// eval_choice.h — multiple-choice tests scored by likelihood (cxxgpt
// eval-choice): no text is generated. For each question the model reads the
// context once per answer option, and the option whose tokens it finds most
// likely wins. This is how small models are usually tested on ARC, SciQ,
// HellaSwag and the like: they cannot yet answer in free text, but they can
// already tell which of four endings reads best.
//
// One row per (question, option): [BOS] context tokens, option tokens,
// padded on the right (attention is causal, so padding after the option
// changes nothing before it). Only the option's tokens have targets: the
// last context token predicts the option's first token, and so on. The
// context and the option are tokenized apart and joined, so every option is
// cut into tokens the same way whatever comes before it (lm-eval's
// convention); a test's converter puts the space between them at the start
// of each option (" Paris"). A row longer than the model's context loses
// context tokens from the start (after BOS), never option tokens.
//
// Each option gets three scores, and each gives an accuracy:
// - sum: the summed log-probability of its tokens (lm-eval's "acc");
// - per token: the sum over the option's token count;
// - per character: the sum over the option's length in characters (UTF-8
//   code points, not counting a space in front): lm-eval's "acc_norm",
//   which most published tables of small models report.
// Long options collect more negative log-probability, so the sum favours
// short options; the normalized scores take that out. A tie (two options
// with the same best score) counts as wrong.
//
// Each accuracy comes with its standard error, sqrt(p (1 - p) / n): about
// two of them either side is where a repeat on new questions of the same
// kind would land 95% of the time. Chance is the mean over questions of one
// over the number of options; "the most common gold position" is what
// always picking the same option would score.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include "common.h"
#include "json.h"
#include "tokenizer.h"

namespace eval_choice {

// One question of a test file: one JSON object per line,
// {"id": "...", "context": "...", "options": ["...", ...], "gold": 0}.
struct Item {
    std::string id, context;
    std::vector<std::string> options;
    int gold = 0;
};

inline Item parse_item(const std::string& line, const std::string& where) {
    Json j;
    std::string error;
    CHECK(Json::parse(line, j, &error), "%s: %s", where.c_str(), error.c_str());
    CHECK(j.type == Json::Object, "%s: not a JSON object", where.c_str());
    Item it;
    const Json *id = j.get("id"), *ctx = j.get("context"), *opts = j.get("options"), *gold = j.get("gold");
    CHECK(ctx && ctx->type == Json::String, "%s: no \"context\" string", where.c_str());
    CHECK(opts && opts->type == Json::Array && opts->items.size() >= 2, "%s: \"options\" needs at least two strings",
          where.c_str());
    CHECK(gold && gold->type == Json::Number, "%s: no \"gold\" number", where.c_str());
    if (id) it.id = id->type == Json::String ? id->str : std::to_string(long(id->number));
    it.context = ctx->str;
    for (const Json& o : opts->items) {
        CHECK(o.type == Json::String && !o.str.empty(), "%s: every option must be a non-empty string", where.c_str());
        it.options.push_back(o.str);
    }
    it.gold = int(gold->number);
    CHECK(double(it.gold) == gold->number && it.gold >= 0 && it.gold < int(it.options.size()),
          "%s: \"gold\" must be an option's index, 0 to %zu", where.c_str(), it.options.size() - 1);
    return it;
}

inline std::vector<Item> load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    CHECK(in, "cannot open %s", path.c_str());
    std::vector<Item> items;
    std::string line;
    for (size_t n = 1; std::getline(in, line); ++n) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        items.push_back(parse_item(line, path + ":" + std::to_string(n)));
    }
    CHECK(!in.bad(), "error reading %s", path.c_str());
    return items;
}

// The (question, option) rows of a test, as token ids.
struct Row {
    std::vector<int> tokens;  // [BOS] context option
    int option_start = 0;     // tokens[option_start..] are the option's
    int item = 0, option = 0;
};

// rows for items; seq_len is the model's context: a longer row keeps BOS
// and its last seq_len tokens but one (the last token is only a target).
// *cut counts the questions that lost context tokens.
inline std::vector<Row> make_rows(const std::vector<Item>& items, const Tokenizer& tok, int seq_len,
                                  size_t* cut = nullptr) {
    std::vector<Row> rows;
    if (cut) *cut = 0;
    for (size_t i = 0; i < items.size(); ++i) {
        const std::vector<int> ctx = tok.encode(items[i].context);
        bool cut_here = false;
        for (size_t o = 0; o < items[i].options.size(); ++o) {
            const std::vector<int> opt = tok.encode(items[i].options[o]);
            CHECK(!opt.empty(), "question %zu, option %zu has no tokens", i + 1, o + 1);
            // The row is BOS + ctx + opt; its first seq_len tokens are inputs,
            // so at most seq_len + 1 tokens fit, the option's whole.
            CHECK(int(opt.size()) + 1 <= seq_len, "question %zu, option %zu: %zu tokens, longer than the context %d",
                  i + 1, o + 1, opt.size(), seq_len);
            const size_t keep = std::min(ctx.size(), size_t(seq_len + 1) - 1 - opt.size());
            cut_here = cut_here || keep < ctx.size();
            Row r;
            r.tokens.push_back(kBos);
            r.tokens.insert(r.tokens.end(), ctx.end() - std::ptrdiff_t(keep), ctx.end());
            r.option_start = int(r.tokens.size());
            r.tokens.insert(r.tokens.end(), opt.begin(), opt.end());
            r.item = int(i);
            r.option = int(o);
            rows.push_back(std::move(r));
        }
        if (cut && cut_here) ++*cut;
    }
    return rows;
}

// A model's per-position losses: for tokens and targets [B * Tn] (-1 = no
// target), -log p(target) at each position (anything where the target is
// -1), as Model::forward and GpuTrainer::forward_losses compute them.
using LossFn = std::function<std::vector<double>(const std::vector<int>& tokens, const std::vector<int>& targets,
                                                 int B, int Tn)>;

// The summed log-probability of each row's option tokens, rows scored B at
// a time. Rows are sorted longest first, so a batch is as long as its
// longest row and the first batch is the largest (the model allocates once).
inline std::vector<double> score_rows(const std::vector<Row>& rows, int B, const LossFn& losses) {
    CHECK(B > 0, "the batch must be positive");
    std::vector<size_t> order(rows.size());
    std::iota(order.begin(), order.end(), size_t(0));
    std::stable_sort(order.begin(), order.end(),
                     [&](size_t a, size_t b) { return rows[a].tokens.size() > rows[b].tokens.size(); });
    std::vector<double> logp(rows.size(), 0.0);
    for (size_t at = 0; at < order.size(); at += size_t(B)) {
        const size_t n = std::min(size_t(B), order.size() - at);
        const int Tn = int(rows[order[at]].tokens.size()) - 1;
        std::vector<int> inp(n * size_t(Tn), 0), tgt(inp.size(), -1);
        for (size_t b = 0; b < n; ++b) {
            const Row& r = rows[order[at + b]];
            for (size_t t = 0; t + 1 < r.tokens.size(); ++t) {
                inp[b * size_t(Tn) + t] = r.tokens[t];
                if (int(t) + 1 >= r.option_start) tgt[b * size_t(Tn) + t] = r.tokens[t + 1];
            }
        }
        const std::vector<double> l = losses(inp, tgt, int(n), Tn);
        CHECK(l.size() >= inp.size(), "the model returned %zu losses for %zu positions", l.size(), inp.size());
        for (size_t b = 0; b < n; ++b) {
            double s = 0;
            for (size_t t = 0; t < size_t(Tn); ++t)
                if (tgt[b * size_t(Tn) + t] >= 0) s -= l[b * size_t(Tn) + t];
            logp[order[at + b]] = s;
        }
    }
    return logp;
}

struct Accuracy {
    long right = 0, n = 0;
    double value() const { return n ? double(right) / double(n) : 0.0; }
    // The binomial standard error, sqrt(p (1 - p) / n).
    double se() const { return n ? std::sqrt(value() * (1 - value()) / double(n)) : 0.0; }
};

struct Result {
    Accuracy sum, per_token, per_char;
    double chance = 0;           // mean of 1 / options
    double most_common_gold = 0;  // share of the most frequent gold index
    int most_common_index = 0;
    size_t questions = 0, rows = 0;
    std::vector<int> pick_sum, pick_token, pick_char;  // per question: the chosen option
};

// UTF-8 code points in s, without a space in front (at least 1).
inline size_t characters(const std::string& s) {
    size_t n = 0;
    for (size_t i = s.empty() || s[0] != ' ' ? 0 : 1; i < s.size(); ++i)
        n += (static_cast<unsigned char>(s[i]) & 0xC0) != 0x80;
    return std::max<size_t>(n, 1);
}

// The index of the highest score, or -1 for a tie at the top.
inline int best(const std::vector<double>& s) {
    int b = 0;
    bool tie = false;
    for (int i = 1; i < int(s.size()); ++i) {
        if (s[size_t(i)] > s[size_t(b)]) b = i, tie = false;
        else if (s[size_t(i)] == s[size_t(b)]) tie = true;
    }
    return tie ? -1 : b;
}

// Scores items from their rows' summed log-probabilities (score_rows), with
// skip[i] true for questions left out (a leak), if given.
inline Result tally(const std::vector<Item>& items, const std::vector<Row>& rows, const std::vector<double>& logp,
                    const std::vector<bool>& skip = {}) {
    std::vector<std::vector<double>> sums(items.size()), tokens(items.size()), chars(items.size());
    for (size_t k = 0; k < rows.size(); ++k) {
        const Row& r = rows[k];
        const size_t i = size_t(r.item);
        const double nt = double(r.tokens.size()) - r.option_start;
        const double nc = double(characters(items[i].options[size_t(r.option)]));
        sums[i].push_back(logp[k]);
        tokens[i].push_back(logp[k] / nt);
        chars[i].push_back(logp[k] / nc);
    }
    Result res;
    res.rows = rows.size();
    std::vector<long> gold_count;
    for (size_t i = 0; i < items.size(); ++i) {
        CHECK(sums[i].size() == items[i].options.size(), "question %zu has %zu scored options of %zu", i + 1,
              sums[i].size(), items[i].options.size());
        const int s = best(sums[i]), t = best(tokens[i]), c = best(chars[i]);
        res.pick_sum.push_back(s);
        res.pick_token.push_back(t);
        res.pick_char.push_back(c);
        if (!skip.empty() && skip[i]) continue;
        const int g = items[i].gold;
        ++res.questions;
        res.chance += 1.0 / double(items[i].options.size());
        if (size_t(g) >= gold_count.size()) gold_count.resize(size_t(g) + 1, 0);
        ++gold_count[size_t(g)];
        for (auto [acc, pick] : {std::pair{&res.sum, s}, {&res.per_token, t}, {&res.per_char, c}}) {
            ++acc->n;
            acc->right += pick == g;
        }
    }
    if (res.questions) {
        res.chance /= double(res.questions);
        const auto top = std::max_element(gold_count.begin(), gold_count.end());
        res.most_common_index = int(top - gold_count.begin());
        res.most_common_gold = double(*top) / double(res.questions);
    }
    return res;
}

// 1-based question numbers, one per line (leak-check --hits), as a mask
// over n questions.
inline std::vector<bool> read_skip(const std::string& path, size_t n) {
    std::ifstream in(path);
    CHECK(in, "cannot open %s", path.c_str());
    std::vector<bool> skip(n, false);
    long q = 0;
    while (in >> q) {
        CHECK(q >= 1 && size_t(q) <= n, "%s: question %ld is not in 1..%zu", path.c_str(), q, n);
        skip[size_t(q - 1)] = true;
    }
    CHECK(in.eof(), "%s: expected one question number per line", path.c_str());
    return skip;
}

}  // namespace eval_choice
