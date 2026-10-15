// generate.h — batched text generation with the KV cache, and the
// reasoning evaluation built on it.
#pragma once

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "model.h"
#include "reasoning.h"
#include "tokenizer.h"

// Picks the next token from one row of logits: argmax if temperature <= 0,
// otherwise a sample from softmax(logits / temperature).
template <class T>
int pick_token(const T* logits, int V, double temperature, Rng* rng) {
    int best = 0;
    for (int i = 1; i < V; ++i)
        if (logits[i] > logits[best]) best = i;
    if (temperature <= 0 || !rng) return best;
    std::vector<double> p(V);
    double sum = 0;
    for (int i = 0; i < V; ++i) sum += p[i] = std::exp((double(logits[i]) - logits[best]) / temperature);
    double u = rng->uniform() * sum;
    for (int i = 0; i < V; ++i)
        if ((u -= p[i]) < 0) return i;
    return best;
}

// Continues every prompt by max_new tokens, all in one batch, or by fewer
// when the context runs out: with prompts of up to L tokens, at most
// seq_len - L + 1 (the last token is never fed back, so it needs no slot).
// done(generated) is asked after each new token; a sequence that is done
// stops growing (its slot keeps computing until the batch finishes).
// The model only needs cfg (vocab, seq_len), decode_begin and decode_step,
// so tests can pass a scripted fake instead of a real Model.
template <class M>
std::vector<std::vector<int>> generate(M& model, const std::vector<std::vector<int>>& prompts,
                                       int max_new,
                                       const std::function<bool(const std::vector<int>&)>& done,
                                       double temperature = 0.0, Rng* rng = nullptr) {
    const int B = int(prompts.size());
    std::vector<std::vector<int>> out(B);
    if (B == 0) return out;
    int L = 0;
    for (const auto& p : prompts) {
        CHECK(!p.empty(), "empty prompt");
        L = std::max(L, int(p.size()));
    }
    CHECK(max_new > 0, "nothing to generate: max_new is %d", max_new);
    CHECK(L < model.cfg.seq_len, "a prompt of %d tokens fills the model's context of %d", L, model.cfg.seq_len);
    // Positions fed: the prompt, then each picked token but the last.
    // Clamp before addition: even an INT_MAX CLI budget cannot overflow.
    const int room = model.cfg.seq_len - L + 1;
    const int total = L - 1 + std::min(max_new, room);
    std::vector<int> start(B);
    for (int b = 0; b < B; ++b) start[b] = L - int(prompts[b].size());
    model.decode_begin(B, start);
    std::vector<int> cur(B), finished(B, 0);
    int n_finished = 0;
    for (int pos = 0; pos < total; ++pos) {
        for (int b = 0; b < B; ++b) {
            if (pos < L) cur[b] = pos < start[b] ? prompts[b][0] : prompts[b][pos - start[b]];
            // else cur[b] holds the token picked at the previous step
        }
        const auto* logits = model.decode_step(cur.data(), pos);
        if (pos < L - 1) continue;
        for (int b = 0; b < B; ++b) {
            if (finished[b]) continue;  // stopped slots consume no further random draws
            const int t = pick_token(logits + size_t(b) * model.cfg.vocab, model.cfg.vocab, temperature, rng);
            cur[b] = t;
            out[b].push_back(t);
            if (done(out[b])) finished[b] = 1, ++n_finished;
        }
        if (n_finished == B) break;
    }
    return out;
}

// A completion is over once the "#### answer" line has ended.
inline bool completion_done(const std::string& text) {
    return !text.empty() && text.back() == '\n' && text.find("#### ") != std::string::npos;
}

// The answer the model gave, normalized the way scoring normalizes it:
// the text after "#### " up to the end of the line, trimmed, with digit-
// grouping commas removed.
inline std::string normalized_answer(const std::string& completion) {
    return strip_digit_commas(trim(extract_answer(completion)));
}

// How many times each problem is sampled. With tries > 1, every problem is
// generated `tries` times at `temperature` (sampling needs a seeded Rng), and
// the reported answer is the most common final answer among the tries that
// finished one. With tries == 1 the
// evaluation is exactly what it was before votes existed.
struct VoteOpts {
    int tries = 1;
    double temperature = 0.0;
    Rng* rng = nullptr;
};

// The most common answer; on a tie, the one that appeared first.
inline std::string vote_answer(const std::vector<std::string>& answers) {
    std::map<std::string, std::pair<int, size_t>> seen;  // answer -> (count, first index)
    for (size_t i = 0; i < answers.size(); ++i) {
        auto it = seen.find(answers[i]);
        if (it == seen.end()) seen[answers[i]] = {1, i};
        else ++it->second.first;
    }
    const std::string* best = nullptr;
    int best_count = 0;
    size_t best_first = 0;
    for (const auto& [ans, c] : seen) {
        if (!best || c.first > best_count || (c.first == best_count && c.second < best_first)) {
            best = &ans;
            best_count = c.first;
            best_first = c.second;
        }
    }
    return best ? *best : "";
}

// How one problem's tries scored.
struct TryScore {
    bool first_ok = false;  // the first try alone (with one try: the plain score)
    bool vote_ok = false;   // the most common answer (maj@N)
    bool any_ok = false;    // at least one try (pass@N)
    int distinct = 0;       // how many different answers the tries gave
    size_t shown = 0;       // the try to show when the vote is wrong
};

// Scores one problem's tries (their completion texts) against the gold
// answer. Only a try that finished its "#### answer" line has an answer; a
// try that ran out of tokens, or wrote no answer line, abstains: it does not
// vote and is not counted among the distinct answers. A problem where no try
// finished is wrong. With one try this is exactly the plain check
// completion_done(text) && answer_matches(text, gold).
inline TryScore score_tries(const std::vector<std::string>& texts, const std::string& gold) {
    std::vector<std::string> answers;  // the finished tries' answers
    std::vector<size_t> from;          // which try each answer came from
    for (size_t t = 0; t < texts.size(); ++t) {
        if (!completion_done(texts[t])) continue;
        answers.push_back(normalized_answer(texts[t]));
        from.push_back(t);
    }
    TryScore s;
    s.first_ok = !texts.empty() && completion_done(texts[0]) && answer_matches(texts[0], gold);
    s.any_ok = std::find(answers.begin(), answers.end(), gold) != answers.end();
    s.distinct = int(std::set<std::string>(answers.begin(), answers.end()).size());
    if (answers.empty()) return s;  // no try finished: wrong, and the first try is shown
    const std::string voted = vote_answer(answers);
    s.vote_ok = voted == gold;
    // Show the first try that gave the voted answer.
    s.shown = from[size_t(std::find(answers.begin(), answers.end(), voted) - answers.begin())];
    return s;
}

// The scores of a group of problems (one level, or all of them): how many
// there are, how many the first try, the vote (maj@N) and any try (pass@N)
// got right, and the distinct answers summed over the problems (divide by
// total for the mean). With one try the three counts are equal.
struct LevelScore {
    int total = 0;
    int first_correct = 0, vote_correct = 0, any_correct = 0;
    int distinct_answers = 0;

    void add(const TryScore& s) {
        total += 1;
        first_correct += s.first_ok;
        vote_correct += s.vote_ok;
        any_correct += s.any_ok;
        distinct_answers += s.distinct;
    }
    void add(const LevelScore& o) {
        total += o.total;
        first_correct += o.first_correct;
        vote_correct += o.vote_correct;
        any_correct += o.any_correct;
        distinct_answers += o.distinct_answers;
    }
};

struct EvalResult {
    // (task, difficulty) -> that level's scores.
    std::map<std::pair<std::string, int>, LevelScore> by_task;
    // The headline: problems the vote got right (maj@N; with one try, the
    // plain score), out of all of them.
    int correct = 0, total = 0;
    int tries = 1;
    int max_new = 200;
    size_t gold_over_cap = 0;
    std::vector<std::pair<Problem, std::string>> failures;  // a few, for inspection
    std::vector<std::string> completions;  // optional, the first tries in input order, for trace audits

    // All levels together.
    LevelScore overall() const {
        LevelScore all;
        for (const auto& entry : by_task) all.add(entry.second);
        return all;
    }
};

// The tokens a problem's prompt is fed as: BOS, then "Q: ...\nA: ". The
// completion is encoded separately (training does the same), so no token
// straddles the boundary the model starts generating from.
inline std::vector<int> prompt_tokens(const Problem& p, const Tokenizer& tok) {
    std::vector<int> t = tok.encode(p.prompt());
    t.insert(t.begin(), kBos);
    return t;
}

// Decodes every problem from "A: " and checks the answer exactly. With
// vote.tries == 1 (the default) this is the plain greedy evaluation. With
// tries > 1, each problem is sampled `tries` times at vote.temperature (an
// Rng must be given) and the reported answer is the most common one among
// the tries that finished an answer line, ties going to the answer that
// appeared first (score_tries).
// The default Tokenizer (no merges) is byte-level.
template <class M>
EvalResult eval_reasoning(M& model, const std::vector<Problem>& problems, const Tokenizer& tok = {},
                          int batch = 64, int max_new = 200, size_t keep_failures = 8,
                          bool keep_completions = false, const VoteOpts& vote = {}) {
    CHECK(batch > 0 && max_new > 0, "evaluation batch and max_new must be positive");
    CHECK(vote.tries > 0, "--tries must be positive");
    CHECK(vote.tries == 1 || (vote.temperature > 0 && vote.rng),
          "sampling more than one try needs --temp above 0 and a seed");
    CHECK(model.cfg.vocab >= tok.vocab_size(), "model vocab %d < tokenizer vocab %d", model.cfg.vocab,
          tok.vocab_size());
    // Ids the tokenizer cannot decode (model vocab padding) end a completion.
    auto text_of = [&](const std::vector<int>& gen) {
        std::vector<int> ok;
        for (int t : gen) ok.push_back(t < tok.vocab_size() ? t : '?');
        return tok.decode(ok);
    };
    // Batches of similar prompt lengths: prompts are padded to the longest
    // in their batch, and the context left after it is all the room any of
    // them has to answer in. A prompt that fills the whole context cannot be
    // answered and counts as wrong. With tries > 1, a batch holds as many
    // whole problems as fit: each contributes `tries` copies of its prompt,
    // so one generate() call samples every try of those problems together.
    const size_t n = problems.size();
    const size_t group_problems = std::max<size_t>(1, size_t(batch) / size_t(vote.tries));
    std::vector<std::vector<int>> prompts(n);
    std::vector<size_t> order;
    for (size_t i = 0; i < n; ++i) {
        prompts[i] = prompt_tokens(problems[i], tok);
        if (int(prompts[i].size()) < model.cfg.seq_len) order.push_back(i);
    }
    std::stable_sort(order.begin(), order.end(),
                     [&](size_t x, size_t y) { return prompts[x].size() < prompts[y].size(); });
    std::vector<std::vector<std::string>> texts(n, std::vector<std::string>(size_t(vote.tries),
                                                                            "(the prompt does not fit in the context)"));
    for (size_t k0 = 0; k0 < order.size(); k0 += group_problems) {
        const size_t k1 = std::min(order.size(), k0 + group_problems);
        std::vector<std::vector<int>> group;
        for (size_t k = k0; k < k1; ++k)
            for (int t = 0; t < vote.tries; ++t) group.push_back(prompts[order[k]]);
        const auto outs = generate(model, group, max_new,
                                   [&](const std::vector<int>& gen) { return completion_done(text_of(gen)); },
                                   vote.temperature, vote.temperature > 0 ? vote.rng : nullptr);
        // Problem order[k]'s tries sit together in the group, in try order.
        for (size_t k = k0; k < k1; ++k)
            for (int t = 0; t < vote.tries; ++t)
                texts[order[k]][size_t(t)] = text_of(outs[(k - k0) * size_t(vote.tries) + size_t(t)]);
    }
    EvalResult r;
    r.max_new = max_new;
    r.tries = vote.tries;
    for (size_t i = 0; i < n; ++i) {
        const Problem& p = problems[i];
        r.gold_over_cap += tok.encode(p.completion()).size() > size_t(max_new);
        const TryScore s = score_tries(texts[i], p.answer);
        r.by_task[{p.task, p.difficulty}].add(s);
        r.correct += s.vote_ok;  // the headline number is the vote's (maj@N)
        r.total += 1;
        if (!s.vote_ok && r.failures.size() < keep_failures) r.failures.push_back({p, texts[i][s.shown]});
    }
    if (keep_completions) {
        r.completions.resize(n);
        for (size_t i = 0; i < n; ++i) r.completions[i] = texts[i][0];
    }
    return r;
}
