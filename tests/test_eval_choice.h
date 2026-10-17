#pragma once

#include <map>

#include "../src/eval_choice.h"
#include "../src/model.h"
#include "test_util.h"

// eval-choice (eval_choice.h): the rows, the scores and the tallies.
//
// First a case worked out by hand: a fake model whose loss for a target is
// a fixed cost per byte (the byte tokenizer: one token a byte), which also
// checks that each row is [BOS] context option with targets on the option's
// tokens only, each predicted from the token before it. Then the tallies
// (chance, ties, the standard error, skipped questions), a context cut to
// fit, the JSON Lines format, and a real model in double: every option's
// score in a padded batch of mixed rows equals its row scored alone.
inline void test_eval_choice() {
    using namespace eval_choice;
    const Tokenizer tok;  // bytes
    std::map<char, double> cost = {{' ', 0.1}, {'a', 1.0}, {'b', 1.0}, {'c', 0.9}, {'d', 0.5}, {'e', 0.2}};
    std::vector<Item> items = {
        {"one", "Q:", {" ab", " c", " dd"}, 2},
        {"two", "Hm", {" e", " ab"}, 0},
    };
    const std::vector<Row> rows = make_rows(items, tok, 64);
    EXPECT(rows.size() == 5 && rows[0].tokens == std::vector<int>({kBos, 'Q', ':', ' ', 'a', 'b'}) &&
               rows[0].option_start == 3 && rows[4].item == 1 && rows[4].option == 1,
           "rows: %zu, the first has %zu tokens", rows.size(), rows.empty() ? size_t(0) : rows[0].tokens.size());
    bool aligned = true;
    int calls = 0;
    const LossFn fake = [&](const std::vector<int>& inp, const std::vector<int>& tgt, int B, int Tn) {
        ++calls;
        std::vector<double> l(inp.size(), 0.0);
        for (int b = 0; b < B; ++b) {
            // The row this batch row is: the one whose inputs match.
            const Row* row = nullptr;
            for (const Row& r : rows) {
                bool same = int(r.tokens.size()) - 1 <= Tn;
                for (size_t t = 0; same && t + 1 < r.tokens.size(); ++t) same = inp[size_t(b * Tn) + t] == r.tokens[t];
                if (same && (!row || r.tokens.size() > row->tokens.size())) row = &r;
            }
            aligned &= row != nullptr;
            if (!row) continue;
            for (int t = 0; t < Tn; ++t) {
                const int y = tgt[size_t(b * Tn + t)];
                const bool want = t + 1 >= row->option_start && t + 1 < int(row->tokens.size());
                aligned &= want ? y == row->tokens[size_t(t + 1)] : y == -1;
                if (y >= 0) l[size_t(b * Tn + t)] = cost[char(y)];
            }
        }
        return l;
    };
    const std::vector<double> logp = score_rows(rows, 11, fake);  // two rows a forward
    // " ab" 0.1 + 1 + 1, " c" 0.1 + 0.9, " dd" 0.1 + 0.5 + 0.5; " e" 0.1 + 0.2.
    const std::vector<double> want = {-2.1, -1.0, -1.1, -0.3, -2.1};
    double worst = 0;
    for (size_t i = 0; i < want.size(); ++i) worst = std::max(worst, std::abs(logp[i] - want[i]));
    EXPECT(aligned && calls == 3 && worst < 1e-12, "hand case: aligned %d, %d calls, scores off by %g", int(aligned),
           calls, worst);
    // By the sum " c" wins (-1.0 against -1.1); per token " dd" (-1.1 / 3
    // against -1.0 / 2); per character " dd" too (-0.55 against -1.0).
    const Result r = tally(items, rows, logp);
    EXPECT(r.pick_sum == std::vector<int>({1, 0}) && r.pick_token == std::vector<int>({2, 0}) &&
               r.pick_char == std::vector<int>({2, 0}),
           "hand case picks: sum %d %d, token %d %d, char %d %d", r.pick_sum[0], r.pick_sum[1], r.pick_token[0],
           r.pick_token[1], r.pick_char[0], r.pick_char[1]);
    EXPECT(r.sum.right == 1 && r.per_token.right == 2 && r.per_char.right == 2 && r.sum.n == 2 &&
               std::abs(r.chance - (1.0 / 3 + 0.5) / 2) < 1e-12 && std::abs(r.sum.se() - std::sqrt(0.25 / 2)) < 1e-12,
           "hand case tallies: %ld %ld %ld right, chance %g, se %g", r.sum.right, r.per_token.right, r.per_char.right,
           r.chance, r.sum.se());
    // The second question left out (leak-check --hits), and a tie at the top
    // counted as wrong.
    const Result skipped = tally(items, rows, logp, {false, true});
    EXPECT(skipped.questions == 1 && skipped.sum.right == 0 && skipped.per_char.right == 1 &&
               std::abs(skipped.chance - 1.0 / 3) < 1e-12,
           "skip: %zu questions, %ld / %ld right", skipped.questions, skipped.sum.right, skipped.per_char.right);
    EXPECT(best({-1.0, -0.5, -0.5}) == -1 && best({-0.5, -1.0}) == 0 && characters(" d\xc3\xa9") == 2 &&
               characters("x") == 1,
           "best() and characters()");

    // A context too long for the model loses its start, never BOS or the
    // option: seq_len 6 takes 7 tokens (the last is only a target).
    size_t cut = 0;
    const std::vector<Row> tight = make_rows({{"long", "abcdefgh", {" xy", " z"}, 0}}, tok, 6, &cut);
    EXPECT(cut == 1 && tight[0].tokens == std::vector<int>({kBos, 'f', 'g', 'h', ' ', 'x', 'y'}) &&
               tight[1].tokens == std::vector<int>({kBos, 'e', 'f', 'g', 'h', ' ', 'z'}) && tight[0].option_start == 4,
           "cut: %zu, first row %zu tokens", cut, tight[0].tokens.size());

    // The file format.
    const Item p = parse_item(R"({"id": 7, "context": "Q: 2+2?\nA:", "options": [" 3", " 4"], "gold": 1})", "line");
    EXPECT(p.id == "7" && p.context == "Q: 2+2?\nA:" && p.options.size() == 2 && p.options[1] == " 4" && p.gold == 1,
           "parse_item: id %s, gold %d", p.id.c_str(), p.gold);

    // A real model in double: options scored in batches of mixed lengths
    // (padded on the right) give each option's score alone, and that
    // score is the summed loss Model::forward reports for the option.
    Config cfg;
    cfg.vocab = 256;
    cfg.seq_len = 24;
    cfg.d_model = 32;
    cfg.n_layers = 2;
    cfg.n_heads = 2;
    cfg.n_experts = 4;
    cfg.top_k = 2;
    cfg.expert_hidden = 16;
    Model<double> m(cfg);
    m.init(3);
    const LossFn model_losses = [&](const std::vector<int>& inp, const std::vector<int>& tgt, int B, int Tn) {
        m.forward(inp.data(), tgt.data(), B, Tn, false);
        return m.last_losses();
    };
    const std::vector<Item> mixed = {{"a", "The cat sat on the", {" mat", " moon", " a"}, 0},
                                     {"b", "1 2 3", {" 4", " 5 6 7 8 9 10"}, 0},
                                     {"c", "", {"x", "yy"}, 1}};
    const std::vector<Row> mrows = make_rows(mixed, tok, cfg.seq_len);
    const std::vector<double> batched = score_rows(mrows, 40, model_losses);
    double gap = 0;
    for (size_t k = 0; k < mrows.size(); ++k) {
        const Row& row = mrows[k];
        const std::vector<double> alone = score_rows({row}, 1, model_losses);
        const int Tn = int(row.tokens.size()) - 1;
        std::vector<int> inp(row.tokens.begin(), row.tokens.end() - 1), tgt(size_t(Tn), -1);
        for (int t = row.option_start - 1; t < Tn; ++t) tgt[size_t(t)] = row.tokens[size_t(t + 1)];
        const double mean = m.forward(inp.data(), tgt.data(), 1, Tn, false);
        const double n = double(row.tokens.size()) - row.option_start;
        gap = std::max({gap, std::abs(batched[k] - alone[0]), std::abs(alone[0] + mean * n)});
    }
    EXPECT(gap < 1e-12, "model in double: batched and alone scores differ by %g", gap);
}
