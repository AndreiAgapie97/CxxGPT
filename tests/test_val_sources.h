#pragma once

#include <string>
#include <vector>

#include "../src/val_sources.h"
#include "test_util.h"

// train --val-sources: the final answers of a Q&A file are the trained
// tokens after its "####", also when the hashes are split over tokens; and
// each source is scored on its own targets only.
inline void test_val_sources() {
    // A made-up vocabulary: 0 "Q: ", 1 "x", 2 "A: ", 3 "so 5", 4 "##", 5 "#",
    // 6 " 12", 7 "\n", 8 "#### 7", 9 "#a", 10 "###".
    const std::vector<std::string> vocab = {"Q: ", "x", "A: ", "so 5", "##", "#", " 12", "\n", "#### 7", "#a", "###"};
    const auto bytes = [&](int id) -> const std::string& { return vocab[size_t(id)]; };
    struct Case {
        const char* what;
        std::vector<uint16_t> tokens;
        std::vector<uint8_t> mask, want;
    };
    const std::vector<Case> cases = {
        {"hashes split over two tokens", {0, 1, 2, 3, 4, 4, 6, 7}, {0, 0, 0, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0, 1, 1}},
        {"hashes and answer in one token", {2, 3, 8, 7}, {0, 1, 1, 1}, {0, 0, 1, 1}},
        {"no #### in the answer", {0, 2, 3, 6, 7}, {0, 0, 1, 1, 1}, {0, 0, 0, 0, 0}},
        // Three hashes, then one more in the next token: four in a row.
        {"three then one", {2, 10, 5, 6}, {0, 1, 1, 1}, {0, 0, 0, 1}},
        // A hash run broken by another byte does not count: "#a###".
        {"broken run", {2, 9, 10, 6}, {0, 1, 1, 1}, {0, 0, 0, 0}},
        // A question between two answers starts a new span; hashes do not
        // carry over the question.
        {"two documents", {2, 4, 0, 4, 6, 2, 3, 8, 6}, {0, 1, 0, 1, 1, 0, 1, 1, 1}, {0, 0, 0, 0, 0, 0, 0, 1, 1}},
    };
    for (const auto& c : cases) {
        TokenData d;
        d.vocab = uint32_t(vocab.size());
        d.tokens = c.tokens;
        d.mask = c.mask;
        const std::vector<uint8_t> got = final_answer_mask(d, bytes);
        std::string shown;
        for (const uint8_t v : got) shown += char('0' + v);
        EXPECT(got == c.want, "final answers, %s: got %s", c.what, shown.c_str());
    }

    const auto spec = parse_val_sources("gsm8k-dev=a.tok,books=dir/b.tok");
    EXPECT(spec.size() == 2 && spec[0].first == "gsm8k-dev" && spec[0].second == "a.tok" && spec[1].first == "books" &&
               spec[1].second == "dir/b.tok",
           "parse_val_sources");

    // Scoring: a fake model whose loss for a target is the target token,
    // so each number shows exactly which targets it covered.
    const auto mean_target = [](const int*, const int* tgt, int b, int t) {
        double sum = 0;
        size_t count = 0;
        for (int i = 0; i < b * t; ++i)
            if (tgt[i] >= 0) sum += tgt[i], ++count;
        return sum / double(count);
    };
    ValSource qa{"qa", "qa.tok", {}, {}};
    qa.data.vocab = uint32_t(vocab.size());
    qa.data.tokens = {2, 3, 4, 4, 6, 7, 0, 1, 2, 3, 8, 7};
    qa.data.mask = {0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1};
    qa.final_only = qa.data;
    qa.final_only.mask = final_answer_mask(qa.data, bytes);  // targets 6 7 | 8 7
    const auto r = score_val_source(mean_target, qa, 2, 4, 3);
    EXPECT(r.size() == 2 && r[0].part == "answers" && r[1].part == "final", "a Q&A file gives answers and final");
    if (r.size() == 2) {
        EXPECT(r[0].targets == 8 && std::abs(r[0].loss - (3 + 4 + 4 + 6 + 7 + 3 + 8 + 7) / 8.0) < 1e-12,
               "answers: %zu targets, loss %.6f", r[0].targets, r[0].loss);
        EXPECT(r[1].targets == 4 && std::abs(r[1].loss - (6 + 7 + 8 + 7) / 4.0) < 1e-12,
               "final: %zu targets, loss %.6f", r[1].targets, r[1].loss);
    }
    ValSource text{"text", "t.tok", {}, {}};
    text.data.vocab = 16;
    for (int i = 0; i < 40; ++i) text.data.tokens.push_back(uint16_t(i % 16));
    const auto t = score_val_source(mean_target, text, 2, 4, 3);
    EXPECT(t.size() == 1 && t[0].part == "text" && t[0].targets == 39, "plain text: every target once");
}
