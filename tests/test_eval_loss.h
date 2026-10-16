#pragma once

#include <array>

#include "../src/train.h"
#include "test_util.h"

// full_loss (cxxgpt eval-loss --all) scores every trained target exactly
// once, each after the tokens before it in its window, and pads the rest;
// sampled_loss (train's validation, eval-loss's default) gives the same
// number every time. A fake model whose "loss" for a target is that
// target's position makes every mistake visible: with tokens[i] = i, a
// target is its own position.
inline void test_eval_loss() {
    for (const bool masked : {false, true}) {
        for (const auto [n, lo, hi, B, Tn] : {std::array<int, 5>{50, 0, 50, 2, 7}, {50, 3, 41, 3, 4},
                                              {9, 0, 9, 4, 8}, {8, 0, 8, 1, 7}, {30, 5, 6, 2, 3}}) {
            TokenData d;
            d.vocab = uint32_t(n);
            for (int i = 0; i < n; ++i) d.tokens.push_back(uint16_t(i));
            if (masked)
                for (int i = 0; i < n; ++i) d.mask.push_back(uint8_t(i % 3 != 0));  // every third target ignored
            std::vector<int> seen(size_t(n), 0);
            bool consistent = true;
            const auto forward = [&](const int* inp, const int* tgt, int b, int t) {
                consistent &= b == B && t == Tn;
                double sum = 0;
                size_t count = 0;
                for (int i = 0; i < b * t; ++i) {
                    if (tgt[i] < 0) continue;
                    // The input before a target is the token before it, and a
                    // window's targets follow each other.
                    consistent &= inp[i] == tgt[i] - 1 && (i % t == 0 || tgt[i - 1] < 0 || tgt[i - 1] == tgt[i] - 1);
                    ++seen[size_t(tgt[i])];
                    sum += tgt[i];
                    ++count;
                }
                return sum / double(count);
            };
            size_t targets = 0;
            const double loss = full_loss(forward, d, size_t(lo), size_t(hi), B, Tn, &targets);
            double want = 0;
            size_t want_count = 0;
            bool once = true;
            for (int i = 0; i < n; ++i) {
                const bool trained = i > lo && i < hi && (!masked || d.mask[size_t(i)]);
                once &= seen[size_t(i)] == (trained ? 1 : 0);
                if (trained) want += i, ++want_count;
            }
            want = want_count ? want / double(want_count) : 0.0;
            EXPECT(once && consistent && targets == want_count && std::abs(loss - want) < 1e-12,
                   "full_loss n %d [%d, %d) B %d Tn %d masked %d: loss %.6f want %.6f, %zu targets want %zu", n, lo,
                   hi, B, Tn, int(masked), loss, want, targets, want_count);
        }
    }
    // Sampled windows: the same draws on every call, and a fully masked
    // batch adds nothing.
    TokenData d;
    d.vocab = 100;
    for (int i = 0; i < 100; ++i) d.tokens.push_back(uint16_t(i));
    const auto mean_target = [](const int* inp, const int* tgt, int b, int t) {
        double sum = 0;
        size_t count = 0;
        for (int i = 0; i < b * t; ++i)
            if (tgt[i] >= 0) sum += tgt[i] + 0.001 * inp[i], ++count;
        return sum / double(count);
    };
    const double a = sampled_loss(mean_target, d, 0, 100, 4, 8, 5), b = sampled_loss(mean_target, d, 0, 100, 4, 8, 5);
    EXPECT(a == b && a > 0, "sampled_loss is not repeatable: %.17g vs %.17g", a, b);
    d.mask.assign(100, 0);
    size_t targets = 7;
    EXPECT(full_loss(mean_target, d, 0, 100, 2, 8, &targets) == 0 && targets == 0,
           "fully masked text: loss %zu targets", targets);
}
