// unlikelihood.h — which wrong tokens unlikelihood training (#34) pushes
// down: at each trained position, the tokens that would repeat a run of N
// trained tokens written earlier in the same document. That is the
// no-repeat block's definition (no_repeat.h, #33), applied to the training
// text, so training discourages what decoding would block.
//
// Math-safe by construction: a run shorter than N may repeat, as solutions
// reuse numbers and short phrases on purpose; the right next token is never
// a negative, even when the real text repeats; and masked tokens (the
// question, planted mistakes) neither get negatives nor form runs, so
// copying the question is never penalized.
//
// "Earlier" means earlier in the training window: a window that starts
// inside a document does not see the runs before it. The window's first
// token counts as masked, since whether it is trained is not known here.
#pragma once

#include <vector>

#include "no_repeat.h"
#include "ops.h"

struct RepeatNegatives {
    std::vector<int> offsets;  // NegativeRows' layout: B * Tn + 1 entries
    std::vector<int> tokens;
    size_t fired = 0;  // trained positions with at least one negative

    NegativeRows rows(double alpha) const { return {offsets.data(), tokens.data(), alpha}; }
};

// inputs and targets as TokenData::sample_batch writes them: B rows of Tn,
// targets[t] = inputs[t + 1] where that token is trained, -1 where it is not.
inline void find_repeat_negatives(RepeatNegatives& out, const int* inputs, const int* targets, int B, int Tn,
                                  int n) {
    const size_t N = size_t(B) * size_t(Tn);
    out.offsets.assign(N + 1, 0);
    out.tokens.clear();
    out.fired = 0;
    for (int b = 0; b < B; ++b) {
        const int* x = inputs + size_t(b) * Tn;
        const int* y = targets + size_t(b) * Tn;
        NoRepeat seen(n);
        for (int t = 0; t < Tn; ++t) {
            if (x[t] == kBos)
                seen.add(kBos);  // a new document
            else if (t == 0 || y[t - 1] < 0)
                seen.skip();
            else
                seen.add(x[t]);
            const size_t row = size_t(b) * Tn + size_t(t);
            if (y[t] >= 0) {
                const size_t before = out.tokens.size();
                for (int c : seen.blocked())
                    if (c != y[t]) out.tokens.push_back(c);
                out.fired += out.tokens.size() > before;
            }
            out.offsets[row + 1] = int(out.tokens.size());
        }
    }
}
