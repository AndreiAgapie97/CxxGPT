// qa_context.h — how often training sees an answer without its question
// (issue #15). TokenData::sample_batch starts each window at a random
// offset, so a window can begin in the middle of a question-answer
// document: the answer's tokens are trained, but the question that says
// what to compute has scrolled out of view.
//
// The count is exact, not sampled. A target token j is trained by the
// windows that start at s = j - Tn ... j - 1 (the inputs s .. s + Tn - 1
// must reach j - 1, and the targets end at s + Tn), within the starts
// sample_batch draws from, 0 ... n - Tn - 1, each equally likely. Of those
// starts:
// - the whole question is in view when s <= the document's BOS (the BOS
//   comes right before the question);
// - none of it is when s >= the document's first trained token (the
//   window starts inside the answer);
// - part of it is otherwise.
// Summed over every trained token, that is exactly the share random windows
// give, weighted as training weights them (one loss term per target).
#pragma once

#include <algorithm>
#include <cstdint>

#include "reasoning.h"
#include "train.h"

struct QaContext {
    // (trained target token, window start) pairs, split by what the window
    // shows of that token's question.
    uint64_t pairs = 0, whole = 0, partial = 0, none = 0;
    uint64_t trained = 0;      // trained target tokens
    uint64_t documents = 0;    // documents with at least one trained token
    uint64_t too_long = 0;     // of those, longer than a window's Tn + 1 tokens
    uint64_t doc_tokens = 0;   // their tokens, for the mean length
};

// Every trained token of `d` (mask on) and the windows of Tn inputs that
// train it. A file without a mask has no question/answer split: nothing to
// count. Tokens before the first BOS count as a document starting at 0
// whose question is never wholly in view.
inline QaContext qa_context(const TokenData& d, int Tn) {
    CHECK(Tn > 0, "window length must be positive");
    QaContext q;
    const int64_t n = int64_t(d.tokens.size()), T = Tn;
    if (d.mask.empty() || n < T + 1) return q;
    const int64_t last_start = n - T - 1;  // sample_batch's largest start
    int64_t bos = -1;                      // the current document's BOS (-1: none yet)
    int64_t first_trained = -1;            // its first trained token (-1: none yet)
    int64_t doc_begin = 0;
    auto end_document = [&](int64_t end) {
        if (first_trained < 0) return;
        q.documents += 1;
        q.doc_tokens += uint64_t(end - doc_begin);
        q.too_long += end - doc_begin > T + 1;
    };
    for (int64_t j = 0; j < n; ++j) {
        if (d.tokens[size_t(j)] == kBos) {
            end_document(j);
            bos = j;
            doc_begin = j;
            first_trained = -1;
            continue;
        }
        if (!d.mask[size_t(j)] || j == 0) continue;  // token 0 is never a target
        if (first_trained < 0) first_trained = j;
        q.trained += 1;
        const int64_t lo = std::max<int64_t>(0, j - T), hi = std::min(j - 1, last_start);
        if (lo > hi) continue;
        const uint64_t all = uint64_t(hi - lo + 1);
        const uint64_t whole = bos < 0 ? 0 : uint64_t(std::max<int64_t>(0, std::min(hi, bos) - lo + 1));
        const uint64_t none = uint64_t(std::max<int64_t>(0, hi - std::max(lo, first_trained) + 1));
        q.pairs += all;
        q.whole += whole;
        q.none += none;
        q.partial += all - whole - none;
    }
    end_document(n);
    return q;
}
