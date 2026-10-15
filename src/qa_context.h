// qa_context.h — how often training sees an answer without its question
// (issue #15). TokenData::sample_batch starts each window at a random
// offset, so a window can begin in the middle of a question-answer
// document: the answer's tokens are trained, but the question that says
// what to compute has scrolled out of view. train --sampler qa (QaStarts,
// train.h) moves such starts back to the document's BOS.
//
// The count is exact, not sampled. sample_batch draws every start s in
// 0 ... n - Tn - 1 equally often; with QaStarts a drawn start may move to
// another, so each start x has a weight: how many drawn starts become x.
// A target token j is trained by the windows starting at x = j - Tn ...
// j - 1 (the inputs x .. x + Tn - 1 must reach j - 1, and the targets end
// at x + Tn). Of those, weighted:
// - the whole question is in view when x <= the document's BOS (the BOS
//   comes right before the question);
// - none of it is when x >= the document's first trained token (the
//   window starts inside the answer);
// - part of it is otherwise.
// Summed over every trained token, that is exactly the share the sampler's
// windows give, weighted as training weights them (one loss term per
// target).
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "reasoning.h"
#include "train.h"

struct QaContext {
    // (trained target token, drawn window start) pairs, split by what the
    // window shows of that token's question.
    uint64_t pairs = 0, whole = 0, partial = 0, none = 0;
    uint64_t windows = 0;      // the starts sample_batch draws from
    uint64_t trained = 0;      // trained target tokens
    uint64_t unreached = 0;    // of those, in no window at all
    uint64_t documents = 0;    // documents with at least one trained token
    uint64_t too_long = 0;     // of those, longer than a window's Tn + 1 tokens
    uint64_t doc_tokens = 0;   // their tokens, for the mean length
};

// Every trained token of `d` (mask on) and the windows of Tn inputs that
// train it: random starts, or with `qa` the starts QaStarts moves them to
// (built for the same Tn). A file without a mask has no question/answer
// split: nothing to count. Tokens before the first BOS count as a document
// starting at 0 whose question is never wholly in view.
inline QaContext qa_context(const TokenData& d, int Tn, const QaStarts* qa = nullptr) {
    CHECK(Tn > 0, "window length must be positive");
    QaContext q;
    const int64_t n = int64_t(d.tokens.size()), T = Tn;
    if (d.mask.empty() || n < T + 1) return q;
    const int64_t last_start = n - T - 1;  // sample_batch's largest start
    q.windows = uint64_t(last_start + 1);
    // How many drawn starts become start x.
    size_t k = 0;  // the next QaStarts document that may hold x
    auto weight = [&](int64_t x) -> uint64_t {
        if (qa) {
            while (k < qa->documents() && int64_t(qa->end(k)) <= x) ++k;
            if (k < qa->documents() && int64_t(qa->begin(k)) <= x) {
                if (x != int64_t(qa->begin(k))) return 0;  // moved back to the BOS
                const int64_t last = std::min(int64_t(qa->end(k)) - 1, last_start);
                return last >= x ? uint64_t(last - x + 1) : 0;
            }
        }
        return x <= last_start ? 1 : 0;
    };
    // W(x) = the weights of starts 0 .. x - 1, kept for x = j - Tn ... j.
    std::vector<uint64_t> ring(size_t(T) + 1, 0);
    auto W = [&](int64_t x) { return ring[size_t(x % (T + 1))]; };
    int64_t bos = -1;            // the current document's BOS (-1: none yet)
    int64_t first_trained = -1;  // its first trained token (-1: none yet)
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
        } else if (d.mask[size_t(j)] && j > 0) {  // token 0 is never a target
            if (first_trained < 0) first_trained = j;
            q.trained += 1;
            const int64_t lo = std::max<int64_t>(0, j - T), hi = std::min(j - 1, last_start);
            const uint64_t all = lo <= hi ? W(hi + 1) - W(lo) : 0;
            const uint64_t whole = bos >= lo && lo <= hi ? W(std::min(hi, bos) + 1) - W(lo) : 0;
            const int64_t from = std::max(lo, first_trained);
            const uint64_t none = from <= hi ? W(hi + 1) - W(from) : 0;
            q.unreached += all == 0;
            q.pairs += all;
            q.whole += whole;
            q.none += none;
            q.partial += all - whole - none;
        }
        ring[size_t((j + 1) % (T + 1))] = W(j) + weight(j);
    }
    end_document(n);
    return q;
}
