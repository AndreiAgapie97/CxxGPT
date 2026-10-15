// test_qa_context.h — the exact count of answers trained without their
// question (qa_context.h): a hand-worked file, every window start
// enumerated by brute force on random documents, and TokenData::sample_batch
// itself, whose random windows must show the same share.
#pragma once

#include "../src/qa_context.h"
#include "test_util.h"

namespace {

unsigned long long ull(uint64_t x) { return x; }

// Random question/answer documents: BOS, a question (mask off), an answer
// (mask on), lengths up to max_len, so some are longer than a window.
TokenData qa_documents(uint64_t seed, int docs, int max_len) {
    Rng rng(seed);
    TokenData d;
    d.vocab = 64;
    for (int i = 0; i < docs; ++i) {
        const int question = 1 + int(rng.below(uint64_t(max_len))), answer = 1 + int(rng.below(uint64_t(max_len)));
        d.tokens.push_back(uint16_t(kBos));
        d.mask.push_back(0);
        for (int k = 0; k < question + answer; ++k) {
            d.tokens.push_back(uint16_t(2 + rng.below(60)));
            d.mask.push_back(k >= question);
        }
    }
    return d;
}

// Every start sample_batch can draw, every trained target in its window:
// the question is wholly in view if a BOS is among the inputs before the
// target, and not at all if the window starts at or after the document's
// first trained token.
QaContext qa_context_brute(const TokenData& d, int T) {
    QaContext q;
    const size_t n = d.tokens.size();
    for (size_t s = 0; s + size_t(T) + 1 <= n; ++s) {
        for (int t = 0; t < T; ++t) {
            const size_t j = s + size_t(t) + 1;
            if (!d.mask[j] || d.tokens[j] == kBos) continue;
            size_t begin = j;  // j's document's BOS, or 0
            while (begin > 0 && d.tokens[begin] != kBos) --begin;
            size_t first = begin;
            while (!d.mask[first] || d.tokens[first] == kBos) ++first;
            bool bos_in_view = false;
            for (size_t p = s; p < j; ++p) bos_in_view |= d.tokens[p] == kBos;
            q.pairs += 1;
            if (bos_in_view) q.whole += 1;
            else if (s >= first) q.none += 1;
            else q.partial += 1;
        }
    }
    return q;
}

inline void test_qa_context_hand() {
    // BOS q q a a | BOS q a, windows of 4: starts 0..3 (n - T - 1 = 3).
    //   token 3 (a): starts 0, 1, 2 -> whole at 0, part at 1 and 2
    //   token 4 (a): starts 0..3    -> whole at 0, part at 1, 2, none at 3
    //   token 7 (a): start 3        -> whole (its BOS, token 5, is in view)
    // Pairs 8: whole 3, part 4, none 1.
    TokenData d;
    d.vocab = 8;
    d.tokens = {uint16_t(kBos), 2, 2, 3, 3, uint16_t(kBos), 2, 3};
    d.mask = {0, 0, 0, 1, 1, 0, 0, 1};
    const QaContext q = qa_context(d, 4);
    EXPECT(q.pairs == 8 && q.whole == 3 && q.partial == 4 && q.none == 1,
           "hand file: pairs %llu whole %llu part %llu none %llu, worked out 8, 3, 4, 1",
           ull(q.pairs), ull(q.whole), ull(q.partial),
           ull(q.none));
    EXPECT(q.trained == 3 && q.documents == 2 && q.too_long == 0 && q.doc_tokens == 8,
           "hand file: %llu trained, %llu documents, %llu too long", ull(q.trained),
           ull(q.documents), ull(q.too_long));
    // Windows of 3 hold 4 tokens: the first document (5 tokens) is too long.
    EXPECT(qa_context(d, 3).too_long == 1, "a 5-token document must not fit in a window of 3 + 1");
    // No mask: nothing to count.
    d.mask.clear();
    EXPECT(qa_context(d, 4).pairs == 0, "a file without a mask has no questions");
}

inline void test_qa_context_brute_force() {
    for (uint64_t seed : {1, 2, 3}) {
        for (int T : {4, 9, 16}) {
            const TokenData d = qa_documents(seed, 30, 12);
            const QaContext fast = qa_context(d, T), slow = qa_context_brute(d, T);
            EXPECT(fast.pairs == slow.pairs && fast.whole == slow.whole && fast.partial == slow.partial &&
                       fast.none == slow.none,
                   "seed %llu, windows of %d: exact %llu/%llu/%llu of %llu, brute force %llu/%llu/%llu of %llu",
                   ull(seed), T, ull(fast.whole), ull(fast.partial),
                   ull(fast.none), ull(fast.pairs), ull(slow.whole),
                   ull(slow.partial), ull(slow.none), ull(slow.pairs));
        }
    }
}

// The windows training really draws: the share of trained targets with a
// BOS among their inputs must match the exact count (within sampling noise).
inline void test_qa_context_sample_batch() {
    const TokenData d = qa_documents(7, 400, 40);
    const int T = 32, B = 64;
    const QaContext q = qa_context(d, T);
    Rng rng(11);
    std::vector<int> in(size_t(B) * T), tgt(size_t(B) * T);
    uint64_t trained = 0, whole = 0;
    for (int round = 0; round < 300; ++round) {
        d.sample_batch(rng, B, T, 0, d.tokens.size(), in.data(), tgt.data());
        for (int b = 0; b < B; ++b) {
            bool bos = false;
            for (int t = 0; t < T; ++t) {
                bos |= in[size_t(b) * T + t] == kBos;
                if (tgt[size_t(b) * T + t] < 0) continue;
                trained += 1;
                whole += bos;
            }
        }
    }
    const double sampled = double(whole) / double(trained), exact = double(q.whole) / double(q.pairs);
    EXPECT(std::fabs(sampled - exact) < 0.01, "sample_batch shows the whole question for %.4f of trained targets, "
           "the exact count says %.4f", sampled, exact);
}

inline void test_qa_context() {
    test_qa_context_hand();
    test_qa_context_brute_force();
    test_qa_context_sample_batch();
}

}  // namespace
