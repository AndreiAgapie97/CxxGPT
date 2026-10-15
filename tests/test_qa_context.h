// test_qa_context.h — the exact count of answers trained without their
// question (qa_context.h), and train --sampler qa (QaStarts): hand-worked
// files, every window start enumerated by brute force on random documents,
// and TokenData::sample_batch itself, whose windows must show the same
// shares, from the same random draws with or without the sampler.
#pragma once

#include "../src/qa_context.h"
#include "test_util.h"

namespace {

unsigned long long ull(uint64_t x) { return x; }

// Random question/answer documents: BOS, a question (mask off), an answer
// (mask on), lengths up to max_len, so some are longer than a window. With
// text_every k, every k-th document is plain text instead, trained on every
// token (as an unmasked file is in a mixed --data list).
TokenData qa_documents(uint64_t seed, int docs, int max_len, int text_every = 0) {
    Rng rng(seed);
    TokenData d;
    d.vocab = 64;
    for (int i = 0; i < docs; ++i) {
        const int question = 1 + int(rng.below(uint64_t(max_len))), answer = 1 + int(rng.below(uint64_t(max_len)));
        const bool text = text_every > 0 && i % text_every == text_every - 1;
        d.tokens.push_back(uint16_t(kBos));
        d.mask.push_back(text);
        for (int k = 0; k < question + answer; ++k) {
            d.tokens.push_back(uint16_t(2 + rng.below(60)));
            d.mask.push_back(text || k >= question);
        }
    }
    return d;
}

// Every start sample_batch can draw (moved by `qa` if given), every
// trained target in its window: the question is wholly in view if a BOS is
// among the inputs before the target, and not at all if the window starts
// at or after the document's first trained token.
QaContext qa_context_brute(const TokenData& d, int T, const QaStarts* qa = nullptr) {
    QaContext q;
    const size_t n = d.tokens.size();
    for (size_t drawn = 0; drawn + size_t(T) + 1 <= n; ++drawn) {
        const size_t s = qa ? qa->start(drawn, 0) : drawn;
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

// QaStarts on a hand-worked file, windows of 4 (documents of up to 5 tokens fit):
//   0 BOS 1 q 2 a 3 a | 4 BOS 5 q 6 a | 7 BOS 8 w 9 w (text: trained BOS, no question)
//   | 10 BOS 11 q 12 q 13 a 14 a 15 a (6 tokens: too long)
inline void test_qa_starts_hand() {
    TokenData d;
    d.vocab = 8;
    const uint16_t B = uint16_t(kBos);
    d.tokens = {B, 2, 3, 3, B, 2, 3, B, 4, 4, B, 2, 2, 3, 3, 3};
    d.mask = {0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1};
    const QaStarts qa(d, 4, kBos);
    EXPECT(qa.documents() == 2 && qa.begin(0) == 0 && qa.end(0) == 4 && qa.begin(1) == 4 && qa.end(1) == 7,
           "the two short question-answer documents, and only they, start windows (%zu found)", qa.documents());
    const size_t want[16] = {0, 0, 0, 0, 4, 4, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    for (size_t s = 0; s < 16; ++s)
        EXPECT(qa.start(s, 0) == want[s], "start %zu moved to %zu, expected %zu", s, qa.start(s, 0), want[s]);
    // A document that begins before the training range is not moved into it.
    EXPECT(qa.start(2, 1) == 2 && qa.start(5, 4) == 4, "starts must stay inside [lo, ...)");
    // Windows of 2 hold 3 tokens: only the second document still fits.
    const QaStarts small(d, 2, kBos);
    EXPECT(small.documents() == 1 && small.start(2, 0) == 2 && small.start(6, 0) == 4,
           "with windows of 2 only document 2 fits");
    // No mask: no question-answer documents.
    d.mask.clear();
    EXPECT(QaStarts(d, 4, kBos).documents() == 0, "a file without a mask has no questions");
}

inline void test_qa_context_brute_force() {
    for (uint64_t seed : {1, 2, 3}) {
        for (int T : {4, 9, 16}) {
            // Some documents fit in a window, some do not; every third is text.
            const TokenData d = qa_documents(seed, 30, 12, 3);
            const QaStarts starts(d, T, kBos);
            for (const QaStarts* qa : {static_cast<const QaStarts*>(nullptr), &starts}) {
                const QaContext fast = qa_context(d, T, qa), slow = qa_context_brute(d, T, qa);
                EXPECT(fast.pairs == slow.pairs && fast.whole == slow.whole && fast.partial == slow.partial &&
                           fast.none == slow.none,
                       "seed %llu, windows of %d, %s: exact %llu/%llu/%llu of %llu, brute force %llu/%llu/%llu of %llu",
                       ull(seed), T, qa ? "qa" : "random", ull(fast.whole), ull(fast.partial), ull(fast.none),
                       ull(fast.pairs), ull(slow.whole), ull(slow.partial), ull(slow.none), ull(slow.pairs));
                EXPECT(qa || fast.unreached == 0, "random windows must reach every trained token");
            }
        }
    }
    // Every document fits: with the qa sampler every trained token sees its
    // whole question. Only the end of the range can go untrained: a
    // document whose BOS lies within a window of the end cannot start one,
    // and the windows that reached it before now start at earlier BOSes.
    const TokenData d = qa_documents(4, 60, 6);
    const QaStarts starts(d, 16, kBos);
    const QaContext random = qa_context(d, 16), qa = qa_context(d, 16, &starts);
    EXPECT(qa.whole == qa.pairs && qa.pairs > 0 && qa.unreached <= 16,
           "documents that fit: whole %llu of %llu, %llu unreached", ull(qa.whole), ull(qa.pairs), ull(qa.unreached));
    EXPECT(random.whole < random.pairs, "random windows must miss some questions (%llu of %llu)", ull(random.whole),
           ull(random.pairs));
}

// The windows training really draws: the share of trained targets with a
// BOS among their inputs must match the exact count (within sampling noise),
// for both samplers, from the same random numbers.
inline void test_qa_context_sample_batch() {
    const TokenData d = qa_documents(7, 400, 40);  // documents of 3 to 81 tokens
    const int T = 32, B = 64;
    const QaStarts starts(d, T, kBos);
    EXPECT(starts.documents() > 50 && starts.documents() < 350, "%zu of 400 documents fit in a window",
           starts.documents());
    for (const QaStarts* qa : {static_cast<const QaStarts*>(nullptr), &starts}) {
        const QaContext q = qa_context(d, T, qa);
        Rng rng(11);
        std::vector<int> in(size_t(B) * T), tgt(size_t(B) * T);
        uint64_t trained = 0, whole = 0;
        for (int round = 0; round < 300; ++round) {
            d.sample_batch(rng, B, T, 0, d.tokens.size(), in.data(), tgt.data(), qa);
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
        EXPECT(std::fabs(sampled - exact) < 0.01, "%s: sample_batch shows the whole question for %.4f of trained "
               "targets, the exact count says %.4f", qa ? "qa" : "random", sampled, exact);
        const double per_window = double(trained) / (300.0 * B), exact_per_window = double(q.pairs) / double(q.windows);
        EXPECT(std::fabs(per_window - exact_per_window) < 0.02 * exact_per_window,
               "%s: %.2f trained targets per sampled window, the exact count says %.2f", qa ? "qa" : "random",
               per_window, exact_per_window);
    }
    // The same random numbers either way: a window only moves back, inside
    // its document, and the generator ends in the same state.
    Rng a(5), b(5);
    std::vector<int> in_a(size_t(B) * T), tgt_a(in_a.size()), in_b(in_a.size()), tgt_b(in_a.size());
    for (int round = 0; round < 50; ++round) {
        d.sample_batch(a, B, T, 0, d.tokens.size(), in_a.data(), tgt_a.data());
        d.sample_batch(b, B, T, 0, d.tokens.size(), in_b.data(), tgt_b.data(), &starts);
    }
    EXPECT(a.below(1u << 30) == b.below(1u << 30), "the qa sampler drew different random numbers");
}

inline void test_qa_context() {
    test_qa_context_hand();
    test_qa_starts_hand();
    test_qa_context_brute_force();
    test_qa_context_sample_batch();
}

}  // namespace
