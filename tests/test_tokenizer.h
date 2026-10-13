// test_tokenizer.h — BPE: chunking, exact round trips, digits stay single.
#pragma once

#include "../src/tokenizer.h"
#include "test_util.h"

inline void test_tokenizer() {
    // Chunking rules.
    const std::string s = "Hello, world! 123  foo\n\nbar";
    std::vector<std::string> got;
    for (auto [b, e] : Tokenizer::chunks(s)) got.push_back(s.substr(b, e - b));
    const std::vector<std::string> want = {"Hello", ",", " world", "!", " ", "1", "2", "3",
                                           " ", " foo", "\n\n", "bar"};
    EXPECT(got == want, "chunks of '%s' differ (got %zu chunks)", s.c_str(), got.size());

    // Chunks always tile the input exactly, for arbitrary bytes.
    Rng rng(51);
    for (int t = 0; t < 200; ++t) {
        std::string r(rng.below(40), '\0');
        for (auto& ch : r) ch = char(rng.below(256));
        size_t pos = 0;
        bool ok = true;
        for (auto [b, e] : Tokenizer::chunks(r)) ok &= b == pos && e > b, pos = e;
        EXPECT(ok && pos == r.size(), "chunks do not tile a random string");
    }

    // Train on a small corpus with repetition, so merges are meaningful.
    std::string corpus;
    const char* words[] = {"the", "reason", "model", "train", "token", "merge", "the", "the", "and"};
    for (int i = 0; i < 3000; ++i) {
        corpus += words[rng.below(9)];
        corpus += rng.below(5) == 0 ? ", " : " ";
        if (rng.below(7) == 0) corpus += std::to_string(rng.below(1000)) + " ";
        if (rng.below(20) == 0) corpus += "\n";
    }
    Tokenizer tok;
    tok.train(corpus, 300);
    EXPECT(tok.vocab_size() == 300 || tok.num_merges() > 20, "only %zu merges learned", tok.num_merges());
    for (int id = 256; id < tok.vocab_size(); ++id) {
        const std::string& b = tok.token_bytes(id);
        bool digit = false;
        for (char ch : b) digit |= ch >= '0' && ch <= '9';
        EXPECT(!digit, "token %d '%s' merged a digit", id, b.c_str());
    }
    const std::vector<int> enc = tok.encode(corpus);
    EXPECT(tok.decode(enc) == corpus, "corpus round trip failed");
    EXPECT(enc.size() * 2 < corpus.size(), "poor compression: %zu tokens for %zu bytes", enc.size(), corpus.size());
    // " the" should have become a single token.
    EXPECT(tok.encode(" the").size() == 1, "' the' is %zu tokens", tok.encode(" the").size());

    // Any bytes round-trip, including ones never seen in training.
    for (int t = 0; t < 200; ++t) {
        std::string r(rng.below(60), '\0');
        for (auto& ch : r) ch = char(rng.below(256));
        EXPECT(tok.decode(tok.encode(r)) == r, "random bytes round trip failed");
    }

    // Save and load give the same encoder.
    const std::string path = "/tmp/cxxgpt_test_tokenizer.bpe";
    tok.save(path);
    Tokenizer tok2;
    tok2.load(path);
    std::remove(path.c_str());
    EXPECT(tok2.vocab_size() == tok.vocab_size() && tok2.encode(corpus) == enc, "reloaded tokenizer differs");

    // Training is deterministic.
    Tokenizer tok3;
    tok3.train(corpus, 300);
    EXPECT(tok3.encode(corpus) == enc, "training is not deterministic");
}
