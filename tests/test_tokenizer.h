// test_tokenizer.h — BPE: chunking, exact round trips, digits stay single.
#pragma once

#include "../src/tokenizer.h"
#include "test_util.h"

inline void test_tokenizer() {
    const std::string adjacent = "abc123def-4.5(6)!7+8=9 000xyz\n";
    std::string adjacent_corpus;
    for (int i = 0; i < 100; ++i) adjacent_corpus += adjacent;
    Tokenizer digits; digits.train(adjacent_corpus, 320);
    for (int id = 256; id < digits.vocab_size(); ++id)
        for (char c : digits.token_bytes(id)) EXPECT(c < '0' || c > '9', "adjacent digit entered a BPE merge");
    const auto adjacent_tokens = digits.encode(adjacent);
    size_t digit_tokens = 0;
    for (int id : adjacent_tokens) digit_tokens += id >= '0' && id <= '9';
    const size_t digit_bytes = std::count_if(adjacent.begin(), adjacent.end(), [](char c) { return c >= '0' && c <= '9'; });
    EXPECT(digit_tokens == digit_bytes && digits.decode(adjacent_tokens) == adjacent, "adjacent digits are not standalone exact tokens");
    const std::string a = " alpha beta123!\n", b = " beta alpha456?\n";
    std::string ab, ba;
    for (int i = 0; i < 100; ++i) { ab += a+b; ba += b+a; }
    Tokenizer first, reordered; first.train(ab,320); reordered.train(ba,320);
    EXPECT(first.num_merges() == reordered.num_merges(), "BPE merge count depends on chunk order");
    for (int id = 0; id < std::min(first.vocab_size(), reordered.vocab_size()); ++id)
        EXPECT(first.token_bytes(id) == reordered.token_bytes(id), "BPE vocabulary depends on equal-count chunk order");

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
    const std::string path = test_path("tokenizer.bpe");
    tok.save(path);
    Tokenizer tok2;
    tok2.load(path);
    std::remove(path.c_str());
    EXPECT(tok2.vocab_size() == tok.vocab_size() && tok2.encode(corpus) == enc, "reloaded tokenizer differs");

    // Text cut at next_safe_split points encodes exactly as the whole, even
    // when the text is mostly whitespace, newlines, digits and symbols.
    const char alphabet[] = "  \n\n\t\r\rab9,. ";
    for (int t = 0; t < 300; ++t) {
        std::string r(rng.below(80), '\0');
        for (auto& ch : r) ch = alphabet[rng.below(sizeof alphabet - 1)];
        if (t % 3 == 0) r = corpus.substr(rng.below(corpus.size() - 200), 200);
        std::vector<int> pieces;
        for (size_t b = 0; b < r.size();) {
            const size_t from = b + 1 + rng.below(8);
            const size_t e = Tokenizer::next_safe_split(r, from);
            // text2tok scans from the previous cut; that must find the same cut.
            EXPECT(Tokenizer::next_safe_split(r, from, b) == e, "split search from the previous cut differs");
            for (int id : tok.encode(r.substr(b, e - b))) pieces.push_back(id);
            b = e;
        }
        EXPECT(pieces == tok.encode(r), "encoding in pieces differs from encoding the whole");
    }

    // One-line boundaries and adversarial trailing space are also safe.
    for (const std::string text : {" alpha beta gamma delta", "word  next!  ", "a1b2c3d4e5"}) {
        const size_t cut = Tokenizer::next_safe_split(text, text.size()/2);
        auto pieces = tok.encode(text.substr(0,cut)); const auto tail = tok.encode(text.substr(cut));
        pieces.insert(pieces.end(),tail.begin(),tail.end());
        EXPECT(pieces == tok.encode(text), "lexical split changes whole-text token ids");
    }
    // Training is deterministic.
    Tokenizer tok3;
    tok3.train(corpus, 300);
    EXPECT(tok3.encode(corpus) == enc, "training is not deterministic");

    // A chunk over kMaxChunkBytes: training skips it (same merges as without
    // it), encoding writes it as plain bytes, and both count it.
    const std::string huge = " " + std::string(Tokenizer::kMaxChunkBytes + 1, 'a');
    Tokenizer tok4;
    tok4.train(corpus + huge, 300);
    EXPECT(tok4.long_chunks() == 1, "training did not count the long chunk");
    EXPECT(tok4.num_merges() == tok.num_merges(), "a skipped long chunk changed the merges");
    for (int id = 0; id < tok.vocab_size(); ++id)
        EXPECT(tok4.token_bytes(id) == tok.token_bytes(id), "a skipped long chunk changed the vocabulary");
    Tokenizer tok5 = tok;
    const auto with_huge = tok5.encode("one two" + huge + " three");
    std::vector<int> expected = tok.encode("one two");
    for (unsigned char c : huge) expected.push_back(int(c));
    for (int id : tok.encode(" three")) expected.push_back(id);
    EXPECT(with_huge == expected, "a long chunk is not encoded as plain bytes");
    EXPECT(tok5.long_chunks() == 1, "encoding did not count the long chunk");
}
