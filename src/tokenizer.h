// tokenizer.h — byte-level BPE (byte pair encoding).
//
// Tokens 0..255 are the raw bytes, so any byte string encodes and decodes
// exactly. Training learns merges: the most frequent adjacent pair of tokens
// becomes a new token, repeatedly, until the vocabulary is full.
//
// Text is first split into chunks, and merges never cross a chunk:
//   - a word: an optional leading space plus a run of letters (bytes >= 0x80
//     count as letters, so UTF-8 characters stay inside words);
//   - each digit alone: digits are never merged, so numbers are always
//     spelled digit by digit (much easier arithmetic for the model);
//   - punctuation: an optional leading space plus a run of other symbols;
//   - whitespace: a run of it (a single space before a word or punctuation
//     joins that word instead).
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <queue>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common.h"

class Tokenizer {
public:
    Tokenizer() { reset_vocab(); }

    int vocab_size() const { return int(vocab_.size()); }
    size_t num_merges() const { return merges_.size(); }
    const std::string& token_bytes(int id) const { return vocab_[id]; }

    // Calls fn(begin, end) for each chunk of s (see above), in order.
    template <class Fn>
    static void for_each_chunk(std::string_view s, Fn&& fn) {
        const size_t n = s.size();
        size_t i = 0;
        while (i < n) {
            const size_t start = i;
            const unsigned char c = s[i];
            if (is_digit(c)) {
                ++i;
            } else if (c == ' ' && i + 1 < n && !is_space(s[i + 1]) && !is_digit(s[i + 1])) {
                i = scan_run(s, i + 1);  // " word" or " ..."
            } else if (is_space(c)) {
                size_t j = i;
                while (j < n && is_space(s[j])) ++j;
                // A final ' ' before a word or punctuation leads that instead.
                if (j < n && j - i > 1 && s[j - 1] == ' ' && !is_digit(s[j])) --j;
                i = j;
            } else {
                i = scan_run(s, i);
            }
            fn(start, i);
        }
    }

    // The first position p >= from where a line starts with something other
    // than whitespace (s[p-1] is '\n'), or s.size() if there is none. Such a
    // p always starts a chunk, whatever came before, so text cut there
    // encodes exactly as the whole does: big files can be encoded in
    // independent pieces, in parallel.
    static size_t next_safe_split(std::string_view s, size_t from) {
        for (size_t p = std::max<size_t>(from, 1); p < s.size(); ++p)
            if (s[p - 1] == '\n' && !is_space(s[p])) return p;
        return s.size();
    }

    static std::vector<std::pair<size_t, size_t>> chunks(std::string_view s) {
        std::vector<std::pair<size_t, size_t>> out;
        for_each_chunk(s, [&](size_t b, size_t e) { out.push_back({b, e}); });
        return out;
    }

    // Learns merges from text until the vocabulary has vocab_size tokens.
    void train(const std::string& text, int vocab_size, bool verbose = false) {
        CHECK(vocab_size >= 256, "vocab_size must be at least 256");
        reset_vocab();
        merges_.clear();
        // Unique chunks with their counts; each as a token sequence.
        std::unordered_map<std::string_view, long> freq;
        const std::string_view tv(text);
        for_each_chunk(tv, [&](size_t b, size_t e) { ++freq[tv.substr(b, e - b)]; });
        std::vector<std::vector<int>> words;
        std::vector<long> counts;
        for (const auto& [w, c] : freq) {
            if (w.size() < 2) continue;  // nothing to merge
            words.emplace_back(w.begin(), w.end());
            for (int& t : words.back()) t &= 0xFF;
            counts.push_back(c);
        }
        // Pair counts, where each pair occurs, and a max-heap with lazy
        // deletion (an entry is stale if its count no longer matches).
        std::unordered_map<uint64_t, long> pair_count;
        std::unordered_map<uint64_t, std::vector<int>> where;
        for (int w = 0; w < int(words.size()); ++w)
            for (size_t j = 0; j + 1 < words[w].size(); ++j) {
                const uint64_t p = key(words[w][j], words[w][j + 1]);
                pair_count[p] += counts[w];
                where[p].push_back(w);
            }
        using Entry = std::pair<long, uint64_t>;
        auto cmp = [](const Entry& a, const Entry& b) {
            return a.first != b.first ? a.first < b.first : a.second > b.second;  // ties: lowest pair
        };
        std::priority_queue<Entry, std::vector<Entry>, decltype(cmp)> heap(cmp);
        for (const auto& [p, c] : pair_count) heap.push({c, p});

        while (vocab_size_now() < vocab_size && !heap.empty()) {
            const auto [c, p] = heap.top();
            heap.pop();
            auto it = pair_count.find(p);
            if (it == pair_count.end() || it->second != c) continue;  // stale
            if (c < 2) break;  // merging a pair seen once teaches nothing
            const int a = int(p >> 32), b = int(p & 0xFFFFFFFF), id = vocab_size_now();
            merges_.push_back({a, b});
            vocab_.push_back(vocab_[a] + vocab_[b]);
            // Rewrite every word containing the pair, updating pair counts.
            std::vector<int> ws = std::move(where[p]);
            where.erase(p);
            std::sort(ws.begin(), ws.end());
            ws.erase(std::unique(ws.begin(), ws.end()), ws.end());
            std::unordered_map<uint64_t, long> changed;
            for (int w : ws) {
                auto& t = words[w];
                bool has = false;
                for (size_t j = 0; j + 1 < t.size(); ++j) has |= t[j] == a && t[j + 1] == b;
                if (!has) continue;
                for (size_t j = 0; j + 1 < t.size(); ++j) changed[key(t[j], t[j + 1])] -= counts[w];
                std::vector<int> merged;
                for (size_t j = 0; j < t.size(); ++j) {
                    if (j + 1 < t.size() && t[j] == a && t[j + 1] == b) {
                        merged.push_back(id);
                        ++j;
                    } else {
                        merged.push_back(t[j]);
                    }
                }
                t = std::move(merged);
                for (size_t j = 0; j + 1 < t.size(); ++j) {
                    const uint64_t q = key(t[j], t[j + 1]);
                    changed[q] += counts[w];
                    where[q].push_back(w);
                }
            }
            for (const auto& [q, d] : changed) {
                if (d == 0) continue;
                long& cnt = pair_count[q];
                cnt += d;
                if (cnt <= 0) pair_count.erase(q);
                else heap.push({cnt, q});
            }
            if (verbose && merges_.size() % 500 == 0)
                std::printf("  %zu merges, last '%s' (%ld)\n", merges_.size(), vocab_.back().c_str(), c);
        }
        build_ranks();
    }

    std::vector<int> encode(std::string_view tv) const {
        std::vector<int> out;
        for_each_chunk(tv, [&](size_t b, size_t e) {
            const std::string_view w = tv.substr(b, e - b);
            if (w.size() == 1) return out.push_back(int((unsigned char)w[0]));
            auto it = cache_.find(w);
            if (it == cache_.end()) {
                if (cache_.size() >= kMaxCache) cache_.clear();  // bounded memory on huge corpora
                it = cache_.emplace(std::string(w), encode_chunk(w)).first;
            }
            out.insert(out.end(), it->second.begin(), it->second.end());
        });
        return out;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string s;
        for (int id : ids) {
            CHECK(id >= 0 && id < vocab_size(), "token id %d out of range", id);
            s += vocab_[id];
        }
        return s;
    }

    // Text file: a header line, the merge count, then one "a b" pair per line.
    void save(const std::string& path) const {
        FILE* f = std::fopen(path.c_str(), "w");
        CHECK(f, "cannot write %s", path.c_str());
        std::fprintf(f, "cxxgpt-bpe 1\n%zu\n", merges_.size());
        for (auto [a, b] : merges_) std::fprintf(f, "%d %d\n", a, b);
        CHECK(std::fclose(f) == 0, "error writing %s", path.c_str());
    }

    void load(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "r");
        CHECK(f, "cannot open %s", path.c_str());
        int version = 0;
        size_t n = 0;
        CHECK(std::fscanf(f, "cxxgpt-bpe %d %zu", &version, &n) == 2 && version == 1,
              "%s is not a tokenizer file", path.c_str());
        reset_vocab();
        merges_.clear();
        for (size_t i = 0; i < n; ++i) {
            int a, b;
            CHECK(std::fscanf(f, "%d %d", &a, &b) == 2, "truncated tokenizer file %s", path.c_str());
            CHECK(a >= 0 && b >= 0 && a < vocab_size_now() && b < vocab_size_now(), "bad merge in %s", path.c_str());
            merges_.push_back({a, b});
            vocab_.push_back(vocab_[a] + vocab_[b]);
        }
        std::fclose(f);
        build_ranks();
    }

private:
    static bool is_digit(unsigned char c) { return c >= '0' && c <= '9'; }
    static bool is_space(unsigned char c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r'; }
    static bool is_letter(unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80;
    }
    // A run of letters, or a run of punctuation, starting at i.
    static size_t scan_run(std::string_view s, size_t i) {
        const size_t n = s.size();
        if (i >= n) return i;
        if (is_letter(s[i])) {
            while (i < n && is_letter(s[i])) ++i;
        } else if (!is_digit(s[i]) && !is_space(s[i])) {
            while (i < n && !is_letter(s[i]) && !is_digit(s[i]) && !is_space(s[i])) ++i;
        }
        return i;
    }
    static uint64_t key(int a, int b) { return (uint64_t(uint32_t(a)) << 32) | uint32_t(b); }

    int vocab_size_now() const { return int(vocab_.size()); }

    void reset_vocab() {
        vocab_.clear();
        for (int i = 0; i < 256; ++i) vocab_.push_back(std::string(1, char(i)));
        cache_.clear();
    }

    void build_ranks() {
        ranks_.clear();
        for (size_t i = 0; i < merges_.size(); ++i) ranks_[key(merges_[i].first, merges_[i].second)] = int(i);
        cache_.clear();
    }

    // Apply merges in the order they were learned: repeatedly merge the
    // adjacent pair with the lowest rank.
    std::vector<int> encode_chunk(std::string_view w) const {
        std::vector<int> t(w.begin(), w.end());
        for (int& x : t) x &= 0xFF;
        while (t.size() > 1) {
            int best = -1;
            size_t at = 0;
            for (size_t j = 0; j + 1 < t.size(); ++j) {
                auto it = ranks_.find(key(t[j], t[j + 1]));
                if (it != ranks_.end() && (best < 0 || it->second < best)) best = it->second, at = j;
            }
            if (best < 0) break;
            const int id = 256 + best;
            const auto [a, b] = merges_[best];
            std::vector<int> merged;
            for (size_t j = 0; j < t.size(); ++j) {
                if (j >= at && j + 1 < t.size() && t[j] == a && t[j + 1] == b) {
                    merged.push_back(id);
                    ++j;
                } else {
                    merged.push_back(t[j]);
                }
            }
            t = std::move(merged);
        }
        return t;
    }

    std::vector<std::string> vocab_;             // id -> bytes
    std::vector<std::pair<int, int>> merges_;    // merge i makes token 256 + i
    std::unordered_map<uint64_t, int> ranks_;    // pair -> merge index
    // Encoded chunks. Heterogeneous lookup lets a string_view find a key
    // without building a std::string first.
    struct Hash {
        using is_transparent = void;
        size_t operator()(std::string_view v) const { return std::hash<std::string_view>{}(v); }
    };
    static constexpr size_t kMaxCache = size_t(1) << 19;
    mutable std::unordered_map<std::string, std::vector<int>, Hash, std::equal_to<>> cache_;
};
