#pragma once

// train --val-sources NAME=FILE,...: the validation loss of each held-out
// file on its own, at every evaluation, so a night shows which part of the
// mix the model gets better at (GSM8K, the word problems, textbooks,
// books, papers) rather than one number for all of them.
//
// A file with a mask holds questions and worked answers (prep-qa, gen-word):
// its loss is over the answers only, and a second number covers the final
// answers alone, the trained tokens after "####", which is what the GSM8K
// exam scores.

#include <algorithm>
#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "common.h"
#include "train.h"

// Each trained token whose bytes come after a "####" in its trained span
// (the mask's run of 1s). The four hashes may be split over several tokens;
// a token holding the fourth hash and more is a final-answer token too.
// `bytes(id)` gives a token's bytes (the tokenizer's vocabulary).
template <class Bytes>
std::vector<uint8_t> final_answer_mask(const TokenData& d, Bytes bytes) {
    CHECK(!d.mask.empty(), "final answers need a file with a mask (questions and answers)");
    std::vector<uint8_t> out(d.tokens.size(), 0);
    int hashes = 0;      // '#' in a row so far in this span
    bool after = false;  // past the "####" of this span
    for (size_t i = 0; i < d.tokens.size(); ++i) {
        if (!d.mask[i]) {  // a question (or the text between documents): a new span starts after it
            hashes = 0;
            after = false;
            continue;
        }
        bool tail = after;  // part of this token comes after the "####"
        for (const char c : bytes(d.tokens[i])) {
            if (after) {
                tail = true;
                break;
            }
            hashes = c == '#' ? hashes + 1 : 0;
            after = hashes >= 4;
        }
        out[i] = tail ? 1 : 0;
    }
    return out;
}

struct ValSource {
    std::string name, path;
    TokenData data;
    TokenData final_only;  // the same tokens, trained only on final answers (Q&A files)
};

// "gsm8k=a.tok,books=b.tok": names and files, each name once.
inline std::vector<std::pair<std::string, std::string>> parse_val_sources(const std::string& spec) {
    std::vector<std::pair<std::string, std::string>> out;
    std::set<std::string> names;
    size_t start = 0;
    while (start < spec.size()) {
        const size_t comma = std::min(spec.find(',', start), spec.size());
        const std::string item = spec.substr(start, comma - start);
        const size_t eq = item.find('=');
        CHECK(eq != std::string::npos && eq > 0 && eq + 1 < item.size(),
              "--val-sources takes NAME=FILE,NAME=FILE... (got \"%s\")", item.c_str());
        const std::string name = item.substr(0, eq);
        CHECK(name.find_first_of(" |") == std::string::npos, "--val-sources: name \"%s\" has a space or |",
              name.c_str());
        CHECK(names.insert(name).second, "--val-sources: name %s twice", name.c_str());
        out.emplace_back(name, item.substr(eq + 1));
        start = comma + 1;
    }
    return out;
}

// One number of one source: part is "answers" and "final" for a Q&A file,
// "text" for plain text.
struct ValSourceLoss {
    std::string name, part;
    size_t targets = 0;
    double loss = 0;
};

// Files up to this many tokens are scored on every token once (no sampling
// noise; the dev sets, the textbooks shard); larger ones on the same random
// windows as --val, `batches` of them.
constexpr size_t kFullLossTokens = 300000;

template <class Forward>
std::vector<ValSourceLoss> score_val_source(Forward forward, const ValSource& s, int B, int Tn, int batches) {
    std::vector<ValSourceLoss> out;
    const size_t n = s.data.tokens.size();
    const bool full = n <= kFullLossTokens;
    auto score = [&](const TokenData& d, const char* part) {
        ValSourceLoss r{s.name, part};
        r.loss = full ? full_loss(forward, d, 0, n, B, Tn, &r.targets)
                      : sampled_loss(forward, d, 0, n, B, Tn, batches, &r.targets);
        if (r.targets) out.push_back(r);
    };
    if (s.data.mask.empty()) {
        score(s.data, "text");
    } else {
        score(s.data, "answers");
        score(s.final_only, "final");
    }
    return out;
}
