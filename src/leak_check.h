// leak_check.h — exact 13-word overlap search, with bounded training-file memory.
#pragma once

#include <algorithm>
#include <array>
#include <fstream>
#include <istream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "reasoning.h"
#include "tokenizer.h"

namespace leak_check {

inline bool is_hex_digit(char ch) {
    return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') || (ch >= 'A' && ch <= 'F');
}

// ASCII letters and digits define words; punctuation and line breaks separate
// them. Cap retained word length at the longest question word. A longer word
// cannot match, even when a training file contains a gigantic unbroken token.
//
// prep-qa's .txt files and JSON Lines files store a line break as the two
// characters \n. So that "apples.\nHow" gives the words "apples", "how" (not
// "nhow"), the escapes \n, \t, \r and \u plus up to four hex digits are
// separators. A backslash before anything else is an ordinary separator and
// the next character is read normally; a backslash after a backslash starts
// a new escape. The stored form "\\x" of a real "\x" therefore gives the same
// words as "\x" itself, whichever side of the check is stored escaped.
template<class Emit>
void words(std::istream& in, size_t max_length, Emit emit) {
    std::array<char, 65536> buffer{};
    std::string word;
    bool too_long = false;
    auto finish = [&] {
        if (too_long || !word.empty()) emit(word, too_long);
        word.clear();
        too_long = false;
    };
    // Kept across reads: an escape may straddle the end of the buffer.
    bool after_backslash = false;
    int hex_left = 0;  // hex digits still allowed after \u
    while (in) {
        in.read(buffer.data(), buffer.size());
        for (std::streamsize i = 0; i < in.gcount(); ++i) {
            char ch = buffer[size_t(i)];
            if (after_backslash) {
                after_backslash = false;
                if (ch == 'n' || ch == 't' || ch == 'r') continue;
                if (ch == 'u') { hex_left = 4; continue; }
            } else if (hex_left > 0) {
                if (is_hex_digit(ch)) { --hex_left; continue; }
                hex_left = 0;
            }
            if (ch == '\\') {
                finish();
                after_backslash = true;
                continue;
            }
            if (ch >= 'A' && ch <= 'Z') ch = char(ch - 'A' + 'a');
            if ((ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9')) {
                if (word.size() < max_length) word += ch;
                else too_long = true;
            } else finish();
        }
    }
    CHECK(!in.bad(), "error reading leak-check input");
    finish();
}

class Index {
    using Run = std::array<size_t, 13>;
    std::unordered_map<std::string, size_t> vocabulary_;
    std::map<Run, std::vector<size_t>> runs_;
    size_t max_word_ = 0;
public:
    std::vector<std::string> questions;

    explicit Index(std::vector<std::string> input) : questions(std::move(input)) {
        for (size_t q = 0; q < questions.size(); ++q) {
            Run run{};
            size_t filled = 0;
            // The index itself may use memory proportional to the questions.
            std::istringstream in(questions[q]);
            words(in, questions[q].size(), [&](const std::string& word, bool) {
                max_word_ = std::max(max_word_, word.size());
                const auto [entry, inserted] = vocabulary_.emplace(word, vocabulary_.size());
                (void)inserted;
                std::move(run.begin() + 1, run.end(), run.begin());
                run.back() = entry->second;
                if (filled < run.size()) ++filled;
                if (filled == run.size()) {
                    auto& owners = runs_[run];
                    if (owners.empty() || owners.back() != q) owners.push_back(q);
                }
            });
        }
    }

    std::vector<bool> scan(std::istream& training) const {
        std::vector<bool> hit(questions.size(), false);
        Run run{};
        size_t filled = 0;
        words(training, max_word_, [&](const std::string& word, bool too_long) {
            const auto entry = vocabulary_.find(word);
            if (too_long || entry == vocabulary_.end()) { filled = 0; return; }
            std::move(run.begin() + 1, run.end(), run.begin());
            run.back() = entry->second;
            if (filled < run.size()) ++filled;
            if (filled == run.size()) {
                const auto found = runs_.find(run);
                if (found != runs_.end())
                    for (size_t q : found->second) hit[q] = true;
            }
        });
        return hit;
    }
};

// A token file read as its text: an input stream that decodes `chunk`
// tokens at a time, so a training shard of 100M tokens scans without its
// text ever held whole. A run of words split across two chunks is still one
// run: Index::scan reads the stream as one text.
class TokenText : public std::streambuf {
    const std::vector<uint16_t>& tokens_;
    const Tokenizer& tok_;
    size_t next_ = 0, chunk_;
    std::string text_;

    int_type underflow() override {
        text_.clear();
        while (text_.empty() && next_ < tokens_.size()) {
            const size_t end = std::min(tokens_.size(), next_ + chunk_);
            text_ = tok_.decode(std::vector<int>(tokens_.begin() + std::ptrdiff_t(next_), tokens_.begin() +
                                                                                          std::ptrdiff_t(end)));
            next_ = end;
        }
        if (text_.empty()) return traits_type::eof();
        setg(text_.data(), text_.data(), text_.data() + text_.size());
        return traits_type::to_int_type(text_[0]);
    }

public:
    TokenText(const std::vector<uint16_t>& tokens, const Tokenizer& tok, size_t chunk = size_t(1) << 20)
        : tokens_(tokens), tok_(tok), chunk_(std::max<size_t>(chunk, 1)) {}
};

// Accept prep-qa's five-column problems files, or one plain question per line.
// In a problems file only the question field is indexed, never its solution.
inline std::vector<std::string> read_questions(const std::string& path) {
    std::ifstream in(path);
    CHECK(in, "cannot open %s", path.c_str());
    std::vector<std::string> result;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (std::count(line.begin(), line.end(), '\t') == 4) {
            const size_t first = line.find('\t'), second = line.find('\t', first + 1);
            const size_t third = line.find('\t', second + 1);
            result.push_back(unescape_field(line.substr(second + 1, third - second - 1)));
        } else result.push_back(line);
    }
    CHECK(!in.bad(), "error reading %s", path.c_str());
    return result;
}

}  // namespace leak_check
