#pragma once

#include <regex>
#include <sstream>

#include "../src/leak_check.h"
#include "test_util.h"

// Independent reference: blank out the escapes \n, \t, \r and \u plus up to
// four hex digits, tokenize the complete small strings with a regex, then
// exhaustively compare every pair of 13-word slices.
inline std::vector<std::string> leak_reference_words(std::string text) {
    text = std::regex_replace(text, std::regex(R"(\\(?:[ntr]|u[0-9a-fA-F]{0,4}))"), " ");
    for (char& ch : text) if (ch >= 'A' && ch <= 'Z') ch = char(ch - 'A' + 'a');
    const std::regex word("[a-z0-9]+");
    std::vector<std::string> result;
    for (auto it = std::sregex_iterator(text.begin(), text.end(), word); it != std::sregex_iterator(); ++it)
        result.push_back(it->str());
    return result;
}

inline bool leak_reference(const std::string& question, const std::string& training) {
    const auto q = leak_reference_words(question), t = leak_reference_words(training);
    for (size_t i = 0; i + 13 <= q.size(); ++i)
        for (size_t j = 0; j + 13 <= t.size(); ++j)
            if (std::equal(q.begin() + i, q.begin() + i + 13, t.begin() + j)) return true;
    return false;
}

inline void test_leak_check() {
    const std::string q = "One two three four five six seven eight nine ten eleven twelve 13";
    const std::vector<std::string> questions = {q, q + " extra", "one two three", "other " + q};
    const leak_check::Index index(questions);
    const std::vector<std::string> training = {
        q, "One two three four five six seven eight nine ten eleven twelve", // only twelve
        "ONE,two three four five six\nseven eight nine ten eleven twelve 13!",
        "one two three four five six wrong seven eight nine ten eleven twelve 13",
        std::string(65530, '!') + q, // a word straddles the read-buffer boundary
        "!!!", "", q + " " + q,
        // Stored line breaks and other escapes separate words, as real ones do.
        "one two three four five six\\nseven eight nine ten\\televen\\rtwelve\\u00a013",
        "one two three four five six\\\\nseven eight nine ten eleven twelve 13",  // \\n: \ then \n
        "one two three four five six\\seven eight nine ten eleven twelve 13",  // \s: \ then s
        // The backslash is the buffer's last byte, its "n" the next read's first.
        std::string(65535, '!') + "\\n" + q};
    for (const auto& text : training) {
        std::istringstream in(text);
        const auto hits = index.scan(in);
        for (size_t i = 0; i < questions.size(); ++i)
            EXPECT(hits[i] == leak_reference(questions[i], text), "13-word search differs from reference");
    }
    // The reference changed with the escapes too, so check the cases directly.
    // "-> bool": vector<bool>'s [] returns a proxy into the temporary vector.
    auto first_hit = [&](const std::string& text) -> bool {
        std::istringstream in(text);
        return index.scan(in)[0];
    };
    EXPECT(first_hit("One two three four five six\\nseven eight nine ten eleven twelve 13"),
           "\\n between six and seven must separate them");
    EXPECT(first_hit(std::string(65535, '!') + "\\n" + q), "an escape split across two reads");
    EXPECT(!first_hit("One two three four five six\\nnseven eight nine ten eleven twelve 13"),
           "only one letter after the backslash belongs to the escape");
    // The longest question word is 6 letters ("eleven", "twelve"), so words
    // are cut there. "twelve" followed by 200000 x's is cut to "twelve" plus
    // the too-long mark; it must still break the run: without the mark, the
    // run "one ... twelve 13" would match.
    std::istringstream huge("one two three four five six seven eight nine ten eleven twelve" +
                            std::string(200000, 'x') + " 13");
    const auto huge_hits = index.scan(huge);
    EXPECT(std::none_of(huge_hits.begin(), huge_hits.end(), [](bool hit) { return hit; }),
           "a huge word must break the matching run");
    // Random punctuation, case and escapes, and planted slices, checked by
    // exhaustive search. Words such as "cafe" are hex digits, which \u eats.
    Rng rng(9127);
    const char* vocabulary[] = {"cafe", "Add", "bee", "12", "w9", "fed"};
    const char* separators[] = {" ", "!", "\n", "\\n", "\\t", "\\r", "\\", "\\\\", "\\u", "\\u0", "\\u00e9",
                                "\\x", "\\N"};
    auto random_text = [&](int length) {
        std::string text;
        for (int i = 0; i < length; ++i)
            text += std::string(separators[rng.below(13)]) + vocabulary[rng.below(6)];
        return text;
    };
    for (int trial = 0; trial < 300; ++trial) {
        const std::string question = random_text(20);
        std::string text = random_text(35);
        if (trial % 2) text += " " + question;
        const leak_check::Index random_index({question});
        std::istringstream in(text);
        EXPECT(random_index.scan(in)[0] == leak_reference(question, text), "random overlap reference");
    }
    std::istringstream empty_training(q);
    EXPECT(leak_check::Index({}).scan(empty_training).empty(), "empty question index");
}
