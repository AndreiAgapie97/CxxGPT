#pragma once

#include <regex>
#include <sstream>

#include "../src/leak_check.h"
#include "test_util.h"

// Independent reference: tokenize the complete small strings with a regex,
// then exhaustively compare every pair of 13-word slices.
inline std::vector<std::string> leak_reference_words(std::string text) {
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
        "!!!", "", q + " " + q};
    for (const auto& text : training) {
        std::istringstream in(text);
        const auto hits = index.scan(in);
        for (size_t i = 0; i < questions.size(); ++i)
            EXPECT(hits[i] == leak_reference(questions[i], text), "13-word search differs from reference");
    }
    std::istringstream huge("one two three four five six " + std::string(200000, 'x') +
                            " seven eight nine ten eleven twelve 13");
    const auto huge_hits = index.scan(huge);
    EXPECT(std::none_of(huge_hits.begin(), huge_hits.end(), [](bool hit) { return hit; }),
           "a huge word must break the matching run");
    // Random punctuation/case and planted slices, checked by exhaustive search.
    Rng rng(9127);
    for (int trial = 0; trial < 100; ++trial) {
        std::string question, text;
        for (int i = 0; i < 20; ++i) question += " word" + std::to_string(rng.below(6));
        for (int i = 0; i < 35; ++i) text += (i % 3 ? "!WORD" : "\nword") + std::to_string(rng.below(6));
        if (trial % 2) text += question;
        const leak_check::Index random_index({question});
        std::istringstream in(text);
        EXPECT(random_index.scan(in)[0] == leak_reference(question, text), "random overlap reference");
    }
    std::istringstream empty_training(q);
    EXPECT(leak_check::Index({}).scan(empty_training).empty(), "empty question index");
}
