// Tests for score_checks.h: a small labelled file through self-train's
// checks, counted per set, half, verdict and label.
#pragma once

#include <fstream>

#include "../src/score_checks.h"
#include "test_util.h"

inline void test_score_checks() {
    using score_checks::Label;
    using self_train::Verdict;
    const std::string path = test_path("labels.jsonl");
    {
        std::ofstream f(path, std::ios::binary);
        // A clean try, kept; a lucky one without the base, dropped as unused.
        f << R"({"set":"textbook","split":"tune","question":"Find the area of a triangle with base 14 inches and height 7 inches.","steps":"","answer":"49","text":"The area is 14 x 7 / 2 = <<14*7/2=49>>49 square inches.\n#### 49\n","label":"clean","minor":false})"
          << "\n";
        f << R"({"set":"textbook","split":"tune","question":"Find the area of a triangle with base 14 inches and height 7 inches.","steps":"","answer":"49","text":"The area is 7 x 7 = <<7*7=49>>49 square inches.\n#### 49\n","label":"lucky","minor":false})"
          << "\n";
        // A calculation never used (minor), kept; a clean try dropped for an outside number.
        f << R"({"set":"textbook","split":"held-out","question":"What is 6 x 4?","steps":"","answer":"24","text":"6 x 4 = <<6*4=24>>24 and 6 + 4 = <<6+4=10>>10.\n#### 24\n","label":"padded","minor":true})"
          << "\r\n";
        f << R"({"set":"textbook","split":"held-out","question":"What is 6 x 4?","steps":"","answer":"24","text":"6 x 4 = <<6*4=24>>24, as 11 + 13 = 24.\n#### 24\n","label":"clean","minor":false})"
          << "\n\n";
        // The problem file's own solution: dropped for being known, not wrong.
        f << R"({"set":"gsm8k-dev","split":"held-out","question":"Tom has 3 bags of 5 apples. How many apples does he have?","steps":"3 * 5 = <<3*5=15>>15","answer":"15","text":"3 * 5 = <<3*5=15>>15\n#### 15\n","label":"clean"})"
          << "\n";
    }
    const auto tries = score_checks::load(path);
    EXPECT(tries.size() == 5 && tries[2].label == score_checks::minor && tries[4].label == score_checks::clean,
           "5 labelled tries, the third minor (got %zu)", tries.size());
    const score_checks::Score s = score_checks::score(tries, {});
    const auto first = std::make_pair(std::string("textbook"), std::string("tune"));
    EXPECT(s.groups.size() == 3 && s.groups[0] == first && s.groups[2].first == "gsm8k-dev",
           "3 groups in file order, got %zu", s.groups.size());
    if (s.groups.size() != 3) return;
    auto count = [&](size_t g, Verdict v, Label l) {
        const auto it = s.counts[g].find(v);
        return it == s.counts[g].end() ? 0L : it->second[l];
    };
    EXPECT(s.kept(0, score_checks::clean) == 1 && s.kept_total(0) == 1 &&
               count(0, Verdict::unused, score_checks::lucky) == 1 && s.clean_dropped(0) == 0,
           "tune: the clean try kept, the lucky one dropped as unused");
    EXPECT(s.kept(1, score_checks::minor) == 1 && s.kept_total(1) == 1 &&
               count(1, Verdict::outside, score_checks::clean) == 1 && s.clean_dropped(1) == 1,
           "held-out: the minor try kept, one clean try dropped as outside");
    EXPECT(count(2, Verdict::gold_copy, score_checks::clean) == 1 && s.clean_dropped(2) == 0,
           "a clean gold copy is not counted as a clean try dropped");
    self_train::Options outside;
    outside.allow_outside = true;
    const score_checks::Score t = score_checks::score(tries, outside);
    EXPECT(t.kept(1, score_checks::clean) == 1 && t.clean_dropped(1) == 0,
           "with outside numbers allowed, the clean try is kept");
}
