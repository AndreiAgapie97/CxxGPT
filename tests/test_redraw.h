// test_redraw.h — generated problems written again with other numbers
// (redraw.h): recipe files, versions that keep the story and change the
// numbers, and the generators' own solutions passing the re-draw check.
#pragma once

#include "../src/redraw.h"
#include "../src/self_train.h"
#include "test_util.h"

namespace {

inline void test_redraw_recipes() {
    const auto recipes_of = [](const std::string& topic, uint64_t seed, int64_t count, bool test) {
        std::vector<Recipe> recipes;
        if (topic == "word") make_word_problems(seed, count, {2, 8, true, true, test}, &recipes);
        else if (topic == "equations") make_equation_problems(seed, count, {true, true, test}, &recipes);
        else if (topic == "fractions") make_fraction_problems(seed, count, {false, true, test}, &recipes);
        else make_number_theory_problems(seed, count, {true, true, test}, &recipes);
        return recipes;
    };
    std::vector<Recipe> all;
    for (const char* topic : {"word", "equations", "fractions", "number-theory"}) {
        const std::vector<Recipe> r = recipes_of(topic, 3, 20, std::string(topic) == "fractions");
        EXPECT(r.size() == 20, "%s: %zu recipes for 20 problems", topic, r.size());
        all.insert(all.end(), r.begin(), r.end());
    }
    const std::string path = test_path("recipes.txt");
    redraw::save_recipes(path, all);
    const std::vector<Recipe> back = redraw::load_recipes(path);
    EXPECT(back.size() == all.size(), "%zu recipes read back for %zu", back.size(), all.size());
    for (size_t i = 0; i < std::min(back.size(), all.size()); ++i)
        EXPECT(redraw::recipe_line(back[i]) == redraw::recipe_line(all[i]), "recipe %zu read back as %s", i,
               redraw::recipe_line(back[i]).c_str());
    EXPECT(back.size() > 50 && back[50].topic == "fractions" && back[50].test && !back[50].distractors,
           "the fractions recipes keep their options");
    EXPECT(back.size() > 70 && back[70].topic == "number-theory" && !back[70].test && back[70].distractors,
           "the number-theory recipes keep their options");
    const std::string good = redraw::recipe_line(all[0]);
    EXPECT(redraw::parse_recipe(good).has_value(), "'%s' must read", good.c_str());
    for (const std::string& bad :
         {std::string("sums distractors=1 test=0 steps=5 shape=2 state=1,2,3,4"),
          std::string("word distractors=2 test=0 steps=5 shape=2 state=1,2,3,4"),
          std::string("word distractors=1 test=0 steps=9 shape=2 state=1,2,3,4"),
          std::string("word distractors=1 test=0 steps=5 shape=6 state=1,2,3,4"),
          std::string("equations distractors=1 test=0 steps=0 shape=7 state=1,2,3,4"),
          std::string("number-theory distractors=1 test=0 steps=0 shape=7 state=1,2,3,4"),
          std::string("word distractors=1 test=0 steps=5 shape=2 state=0,0,0,0"),
          std::string("word distractors=1 test=0 steps=5 shape=2 state=1,2,3"), good + " more", std::string("")})
        EXPECT(!redraw::parse_recipe(bad).has_value(), "'%s' must not read", bad.c_str());
}

inline void test_redraw_masked() {
    EXPECT(redraw::masked("Maya has 12 boxes of $2.50 and 1,200 cards; 1/3 of 7.") ==
               "Maya has # boxes of $# and # cards; #/# of #.",
           "masked: '%s'", redraw::masked("Maya has 12 boxes of $2.50 and 1,200 cards; 1/3 of 7.").c_str());
    EXPECT(redraw::masked("No numbers here.") == "No numbers here.", "a question without numbers stays");
}

// Every topic, train and test split: most problems get three versions;
// each has the original's words with other numbers (and, for gen-math, is
// accepted by the topic's second solver: make() asks it); the same seed
// gives the same versions. The generator's own solution, as a try, passes
// the re-draw check wherever the check can follow it, and in word problems
// (every step a calculator note) it can almost always.
inline void test_redraw_versions() {
    struct Topic {
        const char* name;
        bool test;
    };
    for (const Topic topic : {Topic{"word", false}, Topic{"word", true}, Topic{"equations", false},
                              Topic{"equations", true}, Topic{"fractions", false}, Topic{"fractions", true},
                              Topic{"number-theory", false}, Topic{"number-theory", true}}) {
        std::vector<Recipe> recipes;
        std::vector<Problem> problems;
        const std::string name = topic.name;
        if (name == "word") problems = make_word_problems(11, 80, {2, 8, true, true, topic.test}, &recipes);
        else if (name == "equations") problems = make_equation_problems(11, 80, {true, true, topic.test}, &recipes);
        else if (name == "fractions") problems = make_fraction_problems(11, 80, {true, true, topic.test}, &recipes);
        else problems = make_number_theory_problems(11, 80, {true, true, topic.test}, &recipes);
        size_t full = 0, strict = 0, passed = 0, plain_passed = 0;
        for (size_t i = 0; i < problems.size(); ++i) {
            const Problem& p = problems[i];
            const redraw::Versions v = redraw::versions_of(recipes[i], p, 3, 1000 + i);
            full += v.versions.size() == 3;
            const std::string mask = redraw::masked(p.question);
            std::set<std::string> questions = {p.question};
            for (const Problem& q : v.versions) {
                EXPECT(redraw::masked(q.question) == mask, "%s %zu: a version with other words:\n%s\n%s", topic.name,
                       i, p.question.c_str(), q.question.c_str());
                EXPECT(questions.insert(q.question).second, "%s %zu: a version repeats a question", topic.name, i);
                EXPECT(q.task == p.task && redraw::masked(q.steps) == redraw::masked(p.steps),
                       "%s %zu: a version with another plan:\n%s\n%s", topic.name, i, p.steps.c_str(), q.steps.c_str());
            }
            const redraw::Versions again = redraw::versions_of(recipes[i], p, 3, 1000 + i);
            EXPECT(again.versions.size() == v.versions.size() && again.draws == v.draws &&
                       (v.versions.empty() || again.versions.back().question == v.versions.back().question),
                   "%s %zu: the same seed must give the same versions", topic.name, i);
            if (v.versions.size() != 3) continue;
            const labels::AnswerLabel label = labels::label_answer(p, p.completion());
            const self_train::Trace t = self_train::trace_try(p, label, v.versions);
            strict += t.passed && !t.ambiguous;
            passed += t.passed;
            if (!t.passed && name != "fractions" && name != "number-theory")
                std::printf("  %s %zu: the gold solution misses a version (%s, %zu readings):\n%s\n", topic.name, i,
                            t.formula.c_str(), t.readings, p.completion().c_str());
            // The same solution without its notes ("--notes off", and how rp1
            // writes): plain arithmetic, some of it after a word ("30% of 270
            // is 270*30/100 = 81"), must be followed as well.
            Problem plain = p;
            plain.steps = math_detail::without_notes(p.steps);
            const self_train::Trace tp = self_train::trace_try(p, labels::label_answer(p, plain.completion()), v.versions);
            plain_passed += tp.passed;
            if (!tp.passed && t.passed)
                std::printf("  %s %zu: passes with notes, not without (%s):\n%s\n", topic.name, i, tp.formula.c_str(),
                            plain.completion().c_str());
        }
        std::printf("  %s (%s): %zu of %zu problems with 3 versions; gold solutions passing: %zu with one "
                    "reading, %zu with every reading tried, %zu without notes\n",
                    topic.name, topic.test ? "test" : "train", full, problems.size(), strict, passed, plain_passed);
        EXPECT(plain_passed == passed, "%s: %zu gold solutions pass with notes, %zu without", topic.name, passed,
               plain_passed);
        EXPECT(full * 10 >= problems.size() * 9, "%s: only %zu of %zu problems got 3 versions", topic.name, full,
               problems.size());
        // Word problems and equations: every gold solution has a reading
        // that passes. Fractions: a long run of halves can have more
        // combinations than are followed (counted ambiguous). Number
        // theory: most solutions write a number the calculations do not
        // make (the 28 of "7*28 = 196 and 7*29 = 203 is more than 200",
        // a common multiple found by counting up, the first even locker),
        // which the formula keeps as a constant, so other numbers give
        // another answer; only the plain sums pass (README, gen-math
        // number theory).
        if (name == "number-theory")
            EXPECT(passed > 0, "number-theory: none of %zu gold solutions pass", full);
        else
            EXPECT(name == "fractions" ? passed * 50 >= full * 49 : passed == full,
                   "%s: the gold solutions must pass with one of their readings (%zu of %zu)", topic.name, passed,
                   full);
        EXPECT(strict > 0, "%s: some gold solutions must pass with one reading", topic.name);
    }
}

// The story choices are only recorded, never changed: a problem made while
// recording is the same problem.
inline void test_redraw_record() {
    Rng rng(42);
    const WordProblem x = make_word_problem(rng);
    Rng from(1);
    from.set_state(x.recipe.state);
    std::vector<Choices::Choice> story;
    Choices record(from, &story);
    WordProblem y;
    y.shape = x.shape;
    EXPECT(word_detail::candidate(record, {}, x.recipe.steps, y) && y.problem.question == x.problem.question &&
               y.problem.steps == x.problem.steps && !story.empty(),
           "the recipe, recorded, must make the same problem");
    // Playing back: a choice over another range, or one too few or too
    // many, is noticed.
    Rng numbers(1);
    const std::vector<Choices::Choice> two = {{5, 1}, {3, 2}};
    Choices exact(two, numbers), other(two, numbers), short_(two, numbers), long_(two, numbers);
    EXPECT(exact.below(5) == 1 && exact.below(3) == 2 && exact.played_back(), "the recorded choices, in order");
    other.below(5);
    other.below(4);
    EXPECT(!other.played_back(), "a choice over another range is not the story");
    short_.below(5);
    EXPECT(!short_.played_back(), "a story that stops early is not the story");
    long_.below(5);
    long_.below(3);
    long_.below(2);
    EXPECT(!long_.played_back(), "a story that goes on is not the story");
}

inline void test_redraw() {
    test_redraw_recipes();
    test_redraw_masked();
    test_redraw_record();
    test_redraw_versions();
}

}  // namespace
