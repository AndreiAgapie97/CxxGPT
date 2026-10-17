// test_mathgen_number_theory.h — gen-math --topic number-theory
// (src/mathgen_number_theory.h): every problem is checked again here, the
// checker is shown to catch broken problems, and the splits and options
// behave as in the other topics.
#pragma once

#include <set>
#include <sstream>

#include "../src/mathgen_fractions.h"
#include "../src/mathgen_number_theory.h"
#include "test_mathgen.h"
#include "test_util.h"

namespace number_theory_test {

// The numbers the checker uses must each be written in the question (as
// digits; "5th" counts as 5). Not written: the lockers' parity and which
// digit of the puzzle is larger (words).
inline void check_inputs_written(const NumberTheoryProblem& p) {
    std::vector<int64_t> must = p.inputs;
    const bool words_last = (p.kind == NumberTheoryProblem::multiples && p.context == 4) ||
                            (p.kind == NumberTheoryProblem::digits && p.context == 2);
    if (words_last) must.pop_back();
    std::vector<int64_t> written = equation_test::written_numbers(p.problem.question);
    for (int64_t value : must) {
        const auto match = std::find(written.begin(), written.end(), value);
        EXPECT(match != written.end(), "story number %lld is not written in the question: %s", (long long)value,
               p.problem.question.c_str());
        if (match != written.end()) written.erase(match);
    }
}

inline void check_problem(const NumberTheoryProblem& p, bool notes) {
    const Problem& q = p.problem;
    EXPECT(q.task == "number-theory", "wrong task %s", q.task.c_str());
    equation_test::check_singular_after_one(q.question + " " + q.steps);
    EXPECT(q.question.find('{') == std::string::npos && q.question.find('}') == std::string::npos,
           "unfilled placeholder: %s", q.question.c_str());
    EXPECT(extract_answer(q.completion()) == q.answer, "not a GSM8K completion");
    EXPECT(!q.answer.empty() && q.answer.find_first_not_of("0123456789") == std::string::npos && q.answer[0] != '0',
           "answer is not a positive whole number: %s", q.answer.c_str());
    check_inputs_written(p);
    // A person the question names is in the story before it.
    const size_t asked_at = q.question.rfind(". ");
    const std::string story = q.question.substr(0, asked_at), asked = q.question.substr(asked_at);
    for (const std::string& person : {p.name, p.other, p.third})
        EXPECT(asked.find(person) == std::string::npos || story.find(person) != std::string::npos,
               "the question names %s, whom the story does not: %s", person.c_str(), q.question.c_str());
    std::istringstream lines(q.steps);
    for (std::string line; std::getline(lines, line);)
        EXPECT(!line.empty() && line.back() == '.', "solution line does not end a sentence: %s", line.c_str());
    size_t notes_seen = 0;
    for (size_t at = 0; (at = q.steps.find("<<", at)) != std::string::npos; ++at) ++notes_seen;
    EXPECT(notes ? notes_seen == size_t(q.difficulty) && notes_seen >= 1 : notes_seen == 0,
           "difficulty is not the number of steps");
    // Fillers are this topic's own, never gen-word's shared sentences.
    for (const char* shared : word_detail::distractors) {
        // The longer side of the number: " pictures on the wall.", "The room has ".
        const std::string text = shared;
        const size_t at = text.find("{a}");
        const std::string before = text.substr(0, at), after = at == std::string::npos ? "" : text.substr(at + 3);
        const std::string& fixed = before.size() >= after.size() ? before : after;
        EXPECT(q.question.find(fixed) == std::string::npos, "a shared filler in: %s", q.question.c_str());
    }
    if (p.distractor_number) {
        const auto used = equation_test::written_numbers(q.steps + " " + q.answer);
        EXPECT(std::find(used.begin(), used.end(), p.distractor_number) == used.end(),
               "distractor number %lld is used in the solution", (long long)p.distractor_number);
    }
    if (!notes) return;
    EXPECT(number_theory_check::problems_with(p).empty(), "checker rejects: %s", q.question.c_str());
    for (size_t at = 0; (at = q.steps.find("<<", at)) != std::string::npos; ++at) {
        const size_t equals = q.steps.find('=', at), end = q.steps.find(">>", at);
        const auto reply = calc::response(q.steps.substr(0, equals + 1), calc::Mode::notes);
        EXPECT(reply && *reply == q.steps.substr(equals + 1, end - equals - 1) + ">>",
               "the calculator hook disagrees with a note in: %s", q.steps.c_str());
    }
}

// The checker must not pass broken problems. A story number off by one
// can leave the answer as it was (51 cookies fill as many boxes of 6 as
// 50), so each story number is moved one up and one down, and the changes
// the checker catches are counted, as are the problems where it catches
// none: there, every changed story is still right (95, 96 or 97 muffins
// in trays of 18 all need 6 trays).
inline void checker_catches_mistakes(const NumberTheoryProblem& good, int& changed, int& caught, int& unchanged) {
    EXPECT(number_theory_check::problems_with(good).empty(), "checker rejects a good problem");
    NumberTheoryProblem p = good;
    p.problem.answer = good.problem.answer + "1";
    EXPECT(!number_theory_check::problems_with(p).empty(), "checker passed a wrong answer");
    p = good;
    const size_t note = p.problem.steps.find(">>");
    char& digit = p.problem.steps[note - 1];
    digit = digit == '9' ? '0' : char(digit + 1);
    EXPECT(!number_theory_check::problems_with(p).empty(), "checker passed a wrong note");
    p = good;
    p.problem.steps += "\nThat is all.";
    EXPECT(!number_theory_check::problems_with(p).empty(), "checker passed a last line without the answer");
    bool any = false;
    for (size_t i = 0; i < good.inputs.size(); ++i)
        for (int64_t step : {-1, 1}) {
            p = good;
            p.inputs[i] += step;
            const bool rejected = !number_theory_check::problems_with(p).empty();
            ++changed;
            caught += rejected;
            any = any || rejected;
        }
    unchanged += !any;
}

// The checker's answer for a hand-made record, or -1 if it rejects it.
inline int64_t solved(NumberTheoryProblem::Kind kind, int context, int asked, std::vector<int64_t> inputs) {
    NumberTheoryProblem p;
    p.kind = kind;
    p.context = context;
    p.asked = asked;
    p.inputs = std::move(inputs);
    int64_t answer = 0;
    return number_theory_check::solve(p, answer).empty() ? answer : -1;
}

}  // namespace number_theory_test

inline void test_mathgen_number_theory() {
    using number_detail::family_count;
    using number_theory_test::solved;
    using K = NumberTheoryProblem;
    // The checker on stories worked out by hand, and on broken ones.
    EXPECT(solved(K::remainder, 0, 0, {50, 6}) == 8 && solved(K::remainder, 0, 1, {50, 6}) == 2 &&
               solved(K::remainder, 1, 2, {50, 6}) == 9 && solved(K::remainder, 1, 3, {50, 6}) == 4,
           "the remainder check is wrong");
    EXPECT(solved(K::remainder, 0, 1, {48, 6}) == -1, "an exact fit (nothing left over) passed");
    EXPECT(solved(K::lcm, 0, 0, {6, 8}) == 24 && solved(K::lcm, 0, 1, {8, 12, 2}) == 5 &&
               solved(K::lcm, 1, 1, {4, 6, 2}) == 10 && solved(K::lcm, 2, 1, {4, 6, 30}) == 2 &&
               solved(K::lcm, 3, 0, {8, 6}) == 3 &&
               solved(K::lcm, 3, 1, {8, 6}) == 4 && solved(K::lcm, 3, 2, {8, 6}) == 24,
           "the lcm check is wrong");
    EXPECT(solved(K::gcd, 0, 0, {24, 36}) == 12 && solved(K::gcd, 0, 2, {24, 36}) == 3 &&
               solved(K::gcd, 1, 1, {48, 72}) == 5,
           "the gcd check is wrong");
    EXPECT(solved(K::gcd, 0, 0, {12, 36}) == -1, "a gcd story where one number divides the other passed");
    EXPECT(solved(K::factors, 0, 0, {36, 2}) == 7 && solved(K::factors, 1, 0, {12, 2}) == 4 &&
               solved(K::factors, 0, 0, {36, 3}) == 5,
           "the factors check is wrong");
    EXPECT(solved(K::factors, 0, 0, {13, 2}) == -1 && solved(K::factors, 0, 0, {9, 2}) == -1 &&
               solved(K::factors, 0, 0, {16, 3}) == -1,
           "fewer than two arrangements passed");
    EXPECT(solved(K::multiples, 0, 0, {200, 7}) == 28 && solved(K::multiples, 2, 0, {100, 4, 6}) == 8 &&
               solved(K::multiples, 3, 0, {45, 180, 7}) == 19 && solved(K::multiples, 4, 0, {3, 30, 0}) == 14 &&
               solved(K::multiples, 4, 0, {3, 30, 1}) == 14,
           "the multiples check is wrong");
    EXPECT(solved(K::digits, 0, 0, {150}) == 342 && solved(K::digits, 0, 0, {42}) == 75 &&
               solved(K::digits, 1, 0, {342}) == 150 && solved(K::digits, 1, 0, {75}) == 42 &&
               solved(K::digits, 2, 0, {10, 4, 0}) == 73 && solved(K::digits, 2, 0, {10, 4, 1}) == 37,
           "the digits check is wrong");
    EXPECT(solved(K::digits, 2, 0, {10, 3, 0}) == -1, "a digit puzzle without a whole answer passed");
    EXPECT(solved(K::digits, 1, 0, {343}) == -1, "a number of digits that numbers no whole page count passed");
    EXPECT(solved(K::pattern, 0, 0, {3, 2, 47}) == 29 && solved(K::pattern, 0, 1, {3, 2, 47}) == 18 &&
               solved(K::pattern, 1, 1, {5, 2, 30}) == 8,
           "the pattern check is wrong");
    EXPECT(number_detail::ordinal(3) == "3rd" && number_detail::ordinal(11) == "11th" &&
               number_detail::ordinal(12) == "12th" && number_detail::ordinal(21) == "21st" &&
               number_detail::ordinal(22) == "22nd" && number_detail::ordinal(60) == "60th",
           "ordinals are wrong");

    std::set<std::string> questions[2];
    std::set<int> patterns[2];
    std::set<int> kinds_seen[2];
    int checked = 0, changed = 0, caught = 0, unchanged = 0, filled = 0;
    for (bool test : {false, true}) {
        Rng rng(77);
        for (int i = 0; i < 4000; ++i) {
            const auto p = make_number_theory_problem(rng, {true, true, test});
            number_theory_test::check_problem(p, true);
            if (i < 400) number_theory_test::checker_catches_mistakes(p, changed, caught, unchanged);
            questions[test].insert(p.problem.question);
            for (int id : p.patterns) {
                EXPECT((id % 6 >= 4) == test, "pattern %d leaked across splits", id);
                patterns[test].insert(id);
            }
            // Every kind, setting and question asked.
            kinds_seen[test].insert((p.kind * 8 + p.context) * 8 + p.asked);
            if (p.distractor >= 0) EXPECT((p.distractor >= 12) == test, "distractor wording leaked across splits");
            filled += p.distractor >= 0;
            ++checked;
        }
    }
    // Most story numbers moved by one change the answer; in a few problems
    // (8 of 800 with this seed) no such change does.
    EXPECT(caught * 3 >= changed * 2 && unchanged * 50 <= 800,
           "the checker caught only %d of %d story numbers moved by one; none in %d of 800 problems", caught, changed,
           unchanged);
    // Half the problems get a filler.
    EXPECT(filled * 100 >= checked * 45 && filled * 100 <= checked * 55, "%d of %d problems with a filler", filled,
           checked);
    // Story sentences: 4 train and 2 test wordings each; questions and
    // solution lines: 2 and 1.
    using number_detail::phrase_count;
    EXPECT(patterns[0].size() == size_t(family_count) * 4 + size_t(phrase_count) * 2 &&
               patterns[1].size() == size_t(family_count) * 2 + size_t(phrase_count),
           "not all wordings exercised (%zu train, %zu test)", patterns[0].size(), patterns[1].size());
    // remainder 2 x 4, lcm 3 x 2 + 3, gcd 3 + 2, factors 2, multiples 5,
    // digits 3, pattern 2 x 2.
    const size_t asked_total = 8 + 9 + 5 + 2 + 5 + 3 + 4;
    EXPECT(kinds_seen[0].size() == asked_total && kinds_seen[1].size() == asked_total,
           "a kind, setting or question is missing (%zu, %zu)", kinds_seen[0].size(), kinds_seen[1].size());
    for (const auto& q : questions[1]) EXPECT(!questions[0].count(q), "question leaked across splits");
    std::set<std::string> texts;
    for (const auto& family : number_detail::patterns)
        for (const char* pattern : family) EXPECT(texts.insert(pattern).second, "duplicate pattern: %s", pattern);
    for (const auto& phrase : number_detail::phrases)
        for (const char* wording : phrase) EXPECT(texts.insert(wording).second, "duplicate wording: %s", wording);
    for (const auto& family : equation_detail::patterns)
        for (const char* pattern : family) EXPECT(texts.insert(pattern).second, "pattern shared with equations: %s", pattern);
    for (const auto& family : fraction_detail::patterns)
        for (const char* pattern : family) EXPECT(texts.insert(pattern).second, "pattern shared with fractions: %s", pattern);
    for (bool test : {false, true}) {
        Rng noted_rng(9), plain_rng(9), quiet_rng(9);
        for (int i = 0; i < 300; ++i) {
            const auto noted = make_number_theory_problem(noted_rng, {true, true, test});
            const auto plain = make_number_theory_problem(plain_rng, {true, false, test});
            const auto quiet = make_number_theory_problem(quiet_rng, {false, false, test});
            number_theory_test::check_problem(plain, false);
            number_theory_test::check_problem(quiet, false);
            EXPECT(noted.problem.question == plain.problem.question && noted.problem.answer == plain.problem.answer &&
                       math_detail::without_notes(noted.problem.steps) == plain.problem.steps,
                   "notes changed the problem");
            EXPECT(quiet.problem.answer == plain.problem.answer && quiet.problem.steps == plain.problem.steps &&
                       (plain.distractor < 0 ? quiet.problem.question == plain.problem.question
                                             : quiet.problem.question.size() < plain.problem.question.size()),
                   "distractors changed the math");
        }
    }
    const auto a = make_number_theory_problems(31, 200), b = make_number_theory_problems(31, 200);
    bool same = a.size() == b.size();
    for (size_t i = 0; same && i < a.size(); ++i)
        same = a[i].question == b[i].question && a[i].steps == b[i].steps && a[i].answer == b[i].answer;
    EXPECT(same, "the same seed gave other problems");
    std::set<std::string> unique;
    for (const auto& p : a) unique.insert(p.question);
    EXPECT(unique.size() == a.size(), "duplicate questions in one file");
    EXPECT(a[0].question != make_number_theory_problems(32, 1)[0].question, "seed has no effect");
    std::printf("  %d number-theory problems checked by the second solver; all %d sentence and %d question or "
                "solution wordings, both splits; %d with a filler; "
                "%d of %d story numbers moved by one caught (none in %d of 800 problems)\n",
                checked, family_count * 6, phrase_count * 3, filled, caught, changed, unchanged);
}
