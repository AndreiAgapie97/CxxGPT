// test_mathgen_fractions.h — gen-math --topic fractions (src/mathgen_fractions.h):
// every problem is checked again here, the checker is shown to catch broken
// problems, and the splits and options behave as in the equations topic.
#pragma once

#include <set>
#include <sstream>

#include "../src/mathgen_fractions.h"
#include "test_mathgen.h"
#include "test_util.h"

namespace fraction_test {

// The numbers written in a text, in hundredths: "$12.50" and "12.5" are
// 1250, "7" is 700, and "3:5" is 300 and 500.
inline std::vector<int64_t> written_hundredths(const std::string& text) {
    std::vector<int64_t> found;
    auto digit = [&](size_t at) { return at < text.size() && text[at] >= '0' && text[at] <= '9'; };
    for (size_t at = 0; at < text.size(); ++at) {
        if (!digit(at)) continue;
        int64_t whole = 0, cents = 0, places = 0;
        while (digit(at)) whole = whole * 10 + (text[at++] - '0');
        if (at < text.size() && text[at] == '.' && digit(at + 1))
            for (++at; digit(at); ++at, ++places) cents = cents * 10 + (text[at] - '0');
        EXPECT(places <= 2, "a number with more than two decimals: %s", text.c_str());
        found.push_back(whole * 100 + (places == 1 ? cents * 10 : cents));
    }
    return found;
}

// The numbers the checker uses, in hundredths, must each be written in the
// question (the record could otherwise hold a number the reader never saw).
inline void check_inputs_written(const FractionProblem& p) {
    const auto& in = p.inputs;
    std::vector<int64_t> must;
    switch (p.kind) {
        case FractionProblem::part_of:
            must.push_back(in[0] * 100);
            for (size_t i = 0; i < p.ops.size(); ++i) {
                must.push_back(in[1 + 2 * i] * 100);
                if (p.ops[i] == '/') must.push_back(in[2 + 2 * i] * 100);
            }
            break;
        case FractionProblem::price_change:
            must.push_back(in[0]);  // already in cents
            for (size_t i = 1; i < in.size(); ++i) must.push_back(in[i] * 100);
            break;
        case FractionProblem::proportion: must = {in[0] * 100, in[1], in[2] * 100}; break;
        case FractionProblem::units:
            must = {p.context == 2 ? in[0] : in[0] * 100, in[1] * 100};
            break;
        case FractionProblem::ratio_known: must = {in[0] * 100, in[1] * 100, in[3] * 100}; break;
        default:
            for (int64_t v : in) must.push_back(v * 100);
    }
    std::vector<int64_t> written = written_hundredths(p.problem.question);
    for (int64_t value : must) {
        const auto match = std::find(written.begin(), written.end(), value);
        EXPECT(match != written.end(), "story number %lld/100 is not written in the question: %s", (long long)value,
               p.problem.question.c_str());
        if (match != written.end()) written.erase(match);
    }
}

inline void check_problem(const FractionProblem& p, bool notes) {
    const Problem& q = p.problem;
    EXPECT(q.task == "fractions", "wrong task %s", q.task.c_str());
    equation_test::check_singular_after_one(q.question + " " + q.steps);
    EXPECT(q.question.find('{') == std::string::npos && q.question.find('}') == std::string::npos,
           "unfilled placeholder: %s", q.question.c_str());
    EXPECT(extract_answer(q.completion()) == q.answer, "not a GSM8K completion");
    // A positive number with at most two decimals and no trailing zero.
    const size_t dot = q.answer.find('.');
    EXPECT(!q.answer.empty() && q.answer.find_first_not_of("0123456789.") == std::string::npos &&
               (q.answer[0] != '0' || q.answer.starts_with("0.")) &&
               (dot == std::string::npos || (q.answer.size() - dot - 1 <= 2 && q.answer.back() != '0')),
           "answer is not a short positive number: %s", q.answer.c_str());
    check_inputs_written(p);
    std::istringstream lines(q.steps);
    for (std::string line; std::getline(lines, line);)
        EXPECT(!line.empty() && line.back() == '.', "solution line does not end a sentence: %s", line.c_str());
    size_t notes_seen = 0;
    for (size_t at = 0; (at = q.steps.find("<<", at)) != std::string::npos; ++at) ++notes_seen;
    EXPECT(notes ? notes_seen == size_t(q.difficulty) && notes_seen >= 1 : notes_seen == 0,
           "difficulty is not the number of steps");
    if (p.distractor_number) {
        const auto used = written_hundredths(q.steps + " " + q.answer);
        EXPECT(std::find(used.begin(), used.end(), p.distractor_number * 100) == used.end(),
               "distractor number %lld is used in the solution", (long long)p.distractor_number);
    }
    if (!notes) return;
    EXPECT(fraction_check::problems_with(p).empty(), "checker rejects: %s", q.question.c_str());
    for (size_t at = 0; (at = q.steps.find("<<", at)) != std::string::npos; ++at) {
        const size_t equals = q.steps.find('=', at), end = q.steps.find(">>", at);
        const auto reply = calc::response(q.steps.substr(0, equals + 1), calc::Mode::notes);
        EXPECT(reply && *reply == q.steps.substr(equals + 1, end - equals - 1) + ">>",
               "the calculator hook disagrees with a note in: %s", q.steps.c_str());
    }
}

// The checker must not pass broken problems.
inline void checker_catches_mistakes(const FractionProblem& good) {
    EXPECT(fraction_check::problems_with(good).empty(), "checker rejects a good problem");
    FractionProblem p = good;
    p.problem.answer = good.problem.answer + "1";
    EXPECT(!fraction_check::problems_with(p).empty(), "checker passed a wrong answer");
    p = good;
    p.inputs[0] += 1;  // the story's first number is off by one
    EXPECT(!fraction_check::problems_with(p).empty(), "checker passed a changed story number: %s",
           good.problem.question.c_str());
    p = good;
    const size_t note = p.problem.steps.find(">>");
    char& digit = p.problem.steps[note - 1];
    digit = digit == '9' ? '0' : char(digit + 1);
    EXPECT(!fraction_check::problems_with(p).empty(), "checker passed a wrong note");
    p = good;
    p.problem.steps += "\nThat is all.";
    EXPECT(!fraction_check::problems_with(p).empty(), "checker passed a last line without the answer");
}

}  // namespace fraction_test

inline void test_mathgen_fractions() {
    using fraction_detail::family_count;
    {
        FractionProblem p;
        p.kind = FractionProblem::ratio_split;
        p.inputs = {41, 3, 5};  // 41 cannot be split 3:5 in whole dollars
        int64_t hundredths = 0;
        EXPECT(!fraction_check::solve(p, hundredths).empty(), "a ratio without a whole split passed");
        p.inputs = {40, 3, 5};
        p.asked = 1;
        EXPECT(fraction_check::solve(p, hundredths).empty() && hundredths == 2500, "the ratio check is wrong");
        p.kind = FractionProblem::part_of;
        p.ops = "/%";
        p.inputs = {60, 1, 4, 25, 100};  // 60 -> 45 -> 33.75: not whole
        EXPECT(!fraction_check::solve(p, hundredths).empty(), "a part that is not whole passed");
        p.inputs = {80, 1, 4, 25, 100};  // 80 -> 60 -> 45
        EXPECT(fraction_check::solve(p, hundredths).empty() && hundredths == 4500, "the part check is wrong");
        p.kind = FractionProblem::what_percent;
        p.context = 0;
        p.inputs = {1, 3};  // 1 of 3 is not a whole percent
        EXPECT(!fraction_check::solve(p, hundredths).empty(), "a percent that is not whole passed");
        EXPECT(fraction_check::decimal(1250) == "12.5" && fraction_check::decimal(1205) == "12.05" &&
                   fraction_check::decimal(1200) == "12",
               "decimals are not written as GSM8K writes them");
    }
    std::set<std::string> questions[2];
    std::set<int> patterns[2];
    std::set<int> kinds_seen[2];
    int checked = 0;
    for (bool test : {false, true}) {
        Rng rng(77);
        for (int i = 0; i < 4000; ++i) {
            const auto p = make_fraction_problem(rng, {true, true, test});
            fraction_test::check_problem(p, true);
            if (i < 400) fraction_test::checker_catches_mistakes(p);
            questions[test].insert(p.problem.question);
            for (int id : p.patterns) {
                EXPECT((id % 6 >= 4) == test, "pattern %d leaked across splits", id);
                patterns[test].insert(id);
            }
            // Every kind, setting and question asked.
            kinds_seen[test].insert((p.kind * 4 + p.context) * 4 + p.asked);
            if (p.distractor >= 0) EXPECT((p.distractor >= 12) == test, "distractor wording leaked across splits");
            ++checked;
        }
    }
    EXPECT(patterns[0].size() == size_t(family_count) * 4 && patterns[1].size() == size_t(family_count) * 2,
           "not all sentence patterns exercised (%zu train, %zu test)", patterns[0].size(), patterns[1].size());
    // part_of 1, price_change 2, what_percent 2, ratio_split 3+3+3,
    // ratio_known 2, proportion 2, units 4.
    const size_t asked_total = 1 + 2 + 2 + 9 + 2 + 2 + 4;
    EXPECT(kinds_seen[0].size() == asked_total && kinds_seen[1].size() == asked_total,
           "a kind, setting or question is missing (%zu, %zu)", kinds_seen[0].size(), kinds_seen[1].size());
    for (const auto& q : questions[1]) EXPECT(!questions[0].count(q), "question leaked across splits");
    std::set<std::string> texts;
    for (const auto& family : fraction_detail::patterns)
        for (const char* pattern : family) EXPECT(texts.insert(pattern).second, "duplicate pattern: %s", pattern);
    for (const auto& family : equation_detail::patterns)
        for (const char* pattern : family) EXPECT(texts.insert(pattern).second, "pattern shared with equations: %s", pattern);
    for (bool test : {false, true}) {
        Rng noted_rng(9), plain_rng(9), quiet_rng(9);
        for (int i = 0; i < 300; ++i) {
            const auto noted = make_fraction_problem(noted_rng, {true, true, test});
            const auto plain = make_fraction_problem(plain_rng, {true, false, test});
            const auto quiet = make_fraction_problem(quiet_rng, {false, false, test});
            fraction_test::check_problem(plain, false);
            fraction_test::check_problem(quiet, false);
            EXPECT(noted.problem.question == plain.problem.question && noted.problem.answer == plain.problem.answer &&
                       math_detail::without_notes(noted.problem.steps) == plain.problem.steps,
                   "notes changed the problem");
            EXPECT(quiet.problem.answer == plain.problem.answer && quiet.problem.steps == plain.problem.steps &&
                       quiet.problem.question.size() < plain.problem.question.size(),
                   "distractors changed the math");
        }
    }
    const auto a = make_fraction_problems(31, 200), b = make_fraction_problems(31, 200);
    bool same = a.size() == b.size();
    for (size_t i = 0; same && i < a.size(); ++i)
        same = a[i].question == b[i].question && a[i].steps == b[i].steps && a[i].answer == b[i].answer;
    EXPECT(same, "the same seed gave other problems");
    std::set<std::string> unique;
    for (const auto& p : a) unique.insert(p.question);
    EXPECT(unique.size() == a.size(), "duplicate questions in one file");
    EXPECT(a[0].question != make_fraction_problems(32, 1)[0].question, "seed has no effect");
    std::printf("  %d fractions problems checked by the second solver; all %d patterns, both splits\n", checked,
                family_count * 6);
}
