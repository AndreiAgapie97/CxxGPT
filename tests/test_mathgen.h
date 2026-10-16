// test_mathgen.h — the equations generator (src/mathgen.h): every problem is
// checked again here, the checker is shown to catch broken problems, and the
// splits, options and files behave as gen-word's do.
#pragma once

#include <fstream>
#include <iterator>
#include <set>
#include <sstream>

#include "../src/mathgen.h"
#include "test_util.h"

namespace math_check {

// The whole numbers written in a question, in order.
inline std::vector<int64_t> written_numbers(const std::string& text) {
    std::vector<int64_t> found;
    for (size_t at = 0; at < text.size(); ++at) {
        if (text[at] < '0' || text[at] > '9') continue;
        int64_t v = 0;
        while (at < text.size() && text[at] >= '0' && text[at] <= '9') v = v * 10 + (text[at++] - '0');
        found.push_back(v);
    }
    return found;
}

// The checker reads the record's numbers, so each number it uses must be
// written in the question; a pattern that dropped one would go unnoticed.
// Not written as digits: the animals' 4 and 2 legs (common knowledge), and
// the consecutive numbers' count (a word), step and parity.
inline void check_inputs_written(const EquationProblem& p) {
    std::vector<int64_t> must = p.inputs;
    if (p.kind == EquationProblem::mixture && p.context == 1) must = {p.inputs[2], p.inputs[3]};
    if (p.kind == EquationProblem::consecutive) must = {p.inputs[3]};
    // ages_sum writes |d| ("5 years younger"), and k only when it is not 1.
    // ages_sum and rectangle write k only when it is not 1, and d only when
    // it is not 0 (ages_sum writes |d|: "5 years younger").
    if (p.kind == EquationProblem::ages_sum || p.kind == EquationProblem::rectangle) {
        const int64_t k = p.inputs[0], d = p.inputs[1];
        must = {p.inputs[2]};
        if (k != 1) must.push_back(k);
        if (d != 0) must.push_back(d < 0 ? -d : d);
    }
    std::vector<int64_t> written = written_numbers(p.problem.question);
    for (int64_t value : must) {
        const auto match = std::find(written.begin(), written.end(), value);
        EXPECT(match != written.end(), "story number %lld is not written in the question: %s", (long long)value,
               p.problem.question.c_str());
        if (match != written.end()) written.erase(match);
    }
}

inline void check_problem(const EquationProblem& p, bool notes) {
    const Problem& q = p.problem;
    EXPECT(q.task == "equations", "wrong task %s", q.task.c_str());
    EXPECT(q.question.find('{') == std::string::npos && q.question.find('}') == std::string::npos,
           "unfilled placeholder: %s", q.question.c_str());
    EXPECT(extract_answer(q.completion()) == q.answer, "not a GSM8K completion");
    EXPECT(!q.answer.empty() && q.answer.find_first_not_of("0123456789") == std::string::npos && q.answer[0] != '0',
           "answer is not a whole positive number: %s", q.answer.c_str());
    check_inputs_written(p);
    std::istringstream lines(q.steps);
    int lines_seen = 0;
    for (std::string line; std::getline(lines, line); ++lines_seen)
        EXPECT(!line.empty() && line.back() == '.', "solution line does not end a sentence: %s", line.c_str());
    EXPECT(lines_seen >= 2, "solution too short");
    size_t notes_seen = 0;
    for (size_t at = 0; (at = q.steps.find("<<", at)) != std::string::npos; ++at) ++notes_seen;
    EXPECT(notes ? notes_seen == size_t(q.difficulty) : notes_seen == 0, "difficulty is not the number of steps");
    if (p.distractor_number) {
        const auto used = written_numbers(q.steps);
        EXPECT(std::find(used.begin(), used.end(), p.distractor_number) == used.end(),
               "distractor number %lld is used in the solution", (long long)p.distractor_number);
    }
    if (!notes) return;
    EXPECT(equation_check::problems_with(p).empty(), "checker rejects: %s", q.question.c_str());
    // The engine's calculator hook (calc-notes) must give each note's own
    // value when the model has written the note up to its "=".
    for (size_t at = 0; (at = q.steps.find("<<", at)) != std::string::npos; ++at) {
        const size_t equals = q.steps.find('=', at), end = q.steps.find(">>", at);
        const auto reply = calc::response(q.steps.substr(0, equals + 1), calc::Mode::notes);
        EXPECT(reply && *reply == q.steps.substr(equals + 1, end - equals - 1) + ">>",
               "the calculator hook disagrees with a note in: %s", q.steps.c_str());
    }
}

// The checker must not pass broken problems: wrong answers, changed story
// numbers, wrong notes, an answer the last line does not state, and
// stories with no or several solutions.
inline void checker_catches_mistakes(const EquationProblem& good) {
    EXPECT(equation_check::problems_with(good).empty(), "checker rejects a good problem");
    EquationProblem p = good;
    p.problem.answer = std::to_string(std::stoll(good.problem.answer) + 1);
    EXPECT(!equation_check::problems_with(p).empty(), "checker passed a wrong answer");
    p = good;
    p.inputs.back() += 1;  // the stated total (or final number) is off by one
    EXPECT(!equation_check::problems_with(p).empty(), "checker passed a changed story number: %s",
           good.problem.question.c_str());
    p = good;
    const size_t note = p.problem.steps.find(">>");
    if (note != std::string::npos) {
        // Change the last digit of the first note's value, inside the note only.
        char& digit = p.problem.steps[note - 1];
        digit = digit == '9' ? '0' : char(digit + 1);
        EXPECT(!equation_check::problems_with(p).empty(), "checker passed a wrong note");
    }
    p = good;
    p.problem.steps += "\nThat is all.";
    EXPECT(!equation_check::problems_with(p).empty(), "checker passed a last line without the answer");
    // Asking for another quantity must change the answer the checker
    // expects. (Only for kinds where the quantities always differ: in a
    // mixture both kinds can have the same count.)
    p = good;
    p.asked = good.asked == 0 ? 1 : 0;
    if (good.kind == EquationProblem::ages_future || good.kind == EquationProblem::consecutive ||
        good.kind == EquationProblem::rectangle || good.kind == EquationProblem::shares)
        EXPECT(!equation_check::problems_with(p).empty(), "checker ignores which quantity is asked");
}

}  // namespace math_check

inline void test_mathgen() {
    using equation_detail::family_count;
    // Stories with no whole-number solution, and with several, are rejected.
    {
        EquationProblem none;
        none.kind = EquationProblem::mixture;
        none.inputs = {5, 3, 10, 41};  // 5x + 3(10 - x) = 41 has no whole x
        int64_t answer = 0;
        EXPECT(!equation_check::solve(none, answer).empty(), "a story without a solution passed");
        EquationProblem many = none;
        many.inputs = {3, 3, 10, 30};  // both kinds cost the same: every x fits
        EXPECT(!equation_check::solve(many, answer).empty(), "a story with many solutions passed");
        EquationProblem puzzle;
        puzzle.ops = "*+";
        puzzle.inputs = {2, 3, 10};  // 2x + 3 = 10 has no whole x
        EXPECT(!equation_check::solve(puzzle, answer).empty(), "a puzzle without a whole start passed");
        puzzle.inputs = {2, 3, 11};
        EXPECT(equation_check::solve(puzzle, answer).empty() && answer == 4, "the puzzle check is wrong");
    }
    std::set<std::string> questions[2];
    std::set<int> patterns[2];
    std::set<int> kinds_asked[2], contexts, distractors, steps;
    int checked = 0;
    for (bool test : {false, true}) {
        Rng rng(77);  // deliberately the same seed for both splits
        for (int i = 0; i < 3000; ++i) {
            const auto p = make_equation_problem(rng, {true, true, test});
            math_check::check_problem(p, true);
            if (i < 300) math_check::checker_catches_mistakes(p);
            questions[test].insert(p.problem.question);
            for (int id : p.patterns) {
                EXPECT((id % 6 >= 4) == test, "pattern %d leaked across splits", id);
                patterns[test].insert(id);
            }
            kinds_asked[test].insert(p.kind * 4 + p.asked);
            if (p.kind == EquationProblem::mixture) contexts.insert(p.context);
            if (p.distractor >= 0) {
                EXPECT((p.distractor >= 12) == test, "distractor wording leaked across splits");
                distractors.insert(p.distractor);
            }
            steps.insert(p.problem.difficulty);
            ++checked;
        }
    }
    EXPECT(patterns[0].size() == size_t(family_count) * 4 && patterns[1].size() == size_t(family_count) * 2,
           "not all sentence patterns exercised (%zu train, %zu test)", patterns[0].size(), patterns[1].size());
    // Every kind, and every quantity each kind can ask for, in both splits.
    const size_t asked_total = 1 + 2 + 2 + 2 + 2 + 3 + 3;
    EXPECT(kinds_asked[0].size() == asked_total && kinds_asked[1].size() == asked_total,
           "a kind or question is missing");
    EXPECT(contexts.size() == 3 && distractors.size() == 24, "mixture contexts or distractors missing");
    EXPECT(steps.size() >= 5, "too little variety in the number of steps");
    for (const auto& q : questions[1]) EXPECT(!questions[0].count(q), "question leaked across splits");
    std::set<std::string> texts;
    for (const auto& family : equation_detail::patterns)
        for (const char* pattern : family) EXPECT(texts.insert(pattern).second, "duplicate pattern: %s", pattern);
    // Notes and distractors change no random draw: the same problems.
    for (bool test : {false, true}) {
        Rng noted_rng(9), plain_rng(9), quiet_rng(9);
        for (int i = 0; i < 300; ++i) {
            const auto noted = make_equation_problem(noted_rng, {true, true, test});
            const auto plain = make_equation_problem(plain_rng, {true, false, test});
            const auto quiet = make_equation_problem(quiet_rng, {false, false, test});
            math_check::check_problem(plain, false);
            math_check::check_problem(quiet, false);
            EXPECT(noted.problem.question == plain.problem.question && noted.problem.answer == plain.problem.answer &&
                       equation_detail::without_notes(noted.problem.steps) == plain.problem.steps &&
                       plain.problem.difficulty == noted.problem.difficulty,
                   "notes changed the problem");
            EXPECT(quiet.problem.answer == plain.problem.answer && quiet.problem.steps == plain.problem.steps &&
                       quiet.problem.question.size() < plain.problem.question.size() && quiet.distractor == -1,
                   "distractors changed the math");
        }
    }
    // The same seed writes the same file; it loads back unchanged.
    const std::string base = "runs/test-mathgen";
    const auto a = make_equation_problems(31, 100), b = make_equation_problems(31, 100);
    save_problems(base + "-a.txt", a);
    save_problems(base + "-b.txt", b);
    auto bytes = [](const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(f), {});
    };
    EXPECT(bytes(base + "-a.txt") == bytes(base + "-b.txt"), "same seed changed file bytes");
    const auto loaded = load_problems(base + "-a.txt");
    EXPECT(loaded.size() == a.size(), "file did not round trip");
    for (size_t i = 0; i < loaded.size() && i < a.size(); ++i)
        EXPECT(loaded[i].prompt() == a[i].prompt() && loaded[i].completion() == a[i].completion() &&
                   loaded[i].task == "equations" && loaded[i].difficulty == a[i].difficulty,
               "file changed a field");
    std::remove((base + "-a.txt").c_str());
    std::remove((base + "-b.txt").c_str());
    // --exclude: a question copied word for word (rule 1), and one with only
    // the same numbers and answer (rule 2), are both drawn again. Each rule
    // is tried alone, so the problems before the excluded one are the same.
    EXPECT((Exclusions::numbers("It costs $1,000.50, 3 more. 7?") == std::vector<std::string>{"1000.50", "3", "7"}),
           "numbers are not read as written");
    {
        Rng plain_rng(5);
        std::vector<EquationProblem> plain;
        for (int i = 0; i < 60; ++i) plain.push_back(make_equation_problem(plain_rng));
        size_t longest = 0;
        for (size_t i = 0; i < plain.size(); ++i)
            if (plain[i].problem.question.size() > plain[longest].problem.question.size()) longest = i;
        const EquationProblem& copied = plain[longest];
        const EquationProblem& renamed = plain[7];
        auto key = Exclusions::numbers(renamed.problem.question);
        if (renamed.distractor_number)
            key.erase(std::find(key.begin(), key.end(), std::to_string(renamed.distractor_number)));
        std::string other_words = "Some other story with";
        for (const auto& n : key) other_words += " " + n + " and";
        const std::string file1 = base + "-exclude1.txt", file2 = base + "-exclude2.txt";
        std::ofstream(file1, std::ios::binary) << copied.problem.question << "\n";
        std::ofstream(file2, std::ios::binary) << "gsm8k\t1\t" << other_words << " more.\tsteps\t"
                                               << renamed.problem.answer << "\r\n";
        for (int rule : {1, 2}) {
            Exclusions exclusions(rule == 1 ? file1 : file2);
            MathOptions options;
            options.exclude = &exclusions;
            Rng rng(5);
            for (int i = 0; i < 60; ++i) {
                const auto p = make_equation_problem(rng, options);
                if (rule == 1) EXPECT(p.problem.question != copied.problem.question, "an excluded question was kept");
                auto got = Exclusions::numbers(p.problem.question);
                if (p.distractor_number)
                    got.erase(std::find(got.begin(), got.end(), std::to_string(p.distractor_number)));
                if (rule == 2)
                    EXPECT(!(got == key && p.problem.answer == renamed.problem.answer),
                           "a problem with excluded numbers and answer was kept");
                if (i < (rule == 1 ? int(longest) : 7))
                    EXPECT(p.problem.question == plain[size_t(i)].problem.question,
                           "exclusions changed the problems before the excluded one");
            }
            EXPECT(rule == 1 ? exclusions.by_run >= 1 : exclusions.by_numbers >= 1,
                   "rule %d never fired (%zu, %zu)", rule, exclusions.by_run, exclusions.by_numbers);
        }
        std::remove(file1.c_str());
        std::remove(file2.c_str());
    }
    std::set<std::string> unique;
    for (const auto& p : a) unique.insert(p.question);
    EXPECT(unique.size() == a.size(), "duplicate questions in one file");
    EXPECT(make_equation_problems(1, 0).empty(), "zero count is not empty");
    EXPECT(a[0].question != make_equation_problems(32, 1)[0].question, "seed has no effect");
    std::printf("  %d equation problems checked by the second solver; all %d patterns, both splits\n", checked,
                family_count * 6);
}
