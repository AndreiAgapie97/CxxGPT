// test_wordgen.h — independent rational and double-precision references.
#pragma once

#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>

#include "../src/wordgen.h"
#include "test_util.h"

namespace word_check {

// The generator uses fixed hundredths. This reference instead keeps a
// numerator/denominator pair, so it neither calls nor copies that evaluator.
struct Rational {
    int64_t n = 0, d = 1;
    Rational(int64_t numerator = 0, int64_t denominator = 1) : n(numerator), d(denominator) {
        CHECK(d != 0, "zero reference denominator");
        const int64_t divisor = std::gcd(n, d);
        n /= divisor;
        d /= divisor;
        if (d < 0) { n = -n; d = -d; }
    }
    bool operator==(const Rational&) const = default;
};

inline Rational apply(char op, Rational a, Rational b) {
    if (op == '+') return {a.n * b.d + b.n * a.d, a.d * b.d};
    if (op == '-') return {a.n * b.d - b.n * a.d, a.d * b.d};
    if (op == '*') return {a.n * b.n, a.d * b.d};
    CHECK(op == '/' && b.n != 0, "invalid reference operation");
    return {a.n * b.d, a.d * b.n};
}

inline Rational decimal(const std::string& text) {
    int64_t numerator = 0, denominator = 1;
    bool after_point = false;
    for (char ch : text) {
        if (ch == '.') { EXPECT(!after_point, "two decimal points: %s", text.c_str()); after_point = true; }
        else {
            EXPECT(ch >= '0' && ch <= '9', "not a decimal: %s", text.c_str());
            numerator = numerator * 10 + ch - '0';
            if (after_point) denominator *= 10;
        }
    }
    EXPECT(!text.empty(), "empty decimal");
    return {numerator, denominator};
}

inline Rational expression(const std::string& text) {
    const size_t op = text.find_first_of("+-*/");
    CHECK(op != std::string::npos, "missing reference operator: %s", text.c_str());
    return apply(text[op], decimal(text.substr(0, op)), decimal(text.substr(op + 1)));
}

// Accept only a plan as input: neither the stored results nor the answer
// string is accessible while solving it. Double precision is checked too.
inline std::vector<Rational> solve(const std::vector<WordNode>& plan) {
    std::vector<Rational> results;
    std::vector<double> doubles;
    for (const auto& node : plan) {
        Rational value(node.number, 100);
        double fp = double(node.number) / 100;
        if (node.op) {
            CHECK(node.left < results.size() && node.right < results.size(), "plan must refer backward");
            value = apply(node.op, results[node.left], results[node.right]);
            const double a = doubles[node.left], b = doubles[node.right];
            if (node.op == '+') fp = a + b;
            if (node.op == '-') fp = a - b;
            if (node.op == '*') fp = a * b;
            if (node.op == '/') fp = a / b;
        }
        EXPECT(value.n >= 0 && value.n <= 1000000 * value.d, "reference value outside cap");
        EXPECT((value.n * 100) % value.d == 0, "reference needs finer than hundredths");
        EXPECT(!node.whole || value.d == 1, "reference found fractional objects");
        EXPECT(std::abs(fp - double(value.n) / double(value.d)) < 1e-8, "double reference disagrees");
        results.push_back(value);
        doubles.push_back(fp);
    }
    return results;
}

// Solve the new stories a second way, directly from their independent
// quantities. This catches a wrong plan as well as wrong plan arithmetic.
inline Rational solve_shape(WordProblem::Shape shape, const std::array<int64_t, 4>& inputs,
                            const std::vector<int64_t>& additions, bool three_people, bool percentage) {
    const Rational a(inputs[0], 100), b(inputs[1], 100), c(inputs[2], 100), d(inputs[3], 100);
    Rational first = shape == WordProblem::part && !percentage ? apply('/', a, b) : apply('*', a, b);
    if (percentage) first = apply('/', first, Rational(100));
    for (int64_t added : additions) first = apply('+', first, Rational(added, 100));
    if (shape == WordProblem::comparison) return apply('-', first, c);
    if (shape == WordProblem::change) return apply('-', c, first);
    CHECK(shape == WordProblem::total || shape == WordProblem::rate || shape == WordProblem::part,
          "unexpected story shape");
    Rational result = apply('+', first, c);
    return three_people ? apply('+', result, d) : result;
}

inline void check_money(const std::string& question) {
    // Check both dollar signs and amounts spelled as "7.50 dollars".
    for (size_t at = 0; at < question.size(); ++at) {
        if (question[at] < '0' || question[at] > '9') continue;
        const size_t begin = at;
        while (at < question.size() && ((question[at] >= '0' && question[at] <= '9') ||
               (question[at] == '.' && at + 1 < question.size() && question[at + 1] >= '0' && question[at + 1] <= '9'))) ++at;
        const std::string value = question.substr(begin, at - begin);
        if ((begin && question[begin - 1] == '$') || question.substr(at, 8) == " dollars") {
            const size_t dot = value.find('.');
            EXPECT(dot == std::string::npos || value.size() - dot - 1 == 2,
                   "money must have zero or two decimals: %s", question.c_str());
        }
    }
}

inline void check_solution(const WordProblem& generated, bool notes) {
    const auto reference = solve(generated.plan);
    const auto& p = generated.problem;
    EXPECT(reference.size() == generated.values.size(), "missing computed values");
    for (size_t i = 0; i < reference.size(); ++i)
        EXPECT((reference[i] == Rational(generated.values[i], 100)), "plan node %zu has a wrong value", i);
    EXPECT(decimal(p.answer) == reference.back(), "answer disagrees with independently solved plan");
    if (generated.shape != WordProblem::chain)
        EXPECT(decimal(p.answer) == solve_shape(generated.shape, generated.inputs, generated.additions,
                                               generated.three_people, generated.percentage),
               "answer disagrees with independently solved story shape");
    check_money(p.question);
    if (generated.distractor_number) {
        const Rational distractor(generated.distractor_number, 100);
        for (const auto& value : reference)
            EXPECT(!(value == distractor), "distractor number used in solution arithmetic");
    }
    EXPECT(extract_answer(p.completion()) == p.answer, "not a GSM8K completion");
    std::istringstream lines(p.steps);
    std::string line;
    size_t node = 0;
    int operations = 0;
    while (std::getline(lines, line)) {
        while (node < generated.plan.size() && !generated.plan[node].op) ++node;
        CHECK(node < reference.size(), "extra solution step");
        const size_t eq = line.find(" = ");
        CHECK(eq != std::string::npos && line.back() == '.', "bad solution line");
        size_t begin = eq;
        while (begin && std::string("0123456789.+-*/").find(line[begin - 1]) != std::string::npos) --begin;
        const std::string expr = line.substr(begin, eq - begin);
        const size_t op = expr.find_first_of("+-*/");
        CHECK(op != std::string::npos, "missing solution operator");
        const WordNode& plan = generated.plan[node];
        EXPECT(expr[op] == plan.op && decimal(expr.substr(0, op)) == reference[plan.left] &&
                   decimal(expr.substr(op + 1)) == reference[plan.right], "solution uses wrong inputs");
        const Rational result = expression(expr);
        EXPECT(result == reference[node], "solution step disagrees with plan");
        std::string tail = line.substr(eq + 3, line.size() - eq - 4);
        if (notes) {
            const size_t equals = tail.find('='), end = tail.find(">>");
            CHECK(tail.starts_with("<<") && equals != std::string::npos && end != std::string::npos,
                  "missing calculator note");
            const Rational note = expression(tail.substr(2, equals - 2));
            EXPECT(note == result && note == decimal(tail.substr(equals + 1, end - equals - 1)),
                   "calculator note has a wrong expression/result");
            tail = tail.substr(end + 2);
        } else EXPECT(tail.find("<<") == std::string::npos, "notes present while disabled");
        const size_t result_end = tail.find_first_not_of("0123456789.");
        EXPECT(decimal(tail.substr(0, result_end)) == reference[node], "written step result is wrong");
        ++node;
        ++operations;
    }
    EXPECT(operations == p.difficulty && node == generated.plan.size(), "wrong number of solution operations");
    EXPECT(p.question.find('{') == std::string::npos, "unfilled sentence placeholder");
}

inline void arithmetic() {
    // Boundaries include negative inputs, division by zero, non-cent results,
    // fractional counts, the cap itself, and values that would overflow if
    // the implementation multiplied before checking its inputs.
    const std::vector<int64_t> inputs = {-1, 0, 1, 25, 50, 99, 100, 125, 200, 300, 1000,
        99999999, 100000000, 100000001, std::numeric_limits<int64_t>::max()};
    for (int64_t a : inputs) for (int64_t b : inputs) for (char op : std::string("+-*/"))
        for (bool whole : {false, true}) {
            bool valid = a >= 0 && a <= 100000000 && b >= 0 && b <= 100000000 && !(op == '/' && b == 0);
            Rational ref;
            if (valid) {
                ref = apply(op, Rational(a, 100), Rational(b, 100));
                valid = ref.n >= 0 && ref.n <= 1000000 * ref.d && (ref.n * 100) % ref.d == 0 &&
                        (!whole || ref.d == 1);
            }
            int64_t value = -123;
            const bool ok = word_detail::calculate(op, a, b, whole, value);
            EXPECT(ok == valid, "arithmetic acceptance disagrees for %lld %c %lld (whole %d)",
                   static_cast<long long>(a), op, static_cast<long long>(b), int(whole));
            if (ok) {
                EXPECT((Rational(value, 100) == ref), "arithmetic value disagrees");
                EXPECT(decimal(word_detail::number(value)) == ref, "decimal formatting changed a value");
            } else EXPECT(value == -123, "rejected arithmetic changed output");
        }
    int64_t unused = 0;
    EXPECT(!word_detail::calculate('?', 100, 200, false, unused), "unknown operation accepted");
    // Exercise every possible fractional suffix, including trailing zeroes.
    for (int64_t value = 0; value <= 10000; ++value) {
        EXPECT((decimal(word_detail::number(value)) == Rational(value, 100)), "bad decimal spelling");
        EXPECT((decimal(word_detail::money(value)) == Rational(value, 100)), "money formatting changed value");
        check_money("An item costs $" + word_detail::money(value) + " each.");
    }
}

}  // namespace word_check

inline void test_wordgen() {
    word_check::arithmetic();
    std::set<std::string> train_questions, test_questions;
    std::set<int> train_patterns, test_patterns;
    std::array<std::set<int>, WordProblem::shape_count> shape_patterns[2];
    std::set<int> seen_distractors, distractor_positions;
    std::set<int> shapes_by_length;
    std::set<std::string> rate_units;
    bool saw_two_people = false, saw_three_people = false, saw_fraction = false, saw_percentage = false;
    int inline_steps = 0, converted_stories = 0;
    int checked = 0;
    for (bool test : {false, true}) for (int steps = 2; steps <= 8; ++steps) {
        Rng rng(123 + uint64_t(steps));  // deliberately the same seeds for both splits
        const WordOptions options{steps, steps, true, true, test};
        for (int i = 0; i < 600; ++i) {
            const auto p = make_word_problem(rng, options);
            word_check::check_solution(p, true);
            shapes_by_length.insert((int(test) * 7 + steps - 2) * WordProblem::shape_count + p.shape);
            if (p.shape == WordProblem::total) (p.three_people ? saw_three_people : saw_two_people) = true;
            if (p.shape == WordProblem::part) (p.percentage ? saw_percentage : saw_fraction) = true;
            if (p.shape == WordProblem::rate) for (const std::string unit : {"hour", "day", "week"})
                if (p.problem.question.find(unit) != std::string::npos) rate_units.insert(unit);
            // Kept-items stories add items with "keeps" sentences only, never gifts.
            if (p.shape == WordProblem::part)
                for (size_t k = 1; k < p.patterns.size(); ++k)
                    EXPECT(p.patterns[k] / 6 == word_detail::part_extra, "part story adds items without keeping them");
            if (p.problem.steps.find(" as the result.") != std::string::npos) ++inline_steps;
            seen_distractors.insert(p.distractor);
            distractor_positions.insert(p.distractor_position);
            EXPECT((p.distractor >= 12) == test, "distractor wording leaked across splits");
            size_t ml = 0;
            while ((ml = p.problem.question.find(" milliliters", ml)) != std::string::npos) {
                size_t begin = ml;
                while (begin && std::string("0123456789.").find(p.problem.question[begin - 1]) != std::string::npos) --begin;
                if (begin != ml) {
                    const auto amount = word_check::decimal(p.problem.question.substr(begin, ml - begin));
                    EXPECT(amount.n >= 1000 * amount.d && amount.n <= 24750 * amount.d &&
                           amount.n % (250 * amount.d) == 0, "follow-up milliliters were not scaled from liters");
                    ++converted_stories;
                }
                ++ml;
            }
            EXPECT(p.problem.difficulty == steps, "requested step count was not honored");
            (test ? test_questions : train_questions).insert(p.problem.question);
            for (int id : p.patterns) {
                EXPECT((id % 6 >= 4) == test, "pattern %d leaked across splits", id);
                (test ? test_patterns : train_patterns).insert(id);
                shape_patterns[test][p.shape].insert(id);
            }
            ++checked;
        }
    }
    EXPECT(train_patterns.size() == word_detail::family_count * 4 &&
           test_patterns.size() == word_detail::family_count * 2, "not all sentence patterns exercised");
    EXPECT(seen_distractors.size() == 24 && distractor_positions.count(0) && distractor_positions.count(1) &&
           distractor_positions.count(3), "distractor variety/placement missing");
    EXPECT(shapes_by_length.size() == 2 * 7 * WordProblem::shape_count, "a shape is missing at a requested length/split");
    EXPECT(saw_two_people && saw_three_people && saw_fraction && saw_percentage && rate_units.size() == 3,
           "missing a story subtype or time unit");
    EXPECT(inline_steps > 0 && converted_stories > 0, "inline wording or converted units not exercised");
    for (int shape = 0; shape < WordProblem::shape_count; ++shape) {
        EXPECT(!shape_patterns[0][shape].empty() && !shape_patterns[1][shape].empty(), "missing story shape");
        for (int id : shape_patterns[1][shape])
            EXPECT(!shape_patterns[0][shape].count(id), "held-out pattern leaked for a shape");
    }
    for (const auto& q : test_questions) EXPECT(!train_questions.count(q), "question leaked across splits");
    for (int id : test_patterns) EXPECT(!train_patterns.count(id), "held-out pattern leaked into training");
    std::set<std::string> pattern_text;
    for (const auto& group : word_detail::patterns) for (const char* pattern : group)
        EXPECT(pattern_text.insert(pattern).second, "duplicate sentence pattern");
    for (const char* pattern : word_detail::distractors)
        EXPECT(pattern_text.insert(pattern).second, "duplicate distractor wording");
    for (bool test : {false, true}) {
        Rng noted_rng(7), plain_rng(7), quiet_rng(7);
        for (int i = 0; i < 200; ++i) {
            const auto noted = make_word_problem(noted_rng, {2, 8, true, true, test});
            const auto plain = make_word_problem(plain_rng, {2, 8, true, false, test});
            const auto quiet = make_word_problem(quiet_rng, {2, 8, false, false, test});
            word_check::check_solution(plain, false);
            word_check::check_solution(quiet, false);
            EXPECT(noted.problem.question == plain.problem.question && noted.problem.answer == plain.problem.answer,
                   "notes changed the question or answer");
            EXPECT(quiet.problem.answer == plain.problem.answer && quiet.problem.steps == plain.problem.steps &&
                   quiet.problem.question.size() < plain.problem.question.size(), "distractors changed the math");
        }
    }
    // Serialize, load, and compare files too, including escaped solution newlines.
    const std::string base = "runs/test-wordgen";
    const auto a = make_word_problems(31, 100), b = make_word_problems(31, 100);
    save_problems(base + "-a.txt", a);
    save_problems(base + "-b.txt", b);
    auto bytes = [](const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(f), {});
    };
    EXPECT(bytes(base + "-a.txt") == bytes(base + "-b.txt"), "same seed changed file bytes");
    const auto loaded = load_problems(base + "-a.txt");
    EXPECT(loaded.size() == a.size(), "word file did not round trip");
    for (size_t i = 0; i < loaded.size(); ++i)
        EXPECT(loaded[i].prompt() == a[i].prompt() && loaded[i].completion() == a[i].completion() &&
                   loaded[i].task == "word" && loaded[i].difficulty == a[i].difficulty, "word file changed a field");
    std::remove((base + "-a.txt").c_str());
    std::remove((base + "-b.txt").c_str());
    EXPECT(make_word_problems(1, 0).empty(), "zero count is not empty");
    EXPECT(a[0].question != make_word_problems(32, 1)[0].question, "seed has no effect");
    std::printf("  %d word problems independently solved; all %d patterns, six shapes and calculator notes checked\n",
                checked, word_detail::family_count * 6);
}
