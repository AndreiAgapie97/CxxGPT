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

inline void check_solution(const WordProblem& generated, bool notes) {
    const auto reference = solve(generated.plan);
    const auto& p = generated.problem;
    EXPECT(reference.size() == generated.values.size(), "missing computed values");
    for (size_t i = 0; i < reference.size(); ++i)
        EXPECT((reference[i] == Rational(generated.values[i], 100)), "plan node %zu has a wrong value", i);
    EXPECT(decimal(p.answer) == reference.back(), "answer disagrees with independently solved plan");
    EXPECT(extract_answer(p.completion()) == p.answer, "not a GSM8K completion");
    std::istringstream lines(p.steps);
    std::string line;
    size_t node = 0;
    int operations = 0;
    while (std::getline(lines, line)) {
        while (node < generated.plan.size() && !generated.plan[node].op) ++node;
        CHECK(node < reference.size(), "extra solution step");
        const size_t colon = line.find(": "), eq = line.find(" = ");
        CHECK(colon != std::string::npos && eq != std::string::npos && line.back() == '.', "bad solution line");
        const std::string expr = line.substr(colon + 2, eq - colon - 2);
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
            EXPECT(note == decimal(tail), "calculator note differs from the number following it");
        } else EXPECT(tail.find("<<") == std::string::npos, "notes present while disabled");
        EXPECT(decimal(tail) == reference[node], "written step result is wrong");
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
    for (int64_t value = 0; value <= 10000; ++value)
        EXPECT((decimal(word_detail::number(value)) == Rational(value, 100)), "bad decimal spelling");
}

}  // namespace word_check

inline void test_wordgen() {
    word_check::arithmetic();
    std::set<std::string> train_questions, test_questions;
    std::set<int> train_patterns, test_patterns;
    int checked = 0;
    for (bool test : {false, true}) for (int steps = 2; steps <= 8; ++steps) {
        Rng rng(123 + uint64_t(steps));  // deliberately the same seeds for both splits
        const WordOptions options{steps, steps, true, true, test};
        for (int i = 0; i < 600; ++i) {
            const auto p = make_word_problem(rng, options);
            word_check::check_solution(p, true);
            EXPECT(p.problem.difficulty == steps, "requested step count was not honored");
            (test ? test_questions : train_questions).insert(p.problem.question);
            for (int id : p.patterns) {
                EXPECT((id % 6 >= 4) == test, "pattern %d leaked across splits", id);
                (test ? test_patterns : train_patterns).insert(id);
            }
            ++checked;
        }
    }
    EXPECT(train_patterns.size() == 48 && test_patterns.size() == 24, "not all 72 sentence patterns exercised");
    for (const auto& q : test_questions) EXPECT(!train_questions.count(q), "question leaked across splits");
    for (int id : test_patterns) EXPECT(!train_patterns.count(id), "held-out pattern leaked into training");
    std::set<std::string> pattern_text;
    for (const auto& group : word_detail::patterns) for (const char* pattern : group)
        EXPECT(pattern_text.insert(pattern).second, "duplicate sentence pattern");
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
    std::printf("  %d word problems independently solved; all 72 patterns and calculator notes checked\n", checked);
}
