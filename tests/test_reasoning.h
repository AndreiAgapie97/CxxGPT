// test_reasoning.h — every generated problem is checked by an independent
// solver that only reads the question text.
#pragma once

#include <cctype>
#include <map>

#include "../src/reasoning.h"
#include "test_util.h"

namespace reasoning_check {

// Recursive descent over + - * and parentheses, usual precedence.
struct Parser {
    const std::string& s;
    size_t i = 0;
    long number() {
        long v = 0;
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) v = v * 10 + (s[i++] - '0');
        return v;
    }
    long atom() {
        if (s[i] == '(') {
            ++i;
            const long v = sum();
            ++i;  // ')'
            return v;
        }
        return number();
    }
    long product() {
        long v = atom();
        while (i < s.size() && s[i] == '*') ++i, v *= atom();
        return v;
    }
    long sum() {
        long v = product();
        while (i < s.size() && (s[i] == '+' || s[i] == '-')) {
            const char op = s[i++];
            const long r = product();
            v = op == '+' ? v + r : v - r;
        }
        return v;
    }
};

inline long eval(const std::string& s) {
    Parser p{s};
    return p.sum();
}

// "b=a+4; a=3; c=b*k; ...; c?" -> value of c
inline long solve_chain(const std::string& q) {
    std::map<char, std::string> defs;
    size_t pos = 0;
    char target = 0;
    while (pos < q.size()) {
        size_t end = q.find("; ", pos);
        const std::string d = q.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
        if (d.size() == 2 && d[1] == '?') target = d[0];
        else defs[d[0]] = d.substr(2);
        if (end == std::string::npos) break;
        pos = end + 2;
    }
    std::function<long(char)> value = [&](char v) -> long {
        std::string e;
        for (char ch : defs.at(v)) e += std::islower(static_cast<unsigned char>(ch)) ? std::to_string(value(ch)) : std::string(1, ch);
        return eval(e);
    };
    return value(target);
}

inline std::string solve(const Problem& p) {
    const std::string& q = p.question;
    if (p.task == "chain") return std::to_string(solve_chain(q));
    if (p.task == "add" || p.task == "mul" || p.task == "expr") return std::to_string(eval(q));
    if (p.task == "eq") {  // a*x+b=c or a*x-b=c: try every x
        const size_t eqpos = q.find('=');
        const long c = std::stol(q.substr(eqpos + 1));
        for (long x = 0; x < 100000; ++x) {
            const std::string lhs = q.substr(0, eqpos);
            std::string e;
            for (char ch : lhs) e += ch == 'x' ? std::to_string(x) : std::string(1, ch);
            if (eval(e) == c) return std::to_string(x);
        }
        return "?";
    }
    if (p.task == "cmp") {  // max(a,b)
        const size_t comma = q.find(',');
        return std::to_string(std::max(std::stol(q.substr(4, comma - 4)), std::stol(q.substr(comma + 1))));
    }
    return "?";
}

}  // namespace reasoning_check

inline void test_reasoning() {
    Rng rng(41);
    int checked = 0;
    for (const auto& task : reasoning_tasks())
        for (int d = 1; d <= 5; ++d)
            for (int i = 0; i < 300; ++i) {
                const Problem p = make_problem(task, d, rng);
                const std::string want = reasoning_check::solve(p);
                EXPECT(p.answer == want, "%s d=%d: %s -> answer %s, solver says %s", task.c_str(), d,
                       p.question.c_str(), p.answer.c_str(), want.c_str());
                // The steps end in the answer for the tasks whose last step computes it.
                if (task == "chain" || task == "expr" || task == "eq") {
                    const std::string tail = "=" + p.answer;
                    EXPECT(p.steps.size() >= tail.size() &&
                               p.steps.compare(p.steps.size() - tail.size(), tail.size(), tail) == 0,
                           "%s: steps '%s' do not end in the answer %s", task.c_str(), p.steps.c_str(),
                           p.answer.c_str());
                }
                for (char ch : p.prompt() + p.completion())
                    EXPECT(ch >= 10 && ch < 127 && ch != kBos, "non-text byte %d in a problem", int(ch));
                EXPECT(extract_answer(p.completion()) == p.answer, "extract_answer failed on %s",
                       p.completion().c_str());
                if (g_failures > 20) return;
                ++checked;
            }
    if (std::getenv("SHOW_PROBLEMS")) {
        for (const auto& task : reasoning_tasks())
            for (int d : {1, 3, 5}) {
                const Problem p = make_problem(task, d, rng);
                std::printf("--- %s d=%d\n%s%s", task.c_str(), d, p.prompt().c_str(), p.completion().c_str());
            }
    }
    std::printf("  %d problems verified by an independent solver\n", checked);
}
