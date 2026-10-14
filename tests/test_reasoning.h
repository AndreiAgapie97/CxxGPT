// test_reasoning.h — every generated problem is checked by an independent
// solver that only reads the question text.
#pragma once

#include <algorithm>
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

inline std::vector<std::string> split_steps(const std::string& s) {
    std::vector<std::string> out;
    size_t at = 0;
    for (size_t semi; (semi = s.find("; ", at)) != std::string::npos; at = semi + 2) out.push_back(s.substr(at, semi - at));
    out.push_back(s.substr(at));
    return out;
}

// expr_rewrite: every step but the last also writes the expression left,
// which has one operation fewer and still evaluates to the answer.
inline void check_expr_rewrite(const Problem& plain, const Problem& rw) {
    const std::vector<std::string> ps = split_steps(plain.steps), qs = split_steps(rw.steps);
    EXPECT(ps.size() == qs.size(), "expr rewrite of %s: %zu steps, plain has %zu", rw.question.c_str(), qs.size(),
           ps.size());
    if (ps.size() != qs.size()) return;
    for (size_t i = 0; i < ps.size(); ++i) {
        if (i + 1 == ps.size()) {
            EXPECT(qs[i] == ps[i], "expr rewrite: last step '%s', plain '%s'", qs[i].c_str(), ps[i].c_str());
            continue;
        }
        const size_t colon = qs[i].find(": ");
        EXPECT(colon != std::string::npos && qs[i].substr(0, colon) == ps[i], "expr rewrite: step '%s', plain '%s'",
               qs[i].c_str(), ps[i].c_str());
        if (colon == std::string::npos) continue;
        const std::string left = qs[i].substr(colon + 2);
        const long ops = std::count_if(left.begin(), left.end(), [](char c) { return c == '+' || c == '-' || c == '*'; });
        EXPECT(ops == long(ps.size() - 1 - i) && std::to_string(reasoning_check::eval(left)) == rw.answer,
               "expr rewrite of %s: '%s' after step %zu", rw.question.c_str(), left.c_str(), i + 1);
    }
}

// Formats change only how solutions are written: the same random draws give
// the same questions and answers. The chain plan, rebuilt here from the
// question alone, follows the definitions back from the queried variable.
inline void test_formats() {
    Rng plain_rng(43), fmt_rng(43);
    const Formats all{.chain_plan = true, .expr_rewrite = true};
    int planned = 0, rewritten = 0;
    for (const auto& task : reasoning_tasks())
        for (int d = 1; d <= 5; ++d)
            for (int i = 0; i < 100; ++i) {
                const Problem p = make_problem(task, d, plain_rng);
                const Problem q = make_problem(task, d, fmt_rng, all);
                EXPECT(p.question == q.question && p.answer == q.answer, "%s d=%d: a format changed the problem",
                       task.c_str(), d);
                if (task == "expr") {
                    check_expr_rewrite(p, q);
                    ++rewritten;
                    continue;
                }
                if (task != "chain") {
                    EXPECT(p.steps == q.steps, "%s: a format changed its steps", task.c_str());
                    continue;
                }
                // "h=6; a=c*5; m=h+a; c=h+8; c?" -> plan "c=h+8; h=6".
                std::map<char, std::string> defs;
                size_t at = 0;
                for (size_t semi; (semi = q.question.find("; ", at)) != std::string::npos; at = semi + 2)
                    defs[q.question[at]] = q.question.substr(at, semi - at);
                std::string want;
                for (char v = q.question[at]; defs.count(v);) {
                    const std::string& def = defs[v];
                    want += (want.empty() ? "" : "; ") + def;
                    v = std::isalpha(static_cast<unsigned char>(def[2])) ? def[2] : 0;  // "c=h+8": next is h
                }
                EXPECT(q.steps == want + " => " + p.steps, "chain plan for %s: '%s', expected '%s => %s'",
                       q.question.c_str(), q.steps.c_str(), want.c_str(), p.steps.c_str());
                ++planned;
            }
    std::printf("  %d chain plans rebuilt from their questions, %d expression rewrites checked\n", planned,
                rewritten);
}

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
    test_formats();
}
