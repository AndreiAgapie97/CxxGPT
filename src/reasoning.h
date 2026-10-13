// reasoning.h — synthetic reasoning problems with step-by-step solutions.
//
// Every problem is plain text, one byte per token:
//   <BOS>Q: <question>\nA: <steps>\n#### <answer>\n
// The model is trained only on what follows "A: " (the loss mask), and
// evaluated by generating from "A: " and comparing the text after "#### "
// with the answer, exactly.
//
// Task families, each with a difficulty knob d = 1..5:
//   chain  multi-hop variable resolution: definitions in shuffled order,
//          with distractors; the solution resolves them in dependency order
//   add    multi-digit addition, column by column with carries
//   mul    multi-digit multiplication via partial products
//   expr   nested arithmetic expression, innermost operation first
//   eq     linear equation a*x+b=c, solved in two steps
//   cmp    which of two numbers is larger, digit by digit
#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include "common.h"

constexpr int kBos = 1;  // a byte that never occurs in the text

struct Problem {
    std::string task;
    int difficulty = 1;
    std::string question;  // without "Q: "
    std::string steps;     // without "A: "
    std::string answer;

    std::string prompt() const { return "Q: " + question + "\nA: "; }
    std::string completion() const { return steps + "\n#### " + answer + "\n"; }
};

namespace reasoning_detail {

inline int rand_in(Rng& rng, int lo, int hi) { return lo + int(rng.below(uint64_t(hi - lo + 1))); }

// A number with exactly `digits` digits (no leading zero unless 1 digit).
inline long rand_digits(Rng& rng, int digits) {
    if (digits <= 1) return rand_in(rng, 0, 9);
    long lo = 1;
    for (int i = 1; i < digits; ++i) lo *= 10;
    return lo + long(rng.below(uint64_t(9 * lo)));
}

inline std::string join(const std::vector<std::string>& parts, const std::string& sep) {
    std::string s;
    for (size_t i = 0; i < parts.size(); ++i) s += (i ? sep : "") + parts[i];
    return s;
}

inline Problem chain(Rng& rng, int d) {
    // Variables v[0..n): v[0] is a literal; v[i] = v[j] op (number or v[k])
    // with j, k < i. The question asks for v[n-1]; distractors use the same
    // variables but nothing depends on them.
    const int hops = d + 1, distract = (d + 1) / 2 + 1;
    std::string letters = "abcdefghijklmnopqrstuvwxyz";
    for (int i = 25; i > 0; --i) std::swap(letters[i], letters[rng.below(i + 1)]);
    std::vector<int> val;
    std::vector<std::string> defs, steps;
    std::vector<std::vector<std::string>> lhs_rhs;  // for the solution
    auto name = [&](int i) { return std::string(1, letters[i]); };
    auto make = [&](int i, bool needed) {
        if (i == 0) {
            const int v = rand_in(rng, 1, 9);
            val.push_back(v);
            defs.push_back(name(0) + "=" + std::to_string(v));
            steps.push_back(name(0) + "=" + std::to_string(v));
            return;
        }
        // The needed chain links to the previous variable, so every hop counts.
        const int j = needed ? i - 1 : int(rng.below(uint64_t(i)));
        const bool use_var = i >= 2 && rng.below(3) == 0;
        const int k = use_var ? int(rng.below(uint64_t(i))) : -1;
        for (int tries = 0; tries < 50; ++tries) {
            const int a = val[j];
            const int b = use_var && tries < 25 ? val[k] : rand_in(rng, 1, 9);
            const bool b_is_var = use_var && tries < 25;
            const char op = "+-*"[rng.below(3)];
            int r;
            if (op == '+') r = a + b;
            else if (op == '-') r = a - b;
            else r = a * b;
            if (r < 0 || r > 99 || (op == '*' && b < 2)) continue;
            const std::string bs = b_is_var ? name(k) : std::to_string(b);
            val.push_back(r);
            defs.push_back(name(i) + "=" + name(j) + op + bs);
            steps.push_back(name(i) + "=" + name(j) + op + bs + "=" + std::to_string(a) + op +
                            std::to_string(b) + "=" + std::to_string(r));
            return;
        }
        // No operation fit in 50 tries (rare): v[i] = v[j]+0 always does.
        val.push_back(val[j]);
        defs.push_back(name(i) + "=" + name(j) + "+0");
        steps.push_back(name(i) + "=" + name(j) + "+0=" + std::to_string(val[j]) + "+0=" +
                        std::to_string(val[j]));
    };
    for (int i = 0; i < hops; ++i) make(i, true);
    const std::vector<std::string> solution(steps.begin(), steps.end());
    for (int i = hops; i < hops + distract; ++i) make(i, false);
    std::vector<std::string> shuffled = defs;
    for (int i = int(shuffled.size()) - 1; i > 0; --i) std::swap(shuffled[i], shuffled[rng.below(i + 1)]);
    Problem p;
    p.task = "chain";
    p.difficulty = d;
    p.question = join(shuffled, "; ") + "; " + name(hops - 1) + "?";
    p.steps = join(solution, "; ");
    p.answer = std::to_string(val[hops - 1]);
    return p;
}

inline Problem add(Rng& rng, int d) {
    const int digits = d + 1;
    const long a = rand_digits(rng, digits), b = rand_digits(rng, rand_in(rng, std::max(1, digits - 1), digits));
    std::vector<std::string> cols;
    long x = a, y = b;
    int carry = 0;
    while (x > 0 || y > 0) {
        const int dx = int(x % 10), dy = int(y % 10), s = dx + dy + carry;
        cols.push_back(std::to_string(dx) + "+" + std::to_string(dy) + "+" + std::to_string(carry) +
                       "=" + std::to_string(s));
        carry = s / 10;
        x /= 10;
        y /= 10;
    }
    Problem p;
    p.task = "add";
    p.difficulty = d;
    p.question = std::to_string(a) + "+" + std::to_string(b);
    p.steps = join(cols, ", ");
    p.answer = std::to_string(a + b);
    return p;
}

inline Problem mul(Rng& rng, int d) {
    // d=1: 2-digit x 1-digit ... d=5: 3-digit x 3-digit.
    static const int shapes[5][2] = {{2, 1}, {3, 1}, {2, 2}, {3, 2}, {3, 3}};
    const auto& s = shapes[std::clamp(d, 1, 5) - 1];
    const long a = rand_digits(rng, s[0]);
    const long b = s[1] == 1 ? rand_in(rng, 2, 9) : rand_digits(rng, s[1]);
    // a times each nonzero digit of b (at its place value), each product
    // split over a's digits; then the partial products are added.
    std::vector<std::string> steps, partials;
    long total = 0, place = 1;
    for (long bb = b; bb > 0; bb /= 10, place *= 10) {
        const long m = (bb % 10) * place;
        if (m == 0) continue;
        std::vector<std::string> terms;
        long ap = 1;
        for (long aa = a; aa > 0; aa /= 10, ap *= 10)
            if (aa % 10) terms.push_back(std::to_string((aa % 10) * ap * m));
        std::reverse(terms.begin(), terms.end());
        const long prod = a * m;
        std::string step = std::to_string(a) + "*" + std::to_string(m) + "=";
        if (terms.size() > 1) step += join(terms, "+") + "=";
        steps.push_back(step + std::to_string(prod));
        partials.push_back(std::to_string(prod));
        total += prod;
    }
    if (partials.size() > 1) steps.push_back(join(partials, "+") + "=" + std::to_string(total));
    Problem p;
    p.task = "mul";
    p.difficulty = d;
    p.question = std::to_string(a) + "*" + std::to_string(b);
    p.steps = join(steps, "; ");
    p.answer = std::to_string(a * b);
    return p;
}

// A random expression tree with `ops` operators over single digits.
struct Expr {
    char op = 0;  // 0 = leaf
    long value = 0;
    int left = -1, right = -1;
};

inline Problem expr(Rng& rng, int d) {
    const int ops = d + 1;
    std::vector<Expr> nodes;
    std::function<int(int)> build = [&](int n_ops) -> int {
        if (n_ops == 0) {
            nodes.push_back({0, rand_in(rng, 1, 9), -1, -1});
            return int(nodes.size()) - 1;
        }
        const int left_ops = int(rng.below(uint64_t(n_ops)));
        const int l = build(left_ops), r = build(n_ops - 1 - left_ops);
        char op = "+-*"[rng.below(3)];
        const long a = nodes[l].value, b = nodes[r].value;
        if (op == '*' && a * b > 999) op = '+';
        if (op == '-' && a < b) op = '+';
        const long v = op == '+' ? a + b : op == '-' ? a - b : a * b;
        nodes.push_back({op, v, l, r});
        return int(nodes.size()) - 1;
    };
    const int root = build(ops);
    std::vector<std::string> steps;
    // Post-order: each operation once both operands are numbers.
    std::function<void(int)> solve = [&](int i) {
        if (!nodes[i].op) return;
        solve(nodes[i].left);
        solve(nodes[i].right);
        steps.push_back(std::to_string(nodes[nodes[i].left].value) + nodes[i].op +
                        std::to_string(nodes[nodes[i].right].value) + "=" + std::to_string(nodes[i].value));
    };
    solve(root);
    // Print with parentheses around every non-leaf operand.
    std::function<std::string(int, bool)> print = [&](int i, bool top) -> std::string {
        if (!nodes[i].op) return std::to_string(nodes[i].value);
        const std::string s = print(nodes[i].left, false) + nodes[i].op + print(nodes[i].right, false);
        return top ? s : "(" + s + ")";
    };
    Problem p;
    p.task = "expr";
    p.difficulty = d;
    p.question = print(root, true);
    p.steps = join(steps, "; ");
    p.answer = std::to_string(nodes[root].value);
    return p;
}

inline Problem eq(Rng& rng, int d) {
    int max_b = 9;
    for (int i = 1; i < d; ++i) max_b = max_b * 10 + 9;
    const int a = rand_in(rng, 2, 9), x = rand_in(rng, 0, std::min(99, 9 * d + 3));
    const int b = rand_in(rng, 1, std::min(max_b, 999));
    const bool minus = rng.below(2) && a * x >= b;
    const int c = minus ? a * x - b : a * x + b;
    const std::string ax = std::to_string(a) + "*x";
    Problem p;
    p.task = "eq";
    p.difficulty = d;
    p.question = ax + (minus ? "-" : "+") + std::to_string(b) + "=" + std::to_string(c);
    p.steps = ax + "=" + std::to_string(c) + (minus ? "+" : "-") + std::to_string(b) + "=" +
              std::to_string(a * x) + "; x=" + std::to_string(a * x) + "/" + std::to_string(a) +
              "=" + std::to_string(x);
    p.answer = std::to_string(x);
    return p;
}

inline Problem cmp(Rng& rng, int d) {
    const int digits = d + 1;
    const long a = rand_digits(rng, digits);
    // Usually share a prefix with a, so the comparison needs several digits.
    std::string bs = std::to_string(a);
    const int keep = int(rng.below(uint64_t(digits)));
    for (int i = keep; i < digits; ++i) bs[i] = char('0' + rng.below(10));
    if (bs[0] == '0' && digits > 1) bs[0] = '1';
    long b = std::stol(bs);
    if (b == a) b = a + 1;
    const std::string sa = std::to_string(a), sb = std::to_string(b);
    std::vector<std::string> steps;
    if (sa.size() != sb.size()) {
        steps.push_back(std::to_string(sa.size()) + " digits vs " + std::to_string(sb.size()));
    } else {
        for (size_t i = 0; i < sa.size(); ++i) {
            const char rel = sa[i] == sb[i] ? '=' : sa[i] > sb[i] ? '>' : '<';
            steps.push_back(std::string(1, sa[i]) + rel + sb[i]);
            if (rel != '=') break;
        }
    }
    Problem p;
    p.task = "cmp";
    p.difficulty = d;
    p.question = "max(" + sa + "," + sb + ")";
    p.steps = join(steps, "; ");
    p.answer = std::to_string(std::max(a, b));
    return p;
}

}  // namespace reasoning_detail

inline const std::vector<std::string>& reasoning_tasks() {
    static const std::vector<std::string> t = {"chain", "add", "mul", "expr", "eq", "cmp"};
    return t;
}

inline Problem make_problem(const std::string& task, int difficulty, Rng& rng) {
    using namespace reasoning_detail;
    if (task == "chain") return chain(rng, difficulty);
    if (task == "add") return add(rng, difficulty);
    if (task == "mul") return mul(rng, difficulty);
    if (task == "expr") return expr(rng, difficulty);
    if (task == "eq") return eq(rng, difficulty);
    if (task == "cmp") return cmp(rng, difficulty);
    CHECK(false, "unknown task '%s'", task.c_str());
    return {};
}

// Problem sets on disk: one per line, tab-separated
// task, difficulty, question, steps, answer.
// One problem per line: task, difficulty, question, steps, answer, separated
// by tabs. Backslash, tab and newline inside a field are written as \\, \t
// and \n.
inline std::string escape_field(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '\\') o += "\\\\";
        else if (c == '\t') o += "\\t";
        else if (c == '\n') o += "\\n";
        else o += c;
    }
    return o;
}

inline std::string unescape_field(const std::string& s) {
    std::string o;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] != '\\' || i + 1 == s.size()) {
            o += s[i];
            continue;
        }
        const char c = s[++i];
        o += c == 't' ? '\t' : c == 'n' ? '\n' : c;
    }
    return o;
}

inline void save_problems(const std::string& path, const std::vector<Problem>& ps) {
    FILE* f = std::fopen(path.c_str(), "w");
    CHECK(f, "cannot write %s", path.c_str());
    for (const auto& p : ps)
        std::fprintf(f, "%s\t%d\t%s\t%s\t%s\n", escape_field(p.task).c_str(), p.difficulty,
                     escape_field(p.question).c_str(), escape_field(p.steps).c_str(),
                     escape_field(p.answer).c_str());
    CHECK(std::fclose(f) == 0, "error writing %s", path.c_str());
}

inline std::vector<Problem> load_problems(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "r");
    CHECK(f, "cannot open %s", path.c_str());
    std::vector<Problem> ps;
    std::string line;
    for (int ch; (ch = std::fgetc(f)) != EOF;) {
        if (ch != '\n') {
            line += char(ch);
            continue;
        }
        std::vector<std::string> fields;
        size_t pos = 0;
        for (size_t tab; (tab = line.find('\t', pos)) != std::string::npos; pos = tab + 1)
            fields.push_back(line.substr(pos, tab - pos));
        fields.push_back(line.substr(pos));
        CHECK(fields.size() == 5, "bad line in %s: %s", path.c_str(), line.c_str());
        Problem p;
        p.task = unescape_field(fields[0]);
        p.difficulty = std::atoi(fields[1].c_str());
        p.question = unescape_field(fields[2]);
        p.steps = unescape_field(fields[3]);
        p.answer = unescape_field(fields[4]);
        ps.push_back(std::move(p));
        line.clear();
    }
    std::fclose(f);
    return ps;
}

// The answer the model gave: the text after "#### " up to the end of the line.
inline std::string extract_answer(const std::string& completion) {
    const size_t i = completion.find("#### ");
    if (i == std::string::npos) return "";
    const size_t j = completion.find('\n', i);
    return completion.substr(i + 5, j == std::string::npos ? std::string::npos : j - i - 5);
}
