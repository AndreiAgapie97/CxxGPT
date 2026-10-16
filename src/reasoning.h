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
//          (with chain_plan, it first looks them up backward from the
//          question: "c=h+8; h=6 => h=6; c=h+8=6+8=14")
//   add    multi-digit addition, column by column with carries
//   mul    multi-digit multiplication via partial products
//   expr   nested arithmetic expression, innermost operation first
//          (with expr_rewrite, each step also writes what is left:
//          "(2+3)*4" -> "2+3=5: 5*4; 5*4=20")
//   eq     linear equation a*x+b=c, solved in two steps
//   cmp    which of two numbers is larger, digit by digit
#pragma once
#include <charconv>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "common.h"

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

}  // namespace reasoning_detail

// How solutions are written. The problems themselves (the questions and
// answers) are the same in every format: formats draw no random numbers.
struct Formats {
    bool chain_plan = false;    // chain: look the variables up first
    bool expr_rewrite = false;  // expr: write the expression left after each step
};

namespace reasoning_detail {

// A number with exactly `digits` digits (no leading zero unless 1 digit).
inline int64_t rand_digits(Rng& rng, int digits) {
    if (digits <= 1) return rand_in(rng, 0, 9);
    int64_t lo = 1;
    for (int i = 1; i < digits; ++i) lo *= 10;
    return lo + int64_t(rng.below(uint64_t(9 * lo)));
}

inline std::string join(const std::vector<std::string>& parts, const std::string& sep) {
    std::string s;
    for (size_t i = 0; i < parts.size(); ++i) s += (i ? sep : "") + parts[i];
    return s;
}

inline Problem chain(Rng& rng, int d, bool plan) {
    // Variables v[0..n): v[0] is a literal; v[i] = v[j] op (number or v[k])
    // with j, k < i. The question asks for v[n-1]; distractors use the same
    // variables but nothing depends on them.
    //
    // The plain solution computes v[0], v[1], ... in order, so its first
    // step already has to know which of the variables defined from v[0]
    // leads to the question. With plan, the solution first copies the
    // definitions backward from the question: v[n-1]'s, then that of the
    // first variable on its right-hand side (v[n-2]), and so on down to
    // v[0]. Every step is then a lookup of the variable just written.
    // Both versions draw the same random numbers: same questions, same
    // answers.
    const int hops = d + 1, distract = (d + 1) / 2 + 1;
    std::string letters = "abcdefghijklmnopqrstuvwxyz";
    for (int i = 25; i > 0; --i) std::swap(letters[i], letters[rng.below(i + 1)]);
    std::vector<int> val;
    std::vector<std::string> defs, steps;
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
            // No variable is 0: "u=n-n" or "f=t-2" with t=2 would make every
            // later step "0+0" or "0*8", and the answer 0 whatever the numbers.
            if (r < 1 || r > 99 || (op == '*' && b < 2)) continue;
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
    if (plan) {
        std::vector<std::string> lookup(defs.rbegin() + int(defs.size()) - hops, defs.rend());
        p.steps = join(lookup, "; ") + " => " + p.steps;
    }
    p.answer = std::to_string(val[hops - 1]);
    return p;
}

inline Problem add(Rng& rng, int d) {
    const int digits = d + 1;
    // b has digits - 1 or digits digits; a one-digit b is 1..9, never "88+0".
    const int64_t a = rand_digits(rng, digits);
    const int b_digits = rand_in(rng, std::max(1, digits - 1), digits);
    const int64_t b = b_digits == 1 ? rand_in(rng, 1, 9) : rand_digits(rng, b_digits);
    std::vector<std::string> cols;
    int64_t x = a, y = b;
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
    const int64_t a = rand_digits(rng, s[0]);
    const int64_t b = s[1] == 1 ? rand_in(rng, 2, 9) : rand_digits(rng, s[1]);
    // a times each nonzero digit of b (at its place value), each product
    // split over a's digits; then the partial products are added.
    std::vector<std::string> steps, partials;
    int64_t total = 0, place = 1;
    for (int64_t bb = b; bb > 0; bb /= 10, place *= 10) {
        const int64_t m = (bb % 10) * place;
        if (m == 0) continue;
        std::vector<std::string> terms;
        int64_t ap = 1;
        for (int64_t aa = a; aa > 0; aa /= 10, ap *= 10)
            if (aa % 10) terms.push_back(std::to_string((aa % 10) * ap * m));
        std::reverse(terms.begin(), terms.end());
        const int64_t prod = a * m;
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
    int64_t value = 0;
    int left = -1, right = -1;
};

inline Problem expr(Rng& rng, int d, bool rewrite) {
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
        const int64_t a = nodes[l].value, b = nodes[r].value;
        if (op == '*' && a * b > 999) op = '+';
        if (op == '-' && a <= b) op = '+';  // never negative, and never a-a = 0
        const int64_t v = op == '+' ? a + b : op == '-' ? a - b : a * b;
        nodes.push_back({op, v, l, r});
        return int(nodes.size()) - 1;
    };
    const int root = build(ops);
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
    std::vector<std::string> steps;
    // Post-order: each operation once both operands are numbers. A computed
    // node becomes a number, so printing the tree shows what is left.
    std::function<void(int)> solve = [&](int i) {
        if (!nodes[i].op) return;
        solve(nodes[i].left);
        solve(nodes[i].right);
        std::string step = std::to_string(nodes[nodes[i].left].value) + nodes[i].op +
                           std::to_string(nodes[nodes[i].right].value) + "=" + std::to_string(nodes[i].value);
        nodes[i].op = 0;
        if (rewrite && i != root) step += ": " + print(root, true);
        steps.push_back(step);
    };
    solve(root);
    p.steps = join(steps, "; ");
    p.answer = std::to_string(nodes[root].value);
    return p;
}

inline Problem eq(Rng& rng, int d) {
    int max_b = 9;
    for (int i = 1; i < d; ++i) max_b = max_b * 10 + 9;
    const int a = rand_in(rng, 2, 9), x = rand_in(rng, 1, std::min(99, 9 * d + 3));  // x = 0 is trivial
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
    const int64_t a = rand_digits(rng, digits);
    // Usually share a prefix with a, so the comparison needs several digits.
    std::string bs = std::to_string(a);
    const int keep = int(rng.below(uint64_t(digits)));
    for (int i = keep; i < digits; ++i) bs[i] = char('0' + rng.below(10));
    if (bs[0] == '0' && digits > 1) bs[0] = '1';
    int64_t b = std::stoll(bs);
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

inline Problem make_problem(const std::string& task, int difficulty, Rng& rng, const Formats& f = {}) {
    CHECK(difficulty >= 1 && difficulty <= 5, "reasoning difficulty must be in [1,5]");
    using namespace reasoning_detail;
    if (task == "chain") return chain(rng, difficulty, f.chain_plan);
    if (task == "add") return add(rng, difficulty);
    if (task == "mul") return mul(rng, difficulty);
    if (task == "expr") return expr(rng, difficulty, f.expr_rewrite);
    if (task == "eq") return eq(rng, difficulty);
    if (task == "cmp") return cmp(rng, difficulty);
    CHECK(false, "unknown task '%s'", task.c_str());
    return {};
}

// ---------------------------------------------------------------------------
// Planted mistakes (gen-reasoning --mistakes P)
//
// One statement of a solution is written wrong first, then right:
//   b=a+3=4+3=7   becomes   [[no-loss]]b=a+3=4+3=8[[/no-loss]]. Wait: b=a+3=4+3=7
// The marked wrong statement is context only: it is never trained on
// (loss_mask.h), so the model learns to correct a mistake, not to make one.
//
// Most tasks have calculations: numbers and operators worked out into a
// result at the end of a statement. The one corrupted, per task:
//   chain  "b=a+3=4+3=7"        4+3 (the definition "a=4" and the
//                               chain_plan lookup "b=a+3" work nothing out)
//   add    "4+3+1=8"            one column: digit + digit + carry
//   mul    "23*4=80+12=92"      80+12 ("23*4=80+12" only splits the product)
//   expr   "7-2=5: 5*3"         7-2 (the expr_rewrite text after ':' is not)
//   eq     "3*x=20-8=12; x=12/3=4"   20-8 or 12/3
//   cmp    "1=1; 2<3", "3 digits vs 2"   nothing is worked out ("1=1" is
//          not a calculation), so plant_comparison_mistake turns the
//          deciding relation around or miscounts the digits instead.

namespace mistake_detail {

// steps[begin, end) is one statement, e.g. "b=a+3=4+3=7": steps[expr,
// value - 1) is what is worked out ("4+3"), steps[value, end) the result.
struct Calculation {
    size_t begin, expr, value, end;
    int64_t correct;
};

inline bool is_digit(char c) { return c >= '0' && c <= '9'; }
inline bool is_operator(char c) { return c == '+' || c == '-' || c == '*' || c == '/'; }

inline std::vector<Calculation> find_calculations(const std::string& s) {
    std::vector<Calculation> found;
    size_t begin = 0;  // where the current statement starts
    for (size_t i = 0; i < s.size(); ++i) {
        if (s.compare(i, 2, "; ") == 0 || s.compare(i, 2, ", ") == 0) begin = i + 2;
        if (s.compare(i, 4, " => ") == 0) begin = i + 4;
        if (s[i] != '=') continue;
        // A result: digits right after '=', ending the statement (or followed
        // by expr_rewrite's ':'). "23*4=80+12" goes on, so 80 is not one.
        size_t end = i + 1;
        while (end < s.size() && is_digit(s[end])) ++end;
        if (end == i + 1 || (end < s.size() && s[end] != ';' && s[end] != ',' && s[end] != ':')) continue;
        // What is worked out: the numbers and operators just before '='.
        size_t expr = i;
        while (expr > begin && (is_digit(s[expr - 1]) || is_operator(s[expr - 1]))) --expr;
        bool has_operator = false;
        for (size_t k = expr; k < i; ++k) has_operator = has_operator || is_operator(s[k]);
        if (!has_operator) continue;  // "a=4" or "8=8": nothing worked out
        int64_t correct = 0;
        const auto parsed = std::from_chars(s.data() + i + 1, s.data() + end, correct);
        CHECK(parsed.ec == std::errc{}, "bad generated numerical step");
        found.push_back({begin, expr, i + 1, end, correct});
    }
    return found;
}

// The value of numbers joined by + - * / ("4*3+0"), * and / first.
inline int64_t evaluate(std::string_view e) {
    std::vector<int64_t> numbers;
    std::string ops;
    for (size_t i = 0; i < e.size();) {
        int64_t v = 0;
        const auto r = std::from_chars(e.data() + i, e.data() + e.size(), v);
        CHECK(r.ec == std::errc{}, "bad calculation '%.*s'", int(e.size()), e.data());
        numbers.push_back(v);
        i = size_t(r.ptr - e.data());
        if (i < e.size()) ops += e[i++];
    }
    CHECK(numbers.size() == ops.size() + 1, "bad calculation '%.*s'", int(e.size()), e.data());
    // Fold each * or / into the number before it; then add and subtract.
    std::vector<int64_t> terms = {numbers[0]};
    std::string signs;
    for (size_t k = 0; k < ops.size(); ++k) {
        const int64_t b = numbers[k + 1];
        if (ops[k] == '*') {
            terms.back() *= b;
        } else if (ops[k] == '/') {
            CHECK(b != 0, "division by zero in '%.*s'", int(e.size()), e.data());
            terms.back() /= b;
        } else {
            terms.push_back(b);
            signs += ops[k];
        }
    }
    int64_t value = terms[0];
    for (size_t k = 0; k < signs.size(); ++k) value += signs[k] == '+' ? terms[k + 1] : -terms[k + 1];
    return value;
}

enum MistakeKind { kWrongDigit, kMissedCarry, kWrongOperator };

// A wrong result for the calculation `expr`, whose right result is
// `correct`; or -1 when this kind of mistake does not fit it (a result is
// never negative, so -1 cannot be confused with one).
inline int64_t wrong_value(std::string_view expr, int64_t correct, MistakeKind kind, Rng& rng) {
    switch (kind) {
    case kWrongDigit: {  // one digit of the result wrong, the others right
        std::string digits = std::to_string(correct);
        const size_t i = size_t(rng.below(digits.size()));
        digits[i] = char('0' + (digits[i] - '0' + 1 + rng.below(9)) % 10);
        return std::stoll(digits);
    }
    case kMissedCarry:  // a carry or borrow forgotten (one off for a single digit)
        return correct >= 10 ? correct - 10 : correct + 1;
    case kWrongOperator: {
        // A learner who slips on an operator still works out the rest of
        // the calculation right. So the whole calculation is redone with
        // its first operator swapped for the one it is most often mixed up
        // with (in an add column the first one joins the two digits, and
        // the carry is still added; elsewhere there is usually only one):
        //   7*4    as 7+4   = 11    (* and + mixed up)
        //   4+3+0  as 4*3+0 = 12    (+ between single digits read as *)
        //   20-8   as 20+8  = 28    (a sign slip, e.g. moving -8 across the =)
        //   80+12  as 80-12 = 68    (a sign slip between bigger numbers)
        //   12/3   as 12*3  = 36    (3 moved across the = without inverting it)
        const size_t op = expr.find_first_of("+-*/");
        if (op == expr.npos) return -1;
        const bool single_digits = op == 1 && (expr.size() == 3 || (expr.size() > 3 && is_operator(expr[3])));
        std::string slipped(expr);
        slipped[op] = expr[op] == '*' ? '+' : expr[op] == '/' ? '*' : expr[op] == '-' ? '+'
                      : single_digits ? '*' : '-';
        const int64_t wrong = evaluate(slipped);
        // 2+2 as 2*2 is still 4, and 7+12 as 7-12 is negative: no mistake.
        return wrong >= 0 && wrong != correct ? wrong : -1;
    }
    }
    return -1;
}

// statement = steps[begin, end), which is right: put `wrong` before it,
// marked as not trained on, and a phrase that introduces the correction.
inline void insert_mistake(Problem& p, size_t begin, size_t end, const std::string& wrong, Rng& rng) {
    static const char* corrections[] = {"Wait: ", "Correction: ", "Let me check: ", "That was wrong: "};
    const std::string right = p.steps.substr(begin, end - begin);
    CHECK(wrong != right, "planted statement must be wrong");
    p.steps.replace(begin, end - begin, "[[no-loss]]" + wrong + "[[/no-loss]]. " + corrections[rng.below(4)] + right);
}

// cmp: "1=1; 2<3" (same length: digit by digit until one differs) or
// "3 digits vs 2". The last step decides; it is the one corrupted:
//   "2<3" -> "2>3"                      (the relation turned around)
//   "3 digits vs 2" -> "4 digits vs 2"  (one number's digits miscounted)
// The "1=1" steps are left alone: "1<1" is not a slip anyone makes.
inline bool plant_comparison_mistake(Problem& p, Rng& rng) {
    const size_t last = p.steps.rfind("; ");
    const size_t begin = last == std::string::npos ? 0 : last + 2;
    const std::string right = p.steps.substr(begin);
    std::string wrong = right;
    const std::string vs = " digits vs ";
    const size_t at = right.find(vs);
    if (at != std::string::npos) {
        // Miscount one side by one, never down to zero digits.
        const bool left = rng.below(2) == 0;
        const size_t from = left ? 0 : at + vs.size(), to = left ? at : right.size();
        const int count = std::stoi(right.substr(from, to - from));
        const int miscount = count > 1 && rng.below(2) ? count - 1 : count + 1;
        wrong.replace(from, to - from, std::to_string(miscount));
    } else {
        // "2<3": a digit, the relation, a digit.
        if (right.size() != 3 || (right[1] != '<' && right[1] != '>')) return false;
        wrong[1] = right[1] == '<' ? '>' : '<';
    }
    insert_mistake(p, begin, p.steps.size(), wrong, rng);
    return true;
}

}  // namespace mistake_detail

// Plants one mistake in p's solution, followed by its correction; false if
// p has nothing that fits (every generated problem has something).
// The CLI uses its own RNG for this, so the problems stay the same.
inline bool plant_mistake(Problem& p, Rng& rng) {
    using namespace mistake_detail;
    if (p.task == "cmp") return plant_comparison_mistake(p, rng);
    const std::vector<Calculation> calculations = find_calculations(p.steps);
    if (calculations.empty()) return false;
    const Calculation c = calculations[rng.below(calculations.size())];
    const std::string_view expr(p.steps.data() + c.expr, c.value - 1 - c.expr);
    int64_t wrong = wrong_value(expr, c.correct, MistakeKind(rng.below(3)), rng);
    // A wrong operator that changes nothing: a wrong digit or carry instead.
    if (wrong < 0) wrong = wrong_value(expr, c.correct, MistakeKind(rng.below(2)), rng);
    CHECK(wrong >= 0 && wrong != c.correct, "planted value must be wrong");
    insert_mistake(p, c.begin, c.end, p.steps.substr(c.begin, c.value - c.begin) + std::to_string(wrong), rng);
    return true;
}

// Problem sets on disk: one per line, tab-separated
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

// Written in binary, so the file has the same bytes on Windows as on Linux
// (scripts record its sha256); load_problems accepts "\r\n" too.
inline void save_problems(const std::string& path, const std::vector<Problem>& ps) {
    FILE* f = create_file(path, "wb");
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
    auto parse_line = [&] {
        if (!line.empty() && line.back() == '\r') line.pop_back();  // CRLF record terminator
        std::vector<std::string> fields;
        size_t pos = 0;
        for (size_t tab; (tab = line.find('\t', pos)) != std::string::npos; pos = tab + 1)
            fields.push_back(line.substr(pos, tab - pos));
        fields.push_back(line.substr(pos));
        CHECK(fields.size() == 5, "bad line in %s: %s", path.c_str(), line.c_str());
        Problem p;
        p.task = unescape_field(fields[0]);
        const auto parsed = std::from_chars(fields[1].data(), fields[1].data() + fields[1].size(), p.difficulty);
        CHECK(parsed.ec == std::errc{} && parsed.ptr == fields[1].data() + fields[1].size() && p.difficulty > 0,
              "bad difficulty in %s: %s", path.c_str(), fields[1].c_str());
        p.question = unescape_field(fields[2]);
        p.steps = unescape_field(fields[3]);
        p.answer = unescape_field(fields[4]);
        ps.push_back(std::move(p));
        line.clear();
    };
    for (int ch; (ch = std::fgetc(f)) != EOF;) {
        if (ch == '\n') parse_line();
        else line += char(ch);
    }
    CHECK(!std::ferror(f), "error reading %s", path.c_str());
    if (!line.empty()) parse_line();  // a final record need not end with a newline
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

// Shared by QA preparation and generated-answer scoring.
inline std::string trim(const std::string& s) {
    const size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    return s.substr(b, s.find_last_not_of(" \t\r\n") - b + 1);
}

// "1,080" -> "1080": commas that group digits go, so the exact-match check
// ignores digit grouping; other commas, as in "(1, 2)", stay.
inline std::string strip_digit_commas(const std::string& s) {
    auto digit = [&](size_t i) { return i < s.size() && s[i] >= '0' && s[i] <= '9'; };
    std::string o;
    for (size_t i = 0; i < s.size(); ++i) {
        const bool grouping = s[i] == ',' && i > 0 && digit(i - 1) && digit(i + 1) && digit(i + 2) &&
                              digit(i + 3) && !digit(i + 4);
        if (!grouping) o += s[i];
    }
    return o;
}

inline bool answer_matches(const std::string& completion, const std::string& gold) {
    return strip_digit_commas(trim(extract_answer(completion))) == gold;
}
