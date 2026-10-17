// test_reasoning.h — every generated problem is checked by an independent
// solver that only reads the question text.
#pragma once

#include <algorithm>
#include <cctype>
#include <map>
#include <tuple>

#include "../src/reasoning.h"
#include "../src/loss_mask.h"
#include "../src/train.h"
#include "test_util.h"

namespace reasoning_check {

// Recursive descent over + - * and parentheses, usual precedence.
struct Parser {
    const std::string& s;
    size_t i = 0;
    int64_t number() {
        int64_t v = 0;
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) v = v * 10 + (s[i++] - '0');
        return v;
    }
    int64_t atom() {
        if (s[i] == '(') {
            ++i;
            const int64_t v = sum();
            ++i;  // ')'
            return v;
        }
        return number();
    }
    int64_t product() {
        int64_t v = atom();
        while (i < s.size() && (s[i] == '*' || s[i] == '/')) {
            const char op = s[i++];
            const int64_t rhs = atom();
            v = op == '*' ? v * rhs : v / rhs;
        }
        return v;
    }
    int64_t sum() {
        int64_t v = product();
        while (i < s.size() && (s[i] == '+' || s[i] == '-')) {
            const char op = s[i++];
            const int64_t r = product();
            v = op == '+' ? v + r : v - r;
        }
        return v;
    }
};

inline int64_t eval(const std::string& s) {
    Parser p{s};
    return p.sum();
}

// "b=a+4; a=3; c=b*k; ...; c?" -> value of c
inline int64_t solve_chain(const std::string& q) {
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
    std::function<int64_t(char)> value = [&](char v) -> int64_t {
        std::string e;
        for (char ch : defs.at(v)) e += std::islower(static_cast<unsigned char>(ch)) ? std::to_string(value(ch)) : std::string(1, ch);
        return eval(e);
    };
    return value(target);
}

// Since 2026-10-16 the curriculum has no degenerate zeros: no chain
// variable (needed or not) is 0, no add operand is 0, eq's x is at least 1,
// and expr never subtracts a number from itself, so no expr step gives 0.
inline void check_no_zeros(const Problem& p) {
    const std::string& q = p.question;
    if (p.task == "chain") {
        for (size_t pos = 0; pos < q.size();) {
            const size_t end = q.find("; ", pos);
            const std::string d = q.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
            if (d.size() > 2 && d[1] == '=') {
                // The same definitions, asking for this variable instead.
                const int64_t v = solve_chain(q.substr(0, q.rfind("; ")) + "; " + d[0] + "?");
                EXPECT(v != 0, "chain variable %c is 0: %s", d[0], q.c_str());
            }
            if (end == std::string::npos) break;
            pos = end + 2;
        }
    }
    if (p.task == "add") {
        const size_t plus = q.find('+');
        EXPECT(std::stoll(q.substr(0, plus)) > 0 && std::stoll(q.substr(plus + 1)) > 0, "add with a 0: %s", q.c_str());
    }
    if (p.task == "eq") EXPECT(p.answer != "0", "eq with x = 0: %s", q.c_str());
    if (p.task == "expr")
        for (size_t at = 0; (at = p.steps.find('=', at)) != std::string::npos; ++at) {
            const size_t end = p.steps.find_first_not_of("0123456789", at + 1);
            EXPECT(p.steps.substr(at + 1, end == std::string::npos ? std::string::npos : end - at - 1) != "0",
                   "an expr step gives 0: %s", p.steps.c_str());
        }
}

inline std::string solve(const Problem& p) {
    const std::string& q = p.question;
    if (p.task == "chain") return std::to_string(solve_chain(q));
    if (p.task == "add" || p.task == "mul" || p.task == "expr") return std::to_string(eval(q));
    if (p.task == "eq") {  // a*x+b=c or a*x-b=c: try every x
        const size_t eqpos = q.find('=');
        const int64_t c = std::stoll(q.substr(eqpos + 1));
        for (int64_t x = 0; x < 100000; ++x) {
            const std::string lhs = q.substr(0, eqpos);
            std::string e;
            for (char ch : lhs) e += ch == 'x' ? std::to_string(x) : std::string(1, ch);
            if (eval(e) == c) return std::to_string(x);
        }
        return "?";
    }
    if (p.task == "cmp") {  // max(a,b)
        const size_t comma = q.find(',');
        return std::to_string(std::max(std::stoll(q.substr(4, comma - 4)), std::stoll(q.substr(comma + 1))));
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
        const int64_t ops = std::count_if(left.begin(), left.end(), [](char c) { return c == '+' || c == '-' || c == '*'; });
        EXPECT(ops == int64_t(ps.size() - 1 - i) && std::to_string(reasoning_check::eval(left)) == rw.answer,
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

inline bool multiplication_steps_correct(const Problem& p) {
    const size_t star = p.question.find('*');
    const int64_t a = std::stoll(p.question.substr(0,star)), b = std::stoll(p.question.substr(star+1));
    const auto statements = split_steps(p.steps);
    std::vector<int64_t> multipliers;
    for (int64_t rest=b, place=1; rest; rest/=10, place*=10)
        if (rest%10) multipliers.push_back((rest%10)*place);
    if (statements.size() != multipliers.size() + (multipliers.size()>1)) return false;
    std::string partials;
    for (size_t i = 0; i < multipliers.size(); ++i) {
        const auto& statement = statements[i];
        const size_t eq = statement.find('=');
        if (eq == statement.npos || eq+1 >= statement.size() || statement.back()=='=' || statement.substr(0,eq) != std::to_string(a)+"*"+std::to_string(multipliers[i])) return false;
        const int64_t expected = a*multipliers[i];
        for (size_t pos=eq+1; pos<statement.size();) {
            const auto end=statement.find('=',pos);
            const auto expression=statement.substr(pos,end-pos);
            if (expression.empty() || reasoning_check::eval(expression) != expected) return false;
            if (end==statement.npos) break;
            pos=end+1;
        }
        partials += (partials.empty() ? "" : "+") + std::to_string(expected);
    }
    return multipliers.size()==1 || statements.back()==partials+"="+std::to_string(a*b);
}

// cmp's planted statement must be false for the question "max(A,B)": a
// deciding digit pair "x<y" with the relation turned around, or digit
// counts "N digits vs M" that are not A's and B's.
inline bool false_comparison(const std::string& question, const std::string& s) {
    const size_t comma = question.find(',');
    const std::string a = question.substr(4, comma - 4), b = question.substr(comma + 1, question.size() - comma - 2);
    const std::string vs = " digits vs ";
    const size_t at = s.find(vs);
    if (at != std::string::npos) {
        const unsigned long n = std::stoul(s.substr(0, at)), m = std::stoul(s.substr(at + vs.size()));
        return n >= 1 && m >= 1 && (n != a.size() || m != b.size());
    }
    if (s.size() != 3 || !std::isdigit(static_cast<unsigned char>(s[0])) ||
        !std::isdigit(static_cast<unsigned char>(s[2])) || s[0] == s[2]) return false;
    return s[1] == (s[0] > s[2] ? '<' : '>');
}

// A planted problem has one marked statement that is really wrong, then a
// correction that is right, the rest of the solution unchanged, the same
// answer, and a loss mask that hides exactly the wrong statement.
inline void check_planted_mistake(const Problem& plain, const Problem& planted) {
    constexpr size_t npos = std::string::npos;
    const char* steps = planted.steps.c_str();
    EXPECT(planted.question == plain.question && planted.answer == reasoning_check::solve(plain) &&
           extract_answer(planted.completion()) == plain.answer, "mistake changed the true answer: %s", steps);
    const size_t begin = planted.steps.find(kNoLossBegin), close = planted.steps.find(kNoLossEnd);
    EXPECT(begin != npos && close != npos && planted.steps.find(kNoLossBegin, begin + 1) == npos,
           "solution does not have exactly one wrong span: %s", steps);
    if (begin == npos || close == npos) return;
    const std::string wrong = planted.steps.substr(begin + kNoLossBegin.size(), close - begin - kNoLossBegin.size());
    const size_t correction = planted.steps.find(": ", close) + 2;  // after "Wait: " and the like
    if (plain.task == "cmp") {
        EXPECT(false_comparison(plain.question, wrong), "planted comparison is not false: %s", steps);
    } else {
        // "b=a+3=4+3=8": what is worked out lies between the last two '='.
        const size_t eq = wrong.rfind('=');
        EXPECT(eq != npos && eq > 0, "planted statement has no result: %s", steps);
        if (eq == npos || eq == 0) return;
        const size_t prefix_eq = wrong.rfind('=', eq - 1);
        const size_t from = prefix_eq == npos ? 0 : prefix_eq + 1;
        const std::string lhs = wrong.substr(from, eq - from);
        EXPECT(lhs.find_first_of("+-*/") != npos, "planted a statement that works nothing out: %s", steps);
        const int64_t expected = reasoning_check::eval(lhs);
        EXPECT(std::stoll(wrong.substr(eq + 1)) != expected, "planted value is actually right: %s", steps);
        const std::string right = wrong.substr(0, eq + 1) + std::to_string(expected);
        EXPECT(planted.steps.compare(correction, right.size(), right) == 0,
               "correction line is numerically wrong: %s", steps);
    }
    std::string recovered = planted.steps;
    recovered.erase(begin, correction - begin);
    EXPECT(recovered == plain.steps, "other solution steps changed: %s", steps);
    Tokenizer tok;
    std::vector<uint16_t> ids;
    std::vector<uint8_t> mask;
    encode_loss_marked(planted.completion(), tok, ids, mask);
    std::string hidden;
    for (size_t j = 0; j < ids.size(); ++j)
        if (!mask[j]) hidden += tok.token_bytes(ids[j]);
    EXPECT(hidden == wrong, "masked token bytes differ from precisely the wrong step: %s", steps);
}

inline void test_planted_mistakes() {
    // The wrong-operator examples documented in wrong_value.
    using mistake_detail::kWrongOperator;
    Rng unused(1);
    for (const auto& [expr, correct, wrong] : std::vector<std::tuple<std::string, int64_t, int64_t>>{
             {"7*4", 28, 11}, {"4+3+0", 7, 12}, {"20-8", 12, 28}, {"80+12", 92, 68}, {"12/3", 4, 36},
             {"2+2", 4, -1}, {"7+12", 19, -1}})
        EXPECT(mistake_detail::wrong_value(expr, correct, kWrongOperator, unused) == wrong,
               "wrong operator in %s", expr.c_str());
    int checked = 0;
    // The default formats and chain_plan + expr_rewrite write different
    // statements (lookups first, ':' after expr steps): check both.
    for (const Formats& formats : {Formats{}, Formats{true, true}}) {
        Rng problems(551), errors(19), repeat_errors(19);
        for (const auto& task : reasoning_tasks())
            for (int difficulty = 1; difficulty <= 5; ++difficulty)
                for (int i = 0; i < 70; ++i) {
                    const Problem plain = make_problem(task, difficulty, problems, formats);
                    Problem planted = plain, repeat = plain;
                    const bool planted_one = plant_mistake(planted, errors);
                    plant_mistake(repeat, repeat_errors);
                    EXPECT(planted_one, "no mistake fits: %s", plain.steps.c_str());
                    EXPECT(planted.steps == repeat.steps, "mistakes are not repeatable");
                    check_planted_mistake(plain, planted);
                    ++checked;
                }
    }
    // Deliberately put annotations inside a word that BPE can merge. Every
    // mask-zero token must decode to the wrong span, including its edges.
    Tokenizer bpe;
    bpe.train("prefixmistakesuffix prefixmistakesuffix Wait correction Wait correction", 290);
    const std::string text = "prefix[[no-loss]]mistake[[/no-loss]]suffix. Wait: correction";
    TokenData data; data.vocab = uint32_t(bpe.vocab_size());
    encode_loss_marked(text, bpe, data.tokens, data.mask);
    std::string all, wrong;
    for (size_t i = 0; i < data.tokens.size(); ++i) {
        all += bpe.token_bytes(data.tokens[i]);
        if (!data.mask[i]) wrong += bpe.token_bytes(data.tokens[i]);
    }
    EXPECT(all == "prefixmistakesuffix. Wait: correction" && wrong == "mistake", "BPE mask boundary crossed");
    // The training target is one token ahead of its input, so the mask must
    // exclude a wrong token when it is the target, including the first one.
    const int T = int(data.tokens.size()) - 1;
    std::vector<int> inputs(T), targets(T);
    Rng sampling(4);
    data.sample_batch(sampling, 1, T, 0, data.tokens.size(), inputs.data(), targets.data());
    for (int i = 0; i < T; ++i)
        EXPECT(targets[i] == (data.mask[i + 1] ? int(data.tokens[i + 1]) : -1), "wrong shifted-target mask");
    EXPECT(loss_marker_safe_split(text, 12) == text.find(kNoLossEnd) + kNoLossEnd.size(),
           "stream cut split a marked span");
    EXPECT(loss_marker_safe_split("[[no-loss]]partial", 13) == std::string::npos,
           "unfinished streaming span was accepted");
    std::printf("  %d wrong steps and corrections independently checked; exact byte/BPE/target masks checked\n", checked);
}

// Answers are compared as numbers (canonical_answer): what may surround the
// number, equal numbers written differently, and what must not match.
inline void test_answer_matches() {
    auto line = [](const std::string& a) { return "work\n#### " + a + "\n"; };
    const std::vector<std::pair<std::string, std::string>> same = {
        {"57", "57"},           {"57.0", "57"},          {"57.00", "57"},       {"$57", "57"},
        {"$ 57", "57"},         {"57 dollars", "57"},    {"57%", "57"},         {"$57.00 dollars.", "57"},
        {"57.", "57"},          {"57 square feet", "57"}, {"1,080", "1080"},    {"0.8", "0.80"},
        {"0.80", "0.8"},        {"-28.5", "-28.50"},     {"-$28.50", "-28.5"},  {"$-28.5", "-28.50"},
        {"57", "57.00"},        {"0", "0.0"},            {"-0", "0"},           {"0.05", "0.050"},
        {"698,201.57", "698201.57"}, {"123456789012345678901234567890", "123456789012345678901234567890.0"},
        {"(1, 2)", "(1, 2)"},   {"x = 4", "x = 4"},      {"", ""},
    };
    for (const auto& [answer, gold] : same)
        EXPECT(answer_matches(line(answer), gold), "'%s' must match gold '%s' (forms '%s', '%s')", answer.c_str(),
               gold.c_str(), canonical_answer(answer).c_str(), canonical_answer(gold).c_str());
    const std::vector<std::pair<std::string, std::string>> different = {
        {"57 or 58", "57"}, {"5.7", "57"},     {"570", "57"},        {"3", "-3"},          {"-3", "3"},
        {"57.5", "57"},     {"57.01", "57"},   {"fifty-seven", "57"}, {"", "57"},          {"0.5", "1/2"},
        {"1/2", "0.5"},     {"50+7", "57"},    {"about 57", "57"},   {"057", "57"},        {".5", "0.5"},
        {"57th", "57"},     {"57 dollars and 50 cents", "57"},       {"57 (rounded)", "57"},
        {"$57 each, $114 total", "57"},       {"57..", "57"},       {"(1,2)", "(1, 2)"},  {"x = 5", "x = 4"},
    };
    for (const auto& [answer, gold] : different)
        EXPECT(!answer_matches(line(answer), gold), "'%s' must not match gold '%s' (forms '%s', '%s')",
               answer.c_str(), gold.c_str(), canonical_answer(answer).c_str(), canonical_answer(gold).c_str());
    // The shortest forms themselves.
    EXPECT(canonical_answer(" $1,080.50 dollars. ") == "1080.5" && canonical_answer("-0.000") == "0" &&
               canonical_answer("0.80%") == "0.8" && canonical_answer("100") == "100",
           "shortest form changed");
    // Not one number: kept as written, trimmed and without grouping commas.
    EXPECT(canonical_answer(" 57 or 1,080 ") == "57 or 1080" && canonical_answer("057") == "057",
           "text that is not one number must stay as written");
}

inline void test_reasoning() {
    Rng mul_rng(69);
    for (int d=1; d<=5; ++d) for (int i=0; i<100; ++i) {
        auto p = make_problem("mul",d,mul_rng);
        EXPECT(multiplication_steps_correct(p), "question-grounded multiplication work is wrong: %s", p.question.c_str());
        const size_t first_eq=p.steps.find('='), first_end=p.steps.find("; ");
        p.steps.replace(first_eq+1, (first_end==p.steps.npos ? p.steps.size() : first_end)-first_eq-1, "0");
        EXPECT(!multiplication_steps_correct(p), "false intermediate product passed with correct final answer");
    }

    Rng eq_rng(67);
    for (int d : {3, 4}) for (int i = 0; i < 100; ++i) {
        const auto p = make_problem("eq", d, eq_rng);
        const size_t start = p.question.find("*x") + 3, end = p.question.find('=');
        const int b = std::stoi(p.question.substr(start, end - start));
        const int x = std::stoi(reasoning_check::solve(p));
        EXPECT(b >= 1 && b <= 999 && x >= 0 && x <= (d == 3 ? 30 : 39),
               "documented eq support changed: %s", p.question.c_str());
    }
    for (const std::string text : {"#### 1,080\n", "#### 1080 \n", "#### 1080\r\n"})
        EXPECT(answer_matches(text, "1080"), "normalized generated answer was marked wrong: %s", text.c_str());
    EXPECT(answer_matches("#### 14 \n", "14") && answer_matches("#### 14\r\n", "14"),
           "generated trailing whitespace was not trimmed");
    EXPECT(!answer_matches("#### 140\n", "14") && !answer_matches("no delimiter 14", "14"),
           "normalization accepted a wrong/missing answer");
    EXPECT(strip_digit_commas("1,234,567") == "1234567" && strip_digit_commas("(1, 2)") == "(1, 2)",
           "normalization changed non-grouping commas");
    test_answer_matches();
    Rng rng(41);
    int checked = 0;
    for (const auto& task : reasoning_tasks())
        for (int d = 1; d <= 5; ++d)
            for (int i = 0; i < 300; ++i) {
                const Problem p = make_problem(task, d, rng);
                const std::string want = reasoning_check::solve(p);
                EXPECT(p.answer == want, "%s d=%d: %s -> answer %s, solver says %s", task.c_str(), d,
                       p.question.c_str(), p.answer.c_str(), want.c_str());
                reasoning_check::check_no_zeros(p);
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
    test_planted_mistakes();
}
