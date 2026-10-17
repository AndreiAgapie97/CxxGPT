#pragma once

#include "../src/generate.h"
#include "test_util.h"

// An exact reference for the calculator that never uses fractions: the
// value n / 10^k, with n in 64 bits. fits turns false once a step does not
// fit (n beyond 64 bits, or k above 18). That says nothing about the
// calculator, whose reduced fractions can be much smaller (1.0 * 1.0 is
// 100 / 10^2 here but 1/1 there); such cases are skipped. When the
// reference fits, every numerator and denominator the calculator meets is
// at most the reference's, so the calculator must give the same answer.
// Like the calculator, it never keeps INT64_MIN. No 128-bit arithmetic, as
// the tests also build with clang-cl on Windows.
struct CalcDecimal {
    int64_t n = 0;
    int k = 0;
    bool fits = true;
};

inline CalcDecimal calc_no_fit() { return CalcDecimal{0, 0, false}; }

inline CalcDecimal calc_scaled(CalcDecimal x, int k) {  // the same value with k >= x.k decimals
    if (!x.fits || k > 18) return calc_no_fit();
    int64_t n = x.n;
    for (int i = x.k; i < k; ++i)
        if (__builtin_mul_overflow(n, int64_t(10), &n)) return calc_no_fit();
    return CalcDecimal{n, k, true};
}

inline CalcDecimal calc_negate(CalcDecimal x) { return CalcDecimal{-x.n, x.k, x.fits}; }

inline CalcDecimal calc_add(CalcDecimal a, CalcDecimal b) {
    const int k = std::max(a.k, b.k);
    a = calc_scaled(a, k);
    b = calc_scaled(b, k);
    int64_t n = 0;
    if (!a.fits || !b.fits || __builtin_add_overflow(a.n, b.n, &n) || n == std::numeric_limits<int64_t>::min())
        return calc_no_fit();
    return CalcDecimal{n, k, true};
}

inline CalcDecimal calc_times(CalcDecimal a, CalcDecimal b) {
    int64_t n = 0;
    if (!a.fits || !b.fits || a.k + b.k > 18 || __builtin_mul_overflow(a.n, b.n, &n) ||
        n == std::numeric_limits<int64_t>::min())
        return calc_no_fit();
    return CalcDecimal{n, a.k + b.k, true};
}

// x / d for a d made only of 2s and 5s: find the smallest power of 10 that
// d divides, 10^m, and multiply by 10^m / d with m more decimals.
inline CalcDecimal calc_divide(CalcDecimal x, int64_t d) {
    int64_t p = 1;
    int m = 0;
    while (p % d != 0) p *= 10, ++m;
    return calc_times(x, CalcDecimal{p / d, m, true});
}

// Like the calculator's output: no trailing decimal zeros, no "." alone.
inline std::string calc_format(CalcDecimal x) {
    int64_t p = 1;
    for (int i = 0; i < x.k; ++i) p *= 10;
    const uint64_t a = uint64_t(x.n < 0 ? -x.n : x.n);
    std::string out = (x.n < 0 ? "-" : "") + std::to_string(a / uint64_t(p));
    if (x.k > 0) {
        std::string digits = std::to_string(a % uint64_t(p));
        digits = std::string(size_t(x.k) - digits.size(), '0') + digits;  // leading zeros, as in 0.05
        while (!digits.empty() && digits.back() == '0') digits.pop_back();
        if (!digits.empty()) out += "." + digits;
    }
    return out;
}

inline void test_calc() {
    const std::pair<const char*, const char*> exact[] = {
        {"\r1\t+2\n", "3"}, {"\f1 + 2\v", "3"}, {"48/2", "24"}, {"7/2*2", "7"}, {"1/3*3", "1"}, {"0.1+0.2", "0.3"},
        {"-(2+3)*4/8", "-2.5"}, {"12.5000", "12.5"}, {"0/-2", "0"},
        {"3-2-1", "0"}, {"2+3*4", "14"}, {".5 + 1.", "1.5"},
        {"9223372036854775807/9223372036854775807", "1"}, {"1/8", "0.125"},
        {"1/1000000000000000000", "0.000000000000000001"},
        // Digit-grouping commas, strict groups only (calc::ungroup).
        {"1,000", "1000"}, {"24*600,000", "14400000"}, {"100,000*30", "3000000"}, {"24,600,000", "24600000"},
        {"1,234.5*2", "2469"}, {"-1,000+1", "-999"}, {"(1,000)/8", "125"}, {"1,000,000.25", "1000000.25"},
        {"999,999,999,999,999,999", "999999999999999999"}
    };
    for (const auto& [expr, expected] : exact)
        EXPECT(calc::evaluate(expr) == std::optional<std::string>(expected), "calculator exact: %s", expr);
    for (const char* expr : {"", " ", "1/0", "1/3", "2^3", "sqrt(4)", "2x3", "2(3)",
                            "1e3", "1..2", "(2+3", "2+3)", "2+", "1=1", "nan", "inf", "a+2",
                            "9223372036854775808", "9223372036854775807+1", "1/0.0000000000000000001",
                            // commas that are not strict groups stay, and are refused
                            "1,2345", "1,2", "2,3+4", "1234,567", "1,234,56", "1,000,5", "1.5,000", ",000",
                            "1,000,", "0,125", "01,000", "1,0000", "1 ,000", "1, 000", "1,,000",
                            "9,223,372,036,854,775,808"})
        EXPECT(!calc::evaluate(expr), "calculator must reject: %s", expr);
    EXPECT(!calc::evaluate(std::string(129, '(') + "1" + std::string(129, ')')), "calculator depth bound");
    EXPECT(!calc::evaluate(std::string(4097, '1')), "calculator length bound");
    EXPECT(!calc::evaluate("1,000" + std::string(4092, ' ')), "the length bound is on the text before its commas go");
    EXPECT(calc::evaluate("1,000" + std::string(4091, ' ')) == std::optional<std::string>("1000"), "4096 bytes with commas");

    // Random expression trees against an exact reference that never uses
    // fractions (see CalcDecimal). The calculator's answer must be exactly
    // the reference's string, rejected when some value in the tree is far
    // too big for 64 bits, and rejected for a repeating decimal.
    Rng rng(9182);
    struct Node {
        std::string text;
        CalcDecimal exact;
        double value = 0;   // only to see when a value is far beyond 64 bits
        bool huge = false;  // some value in this tree is above 1.1e19 > 2^63
    };
    // only_times: every operation multiplies, to reach overflow often.
    std::function<Node(int, bool)> tree = [&](int depth, bool only_times) -> Node {
        if (!depth) {
            // A leaf: tenths with 1 to 7 digits, from 0.0 to 999999.9.
            const int digits = 1 + int(rng.below(7));
            int64_t n = 0;
            for (int i = 0; i < digits; ++i) n = n * 10 + int64_t(rng.below(10));
            return {std::to_string(n / 10) + "." + std::to_string(n % 10), CalcDecimal{n, 1, true}, double(n) / 10, false};
        }
        const Node a = tree(depth - 1, only_times);
        Node b = tree(depth - 1, only_times);
        const int op = only_times ? 2 : int(rng.below(4));
        if (op == 3) {
            // Inner divisors are made of 2s and 5s, so every value stays a
            // finite decimal.
            const int64_t ds[] = {1, 2, 4, 5, 8, 10, 16, 20, 25, 40, 125};
            const int64_t d = ds[rng.below(11)];
            b = {std::to_string(d), CalcDecimal{d, 0, true}, double(d), false};
        }
        Node r;
        r.text = "(" + a.text + "+-*/"[op] + b.text + ")";
        r.exact = op == 0 ? calc_add(a.exact, b.exact) : op == 1 ? calc_add(a.exact, calc_negate(b.exact)) :
                  op == 2 ? calc_times(a.exact, b.exact) : calc_divide(a.exact, b.exact.n);
        r.value = op == 0 ? a.value + b.value : op == 1 ? a.value - b.value : op == 2 ? a.value * b.value :
                  a.value / b.value;
        r.huge = a.huge || b.huge || std::abs(r.value) > 1.1e19;
        return r;
    };
    size_t exact_cases = 0, after_odd = 0, repeating = 0, overflow = 0, skipped = 0;
    for (int i = 0; i < 4500; ++i) {
        const bool only_times = i >= 4000;
        const Node t = only_times ? tree(3, true) : tree(1 + int(rng.below(4)), false);
        std::string expr = t.text;
        CalcDecimal ref = t.exact;
        bool odd_divisor = false, repeats = false;
        if (!only_times && rng.below(10) < 3) {
            // Divide the whole tree by a number with a prime factor other
            // than 2 and 5: q = odd * two_five, with odd coprime to 10. The
            // answer is a finite decimal only if odd divides the numerator.
            const int64_t qs[] = {3, 6, 7, 9, 11, 12, 13, 15, 21};
            const int64_t q = qs[rng.below(9)];
            expr += "/" + std::to_string(q);
            int64_t odd = q, two_five = 1;
            while (odd % 2 == 0) odd /= 2, two_five *= 2;
            while (odd % 5 == 0) odd /= 5, two_five *= 5;
            odd_divisor = true;
            if (ref.fits && ref.n % odd != 0) repeats = true;
            else if (ref.fits) ref = calc_divide(CalcDecimal{ref.n / odd, ref.k, true}, two_five);
        }
        const auto result = calc::evaluate(expr);
        if (t.huge) {
            ++overflow;
            EXPECT(!result, "calculator must reject overflow: %s", expr.c_str());
        } else if (!ref.fits) {
            ++skipped;  // the unreduced reference overflowed: it cannot judge
        } else if (repeats) {
            ++repeating;
            EXPECT(!result, "calculator must reject a repeating decimal: %s", expr.c_str());
        } else {
            ++(odd_divisor ? after_odd : exact_cases);
            EXPECT(result == std::optional<std::string>(calc_format(ref)), "calculator exact reference: %s = %s, got %s",
                   expr.c_str(), calc_format(ref).c_str(), result ? result->c_str() : "nothing");
        }
    }
    // Each kind of case must actually occur (expected counts are several
    // times these bounds), so a broken reference cannot skip everything.
    EXPECT(exact_cases >= 500 && after_odd >= 20 && repeating >= 50 && overflow >= 50,
           "random calculator cases: %zu exact, %zu finite after an odd divisor, %zu repeating, %zu overflow, %zu skipped",
           exact_cases, after_odd, repeating, overflow, skipped);

    // The overflow boundary itself, with an exact oracle: products and sums
    // of two integers right around INT64_MAX, with either sign. The
    // calculator never keeps INT64_MIN, so it counts as overflow here too.
    constexpr int64_t limit = std::numeric_limits<int64_t>::max();
    size_t fit = 0, over = 0;
    for (int i = 0; i < 1000; ++i) {
        const bool product = i % 2 == 0, negative = rng.below(2) == 1;
        const int64_t a = product ? 2 + int64_t(rng.below(4000000000u)) : (int64_t(1) << 61) + int64_t(rng.next() >> 2);
        const int64_t b = (product ? limit / a : limit - a) + int64_t(rng.below(3)) - 1;
        const int64_t sa = negative ? -a : a, sb = !product && negative ? -b : b;
        int64_t value = 0;
        const bool overflows = (product ? __builtin_mul_overflow(sa, b, &value) : __builtin_add_overflow(sa, sb, &value)) ||
                               value == std::numeric_limits<int64_t>::min();
        const std::string expr = std::string(negative ? "-" : "") + std::to_string(a) +
                                 (product ? "*" : negative ? "-" : "+") + std::to_string(b);
        const auto result = calc::evaluate(expr);
        ++(overflows ? over : fit);
        if (overflows) EXPECT(!result, "calculator boundary overflow: %s", expr.c_str());
        else EXPECT(result == std::optional<std::string>(std::to_string(value)), "calculator boundary: %s", expr.c_str());
    }
    EXPECT(fit >= 200 && over >= 200, "calculator boundary cases: %zu fit, %zu overflow", fit, over);

    for (const char* expr : {"16 x 7 = ", "16 * 7 = ", "16 × 7 = ", "She has 16 x 7 = ", "Then: 16 * 7 = ", "next; 16 * 7 = ",
                             "16 x 7 = $", "$16 x 7 = $"})
        EXPECT(calc::response(expr, calc::Mode::plain) == std::optional<std::string>("112"), "plain multiplication: %s", expr);
    EXPECT(calc::response("-3 * 2 = ", calc::Mode::plain) == std::optional<std::string>("-6"), "unary plain expression");
    for (const char* expr : {"2^3 + 4 = ", "2 ^ 3 + 4 = ", "2 @ 3 + 4 = ", "2 & 3 + 4 = ", "2 X 3 + 4 = ",
                            "2 \\times 3 + 4 = ", "2 times 3 + 4 = ", "a + 3 = ", "a -3*2 = ", "sqrt(4)+3 = ", "1,000+3 = ", "5/0 = "}) {
        const auto r = calc::response(expr, calc::Mode::plain);
        EXPECT(r && r->empty(), "malformed plain expression must not be partially computed: %s", expr);
    }
    // ". , ; ! ?" followed by whitespace end a clause; the expression after
    // them is computed, not rejected. Without the whitespace they are not.
    const std::pair<const char*, const char*> clauses[] = {
        {"He has 3 bags. 16 * 7 = ", "112"}, {"So, 16 * 7 = ", "112"}, {"8, and 8 * 2 = ", "16"},
        {"It costs 4; 2 * 3 = ", "6"}, {"Really! 2 + 3 = ", "5"}, {"Why? 2 + 3 = ", "5"},
        {"He has 3 bags. -3 * 2 = ", "-6"}, {"So, he pays 2 * 3 = ", "6"}, {"3.5 * 2 = ", "7"}};
    for (const auto& [expr, expected] : clauses)
        EXPECT(calc::response(expr, calc::Mode::plain) == std::optional<std::string>(expected), "clause boundary: %s", expr);
    // Plain mode keeps its rule: a comma ends the expression, so a grouped
    // number is refused there (the calculator never sees the comma).
    for (const char* expr : {"1,000 * 2 = ", "24 * 600,000 = ", "8,and 8 * 2 = ", "3 + 4.5 * 2,3 * 2 = "}) {
        const auto r = calc::response(expr, calc::Mode::plain);
        EXPECT(r && r->empty(), "punctuation without a space is no boundary: %s", expr);
    }

    // Algebra is not arithmetic: an "x" next to a number, or a sign right
    // after an operator, must never give a result (3x + 2 once gave 6).
    for (const char* expr : {"If 3x + 2 = ", "2x - 4 = ", "3x = ", "4 + 2x = ", "x + 5 = ", "3 * x + 2 = ",
                             "3 x -2 = ", "2 x3 + 4 = ", "3 * -2 = ", "5 - -3 = ", "2 * + 3 = "}) {
        const auto r = calc::response(expr, calc::Mode::plain);
        EXPECT(!r || r->empty(), "algebra must not fire: %s", expr);
    }
    EXPECT(calc::response("3 x 4 = ", calc::Mode::plain) == std::optional<std::string>("12"), "spaced x between numbers");
    EXPECT(calc::response("2 x 3 x 4 = ", calc::Mode::plain) == std::optional<std::string>("24"), "two x signs");
    EXPECT(calc::response("3 * (-2) = ", calc::Mode::plain) == std::optional<std::string>("-6"), "sign in brackets");
    EXPECT(!calc::response("The answer = ", calc::Mode::plain), "prose is not a call");
    EXPECT(!calc::response("16 * 7 =", calc::Mode::plain), "plain waits for trailing space");
    EXPECT(calc::response("<<48/2=", calc::Mode::notes) == std::optional<std::string>("24>>"), "notes result");
    EXPECT(!calc::response("<<2=2>> then =", calc::Mode::notes), "closed note ignored");
    const auto open_note = calc::response("<<5/0=", calc::Mode::notes);
    EXPECT(open_note && open_note->empty(), "an open note is malformed at its first =");
    EXPECT(!calc::response("<<5/0=unknown =", calc::Mode::notes), "an open note is not counted again at a later =");
    EXPECT(!calc::response("<<5/0=unknown, so 2+2=", calc::Mode::notes), "nor at a later = with arithmetic");
    // A note may copy the question's numbers with their grouping commas.
    std::string worked;
    EXPECT(calc::response("<<24*600,000=", calc::Mode::notes, &worked) == std::optional<std::string>("14400000>>") &&
               worked == "24*600000", "a note with grouping commas, worked out without them");
    EXPECT(calc::response("so <<100,000*30=", calc::Mode::notes) == std::optional<std::string>("3000000>>"), "grouped note");
    for (const char* expr : {"<<1,2345*2=", "<<2,3+4=", "<<1.5,000*2=", "<<0,125*8="}) {
        const auto r = calc::response(expr, calc::Mode::notes);
        EXPECT(r && r->empty(), "a comma that is not a strict group is malformed: %s", expr);
    }
}

// At each decode position the fake predicts the next scripted token. Poison
// predictions at injected positions prove they come from the tool instead.
// Every token fed back is checked, including result tokens before continuation.
struct CalcFakeModel {
    Config cfg;
    std::vector<std::vector<int>> script, prediction, received;
    std::vector<float> logits;
    void decode_begin(int batch, const std::vector<int>&) {
        received.assign(batch, {}); logits.resize(size_t(batch) * cfg.vocab);
    }
    const float* decode_step(const int* tokens, int pos) {
        std::fill(logits.begin(), logits.end(), -100.0f);
        for (size_t b = 0; b < script.size(); ++b) {
            if (pos) received[b].push_back(tokens[b]);
            const int next = size_t(pos) < prediction[b].size() ? prediction[b][pos] : kBos;
            logits[b * cfg.vocab + next] = 100.0f;
        }
        return logits.data();
    }
};

inline void test_calc_hook() {
    const Tokenizer tok;
    for (const auto mode : {calc::Mode::notes, calc::Mode::plain}) {
        const bool notes = mode == calc::Mode::notes;
        const std::vector<std::string> expected = notes ?
            std::vector<std::string>{"<<48/2=24>> continue\n", "bad <<5/0=unknown = 1; <<7/2*2=7>> ok\n"} :
            std::vector<std::string>{"16 x 7 = 112 continue\n", "bad 5/0 = unknown; 13 * 13 = 169 ok\n"};
        CalcFakeModel m; m.cfg.vocab = 256; m.cfg.seq_len = 180;
        for (const auto& text : expected) m.script.push_back(tok.encode(text));
        m.prediction = m.script;
        const std::vector<std::string> insertions = notes ? std::vector<std::string>{"24>>", "7>>"} : std::vector<std::string>{"112", "169"};
        for (size_t b = 0; b < expected.size(); ++b) {
            const size_t begin = expected[b].find(insertions[b]);
            for (size_t i = begin; i < begin + insertions[b].size(); ++i) m.prediction[b][i] = '?';
        }
        calc::Counts calls;
        Rng actual(42), reference(42);
        const auto out = generate(m, {{kBos},{kBos}}, 120, [](const auto& ids) { return ids.back() == '\n'; },
                                  1, &actual, nullptr, mode, &tok, &calls);
        EXPECT(calls.fired == 2 && calls.malformed == 1, "independent batch calculator counts");
        size_t draws = 0;
        for (size_t b = 0; b < expected.size(); ++b) draws += expected[b].size() - insertions[b].size();
        for (size_t i = 0; i < draws; ++i) reference.uniform();
        for (int i = 0; i < 4; ++i)
            EXPECT(actual.state()[i] == reference.state()[i], "injected/finished tokens use no random draws");
        for (size_t b = 0; b < out.size(); ++b) {
            EXPECT(tok.decode(out[b]) == expected[b], "injected results and continuation, slot %zu", b);
            EXPECT(std::equal(out[b].begin(), out[b].end() - 1, m.received[b].begin()), "results fed back into cache, slot %zu", b);
        }
    }
    CalcFakeModel m; m.cfg.vocab = 256; m.cfg.seq_len = 30;
    m.script = {tok.encode("<<48/2=wrong")}; m.prediction = m.script;
    calc::Counts calls;
    const auto capped = generate(m, {{kBos}}, 8, [](const auto&) { return false; }, 0, nullptr, nullptr, calc::Mode::notes, &tok, &calls);
    EXPECT(tok.decode(capped[0]) == "<<48/2=w" && calls.fired == 0, "budget never inserts partial result");
    m.cfg.seq_len = 8;
    const auto context = generate(m, {{kBos}}, 100, [](const auto&) { return false; }, 0, nullptr, nullptr, calc::Mode::notes, &tok, &calls);
    EXPECT(context[0] == capped[0], "context cap includes result tokens");
    m.cfg.seq_len = 30;
    const auto disabled = generate(m, {{kBos}}, 12, [](const auto&) { return false; });
    EXPECT(tok.decode(disabled[0]) == "<<48/2=wrong", "default generation unchanged");

    // The prompt is watched too: "16 x 7 =" at its end fires on the model's
    // first token, " ". (The prompt's BOS starts the watched text afresh;
    // kept as a byte, it would make the call malformed.)
    {
        std::vector<int> prompt = tok.encode("16 x 7 =");
        prompt.insert(prompt.begin(), kBos);
        const size_t L = prompt.size();
        // Predictions start at position L - 1, after the prompt.
        std::vector<int> wanted(L - 1, '?');
        for (int t : tok.encode(" ??? ok\n")) wanted.push_back(t);
        CalcFakeModel p; p.cfg.vocab = 256; p.cfg.seq_len = 60;
        p.script = {wanted}; p.prediction = p.script;
        calc::Counts prompt_calls;
        const auto out = generate(p, {prompt}, 30, [](const auto& ids) { return ids.back() == '\n'; },
                                  0, nullptr, nullptr, calc::Mode::plain, &tok, &prompt_calls);
        EXPECT(tok.decode(out[0]) == " 112 ok\n" && prompt_calls.fired == 1 && prompt_calls.malformed == 0,
               "a trigger at the end of the prompt fires, got \"%s\"", tok.decode(out[0]).c_str());
        EXPECT(std::equal(out[0].begin(), out[0].end() - 1, p.received[0].begin() + ptrdiff_t(L - 1)),
               "prompt-triggered result fed back");
    }

    // A merged BPE token can hold the trigger and the model's own start of
    // the answer (the tokenizer joins a space to the punctuation after it,
    // so "= $150" arrives as " =", " $"). The calculator keeps `kept`, the
    // start of the token, and puts its result in place of the rest: nothing
    // of the rest comes back, so "= $150" never becomes "= 150$150".
    struct Merged { const char* merge; calc::Mode mode; const char* before; const char* kept; const char* result; };
    const Merged merges[] = {
        {"=$", calc::Mode::notes, "<<48/2", "=", "24>>"},
        {"=-", calc::Mode::notes, "<<3-5", "=", "-2>>"},
        {"=.", calc::Mode::notes, "<<1/2", "=", "0.5>>"},
        {" $", calc::Mode::plain, "16 x 7 =", " $", "112"},
        {" -", calc::Mode::plain, "3 - 5 =", " ", "-2"},
    };
    for (const Merged& c : merges) {
        // "a" between copies keeps each copy its own chunk, so the only
        // pair to merge is the one we want.
        std::string corpus;
        for (int i = 0; i < 8; ++i) corpus += std::string("a") + c.merge;
        Tokenizer bpe;
        bpe.train(corpus, 257);
        EXPECT(bpe.encode(c.merge).size() == 1, "BPE fixture merges \"%s\"", c.merge);
        // The model writes `before`, then the merged token; poison follows
        // where the result's tokens go, then the model's continuation.
        auto replacement = bpe.encode(c.kept);
        const auto answer = bpe.encode(c.result), after = bpe.encode(" ok\n");
        replacement.insert(replacement.end(), answer.begin(), answer.end());
        std::vector<int> wanted = bpe.encode(c.before);
        wanted.push_back(bpe.encode(c.merge)[0]);
        wanted.insert(wanted.end(), replacement.size() - 1, '?');
        wanted.insert(wanted.end(), after.begin(), after.end());
        CalcFakeModel m; m.cfg.vocab = bpe.vocab_size(); m.cfg.seq_len = 60;
        m.script = {wanted}; m.prediction = m.script;
        calc::Counts calls;
        const auto out = generate(m, {{kBos}}, 50, [](const auto& ids) { return ids.back() == '\n'; },
                                  0, nullptr, nullptr, c.mode, &bpe, &calls);
        const std::string expected = std::string(c.before) + c.kept + c.result + " ok\n";
        EXPECT(bpe.decode(out[0]) == expected && calls.fired == 1 && calls.malformed == 0,
               "merged token \"%s\": the result replaces its rest, got \"%s\"", c.merge, bpe.decode(out[0]).c_str());
        EXPECT(std::equal(out[0].begin(), out[0].end() - 1, m.received[0].begin()), "merged token \"%s\" fed back", c.merge);
    }

    // With the no-repeat block (--no-repeat 8): the second "<<48/2=" is the
    // model's own, and new as a run of 8 ("B<<48/2=" after "A<<48/2=").
    // The calculator's "24>>" then repeats runs of 8 but is never blocked;
    // the model's next token, '.', would repeat "/2=24>>." and is: the
    // fake's only other choice is token 0, so 0 comes instead.
    {
        const std::string text = "A<<48/2=24>>.B<<48/2=24>>.";
        CalcFakeModel b; b.cfg.vocab = 256; b.cfg.seq_len = 60;
        std::vector<int> wanted = tok.encode(text);
        for (size_t at = text.find("24>>"); at != std::string::npos; at = text.find("24>>", at + 1))
            for (size_t i = at; i < at + 4; ++i) wanted[i] = '?';
        b.script = {wanted}; b.prediction = b.script;
        calc::Counts block_calls;
        const auto out = generate(b, {{kBos}}, int(text.size()), [](const auto&) { return false; }, 0, nullptr,
                                  nullptr, calc::Mode::notes, &tok, &block_calls, 8);
        std::vector<int> expected = tok.encode(text);
        expected.back() = 0;
        EXPECT(out[0] == expected && block_calls.fired == 2,
               "calculator results pass the block, the model's next token does not: got \"%s\"",
               tok.decode(out[0]).c_str());
    }
}
