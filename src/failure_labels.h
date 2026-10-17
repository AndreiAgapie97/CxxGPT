// failure_labels.h — why an exam answer is wrong (issue #16).
//
// An exact-answer score says how many answers are right, not what goes
// wrong in the others. Each first try gets one label, decided mechanically
// so that anyone can check it:
//   correct     the answer matches;
//   unfinished  no "#### answer" line before the token budget ran out
//               (usually a loop: see the flags);
//   format      the answer line holds the right number, written another way
//               ("$18", "18 dollars");
//   arithmetic  a written calculation is wrong ("480 * 480 = 112,400"),
//               and it comes before any mention of the right answer;
//   passed_gold the right answer appears in the work before anything goes
//               wrong in it, and the final answer is another number: the
//               model computed what was asked and kept going (or answered
//               another quantity);
//   other       wrong, with no wrong calculation and no right number in the
//               work: a wrong plan or a wrong goal, for a person to tell.
// Calculations are checked with the exact calculator (calc.h), in both of
// its forms: "<<48/2=24>>" notes and plain "16 x 7 = 112", on the work with
// "$" signs and digit-grouping commas dropped ("1,200 x $50" reads as
// "1200 x 50"; the calculator takes neither). An "=" after a bare number
// ("half of 10 = 5") is not a calculation. A chain "6 * 4 + 2 = 24 + 2 = 26"
// is checked link by link: "6 * 4 + 2 = 24 + 2", then "24 + 2 = 26" (a
// link is the text up to the next " = " on the line, when all of it is
// arithmetic). Every check is kept, so a label can be audited from its own
// evidence.
//
// Flags, not labels (they also fire on right answers, so they are reported
// for right and wrong answers separately):
//   loop         a line of 12 or more characters written twice;
//   repeats_run  the answer repeats a run of N tokens (NoRepeat's
//                definition, as --no-repeat N and --unlikelihood use it), on
//                the answer's text encoded again;
//   outside      numbers that are neither in the question (digits or number
//                words), nor a result written after "=", nor one +, -, * or
//                / away from numbers already seen or the constants 1, 2 and
//                100: knowledge brought in from outside ("30 days in a
//                week", but also a right "7 days in a week").
// Numbers are compared as exact fractions, without their sign.
#pragma once

#include <algorithm>
#include <cstdio>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "calc.h"
#include "no_repeat.h"
#include "reasoning.h"

namespace labels {

// The run length for repeats_run, as --no-repeat 8 would block.
inline constexpr int kRunTokens = 8;

enum class Label { correct, unfinished, format, arithmetic, passed_gold, other };
inline constexpr Label kAllLabels[] = {Label::correct,    Label::unfinished,  Label::format,
                                       Label::arithmetic, Label::passed_gold, Label::other};

inline const char* label_name(Label l) {
    switch (l) {
        case Label::correct: return "correct";
        case Label::unfinished: return "unfinished";
        case Label::format: return "format";
        case Label::arithmetic: return "arithmetic";
        case Label::passed_gold: return "passed_gold";
        case Label::other: return "other";
    }
    return "?";
}

// One written calculation the calculator could read: where its "=" is, the
// text it sits in, what the model wrote after the "=" (a number, or the
// next link of a chain: "24 + 2") and the exact value. `negative`: what was
// written is below zero ("-3", or "-5 + 2"); ok ignores signs.
struct Calculation {
    size_t at = 0;
    std::string text, wrote, exact;
    bool ok = false, negative = false;
};

struct AnswerLabel {
    Label label = Label::other;
    std::string answer;               // normalized as scoring does
    std::vector<Calculation> calcs;   // every calculation that was checked
    size_t unreadable = 0;            // "=" the calculator could not check (no number written, 10/3, ...)
    bool loop = false, repeats_run = false;
    bool gold_in_work = false;        // the right number appears before the answer line
    std::vector<std::string> outside;  // numbers from outside the question, in order

    size_t wrong_calcs() const {
        return size_t(std::count_if(calcs.begin(), calcs.end(), [](const Calculation& c) { return !c.ok; }));
    }
};

namespace detail {

inline bool digit(char c) { return c >= '0' && c <= '9'; }

inline bool same(const calc::Rational& a, const calc::Rational& b) { return a.num == b.num && a.den == b.den; }

inline calc::Rational magnitude(calc::Rational r) {
    if (r.num < 0) r.num = -r.num;
    return r;
}

inline std::optional<calc::Rational> value_of(std::string_view s) {
    const auto r = calc::Parser(s).parse();
    if (!r) return {};
    return magnitude(*r);
}

// A number in the text: digits, digit-grouping commas, one decimal point.
struct Number {
    size_t at = 0, end = 0;
    std::string written;  // without commas
    calc::Rational value;
};

inline std::vector<Number> numbers_in(std::string_view s) {
    std::vector<Number> out;
    size_t i = 0;
    while (i < s.size()) {
        if (!digit(s[i]) || (i > 0 && (digit(s[i - 1]) || s[i - 1] == '.'))) {
            ++i;
            continue;
        }
        Number n;
        n.at = i;
        while (i < s.size() && digit(s[i])) n.written += s[i++];
        // ",ddd" groups, but not ",dddd" (a list "1,2345" is not one number).
        while (i + 3 < s.size() && s[i] == ',' && digit(s[i + 1]) && digit(s[i + 2]) && digit(s[i + 3]) &&
               (i + 4 >= s.size() || !digit(s[i + 4]))) {
            n.written.append(s.substr(i + 1, 3));
            i += 4;
        }
        if (i + 1 < s.size() && s[i] == '.' && digit(s[i + 1])) {
            n.written += s[i++];
            while (i < s.size() && digit(s[i])) n.written += s[i++];
        }
        n.end = i;
        const auto v = value_of(n.written);
        if (v) {
            n.value = *v;
            out.push_back(n);
        }
    }
    return out;
}

// Numbers a question gives in words. "half" and "twice" give 2, "dozen"
// 12; ordinary words count only as whole words.
inline std::vector<calc::Rational> number_words(std::string_view s) {
    static const std::pair<const char*, int64_t> words[] = {
        {"one", 1},        {"two", 2},        {"three", 3},     {"four", 4},     {"five", 5},
        {"six", 6},        {"seven", 7},      {"eight", 8},     {"nine", 9},     {"ten", 10},
        {"eleven", 11},    {"twelve", 12},    {"thirteen", 13}, {"fourteen", 14}, {"fifteen", 15},
        {"sixteen", 16},   {"seventeen", 17}, {"eighteen", 18}, {"nineteen", 19}, {"twenty", 20},
        {"thirty", 30},    {"forty", 40},     {"fifty", 50},    {"sixty", 60},   {"seventy", 70},
        {"eighty", 80},    {"ninety", 90},    {"hundred", 100}, {"thousand", 1000}, {"million", 1000000},
        {"dozen", 12},     {"half", 2},       {"twice", 2},     {"double", 2},   {"triple", 3},
        {"third", 3},      {"quarter", 4},    {"once", 1}};
    std::vector<calc::Rational> out;
    auto letter = [](char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); };
    size_t i = 0;
    while (i < s.size()) {
        if (!letter(s[i])) {
            ++i;
            continue;
        }
        size_t j = i;
        std::string w;
        while (j < s.size() && letter(s[j])) w += char(s[j] >= 'A' && s[j] <= 'Z' ? s[j] - 'A' + 'a' : s[j]), ++j;
        // "fourths", "thirds", "halves", "dozens": the plural counts too.
        for (const auto& [word, v] : words) {
            const std::string_view ww(w), base(word);
            if (ww == base || (ww.size() == base.size() + 1 && ww.starts_with(base) && ww.back() == 's') ||
                (base == "half" && ww == "halves"))
                out.push_back({v, 1});
        }
        i = j;
    }
    return out;
}

inline bool contains(const std::vector<calc::Rational>& set, const calc::Rational& v) {
    return std::any_of(set.begin(), set.end(), [&](const calc::Rational& x) { return same(x, v); });
}

// v is a + b, a - b, a * b or a / b for some a, b in `seen`.
inline bool one_step(const std::vector<calc::Rational>& seen, const calc::Rational& v) {
    for (const calc::Rational& a : seen)
        for (const calc::Rational& b : seen) {
            for (const auto& r : {calc::plus(a, b), calc::plus(a, calc::Rational{-b.num, b.den}), calc::times(a, b),
                                  b.num != 0 ? calc::divide(a, b) : std::optional<calc::Rational>()})
                if (r && same(magnitude(*r), v)) return true;
        }
    return false;
}

// The text around a calculation, for the record: from the start of its line
// (at most 80 characters back) to the end of what was written after "=".
inline std::string around(std::string_view s, size_t eq, size_t end) {
    size_t begin = s.rfind('\n', eq);
    begin = begin == std::string_view::npos ? 0 : begin + 1;
    if (eq - begin > 80) begin = eq - 80;
    return std::string(s.substr(begin, end - begin));
}

inline std::string_view trim_blanks(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.remove_suffix(1);
    return s;
}

// `s` as one whole calculation, ready for the calculator ("6 x 4 + 2" gives
// "6 * 4 + 2"): only arithmetic, all of it, with an operator after its first
// character. A bare number ("24", "-7") is not a calculation, and neither is
// "9 eggs. 9 * 2", though the calculator would read "9 * 2" in it.
inline std::optional<std::string> whole_calculation(std::string_view s) {
    s = trim_blanks(s);
    std::string expr;
    for (size_t k = 0; k < s.size(); ++k) {
        const char c = s[k];
        if (s.substr(k).starts_with("\xc3\x97")) {  // "×"
            expr += '*';
            ++k;
        } else if (c == 'x' && calc::times_sign(s, k)) {
            expr += '*';
        } else if (digit(c) || std::string_view(".+-*/() \t").find(c) != std::string_view::npos) {
            expr += c;
        } else {
            return {};
        }
    }
    if (expr.find_first_of("+-*/", 1) == std::string::npos || !calc::Parser(expr).parse()) return {};
    return expr;
}

// Every calculation in the work, checked: notes ("<<16-3-4=9>>") and plain
// ("16 - 3 - 4 = 9", "$50 x 3 = $150"), found as the calculator finds its
// calls, and chains ("6 * 4 + 2 = 24 + 2 = 26") link by link.
inline void check_calculations(std::string_view work, AnswerLabel& out) {
    for (size_t i = 0; i < work.size(); ++i) {
        if (work[i] != '=') continue;
        // Notes: the prefix ends at the "=".
        if (const auto r = calc::response(work.substr(0, i + 1), calc::Mode::notes)) {
            const size_t close = work.find(">>", i);
            const std::string_view wrote =
                work.substr(i + 1, close == std::string_view::npos ? 0 : close - i - 1);
            const std::string result = r->ends_with(">>") ? r->substr(0, r->size() - 2) : *r;
            const auto got = calc::Parser(wrote).parse();
            const auto exact = result.empty() ? std::nullopt : value_of(result);
            if (got && exact && close != std::string_view::npos) {
                out.calcs.push_back({i, around(work, i, close + 2), std::string(wrote), result,
                                     same(magnitude(*got), *exact), got->num < 0});
            } else {
                ++out.unreadable;
            }
            continue;
        }
        // Plain: the prefix ends at " = ".
        if (i + 1 >= work.size() || work[i + 1] != ' ' || i == 0 || work[i - 1] != ' ') continue;
        const size_t after = i + 2;
        if (work.substr(after).starts_with("<<")) continue;  // "48/2 = <<48/2=24>>24": the note is checked
        const size_t newline = work.rfind('\n', i), line_begin = newline == std::string_view::npos ? 0 : newline + 1;
        // The left side: what the calculator reads before " = ", or else, in
        // a chain, the link before this one ("24 + 2" in "... = 24 + 2 = 26";
        // the calculator stops at the earlier "=" and finds nothing to read).
        auto r = calc::response(work.substr(0, after), calc::Mode::plain);
        if (!r || r->empty()) {
            const size_t prev = i >= 4 ? work.rfind(" = ", i - 4) : std::string_view::npos;
            if (prev != std::string_view::npos && prev >= line_begin)
                if (const auto link = whole_calculation(work.substr(prev + 3, i - 1 - (prev + 3))))
                    r = calc::evaluate(*link).value_or("");
        }
        if (!r) continue;  // not a calculation: "x = 5", "The total = ..."
        // The left side's value (a plain Rational and a flag, not an optional:
        // g++ -O3 then warns that the optional's value may be uninitialized).
        calc::Rational exact;
        bool readable = false;
        if (!r->empty())
            if (const auto v = value_of(*r)) exact = *v, readable = true;
        // The right side: the next link of a chain, if this "=" has one (the
        // text up to the next " = " on the line, all of it arithmetic), or
        // else the number written right after the "=".
        const size_t line_end = std::min(work.find('\n', after), work.size());
        const size_t next = work.find(" = ", after);
        if (next != std::string_view::npos && next < line_end) {
            if (const auto link = whole_calculation(work.substr(after, next - after))) {
                if (!readable) {
                    ++out.unreadable;
                    continue;
                }
                const calc::Rational v = calc::Parser(*link).parse().value_or(calc::Rational{});
                const std::string wrote(trim_blanks(work.substr(after, next - after)));
                out.calcs.push_back({i, around(work, i, next), wrote, *r, same(magnitude(v), exact), v.num < 0});
                continue;
            }
        }
        const auto nums = numbers_in(work.substr(after));
        const bool written = !nums.empty() && nums[0].at == 0;
        if (!written || !readable) {
            ++out.unreadable;
            continue;
        }
        const Number& n = nums[0];
        out.calcs.push_back({i, around(work, i, after + n.end), std::string(work.substr(after, n.end)), *r,
                             same(n.value, exact), false});
    }
}

}  // namespace detail

// Labels one answer. `tokens`, if given, is the answer's text encoded, for
// the repeats_run flag with runs of `ngram` tokens.
inline AnswerLabel label_answer(const Problem& p, const std::string& completion,
                                const std::vector<int>* tokens = nullptr, int ngram = kRunTokens) {
    using namespace detail;
    AnswerLabel out;
    out.answer = strip_digit_commas(trim(extract_answer(completion)));
    const size_t hashes = completion.find("#### ");
    // The work before the answer line, without "$" and digit-grouping commas.
    std::string work = strip_digit_commas(completion.substr(0, hashes));
    work.erase(std::remove(work.begin(), work.end(), '$'), work.end());

    // Flags.
    {
        std::vector<std::string> lines;
        size_t b = 0;
        while (b <= completion.size()) {
            size_t e = completion.find('\n', b);
            if (e == std::string::npos) e = completion.size();
            const std::string line = trim(completion.substr(b, e - b));
            if (line.size() >= 12) {
                if (std::find(lines.begin(), lines.end(), line) != lines.end()) out.loop = true;
                lines.push_back(line);
            }
            b = e + 1;
        }
    }
    if (tokens) {
        NoRepeat block(ngram);
        for (int t : *tokens) {
            const auto& banned = block.blocked();
            if (std::find(banned.begin(), banned.end(), t) != banned.end()) out.repeats_run = true;
            block.add(t);
        }
    }
    check_calculations(work, out);

    // Numbers from outside the question, and where the right one first appears.
    const auto gold = value_of(p.answer);
    std::vector<calc::Rational> seen = number_words(p.question);
    for (const Number& n : numbers_in(p.question)) seen.push_back(n.value);
    const bool gold_in_question = gold && contains(seen, *gold);
    for (const int64_t c : {1, 2, 100}) seen.push_back({c, 1});
    size_t gold_at = std::string::npos;
    for (const Number& n : numbers_in(work)) {
        if (gold && !gold_in_question && gold_at == std::string::npos && same(n.value, *gold)) gold_at = n.at;
        if (contains(seen, n.value)) continue;
        // A result: right after "=" (and spaces), checked or not.
        size_t k = n.at;
        while (k > 0 && work[k - 1] == ' ') --k;
        const bool result = k > 0 && work[k - 1] == '=';
        if (!result && !one_step(seen, n.value)) out.outside.push_back(n.written);
        seen.push_back(n.value);
    }
    out.gold_in_work = gold_at != std::string::npos;

    // Finished as completion_done() (generate.h) has it, and right as scoring has it.
    const bool done = hashes != std::string::npos && completion.back() == '\n';
    if (!done) {
        out.label = Label::unfinished;  // counted in EvalResult::unfinished too
    } else if (answer_matches(completion, p.answer)) {
        out.label = Label::correct;
    } else {
        bool right_number = false;
        for (const Number& n : numbers_in(out.answer)) right_number |= gold && same(n.value, *gold);
        size_t first_wrong = std::string::npos;
        for (const Calculation& c : out.calcs)
            if (!c.ok) {
                first_wrong = c.at;
                break;
            }
        if (right_number) out.label = Label::format;
        else if (first_wrong != std::string::npos && first_wrong < gold_at) out.label = Label::arithmetic;
        else if (gold_at != std::string::npos) out.label = Label::passed_gold;
        else out.label = Label::other;
    }
    return out;
}

// JSON string contents: quotes, backslashes and control characters escaped,
// valid UTF-8 kept, and any byte that is not valid UTF-8 (a model can write
// half a character) replaced by U+FFFD.
inline std::string json_escape(std::string_view s) {
    std::string o;
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = s[i];
        if (c == '"' || c == '\\') o += '\\', o += char(c), ++i;
        else if (c == '\n') o += "\\n", ++i;
        else if (c == '\r') o += "\\r", ++i;
        else if (c == '\t') o += "\\t", ++i;
        else if (c < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof buf, "\\u%04x", c);
            o += buf;
            ++i;
        } else if (c < 0x80) {
            o += char(c);
            ++i;
        } else {
            // A lead byte and its continuation bytes; overlong forms,
            // surrogates and values past U+10FFFF are invalid.
            const int len = c >= 0xf0 && c <= 0xf4 ? 4 : c >= 0xe0 ? 3 : c >= 0xc2 && c <= 0xdf ? 2 : 0;
            bool ok = len > 0 && c < 0xf5 && i + size_t(len) <= s.size();
            for (int k = 1; ok && k < len; ++k) ok = (static_cast<unsigned char>(s[i + size_t(k)]) & 0xc0) == 0x80;
            if (ok && len == 3) {
                const unsigned char c1 = s[i + 1];
                ok = !(c == 0xe0 && c1 < 0xa0) && !(c == 0xed && c1 >= 0xa0);
            }
            if (ok && len == 4) {
                const unsigned char c1 = s[i + 1];
                ok = !(c == 0xf0 && c1 < 0x90) && !(c == 0xf4 && c1 >= 0x90);
            }
            if (ok) {
                o.append(s.substr(i, size_t(len)));
                i += size_t(len);
            } else {
                o += "\xef\xbf\xbd";
                ++i;
            }
        }
    }
    return o;
}

inline std::string json_line(int64_t problem, const Problem& p, const AnswerLabel& l, const std::string& completion) {
    auto str = [](std::string_view s) { return "\"" + json_escape(s) + "\""; };
    auto boolean = [](bool b) { return b ? std::string("true") : std::string("false"); };
    std::string o = "{\"problem\":" + std::to_string(problem) + ",\"task\":" + str(p.task) +
                    ",\"difficulty\":" + std::to_string(p.difficulty) + ",\"label\":" + str(label_name(l.label)) +
                    ",\"gold\":" + str(p.answer) + ",\"answer\":" + str(l.answer) + ",\"loop\":" + boolean(l.loop) +
                    ",\"repeats_run\":" + boolean(l.repeats_run) + ",\"gold_in_work\":" + boolean(l.gold_in_work) +
                    ",\"outside\":[";
    for (size_t i = 0; i < l.outside.size(); ++i) o += (i ? "," : "") + str(l.outside[i]);
    o += "],\"calcs\":[";
    for (size_t i = 0; i < l.calcs.size(); ++i) {
        const Calculation& c = l.calcs[i];
        o += std::string(i ? "," : "") + "{\"text\":" + str(c.text) + ",\"wrote\":" + str(c.wrote) +
             ",\"exact\":" + str(c.exact) + ",\"ok\":" + boolean(c.ok) + "}";
    }
    o += "],\"unreadable\":" + std::to_string(l.unreadable) + ",\"question\":" + str(p.question) +
         ",\"completion\":" + str(completion) + "}";
    return o;
}

// The labels' counts, and the flags' among right and wrong answers.
struct LabelSummary {
    size_t count[std::size(kAllLabels)] = {};
    size_t total = 0, loops_unfinished = 0, runs_unfinished = 0;
    size_t right_with_wrong_calc = 0, calcs = 0, wrong_calcs = 0, unreadable = 0;
    size_t outside_right = 0, outside_wrong = 0;

    void add(const AnswerLabel& l) {
        ++count[size_t(l.label)];
        ++total;
        const bool right = l.label == Label::correct;
        loops_unfinished += l.label == Label::unfinished && l.loop;
        runs_unfinished += l.label == Label::unfinished && l.repeats_run;
        right_with_wrong_calc += right && l.wrong_calcs() > 0;
        calcs += l.calcs.size();
        wrong_calcs += l.wrong_calcs();
        unreadable += l.unreadable;
        (right ? outside_right : outside_wrong) += !l.outside.empty();
    }

    void print(int ngram) const {
        auto pct = [](size_t part, size_t whole) { return 100.0 * double(part) / double(std::max<size_t>(1, whole)); };
        std::printf("failure labels (first tries, %zu problems):\n", total);
        for (const Label l : kAllLabels)
            std::printf("  %-12s %5zu  %5.1f%%\n", label_name(l), count[size_t(l)], pct(count[size_t(l)], total));
        const size_t right = count[size_t(Label::correct)];
        std::printf("  unfinished: %zu with a line written twice, %zu repeating a run of %d tokens\n",
                    loops_unfinished, runs_unfinished, ngram);
        std::printf("  calculations: %zu checked, %zu wrong, %zu unreadable; %zu right answers hold a wrong one\n",
                    calcs, wrong_calcs, unreadable, right_with_wrong_calc);
        std::printf("  a number from outside the question: %.1f%% of right answers, %.1f%% of wrong ones\n",
                    pct(outside_right, right), pct(outside_wrong, total - right));
    }
};

// Labels every answer and writes `path`: a first line with the run's
// settings (`settings`, a JSON object), then one line per problem, with
// ids[i] as problem i's number. encode(text) gives an answer's tokens.
template <class Encode>
LabelSummary write_labels(const std::string& path, const std::string& settings, const std::vector<Problem>& problems,
                          const std::vector<std::string>& completions, const std::vector<size_t>& ids, Encode encode) {
    CHECK(completions.size() == problems.size() && ids.size() == problems.size(),
          "%zu answers and %zu ids for %zu problems", completions.size(), ids.size(), problems.size());
    FILE* f = create_file(path, "wb");
    CHECK(f, "cannot write %s", path.c_str());
    std::string out = settings + "\n";
    LabelSummary s;
    for (size_t i = 0; i < problems.size(); ++i) {
        const std::vector<int> tokens = encode(completions[i]);
        const AnswerLabel l = label_answer(problems[i], completions[i], &tokens);
        s.add(l);
        out += json_line(int64_t(ids[i]), problems[i], l, completions[i]) + "\n";
    }
    const bool ok = std::fwrite(out.data(), 1, out.size(), f) == out.size();
    CHECK(std::fclose(f) == 0 && ok, "cannot write %s", path.c_str());
    return s;
}

}  // namespace labels
