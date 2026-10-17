// redraw_qa.h — human-written word problems (GSM8K train) written again with
// other numbers ("cxxgpt redraw-qa"): the question's own words, new
// numbers, and an exact worked solution and answer for them
// (docs/redraw-qa.md).
//
// Why: the model learns GSM8K train's gold solutions by heart, and the
// generators' problems (wordgen.h, mathgen.h) by their templates. The same
// human question with other numbers keeps the varied wording, and its
// answer cannot be recited (in the spirit of GSM-Symbolic).
//
// How, for one problem (decimals written ".5" are read as "0.5" first):
//   1. The gold solution's calculations (its "<<48/2=24>>" notes and plain
//      "16 x 7 = 112", read by failure_labels.h) are turned into formulas
//      over the question's numbers by self-train's re-draw check
//      (self_train::redraw_detail::read_calcs): "q1/2", "q1+(q1/2)".
//      A problem is kept only if every calculation is right and readable,
//      every number in a calculation has one clear source (a number of the
//      question, a percentage as its hundredth: "0.3" for "30%", an earlier
//      result, a number the question gives in words, 1/2 for "half", or a
//      usual constant: 1, 2, 100, 0.01 where it speaks of percent, and 7,
//      12, 24, 52, 60, 365, 1000 where it speaks of their units), one
//      calculation gives the gold answer, and every number of the question
//      that can change is used.
//   2. Some question numbers never change: 1 and 2, 100, numbers the
//      question also gives in words ("2" where it says "half"), units
//      ("60 minutes", "a gallon is 8 pints"), ordinals ("3rd"), times and
//      ratios ("3:30", "2:5", "8 am"), bank notes ("a $20 bill"), fractions
//      ("1/3") and years. Keeping a number is always right; it only gives
//      less variety. Question numbers with the same value are one number:
//      they change together (whichever one the solution meant, it stays
//      right).
//   3. New values are drawn for the others: about half to twice the
//      original, as round as the original ("120" gives a multiple of 10,
//      "$2.50" one of 0.50), the same decimal places, a percentage under
//      100 stays under 100, "5 days a week" at most 7 (24 hours a day, 12
//      months a year, 60 minutes an hour), never 1. Every calculation is
//      worked out again exactly (calc.h) and must keep its kind: a whole
//      number stays whole (no "2.5 people"), a positive one positive, 1
//      stays 1 and only 1 is 1 ("1 hours"), a decimal gets at most two
//      places (or its own), and it stays within 25 times the original
//      either way.
//   4. Every number in the question and the worked solution is written
//      again: the solution's numbers are mapped by value to their source (a
//      question number, a result worked out before or on its line, a
//      constant; a note's result and the number right after it are that
//      note's), which needs every such value to have one source (a result
//      equal to a question number is "text_ambiguous"), and a number tied
//      to a question number to name the same unit ("x 5 days" for the
//      weekdays is not the question's "5 hours": "units"). Each note's
//      result is the calculator's own answer for the new expression, "a 8"
//      becomes "an 8", and the final answer is the formula's value.
//   5. The variant is checked again from its text alone: every calculation
//      right by calc.h (failure_labels.h), as many as the original's; the
//      question and solution are the original's word for word once every
//      number is masked (redraw.h); the "#### " answer is the answer
//      calculation's new result; and, unless the solution uses a hundredth
//      (which it cannot read), self-train's re-draw check
//      (self_train::trace_try) accepts the original gold solution against
//      the variants: its formula gives every variant's answer.
// What no check can see: a number of the solution worked out in the head
// that has a question number's value and unit by chance, and wording that
// the new numbers make odd ("290 campers, 245 of them last week").
//
// Everything is decided from the problem's row and the seed, so the same
// seed gives the same file, whatever --limit is.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "redraw.h"
#include "self_train.h"

namespace redraw_qa {

using calc::Rational;
using labels::detail::Number;

// Why a problem gave no variant (or `converted`), in the order of the checks.
enum class Reason {
    converted,
    no_calcs,        // no calculation the calculator can read
    unreadable,      // an "=" the calculator cannot check, a digit not read as a number, a formula not followed
    wrong,           // a calculation of the gold solution is wrong, or below zero
    ambiguous,       // a number in a calculation could come from two places
    constant,        // a number in a calculation from nowhere that is no usual constant (mental arithmetic?)
    answer,          // no single calculation gives the gold answer
    unused,          // a number of the question that can change is never used by a calculation
    text_ambiguous,  // a number in the solution's text could stand for two things
    text_unsourced,  // a number in the solution's text comes from nowhere
    words,           // the solution writes a number that changes in words ("five")
    units,           // the solution writes a changing number with another unit ("5 days" for "5 hours")
    nothing,         // no number of the question can change
    no_draw,         // no draw of new numbers passed the checks
    check,           // a variant failed the final check: a bug, never expected
};
inline constexpr int kReasons = 15;

inline const char* reason_name(Reason r) {
    switch (r) {
        case Reason::converted: return "converted";
        case Reason::no_calcs: return "no_calcs";
        case Reason::unreadable: return "unreadable";
        case Reason::wrong: return "wrong";
        case Reason::ambiguous: return "ambiguous";
        case Reason::constant: return "constant";
        case Reason::answer: return "answer";
        case Reason::unused: return "unused";
        case Reason::text_ambiguous: return "text_ambiguous";
        case Reason::text_unsourced: return "text_unsourced";
        case Reason::words: return "words";
        case Reason::units: return "units";
        case Reason::nothing: return "nothing";
        case Reason::no_draw: return "no_draw";
        case Reason::check: return "check";
    }
    return "?";
}

struct Options {
    int versions = 1;  // variants per problem at most
    int draws = 300;   // draws of new numbers per variant at most
    uint64_t seed = 1;
};

namespace detail {

using self_train::redraw_detail::rational_text;
using self_train::redraw_detail::same_value;

inline bool letter(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }

inline std::string lower(std::string s) {
    for (char& c : s) c = char(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c);
    return s;
}

inline bool whole(const Rational& v) { return v.den == 1; }

// The exact decimal of v ("2.5"), or nothing for a repeating one (1/3).
inline std::optional<std::string> decimal(const Rational& v) { return calc::evaluate(rational_text(v)); }

inline size_t places(const std::string& decimal_text) {
    const size_t point = decimal_text.find('.');
    return point == std::string::npos ? 0 : decimal_text.size() - point - 1;
}

// The usual constants with the words that make them one: 7 days a week, 60
// minutes an hour, ... (self_train::kConstants). A constant from nowhere
// counts only where the problem speaks of its unit; 1, 2 and 100 always
// count (as self-train's re-draw check has them), and 0.01 where the
// problem speaks of percent ("2 * 20 * 0.01" for 20% of 2).
inline bool usual_constant(const Rational& v, const std::string& problem_lower) {
    if (v.num == 1 && v.den == 100)
        return problem_lower.find('%') != std::string::npos || problem_lower.find("percent") != std::string::npos;
    if (!whole(v)) return false;
    if (v.num == 1 || v.num == 2 || v.num == 100) return true;
    struct Unit {
        int64_t value;
        std::vector<const char*> words;
    };
    static const std::vector<Unit> units = {
        {7, {"week"}},
        {12, {"month", "year", "inch", "foot", "feet", "dozen"}},
        {24, {"hour", "day"}},
        {52, {"week"}},
        {60, {"minute", "hour", "second"}},
        {365, {"year"}},
        {1000, {"kilo", "milli", "gram", "meter", "metre", "liter", "litre", "thousand"}},
    };
    for (const Unit& u : units)
        if (u.value == v.num)
            for (const char* w : u.words)
                if (problem_lower.find(w) != std::string::npos) return true;
    return false;
}

// The line of `text` around position `at`, in lower case.
inline std::string line_at(const std::string& text, size_t at) {
    const size_t begin = at == 0 ? 0 : text.rfind('\n', at - 1), end = text.find('\n', at);
    const size_t from = begin == std::string::npos || at == 0 ? 0 : begin + 1;
    return lower(text.substr(from, (end == std::string::npos ? text.size() : end) - from));
}

// A constant from nowhere in the worked solution, on line `line`: 1, 2,
// 100 and 0.01 as usual_constant has them; a unit's number (60, 12, 24...)
// only as a conversion: the problem names both units (minutes and hours
// for 60, months and years for 12) and the line names one of them. "6
// apples a month" over "two months" makes 12, worked out in the head: no
// year is named, so it is not 12 months a year.
inline bool constant_here(const Rational& v, const std::string& problem_lower, const std::string& line) {
    if ((v.num == 1 && v.den == 100) || (whole(v) && (v.num == 1 || v.num == 2 || v.num == 100)))
        return usual_constant(v, problem_lower);
    struct Conversion {
        int64_t value;
        const char *small, *big;
    };
    static const Conversion conversions[] = {
        {7, "day", "week"},       {12, "month", "year"},    {12, "inch", "foot"},    {12, "inch", "feet"},
        {24, "hour", "day"},      {52, "week", "year"},     {60, "minute", "hour"},  {60, "second", "minute"},
        {365, "day", "year"},     {1000, "gram", "kilo"},   {1000, "meter", "kilo"}, {1000, "metre", "kilo"},
        {1000, "milli", "liter"}, {1000, "milli", "litre"}, {1000, "ml", "liter"},   {1000, "ml", "litre"},
        {1000, "gram", "kg"},     {1000, "meter", "km"}};
    auto has = [](const std::string& text, const char* word) { return text.find(word) != std::string::npos; };
    for (const Conversion& c : conversions)
        if (whole(v) && v.num == c.value && has(problem_lower, c.small) && has(problem_lower, c.big) &&
            (has(line, c.small) || has(line, c.big)))
            return true;
    return false;
}

// What a number of the question is: a number that can change (its
// variable), or one that stays (-1).
struct QuestionNumber {
    Number n;
    bool percent = false;  // "30%", "30 percent"
    size_t places = 0;     // decimal places as written
    bool commas = false;   // written with digit-grouping commas
    int variable = -1;
    std::string fixed_why;  // why it stays, for the audit ("" for a variable)
    int64_t cap = 0;        // the most it may become ("5 days a week": 7), 0 for none
};

// Up to three words right after a number ("5 red chairs", "120-page
// book"), in lower case: none if a word does not follow at once.
inline std::vector<std::string> words_after(const std::string& text, size_t end) {
    std::vector<std::string> out;
    while (out.size() < 3) {
        if (end < text.size() && (text[end] == ' ' || text[end] == '-')) ++end;
        std::string w;
        while (end < text.size() && letter(text[end])) w += text[end++];
        if (w.empty()) break;
        out.push_back(lower(w));
    }
    return out;
}

// A unit's usual name: "ft" is "feet", "guests" "people".
inline std::string unit_name(const std::string& w) {
    static const std::pair<const char*, const char*> names[] = {
        {"ft", "feet"},        {"foot", "feet"},    {"lb", "pound"},     {"lbs", "pound"},     {"hr", "hour"},
        {"hrs", "hour"},       {"min", "minute"},   {"mins", "minute"},  {"sec", "second"},    {"secs", "second"},
        {"yr", "year"},        {"yrs", "year"},     {"oz", "ounce"},     {"in", "inch"},       {"inches", "inch"},
        {"mi", "mile"},        {"km", "kilometer"}, {"kg", "kilogram"},  {"cm", "centimeter"}, {"person", "people"},
        {"persons", "people"}, {"guest", "people"}, {"guests", "people"}};
    for (const auto& [from, to] : names)
        if (w == from) return to;
    return w;
}

// Two words for the same unit: one starts the other ("day", "days"), or they
// share their first four letters ("hours", "hourly").
inline bool same_unit(std::string a, std::string b) {
    a = unit_name(a), b = unit_name(b);
    return a.starts_with(b) || b.starts_with(a) || (a.size() >= 4 && b.size() >= 4 && a.substr(0, 4) == b.substr(0, 4));
}

// The words after a solution's number and after the question's: the first
// word naming a thing on either side ("5 red cartons": "cartons") is
// another unit than the other's three words name (and neither is a word
// without a unit: "5 more"). None to compare: no conflict.
inline bool units_differ(const std::vector<std::string>& solution, const std::vector<std::string>& question);

// Words after a number that say nothing of its unit ("5 more", "5 times").
inline bool no_unit(const std::string& w) {
    static const char* const words[] = {
        "a",     "an",   "and",  "are",  "as",    "at",   "by",  "each",  "every", "for",   "from", "in",   "is",
        "it",    "less", "more", "of",   "on",    "or",   "per", "so",    "than",  "that",  "the",  "then", "this",
        "times", "to",   "was",  "were", "which", "with", "x",   "total", "left",  "fewer", "one",  "ones"};
    for (const char* x : words)
        if (w == x) return true;
    return false;
}

// Words that describe a thing without naming it ("5 red cards"): two
// numbers sharing one of these may still count different things ("5 red
// cartons").
inline bool modifier(const std::string& w) {
    static const char* const words[] = {
        "red",    "blue",  "green", "yellow", "white",   "black",     "pink",   "purple", "orange",    "brown",
        "gray",   "grey",  "gold",  "silver", "large",   "small",     "big",    "little", "huge",      "tiny",
        "new",    "old",   "extra", "full",   "whole",   "empty",     "same",   "other",  "different", "regular",
        "medium", "short", "long",  "tall",   "cheap",   "expensive", "free",   "first",  "second",    "third",
        "last",   "next",  "more",  "less",   "special", "plastic",   "wooden", "paper",  "metal",     "glass",
        "rubber", "size",  "sized", "cotton", "leather", "steel",     "stone",  "frozen", "fresh",     "dark"};
    for (const char* x : words)
        if (w == x) return true;
    return false;
}

inline bool units_differ(const std::vector<std::string>& solution, const std::vector<std::string>& question) {
    if (solution.empty() || question.empty() || no_unit(solution[0]) || no_unit(question[0])) return false;
    // The first word that names a thing on each side ("cartons" in "5 red
    // cartons"); none: nothing to compare.
    auto noun = [](const std::vector<std::string>& words) -> std::string {
        for (const std::string& w : words)
            if (!modifier(w)) return w;
        return "";
    };
    const std::string s = noun(solution), q = noun(question);
    if (s.empty() || q.empty() || no_unit(s) || no_unit(q)) return false;
    for (const std::string& w : question)
        if (!modifier(w) && same_unit(s, w)) return false;
    for (const std::string& w : solution)
        if (!modifier(w) && same_unit(q, w)) return false;
    return true;
}

// The most a number may become, from the words after it: "5 days a week"
// at most 7, "8 hours a day" or "an 8-hour day" 24, "3 months a year" 12,
// "20 minutes an hour" 60. 0 for no bound.
inline int64_t unit_cap(const std::string& q, size_t end) {
    const std::string after = lower(q.substr(end, 40));
    struct Bound {
        const char* unit;
        const char* whole;
        int64_t most;
    };
    static const Bound bounds[] = {
        {"day", "week", 7}, {"hour", "day", 24}, {"month", "year", 12}, {"minute", "hour", 60}};
    for (const Bound& b : bounds) {
        size_t at = 0;
        if (at < after.size() && (after[at] == ' ' || after[at] == '-')) ++at;
        if (after.compare(at, std::strlen(b.unit), b.unit) != 0) continue;
        at += std::strlen(b.unit);
        if (at < after.size() && after[at] == 's') ++at;
        for (const char* link :
             {" a ", " an ", " per ", " each ", " every ", " of the ", " in a ", " in an ", " ", " work "}) {
            if (after.compare(at, std::strlen(link), link) != 0) continue;
            std::string rest = after.substr(at + std::strlen(link));
            if (rest.starts_with("work ")) rest = rest.substr(5);
            if (rest.starts_with("week") && std::string(b.whole) == "day") rest = rest.substr(4);  // "every weekday"
            if (rest.starts_with(b.whole)) return b.most;
        }
    }
    return 0;
}

// Why a question number never changes (see the top of this file), or "".
inline std::string why_fixed(const std::string& q, const Number& n, const std::vector<Rational>& words,
                             const std::string& problem_lower) {
    const Rational& v = n.value;
    if (v.num == 0) return "zero";
    if (whole(v) && (v.num == 1 || v.num == 2 || v.num == 100)) return "usual";
    if (labels::detail::contains(words, v)) return "in words";
    const auto at = [&](size_t i) { return i < q.size() ? q[i] : '\0'; };
    const bool digit_after = labels::detail::digit(at(n.end + 1)),
               digit_before = n.at >= 2 && labels::detail::digit(q[n.at - 2]);
    if ((at(n.end) == ':' && digit_after) || (n.at > 0 && q[n.at - 1] == ':' && digit_before)) return "time or ratio";
    const std::string after = lower(q.substr(n.end, 5));
    if (after.starts_with(" o'c")) return "time";
    // Bank notes come in fixed values: no "$40 bills".
    const std::string note = lower(q.substr(n.end, 14));
    if ((n.at > 0 && q[n.at - 1] == '$' && note.starts_with(" bill")) || note.starts_with(" dollar bill") ||
        note.starts_with("-dollar bill"))
        return "bank note";
    for (const std::string clock : {"am", "pm", " am", " pm", "a.m", "p.m", " a.m", " p.m"})
        if (after.starts_with(clock) && (after.size() == clock.size() || !letter(after[clock.size()]))) return "time";
    if ((at(n.end) == '/' && digit_after) || (n.at > 0 && q[n.at - 1] == '/' && digit_before)) return "fraction";
    if (letter(at(n.end)) && letter(at(n.end + 1))) {
        const std::string suffix = lower(q.substr(n.end, 2));
        if (suffix == "st" || suffix == "nd" || suffix == "rd" || suffix == "th") return "ordinal";
    }
    if (whole(v) && v.num >= 1900 && v.num <= 2100 && n.written.size() == 4 && (n.at == 0 || q[n.at - 1] != '$'))
        return "year";
    if (whole(v) && v.num != 2 && v.num != 1 && v.num != 100 && usual_constant(v, problem_lower)) return "unit";
    // Facts about units the question states: "a gallon (the same as 8
    // pints)", "16 ounces", "3 feet" where it speaks of yards.
    struct Fact {
        int64_t value;
        const char* unit;
        const char* whole;
    };
    static const Fact facts[] = {{8, "pint", "gallon"}, {4, "quart", "gallon"},   {16, "ounce", "pound"},
                                 {3, "feet", "yard"},   {3, "foot", "yard"},      {12, "inch", "foot"},
                                 {12, "inch", "feet"},  {4, "quarter", "dollar"}, {10, "dime", "dollar"}};
    const std::vector<std::string> next = words_after(q, n.end);
    for (const Fact& f : facts)
        if (whole(v) && v.num == f.value && !next.empty() && next[0].starts_with(f.unit) &&
            problem_lower.find(f.whole) != std::string::npos)
            return "unit";
    return "";
}

// "an" before a number said with a vowel sound: 8, 11, 18, 80-89, 800...,
// 11,000, 18,000, ... ("an 8-hour shift", "an 11-year-old").
inline bool needs_an(const std::string& number) {
    std::string digits;
    for (char c : number) {
        if (c == '.') break;
        if (labels::detail::digit(c)) digits += c;
    }
    if (digits.empty()) return false;
    if (digits[0] == '8') return true;
    return digits.size() % 3 == 2 && (digits.starts_with("11") || digits.starts_with("18"));
}

// `text` with every "a"/"an" right before a number (or "$" and a number)
// matching how the number is said. Only articles before `changed` positions
// are touched (start offsets of numbers written anew).
inline std::string fix_articles(const std::string& text, const std::vector<size_t>& changed) {
    std::string out = text;
    // Back to front, so earlier offsets stay right.
    for (auto it = changed.rbegin(); it != changed.rend(); ++it) {
        size_t at = *it;
        size_t end = at;
        while (end < out.size() && (labels::detail::digit(out[end]) || out[end] == ',' || out[end] == '.')) ++end;
        const std::string number = out.substr(at, end - at);
        if (at > 0 && out[at - 1] == '$') --at;
        if (at < 2 || out[at - 1] != ' ') continue;
        size_t w = at - 1;
        while (w > 0 && letter(out[w - 1])) --w;
        const std::string word = out.substr(w, at - 1 - w);
        const std::string lw = lower(word);
        if (lw != "a" && lw != "an") continue;
        const bool upper = word[0] == 'A';
        const std::string want = needs_an(number) ? (upper ? "An" : "an") : (upper ? "A" : "a");
        out.replace(w, word.size(), want);
    }
    return out;
}

// `text` masked (redraw::masked) with "an #" read as "a #": the variant's
// words must be the original's, its articles aside.
inline std::string masked_words(const std::string& text) {
    std::string m = redraw::masked(text);
    for (const char* from : {"an #", "An #", "an $#", "An $#"}) {
        const std::string f(from), to = f.substr(0, 1) + f.substr(2);
        for (size_t at = 0; (at = m.find(f, at)) != std::string::npos;) {
            if (at > 0 && letter(m[at - 1])) {
                ++at;
                continue;
            }
            m.replace(at, f.size(), to);
            at += to.size();
        }
    }
    return m;
}

// The number v written as an occurrence was: with its decimal places (at
// least), its digit-grouping commas if it had them.
inline std::string written_like(const Rational& v, size_t at_least_places, bool commas) {
    std::string s = decimal(v).value_or("?");
    const size_t p = places(s);
    if (p < at_least_places) s += (p == 0 ? "." : "") + std::string(at_least_places - p, '0');
    if (commas) {
        const size_t point = std::min(s.find('.'), s.size());
        for (size_t i = point; i > 3; i -= 3) s.insert(i - 3, ",");
    }
    return s;
}

// Where a number of the worked solution comes from.
struct Slot {
    enum Kind { kept, variable, hundredth, result } kind = kept;
    int index = -1;  // the variable (also for its hundredth: "0.3" for "30%"), or the calculation
    Number n;
    size_t places = 0;
    bool commas = false;
    bool in_note_result = false;  // between "=" and ">>" of a note: the calculator writes it
};

// Where each character of the labels' work (failure_labels.h: the text
// without digit-grouping commas, then without "$") was in `text`.
inline std::vector<size_t> work_positions(const std::string& text) {
    auto digit = [&](size_t i) { return i < text.size() && labels::detail::digit(text[i]); };
    std::vector<size_t> out;
    for (size_t i = 0; i < text.size(); ++i) {
        const bool grouping = text[i] == ',' && i > 0 && digit(i - 1) && digit(i + 1) && digit(i + 2) && digit(i + 3) &&
                              !digit(i + 4);  // as strip_digit_commas has it
        if (!grouping && text[i] != '$') out.push_back(i);
    }
    return out;
}

// A decimal written without its 0 (".5", "$.50", "18*.5") written with it:
// the labels and the formulas do not read ".5" as a number. Only after the
// start of a line, a blank, "$", "(" or an operator ("No.5" stays).
inline std::string with_zero_points(const std::string& text) {
    std::string out;
    for (size_t i = 0; i < text.size(); ++i) {
        const bool point = text[i] == '.' && i + 1 < text.size() && labels::detail::digit(text[i + 1]);
        const bool before = i == 0 || std::string_view(" \t\n$(+-*/x=<>").find(text[i - 1]) != std::string_view::npos;
        if (point && before && (i == 0 || text[i - 1] != 'x' || calc::times_sign(text, i - 1))) out += '0';
        out += text[i];
    }
    return out;
}

// The problem as redraw-qa reads it: its decimals with their 0 (above).
inline Problem normalized(Problem p) {
    p.question = with_zero_points(p.question);
    p.steps = with_zero_points(p.steps);
    return p;
}

inline bool digits_all_read(const std::string& text) {
    std::vector<bool> read(text.size(), false);
    for (const Number& n : labels::detail::numbers_in(text))
        for (size_t i = n.at; i < n.end; ++i) read[i] = true;
    for (size_t i = 0; i < text.size(); ++i)
        if (labels::detail::digit(text[i]) && !read[i]) return false;
    return true;
}

}  // namespace detail

// One problem, read: what changes, what stays, and its formulas.
struct Analysis {
    Reason reason = Reason::converted;
    std::string detail;  // the number or calculation a rejection is about
    std::vector<detail::QuestionNumber> question;
    std::vector<Rational> variables;  // each variable's original value
    std::vector<size_t> variable_places;
    std::vector<bool> variable_percent;
    std::vector<int64_t> variable_cap;  // the most each may become (0: no bound)
    std::vector<Rational> probe;        // each variable's probe value (distinct primes)
    std::vector<Rational> words;      // numbers the question gives in words, and 1/2, 1/3, 1/4 for half, third, quarter
    labels::AnswerLabel label;        // the gold solution's calculations
    std::vector<Rational> results;    // each calculation's value
    size_t answer_calc = 0;           // the calculation whose result is the answer
    std::string formula;              // the answer's formula ("q1+(q1/2)")
    bool hundredths = false;          // a calculation uses a percentage as its hundredth ("20*0.2" for 20%)
    std::vector<detail::Slot> slots;  // the worked solution's numbers, in order
};

namespace detail {

// The number sets read_calcs works on: the question's numbers (set 0), and
// each set of variable values (a fixed number keeps its value everywhere).
// After them, each percentage's hundredth ("30%" used as 0.3), in order.
inline std::vector<Rational> numbers_with(const Analysis& a, const std::vector<Rational>& values) {
    std::vector<Rational> out;
    for (const QuestionNumber& q : a.question) out.push_back(q.variable < 0 ? q.n.value : values[size_t(q.variable)]);
    for (size_t i = 0; i < a.question.size(); ++i)
        if (a.question[i].percent) out.push_back(*calc::divide(out[i], {100, 1}));
    return out;
}

// The calculations' values on each set of variable values: values[j][k] is
// calculation k's on set j (set 0 the original). Nothing when one does not
// work out (a division by zero, a value that does not fit) or has more than
// one reading.
inline std::optional<std::vector<std::vector<Rational>>> work_out(
    const Analysis& a, const std::vector<std::vector<Rational>>& variable_sets) {
    std::vector<std::vector<Rational>> numbers = {numbers_with(a, a.variables)};
    for (const auto& s : variable_sets) numbers.push_back(numbers_with(a, s));
    const auto calcs = self_train::redraw_detail::read_calcs(numbers, a.words, a.label);
    std::vector<std::vector<Rational>> out(numbers.size());
    for (const auto& c : calcs) {
        if (c.readings.size() != 1 || c.readings[0].values.size() != numbers.size()) return {};
        for (size_t j = 0; j < numbers.size(); ++j) out[j].push_back(c.readings[0].values[j]);
    }
    return out;
}

inline std::vector<Rational> primes_avoiding(size_t count, const std::vector<Rational>& avoid) {
    std::vector<Rational> out;
    for (int64_t p = 211; out.size() < count; ++p) {
        bool prime = true;
        for (int64_t d = 2; d * d <= p && prime; ++d) prime = p % d != 0;
        if (prime && !labels::detail::contains(avoid, {p, 1})) out.push_back({p, 1});
    }
    return out;
}

}  // namespace detail

// Reads one problem (steps 1 and 2 of the top of this file).
inline Analysis analyze(const Problem& p) {
    using namespace detail;
    Analysis a;
    auto reject = [&](Reason r, std::string why) {
        a.reason = r;
        a.detail = std::move(why);
        return a;
    };
    const std::string problem_lower = lower(p.question + "\n" + p.steps);
    if (!digits_all_read(p.question) || !digits_all_read(p.steps))
        return reject(Reason::unreadable, "a digit not read as a number");
    a.words = labels::detail::number_words(p.question);
    for (const auto& [word, part] :
         {std::pair<const char*, int64_t>{"half", 2}, {"halves", 2}, {"third", 3}, {"quarter", 4}})
        if (lower(p.question).find(word) != std::string::npos && !labels::detail::contains(a.words, {1, part}))
            a.words.push_back({1, part});
    a.label = labels::label_answer(p, p.completion());
    if (a.label.calcs.empty()) return reject(Reason::no_calcs, "");
    if (a.label.unreadable > 0) return reject(Reason::unreadable, "an \"=\" the calculator cannot check");
    for (const labels::Calculation& c : a.label.calcs) {
        if (!c.ok) return reject(Reason::wrong, c.text);
        if (c.negative || (!c.exact.empty() && c.exact[0] == '-'))
            return reject(Reason::wrong, "below zero: " + c.text);
    }

    // The question's numbers: fixed, or variables (equal values: one variable).
    for (const Number& n : labels::detail::numbers_in(p.question)) {
        QuestionNumber q;
        q.n = n;
        const std::string_view rest = std::string_view(p.question).substr(n.end);
        q.percent = rest.starts_with("%") || rest.starts_with(" percent") || rest.starts_with("-percent");
        q.places = n.written.find('.') == std::string::npos ? 0 : n.written.size() - n.written.find('.') - 1;
        q.commas = p.question.substr(n.at, n.end - n.at).find(',') != std::string::npos;
        q.fixed_why = why_fixed(p.question, n, a.words, problem_lower);
        q.cap = unit_cap(p.question, n.end);
        a.question.push_back(q);
    }
    // A value that stays anywhere stays everywhere ("2 hours" and "the 2nd").
    for (QuestionNumber& q : a.question)
        if (q.fixed_why.empty())
            for (const QuestionNumber& o : a.question)
                if (!o.fixed_why.empty() && same_value(o.n.value, q.n.value)) q.fixed_why = "same as a fixed number";
    for (QuestionNumber& q : a.question) {
        if (!q.fixed_why.empty()) continue;
        for (size_t v = 0; v < a.variables.size(); ++v)
            if (same_value(a.variables[v], q.n.value)) q.variable = int(v);
        if (q.variable < 0) {
            q.variable = int(a.variables.size());
            a.variables.push_back(q.n.value);
            a.variable_places.push_back(0);
            a.variable_percent.push_back(false);
            a.variable_cap.push_back(0);
        }
        a.variable_places[size_t(q.variable)] = std::max(a.variable_places[size_t(q.variable)], q.places);
        a.variable_percent[size_t(q.variable)] = a.variable_percent[size_t(q.variable)] || q.percent;
        if (q.cap > 0) {
            int64_t& cap = a.variable_cap[size_t(q.variable)];
            cap = cap == 0 ? q.cap : std::min(cap, q.cap);
        }
    }
    std::vector<Rational> taken = a.words;
    for (const QuestionNumber& q : a.question) taken.push_back(q.n.value);
    a.probe = primes_avoiding(a.variables.size(), taken);

    // Where each calculation's "=" is in the worked solution.
    const std::string& s = p.steps;
    const std::vector<size_t> where = work_positions(s);
    std::vector<size_t> equals_at;
    for (const labels::Calculation& c : a.label.calcs) {
        if (c.at >= where.size() || s[where[c.at]] != '=')
            return reject(Reason::unreadable, "a calculation not found: " + c.text);
        equals_at.push_back(where[c.at]);
    }

    // The formulas, on the original numbers and the probe: two sources are
    // told apart by their probe values.
    const std::vector<std::vector<Rational>> sets = {a.variables, a.probe};
    std::vector<std::vector<Rational>> numbers;
    for (const auto& s : sets) numbers.push_back(numbers_with(a, s));
    const auto calcs = self_train::redraw_detail::read_calcs(numbers, a.words, a.label);
    std::vector<size_t> percents;  // the question numbers whose hundredths follow theirs
    for (size_t i = 0; i < a.question.size(); ++i)
        if (a.question[i].percent) percents.push_back(i);
    std::vector<bool> used(a.variables.size(), false);
    std::vector<Rational> probe_results;
    std::set<std::string> constants_seen;
    for (size_t k = 0; k < calcs.size(); ++k) {
        const auto& c = calcs[k];
        const std::string& text = a.label.calcs[k].text;
        if (c.ambiguous) return reject(Reason::ambiguous, text);
        if (!c.readable || c.readings.size() != 1) return reject(Reason::unreadable, "not followed: " + text);
        const auto& r = c.readings[0];
        if (r.formula.ends_with("...")) return reject(Reason::unreadable, "formula too long: " + text);
        // A formula carries the constants of the results it uses: each is
        // judged once, on the line of the calculation that uses it first.
        for (const std::string& k_text : r.constants) {
            if (constants_seen.count(k_text)) continue;
            const auto v = labels::detail::value_of(k_text);
            if (!v || !constant_here(*v, problem_lower, line_at(s, equals_at[k])))
                return reject(Reason::constant, k_text + " in " + text);
            constants_seen.insert(k_text);
        }
        for (size_t i = 0; i + 1 < r.formula.size(); ++i)
            if (r.formula[i] == 'q' && labels::detail::digit(r.formula[i + 1])) {
                size_t q = size_t(std::stoul(r.formula.substr(i + 1))) - 1;
                if (q >= a.question.size()) q = percents[q - a.question.size()], a.hundredths = true;
                if (q < a.question.size() && a.question[q].variable >= 0) used[size_t(a.question[q].variable)] = true;
            }
        a.results.push_back(r.values[0]);
        probe_results.push_back(r.values[1]);
    }

    // The answer: the one calculation (or the same one written again) giving it.
    const auto gold = calc::Parser(strip_digit_commas(trim(p.answer))).parse();
    if (!gold) return reject(Reason::answer, "the answer is no number: " + p.answer);
    std::optional<size_t> answer;
    for (size_t k = a.results.size(); k-- > 0;) {
        if (!same_value(a.results[k], *gold)) continue;
        if (answer && !same_value(probe_results[*answer], probe_results[k]))
            return reject(Reason::answer, "two calculations give the answer");
        if (!answer) answer = k;
    }
    if (!answer) return reject(Reason::answer, "no calculation gives " + p.answer);
    a.answer_calc = *answer;
    // The formula, with each hundredth written as one: "q9" is "(q3/100)".
    const std::string& f = calcs[*answer].readings[0].formula;
    for (size_t i = 0; i < f.size(); ++i) {
        if (f[i] == 'q' && i + 1 < f.size() && labels::detail::digit(f[i + 1])) {
            size_t j = i + 1;
            while (j < f.size() && labels::detail::digit(f[j])) ++j;
            const size_t q = size_t(std::stoul(f.substr(i + 1, j - i - 1))) - 1;
            a.formula += q < a.question.size() ? f.substr(i, j - i)
                                               : "(q" + std::to_string(percents[q - a.question.size()] + 1) + "/100)";
            i = j - 1;
        } else {
            a.formula += f[i];
        }
    }
    for (const QuestionNumber& q : a.question)
        if (q.variable >= 0 && !used[size_t(q.variable)]) return reject(Reason::unused, q.n.written);
    if (a.variables.empty()) return reject(Reason::nothing, "");

    // The worked solution's numbers, each with its one source.
    struct Source {
        Rational value, probe;
        Slot::Kind kind;
        int index;
    };
    std::vector<Source> sources;
    for (const QuestionNumber& q : a.question)
        sources.push_back(q.variable < 0 ? Source{q.n.value, q.n.value, Slot::kept, -1}
                                         : Source{q.n.value, a.probe[size_t(q.variable)], Slot::variable, q.variable});
    for (const size_t i : percents) {
        const QuestionNumber& q = a.question[i];
        const Rational v = *calc::divide(q.n.value, {100, 1});
        sources.push_back(q.variable < 0 ? Source{v, v, Slot::kept, -1}
                                         : Source{v, *calc::divide(a.probe[size_t(q.variable)], {100, 1}),
                                                  Slot::hundredth, q.variable});
    }
    for (const Rational& w : a.words) sources.push_back({w, w, Slot::kept, -1});
    for (size_t k = 0; k < a.results.size(); ++k)
        sources.push_back({a.results[k], probe_results[k], Slot::result, int(k)});
    // The notes, in order, are the calculations the labels read from notes
    // (their text ends with the note): a note's result, and the number
    // written right after it, is that calculation's result.
    // A number can only stand for a result already worked out, or worked
    // out on its own line ("He eats 32 because 2 x 16 = 32"): one whose "="
    // (equals_at) comes before the end of its line.
    std::vector<size_t> note_calcs;
    for (size_t k = 0; k < a.label.calcs.size(); ++k)
        if (a.label.calcs[k].text.ends_with(">>")) note_calcs.push_back(k);
    size_t notes = 0;
    for (size_t at = 0; (at = s.find("<<", at)) != std::string::npos; at += 2) ++notes;
    if (notes != note_calcs.size()) return reject(Reason::unreadable, "a note not read as a calculation");
    for (const Number& n : labels::detail::numbers_in(s)) {
        Slot slot;
        slot.n = n;
        slot.places = n.written.find('.') == std::string::npos ? 0 : n.written.size() - n.written.find('.') - 1;
        slot.commas = s.substr(n.at, n.end - n.at).find(',') != std::string::npos;
        const size_t open = s.rfind("<<", n.at), close = s.rfind(">>", n.at), equals = s.rfind('=', n.at);
        slot.in_note_result = open != std::string::npos && (close == std::string::npos || close < open) &&
                              equals != std::string::npos && equals > open;
        const size_t line_end = std::min(s.find('\n', n.at), s.size());
        const Source* found = nullptr;
        size_t gap = close == std::string::npos ? 0 : close + 2;  // after ">>": "$" and spaces may come first
        while (gap < n.at && (s[gap] == '$' || s[gap] == ' ')) ++gap;
        const bool after_note =
            close != std::string::npos && (open == std::string::npos || close > open) && gap == n.at;
        if (slot.in_note_result || after_note) {
            size_t note = 0;  // the note's place among the notes
            for (size_t at = 0; (at = s.find("<<", at)) != std::string::npos && at < n.at; at += 2) ++note;
            const size_t k = note_calcs[note - 1];
            if (same_value(a.results[k], n.value))
                found = &sources[a.question.size() + percents.size() + a.words.size() + k];
        }
        for (const Source& src : sources) {
            if (found && (slot.in_note_result || after_note)) break;
            if (!same_value(src.value, n.value)) continue;
            if (src.kind == Slot::result && equals_at[size_t(src.index)] > line_end) continue;  // not yet worked out
            if (found && !same_value(found->probe, src.probe))
                return reject(Reason::text_ambiguous, n.written + " in: " + labels::detail::around(s, n.at, n.end));
            if (!found) found = &src;
        }
        // A result worked out later on the line, or a constant: "4 divided
        // by 2 equals <<4/2=2>>2" (the first 2 is the constant).
        if (found && found->kind == Slot::result && equals_at[size_t(found->index)] > n.at && !slot.in_note_result &&
            !after_note && usual_constant(n.value, problem_lower))
            return reject(Reason::text_ambiguous, n.written + " in: " + labels::detail::around(s, n.at, n.end));
        if (!found) {
            if (!constant_here(n.value, problem_lower, line_at(s, n.at)))
                return reject(Reason::text_unsourced, n.written + " in: " + labels::detail::around(s, n.at, n.end));
            slot.kind = Slot::kept;
        } else {
            slot.kind = found->kind;
            slot.index = found->index;
        }
        a.slots.push_back(slot);
    }
    // A changing number the solution writes with another unit than the
    // question gives it: "x 5 days" (the five weekdays) where the question's
    // 5 is "5 hours" on Saturdays. Its value is the question's by chance.
    for (const Slot& slot : a.slots) {
        if (slot.kind != Slot::variable) continue;
        const std::vector<std::string> unit = words_after(s, slot.n.end);
        bool differ = true;  // from every copy in the question
        for (const QuestionNumber& q : a.question)
            if (q.variable == slot.index) differ = differ && units_differ(unit, words_after(p.question, q.n.end));
        if (differ)
            return reject(Reason::units,
                          slot.n.written + " " + unit[0] + " in: " + labels::detail::around(s, slot.n.at, slot.n.end));
    }
    // A changing number the solution writes in words ("five friends").
    const std::vector<Rational> step_words = labels::detail::number_words(s);
    for (const Rational& v : a.variables)
        if (labels::detail::contains(step_words, v)) return reject(Reason::words, rational_text(v));
    return a;
}

namespace detail {

// One new value for a variable (step 3 at the top of this file), or nothing
// if its range has no other value.
inline std::optional<Rational> draw_value(const Rational& original, size_t decimals, bool percent, Rng& rng,
                                          int64_t cap = 0) {
    int64_t scale = 1;
    for (size_t i = 0; i < decimals; ++i) scale *= 10;
    const auto scaled = calc::times(original, {scale, 1});
    if (!scaled || scaled->den != 1 || scaled->num <= 0 || scaled->num > int64_t(1) << 40) return {};
    const int64_t m = scaled->num;
    // As round as the original: 120 gives multiples of 10, 25 and 250 of 5
    // and 50 (15 is an ordinary number).
    int64_t ten = 1;
    while (m % (ten * 10) == 0 && ten * 10 < m) ten *= 10;
    const int64_t t = m / ten, step = (t % 10 == 5 && t >= 25) ? 5 * ten : ten;
    int64_t lo = std::max((m / 2 + step - 1) / step * step, step), hi = std::max(2 * m, m + 5 * step) / step * step;
    if (decimals == 0) lo = std::max<int64_t>(lo, ((2 + step - 1) / step) * step);
    if (percent && m <= 100 * scale) hi = std::min(hi, (100 * scale - 1) / step * step);
    if (cap > 0 && m <= cap * scale) hi = std::min(hi, cap * scale / step * step);
    if (hi < lo) return {};
    const int64_t choices = (hi - lo) / step + 1;
    const bool has_original = m >= lo && m <= hi && (m - lo) % step == 0;
    if (choices - (has_original ? 1 : 0) <= 0) return {};
    int64_t pick = lo + int64_t(rng.below(uint64_t(choices - (has_original ? 1 : 0)))) * step;
    if (has_original && pick >= m) pick += step;
    // A decimal whose last digit is not 0 keeps one that is not: "2.5
    // times" never becomes "5.0 times" (a draw like that is drawn again).
    if (decimals > 0 && m % 10 != 0 && pick % 10 == 0) return {};
    return calc::lowest(pick, scale);
}

// A new result keeps the original's kind (step 3).
inline bool same_kind(const Rational& was, const Rational& now) {
    const auto text = decimal(now);
    if (!text) return false;  // no exact decimal: a note could not show it
    if ((was.num > 0) != (now.num > 0) || (was.num == 0) != (now.num == 0)) return false;
    const bool was_one = was.num == 1 && was.den == 1, now_one = now.num == 1 && now.den == 1;
    if (was_one != now_one) return false;
    if (whole(was) ? !whole(now) : places(*text) > std::max<size_t>(2, places(decimal(was).value_or("")))) return false;
    if (was.num > 0) {
        // Within 25 times the original either way: now <= 25 was, was <= 25 now.
        const auto up = calc::times(was, {25, 1}), down = calc::times(now, {25, 1});
        if (!up || !down) return false;
        const auto a = calc::plus(*up, {-now.num, now.den}), b = calc::plus(*down, {-was.num, was.den});
        if (!a || !b || a->num < 0 || b->num < 0) return false;
    }
    return true;
}

// The text with its numbers written anew: `value(slot)` gives each slot's
// new text, or nothing to keep it. Returns the text and where the new
// numbers start (for the articles).
template <class F>
inline std::string rewrite(const std::string& text, const std::vector<Number>& numbers, F&& value,
                           std::vector<size_t>& changed) {
    std::string out;
    size_t at = 0;
    for (size_t i = 0; i < numbers.size(); ++i) {
        out += text.substr(at, numbers[i].at - at);
        const std::optional<std::string> now = value(i);
        if (now) changed.push_back(out.size());
        out += now ? *now : text.substr(numbers[i].at, numbers[i].end - numbers[i].at);
        at = numbers[i].end;
    }
    return out + text.substr(at);
}

// Each note's result written as the calculator writes it for its
// expression: nothing if one has none.
inline std::optional<std::string> renote(const std::string& steps) {
    std::string out;
    size_t at = 0;
    for (size_t open; (open = steps.find("<<", at)) != std::string::npos;) {
        const size_t equals = steps.find('=', open), close = steps.find(">>", open);
        if (equals == std::string::npos || close == std::string::npos || equals > close) return {};
        const auto result = calc::evaluate(steps.substr(open + 2, equals - open - 2));
        if (!result) return {};
        out += steps.substr(at, equals + 1 - at) + *result + ">>";
        at = close + 2;
    }
    return out + steps.substr(at);
}

}  // namespace detail

// The final check of one variant (step 5 at the top of this file): "" if it
// passes, else what failed.
inline std::string check_variant(const Problem& original, const Analysis& a, const Problem& v, const Rational& answer) {
    using namespace detail;
    if (masked_words(v.question) != masked_words(original.question)) return "the question's words changed";
    if (masked_words(v.steps) != masked_words(original.steps)) return "the solution's words changed";
    if (v.question == original.question) return "the question did not change";
    const labels::AnswerLabel l = labels::label_answer(v, v.completion());
    if (l.calcs.size() != a.label.calcs.size() || l.unreadable != 0) return "the calculations read differently";
    if (l.wrong_calcs() != 0 || self_train::sign_wrong(l) || self_train::note_miscopied(v.steps) ||
        self_train::note_inexact(v.steps))
        return "a calculation is wrong";
    if (l.label != labels::Label::correct) return "the answer line is not right";
    const auto stated = calc::Parser(v.answer).parse();
    if (!stated || !same_value(*stated, answer)) return "the answer is not the formula's";
    const auto last = calc::Parser(l.calcs[a.answer_calc].exact).parse();
    if (!last || !same_value(*last, answer)) return "the answer calculation does not give the answer";
    return "";
}

// What became of one problem.
struct Outcome {
    Analysis analysis;
    std::vector<Problem> variants;
    size_t draws = 0;     // draws of new numbers made
    bool traced = false;  // self-train's re-draw check accepted the variants (it cannot read hundredths)
    std::string failed;   // the final check's message, for Reason::check
    Reason reason() const { return analysis.reason; }
};

// Up to o.versions variants of problem number `row` (its place in the
// input, from 0: the seed's stream for it).
inline Outcome redraw_problem(const Problem& problem, size_t row, const Options& o) {
    using namespace detail;
    Outcome out;
    const Problem p = normalized(problem);  // the variants are compared with this spelling
    out.analysis = analyze(p);
    Analysis& a = out.analysis;
    if (a.reason != Reason::converted) return out;
    const std::vector<Number> qnums = labels::detail::numbers_in(p.question),
                              snums = labels::detail::numbers_in(p.steps);
    std::vector<Rational> kept;  // values a new variable must not take: they would tie it to another number
    for (const auto& q : a.question)
        if (q.variable < 0) kept.push_back(q.n.value);
    kept.insert(kept.end(), a.words.begin(), a.words.end());
    for (const int64_t c : {1, 2, 100}) kept.push_back({c, 1});
    std::set<std::string> seen = {p.question};
    std::vector<Rational> answers;
    while (int(out.variants.size()) < o.versions && out.draws < size_t(o.draws) * size_t(o.versions)) {
        Rng rng(o.seed * 0x9E3779B97F4A7C15ull + (uint64_t(row) + 1) * 0xBF58476D1CE4E5B9ull + ++out.draws);
        std::vector<Rational> values;
        bool ok = true;
        for (size_t v = 0; v < a.variables.size() && ok; ++v) {
            const auto x =
                draw_value(a.variables[v], a.variable_places[v], a.variable_percent[v], rng, a.variable_cap[v]);
            ok = x && !labels::detail::contains(kept, *x) && !labels::detail::contains(values, *x);
            if (ok) values.push_back(*x);
        }
        if (!ok) continue;
        const auto worked = work_out(a, {values});
        if (!worked) continue;
        const std::vector<Rational>& now = (*worked)[1];
        for (size_t k = 0; k < now.size() && ok; ++k) ok = same_kind(a.results[k], now[k]);
        if (!ok) continue;
        // The question: its variables written anew.
        std::vector<size_t> changed;
        Problem v = p;
        v.task = "gsm8k-redraw";
        v.question = rewrite(
            p.question, qnums,
            [&](size_t i) -> std::optional<std::string> {
                const QuestionNumber& q = a.question[i];
                if (q.variable < 0) return {};
                return written_like(values[size_t(q.variable)], q.places, q.commas);
            },
            changed);
        v.question = fix_articles(v.question, changed);
        changed.clear();
        v.steps = rewrite(
            p.steps, snums,
            [&](size_t i) -> std::optional<std::string> {
                const Slot& s = a.slots[i];
                if (s.kind == Slot::kept) return {};
                const Rational x = s.kind == Slot::variable    ? values[size_t(s.index)]
                                   : s.kind == Slot::hundredth ? *calc::divide(values[size_t(s.index)], {100, 1})
                                                               : now[size_t(s.index)];
                return written_like(x, s.in_note_result ? 0 : s.places, s.commas && !s.in_note_result);
            },
            changed);
        v.steps = fix_articles(v.steps, changed);
        const auto renoted = renote(v.steps);
        if (!renoted) continue;
        v.steps = *renoted;
        const Rational answer = now[a.answer_calc];
        v.answer = decimal(answer).value_or("?");
        if (!seen.insert(v.question).second) continue;
        const std::string failed = check_variant(p, a, v, answer);
        if (!failed.empty()) {
            a.reason = Reason::check;
            out.failed = failed;
            out.variants.clear();
            return out;
        }
        out.variants.push_back(v);
    }
    if (out.variants.empty()) {
        a.reason = Reason::no_draw;
        return out;
    }
    // Self-train's re-draw check, on its own code path: the gold solution's
    // formula gives every variant's answer. It reads "20*0.2" for 20% as a
    // constant 0.2, so it cannot check a solution that uses a hundredth.
    if (a.hundredths) return out;
    const self_train::Trace t = self_train::trace_try(p, a.label, out.variants);
    out.traced = t.passed;
    if (!t.passed) {
        a.reason = Reason::check;
        out.failed = "self-train's re-draw check rejects the gold solution (" + t.formula + ")";
        out.variants.clear();
    }
    return out;
}

}  // namespace redraw_qa
