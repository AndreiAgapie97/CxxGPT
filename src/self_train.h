// self_train.h — the model's own checked solutions as training data
// (docs/research-plan.md, "Learning from its own checked solutions").
//
// The model tries each problem several times (sample_tries, generate.h). A
// try is kept only if all of these hold:
//   - it finished its "#### answer" line, and the answer is the right one
//     (scored exactly as eval-reasoning scores it);
//   - it shows some work before the answer line;
//   - it holds no "[[no-loss]]" marker and no BOS byte: those would change
//     how its tokens are masked or where its document ends;
//   - every calculation it wrote is right by the exact calculator (calc.h),
//     both forms: "<<48/2=24>>" notes and plain "16 x 7 = 112", found the
//     way failure_labels.h finds them, with the right sign, and the number
//     written right after a note is the note's, sign included; a note the
//     calculator leaves to the model (a value with no finite decimal,
//     "<<20/3=") is right only as that value rounded to its decimal places,
//     never as a whole number. A right answer reached through a wrong
//     calculation is a lucky answer, and would teach the mistake;
//   - it does not repeat a line of 12 or more characters (the failure
//     labels' loop flag), unless loops are allowed;
//   - it uses no number from outside the question (the failure labels'
//     outside flag: not in the question, not a result after "=", not one
//     +, -, * or / away from numbers already seen), other than a few
//     common constants (kConstants: 7 days a week, 60 minutes an hour,
//     ...), unless outside numbers are allowed: an invented quantity is a
//     wrong plan even when the answer comes out right;
//   - it uses every number of the question (but 1, and a percentage may be
//     used as its hundredth: "30%" as 0.3), unless unused numbers are
//     allowed or the re-draw check runs: rp1's lucky answers on the textbook
//     problems ignore one ("7 x 7 = 49" for a triangle of base 14, height
//     7). On GSM8K dev it drops 57% of the right tries that reach under
//     half the gold's results and 3% of those that reach all of them; 8%
//     of GSM8K's own train solutions leave a number unused ("a 2008
//     study"). Generated problems add numbers on purpose (a distractor
//     sentence), so their re-draw check replaces this one;
//   - it is not the problem file's own solution: a model that has learned
//     its training problems, or a generator's templates, by heart writes
//     those back, and they teach it nothing new;
//   - for a generated problem with a recipe (redraw.h), the re-draw check:
//     the try's calculations, as a formula over the question's numbers,
//     give the answer of every version of the story with other numbers
//     (see "The re-draw check" below).
// Of the tries left, one that is the same solution as an earlier kept try
// is dropped, and at most `keep` per problem are kept (0: all), in try
// order. "The same solution" (for the gold solution too) means the same
// outline by default: the same calculations' values in the same order, in
// whatever words ("12*4 = 48 pencils" and "The start is 12 x 4 = 48");
// work without a readable calculation is compared as text. With
// Same::text, only the same text (spaces around it aside) is the same.
// Everything is decided in a fixed order, so the same seed gives the same
// file.
//
// A kept try is written as prep-qa writes a problem: BOS and "Q: ...\nA: "
// masked out (not trained on), then the model's work and "#### " with the
// right answer as the problem file gives it ("1200", where the model may
// have written "1,200"), trained on.
#pragma once

#include <algorithm>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "failure_labels.h"
#include "generate.h"
#include "loss_mask.h"
#include "reasoning.h"
#include "train.h"

// One problem appended to a token file the way training reads a question
// and its answer: BOS and the prompt masked out, the completion trained on
// (except [[no-loss]] spans, loss_mask.h). prep-qa and gen-reasoning use it
// too.
inline void append_problem(TokenData& d, const Problem& p, const Tokenizer& tok) {
    const std::vector<int> prompt = prompt_tokens(p, tok);
    for (int t : prompt) d.tokens.push_back(uint16_t(t)), d.mask.push_back(0);
    encode_loss_marked(p.completion(), tok, d.tokens, d.mask);
}

namespace self_train {

// What became of one try, in the order the checks are made.
enum class Verdict {
    kept,
    too_long,     // the prompt does not fit in the model's context
    unfinished,   // no finished "#### answer" line
    wrong,        // another final answer
    no_work,      // the right answer with nothing written before it
    markers,      // a "[[no-loss]]" marker or a BOS byte in the text
    arithmetic,   // the right answer, but a calculation in the work is wrong
    loop,         // the right answer, but a line repeats
    outside,      // the right answer, but a number comes from outside the question
    unused,       // the right answer, but a number of the question is never used
    gold_copy,    // the problem file's own solution (outline or text): nothing new
    ambiguous,    // a number on the answer's path could come from two places
    unchecked,    // the re-draw check could not run: no formula for the answer, or no versions
    redraw,       // the formula misses a version's answer: right by luck
    dead,         // a calculation whose result nothing uses (dead_calculations)
    mismatch,     // the text before a note is another calculation than the note's (note_mismatch)
    broken,       // a note not written "<<calculation=result>>" (broken_note)
    uncomputed,   // no calculation gives the answer (answer_computed)
    duplicate,    // the same solution as an earlier kept try of the problem
    over_cap,     // the problem already has `keep` kept tries
};
inline constexpr int kVerdicts = 20;

inline const char* verdict_name(Verdict v) {
    switch (v) {
        case Verdict::kept: return "kept";
        case Verdict::too_long: return "too_long";
        case Verdict::unfinished: return "unfinished";
        case Verdict::wrong: return "wrong";
        case Verdict::no_work: return "no_work";
        case Verdict::markers: return "markers";
        case Verdict::arithmetic: return "arithmetic";
        case Verdict::loop: return "loop";
        case Verdict::outside: return "outside";
        case Verdict::unused: return "unused";
        case Verdict::gold_copy: return "gold_copy";
        case Verdict::ambiguous: return "ambiguous";
        case Verdict::unchecked: return "unchecked";
        case Verdict::redraw: return "redraw";
        case Verdict::dead: return "dead";
        case Verdict::mismatch: return "mismatch";
        case Verdict::broken: return "broken";
        case Verdict::uncomputed: return "uncomputed";
        case Verdict::duplicate: return "duplicate";
        case Verdict::over_cap: return "over_cap";
    }
    return "?";
}

// What makes two tries the same solution (see the top of this file).
enum class Same { outline, text };

struct Options {
    int keep = 0;               // kept tries per problem at most; 0: all
    bool allow_loops = false;   // keep right tries that repeat a line
    bool allow_outside = false; // keep right tries with numbers from outside the question
    bool allow_unused = false;  // keep right tries that leave a number of the question unused
    bool keep_gold = false;     // keep tries that are the problem's own solution
    Same same = Same::outline;
    bool keep_unchecked = false;  // keep tries the re-draw check cannot check
    bool every_reading = true;    // an ambiguous try passes if one of its readings does (false: --strict-readings)
    bool strict_outside = false;  // no number one + - * / away from the given ones (unearned_numbers)
    bool check_dead = false, check_mismatch = false, check_computed = false;
    bool check_broken = false;
};

// Numbers a solution may bring in without the question giving them: days
// in a week, months and hours, minutes and seconds, weeks and days in a
// year, and metric units. (The failure labels already allow 1, 2 and 100.)
inline constexpr int64_t kConstants[] = {7, 12, 24, 52, 60, 365, 1000};

inline bool is_constant(const calc::Rational& v) {
    return v.den == 1 && std::find(std::begin(kConstants), std::end(kConstants), v.num) != std::end(kConstants);
}

// The question's numbers the work never writes: digits only, "$" and
// digit-grouping commas dropped (as the labels read numbers), 1 aside (the
// unit in "1 hour each day"), and a percentage written as its hundredth
// counts ("30%" used as 0.3). In the order of the question.
inline std::vector<std::string> unused_numbers(const Problem& p, const std::string& work) {
    auto clean = [](std::string s) {
        s = strip_digit_commas(s);
        s.erase(std::remove(s.begin(), s.end(), '$'), s.end());
        return s;
    };
    std::vector<calc::Rational> written;
    for (const auto& n : labels::detail::numbers_in(clean(work))) written.push_back(n.value);
    auto has = [&](const calc::Rational& v) {
        return std::any_of(written.begin(), written.end(), [&](const calc::Rational& w) { return labels::detail::same(w, v); });
    };
    std::vector<std::string> out;
    const std::string question = clean(p.question);
    for (const auto& n : labels::detail::numbers_in(question)) {
        if (n.value.num == n.value.den) continue;
        const std::string_view rest = std::string_view(question).substr(n.end);
        const bool percent = rest.starts_with("%") || rest.starts_with(" percent");
        const auto hundredth = calc::times(n.value, calc::Rational{1, 100});
        if (!has(n.value) && !(percent && hundredth && has(*hundredth))) out.push_back(n.written);
    }
    return out;
}

// The work's numbers that nothing gives it, a stricter outside check: a
// number must be in the question (in digits or words, a fraction word as
// its parts, "four-fifths" giving 4 and 5, and a percentage as its
// hundredth too), 1, 2, 100 or a constant, the result of a calculation
// (right after "="), or written earlier in the work. The outside check
// also lets in a number one + - * / away from those, which lets in almost
// any small number (4 is 2 * 2); rp1's lucky tries use that ("12/4 = 3,
// 12*3 = 36" for the LCM of 9 and 12). In the order of the work.
inline std::vector<std::string> unearned_numbers(const Problem& p, const std::string& work) {
    using namespace labels::detail;
    auto clean = [](std::string s) {
        s = strip_digit_commas(s);
        s.erase(std::remove(s.begin(), s.end(), '$'), s.end());
        return s;
    };
    const std::string question = clean(p.question);
    std::vector<calc::Rational> given = number_words(question);
    // Fraction words number_words leaves out ("fifths", "tenth").
    static const std::pair<const char*, int64_t> parts[] = {{"fourth", 4}, {"fifth", 5}, {"sixth", 6},
                                                            {"seventh", 7}, {"eighth", 8}, {"ninth", 9},
                                                            {"tenth", 10}};
    for (const auto& [word, v] : parts)
        if (question.find(word) != std::string::npos) given.push_back({v, 1});
    for (const auto& n : numbers_in(question)) {
        given.push_back(n.value);
        if (const auto h = calc::times(n.value, calc::Rational{1, 100})) given.push_back(*h);
    }
    for (const int64_t c : {1, 2, 100}) given.push_back({c, 1});
    std::vector<std::string> out;
    const std::string w = clean(work);
    for (const auto& n : numbers_in(w)) {
        size_t k = n.at;
        while (k > 0 && w[k - 1] == ' ') --k;
        const bool result = k > 0 && w[k - 1] == '=';
        if (!result && !contains(given, n.value) && !is_constant(n.value)) out.push_back(n.written);
        given.push_back(n.value);
    }
    return out;
}

// The work as the failure labels read it (label_answer): the text before
// the answer line, "$" and digit-grouping commas dropped. Their
// calculations' positions (Calculation::at) are in this text.
inline std::string labels_work(const std::string& text) {
    std::string w = strip_digit_commas(text.substr(0, text.find("#### ")));
    w.erase(std::remove(w.begin(), w.end(), '$'), w.end());
    return w;
}

// Decimals written without their 0 (".4 liters", "60 x .75") with it, so
// that numbers_in reads them (it skips a number right after a "."). For
// reading values only: positions move.
inline std::string with_zeros(std::string s) {
    for (size_t i = 0; i + 1 < s.size(); ++i)
        if (s[i] == '.' && labels::detail::digit(s[i + 1]) && (i == 0 || !labels::detail::digit(s[i - 1])))
            s.insert(i, "0"), ++i;
    return s;
}

// a is b, or b as a percentage of it or a share: "100 - 25 = 75" is used
// as ".75", "0.25" is "25" percent.
inline bool same_or_percent(const calc::Rational& a, const calc::Rational& b) {
    using labels::detail::same;
    if (same(a, b)) return true;
    const auto up = calc::times(a, calc::Rational{100, 1}), down = calc::times(a, calc::Rational{1, 100});
    return (up && same(*up, b)) || (down && same(*down, b));
}

// Where the result written after a calculation's "=" ends: after the note's
// ">>" and the number copied after it, or after a plain "= 24".
inline size_t result_end(const std::string& w, size_t eq) {
    size_t pos = eq + 1;
    const size_t open = w.rfind("<<", eq), before = w.rfind(">>", eq), close = w.find(">>", eq);
    const bool in_note = open != std::string::npos && (before == std::string::npos || before < open) &&
                         close != std::string::npos && close < w.find('\n', eq);
    if (in_note) pos = close + 2;
    while (pos < w.size() && w[pos] == ' ') ++pos;
    const auto n = labels::detail::numbers_in(std::string_view(w).substr(pos));
    if (!n.empty() && n[0].at == 0) pos += n[0].end;
    return pos;
}

// Calculations whose result nothing uses: no later number of the work is
// that value (or it as a percentage), and it is not the answer ("171 + 380
// = 551" and then never 551 again). A clean solution computes what it
// needs.
inline std::vector<std::string> dead_calculations(const labels::AnswerLabel& label, const Problem& p,
                                                  const std::string& text) {
    using namespace labels::detail;
    const std::string w = labels_work(text);
    const auto answer = value_of(strip_digit_commas(p.answer));
    std::vector<std::string> out;
    for (const auto& c : label.calcs) {
        const auto v = value_of(c.exact);
        if (!v) continue;
        const calc::Rational m = magnitude(*v);
        if (answer && same_or_percent(m, magnitude(*answer))) continue;
        bool used = false;
        for (const auto& n : numbers_in(with_zeros(w.substr(result_end(w, c.at))))) used |= same_or_percent(m, n.value);
        if (!used) out.push_back(c.text);
    }
    return out;
}

// The calculation written right before a note, when it is another one:
// a whole calculation with another value (".71 x 100 = <<171*100=17100>>",
// "60 / 1 = <<60/4=15>>"; "16 + 20 + 48 = <<8*2+4*5+12*4=84>>" is the same),
// or else a number the note leaves out. The note may add numbers ("15% *
// 1000 = <<15*.01*1000=150>>"), and only the arithmetic right before the
// " = " is read ("20 minutes/day * 3 = <<20*3=60>>" gives "* 3").
inline bool note_mismatch(const std::string& text) {
    using namespace labels::detail;
    const std::string w = labels_work(text);
    for (size_t at = 0; (at = w.find("<<", at)) != std::string::npos; at += 2) {
        const size_t eq = w.find('=', at), close = w.find(">>", at);
        if (eq == std::string::npos || close == std::string::npos || eq > close) continue;
        size_t end = at;
        while (end > 0 && w[end - 1] == ' ') --end;
        if (end == 0 || w[end - 1] != '=') continue;
        --end;
        size_t begin = end;
        while (begin > 0) {
            const char c = w[begin - 1];
            // Digits, signs, and the bytes of "×", "÷" and "−" (U+2212).
            if (digit(c) || std::string_view(".+-*/() \t\xc3\x97\xb7\xe2\x88\x92").find(c) != std::string_view::npos ||
                (c == 'x' && calc::times_sign(w, begin - 1)))
                --begin;
            else
                break;
        }
        const std::string run = with_zeros(w.substr(begin, end - begin));
        if (const auto whole = whole_calculation(run)) {
            const auto mine = calc::Parser(*whole).parse(), theirs = calc::Parser(w.substr(at + 2, eq - at - 2)).parse();
            if (mine && theirs && !same(magnitude(*mine), magnitude(*theirs))) return true;
            continue;
        }
        std::vector<calc::Rational> note;
        for (const auto& n : numbers_in(with_zeros(w.substr(at + 2, eq - at - 2)))) note.push_back(n.value);
        for (const auto& n : numbers_in(run)) {
            const auto it =
                std::find_if(note.begin(), note.end(), [&](const calc::Rational& x) { return same(x, n.value); });
            if (it == note.end()) return true;
            note.erase(it);
        }
    }
    return false;
}

// A note that is not "<<calculation=result>>": no "=" ("<<19.45>>",
// "<<27%>>"), no operator before it ("<<60=60>>"), or a ">>" that closes
// no note ("<<300-255=45>>45>>45").
inline bool broken_note(const std::string& text) {
    const std::string w = labels_work(text);
    size_t opens = 0, closes = 0;
    for (size_t at = 0; (at = w.find(">>", at)) != std::string::npos; at += 2) ++closes;
    for (size_t at = 0; (at = w.find("<<", at)) != std::string::npos; at += 2) {
        ++opens;
        const size_t close = w.find(">>", at);
        if (close == std::string::npos) return true;
        const std::string_view inner = std::string_view(w).substr(at + 2, close - at - 2);
        const size_t eq = inner.find('=');
        if (eq == std::string_view::npos || eq == 0) return true;
        if (inner.substr(0, eq).find_first_of("+-*/", 1) == std::string_view::npos &&
            inner.substr(0, eq).find("\xc3\x97") == std::string_view::npos)
            return true;
    }
    return closes > opens;
}

// The answer is computed: some calculation of the work gives it (or it as
// a percentage), or it is written right after an "=" with its sign
// ("4+6=10", "x = 2350", which the calculation check does not read). A
// number answer only.
inline bool answer_computed(const labels::AnswerLabel& label, const Problem& p, const std::string& text) {
    using namespace labels::detail;
    const auto answer = value_of(strip_digit_commas(p.answer));
    if (!answer) return true;
    for (const auto& c : label.calcs)
        if (const auto v = value_of(c.exact); v && same_or_percent(magnitude(*v), magnitude(*answer))) return true;
    const std::string w = with_zeros(labels_work(text));
    for (size_t i = 0; (i = w.find('=', i)) != std::string::npos; ++i) {
        size_t k = i + 1;
        while (k < w.size() && w[k] == ' ') ++k;
        const bool minus = k < w.size() && w[k] == '-';
        const auto n = numbers_in(std::string_view(w).substr(k + minus));
        if (!n.empty() && n[0].at == 0 && same(n[0].value, magnitude(*answer)) && minus == (answer->num < 0))
            return true;
    }
    return false;
}

// The outside numbers that are not common constants: the failure labels'
// (digits, as written), then numbers the work writes in words ("Each box
// holds thirteen"), which that flag does not read. A number word is
// outside unless the question gives it (in digits or words), the work
// writes it in digits too, it is one +, -, * or / away from those, or it
// is 1, 2, 100 or a constant. Words are reported as "13 (in words)".
inline std::vector<std::string> outside_numbers(const labels::AnswerLabel& label, const Problem& p,
                                                const std::string& work) {
    using namespace labels::detail;
    std::vector<std::string> out;
    for (const std::string& n : label.outside) {
        const auto v = value_of(n);
        if (!v || !is_constant(*v)) out.push_back(n);
    }
    const std::string digits = strip_digit_commas(work);
    std::vector<calc::Rational> seen = number_words(p.question);
    for (const Number& n : numbers_in(strip_digit_commas(p.question))) seen.push_back(n.value);
    for (const Number& n : numbers_in(digits)) seen.push_back(n.value);
    for (const int64_t c : {1, 2, 100}) seen.push_back({c, 1});
    for (const calc::Rational& v : number_words(work))
        if (!contains(seen, v) && !is_constant(v) && !one_step(seen, v))
            out.push_back((v.den == 1 ? std::to_string(v.num) : std::to_string(v.num) + "/" + std::to_string(v.den)) +
                          " (in words)");
    return out;
}

// What a solution is, for telling two apart: its outline (the exact values
// of its checked calculations, in order: "48;24;32;") or, with Same::text
// or no readable calculation, its text.
inline std::string solution_key(const labels::AnswerLabel& label, const std::string& work, Same same) {
    if (same == Same::outline && !label.calcs.empty()) {
        std::string key = "outline:";
        for (const labels::Calculation& c : label.calcs) key += c.exact + ";";
        return key;
    }
    return "text:" + work;
}

// The work before the "#### " line, trimmed: what a kept try trains on.
inline std::string work_of(const std::string& text) {
    const size_t hashes = text.find("#### ");
    return trim(text.substr(0, hashes == std::string::npos ? text.size() : hashes));
}

// A right note followed by another number: "<<3+4=7>>8", or by the same
// number without its sign: "<<14-21=-7>>7" (rp1 drops the minus after the
// calculator's negative results). The calculation check reads the note, not
// the number the model copied after it. Read as that check reads the work:
// "$" and digit-grouping commas dropped.
inline bool note_miscopied(std::string work) {
    work = strip_digit_commas(work);
    work.erase(std::remove(work.begin(), work.end(), '$'), work.end());
    for (size_t at = 0; (at = work.find("<<", at)) != std::string::npos; at += 2) {
        const size_t equals = work.find('=', at), end = work.find(">>", at);
        if (equals == std::string::npos || end == std::string::npos || equals > end) continue;
        const auto value = calc::Parser(work.substr(equals + 1, end - equals - 1)).parse();  // with its sign
        const std::string_view rest = std::string_view(work).substr(end + 2);
        const bool minus = rest.starts_with('-');
        const auto after = labels::detail::numbers_in(rest.substr(minus ? 1 : 0));
        if (value && !after.empty() && after[0].at == 0 &&
            (!labels::detail::same(after[0].value, labels::detail::magnitude(*value)) || minus != (value->num < 0)))
            return true;
    }
    return false;
}

// A note whose exact value has no finite decimal, so the calculator left
// it to the model ("<<20/3=" is no call), written wrong: as a whole number
// ("<<20/3=10>>", "<<304/82=4>>": a whole number never stands for a value
// that is not whole), or as a decimal that is not the value rounded to its
// places ("<<10/3=3.34>>"; "<<10/3=3.33>>" and "<<20/3=6.7>>" are right).
// The failure labels count these notes as unreadable, not wrong. GSM8K's
// own notes never have such a value (0 of its 23,716 train notes), so a
// right try needs none; this only keeps a wrong one out. Read as the
// calculation check reads the work: "$" and digit-grouping commas dropped.
inline bool note_inexact(std::string work) {
    work = strip_digit_commas(work);
    work.erase(std::remove(work.begin(), work.end(), '$'), work.end());
    for (size_t at = 0; (at = work.find("<<", at)) != std::string::npos; at += 2) {
        const size_t equals = work.find('=', at), end = work.find(">>", at);
        if (equals == std::string::npos || end == std::string::npos || equals > end) continue;
        const std::string_view expression = std::string_view(work).substr(at + 2, equals - at - 2);
        if (calc::evaluate(expression)) continue;  // the calculator's own result: checked by the labels
        const auto exact = calc::Parser(expression).parse();
        if (!exact) continue;  // not arithmetic it can read ("5%*20", "2x+3"), or too large
        const std::string_view wrote =
            labels::detail::trim_blanks(std::string_view(work).substr(equals + 1, end - equals - 1));
        const auto value = calc::Parser(wrote).parse();
        if (!value || wrote.find_first_not_of("-.0123456789") != std::string_view::npos) continue;
        const size_t point = wrote.find('.');
        const size_t places = point == std::string_view::npos ? 0 : wrote.size() - point - 1;
        if (places == 0) return true;  // "<<20/3=10>>": a whole number for a value that is not whole
        // Rounded right: |value - exact| <= 1/2 * 10^-places, that is
        // 2 * 10^places * |num| <= den for the difference num/den. A
        // difference that does not fit, or a product that overflows, is far
        // from a rounding.
        const auto diff = calc::plus(*value, calc::Rational{-exact->num, exact->den});
        if (!diff || places > 18) return true;
        int64_t scale = 2, bound = 0;
        for (size_t i = 0; i < places; ++i) scale *= 10;
        const int64_t num = diff->num < 0 ? -diff->num : diff->num;
        if (__builtin_mul_overflow(scale, num, &bound) || bound > diff->den) return true;
    }
    return false;
}

// A calculation right up to its sign: "30 - 35 = 5", or "<<4-1=-3>>". The
// failure labels compare numbers without their sign (prose rarely writes
// one), so these pass there; training text must not state them. A plain
// result is read from its first digit, so "30 - 35 = -5" is unreadable
// there (counted, not wrong) and never reaches this check; a chain's link
// keeps its sign ("2 - 7 + 8 = -5 + 8 = 3" is right: -5 + 8 is 3).
inline bool sign_wrong(const labels::AnswerLabel& label) {
    for (const labels::Calculation& c : label.calcs)
        if (!c.exact.empty() && (c.exact[0] == '-') != c.negative) return true;
    return false;
}

// The re-draw check (docs/research-plan.md, item 3). Every calculation the
// labels checked has its left side as the calculator read it ("84+4"). Each
// number in it is traced to its sources:
//   - a number of the question with that value (the 3rd one: "q3");
//   - the result of an earlier calculation of the try with that value,
//     whose formula then takes its place (a chain "6 * 4 + 2 = 24 + 2 = 26"
//     is one calculation: a link whose left side is the previous link has
//     the previous link's formula);
//   - a number the question gives in words ("half", "a dozen"), or 1, 2 or
//     100 (the constants the failure labels allow: "/ 2" for half, "* 100"
//     for a percentage), even where the question has the same number: a
//     constant;
//   - none of these: a constant too ("60" minutes, "7" days).
//     It stays the same in every version, so a number the model worked out
//     without writing it down fails, and a real constant passes.
// Each formula is worked out exactly (calc.h) with the problem's own numbers
// and with each version's. Two sources are the same source if they give the
// same value in every version (a calculation written twice). A number with
// two different sources (two question numbers alike, a question number
// equal to an earlier result) is ambiguous: the model's text does not say
// which one it used. Each choice of sources is a reading of the try.
// The answer's sources are the calculations whose result is the answer
// (more than one is ambiguous too), or else the question numbers equal to
// it. The try passes if one of its readings gives every version's own
// answer (Options::every_reading, the default since 2026-10-17, the
// owner's decision: generator questions repeat small numbers, so a third
// to two thirds of right tries are ambiguous). With strict readings
// (--strict-readings) only a try with one reading can pass, and an
// ambiguous one is dropped (Verdict::ambiguous). Every reading is worked
// out either way, so the record says how many ambiguous tries pass.
struct Trace {
    bool ran = false;        // the try reached the check
    std::string formula;     // the answer's formula, "q1 - (q1 * q2 / 100)"; empty: none found
    bool ambiguous = false;  // a number on the answer's path has two sources
    size_t readings = 0;     // the answer's readings
    std::vector<std::string> constants;  // numbers on the formula's path from neither source
    std::vector<std::string> values;     // the formula's value on each version ("?": none)
    bool passed = false;     // the formula (one of the readings, if ambiguous) gave every version's answer
};

namespace redraw_detail {

using calc::Rational;

// One reading of a calculation: its value with the problem's own numbers
// (values[0]) and with each version's, and its formula, for the record.
struct Reading {
    std::vector<Rational> values;
    std::string formula;
    std::vector<std::string> constants;
};

struct Calc {
    Rational value;  // as the calculator found it
    bool has_value = false, readable = false, ambiguous = false;
    std::vector<Reading> readings;  // none: not followed (unreadable, or too many readings)
};

// More combinations of readings than this per calculation are not
// followed: the calculation is ambiguous with no reading.
inline constexpr size_t kMaxReadings = 4096;
// Formulas are recorded up to this length (an earlier result is copied in
// whole, so a long try could grow them without end).
inline constexpr size_t kMaxFormula = 2000;

// Constants that may stand for themselves even where the question has the
// same number: "/ 2" for half, "* 100" for a percentage, "* 1".
inline const std::vector<Rational> kUsualConstants = {{1, 1}, {2, 1}, {100, 1}};

inline bool same_value(const Rational& a, const Rational& b) { return a.num == b.num && a.den == b.den; }

inline bool same_values(const std::vector<Rational>& a, const std::vector<Rational>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!same_value(a[i], b[i])) return false;
    return true;
}

inline std::string rational_text(const Rational& v) {
    return std::to_string(v.num) + (v.den == 1 ? "" : "/" + std::to_string(v.den));
}

inline std::vector<Rational> question_numbers(const std::string& question) {
    std::vector<Rational> out;
    for (const labels::detail::Number& n : labels::detail::numbers_in(question)) out.push_back(n.value);
    return out;
}

// Adds `r` to `set` unless a reading with the same values is there.
inline void add_reading(std::vector<Reading>& set, Reading r) {
    for (const Reading& s : set)
        if (same_values(s.values, r.values)) return;
    if (r.formula.size() > kMaxFormula) r.formula = r.formula.substr(0, kMaxFormula) + "...";
    set.push_back(std::move(r));
}

// Every calculation of `label`, read: numbers[0] are the question's
// numbers, numbers[j] version j's (in the same places).
inline std::vector<Calc> read_calcs(const std::vector<std::vector<Rational>>& numbers,
                                    const std::vector<Rational>& words, const labels::AnswerLabel& label) {
    const size_t sets = numbers.size();
    std::vector<Calc> out;
    for (size_t k = 0; k < label.calcs.size(); ++k) {
        const labels::Calculation& c = label.calcs[k];
        Calc calc;
        const auto value = calc::Parser(c.exact).parse();
        calc.readable = value.has_value() && !c.expression.empty();
        if (value) calc.value = *value, calc.has_value = true;
        // A chain's link: its left side is the previous link ("24 + 2").
        if (k > 0 && calc.readable && labels::detail::whole_calculation(label.calcs[k - 1].wrote) == c.expression) {
            Calc link = out[k - 1];
            link.value = calc.value;
            out.push_back(std::move(link));
            continue;
        }
        if (!calc.readable) {
            out.push_back(std::move(calc));
            continue;
        }
        // The expression: texts between its numbers, and each number's
        // sources as readings.
        std::vector<std::string> texts;
        std::vector<std::vector<Reading>> leaves;
        size_t at = 0;
        size_t combinations = 1;
        for (const labels::detail::Number& n : labels::detail::numbers_in(c.expression)) {
            texts.push_back(c.expression.substr(at, n.at - at));
            at = n.end;
            std::vector<Reading> sources;
            for (size_t q = 0; q < numbers[0].size(); ++q) {
                if (!same_value(numbers[0][q], n.value)) continue;
                Reading r{{}, "q" + std::to_string(q + 1), {}};
                for (size_t j = 0; j < sets; ++j) r.values.push_back(q < numbers[j].size() ? numbers[j][q] : Rational{});
                add_reading(sources, r);
            }
            // An earlier result that was not followed (too many readings, or
            // an expression not read) may be this number's source: then
            // nothing is known about it, and it is no constant.
            bool lost = false;
            for (size_t e = 0; e < k; ++e) {
                if (!out[e].has_value || !same_value(out[e].value, n.value)) continue;
                calc.ambiguous |= out[e].ambiguous;
                lost |= out[e].readings.empty();
                for (const Reading& r : out[e].readings) add_reading(sources, {r.values, "(" + r.formula + ")", r.constants});
            }
            if (lost) {
                calc.ambiguous = true;
                sources.clear();
            }
            // A constant: given in words, one of the usual ones (1, 2, 100,
            // as the failure labels allow them), or from nowhere else.
            const bool in_words = labels::detail::contains(words, n.value);
            const bool usual = labels::detail::contains(kUsualConstants, n.value);
            if (!lost && (in_words || usual || sources.empty()))
                add_reading(sources, {std::vector<Rational>(sets, n.value), n.written,
                                      in_words ? std::vector<std::string>() : std::vector<std::string>{n.written}});
            if (sources.size() > 1) calc.ambiguous = true;
            combinations *= std::max<size_t>(1, sources.size());
            if (combinations > kMaxReadings) combinations = kMaxReadings + 1;
            leaves.push_back(std::move(sources));
        }
        texts.push_back(c.expression.substr(std::min(at, c.expression.size())));
        // A digit or a point left between the numbers: not read as numbers.
        for (const std::string& t : texts)
            if (t.find_first_of("0123456789.") != std::string::npos) calc.readable = false;
        bool empty = false;
        for (const auto& l : leaves) empty |= l.empty();  // an earlier result with no reading
        if (!calc.readable || empty || combinations > kMaxReadings) {
            calc.ambiguous |= combinations > kMaxReadings;
            out.push_back(std::move(calc));
            continue;
        }
        // Every combination of the leaves' readings, worked out on every set
        // of numbers.
        std::vector<size_t> pick(leaves.size(), 0);
        for (size_t combination = 0; combination < combinations; ++combination) {
            Reading r;
            r.formula = texts[0];
            for (size_t l = 0; l < leaves.size(); ++l) {
                const Reading& leaf = leaves[l][pick[l]];
                r.formula += leaf.formula + texts[l + 1];
                r.constants.insert(r.constants.end(), leaf.constants.begin(), leaf.constants.end());
            }
            bool ok = true;
            for (size_t j = 0; j < sets && ok; ++j) {
                std::string text = texts[0];
                for (size_t l = 0; l < leaves.size(); ++l)
                    text += "(" + rational_text(leaves[l][pick[l]].values[j]) + ")" + texts[l + 1];
                const auto v = calc::Parser(text).parse();
                ok = v.has_value();
                if (ok) r.values.push_back(*v);
            }
            if (ok) add_reading(calc.readings, std::move(r));
            for (size_t l = 0; l < leaves.size() && ++pick[l] == leaves[l].size(); ++l) pick[l] = 0;
        }
        out.push_back(std::move(calc));
    }
    return out;
}

}  // namespace redraw_detail

// The re-draw check of one try, whose work `label` has (labels::label_answer),
// against the versions of its problem. Every reading is tried, so an
// ambiguous try's record says whether one of its readings passes, though
// strict readings drop it.
inline Trace trace_try(const Problem& p, const labels::AnswerLabel& label, const std::vector<Problem>& versions) {
    using namespace redraw_detail;
    Trace t;
    t.ran = true;
    std::vector<std::vector<Rational>> numbers = {question_numbers(p.question)};
    std::vector<std::optional<Rational>> answers = {calc::Parser(strip_digit_commas(trim(p.answer))).parse()};
    for (const Problem& v : versions) {
        numbers.push_back(question_numbers(v.question));
        answers.push_back(calc::Parser(strip_digit_commas(trim(v.answer))).parse());
    }
    if (!answers[0]) return t;
    const std::vector<Calc> calcs = read_calcs(numbers, labels::detail::number_words(p.question), label);
    // Every calculation whose result is the answer (a price cut by 50% is
    // as much as what is left of it), or else the question numbers that are.
    std::vector<Reading> readings;
    bool found = false;
    for (size_t k = calcs.size(); k-- > 0;)
        if (calcs[k].has_value && same_value(calcs[k].value, *answers[0])) {
            found = true;
            for (const Reading& r : calcs[k].readings) add_reading(readings, r);
            t.ambiguous |= calcs[k].ambiguous;
        }
    t.ambiguous |= readings.size() > 1;
    if (!found)
        for (size_t q = 0; q < numbers[0].size(); ++q) {
            if (!same_value(numbers[0][q], *answers[0])) continue;
            Reading r{{}, "q" + std::to_string(q + 1), {}};
            for (const auto& set : numbers) r.values.push_back(q < set.size() ? set[q] : Rational{});
            add_reading(readings, r);
            t.ambiguous = readings.size() > 1;
        }
    t.readings = readings.size();
    if (readings.empty()) return t;
    // The first reading that gives every version's answer, or else the first.
    size_t chosen = 0;
    if (!versions.empty())
        for (size_t i = 0; i < readings.size() && !t.passed; ++i) {
            bool all = true;
            for (size_t j = 1; j < answers.size(); ++j)
                all = all && answers[j] && same_value(readings[i].values[j], *answers[j]);
            if (all) t.passed = true, chosen = i;
        }
    const Reading& r = readings[chosen];
    t.formula = r.formula;
    t.constants = r.constants;
    for (size_t j = 1; j < r.values.size(); ++j) t.values.push_back(rational_text(r.values[j]));
    return t;
}

// What check_try found besides the verdict.
struct TryInfo {
    std::string key;                   // solution_key, for a try that passed every check
    std::vector<std::string> outside;  // numbers from outside the question (constants aside)
    std::vector<std::string> unused;   // numbers of the question the work never writes (1 aside)
    size_t calcs = 0, unreadable = 0;  // calculations, and those the calculator could not read
    Trace trace;                       // the re-draw check's record, if it ran
};

// Every check that looks at one try alone (all but duplicate and over_cap).
// info, if given, gets the try's key, its outside numbers, how many
// calculations its work had and how many the calculator could not read
// ("10/3 = 3.33"; those are not wrong, only unchecked), and the re-draw
// check's record. versions, if given, are the problem's re-drawn versions
// (none: it could not be re-drawn), and the re-draw check runs.
inline Verdict check_try(const Problem& p, const std::string& text, const Options& o, TryInfo* info = nullptr,
                         const std::vector<Problem>* versions = nullptr) {
    if (text == kPromptTooLong) return Verdict::too_long;
    if (!completion_done(text)) return Verdict::unfinished;
    if (!answer_matches(text, p.answer)) return Verdict::wrong;
    const std::string work = work_of(text);
    if (work.empty()) return Verdict::no_work;
    if (text.find(kNoLossBegin) != std::string::npos || text.find(kNoLossEnd) != std::string::npos ||
        text.find(char(kBos)) != std::string::npos)
        return Verdict::markers;
    const labels::AnswerLabel label = labels::label_answer(p, text);
    TryInfo local;
    TryInfo& in = info ? *info : local;
    in.calcs = label.calcs.size();
    in.unreadable = label.unreadable;
    in.outside = outside_numbers(label, p, work);
    if (o.strict_outside)
        for (const std::string& n : unearned_numbers(p, work))
            if (std::find(in.outside.begin(), in.outside.end(), n) == in.outside.end()) in.outside.push_back(n);
    if (label.wrong_calcs() > 0 || sign_wrong(label) || note_miscopied(work) || note_inexact(work))
        return Verdict::arithmetic;
    if (o.check_mismatch && note_mismatch(text)) return Verdict::mismatch;
    if (o.check_broken && broken_note(text)) return Verdict::broken;
    if (label.loop && !o.allow_loops) return Verdict::loop;
    if (o.check_computed && !answer_computed(label, p, text)) return Verdict::uncomputed;
    if (o.check_dead && !dead_calculations(label, p, text).empty()) return Verdict::dead;
    if (!in.outside.empty() && !o.allow_outside) return Verdict::outside;
    in.key = solution_key(label, work, o.same);
    // A model that learned its training problems by heart (rp1 on GSM8K
    // train), or a generator's templates (rp1 on gen-math), writes their
    // solutions back; those teach it nothing new.
    if (!o.keep_gold) {
        const std::string gold = trim(p.steps);
        if (work == gold || in.key == solution_key(labels::label_answer(p, p.completion()), gold, o.same))
            return Verdict::gold_copy;
    }
    // Generated problems' distractor numbers are left out on purpose: there
    // the re-draw check below tells method from luck.
    if (!versions) {
        in.unused = unused_numbers(p, work);
        if (!in.unused.empty() && !o.allow_unused) return Verdict::unused;
    }
    if (versions) {
        in.trace = trace_try(p, label, *versions);
        const Trace& t = in.trace;
        if (t.ambiguous && (!o.every_reading || t.readings == 0)) return Verdict::ambiguous;
        if (t.readings == 0 || versions->empty()) return o.keep_unchecked ? Verdict::kept : Verdict::unchecked;
        if (!t.passed) return Verdict::redraw;
    }
    return Verdict::kept;
}

struct Result {
    std::vector<Problem> kept;  // in problem order, then try order
    // Per problem, one verdict per try (texts' layout), and each try's
    // numbers from outside the question (TryInfo::outside).
    std::vector<std::vector<Verdict>> verdicts;
    std::vector<std::vector<std::vector<std::string>>> outside;
    size_t problems = 0, tries = 0;
    size_t by_verdict[kVerdicts] = {};
    size_t first_right = 0;  // problems whose first try has the right answer
    size_t solved = 0;       // problems with a right answer in any try (pass@N)
    size_t solved_kept = 0;  // problems with at least one kept try
    size_t calcs = 0, unreadable = 0;  // calculations in the right tries' work, and those unchecked
    // The re-draw check: per problem, one record per try (texts' layout;
    // empty when the check did not run on that try).
    std::vector<std::vector<Trace>> traces;
    size_t no_recipe = 0;        // tries kept without the check: their problem has no recipe
    size_t kept_unchecked = 0;   // tries kept by --keep-unchecked
    // Tries compared with the versions with a constant on the answer's path, and those that passed.
    size_t with_constants = 0, with_constants_passed = 0;
    size_t ambiguous_passed = 0;  // ambiguous tries one of whose readings passes (dropped by strict readings)

    size_t count(Verdict v) const { return by_verdict[int(v)]; }
};

// Checks every try (texts[problem][try], as sample_tries returns them) and
// keeps the good ones. versions, if given, has one entry per problem: its
// re-drawn versions (redraw.h), or nothing if the problem has no recipe
// (then its tries are kept without the re-draw check).
inline Result keep_checked(const std::vector<Problem>& problems, const std::vector<std::vector<std::string>>& texts,
                           const Options& o,
                           const std::vector<std::optional<std::vector<Problem>>>* versions = nullptr) {
    CHECK(texts.size() == problems.size(), "%zu problems but tries for %zu", problems.size(), texts.size());
    CHECK(!versions || versions->size() == problems.size(), "%zu problems but versions for %zu", problems.size(),
          versions->size());
    CHECK(o.keep >= 0, "--keep must not be negative");
    // With recipes the problems are generated, and generated problems add
    // numbers on purpose (a distractor sentence): no try need use them all,
    // not even one of a problem without a recipe of its own.
    Options checks = o;
    if (versions) checks.allow_unused = true;
    Result r;
    r.problems = problems.size();
    r.verdicts.resize(problems.size());
    r.traces.resize(problems.size());
    for (size_t i = 0; i < problems.size(); ++i) {
        const Problem& p = problems[i];
        std::set<std::string> seen;  // the keys of this problem's kept tries
        bool right = false;
        r.outside.emplace_back();
        for (size_t t = 0; t < texts[i].size(); ++t) {
            const std::string& text = texts[i][t];
            const std::vector<Problem>* redrawn = versions && (*versions)[i] ? &*(*versions)[i] : nullptr;
            TryInfo info;
            Verdict v = check_try(p, text, checks, &info, redrawn);
            r.calcs += info.calcs;
            r.unreadable += info.unreadable;
            r.outside.back().push_back(info.outside);
            const Trace& trace = info.trace;
            if (v == Verdict::kept && versions && !redrawn) ++r.no_recipe;
            if (v == Verdict::kept && trace.ran && (trace.readings == 0 || redrawn->empty())) ++r.kept_unchecked;
            const bool compared = trace.ran && trace.readings > 0 && !redrawn->empty();
            if (compared && (!trace.ambiguous || o.every_reading) && !trace.constants.empty()) {
                ++r.with_constants;
                r.with_constants_passed += trace.passed;
            }
            r.ambiguous_passed += compared && trace.ambiguous && trace.passed;
            r.traces[i].push_back(trace);
            const bool answer_right = v != Verdict::too_long && v != Verdict::unfinished && v != Verdict::wrong;
            right |= answer_right;
            if (t == 0 && answer_right) ++r.first_right;
            if (v == Verdict::kept) {
                if (seen.count(info.key)) v = Verdict::duplicate;
                else if (o.keep > 0 && seen.size() >= size_t(o.keep)) v = Verdict::over_cap;
                else {
                    seen.insert(info.key);
                    Problem k = p;
                    k.steps = work_of(text);  // the answer stays the problem file's
                    r.kept.push_back(std::move(k));
                }
            }
            r.verdicts[i].push_back(v);
            ++r.by_verdict[int(v)];
            ++r.tries;
        }
        r.solved += right;
        r.solved_kept += !seen.empty();
    }
    return r;
}

// The kept solutions as OUT.txt (a problems file: eval-reasoning,
// leak-check and a person can read it) and OUT.tok (prep-qa's masked
// format, ready for train --data).
inline void write(const std::string& out, const std::vector<Problem>& kept, const Tokenizer& tok) {
    save_problems(out + ".txt", kept);
    TokenData d;
    d.vocab = uint32_t(tok.vocab_size());
    for (const Problem& p : kept) append_problem(d, p, tok);
    d.save(out + ".tok");
}

}  // namespace self_train
