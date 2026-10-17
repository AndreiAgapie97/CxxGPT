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
//     written right after a note is the note's. A right answer reached through a wrong
//     calculation is a lucky answer, and would teach the mistake;
//   - it does not repeat a line of 12 or more characters (the failure
//     labels' loop flag), unless loops are allowed;
//   - it uses no number from outside the question (the failure labels'
//     outside flag: not in the question, not a result after "=", not one
//     +, -, * or / away from numbers already seen), other than a few
//     common constants (kConstants: 7 days a week, 60 minutes an hour,
//     ...), unless outside numbers are allowed: an invented quantity is a
//     wrong plan even when the answer comes out right;
//   - it is not the problem file's own solution: a model that has learned
//     its training problems, or a generator's templates, by heart writes
//     those back, and they teach it nothing new.
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
    gold_copy,    // the problem file's own solution (outline or text): nothing new
    duplicate,    // the same solution as an earlier kept try of the problem
    over_cap,     // the problem already has `keep` kept tries
};
inline constexpr int kVerdicts = 12;

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
        case Verdict::gold_copy: return "gold_copy";
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
    bool keep_gold = false;     // keep tries that are the problem's own solution
    Same same = Same::outline;
};

// Numbers a solution may bring in without the question giving them: days
// in a week, months and hours, minutes and seconds, weeks and days in a
// year, and metric units. (The failure labels already allow 1, 2 and 100.)
inline constexpr int64_t kConstants[] = {7, 12, 24, 52, 60, 365, 1000};

inline bool is_constant(const calc::Rational& v) {
    return v.den == 1 && std::find(std::begin(kConstants), std::end(kConstants), v.num) != std::end(kConstants);
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

// A right note followed by another number: "<<3+4=7>>8". The calculation
// check reads the note, not the number the model copied after it. Read as
// that check reads the work: "$" and digit-grouping commas dropped.
inline bool note_miscopied(std::string work) {
    work = strip_digit_commas(work);
    work.erase(std::remove(work.begin(), work.end(), '$'), work.end());
    for (size_t at = 0; (at = work.find("<<", at)) != std::string::npos; at += 2) {
        const size_t equals = work.find('=', at), end = work.find(">>", at);
        if (equals == std::string::npos || end == std::string::npos || equals > end) continue;
        const auto value = labels::detail::value_of(work.substr(equals + 1, end - equals - 1));
        const auto after = labels::detail::numbers_in(work.substr(end + 2));
        if (value && !after.empty() && after[0].at == 0 && !labels::detail::same(after[0].value, *value))
            return true;
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

// What check_try found besides the verdict.
struct TryInfo {
    std::string key;                   // solution_key, for a try that passed every check
    std::vector<std::string> outside;  // numbers from outside the question (constants aside)
    size_t calcs = 0, unreadable = 0;  // calculations, and those the calculator could not read
};

// Every check that looks at one try alone (all but duplicate and over_cap).
// info, if given, gets the try's key, its outside numbers, and how many
// calculations its work had and how many the calculator could not read
// ("10/3 = 3.33"; those are not wrong, only unchecked).
inline Verdict check_try(const Problem& p, const std::string& text, const Options& o, TryInfo* info = nullptr) {
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
    if (label.wrong_calcs() > 0 || sign_wrong(label) || note_miscopied(work)) return Verdict::arithmetic;
    if (label.loop && !o.allow_loops) return Verdict::loop;
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

    size_t count(Verdict v) const { return by_verdict[int(v)]; }
};

// Checks every try (texts[problem][try], as sample_tries returns them) and
// keeps the good ones.
inline Result keep_checked(const std::vector<Problem>& problems, const std::vector<std::vector<std::string>>& texts,
                     const Options& o) {
    CHECK(texts.size() == problems.size(), "%zu problems but tries for %zu", problems.size(), texts.size());
    CHECK(o.keep >= 0, "--keep must not be negative");
    Result r;
    r.problems = problems.size();
    r.verdicts.resize(problems.size());
    for (size_t i = 0; i < problems.size(); ++i) {
        const Problem& p = problems[i];
        std::set<std::string> seen;  // the keys of this problem's kept tries
        bool right = false;
        r.outside.emplace_back();
        for (size_t t = 0; t < texts[i].size(); ++t) {
            const std::string& text = texts[i][t];
            TryInfo info;
            Verdict v = check_try(p, text, o, &info);
            r.calcs += info.calcs;
            r.unreadable += info.unreadable;
            r.outside.back().push_back(info.outside);
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
