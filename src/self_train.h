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
//     way failure_labels.h finds them, and the number written right after
//     a note is the note's. A right answer reached through a wrong
//     calculation is a lucky answer, and would teach the mistake;
//   - it does not repeat a line of 12 or more characters (the failure
//     labels' loop flag), unless loops are allowed.
// Of the tries left, an exact copy of an earlier try's work is dropped, and
// at most `keep` per problem are kept (0: all), in try order. Everything is
// decided in a fixed order, so the same seed gives the same file.
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
    duplicate,    // the same work as an earlier kept try of the problem
    over_cap,     // the problem already has `keep` kept tries
};
inline constexpr int kVerdicts = 10;

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
        case Verdict::duplicate: return "duplicate";
        case Verdict::over_cap: return "over_cap";
    }
    return "?";
}

struct Options {
    int keep = 0;             // kept tries per problem at most; 0: all
    bool allow_loops = false; // keep right tries that repeat a line
};

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

// Every check that looks at one try alone (all but duplicate and over_cap).
// calcs, if given, gets how many calculations the try's work had and how
// many the calculator could not read ("10/3 = 3.33"; those are not wrong,
// only unchecked).
inline Verdict check_try(const Problem& p, const std::string& text, const Options& o,
                         size_t* calcs = nullptr, size_t* unreadable = nullptr) {
    if (text == kPromptTooLong) return Verdict::too_long;
    if (!completion_done(text)) return Verdict::unfinished;
    if (!answer_matches(text, p.answer)) return Verdict::wrong;
    const std::string work = work_of(text);
    if (work.empty()) return Verdict::no_work;
    if (text.find(kNoLossBegin) != std::string::npos || text.find(kNoLossEnd) != std::string::npos ||
        text.find(char(kBos)) != std::string::npos)
        return Verdict::markers;
    const labels::AnswerLabel label = labels::label_answer(p, text);
    if (calcs) *calcs += label.calcs.size();
    if (unreadable) *unreadable += label.unreadable;
    if (label.wrong_calcs() > 0 || note_miscopied(work)) return Verdict::arithmetic;
    if (label.loop && !o.allow_loops) return Verdict::loop;
    return Verdict::kept;
}

struct Result {
    std::vector<Problem> kept;  // in problem order, then try order
    // Per problem, one verdict per try (texts' layout).
    std::vector<std::vector<Verdict>> verdicts;
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
        std::set<std::string> seen;  // the work of this problem's kept tries
        bool right = false;
        for (size_t t = 0; t < texts[i].size(); ++t) {
            const std::string& text = texts[i][t];
            Verdict v = check_try(p, text, o, &r.calcs, &r.unreadable);
            const bool answer_right = v != Verdict::too_long && v != Verdict::unfinished && v != Verdict::wrong;
            right |= answer_right;
            if (t == 0 && answer_right) ++r.first_right;
            if (v == Verdict::kept) {
                const std::string work = work_of(text);
                if (seen.count(work)) v = Verdict::duplicate;
                else if (o.keep > 0 && seen.size() >= size_t(o.keep)) v = Verdict::over_cap;
                else {
                    seen.insert(work);
                    Problem k = p;
                    k.steps = work;  // the answer stays the problem file's
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
