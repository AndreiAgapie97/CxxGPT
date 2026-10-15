// test_vote.h — the many-tries vote: a scripted fake model returns fixed
// answer sequences, and the reported single/vote/any accuracies and distinct
// counts are checked against numbers computed by hand from those sequences.
// A real tiny model with one try must reproduce a plain greedy decode.
#pragma once

#include <map>
#include <set>

#include "../src/generate.h"
#include "../src/model.h"
#include "test_util.h"

namespace {

// Plays one fixed completion text per sequence slot, one byte per generated
// token, whatever the temperature: the desired token's logit towers over the
// rest, so sampling it is certain. Once a script runs out it repeats '.',
// so a completion without an "#### " line simply fills its token budget.
class ScriptedModel {
public:
    Config cfg;
    std::vector<std::string> queue;  // scripts, consumed by decode_begin
    int prompt_len = 0;              // every prompt has this many tokens

    ScriptedModel() {
        cfg.vocab = 256;  // byte-level
        cfg.seq_len = 128;
    }
    void decode_begin(int B, const std::vector<int>& start) {
        CHECK(int(start.size()) == B, "decode_begin: %zu starts for %d sequences", start.size(), B);
        CHECK(size_t(served_ + B) <= queue.size(), "scripted model ran out of scripts");
        scripts_.assign(queue.begin() + served_, queue.begin() + served_ + B);
        served_ += B;
        B_ = B;
        logits_.assign(size_t(B) * cfg.vocab, 0.0f);
    }
    const float* decode_step(const int*, int pos) {
        for (int b = 0; b < B_; ++b) {
            const size_t at = size_t(b) * cfg.vocab;
            for (int v = 0; v < cfg.vocab; ++v) logits_[at + v] = 0.0f;
            const size_t index = size_t(pos - (prompt_len - 1));
            const int token = index < scripts_[b].size() ? int(scripts_[b][index]) : int('.');
            logits_[at + token] = 1e6f;
        }
        return logits_.data();
    }

private:
    int B_ = 0;
    size_t served_ = 0;
    std::vector<std::string> scripts_;
    std::vector<float> logits_;
};

Problem vote_problem(const char* task, int difficulty, const std::string& question, const char* answer) {
    Problem p;
    p.task = task;
    p.difficulty = difficulty;
    p.question = question;  // all questions the same length, so batching order is input order
    p.steps = "work";
    p.answer = answer;
    return p;
}

// A completion with the given final answer, or without an answer line at all.
std::string vote_script(const char* answer) {
    return std::string("work\n#### ") + answer + "\n";
}

inline void test_vote_answer() {
    // The vote itself: most common answer, ties to the one that appeared first.
    EXPECT(vote_answer({"7", "3", "7", "7"}) == "7", "plain majority lost");
    EXPECT(vote_answer({"4", "9", "4", "4"}) == "4", "majority after a wrong first try lost");
    EXPECT(vote_answer({"8", "3", "8", "3"}) == "8", "a tie must go to the first-seen answer");
    EXPECT(vote_answer({"5", "5", "12", "5"}) == "5", "counts, not the last answer, decide");
    // An empty answer here is a finished but empty "#### " line: one vote
    // like any other. (Unfinished tries never reach vote_answer.)
    EXPECT(vote_answer({"", "6", "6", "6"}) == "6", "one empty answer line must lose to three 6s");
    EXPECT(vote_answer({}) == "", "no tries gave no answer");
    EXPECT(vote_answer({"", ""}) == "", "two empty answer lines vote for the empty answer");
}

// score_tries directly: unfinished tries abstain, a problem without a
// finished try is wrong, and one try scores exactly as the plain check.
inline void test_score_tries() {
    const std::string unfinished = "work, but no answer line";
    auto answer = [](const char* a) { return std::string("work\n#### ") + a + "\n"; };
    // Gold 7, tries "", "", 7, 7: the two answers both say 7, so 7 wins.
    TryScore s = score_tries({unfinished, unfinished, answer("7"), answer("7")}, "7");
    EXPECT(s.vote_ok && s.any_ok && !s.first_ok && s.distinct == 1 && s.shown == 2,
           "unfinished, unfinished, 7, 7 vs gold 7: vote %d any %d first %d distinct %d shown %zu", s.vote_ok,
           s.any_ok, s.first_ok, s.distinct, s.shown);
    // Gold 7, tries "", "", "", 7: the only answer is 7.
    s = score_tries({unfinished, unfinished, unfinished, answer("7")}, "7");
    EXPECT(s.vote_ok && s.distinct == 1, "three unfinished and one 7 vs gold 7 must be right (vote %d)", s.vote_ok);
    // Gold "", three unfinished and one "#### 5": the only answer is 5.
    s = score_tries({unfinished, unfinished, unfinished, answer("5")}, "");
    EXPECT(!s.vote_ok && !s.any_ok && s.distinct == 1 && s.shown == 3,
           "three unfinished and one 5 vs gold \"\" must be wrong (vote %d any %d)", s.vote_ok, s.any_ok);
    // No try finished: wrong even against gold "", no answers, first try shown.
    s = score_tries({unfinished, unfinished}, "");
    EXPECT(!s.vote_ok && !s.any_ok && s.distinct == 0 && s.shown == 0,
           "no finished try must be wrong with no answers (vote %d distinct %d)", s.vote_ok, s.distinct);
    // A finished empty answer line against gold "" is right.
    s = score_tries({unfinished, answer(""), answer("")}, "");
    EXPECT(s.vote_ok && s.distinct == 1, "finished empty answers vs gold \"\" must be right (vote %d)", s.vote_ok);
    // The vote loses: the shown try is the first one that gave the voted answer.
    s = score_tries({unfinished, answer("3"), answer("5"), answer("5")}, "3");
    EXPECT(!s.vote_ok && s.any_ok && s.distinct == 2 && s.shown == 2,
           "3, 5, 5 vs gold 3: vote %d any %d distinct %d shown %zu", s.vote_ok, s.any_ok, s.distinct, s.shown);
    // One try: all three scores equal the plain check, on every kind of text.
    const std::vector<std::pair<std::string, std::string>> single = {
        {answer("7"), "7"}, {answer("8"), "7"}, {unfinished, "7"}, {answer(""), ""},
        {unfinished, ""},   {"#### 7", "7"},    {answer("1,080"), "1080"},
    };
    for (const auto& [text, gold] : single) {
        const bool plain = completion_done(text) && answer_matches(text, gold);
        s = score_tries({text}, gold);
        EXPECT(s.first_ok == plain && s.vote_ok == plain && s.any_ok == plain,
               "one try '%s' vs gold '%s': first %d vote %d any %d, plain %d", text.c_str(), gold.c_str(),
               s.first_ok, s.vote_ok, s.any_ok, plain);
    }
}

inline void test_vote_report() {
    // Five problems, four tries each. The scripts are fixed, so every number
    // below was computed by hand from them:
    //   gold 7 : tries 7, 7, 3, 7   -> single ok, vote 7 ok, any ok, 2 distinct
    //   gold 9 : tries 9, 4, 4, 4   -> single ok, vote 4 no, any ok, 2 distinct
    //   gold 12: tries 5, 5, 12, 5  -> single no, vote 5 no, any ok, 2 distinct
    //   gold 3 : tries 8, 3, 8, 3   -> tie 2-2, vote 8 (first seen) no, any ok
    //   gold 6 : tries -, 6, 6, 6   -> single no, vote 6 ok, any ok, 1 distinct
    //            (the first try writes no answer line: it abstains)
    // Overall: single 2/5, vote 2/5, any 5/5, distinct answers 9 (mean 1.8).
    // Per level: add d1 1/2 by vote, add d2 0/1, mul d1 0/1, mul d2 1/1.
    std::vector<Problem> ps = {
        vote_problem("add", 1, "12+45?", "7"),
        vote_problem("add", 1, "34+50?", "9"),
        vote_problem("mul", 1, "11x2+1", "12"),
        vote_problem("add", 2, "40-37?", "3"),
        vote_problem("mul", 2, "2x3+0?", "6"),
    };
    ScriptedModel fake;
    for (const char* t : {"7", "7", "3", "7"}) fake.queue.push_back(vote_script(t));
    for (const char* t : {"9", "4", "4", "4"}) fake.queue.push_back(vote_script(t));
    for (const char* t : {"5", "5", "12", "5"}) fake.queue.push_back(vote_script(t));
    for (const char* t : {"8", "3", "8", "3"}) fake.queue.push_back(vote_script(t));
    fake.queue.push_back("no answer line here\n");
    for (const char* t : {"6", "6", "6"}) fake.queue.push_back(vote_script(t));
    fake.prompt_len = 1 + int(ps[0].prompt().size());
    VoteOpts vote;
    vote.tries = 4;
    vote.temperature = 1.0;
    vote.seed = 5;
    const EvalResult r = eval_reasoning(fake, ps, Tokenizer{}, 4, 40, 8, false, vote);
    const LevelScore all = r.overall();
    EXPECT(r.total == 5 && all.total == 5 && r.tries == 4, "the vote report scored %d problems at %d tries",
           r.total, r.tries);
    EXPECT(all.first_correct == 2, "first-try accuracy %d, hand-computed 2", all.first_correct);
    EXPECT(all.vote_correct == 2, "vote accuracy %d, hand-computed 2", all.vote_correct);
    EXPECT(all.any_correct == 5, "any-try accuracy %d, hand-computed 5", all.any_correct);
    EXPECT(r.correct == all.vote_correct, "the headline score must be the vote's");
    EXPECT(all.distinct_answers == 9, "distinct answers %d, hand-computed 9", all.distinct_answers);
    const LevelScore& add1 = r.by_task.at({"add", 1});
    EXPECT(add1.total == 2 && add1.vote_correct == 1 && add1.first_correct == 2 && add1.any_correct == 2,
           "add d1 level: first %d vote %d any %d", add1.first_correct, add1.vote_correct, add1.any_correct);
    const LevelScore& add2 = r.by_task.at({"add", 2});
    EXPECT(add2.vote_correct == 0 && add2.any_correct == 1, "add d2 (the tie) must lose the vote but pass on some try");
    const LevelScore& mul2 = r.by_task.at({"mul", 2});
    EXPECT(mul2.vote_correct == 1 && mul2.first_correct == 0 && mul2.distinct_answers == 1,
           "mul d2 (the missing answer line) must win the vote but not the first try");
    EXPECT(r.failures.size() == 3, "%zu failures shown, but 3 problems lost the vote", r.failures.size());
    if (r.failures.size() == 3) {
        // The vote-losers, in problem order, with the completion each must
        // show: the first try that gave the voted answer.
        const std::vector<std::pair<const char*, const char*>> want = {
            {"34+50?", "work\n#### 4\n"},    // gold 9, vote 4, first 4 is try 1
            {"11x2+1", "work\n#### 5\n"},    // gold 12, vote 5, first 5 is try 0
            {"40-37?", "work\n#### 8\n"},    // gold 3, tie 8/3, first 8 is try 0
        };
        for (size_t f = 0; f < r.failures.size(); ++f) {
            EXPECT(r.failures[f].first.question == want[f].first,
                   "failure %zu shows problem '%s', expected '%s'", f, r.failures[f].first.question.c_str(),
                   want[f].first);
            EXPECT(r.failures[f].second == want[f].second, "failure %zu shows '%s', expected '%s'", f,
                   r.failures[f].second.c_str(), want[f].second);
        }
    }
    // (The scripts ignore randomness, so seeds are tested on a real model in
    // test_vote_seeding.)
}

// One try must reproduce exactly what eval-reasoning scored before votes
// existed: decode each problem greedily here, by hand, and compare.
// An empty gold answer is correct when a try finishes an empty "#### " line
// (today's scoring), and only then: unfinished tries never count.
inline void test_vote_empty_answer() {
    std::vector<Problem> ps = {vote_problem("add", 1, "40-40?", "")};
    ScriptedModel finished;
    finished.queue = {"work\n#### \n", "work\n#### \n", "work\n#### 5\n", "work\n#### \n"};
    finished.prompt_len = 1 + int(ps[0].prompt().size());
    VoteOpts vote;
    vote.tries = 4;
    vote.temperature = 1.0;
    vote.seed = 6;
    const EvalResult r = eval_reasoning(finished, ps, Tokenizer{}, 4, 40, 8, false, vote);
    EXPECT(r.correct == 1, "a finished empty answer line must match an empty gold (vote %d)", r.correct);
    // The same problem, but no try ever finishes: still wrong.
    ScriptedModel unfinished;
    unfinished.queue = {"no answer\n", "no answer\n", "no answer\n", "no answer\n"};
    unfinished.prompt_len = finished.prompt_len;
    const EvalResult u = eval_reasoning(unfinished, ps, Tokenizer{}, 4, 40, 8, false, vote);
    EXPECT(u.correct == 0, "an unfinished try must not give the empty answer (vote %d)", u.correct);
}

inline void test_vote_single_try_matches_greedy() {
    Config cfg;
    cfg.vocab = 256;
    cfg.seq_len = 96;
    cfg.d_model = 24;
    cfg.n_layers = 2;
    cfg.n_heads = 2;
    Model<float> m(cfg);
    m.init(21);
    Tokenizer tok;
    Rng rng(3);
    std::vector<Problem> ps;
    for (int i = 0; i < 5; ++i) ps.push_back(make_problem(i % 2 ? "add" : "chain", 1 + i % 2, rng));
    const int max_new = 30;
    const EvalResult r = eval_reasoning(m, ps, tok, 2, max_new, 8, true);
    const LevelScore all = r.overall();
    EXPECT(r.tries == 1 && all.first_correct == all.vote_correct && all.vote_correct == all.any_correct &&
               r.correct == all.vote_correct,
           "with one try all four accuracies must agree");
    for (size_t i = 0; i < ps.size(); ++i) {
        // Independent reference: a fresh greedy decode, one problem at a time.
        m.decode_begin(1, {0});
        const std::vector<int> prompt = prompt_tokens(ps[i], tok);
        std::string text;
        for (int pos = 0; pos < int(prompt.size()) - 1; ++pos) m.decode_step(&prompt[pos], pos);
        std::vector<int> cur = {prompt.back()};
        const int budget = std::min(max_new, cfg.seq_len - int(prompt.size()) + 1);
        for (int pos = int(prompt.size()) - 1; pos < cfg.seq_len; ++pos) {
            const float* logits = m.decode_step(cur.data(), pos);
            int best = 0;
            for (int v = 1; v < cfg.vocab; ++v)
                if (logits[v] > logits[best]) best = v;
            text += char(best);
            if (completion_done(text) || int(text.size()) == budget) break;
            cur[0] = best;
        }
        EXPECT(text == r.completions[i], "try 1 differs from a plain greedy decode on problem %zu", i);
        EXPECT(answer_matches(r.completions[i], ps[i].answer) == answer_matches(text, ps[i].answer),
               "problem %zu scored differently from the reference decode", i);
    }
}

// Each try is seeded from (seed, problem, try), so on a real model sampling
// at temperature 1, a try's text does not depend on --batch, on which other
// problems are evaluated with it (--limit), or on the number of tries; and
// another seed gives other texts.
inline void test_vote_seeding() {
    Config cfg;
    cfg.vocab = 256;
    cfg.seq_len = 128;  // room for 20 new tokens after any of these prompts
    cfg.d_model = 24;
    cfg.n_layers = 2;
    cfg.n_heads = 2;
    Model<float> m(cfg);
    m.init(21);
    Rng rng(4);
    std::vector<Problem> ps;
    for (int i = 0; i < 6; ++i) ps.push_back(make_problem(i % 2 ? "add" : "chain", 1 + i % 2, rng));
    // The first try's completion of every problem.
    auto first_tries = [&](const std::vector<Problem>& list, int batch, int tries, uint64_t seed,
                           std::vector<size_t> ids) {
        VoteOpts v;
        v.tries = tries;
        v.temperature = 1.0;
        v.seed = seed;
        v.problem_ids = std::move(ids);
        return eval_reasoning(m, list, Tokenizer{}, batch, 20, 0, true, v).completions;
    };
    const auto base = first_tries(ps, 8, 2, 1, {});  // a batch of 8 holds 4 problems x 2 tries
    EXPECT(base.size() == ps.size(), "%zu completions for %zu problems", base.size(), ps.size());
    EXPECT(first_tries(ps, 2, 2, 1, {}) == base, "a try's text depends on --batch");
    EXPECT(first_tries(ps, 8, 4, 1, {}) == base, "the first try changed with the number of tries");
    // As --limit runs it: problems 1 and 4 alone, numbered by their place in
    // the full list, sample exactly what they sampled there.
    const auto some = first_tries({ps[1], ps[4]}, 8, 2, 1, {1, 4});
    EXPECT(some.size() == 2 && some[0] == base[1] && some[1] == base[4],
           "a problem's tries changed when other problems were left out");
    // Another seed samples other texts.
    const auto other = first_tries(ps, 8, 2, 2, {});
    size_t differ = 0;
    for (size_t i = 0; i < ps.size() && i < other.size(); ++i) differ += other[i] != base[i];
    EXPECT(differ > 0, "seeds 1 and 2 gave the same completions for all %zu problems", ps.size());
    // The seeds of nearby (seed, problem, try) are all different.
    std::set<uint64_t> seeds;
    for (uint64_t s = 0; s < 3; ++s)
        for (uint64_t p = 0; p < 3; ++p)
            for (uint64_t t = 0; t < 3; ++t) seeds.insert(try_seed(s, p, t));
    EXPECT(seeds.size() == 27, "only %zu distinct seeds for 27 (seed, problem, try)", seeds.size());
}

// Every score is printed with its effort.
inline void test_effort_label() {
    VoteOpts v;
    EXPECT(effort_label(v) == "greedy", "one try at temperature 0 is labelled '%s'", effort_label(v).c_str());
    v.temperature = 0.8;
    EXPECT(effort_label(v) == "1 sampled try, temp 0.8, seed 1", "one sampled try is labelled '%s'",
           effort_label(v).c_str());
    v.tries = 8;
    v.temperature = 0.7;
    v.seed = 3;
    EXPECT(effort_label(v) == "maj@8, temp 0.7, seed 3", "eight tries are labelled '%s'", effort_label(v).c_str());
    std::vector<Problem> ps = {vote_problem("add", 1, "1+1=?", "2")};
    ScriptedModel fake;
    fake.queue = {vote_script("2"), vote_script("2")};
    fake.prompt_len = 1 + int(ps[0].prompt().size());
    v.tries = 2;
    EXPECT(eval_reasoning(fake, ps, Tokenizer{}, 2, 40, 8, false, v).effort == "maj@2, temp 0.7, seed 3",
           "the evaluation does not carry its effort label");
}

inline void test_vote() {
    test_vote_answer();
    test_effort_label();
    test_score_tries();
    test_vote_report();
    test_vote_empty_answer();
    test_vote_single_try_matches_greedy();
    test_vote_seeding();
}

}  // namespace
