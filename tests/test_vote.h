// test_vote.h — the many-tries vote: a scripted fake model returns fixed
// answer sequences, and the reported single/vote/any accuracies and distinct
// counts are checked against numbers computed by hand from those sequences.
// A real tiny model with one try must reproduce a plain greedy decode.
#pragma once

#include <map>

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
    EXPECT(vote_answer({"", "6", "6", "6"}) == "6", "a missing answer line counts as empty and loses");
    EXPECT(vote_answer({}) == "", "no tries gave no answer");
    EXPECT(vote_answer({"", ""}) == "", "all-missing answers vote for the empty answer");
}

inline void test_vote_report() {
    // Five problems, four tries each. The scripts are fixed, so every number
    // below was computed by hand from them:
    //   gold 7 : tries 7, 7, 3, 7   -> single ok, vote 7 ok, any ok, 2 distinct
    //   gold 9 : tries 9, 4, 4, 4   -> single ok, vote 4 no, any ok, 2 distinct
    //   gold 12: tries 5, 5, 12, 5  -> single no, vote 5 no, any ok, 2 distinct
    //   gold 3 : tries 8, 3, 8, 3   -> tie 2-2, vote 8 (first seen) no, any ok
    //   gold 6 : tries "", 6, 6, 6  -> single no, vote 6 ok, any ok, 2 distinct
    // Overall: single 2/5, vote 2/5, any 5/5, distinct answers 10 (mean 2.0).
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
    Rng rng(5);
    vote.rng = &rng;
    const EvalResult r = eval_reasoning(fake, ps, Tokenizer{}, 4, 40, 8, false, vote);
    EXPECT(r.total == 5 && r.tries == 4, "the vote report scored %d problems at %d tries", r.total, r.tries);
    EXPECT(r.single_correct == 2, "single-try accuracy %d, hand-computed 2", r.single_correct);
    EXPECT(r.vote_correct == 2, "vote accuracy %d, hand-computed 2", r.vote_correct);
    EXPECT(r.any_correct == 5, "any-try accuracy %d, hand-computed 5", r.any_correct);
    EXPECT(r.correct == r.vote_correct, "the headline score must be the vote's");
    EXPECT(r.distinct_answers == 10.0, "distinct answers %.1f, hand-computed 10", r.distinct_answers);
    const VoteCell& add1 = r.vote_by_task.at({"add", 1});
    EXPECT(add1.vote_correct == 1 && add1.single_correct == 2 && add1.any_correct == 2,
           "add d1 cell: single %d vote %d any %d", add1.single_correct, add1.vote_correct, add1.any_correct);
    const VoteCell& add2 = r.vote_by_task.at({"add", 2});
    EXPECT(add2.vote_correct == 0 && add2.any_correct == 1, "add d2 (the tie) must lose the vote but pass on some try");
    const VoteCell& mul2 = r.vote_by_task.at({"mul", 2});
    EXPECT(mul2.vote_correct == 1 && mul2.single_correct == 0,
           "mul d2 (the missing answer line) must win the vote but not the first try");
    EXPECT(r.failures.size() == 3, "%zu failures shown, but 3 problems lost the vote", r.failures.size());
    if (r.failures.size() == 3) {
        // Each failure shows a completion that really gave the voted answer.
        for (const auto& [p, text] : r.failures)
            EXPECT(normalized_answer(text) == vote_answer({normalized_answer(text)}),
                   "failure text '%s' does not match its own answer", text.c_str());
    }
    // Seeded: the same seed replays the same scripts and the same report.
    ScriptedModel again;
    again.queue = fake.queue;
    again.prompt_len = fake.prompt_len;
    Rng rng2(5);
    VoteOpts same = vote;
    same.rng = &rng2;
    const EvalResult r2 = eval_reasoning(again, ps, Tokenizer{}, 4, 40, 8, false, same);
    EXPECT(r2.single_correct == r.single_correct && r2.vote_correct == r.vote_correct &&
               r2.any_correct == r.any_correct && r2.distinct_answers == r.distinct_answers,
           "the same seed gave a different report");
}

// One try must reproduce exactly what eval-reasoning scored before votes
// existed: decode each problem greedily here, by hand, and compare.
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
    EXPECT(r.tries == 1 && r.single_correct == r.vote_correct && r.vote_correct == r.any_correct &&
               r.correct == r.vote_correct,
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

inline void test_vote() {
    test_vote_answer();
    test_vote_report();
    test_vote_single_try_matches_greedy();
}

}  // namespace
