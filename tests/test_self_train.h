// test_self_train.h — self-train's host logic (self_train.h): which tries
// are kept (the answer check, the calculation check on notes and plain
// arithmetic, loops, markers), dedupe and --keep, and the files written,
// end to end from the scripted fake model of test_vote.h.
#pragma once

#include "../src/self_train.h"
#include "test_util.h"
#include "test_vote.h"

namespace {

Problem self_train_problem(const std::string& question, const char* answer) {
    Problem p;
    p.task = "qa";
    p.difficulty = 2;
    p.question = question;
    p.steps = "gold work";
    p.answer = answer;
    return p;
}

inline void test_self_train_check() {
    using self_train::Verdict;
    const Problem p = self_train_problem("What is 3 plus 4?", "7");
    // These cases test the arithmetic, so numbers from outside the question
    // ("28 - 35") are allowed here; test_self_train_outside tests that check.
    self_train::Options o;
    o.allow_outside = true;
    auto verdict = [&](const std::string& text, const self_train::Options& opts) {
        return self_train::check_try(p, text, opts);
    };
    const std::pair<std::string, Verdict> cases[] = {
        {"3 + 4 = 7\n#### 7\n", Verdict::kept},
        {"3 + 4 = <<3+4=7>>7 apples.\n#### 7\n", Verdict::kept},   // a right note
        {"3 + 4 = 8, so 7.\n#### 7\n", Verdict::arithmetic},        // right answer, wrong plain step
        {"3 + 4 = <<3+4=8>>8\n#### 7\n", Verdict::arithmetic},     // right answer, wrong note
        {"3 + 4 = <<3+4=7>>8\n#### 7\n", Verdict::arithmetic},     // right note, another number after it
        {"3 + 4 = $<<3+4=7>>7.00\n#### 7\n", Verdict::kept},        // the same number, written another way
        {"35 - 28 = 7\n#### 7\n", Verdict::kept},
        {"28 - 35 = 7\n#### 7\n", Verdict::arithmetic},            // the sign is wrong (rp1 wrote one)
        {"<<28-35=7>>7, so 7\n#### 7\n", Verdict::arithmetic},     // in a note too
        {"<<4-1=-3>>3, so 3 + 4 = 7\n#### 7\n", Verdict::arithmetic},
        {"28 - 35 = -7, so 3 + 4 = 7\n#### 7\n", Verdict::kept},   // a negative result: unreadable, not wrong
        {"<<28-35=-7>>-7 and 3 + 4 = 7\n#### 7\n", Verdict::kept},
        {"#### 9\n", Verdict::wrong},
        {"3 + 4 = 9\n#### 9\n", Verdict::wrong},                    // wrong answer comes before arithmetic
        {"#### 7\n", Verdict::no_work},
        {"   \n#### 7\n", Verdict::no_work},
        {"3 + 4 = 7 and on and on", Verdict::unfinished},          // no answer line
        {"3 + 4 = 7\n#### 7", Verdict::unfinished},                // the answer line never ended
        {kPromptTooLong, Verdict::too_long},
        {"We have seven apples now.\nWe have seven apples now.\n#### 7\n", Verdict::loop},
        {"[[no-loss]]3 + 4 = 8[[/no-loss]] 3 + 4 = 7\n#### 7\n", Verdict::markers},
        {std::string("3 + 4 = 7") + char(kBos) + "\n#### 7\n", Verdict::markers},
    };
    for (const auto& [text, want] : cases) {
        const Verdict got = verdict(text, o);
        EXPECT(got == want, "'%s': %s, expected %s", text.c_str(), self_train::verdict_name(got),
               self_train::verdict_name(want));
    }
    // With loops allowed, the repeated line is kept.
    self_train::Options loops = o;
    loops.allow_loops = true;
    EXPECT(verdict("We have seven apples now.\nWe have seven apples now.\n#### 7\n", loops) == Verdict::kept,
           "--allow-loops must keep a right try that repeats a line");
    // The problem file's own solution, written back word for word, is
    // dropped (spaces around it aside), unless --keep-gold; a solution that
    // differs in a word is the model's own.
    EXPECT(verdict(" gold work\n#### 7\n", o) == Verdict::gold_copy, "a copy of the gold solution must be dropped");
    EXPECT(verdict("gold work, done\n#### 7\n", o) == Verdict::kept, "another solution is not a gold copy");
    self_train::Options gold = o;
    gold.keep_gold = true;
    EXPECT(verdict("gold work\n#### 7\n", gold) == Verdict::kept, "--keep-gold must keep a copy of the gold solution");
    // A calculation the calculator cannot read is unchecked, not wrong.
    self_train::TryInfo info;
    EXPECT(self_train::check_try(p, "10 / 3 = 3.33, then 3 + 4 = 7\n#### 7\n", o, &info) == Verdict::kept &&
               info.calcs == 1 && info.unreadable == 1,
           "an unreadable 10 / 3 must be counted, not rejected (calcs %zu, unreadable %zu)", info.calcs,
           info.unreadable);
    // A chain is checked link by link (rp1's "6 * 4 + 2 = 24 + 2 = 26" was
    // rejected), signs included: -5 + 8 is a positive 3.
    const Problem bags = self_train_problem("6 bags of 4 apples and 2 more: how many apples?", "26");
    EXPECT(self_train::check_try(bags, "6 * 4 + 2 = 24 + 2 = 26\n#### 26\n", o) == Verdict::kept,
           "a right chain must be kept");
    EXPECT(self_train::check_try(bags, "6 * 4 + 2 = 24 + 2 = 28, so 26\n#### 26\n", o) == Verdict::arithmetic,
           "a chain with a wrong second link must be rejected");
    EXPECT(self_train::check_try(bags, "6 * 3 + 2 = 24 + 2 = 26\n#### 26\n", o) == Verdict::arithmetic,
           "a chain with a wrong first link must be rejected");
    const Problem three = self_train_problem("What is 2 minus 7 plus 8?", "3");
    EXPECT(self_train::check_try(three, "2 - 7 + 8 = -5 + 8 = 3\n#### 3\n", o) == Verdict::kept,
           "a link that starts with a minus sign, of a positive value, must be kept");
    EXPECT(self_train::check_try(three, "7 - 2 - 8 = 5 - 8 = 3\n#### 3\n", o) == Verdict::arithmetic,
           "5 - 8 is -3, not 3: the sign of a chain's result is checked");
    // "$" and digit-grouping commas, as GSM8K writes them, are read.
    const Problem money = self_train_problem("How much for 2 at $600?", "1200");
    EXPECT(self_train::check_try(money, "2 x $600 = $1,200\n#### 1,200\n", o) == Verdict::kept,
           "a right GSM8K-style money step must be kept");
    EXPECT(self_train::check_try(money, "2 x $600 = $1,100, so 1,200\n#### 1,200\n", o) == Verdict::arithmetic,
           "a wrong GSM8K-style money step must be rejected");
}

inline void test_self_train_keep() {
    using self_train::Verdict;
    const std::vector<Problem> ps = {self_train_problem("What is 3 plus 4?", "7"),
                                     self_train_problem("What is 2 times 5?", "10"),
                                     self_train_problem("What is 9 minus 1?", "8")};
    const std::vector<std::vector<std::string>> texts = {
        // Lucky first try (right answer, wrong step), then right ones: A, a
        // copy of A (spaces around the work do not make it new), B, C.
        {"3 + 4 = 8, so 7.\n#### 7\n", "#### 6\n", "3 + 4 = 7\n#### 7\n", " 3 + 4 = 7\n#### 7\n",
         "4 + 3 = 7\n#### 7\n", "3 plus 4 makes 7.\n#### 7\n"},
        {"2 * 5 = 10\n#### 10\n", "2 * 5 = 11\n#### 11\n", "5 + 5 = 10\n#### 10\n", "no answer",
         "2 * 5 = 10\n#### 10\n", "10 it is\n#### 10\n"},
        {"#### 7\n", "9 - 1 = 7\n#### 7\n", "9 - 1 = 9\n#### 8\n", "#### 8\n", "#### 9\n", "#### 6\n"},
    };
    // Compared as text here: test_self_train_outline compares outlines.
    self_train::Options o;
    o.keep = 2;
    o.same = self_train::Same::text;
    const self_train::Result r = self_train::keep_checked(ps, texts, o);
    const std::vector<std::vector<Verdict>> want = {
        {Verdict::arithmetic, Verdict::wrong, Verdict::kept, Verdict::duplicate, Verdict::kept, Verdict::over_cap},
        {Verdict::kept, Verdict::wrong, Verdict::kept, Verdict::unfinished, Verdict::duplicate, Verdict::over_cap},
        {Verdict::wrong, Verdict::wrong, Verdict::arithmetic, Verdict::no_work, Verdict::wrong, Verdict::wrong},
    };
    for (size_t i = 0; i < want.size(); ++i)
        for (size_t t = 0; t < want[i].size(); ++t)
            EXPECT(r.verdicts[i][t] == want[i][t], "problem %zu try %zu: %s, expected %s", i, t,
                   self_train::verdict_name(r.verdicts[i][t]), self_train::verdict_name(want[i][t]));
    EXPECT(r.problems == 3 && r.tries == 18, "3 problems, 18 tries: got %zu, %zu", r.problems, r.tries);
    EXPECT(r.kept.size() == 4 && r.count(Verdict::kept) == 4, "4 kept, got %zu", r.kept.size());
    EXPECT(r.count(Verdict::duplicate) == 2 && r.count(Verdict::over_cap) == 2 && r.count(Verdict::wrong) == 6 &&
               r.count(Verdict::arithmetic) == 2 && r.count(Verdict::unfinished) == 1 &&
               r.count(Verdict::no_work) == 1,
           "verdict counts: duplicate %zu over_cap %zu wrong %zu arithmetic %zu unfinished %zu no_work %zu",
           r.count(Verdict::duplicate), r.count(Verdict::over_cap), r.count(Verdict::wrong),
           r.count(Verdict::arithmetic), r.count(Verdict::unfinished), r.count(Verdict::no_work));
    // First tries: lucky 7 (right answer), 10, and "#### 7" (wrong): two
    // right answers. Solved: all three (problem 3 only by its lucky and
    // empty tries); with a kept solution: the first two.
    EXPECT(r.first_right == 2 && r.solved == 3 && r.solved_kept == 2,
           "first right %zu (want 2), solved %zu (3), solved and kept %zu (2)", r.first_right, r.solved,
           r.solved_kept);
    // The kept work, in order, with the problem file's answer.
    const char* work[] = {"3 + 4 = 7", "4 + 3 = 7", "2 * 5 = 10", "5 + 5 = 10"};
    for (size_t k = 0; k < 4; ++k)
        EXPECT(r.kept[k].steps == work[k] && r.kept[k].answer == ps[k < 2 ? 0 : 1].answer &&
                   r.kept[k].question == ps[k < 2 ? 0 : 1].question && r.kept[k].task == "qa" &&
                   r.kept[k].difficulty == 2,
               "kept %zu: '%s'", k, r.kept[k].steps.c_str());
    // Without a cap every distinct right try is kept.
    o.keep = 0;
    const self_train::Result all = self_train::keep_checked(ps, texts, o);
    EXPECT(all.kept.size() == 6 && all.count(Verdict::over_cap) == 0, "no cap: %zu kept", all.kept.size());
    // By outline (the default), "4 + 3 = 7" is "3 + 4 = 7" again, and
    // "5 + 5 = 10" is "2 * 5 = 10": the same values in the same order.
    // Work without a calculation ("3 plus 4 makes 7.") is compared as text.
    const self_train::Result outline = self_train::keep_checked(ps, texts, {});
    const std::vector<std::vector<Verdict>> by_outline = {
        {Verdict::arithmetic, Verdict::wrong, Verdict::kept, Verdict::duplicate, Verdict::duplicate, Verdict::kept},
        {Verdict::kept, Verdict::wrong, Verdict::duplicate, Verdict::unfinished, Verdict::duplicate, Verdict::kept},
        want[2],
    };
    for (size_t i = 0; i < by_outline.size(); ++i)
        for (size_t t = 0; t < by_outline[i].size(); ++t)
            EXPECT(outline.verdicts[i][t] == by_outline[i][t], "by outline, problem %zu try %zu: %s, expected %s", i,
                   t, self_train::verdict_name(outline.verdicts[i][t]), self_train::verdict_name(by_outline[i][t]));
    EXPECT(outline.kept.size() == 4 && outline.count(Verdict::duplicate) == 4, "by outline: %zu kept, %zu duplicates",
           outline.kept.size(), outline.count(Verdict::duplicate));
}

// The same solution in other words, against another try and against the
// problem's own solution.
inline void test_self_train_outline() {
    using self_train::Verdict;
    Problem p = self_train_problem("Lina buys 12 packs of 4 pencils and keeps half. How many pencils does she keep?",
                                   "24");
    p.steps = "Lina starts with 12*4 = 48 pencils.\nLina keeps 48/2 = 24 pencils.";
    const std::string reworded = "The starting amount is 12 x 4 = 48.\nThe half kept is 48 / 2 = 24.\n#### 24\n";
    const std::string other_way = "Half of the packs is 12 / 2 = 6, and 6 * 4 = 24 pencils.\n#### 24\n";
    const self_train::Options o;
    EXPECT(self_train::check_try(p, reworded, o) == Verdict::gold_copy,
           "the gold solution in other words (same outline) must be a gold copy");
    EXPECT(self_train::check_try(p, other_way, o) == Verdict::kept, "another way to the answer must be kept");
    self_train::Options text;
    text.same = self_train::Same::text;
    EXPECT(self_train::check_try(p, reworded, text) == Verdict::kept,
           "--same text: the gold solution in other words is the model's own");
    self_train::Options gold;
    gold.keep_gold = true;
    EXPECT(self_train::check_try(p, reworded, gold) == Verdict::kept, "--keep-gold must keep it");
    // Against each other: the second wording of an outline is a duplicate.
    p.steps = "gold work";
    const self_train::Result r =
        self_train::keep_checked({p}, {{other_way, "Half the packs: 12/2 = 6. Then 6*4 = 24.\n#### 24\n", reworded}}, o);
    EXPECT(r.verdicts[0] == std::vector<Verdict>({Verdict::kept, Verdict::duplicate, Verdict::kept}),
           "by outline: kept, duplicate, kept; got %s, %s, %s", self_train::verdict_name(r.verdicts[0][0]),
           self_train::verdict_name(r.verdicts[0][1]), self_train::verdict_name(r.verdicts[0][2]));
}

// Numbers from outside the question: an invented quantity is rejected,
// common constants (7 days a week) are not.
inline void test_self_train_outside() {
    using self_train::Verdict;
    const self_train::Options o;
    const Problem p = self_train_problem("What is 3 plus 4?", "7");
    self_train::TryInfo info;
    EXPECT(self_train::check_try(p, "Each box holds 37, so 3 + 4 = 7\n#### 7\n", o, &info) == Verdict::outside &&
               info.outside == std::vector<std::string>({"37"}),
           "an invented 37 must be rejected and named (%zu outside numbers)", info.outside.size());
    self_train::Options allow;
    allow.allow_outside = true;
    EXPECT(self_train::check_try(p, "Each box holds 37, so 3 + 4 = 7\n#### 7\n", allow) == Verdict::kept,
           "--allow-outside must keep it");
    // In words too (Codex's review on #69): the failure labels read digits only.
    info = {};
    EXPECT(self_train::check_try(p, "Each box holds thirteen, so 3 + 4 = 7\n#### 7\n", o, &info) ==
                   Verdict::outside &&
               info.outside == std::vector<std::string>({"13 (in words)"}),
           "an invented thirteen must be rejected and named (%zu outside numbers)", info.outside.size());
    // A word for a number the question gives, a result, or one step away, is not outside.
    const Problem words = self_train_problem("What is three plus four?", "7");
    EXPECT(self_train::check_try(words, "Three and four make 3 + 4 = 7, which is seven\n#### 7\n", o) == Verdict::kept,
           "number words from the question and the result must not be outside");
    EXPECT(self_train::check_try(p, "3 + 4 = 7, twelve less than a dozen and seven\n#### 7\n", o) == Verdict::kept,
           "a constant in words (a dozen, twelve) must not be outside");
    // Results, and numbers one step from the question's, are not outside.
    EXPECT(self_train::check_try(p, "3 + 4 = 7, which is 6 more than 1\n#### 7\n", o) == Verdict::kept,
           "a result and a number one step away must not be outside");
    // 7 days in a week: not in the question, not one step from 3 and 2.
    const Problem weeks = self_train_problem("Sam reads 3 pages a day. How many pages does Sam read in 2 weeks?", "42");
    EXPECT(self_train::check_try(weeks, "Sam reads 3 * 7 = 21 pages a week, so 21 * 2 = 42\n#### 42\n", o) ==
               Verdict::kept,
           "7 days a week is a common constant");
    // 13 is not one step from 3, 2 (or 1, 2, 100); 9 would be (3 * 3).
    EXPECT(self_train::check_try(weeks, "Sam reads 3 * 13 = 39 pages, then 39 + 3 = 42\n#### 42\n", o) ==
               Verdict::outside,
           "13 days from nowhere must be rejected");
}

// From the scripted model to the files: what sample_tries decodes is what
// is checked, and the written files are the kept problems, masked as
// prep-qa masks them.
inline void test_self_train_files() {
    // Questions of the same length keep the batch order the input order.
    const std::vector<Problem> ps = {self_train_problem("What is 600 x 2?", "1200"),
                                     self_train_problem("What is 600 x 3?", "1800")};
    ScriptedModel fake;
    for (const char* s : {"600 x 2 = 1,200\n#### 1,200\n", "600 x 2 = 1300\n#### 1200\n", "#### 1200\n"})
        fake.queue.push_back(s);
    for (const char* s : {"600 x 3 = 1800\n#### 1800\n", "600 x 3 = 1800\n#### 1800\n", "600 + 1200 = 1800\n#### 1800\n"})
        fake.queue.push_back(s);
    const Tokenizer tok;
    fake.prompt_len = int(prompt_tokens(ps[0], tok).size());
    VoteOpts vote;
    vote.tries = 3;
    vote.temperature = 1.0;
    calc::Counts calls;
    const auto texts = sample_tries(fake, ps, tok, 6, 40, vote, calc::Mode::none, &calls);
    EXPECT(texts.size() == 2 && texts[0].size() == 3 && texts[0][1] == "600 x 2 = 1300\n#### 1200\n",
           "sample_tries must return every try in order (got '%s')", texts[0][1].c_str());
    const self_train::Result r = self_train::keep_checked(ps, texts, {});
    EXPECT(r.kept.size() == 2, "2 kept (1 right, 1 wrong step, 1 no work; 1 outline, written twice), got %zu",
           r.kept.size());
    const std::string out = test_path("self-train");
    self_train::write(out, r.kept, tok);
    const std::vector<Problem> back = load_problems(out + ".txt");
    EXPECT(back.size() == r.kept.size(), "%zu problems read back for %zu kept", back.size(), r.kept.size());
    for (size_t k = 0; k < std::min(back.size(), r.kept.size()); ++k)
        EXPECT(back[k].question == r.kept[k].question && back[k].steps == r.kept[k].steps &&
                   back[k].answer == r.kept[k].answer && back[k].task == r.kept[k].task,
               "problem %zu read back differently", k);
    EXPECT(r.kept[0].completion() == "600 x 2 = 1,200\n#### 1200\n",
           "the kept answer line is the problem file's: '%s'", r.kept[0].completion().c_str());
    const TokenData d = TokenData::load(out + ".tok");
    std::vector<uint16_t> tokens;
    std::vector<uint8_t> mask;
    for (const Problem& p : r.kept) {
        for (int t : prompt_tokens(p, tok)) tokens.push_back(uint16_t(t)), mask.push_back(0);
        for (int t : tok.encode(p.completion())) tokens.push_back(uint16_t(t)), mask.push_back(1);
    }
    EXPECT(d.tokens == tokens && d.mask == mask && d.vocab == uint32_t(tok.vocab_size()),
           "token file: %zu tokens and %zu mask bytes, expected %zu", d.tokens.size(), d.mask.size(),
           tokens.size());
    // Nothing kept: empty files, still readable.
    self_train::write(out + "-empty", {}, tok);
    EXPECT(load_problems(out + "-empty.txt").empty() && TokenData::load(out + "-empty.tok").tokens.empty(),
           "an empty result must write empty, readable files");
}

}  // namespace

inline void test_self_train() {
    test_self_train_check();
    test_self_train_keep();
    test_self_train_outline();
    test_self_train_outside();
    test_self_train_files();
}
