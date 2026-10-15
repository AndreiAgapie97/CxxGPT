// test_no_repeat.h — the no-repeat block (no_repeat.h and generate.h): a fake
// model with a first, second and third choice at every position shows
// exactly which tokens the block forbids; every expected text below was
// worked out by hand. With the block off, or never triggered, generation
// must be exactly what it was without it, greedy and sampled.
#pragma once

#include "../src/generate.h"
#include "../src/model.h"
#include "test_util.h"

namespace {

// Byte-level. At each generated position the script's byte is the first
// choice (logit 1e6), then 'x' (1e5), then 'y' (1e4); for a script byte
// 'x' the fallbacks are 'y' and 'z'. The gaps are so wide that sampling at
// temperature 1 is certain too. After the script, '.' is the first choice.
class PreferenceModel {
public:
    Config cfg;
    std::string script;
    int prompt_len = 1;  // every prompt has this many tokens

    PreferenceModel() {
        cfg.vocab = 256;
        cfg.seq_len = 128;
    }
    void decode_begin(int B, const std::vector<int>&) {
        B_ = B;
        logits_.assign(size_t(B) * cfg.vocab, 0.0f);
    }
    const float* decode_step(const int*, int pos) {
        const size_t index = size_t(pos - (prompt_len - 1));
        const int first = index < script.size() ? int(static_cast<unsigned char>(script[index])) : int('.');
        const int second = first == 'x' ? 'y' : 'x', third = first == 'x' ? 'z' : 'y';
        for (int b = 0; b < B_; ++b) {
            float* row = &logits_[size_t(b) * cfg.vocab];
            std::fill(row, row + cfg.vocab, 0.0f);
            row[first] = 1e6f;
            row[second] = 1e5f;
            row[third] = 1e4f;
        }
        return logits_.data();
    }

private:
    int B_ = 0;
    std::vector<float> logits_;
};

std::string text_of_bytes(const std::vector<int>& tokens) {
    std::string s;
    for (int t : tokens) s += char(t);
    return s;
}

// pick_token with blocked tokens: greedy takes the best allowed token,
// sampling never draws a blocked one, and nothing blocked is the plain pick.
inline void test_pick_token_blocked() {
    const std::vector<float> logits = {0.5f, 3.0f, 2.0f, 1.0f, 0.0f, 2.5f, 0.1f, 0.2f};
    const int V = int(logits.size());
    EXPECT(pick_token(logits.data(), V, 0.0, nullptr, {}) == 1, "greedy without a block must pick 1");
    EXPECT(pick_token(logits.data(), V, 0.0, nullptr, {1}) == 5, "greedy with 1 blocked must pick 5");
    EXPECT(pick_token(logits.data(), V, 0.0, nullptr, {1, 5, 2}) == 3, "greedy with 1, 5, 2 blocked must pick 3");
    const std::vector<int> all = {0, 1, 2, 3, 4, 5, 6, 7};
    EXPECT(pick_token(logits.data(), V, 0.0, nullptr, all) == 1, "with every token blocked the block is ignored");
    // Sampling from equal logits with 2 and 5 blocked: those never come,
    // every other token does.
    const std::vector<float> flat(8, 0.0f);
    Rng rng(7);
    std::vector<int> seen(size_t(V), 0);
    for (int i = 0; i < 4000; ++i) ++seen[size_t(pick_token(flat.data(), V, 1.0, &rng, {2, 5}))];
    EXPECT(seen[2] == 0 && seen[5] == 0, "sampling drew blocked tokens %d and %d times", seen[2], seen[5]);
    for (int t : {0, 1, 3, 4, 6, 7})
        EXPECT(seen[size_t(t)] > 500, "allowed token %d sampled only %d times in 4000", t, seen[size_t(t)]);
    // Nothing blocked: the same draws as the plain pick, seed for seed.
    Rng a(9), b(9);
    for (int i = 0; i < 200; ++i)
        EXPECT(pick_token(logits.data(), V, 1.0, &a, {}) == pick_token(logits.data(), V, 1.0, &b),
               "an empty block changed sampled pick %d", i);
}

// NoRepeat on its own: the tokens that would complete a run of N written
// tokens a second time, and nothing else; a BOS forgets the past.
inline void test_no_repeat_runs() {
    NoRepeat r(3);
    for (char c : std::string("abcab")) r.add(c);
    EXPECT(r.blocked() == std::vector<int>{'c'}, "after 'abcab' only 'c' (completing 'abc') must be blocked");
    r.add('x');
    EXPECT(r.blocked().empty(), "after 'abcabx' nothing is blocked (no earlier 'bx')");
    for (char c : std::string("ab")) r.add(c);
    EXPECT((r.blocked() == std::vector<int>{'c', 'x'}), "after 'abcabxab' both 'c' and 'x' must be blocked");
    r.add(kBos);
    for (char c : std::string("ab")) r.add(c);
    EXPECT(r.blocked().empty(), "a BOS must start a new document with nothing blocked");
    NoRepeat two(2);
    two.add('7');
    EXPECT(two.blocked().empty(), "N = 2 after one token: nothing written twice yet");
    two.add('7');
    EXPECT(two.blocked() == std::vector<int>{'7'}, "N = 2 after '77': a third '7' would repeat '77'");
}

// generate() with the block, on scripts worked out by hand.
inline void test_no_repeat_generate() {
    auto never = [](const std::vector<int>&) { return false; };
    auto run = [&](const std::string& script, int n, double temperature = 0.0, const std::string& prompt = "Q") {
        PreferenceModel m;
        m.script = script;
        std::vector<int> ids = {kBos};
        for (char c : prompt) ids.push_back(c);
        m.prompt_len = int(ids.size());
        Rng rng(5);
        return text_of_bytes(generate(m, {ids}, int(script.size()), never, temperature,
                                      temperature > 0 ? &rng : nullptr, nullptr, calc::Mode::none, nullptr,
                                      nullptr, n)[0]);
    };
    // N = 3, "abcabcabc": the 6th token 'c' would repeat "abc", so 'x'
    // comes instead; the 9th 'c' would repeat "abc" and its fallback 'x'
    // would repeat "abx", so the third choice 'y' comes.
    EXPECT(run("abcabcabc", 3) == "abcabxaby", "N = 3 gave '%s'", run("abcabcabc", 3).c_str());
    // N = 4: "abca" is the first run of 4 written twice, at the 7th token.
    EXPECT(run("abcabcabc", 4) == "abcabcxbc", "N = 4 gave '%s'", run("abcabcabc", 4).c_str());
    EXPECT(run("abcabcabc", 0) == "abcabcabc", "the block off gave '%s'", run("abcabcabc", 0).c_str());
    // Sampling obeys the block the same way (the fake's gaps make it certain).
    EXPECT(run("abcabcabc", 3, 1.0) == "abcabxaby", "N = 3 sampled gave '%s'", run("abcabcabc", 3, 1.0).c_str());
    // The prompt does not count: an answer may copy its question.
    EXPECT(run("abcd", 3, 0.0, "abc") == "abcd", "copying the prompt was blocked: '%s'",
           run("abcd", 3, 0.0, "abc").c_str());
    // A generated BOS starts a new document: "ab" after it may go on with
    // 'c' again.
    const std::string bos_script = std::string("abc") + char(kBos) + "abc";
    EXPECT(run(bos_script, 3) == bos_script, "a BOS did not reset the block");
    // Two sequences in one batch keep their own histories: the long prompt
    // pads the short one with its first token, and the padding is not text.
    PreferenceModel m;
    m.script = "abcabc";
    const std::vector<int> short_prompt = {kBos, 'Q'}, long_prompt = {kBos, 'Q', 'Q', 'Q'};
    m.prompt_len = int(long_prompt.size());
    const auto both = generate(m, {short_prompt, long_prompt}, 6, never, 0.0, nullptr, nullptr, calc::Mode::none,
                               nullptr, nullptr, 3);
    EXPECT(text_of_bytes(both[0]) == "abcabx" && text_of_bytes(both[1]) == "abcabx",
           "batched with the block: '%s' and '%s'", text_of_bytes(both[0]).c_str(),
           text_of_bytes(both[1]).c_str());
}

// A real tiny model: the block never lets a run of N repeat, and when it
// cannot trigger (N longer than the whole text) generation is exactly the
// unblocked one, greedy and sampled.
inline void test_no_repeat_real_model() {
    Config cfg;
    cfg.vocab = 256;
    cfg.seq_len = 64;
    cfg.d_model = 24;
    cfg.n_layers = 2;
    cfg.n_heads = 2;
    Model<float> m(cfg);
    m.init(21);
    auto never = [](const std::vector<int>&) { return false; };
    const std::vector<std::vector<int>> prompts = {{kBos, 'Q', ':'}, {kBos, '1', '+', '2'}};
    const int max_new = 40;
    for (double temperature : {0.0, 1.0}) {
        auto run = [&](int n) {
            std::vector<Rng> rngs = {Rng(3), Rng(4)};
            return generate(m, prompts, max_new, never, temperature, nullptr, temperature > 0 ? &rngs : nullptr,
                            calc::Mode::none, nullptr, nullptr, n);
        };
        EXPECT(run(max_new + 1) == run(0), "a block that cannot trigger changed the text (temperature %g)",
               temperature);
        for (int n : {2, 3, 4}) {
            const auto outs = run(n);
            for (const auto& out : outs) {
                // Within each document (BOS resets), no run of n tokens twice.
                std::set<std::vector<int>> runs;
                size_t begin = 0;
                for (size_t i = 0; i < out.size(); ++i) {
                    if (out[i] == kBos) {
                        runs.clear();
                        begin = i + 1;
                        continue;
                    }
                    if (i + 1 < begin + size_t(n)) continue;
                    const std::vector<int> run_n(out.begin() + long(i) + 1 - n, out.begin() + long(i) + 1);
                    EXPECT(runs.insert(run_n).second, "N = %d, temperature %g: a run repeated at token %zu", n,
                           temperature, i);
                }
            }
        }
    }
}

// eval_reasoning carries the block: its effort label names it, and the
// unfinished count reports first tries without an answer line.
inline void test_no_repeat_eval() {
    VoteOpts v;
    v.no_repeat = 12;
    EXPECT(effort_label(v) == "greedy, no-repeat 12", "greedy with the block is labelled '%s'",
           effort_label(v).c_str());
    v.tries = 8;
    v.temperature = 0.7;
    v.seed = 3;
    EXPECT(effort_label(v) == "maj@8, temp 0.7, seed 3, no-repeat 12", "eight tries with the block are labelled '%s'",
           effort_label(v).c_str());
    Problem p;
    p.task = "add";
    p.difficulty = 1;
    p.question = "1+1=?";
    p.steps = "1+1=2";
    p.answer = "2";
    PreferenceModel finished;
    finished.script = "1+1=2\n#### 2\n";
    finished.prompt_len = 1 + int(p.prompt().size());
    VoteOpts block;
    block.no_repeat = 5;
    const EvalResult ok = eval_reasoning(finished, {p}, Tokenizer{}, 4, 40, 8, true, block);
    EXPECT(ok.correct == 1 && ok.unfinished == 0 && ok.effort == "greedy, no-repeat 5",
           "a finished answer with the block: correct %d, unfinished %zu, effort '%s'", ok.correct, ok.unfinished,
           ok.effort.c_str());
    // A loop with no answer line: unfinished, with or without the block.
    PreferenceModel looping;
    looping.script = std::string(40, 'a');
    looping.prompt_len = finished.prompt_len;
    const EvalResult loop = eval_reasoning(looping, {p}, Tokenizer{}, 4, 40, 8, true);
    EXPECT(loop.correct == 0 && loop.unfinished == 1, "a loop: correct %d, unfinished %zu", loop.correct,
           loop.unfinished);
    // With N = 5 the 6th 'a' would write "aaaaa" a second time, so 'x'
    // comes; the runs of five after it ("aaaax", "aaaxa", ..., "xaaaa") are
    // all new.
    const EvalResult broken = eval_reasoning(looping, {p}, Tokenizer{}, 4, 10, 8, true, block);
    EXPECT(broken.completions.size() == 1 && broken.completions[0] == "aaaaaxaaaa",
           "the block on a loop gave '%s'", broken.completions.empty() ? "" : broken.completions[0].c_str());
}

inline void test_no_repeat() {
    test_pick_token_blocked();
    test_no_repeat_runs();
    test_no_repeat_generate();
    test_no_repeat_real_model();
    test_no_repeat_eval();
}

}  // namespace
