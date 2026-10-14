#pragma once

#include "../src/generate.h"
#include "test_util.h"

inline void test_calc() {
    const std::pair<const char*, const char*> exact[] = {
        {"\r1\t+2\n", "3"}, {"\f1 + 2\v", "3"}, {"48/2", "24"}, {"7/2*2", "7"}, {"1/3*3", "1"}, {"0.1+0.2", "0.3"},
        {"-(2+3)*4/8", "-2.5"}, {"12.5000", "12.5"}, {"0/-2", "0"},
        {"3-2-1", "0"}, {"2+3*4", "14"}, {".5 + 1.", "1.5"},
        {"9223372036854775807/9223372036854775807", "1"}, {"1/8", "0.125"},
        {"1/1000000000000000000", "0.000000000000000001"}
    };
    for (const auto& [expr, expected] : exact)
        EXPECT(calc::evaluate(expr) == std::optional<std::string>(expected), "calculator exact: %s", expr);
    for (const char* expr : {"", " ", "1/0", "1/3", "2^3", "sqrt(4)", "2x3", "1,000", "2(3)",
                            "1e3", "1..2", "(2+3", "2+3)", "2+", "1=1", "nan", "inf", "a+2",
                            "9223372036854775808", "9223372036854775807+1", "1/0.0000000000000000001"})
        EXPECT(!calc::evaluate(expr), "calculator must reject: %s", expr);
    EXPECT(!calc::evaluate(std::string(129, '(') + "1" + std::string(129, ')')), "calculator depth bound");
    EXPECT(!calc::evaluate(std::string(4097, '1')), "calculator length bound");

    // Build expression trees while separately calculating their values in
    // double precision. This reference never parses or uses Rational.
    Rng rng(9182);
    std::function<std::pair<std::string, double>(int)> tree = [&](int depth) -> std::pair<std::string, double> {
        const int n = int(rng.uniform() * 101);
        if (!depth) return {std::to_string(n / 10) + "." + std::to_string(n % 10), double(n) / 10};
        auto [left, a] = tree(depth - 1);
        auto [right, b] = tree(depth - 1);
        const int op = int(rng.uniform() * 4);
        if (op == 3) { const int ds[] = {1,2,4,5,10}; const int d = ds[int(rng.uniform() * 5)]; right = std::to_string(d); b = d; }
        const double ref = op == 0 ? a + b : op == 1 ? a - b : op == 2 ? a * b : a / b;
        return {"(" + left + "+-*/"[op] + right + ")", ref};
    };
    for (int i = 0; i < 1000; ++i) {
        const auto [expr, ref] = tree(3);
        const auto result = calc::evaluate(expr);
        EXPECT(result && std::abs(std::stod(*result) - ref) <= 1e-10 * std::max(1.0, std::abs(ref)),
               "calculator random independent double reference: %s", expr.c_str());
    }

    for (const char* expr : {"16 x 7 = ", "16 * 7 = ", "16 × 7 = ", "She has 16 x 7 = ", "Then: 16 * 7 = ", "next; 16 * 7 = "})
        EXPECT(calc::response(expr, calc::Mode::plain) == std::optional<std::string>("112"), "plain multiplication: %s", expr);
    EXPECT(calc::response("-3 * 2 = ", calc::Mode::plain) == std::optional<std::string>("-6"), "unary plain expression");
    for (const char* expr : {"2^3 + 4 = ", "2 ^ 3 + 4 = ", "2 @ 3 + 4 = ", "2 & 3 + 4 = ", "2 X 3 + 4 = ",
                            "2 \\times 3 + 4 = ", "2 times 3 + 4 = ", "a + 3 = ", "a -3*2 = ", "sqrt(4)+3 = ", "1,000+3 = ", "5/0 = "}) {
        const auto r = calc::response(expr, calc::Mode::plain);
        EXPECT(r && r->empty(), "malformed plain expression must not be partially computed: %s", expr);
    }
    EXPECT(!calc::response("The answer = ", calc::Mode::plain), "prose is not a call");
    EXPECT(!calc::response("16 * 7 =", calc::Mode::plain), "plain waits for trailing space");
    EXPECT(calc::response("<<48/2=", calc::Mode::notes) == std::optional<std::string>("24>>"), "notes result");
    EXPECT(!calc::response("<<2=2>> then =", calc::Mode::notes), "closed note ignored");
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
            std::vector<std::string>{"<<48/2=24>> continue\n", "bad <<5/0=unknown; <<7/2*2=7>> ok\n"} :
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
                                  1, &actual, mode, &tok, &calls);
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
    const auto capped = generate(m, {{kBos}}, 8, [](const auto&) { return false; }, 0, nullptr, calc::Mode::notes, &tok, &calls);
    EXPECT(tok.decode(capped[0]) == "<<48/2=w" && calls.fired == 0, "budget never inserts partial result");
    m.cfg.seq_len = 8;
    const auto context = generate(m, {{kBos}}, 100, [](const auto&) { return false; }, 0, nullptr, calc::Mode::notes, &tok, &calls);
    EXPECT(context[0] == capped[0], "context cap includes result tokens");
    m.cfg.seq_len = 30;
    const auto disabled = generate(m, {{kBos}}, 12, [](const auto&) { return false; });
    EXPECT(tok.decode(disabled[0]) == "<<48/2=wrong", "default generation unchanged");

    // Punctuation merges can contain both '=' and a trailing '$'. The
    // generated text must retain that suffix after the inserted result.
    Tokenizer bpe; bpe.train("=$ =$ =$ =$ =$ =$", 257);
    EXPECT(bpe.encode("=$").size() == 1, "BPE trigger-tail fixture");
    CalcFakeModel merged; merged.cfg.vocab = bpe.vocab_size(); merged.cfg.seq_len = 60;
    merged.script = {bpe.encode("<<48/2=24>>$ done\n")};
    merged.prediction = merged.script;
    const auto prefix = bpe.encode("<<48/2");
    merged.prediction[0][prefix.size()] = bpe.encode("=$")[0];
    for (size_t i = prefix.size() + 1; i < prefix.size() + 5; ++i) merged.prediction[0][i] = '?';
    calc::Counts merged_calls;
    const auto merged_out = generate(merged, {{kBos}}, 50, [](const auto& ids) { return ids.back() == '\n'; },
                                     0, nullptr, calc::Mode::notes, &bpe, &merged_calls);
    EXPECT(bpe.decode(merged_out[0]) == "<<48/2=24>>$ done\n" && merged_calls.fired == 1,
           "BPE trigger inside token preserves trailing text");
}
