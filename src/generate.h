// generate.h — batched text generation with the KV cache, and the
// reasoning evaluation built on it.
#pragma once

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "model.h"
#include "reasoning.h"
#include "tokenizer.h"

// Picks the next token from one row of logits: argmax if temperature <= 0,
// otherwise a sample from softmax(logits / temperature).
template <class T>
int pick_token(const T* logits, int V, double temperature, Rng* rng) {
    int best = 0;
    for (int i = 1; i < V; ++i)
        if (logits[i] > logits[best]) best = i;
    if (temperature <= 0 || !rng) return best;
    std::vector<double> p(V);
    double sum = 0;
    for (int i = 0; i < V; ++i) sum += p[i] = std::exp((double(logits[i]) - logits[best]) / temperature);
    double u = rng->uniform() * sum;
    for (int i = 0; i < V; ++i)
        if ((u -= p[i]) <= 0) return i;
    return best;
}

// Continues every prompt by up to max_new tokens, all in one batch.
// done(generated) is asked after each new token; a sequence that is done
// stops growing (its slot keeps computing until the batch finishes).
template <class T>
std::vector<std::vector<int>> generate(Model<T>& model, const std::vector<std::vector<int>>& prompts,
                                       int max_new,
                                       const std::function<bool(const std::vector<int>&)>& done,
                                       double temperature = 0.0, Rng* rng = nullptr) {
    const int B = int(prompts.size());
    std::vector<std::vector<int>> out(B);
    if (B == 0) return out;
    int L = 0;
    for (const auto& p : prompts) {
        CHECK(!p.empty(), "empty prompt");
        L = std::max(L, int(p.size()));
    }
    const int total = std::min(model.cfg.seq_len, L + max_new);
    CHECK(L < total, "prompt of %d tokens leaves no room in seq_len %d", L, model.cfg.seq_len);
    std::vector<int> start(B);
    for (int b = 0; b < B; ++b) start[b] = L - int(prompts[b].size());
    model.decode_begin(B, start);
    std::vector<int> cur(B), finished(B, 0);
    int n_finished = 0;
    for (int pos = 0; pos < total; ++pos) {
        for (int b = 0; b < B; ++b) {
            if (pos < L) cur[b] = pos < start[b] ? prompts[b][0] : prompts[b][pos - start[b]];
            // else cur[b] holds the token picked at the previous step
        }
        const T* logits = model.decode_step(cur.data(), pos);
        if (pos < L - 1) continue;
        for (int b = 0; b < B; ++b) {
            const int t = pick_token(logits + size_t(b) * model.cfg.vocab, model.cfg.vocab, temperature, rng);
            cur[b] = t;
            if (finished[b]) continue;
            out[b].push_back(t);
            if (done(out[b])) finished[b] = 1, ++n_finished;
        }
        if (n_finished == B) break;
    }
    return out;
}

// A completion is over once the "#### answer" line has ended.
inline bool completion_done(const std::string& text) {
    return !text.empty() && text.back() == '\n' && text.find("#### ") != std::string::npos;
}

struct EvalResult {
    // (task, difficulty) -> (correct, total)
    std::map<std::pair<std::string, int>, std::pair<int, int>> by_task;
    int correct = 0, total = 0;
    std::vector<std::pair<Problem, std::string>> failures;  // a few, for inspection
};

// The tokens a problem's prompt is fed as: BOS, then "Q: ...\nA: ". The
// completion is encoded separately (training does the same), so no token
// straddles the boundary the model starts generating from.
inline std::vector<int> prompt_tokens(const Problem& p, const Tokenizer& tok) {
    std::vector<int> t = tok.encode(p.prompt());
    t.insert(t.begin(), kBos);
    return t;
}

// Greedy-decodes every problem from "A: " and checks the answer exactly.
// The default Tokenizer (no merges) is byte-level.
template <class T>
EvalResult eval_reasoning(Model<T>& model, const std::vector<Problem>& problems, const Tokenizer& tok = {},
                          int batch = 64, int max_new = 200, size_t keep_failures = 8) {
    CHECK(model.cfg.vocab >= tok.vocab_size(), "model vocab %d < tokenizer vocab %d", model.cfg.vocab,
          tok.vocab_size());
    // Ids the tokenizer cannot decode (model vocab padding) end a completion.
    auto text_of = [&](const std::vector<int>& gen) {
        std::vector<int> ok;
        for (int t : gen) ok.push_back(t < tok.vocab_size() ? t : '?');
        return tok.decode(ok);
    };
    EvalResult r;
    for (size_t i0 = 0; i0 < problems.size(); i0 += batch) {
        const size_t i1 = std::min(problems.size(), i0 + batch);
        std::vector<std::vector<int>> prompts;
        for (size_t i = i0; i < i1; ++i) prompts.push_back(prompt_tokens(problems[i], tok));
        const auto outs = generate(model, prompts, max_new,
                                   [&](const std::vector<int>& gen) { return completion_done(text_of(gen)); });
        for (size_t i = i0; i < i1; ++i) {
            const Problem& p = problems[i];
            const std::string text = text_of(outs[i - i0]);
            const bool ok = completion_done(text) && extract_answer(text) == p.answer;
            auto& cell = r.by_task[{p.task, p.difficulty}];
            cell.first += ok;
            cell.second += 1;
            r.correct += ok;
            r.total += 1;
            if (!ok && r.failures.size() < keep_failures) r.failures.push_back({p, text});
        }
    }
    return r;
}
