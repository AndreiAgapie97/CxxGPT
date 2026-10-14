// generate.h — batched text generation with the KV cache, and the
// reasoning evaluation built on it.
#pragma once

#include <algorithm>
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

// Continues every prompt by max_new tokens, all in one batch, or by fewer
// when the context runs out: with prompts of up to L tokens, at most
// seq_len - L + 1 (the last token is never fed back, so it needs no slot).
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
    CHECK(max_new > 0, "nothing to generate: max_new is %d", max_new);
    CHECK(L < model.cfg.seq_len, "a prompt of %d tokens fills the model's context of %d", L, model.cfg.seq_len);
    // Positions fed: the prompt, then each picked token but the last.
    const int total = std::min(model.cfg.seq_len, L - 1 + max_new);
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
    // Batches of similar prompt lengths: prompts are padded to the longest
    // in their batch, and the context left after it is all the room any of
    // them has to answer in. A prompt that fills the whole context cannot be
    // answered and counts as wrong.
    const size_t n = problems.size();
    std::vector<std::vector<int>> prompts(n);
    std::vector<size_t> order;
    for (size_t i = 0; i < n; ++i) {
        prompts[i] = prompt_tokens(problems[i], tok);
        if (int(prompts[i].size()) < model.cfg.seq_len) order.push_back(i);
    }
    std::stable_sort(order.begin(), order.end(),
                     [&](size_t x, size_t y) { return prompts[x].size() < prompts[y].size(); });
    std::vector<std::string> texts(n, "(the prompt does not fit in the context)");
    for (size_t k0 = 0; k0 < order.size(); k0 += size_t(batch)) {
        const size_t k1 = std::min(order.size(), k0 + size_t(batch));
        std::vector<std::vector<int>> group;
        for (size_t k = k0; k < k1; ++k) group.push_back(prompts[order[k]]);
        const auto outs = generate(model, group, max_new,
                                   [&](const std::vector<int>& gen) { return completion_done(text_of(gen)); });
        for (size_t k = k0; k < k1; ++k) texts[order[k]] = text_of(outs[k - k0]);
    }
    EvalResult r;
    for (size_t i = 0; i < n; ++i) {
        const Problem& p = problems[i];
        const bool ok = completion_done(texts[i]) && extract_answer(texts[i]) == p.answer;
        auto& cell = r.by_task[{p.task, p.difficulty}];
        cell.first += ok;
        cell.second += 1;
        r.correct += ok;
        r.total += 1;
        if (!ok && r.failures.size() < keep_failures) r.failures.push_back({p, texts[i]});
    }
    return r;
}
