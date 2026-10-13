// test_model.h — whole-model gradient checks and the BitLinear pieces.
#pragma once

#include "../src/model.h"
#include "test_util.h"

// Random parameters with enough spread that the loss is far from flat and
// MoE routing decisions are far from ties (a tie flipping under a 1e-5
// nudge would break the finite difference, not the gradient).
inline void randomize_for_gradcheck(Model<double>& m, Rng& rng) {
    for (const auto& p : m.infos)
        for (size_t i = 0; i < p.size; ++i)
            m.params[p.offset + i] = p.rows == 1 ? 1.0 + 0.3 * rng.normal() : 0.4 * rng.normal();
    m.weights_dirty = true;
}

inline void model_gradcheck(const char* what, Config cfg, uint64_t seed) {
    Rng rng(seed);
    Model<double> m(cfg);
    randomize_for_gradcheck(m, rng);
    const int B = 2, Tn = 5, N = B * Tn;
    std::vector<int> tokens(N), targets(N);
    for (int n = 0; n < N; ++n) {
        tokens[n] = int(rng.below(cfg.vocab));
        targets[n] = int(rng.below(cfg.vocab));
    }
    targets[3] = -1;  // an ignored position
    m.zero_grad();
    m.forward(tokens.data(), targets.data(), B, Tn);
    m.backward();
    const std::vector<double> analytic = m.grads;
    auto loss = [&] { return m.forward(tokens.data(), targets.data(), B, Tn); };
    // Check every tensor separately, so a failure names the culprit.
    double worst = 0.0;
    for (const auto& p : m.infos) {
        std::vector<double> x(m.params.begin() + p.offset, m.params.begin() + p.offset + p.size);
        std::vector<double> g(analytic.begin() + p.offset, analytic.begin() + p.offset + p.size);
        auto loss_p = [&] {
            std::copy(x.begin(), x.end(), m.params.begin() + p.offset);
            return loss();
        };
        const double e = grad_error(x, g, loss_p, rng, 60);
        std::copy(x.begin(), x.end(), m.params.begin() + p.offset);
        worst = std::max(worst, e);
        EXPECT(e < 1e-6, "%s: gradient of %s off by %g", what, p.name.c_str(), e);
    }
    std::printf("  %s: %zu params in %zu tensors, worst gradient error %.1e\n", what,
                m.num_params(), m.infos.size(), worst);
}

inline void test_model_gradients() {
    Config dense;
    dense.vocab = 11;
    dense.seq_len = 8;
    dense.d_model = 8;
    dense.n_layers = 2;
    dense.n_heads = 2;
    dense.ffn_hidden = 12;
    model_gradcheck("dense", dense, 21);

    Config moe = dense;
    moe.n_experts = 4;
    moe.top_k = 2;
    moe.expert_hidden = 6;
    moe.n_shared = 1;
    model_gradcheck("moe+shared", moe, 22);

    moe.n_shared = 0;
    moe.top_k = 1;
    model_gradcheck("moe top-1", moe, 23);
}

// The float model must compute the same thing as the double one.
inline void test_model_float_vs_double() {
    Config cfg;
    cfg.vocab = 37;
    cfg.seq_len = 24;
    cfg.d_model = 48;
    cfg.n_layers = 2;
    cfg.n_heads = 3;
    cfg.n_experts = 4;
    cfg.top_k = 2;
    cfg.n_shared = 1;
    cfg.expert_hidden = 40;
    Model<float> mf(cfg);
    Model<double> md(cfg);
    mf.init(5);
    std::copy(mf.params.begin(), mf.params.end(), md.params.begin());
    Rng rng(6);
    const int B = 3, Tn = 24, N = B * Tn;
    std::vector<int> tokens(N), targets(N);
    for (int n = 0; n < N; ++n) tokens[n] = int(rng.below(cfg.vocab)), targets[n] = int(rng.below(cfg.vocab));
    mf.zero_grad();
    md.zero_grad();
    const double lf = mf.forward(tokens.data(), targets.data(), B, Tn);
    const double ld = md.forward(tokens.data(), targets.data(), B, Tn);
    mf.backward();
    md.backward();
    EXPECT(std::abs(lf - ld) < 1e-5 * ld, "float loss %.8f vs double %.8f", lf, ld);
    const double eg = max_abs_diff(mf.grads.data(), md.grads.data(), md.grads.size()) /
                      max_abs(md.grads.data(), md.grads.size());
    EXPECT(eg < 1e-4, "float vs double gradients differ by %g", eg);
    // A fresh model predicts roughly uniformly: loss ~ log(vocab).
    EXPECT(std::abs(ld - std::log(double(cfg.vocab))) < 0.1, "initial loss %g vs log V %g", ld,
           std::log(double(cfg.vocab)));
}

inline void test_bitlinear() {
    Rng rng(31);
    // Ternary weights: values in {-gamma, 0, +gamma}, gamma = mean |w|.
    auto w = random_vec<double>(1000, rng);
    std::vector<double> wq(w.size());
    const double gamma = ternary_quantize(wq.data(), w.data(), long(w.size()));
    double mean_abs = 0;
    for (double x : w) mean_abs += std::abs(x);
    mean_abs /= double(w.size());
    EXPECT(std::abs(gamma - mean_abs) < 1e-7, "gamma %g vs mean |w| %g", gamma, mean_abs);
    int counts[3] = {0, 0, 0};
    for (size_t i = 0; i < w.size(); ++i) {
        const double r = wq[i] / gamma;
        const bool ok = std::abs(r) < 1e-12 || std::abs(std::abs(r) - 1) < 1e-12;
        EXPECT(ok, "quantized weight %g is not in {-g, 0, g}", wq[i]);
        if (!ok) break;
        ++counts[int(std::lround(r)) + 1];
        // round(w / gamma) clipped to [-1, 1].
        const double expect = std::clamp(std::nearbyint(w[i] / gamma), -1.0, 1.0);
        EXPECT(std::lround(r) == std::lround(expect), "w=%g quantized to %g", w[i], r);
    }
    EXPECT(counts[0] > 200 && counts[1] > 200 && counts[2] > 200, "odd ternary split %d/%d/%d",
           counts[0], counts[1], counts[2]);

    // int8 activations: per-row max maps to ±127; error at most half a step.
    const int N = 4, C = 50;
    auto x = random_vec<double>(N * C, rng);
    std::vector<double> xq(x.size());
    act_quantize(xq.data(), x.data(), N, C);
    for (int n = 0; n < N; ++n) {
        double amax = 0;
        for (int c = 0; c < C; ++c) amax = std::max(amax, std::abs(x[n * C + c]));
        const double step = amax / 127;
        for (int c = 0; c < C; ++c) {
            const double q = xq[n * C + c] / step;
            EXPECT(std::abs(q - std::nearbyint(q)) < 1e-9 && std::abs(q) <= 127 + 1e-9, "bad level %g", q);
            EXPECT(std::abs(xq[n * C + c] - x[n * C + c]) <= step / 2 + 1e-12, "act error too big");
        }
    }

    // A whole ternary model: quantized copies, a real gradient.
    Config cfg;
    cfg.vocab = 19;
    cfg.seq_len = 8;
    cfg.d_model = 16;
    cfg.n_layers = 1;
    cfg.n_heads = 2;
    cfg.ffn_hidden = 24;
    cfg.ternary = true;
    Model<double> m(cfg);
    m.init(7);
    const int B = 1, Tn = 8;
    std::vector<int> tok(Tn), tgt(Tn);
    for (int i = 0; i < Tn; ++i) tok[i] = int(rng.below(cfg.vocab)), tgt[i] = int(rng.below(cfg.vocab));
    m.zero_grad();
    const double lq = m.forward(tok.data(), tgt.data(), B, Tn);
    m.backward();
    for (const auto& p : m.infos) {
        if (!p.ternary) continue;
        // Every ternary tensor's quantized copy has at most 3 distinct values.
        std::vector<double> vals(m.qparams.begin() + p.offset, m.qparams.begin() + p.offset + p.size);
        std::sort(vals.begin(), vals.end());
        vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
        EXPECT(vals.size() <= 3, "%s has %zu distinct quantized values", p.name.c_str(), vals.size());
    }
    // Quantization is really applied: the same weights in full precision
    // give a different loss.
    Config fp = cfg;
    fp.ternary = false;
    Model<double> mfp(fp);
    mfp.params = m.params;
    const double lfp = mfp.forward(tok.data(), tgt.data(), B, Tn);
    EXPECT(std::abs(lq - lfp) > 1e-6, "ternary loss %g equals full-precision loss %g", lq, lfp);
    double gnorm = 0;
    for (double g : m.grads) gnorm += g * g;
    EXPECT(gnorm > 0 && std::isfinite(gnorm), "ternary model has no gradient");
}

inline void test_model() {
    test_model_gradients();
    test_model_float_vs_double();
    test_bitlinear();
}
