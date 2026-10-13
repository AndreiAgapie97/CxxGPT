// test_model.h — whole-model gradient checks and the BitLinear pieces.
#pragma once

#include "../src/infer.h"
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

// KV-cache decoding of left-padded sequences must give the logits the full
// forward pass gives each sequence on its own.
inline void test_decode_matches_forward() {
    for (int variant = 0; variant < 2; ++variant) {
        Config cfg;
        cfg.vocab = 29;
        cfg.seq_len = 16;
        cfg.d_model = 32;
        cfg.n_layers = 2;
        cfg.n_heads = 4;
        if (variant == 1) {  // our architecture: ternary MoE with a shared expert
            cfg.n_experts = 4;
            cfg.top_k = 2;
            cfg.n_shared = 1;
            cfg.expert_hidden = 24;
            cfg.ternary = true;
        }
        Model<float> m(cfg);
        m.init(3);
        Rng rng(4);
        const std::vector<int> lens = {5, 9, 1};
        const int B = 3, L = 9, extra = 4;
        std::vector<std::vector<int>> seqs(B);
        std::vector<int> start(B);
        for (int b = 0; b < B; ++b) {
            for (int i = 0; i < lens[b] + extra; ++i) seqs[b].push_back(int(rng.below(cfg.vocab)));
            start[b] = L - lens[b];
        }
        // decoded[b][t]: logits after feeding token t of sequence b
        std::vector<std::vector<std::vector<float>>> decoded(B);
        m.decode_begin(B, start);
        for (int pos = 0; pos < L + extra; ++pos) {
            std::vector<int> cur(B);
            for (int b = 0; b < B; ++b) cur[b] = pos < start[b] ? 0 : seqs[b][pos - start[b]];
            const float* lg = m.decode_step(cur.data(), pos);
            for (int b = 0; b < B; ++b)
                if (pos >= start[b])
                    decoded[b].emplace_back(lg + size_t(b) * cfg.vocab, lg + size_t(b + 1) * cfg.vocab);
        }
        double worst = 0;
        for (int b = 0; b < B; ++b) {
            const int len = int(seqs[b].size());
            m.forward(seqs[b].data(), nullptr, 1, len);
            for (int t = 0; t < len; ++t) {
                const float* ref = m.output_row(t);
                double mx = 0, diff = 0;
                for (int v = 0; v < cfg.vocab; ++v) {
                    mx = std::max(mx, std::abs(double(ref[v])));
                    diff = std::max(diff, std::abs(double(ref[v]) - decoded[b][t][v]));
                }
                worst = std::max(worst, diff / mx);
            }
        }
        EXPECT(worst < 1e-4, "decode vs forward (%s): relative difference %g",
               variant ? "ternary MoE" : "dense", worst);
    }
}

// A ternary model on the bitplane int8 kernels computes what the training
// forward pass computes (which fake-quantizes with the same arithmetic).
inline void test_ternary_engine() {
    for (int variant = 0; variant < 2; ++variant) {
        Config cfg;
        cfg.vocab = 40;
        cfg.seq_len = 24;
        cfg.d_model = 64;
        cfg.n_layers = 2;
        cfg.n_heads = 4;
        cfg.ternary = true;
        if (variant == 1) {
            cfg.n_experts = 4;
            cfg.top_k = 2;
            cfg.n_shared = 1;
            cfg.expert_hidden = 64;
        }
        Model<float> m(cfg);
        m.init(17);
        // Larger weights than the init, so activations are far from zero
        // and the comparison is meaningful.
        Rng rng(18);
        for (auto& w : m.params) w = float(rng.normal() * 0.3);
        m.weights_dirty = true;
        TernaryEngine eng(m);
        std::vector<int> toks(cfg.seq_len);
        for (auto& t : toks) t = int(rng.below(cfg.vocab));
        m.decode_begin(1, {0});
        // Training and inference sum in different orders, so an activation
        // within ~1e-5 of an int8 rounding midpoint can round to the
        // neighbouring level in one and not the other. That is rare and
        // small; everything else must agree to float rounding.
        double worst = 0;
        std::vector<double> errs;
        for (int pos = 0; pos < cfg.seq_len; ++pos) {
            const float* ref = m.decode_step(&toks[pos], pos);
            const float* got = eng.step(toks[pos]);
            double mx = 0, diff = 0;
            for (int v = 0; v < cfg.vocab; ++v) {
                mx = std::max(mx, std::abs(double(ref[v])));
                diff = std::max(diff, std::abs(double(ref[v]) - got[v]));
            }
            worst = std::max(worst, diff / mx);
            errs.push_back(diff / mx);
        }
        std::sort(errs.begin(), errs.end());
        const double median = errs[errs.size() / 2];
        EXPECT(median < 1e-5 && worst < 2e-2, "ternary engine vs training forward (%s): median %g, worst %g",
               variant ? "MoE" : "dense", median, worst);
        // Bitplanes: 2 bits per ternary weight, versus 32 in training.
        size_t ternary_weights = 0;
        for (const auto& p : m.infos) ternary_weights += p.ternary ? p.size : 0;
        const size_t float_bytes = (m.num_params() - ternary_weights) * 4;
        EXPECT(eng.weight_bytes() < float_bytes + ternary_weights / 4 + 4096, "weight bytes %zu", eng.weight_bytes());
    }
}

inline void test_model() {
    test_ternary_engine();
    test_decode_matches_forward();
    test_model_gradients();
    test_model_float_vs_double();
    test_bitlinear();
}
