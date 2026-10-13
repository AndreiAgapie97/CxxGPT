// test_optim.h — the optimizer: Newton–Schulz orthogonalization, Muon and
// AdamW steps against a double-precision reimplementation, checkpoints of
// the Muon state, and Muon actually learning.
#pragma once

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../src/train.h"
#include "test_util.h"

namespace optim_test {

// Eigenvalues of a symmetric n x n matrix (cyclic Jacobi rotations).
inline std::vector<double> sym_eigenvalues(std::vector<double> A, int n) {
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j) off += A[i * n + j] * A[i * n + j];
        if (off < 1e-24) break;
        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q) {
                const double apq = A[p * n + q];
                if (std::abs(apq) < 1e-300) continue;
                const double theta = (A[q * n + q] - A[p * n + p]) / (2 * apq);
                const double t = (theta >= 0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1));
                const double c = 1 / std::sqrt(t * t + 1), s = t * c;
                for (int k = 0; k < n; ++k) {
                    const double akp = A[k * n + p], akq = A[k * n + q];
                    A[k * n + p] = c * akp - s * akq;
                    A[k * n + q] = s * akp + c * akq;
                }
                for (int k = 0; k < n; ++k) {
                    const double apk = A[p * n + k], aqk = A[q * n + k];
                    A[p * n + k] = c * apk - s * aqk;
                    A[q * n + k] = s * apk + c * aqk;
                }
            }
    }
    std::vector<double> e(n);
    for (int i = 0; i < n; ++i) e[i] = A[i * n + i];
    return e;
}

inline double rel_diff(const std::vector<float>& a, const std::vector<double>& b) {
    double d = 0, r = 0;
    for (size_t i = 0; i < a.size(); ++i) d += (a[i] - b[i]) * (a[i] - b[i]), r += b[i] * b[i];
    return std::sqrt(d / std::max(r, 1e-300));
}

inline void newton_schulz_cases() {
    Rng rng(81);
    // (m, n): small, and big enough to exercise the packed gemm's blocking.
    const int shapes[][2] = {{12, 40}, {7, 7}, {64, 200}};
    for (const auto& sh : shapes) {
        const int m = sh[0], n = sh[1];
        std::vector<double> xd = random_vec<double>(size_t(m) * n, rng);
        std::vector<float> xf(xd.begin(), xd.end());
        std::vector<double> sd(2 * size_t(m) * m + size_t(m) * n);
        std::vector<float> sf(sd.size());
        newton_schulz(xd.data(), m, n, 5, sd.data(), false);
        newton_schulz(xf.data(), m, n, 5, sf.data(), true);
        EXPECT(rel_diff(xf, xd) < 2e-5, "%dx%d: float and double Newton-Schulz differ by %g", m, n,
               rel_diff(xf, xd));
        // Singular values (square roots of the eigenvalues of X Xᵀ) end up near 1.
        std::vector<double> xxt(size_t(m) * m);
        gemm(m, m, n, MatView<double>{xd.data(), n, 1}, MatView<double>{xd.data(), 1, n}, xxt.data(), m, false);
        double lo = 1e9, hi = 0;
        for (double e : sym_eigenvalues(xxt, m)) lo = std::min(lo, std::sqrt(std::max(e, 0.0))), hi = std::max(hi, std::sqrt(e));
        EXPECT(lo > 0.6 && hi < 1.25, "%dx%d: singular values in [%.3f, %.3f], expected about [0.7, 1.2]", m, n, lo, hi);
    }
}

// One optimizer step, recomputed in double: AdamW elementwise, Muon per
// (tensor, expert copy, fused part) block.
struct RefOpt {
    std::vector<double> m, v;
    long step = 0;
    void update(const AdamW& o, const Model<float>& model, std::vector<double>& p, const std::vector<float>& g,
                double lr) {
        if (m.empty()) m.assign(p.size(), 0.0), v.assign(p.size(), 0.0);
        ++step;
        const double c1 = 1 / (1 - std::pow(o.beta1, double(step))), c2 = 1 / (1 - std::pow(o.beta2, double(step)));
        for (const auto& info : model.infos) {
            const double wd = info.decay ? o.weight_decay : 0.0;
            if (!(o.muon && muon_matrix(info))) {
                for (size_t i = info.offset; i < info.offset + info.size; ++i) {
                    m[i] = o.beta1 * m[i] + (1 - o.beta1) * g[i];
                    v[i] = o.beta2 * v[i] + (1 - o.beta2) * double(g[i]) * g[i];
                    p[i] -= lr * (m[i] * c1 / (std::sqrt(v[i] * c2) + o.eps) + wd * p[i]);
                }
                continue;
            }
            const int parts = muon_parts(info), r = info.rows / parts, c = info.cols;
            for (int copy = 0; copy < info.count; ++copy)
                for (int k = 0; k < parts; ++k) {
                    const size_t off = info.offset + copy * (info.size / info.count) + size_t(k) * r * c;
                    const bool tr = r > c;
                    std::vector<double> X(size_t(r) * c), s(2 * size_t(std::min(r, c)) * std::min(r, c) + size_t(r) * c);
                    for (int i = 0; i < r; ++i)
                        for (int j = 0; j < c; ++j) {
                            const size_t q = off + size_t(i) * c + j;
                            const double mu = o.muon_momentum;
                            m[q] = mu * m[q] + (1 - mu) * g[q];
                            X[tr ? size_t(j) * r + i : size_t(i) * c + j] = (1 - mu) * g[q] + mu * m[q];
                        }
                    newton_schulz(X.data(), std::min(r, c), std::max(r, c), o.muon_steps, s.data(), false);
                    const double scale = 0.2 * std::sqrt(double(std::max(r, c)));
                    for (int i = 0; i < r; ++i)
                        for (int j = 0; j < c; ++j) {
                            const size_t q = off + size_t(i) * c + j;
                            p[q] -= lr * (scale * X[tr ? size_t(j) * r + i : size_t(i) * c + j] + wd * p[q]);
                        }
                }
        }
    }
};

inline void step_matches_reference(bool muon, bool moe) {
    Config cfg;
    cfg.vocab = 11;
    cfg.seq_len = 8;
    cfg.d_model = 16;
    cfg.n_layers = 2;
    cfg.n_heads = 2;
    if (moe) cfg.n_experts = 3, cfg.top_k = 2, cfg.n_shared = 1, cfg.expert_hidden = 8;
    Model<float> model(cfg);
    model.init(3);
    AdamW opt;
    opt.muon = muon;
    RefOpt ref;
    std::vector<double> p(model.params.begin(), model.params.end());
    Rng rng(4);
    for (int step = 0; step < 3; ++step) {
        for (auto& g : model.grads) g = float(0.1 * rng.normal());
        ref.update(opt, model, p, model.grads, 1e-2);
        opt.update(model, 1e-2);
    }
    const double d = rel_diff(model.params, p);
    EXPECT(d < 1e-5, "%s%s: parameters after 3 steps differ from the double reference by %g",
           muon ? "Muon" : "AdamW", moe ? " (MoE)" : "", d);
    if (muon) {
        int matrices = 0;
        for (const auto& info : model.infos) matrices += muon_matrix(info);
        // wqkv, wo, wfc, wproj per layer, plus efc and eproj with experts.
        EXPECT(matrices == (moe ? 6 : 4) * cfg.n_layers, "%d tensors on Muon", matrices);
    }
}

// The Muon state survives a checkpoint, marked as Muon's.
inline void checkpoint_keeps_muon_state() {
    Config cfg;
    cfg.vocab = 11;
    cfg.seq_len = 8;
    cfg.d_model = 16;
    cfg.n_layers = 1;
    cfg.n_heads = 2;
    Model<float> model(cfg);
    model.init(5);
    AdamW opt;
    opt.muon = true;
    Rng rng(6);
    for (auto& g : model.grads) g = float(rng.normal());
    opt.update(model, 1e-2);
    const std::string path = "/tmp/cxxgpt_test_muon.ckpt";
    save_checkpoint(path, model, opt, rng);
    Model<float> model2(cfg);
    AdamW opt2;
    opt2.muon = true;
    load_checkpoint(path, model2, &opt2, nullptr);
    std::remove(path.c_str());
    EXPECT(opt2.step == 1 && opt2.m == opt.m && opt2.v == opt.v && model2.params == model.params,
           "Muon state changed in a checkpoint round trip");
}

// Muon trains: a small model learns a repeating sequence.
inline void muon_learns() {
    Config cfg;
    cfg.vocab = 32;
    cfg.seq_len = 16;
    cfg.d_model = 32;
    cfg.n_layers = 2;
    cfg.n_heads = 2;
    Model<float> model(cfg);
    model.init(7);
    AdamW opt;
    opt.muon = true;
    const int B = 4, N = B * cfg.seq_len;
    std::vector<int> tok(N), tgt(N);
    for (int n = 0; n < N; ++n) tok[n] = (n * 5) % cfg.vocab, tgt[n] = ((n + 1) * 5) % cfg.vocab;
    double first = 0, last = 0;
    for (int step = 0; step < 60; ++step) {
        model.zero_grad();
        last = model.forward(tok.data(), tgt.data(), B, cfg.seq_len, true);
        if (step == 0) first = last;
        model.backward(1.0);
        clip_grad_norm(model, 1.0);
        opt.update(model, 1e-2);
    }
    EXPECT(last < 0.1 * first, "Muon did not learn the pattern: loss %.3f -> %.3f", first, last);
}

}  // namespace optim_test

// Which tensors Muon takes, and how fused ones are split.
inline void muon_policy() {
    Config cfg;
    cfg.vocab = 11;
    cfg.seq_len = 8;
    cfg.d_model = 16;
    cfg.n_layers = 1;
    cfg.n_heads = 2;
    cfg.n_experts = 2;
    cfg.top_k = 1;
    cfg.n_shared = 1;
    const Model<float> model(cfg);
    std::string got;
    for (const auto& info : model.infos)
        got += info.name + (muon_matrix(info) ? ":" + std::to_string(muon_parts(info)) : "") + " ";
    const std::string want = "wte l0.ln1 l0.wqkv:3 l0.wo:1 l0.ln2 l0.wfc:2 l0.wproj:1 l0.router l0.efc:2 l0.eproj:1 lnf ";
    EXPECT(got == want, "Muon tensors: got '%s', want '%s'", got.c_str(), want.c_str());
}

inline void test_optim() {
    using namespace optim_test;
    muon_policy();
    newton_schulz_cases();
    step_matches_reference(false, false);
    step_matches_reference(true, false);
    step_matches_reference(true, true);
    checkpoint_keeps_muon_state();
    muon_learns();
}
