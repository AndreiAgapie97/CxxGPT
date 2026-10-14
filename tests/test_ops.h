// test_ops.h — gemm against a double reference, and every op's backward
// against finite differences of its forward.
#pragma once

#include "../src/ops.h"
#include "test_util.h"

// gemm: every edge of the blocking (MR, NR, KC tails), transposed views,
// accumulate and overwrite, threaded and serial, against double sums of the
// same float inputs.
inline void test_gemm() {
    using namespace gemm_detail;
    Rng rng(11);
    const int sizes[] = {1, 3, MR - 1, MR, MR + 1, NR - 1, NR + 1, 2 * NR + 5, 37};
    int cases = 0;
    double worst = 0.0;
    auto run = [&](int M, int N, int K, bool ta, bool tb, bool acc, bool threaded) {
        std::vector<float> A = random_vec<float>(size_t(M) * K, rng);
        std::vector<float> B = random_vec<float>(size_t(K) * N, rng);
        std::vector<float> C0 = random_vec<float>(size_t(M) * (N + 3), rng);
        // ta: A is stored as [K, M] and read transposed; same for tb.
        const MatView<float> av = ta ? MatView<float>{A.data(), 1, M} : MatView<float>{A.data(), K, 1};
        const MatView<float> bv = tb ? MatView<float>{B.data(), 1, K} : MatView<float>{B.data(), N, 1};
        std::vector<float> C = C0;
        const long ldc = N + 3;  // a row stride wider than the row
        gemm(M, N, K, av, bv, C.data(), ldc, acc, threaded);
        // Big shapes: check a sample of rows (all their columns).
        const int row_step = M > 64 ? 17 : 1;
        for (int i = 0; i < M; i += row_step)
            for (int j = 0; j < N + 3; ++j) {
                const double c0 = C0[size_t(i) * ldc + j];
                const double got = C[size_t(i) * ldc + j];
                if (j >= N) {  // the padding columns must be untouched
                    EXPECT(got == c0, "gemm wrote past column N at (%d, %d)", i, j);
                    continue;
                }
                double ref = acc ? c0 : 0.0, mag = acc ? std::abs(c0) : 0.0;
                for (int k = 0; k < K; ++k) {
                    ref += double(av.at(i, k)) * bv.at(k, j);
                    mag += std::abs(double(av.at(i, k)) * bv.at(k, j));
                }
                const double err = std::abs(got - ref) / (mag + 1e-30);
                worst = std::max(worst, err);
                EXPECT(err < 1e-5, "gemm M=%d N=%d K=%d ta=%d tb=%d acc=%d thr=%d (%d,%d): %g vs %g",
                       M, N, K, ta, tb, acc, threaded, i, j, got, ref);
            }
        ++cases;
    };
    for (int M : sizes)
        for (int N : sizes)
            for (int K : {1, 5, KC - 1, KC + 3})
                run(M, N, K, rng.below(2), rng.below(2), rng.below(2), rng.below(2));
    // Big enough for several MC blocks, NC panels and threads.
    for (bool ta : {false, true})
        for (bool tb : {false, true}) run(2 * MC + 7, NC + NR + 3, 2 * KC + 9, ta, tb, true, true);
    // K = 0: overwrite zeroes C, accumulate leaves it.
    std::vector<float> C(6, 1.0f), Z;
    gemm(2, 3, 0, MatView<float>{Z.data(), 0, 1}, MatView<float>{Z.data(), 3, 1}, C.data(), 3, true);
    EXPECT(C[0] == 1.0f, "K=0 accumulate changed C");
    gemm(2, 3, 0, MatView<float>{Z.data(), 0, 1}, MatView<float>{Z.data(), 3, 1}, C.data(), 3, false);
    EXPECT(C[5] == 0.0f, "K=0 overwrite did not zero C");
    std::printf("  gemm: %d shapes, worst relative error %.2e\n", cases, worst);
}

inline void test_linear_grad() {
    Rng rng(1);
    const int N = 5, IC = 7, OC = 6;
    auto x = random_vec<double>(N * IC, rng), w = random_vec<double>(OC * IC, rng);
    auto g = random_vec<double>(N * OC, rng);
    std::vector<double> y(N * OC), dx(N * IC, 0.0), dw(OC * IC, 0.0);
    auto loss = [&] { linear_forward(y.data(), x.data(), w.data(), N, IC, OC); return project(y, g); };
    linear_backward(dx.data(), dw.data(), g.data(), x.data(), w.data(), N, IC, OC);
    const double ex = grad_error(x, dx, loss, rng), ew = grad_error(w, dw, loss, rng);
    EXPECT(ex < 1e-7 && ew < 1e-7, "linear: dx err %g, dw err %g", ex, ew);
}

inline void test_embedding_grad() {
    Rng rng(2);
    const int V = 5, C = 4, N = 7;
    const int tokens[N] = {0, 3, 3, 1, 4, 3, 0};  // repeats must add up
    auto wte = random_vec<double>(V * C, rng), g = random_vec<double>(N * C, rng);
    std::vector<double> out(N * C), dwte(V * C, 0.0);
    auto loss = [&] { embedding_forward(out.data(), tokens, wte.data(), N, C); return project(out, g); };
    embedding_backward(dwte.data(), g.data(), tokens, N, C);
    const double e = grad_error(wte, dwte, loss, rng);
    EXPECT(e < 1e-7, "embedding: dwte err %g", e);
}

inline void test_rmsnorm_grad() {
    Rng rng(3);
    const int N = 4, C = 9;
    auto x = random_vec<double>(N * C, rng), w = random_vec<double>(C, rng), g = random_vec<double>(N * C, rng);
    std::vector<double> out(N * C), rstd(N), dx(N * C, 0.0), dw(C, 0.0);
    auto loss = [&] { rmsnorm_forward(out.data(), rstd.data(), x.data(), w.data(), N, C); return project(out, g); };
    loss();
    rmsnorm_backward(dx.data(), dw.data(), g.data(), x.data(), w.data(), rstd.data(), N, C);
    const double ex = grad_error(x, dx, loss, rng), ew = grad_error(w, dw, loss, rng);
    EXPECT(ex < 1e-7 && ew < 1e-7, "rmsnorm: dx err %g, dw err %g", ex, ew);
}

inline void test_rope_grad() {
    Rng rng(4);
    const int B = 2, Tn = 5, C = 8, NH = 2;
    RopeTable<double> rope(Tn, C / NH);
    auto qkv = random_vec<double>(B * Tn * 3 * C, rng), g = random_vec<double>(B * Tn * 3 * C, rng);
    std::vector<double> tmp;
    auto loss = [&] { tmp = qkv; rope_forward(tmp.data(), rope, B, Tn, C, NH); return project(tmp, g); };
    std::vector<double> d = g;
    rope_backward(d.data(), rope, B, Tn, C, NH);
    const double e = grad_error(qkv, d, loss, rng);
    EXPECT(e < 1e-7, "rope: err %g", e);
    // Position 0 is not rotated; norms are preserved everywhere.
    tmp = qkv;
    rope_forward(tmp.data(), rope, B, Tn, C, NH);
    EXPECT(tmp[0] == qkv[0] && tmp[C] == qkv[C], "rope rotated position 0");
    double n0 = 0, n1 = 0;
    for (size_t i = 0; i < tmp.size(); ++i) n0 += qkv[i] * qkv[i], n1 += tmp[i] * tmp[i];
    EXPECT(std::abs(n0 - n1) < 1e-9 * n0, "rope changed the norm: %g vs %g", n0, n1);
}

inline void test_attention_grad() {
    Rng rng(5);
    const int B = 2, Tn = 5, C = 8, NH = 2;
    auto qkv = random_vec<double>(B * Tn * 3 * C, rng), g = random_vec<double>(B * Tn * C, rng);
    std::vector<double> out(B * Tn * C), att(B * NH * Tn * Tn), dqkv(qkv.size(), 0.0);
    auto loss = [&] { attention_forward(out.data(), att.data(), qkv.data(), B, Tn, C, NH); return project(out, g); };
    loss();
    attention_backward(dqkv.data(), g.data(), att.data(), qkv.data(), B, Tn, C, NH);
    const double e = grad_error(qkv, dqkv, loss, rng);
    EXPECT(e < 1e-7, "attention: err %g", e);
    // Causality: output at t must not depend on anything after t.
    loss();
    const std::vector<double> before = out;
    qkv[size_t(Tn - 1) * 3 * C + 0] += 1.0;          // q at the last position
    qkv[size_t(Tn - 1) * 3 * C + C + 1] += 1.0;      // k at the last position
    qkv[size_t(Tn - 1) * 3 * C + 2 * C + 2] += 1.0;  // v at the last position
    loss();
    for (int t = 0; t < Tn - 1; ++t)
        for (int c = 0; c < C; ++c)
            EXPECT(out[size_t(t) * C + c] == before[size_t(t) * C + c], "attention leaks the future at t=%d", t);
}

inline void test_swiglu_grad() {
    Rng rng(6);
    const int N = 3, F = 5;
    auto inp = random_vec<double>(N * 2 * F, rng, 2.0), g = random_vec<double>(N * F, rng);
    std::vector<double> out(N * F), d(inp.size(), 0.0);
    auto loss = [&] { swiglu_forward(out.data(), inp.data(), N, F); return project(out, g); };
    swiglu_backward(d.data(), g.data(), inp.data(), N, F);
    const double e = grad_error(inp, d, loss, rng);
    EXPECT(e < 1e-7, "swiglu: err %g", e);
}

inline void test_cross_entropy_grad() {
    Rng rng(7);
    const int N = 5, V = 7;
    const int targets[N] = {3, -1, 0, 6, 3};
    auto logits = random_vec<double>(N * V, rng, 3.0);
    std::vector<double> probs(N * V), losses(N), d(N * V);
    auto loss = [&] { probs = logits; return cross_entropy_forward(probs.data(), losses.data(), targets, N, V); };
    const double l = loss();
    cross_entropy_backward(d.data(), probs.data(), targets, N, V, 1.0 / 4);  // 4 counted rows
    const double e = grad_error(logits, d, loss, rng);
    EXPECT(e < 1e-7, "cross-entropy: err %g", e);
    double ref = 0;
    for (int n = 0; n < N; ++n) {
        if (targets[n] < 0) continue;
        double mx = -1e300, s = 0;
        for (int i = 0; i < V; ++i) mx = std::max(mx, logits[size_t(n) * V + i]);
        for (int i = 0; i < V; ++i) s += std::exp(logits[size_t(n) * V + i] - mx);
        ref += -(logits[size_t(n) * V + targets[n]] - mx - std::log(s));
    }
    EXPECT(std::abs(l - ref / 4) < 1e-12, "cross-entropy value %g vs %g", l, ref / 4);
    for (int i = 0; i < V; ++i) EXPECT(d[size_t(V) + i] == 0.0, "ignored row has a gradient");
}

// The float paths (packed gemm inside attention, threaded row loops) must
// match double on shapes big enough to hit kernel edges.
inline void test_attention_float_vs_double() {
    Rng rng(8);
    const int B = 2, Tn = 37, C = 48, NH = 3;
    auto qd = random_vec<double>(size_t(B) * Tn * 3 * C, rng), gd = random_vec<double>(size_t(B) * Tn * C, rng);
    std::vector<float> qf(qd.begin(), qd.end()), gf(gd.begin(), gd.end());
    std::vector<double> od(size_t(B) * Tn * C), ad(size_t(B) * NH * Tn * Tn), dd(qd.size(), 0.0);
    std::vector<float> of(od.size()), af(ad.size()), df(qd.size(), 0.0f);
    attention_forward(od.data(), ad.data(), qd.data(), B, Tn, C, NH);
    attention_forward(of.data(), af.data(), qf.data(), B, Tn, C, NH);
    attention_backward(dd.data(), gd.data(), ad.data(), qd.data(), B, Tn, C, NH);
    attention_backward(df.data(), gf.data(), af.data(), qf.data(), B, Tn, C, NH);
    const double eo = max_abs_diff(of.data(), od.data(), od.size()) / max_abs(od.data(), od.size());
    const double eg = max_abs_diff(df.data(), dd.data(), dd.size()) / max_abs(dd.data(), dd.size());
    EXPECT(eo < 1e-5 && eg < 1e-5, "attention float vs double: out %g, grad %g", eo, eg);
}

// Sizes big enough that every row-parallel op actually runs on the pool's
// workers (small grad-check shapes run on the calling thread only).
inline void test_ops_threaded_float_vs_double() {
    Rng rng(9);
    const int N = 700, C = 96, F = 80;
    auto xd = random_vec<double>(size_t(N) * C, rng), wd = random_vec<double>(C, rng);
    auto gd = random_vec<double>(size_t(N) * C, rng);
    std::vector<float> xf(xd.begin(), xd.end()), wf(wd.begin(), wd.end()), gf(gd.begin(), gd.end());
    std::vector<double> od(xd.size()), rd(N), dxd(xd.size(), 0.0), dwd(C, 0.0);
    std::vector<float> of(xd.size()), rf(N), dxf(xd.size(), 0.0f), dwf(C, 0.0f);
    rmsnorm_forward(od.data(), rd.data(), xd.data(), wd.data(), N, C);
    rmsnorm_forward(of.data(), rf.data(), xf.data(), wf.data(), N, C);
    rmsnorm_backward(dxd.data(), dwd.data(), gd.data(), xd.data(), wd.data(), rd.data(), N, C);
    rmsnorm_backward(dxf.data(), dwf.data(), gf.data(), xf.data(), wf.data(), rf.data(), N, C);
    auto rel = [](const std::vector<float>& f, const std::vector<double>& d) {
        return max_abs_diff(f.data(), d.data(), d.size()) / max_abs(d.data(), d.size());
    };
    EXPECT(rel(of, od) < 1e-5 && rel(dxf, dxd) < 1e-5 && rel(dwf, dwd) < 1e-5,
           "threaded rmsnorm: out %g dx %g dw %g", rel(of, od), rel(dxf, dxd), rel(dwf, dwd));

    auto id = random_vec<double>(size_t(N) * 2 * F, rng, 3.0), sd = random_vec<double>(size_t(N) * F, rng);
    std::vector<float> iff(id.begin(), id.end()), sf(sd.begin(), sd.end());
    std::vector<double> sod(size_t(N) * F), sdd(id.size(), 0.0);
    std::vector<float> sof(size_t(N) * F), sdf(id.size(), 0.0f);
    swiglu_forward(sod.data(), id.data(), N, F);
    swiglu_forward(sof.data(), iff.data(), N, F);
    swiglu_backward(sdd.data(), sd.data(), id.data(), N, F);
    swiglu_backward(sdf.data(), sf.data(), iff.data(), N, F);
    EXPECT(rel(sof, sod) < 1e-5 && rel(sdf, sdd) < 1e-5, "threaded swiglu: out %g d %g",
           rel(sof, sod), rel(sdf, sdd));

    std::vector<double> ad(xd.size());
    std::vector<float> af(xd.size());
    add_forward(ad.data(), xd.data(), gd.data(), long(xd.size()));
    add_forward(af.data(), xf.data(), gf.data(), long(xd.size()));
    EXPECT(rel(af, ad) < 1e-6, "threaded add");
    zero_fill(af.data(), af.size());
    EXPECT(*std::max_element(af.begin(), af.end()) == 0.0f && *std::min_element(af.begin(), af.end()) == 0.0f,
           "zero_fill left nonzeros");
}

// fast_exp against std::exp over the whole float range it supports.
inline void test_fast_exp() {
    EXPECT(std::isnan(fast_exp(std::numeric_limits<float>::quiet_NaN())), "fast_exp did not propagate NaN");
    double worst = 0.0;
    for (float x = -87.0f; x <= 88.0f; x += 0.0137f) {
        const double ref = std::exp(double(x));
        worst = std::max(worst, std::abs(double(fast_exp(x)) - ref) / ref);
    }
    EXPECT(worst < 3e-7, "fast_exp relative error %g", worst);
    EXPECT(fast_exp(-1000.0f) >= 0.0f && fast_exp(-1000.0f) < 1e-37f, "fast_exp(-1000) = %g", fast_exp(-1000.0f));
    EXPECT(std::isfinite(fast_exp(1000.0f)), "fast_exp(1000) overflowed");
    EXPECT(fast_exp(0.0f) == 1.0f, "fast_exp(0) = %.9g", fast_exp(0.0f));
}

inline void test_ops() {
    test_fast_exp();
    test_ops_threaded_float_vs_double();
    test_gemm();
    test_linear_grad();
    test_embedding_grad();
    test_rmsnorm_grad();
    test_rope_grad();
    test_attention_grad();
    test_swiglu_grad();
    test_cross_entropy_grad();
    test_attention_float_vs_double();
}
