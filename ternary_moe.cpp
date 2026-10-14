// ternary_moe.cpp — tests and benchmarks for the inference kernels in src/ternary.h
//
//
// (two-bitplane ternary weights, int8 activations, AVX2 / AVX-512 / VNNI).
//
// Build:
//   g++ -O3 -march=native -std=c++20 -pthread ternary_moe.cpp -o ternary_moe
//
// Run:
//   ./ternary_moe          # tests + benchmark
//   ./ternary_moe -t N     # force N threads
//   ./ternary_moe --tests  # tests only; exit status 0 iff all passed

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "src/ternary.h"

// =============================================================================
// Correctness tests
// =============================================================================
static int g_failures = 0;

static void check(bool ok, const char* what) {
    if (!ok) { printf("  FAIL: %s\n", what); ++g_failures; }
}

static std::vector<float> random_activations(int K, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> v(size_t(K), 0.0f);
    for (auto& x : v) x = nd(rng);
    return v;
}

// L2 error of got relative to ref.
static float rel_err(const std::vector<float>& ref, const std::vector<float>& got) {
    double se = 0.0, sr = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = double(ref[i]) - double(got[i]);
        se += d * d;
        sr += double(ref[i]) * double(ref[i]);
    }
    return float(std::sqrt(se / std::max(sr, 1e-30)));
}

static void reference_gemv(const std::vector<int8_t>& src, const std::vector<float>& X,
                           int M, int K, float scale, std::vector<float>& Y) {
    Y.assign(size_t(M), 0.0f);
    for (int r = 0; r < M; ++r) {
        double s = 0.0;
        for (int k = 0; k < K; ++k) s += double(src[size_t(r) * K + k]) * double(X[k]);
        Y[r] = float(s * scale);
    }
}

static void run_case(const char* label, const std::vector<int8_t>& src,
                     const std::vector<float>& X, int M, int K) {
    constexpr float kScale = 0.05f;
    TernaryMatrixBitplane Wb(M, K, kScale);
    pack_bitplane(src, Wb);

    // round-trip: packing must be recoverable
    for (int r = 0; r < M; ++r)
        for (int k = 0; k < K; ++k)
            if (Wb.weight(r, k) != src[size_t(r) * K + k]) {
                printf("  FAIL: %s pack round-trip at (%d,%d)\n", label, r, k);
                ++g_failures;
                return;
            }

    std::vector<float> ref;
    reference_gemv(src, X, M, K, kScale, ref);

    // interleaved layout must agree too, when K allows it
    if (K % cfg::kInterleavedPerWord == 0) {
        TernaryMatrixInterleaved Wi(M, K, kScale);
        pack_interleaved(src, Wi);
        std::vector<float> y(size_t(M), 0.0f);
        gemv_interleaved(X.data(), Wi, y.data());
        float m = 0.0f;
        for (int r = 0; r < M; ++r) m = std::max(m, std::fabs(y[r] - ref[r]));
        char buf[128];
        snprintf(buf, sizeof buf, "%s / interleaved (maxdiff %.2e)", label, m);
        check(m < 1e-3f, buf);
    }

    std::vector<int8_t> xq(K);
    const float x_scale = quantize_int8(X.data(), K, xq.data());
    std::vector<float> int8_reference(M);
    gemv_bitplane_int8_scalar(xq.data(), x_scale, Wb, int8_reference.data(), 0, M);
    for (const auto& k : all_kernels()) {
        std::vector<float> y(size_t(M), NAN);
        k.fn(X.data(), Wb, y.data());
        float m = 0.0f, rmax = 0.0f;
        for (int r = 0; r < M; ++r) {
            m = std::max(m, std::fabs(y[r] - ref[r]));
            rmax = std::max(rmax, std::fabs(ref[r]));
        }
        // int8 quantization error scales with the largest activation
        const float tol = k.qrows ? 0.05f * std::max(rmax, 1e-6f) + 1e-4f : 1e-3f;
        char buf[160];
        snprintf(buf, sizeof buf, "%s / %s (maxdiff %.2e, tol %.2e)", label, k.name, m, tol);
        check(m <= tol, buf);
        if (k.qrows) {
            bool exact = true;
            for (int r = 0; r < M; ++r) exact &= std::isfinite(y[r]) && y[r] == int8_reference[r];
            snprintf(buf, sizeof buf, "%s / %s equals integer scalar reference", label, k.name);
            check(exact, buf);
        }
    }
}

// The quantizer's contract with the int8 kernels: the largest magnitude maps
// to exactly +-127 and nothing maps to -128, which the kernels cannot negate.
static void test_quantizer() {
    const std::vector<float> X = {-3.0f, 3.0f, -2.9999f, 1.5f, -1e-9f, 0.0f, -3.0f, 0.0119f};
    std::vector<int8_t> q(X.size());
    const float s = quantize_int8(X.data(), int(X.size()), q.data());

    check(q[0] == -cfg::kInt8Max && q[1] == cfg::kInt8Max && q[2] == -cfg::kInt8Max &&
              q[6] == -cfg::kInt8Max,
          "quantizer: +-amax -> +-127");
    check(*std::min_element(q.begin(), q.end()) >= -cfg::kInt8Max, "quantizer: emits -128");
    float err = 0.0f;
    for (size_t i = 0; i < X.size(); ++i) err = std::max(err, std::fabs(float(q[i]) * s - X[i]));
    check(err <= 0.5f * s * 1.0001f, "quantizer: error above half a step");

    // SIMD path against the scalar loop, bit for bit: every tail length,
    // exact .5 ties (amax 127 makes the scale 1), and +-amax at the edges.
    auto scalar_quantize = [](const std::vector<float>& v, std::vector<int8_t>& o) {
        float a = 0.0f;
        for (float f : v) a = std::max(a, std::fabs(f));
        const float s = a / float(cfg::kInt8Max), inv = 1.0f / s;
        for (size_t i = 0; i < v.size(); ++i)
            o[i] = int8_t(std::min(cfg::kInt8Max, std::max(-cfg::kInt8Max, int(std::lrintf(v[i] * inv)))));
        return s;
    };
    int mismatched = 0;
    for (int len : {1, 7, 8, 9, 15, 16, 17, 31, 33, 64, 100, 512, 2048}) {
        std::vector<float> v = random_activations(len, uint32_t(len) * 31u);
        if (len >= 9) {
            v[0] = 127.0f;
            const float ties[] = {0.5f, 1.5f, 2.5f, -0.5f, -1.5f, -2.5f, 126.5f, -126.5f};
            for (int i = 0; i < 8; ++i) v[size_t(1 + i)] = ties[i];
        }
        std::vector<int8_t> got(v.size()), want(v.size());
        const float sg = quantize_int8(v.data(), len, got.data());
        const float sw = scalar_quantize(v, want);
        if (sg != sw || got != want) ++mismatched;
    }
    check(mismatched == 0, "quantizer: SIMD path differs from scalar");

    const std::vector<float> Z(64, 0.0f);
    std::vector<int8_t> qz(Z.size(), 1);
    const float sz = quantize_int8(Z.data(), int(Z.size()), qz.data());
    check(sz == 0.0f && std::all_of(qz.begin(), qz.end(), [](int8_t v) { return v == 0; }),
          "quantizer: all-zero input");
}

// dot_f32 and silu_mul against double precision, over every SIMD tail length
// and SiLU's whole input range, including where exp(-g) saturates.
static void test_fp32_helpers() {
    int bad_dot = 0;
    for (int n : {1, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 512, 2049}) {
        const std::vector<float> a = random_activations(n, uint32_t(n)), b = random_activations(n, uint32_t(n) + 7);
        double ref = 0.0, mag = 0.0;
        for (int i = 0; i < n; ++i) { ref += double(a[i]) * b[i]; mag += std::fabs(double(a[i]) * b[i]); }
        if (std::fabs(dot_f32(a.data(), b.data(), n) - ref) > 1e-6 * mag) ++bad_dot;
    }
    check(bad_dot == 0, "dot_f32: error above 1e-6 of sum |a*b|");

    std::vector<float> g, u;
    for (float v = -100.0f; v <= 100.0f; v += 0.0137f) { g.push_back(v); u.push_back(1.0f - 0.003f * v); }
    for (float v : {0.0f, -0.0f, 1e-30f, -1e-30f, 87.0f, -87.0f, 88.3f, -88.3f, 120.0f, -120.0f}) {
        g.push_back(v);
        u.push_back(0.5f);
    }
    std::vector<float> out(g.size());
    silu_mul(g.data(), u.data(), out.data(), int(g.size()));
    double worst = 0.0;
    for (size_t i = 0; i < g.size(); ++i) {
        const double ref = double(g[i]) * u[i] / (1.0 + std::exp(-double(g[i])));
        worst = std::max(worst, std::fabs(out[i] - ref) / (std::fabs(ref) + 1e-30));
    }
    char buf[96];
    snprintf(buf, sizeof buf, "silu_mul: worst error %.2e (relative; absolute below 1e-30)", worst);
    check(worst < 1e-6, buf);
}

// parallel_for must hand out every index exactly once: across back-to-back
// jobs (workers still spinning), after an idle gap long enough for workers to
// fall asleep, and with pools created and destroyed around it.
static void test_thread_pool() {
    for (int nt : {2, 3, 4}) {
        ThreadPool tp(nt);
        std::mt19937 rng{uint32_t(nt)};
        std::uniform_int_distribution<int> size(1, 3000);
        int bad_jobs = 0;
        for (int job = 0; job < 400; ++job) {
            if (job == 200)  // let every worker go to sleep, then wake them
                std::this_thread::sleep_for(std::chrono::microseconds(3 * ThreadPool::kSpinMicros));
            const int total = size(rng);
            std::vector<std::atomic<int>> hits(static_cast<size_t>(total));
            for (auto& h : hits) h.store(0, std::memory_order_relaxed);
            tp.parallel_for(total, [&](int r0, int r1) {
                for (int r = r0; r < r1; ++r) hits[size_t(r)].fetch_add(1, std::memory_order_relaxed);
            });
            for (auto& h : hits)
                if (h.load(std::memory_order_relaxed) != 1) { ++bad_jobs; break; }
        }
        char buf[96];
        snprintf(buf, sizeof buf, "thread pool, %d threads: %d of 400 jobs missed or repeated rows", nt, bad_jobs);
        check(bad_jobs == 0, buf);
    }
}

// Two threads calling the kernels at once must each get their own answer:
// no shared quantization scratch, no clobbered thread pool job.
static void test_concurrent_callers() {
    const int M = pool().size() * cfg::kMinRowsPerThread * 2, K = 256;
    constexpr int kReps = 50;
    TernaryMatrixBitplane W(M, K, 0.05f);
    pack_bitplane(random_ternary(M, K, 0.6, 501), W);
    const std::vector<float> xa = random_activations(K, 502), xb = random_activations(K, 503);

    for (const auto& k : all_kernels()) {
        std::vector<float> want_a(size_t(M), 0.0f), want_b(size_t(M), 0.0f);
        k.fn(xa.data(), W, want_a.data());
        k.fn(xb.data(), W, want_b.data());

        std::atomic<int> mismatches{0};
        auto hammer = [&](const std::vector<float>* x, const std::vector<float>* want) {
            std::vector<float> y(size_t(M), 0.0f);
            for (int i = 0; i < kReps; ++i) {
                k.fn(x->data(), W, y.data());
                if (y != *want) ++mismatches;
            }
        };
        std::thread ta(hammer, &xa, &want_a), tb(hammer, &xb, &want_b);
        ta.join();
        tb.join();

        char buf[128];
        snprintf(buf, sizeof buf, "concurrent callers / %s (%d of %d calls wrong)", k.name,
                 mismatches.load(), 2 * kReps);
        check(mismatches == 0, buf);
    }
}

// Double-precision MoE forward, reading weights straight off the bitplanes.
static std::vector<float> reference_moe(const TernaryMoELayer& L, const std::vector<float>& x) {
    auto gemv = [](const TernaryMatrixBitplane& W, const std::vector<double>& v) {
        std::vector<double> y(size_t(W.rows));
        for (int r = 0; r < W.rows; ++r) {
            double s = 0.0;
            for (int k = 0; k < W.cols; ++k) s += W.weight(r, k) * v[size_t(k)];
            y[size_t(r)] = s * W.scale;
        }
        return y;
    };

    const int D = L.d_model;
    const std::vector<double> xd(x.begin(), x.end());
    std::vector<double> logits(size_t(L.num_experts), 0.0);
    for (int e = 0; e < L.num_experts; ++e)
        for (int d = 0; d < D; ++d) logits[size_t(e)] += double(L.W_router[size_t(e) * D + d]) * xd[size_t(d)];

    std::vector<int> order(size_t(L.num_experts));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return logits[a] > logits[b]; });

    double sum_exp = 0.0;
    for (int i = 0; i < L.top_k; ++i) sum_exp += std::exp(logits[order[i]] - logits[order[0]]);

    std::vector<double> out(size_t(D), 0.0);
    for (int i = 0; i < L.top_k; ++i) {
        const TernaryExpert& E = L.experts[size_t(order[i])];
        const double w = std::exp(logits[order[i]] - logits[order[0]]) / sum_exp;
        const std::vector<double> g = gemv(E.W_gate, xd), u = gemv(E.W_up, xd);
        std::vector<double> h(g.size());
        for (size_t j = 0; j < h.size(); ++j) h[j] = g[j] / (1.0 + std::exp(-g[j])) * u[j];
        const std::vector<double> y = gemv(E.W_down, h);
        for (int d = 0; d < D; ++d) out[size_t(d)] += w * y[size_t(d)];
    }
    return std::vector<float>(out.begin(), out.end());
}

// Full layer against the reference, for every kernel: routing, softmax
// weights, and (for int8 kernels) the shared once-per-token quantization.
static void test_moe() {
    constexpr int D = 64, E = 6, TopK = 2;
    TernaryMoELayer L(D, E, TopK, all_kernels().front());
    L.randomize(0.6, 2024);

    for (uint32_t tok_seed : {11u, 12u, 13u}) {
        const std::vector<float> x = random_activations(D, tok_seed);
        const std::vector<float> ref = reference_moe(L, x);
        for (const auto& k : all_kernels()) {
            L.kernel = k;
            std::vector<float> out(size_t(D), NAN);
            L.forward(x.data(), out.data());
            const float err = rel_err(ref, out);
            // int8 error compounds through gate/up, SiLU and down
            const float tol = k.qrows ? 5e-2f : 1e-5f;
            char buf[160];
            snprintf(buf, sizeof buf, "moe token %u / %s (rel err %.2e, tol %.0e)", tok_seed,
                     k.name, err, tol);
            check(err <= tol, buf);
        }
    }
}

static void run_tests() {
    printf("Correctness tests\n");
    constexpr int M = 8, K = 64;
    const std::vector<float> X = random_activations(K, 4242);

    test_quantizer();

    run_case("all zero", std::vector<int8_t>(size_t(M) * K, 0), X, M, K);
    run_case("all +1", std::vector<int8_t>(size_t(M) * K, 1), X, M, K);
    run_case("all -1", std::vector<int8_t>(size_t(M) * K, -1), X, M, K);

    {   // exactly one non-zero weight, at a word boundary
        std::vector<int8_t> s(size_t(M) * K, 0);
        s[size_t(3) * K + 32] = -1;
        run_case("single -1 at k=32", s, X, M, K);
    }
    {   // alternating, exercises both planes in every word
        std::vector<int8_t> s(size_t(M) * K);
        for (size_t i = 0; i < s.size(); ++i) s[i] = int8_t((i % 3) - 1);
        run_case("alternating", s, X, M, K);
    }
    // every activation quantizes to -127, the most negative value allowed
    run_case("all activations -amax", random_ternary(M, K, 0.5, 11),
             std::vector<float>(size_t(K), -1.0f), M, K);
    for (double zf : {0.0, 0.5, 0.9, 0.99})
        run_case("random", random_ternary(M, K, zf, unsigned(zf * 1000) + 1), X, M, K);

    // odd row counts and multiple words per row
    run_case("M=1 K=256", random_ternary(1, 256, 0.6, 77), random_activations(256, 9), 1, 256);
    run_case("M=33 K=96", random_ternary(33, 96, 0.6, 78), random_activations(96, 10), 33, 96);
    // with K=64, 96, 128 and 256 elsewhere, every remainder of the VNNI
    // kernels' 2- and 8-word unrolls
    for (int k : {32, 160, 192, 224}) {
        char label[32];
        snprintf(label, sizeof label, "M=5 K=%d", k);
        run_case(label, random_ternary(5, k, 0.5, unsigned(k)), random_activations(k, unsigned(k) + 1), 5, k);
    }
    {   // enough rows to go through the thread pool, split unevenly
        const int Mt = pool().size() * cfg::kMinRowsPerThread + 37;
        char label[32];
        snprintf(label, sizeof label, "M=%d K=128", Mt);
        run_case(label, random_ternary(Mt, 128, 0.6, 79), random_activations(128, 12), Mt, 128);
    }

    test_fp32_helpers();
    test_thread_pool();
    test_concurrent_callers();
    test_moe();

    printf(g_failures == 0 ? "  all passed\n\n" : "  %d FAILURES\n\n", g_failures);
}

// =============================================================================
// Benchmark
// =============================================================================
struct Stats { double mean, median, sd, min, max; };

template <typename F>
Stats bench(F&& f, int iters) {
    for (int i = 0; i < 3; ++i) f();  // warmup
    std::vector<double> t;
    t.reserve(size_t(iters));
    for (int i = 0; i < iters; ++i) {
        auto a = std::chrono::high_resolution_clock::now();
        f();
        auto b = std::chrono::high_resolution_clock::now();
        t.push_back(std::chrono::duration<double, std::milli>(b - a).count());
    }
    std::sort(t.begin(), t.end());
    Stats s{};
    s.min = t.front();
    s.max = t.back();
    s.median = t[t.size() / 2];
    s.mean = std::accumulate(t.begin(), t.end(), 0.0) / double(t.size());
    double v = 0.0;
    for (double x : t) v += (x - s.mean) * (x - s.mean);
    s.sd = std::sqrt(v / double(t.size()));
    return s;
}

int main(int argc, char** argv) {
    int nthreads = int(std::thread::hardware_concurrency());
    bool tests_only = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-t") == 0 && i + 1 < argc) nthreads = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--tests") == 0) tests_only = true;
        else {
            fprintf(stderr, "usage: %s [-t threads] [--tests]\n", argv[0]);
            return 2;
        }
    }
    if (nthreads < 1) nthreads = 1;

    ThreadPool p(nthreads);
    g_pool = &p;

    run_tests();
    if (tests_only) return g_failures == 0 ? 0 : 1;

    constexpr int M = 4096, K = 4096;
    constexpr float kScale = 0.05f;
    const double zero_frac = 0.60;
    const int iters = 25;

    printf("GEMV  M=%d K=%d  zeros=%.0f%%  threads=%d  iters=%d\n\n",
           M, K, zero_frac * 100, nthreads, iters);

    auto src = random_ternary(M, K, zero_frac, 1234);
    TernaryMatrixInterleaved Wi(M, K, kScale);
    TernaryMatrixBitplane Wb(M, K, kScale);
    pack_interleaved(src, Wi);
    pack_bitplane(src, Wb);

    std::vector<float> Wf(size_t(M) * K);
    for (size_t i = 0; i < Wf.size(); ++i) Wf[i] = float(src[i]) * kScale;

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> X(K);
    for (auto& v : X) v = nd(rng);

    printf("  %-24s %9s %9s %9s %9s %8s  %s\n",
           "kernel", "median", "mean", "sd", "min", "vs fp32", "error");

    std::vector<float> y_ref(M);
    const Stats s_f32 = bench([&] { gemv_fp32(X.data(), Wf.data(), M, K, y_ref.data()); }, iters);
    printf("  %-24s %8.3f %8.3f %8.3f %8.3f %7.2fx  %s\n",
           "fp32 dense (reference)", s_f32.median, s_f32.mean, s_f32.sd, s_f32.min, 1.0, "-");

    std::vector<float> y_i(M);
    const Stats s_i = bench([&] { gemv_interleaved(X.data(), Wi, y_i.data()); }, iters);
    printf("  %-24s %8.3f %8.3f %8.3f %8.3f %7.2fx  %.1e\n",
           "A: interleaved 2-bit", s_i.median, s_i.mean, s_i.sd, s_i.min,
           s_f32.median / s_i.median, rel_err(y_ref, y_i));

    for (const auto& k : all_kernels()) {
        std::vector<float> y(size_t(M), 0.0f);
        const Stats s = bench([&] { k.fn(X.data(), Wb, y.data()); }, iters);
        printf("  %-24s %8.3f %8.3f %8.3f %8.3f %7.2fx  %.1e\n",
               k.name, s.median, s.mean, s.sd, s.min,
               s_f32.median / s.median, rel_err(y_ref, y));
    }

    const double mb_f32 = double(M) * K * 4 / (1024.0 * 1024.0);
    const double mb_tern = double(M) * K * 2 / 8 / (1024.0 * 1024.0);
    printf("\n  weights: fp32 %.1f MB -> ternary %.1f MB (%.0fx)\n", mb_f32, mb_tern,
           mb_f32 / mb_tern);

    // --- MoE layer --------------------------------------------------------
    constexpr int d_model = 512, num_experts = 8, top_k = 2;
    TernaryMoELayer moe(d_model, num_experts, top_k, default_int8_kernel());
    moe.randomize(zero_frac, 99);

    std::vector<float> tok(d_model), out(d_model);
    for (auto& v : tok) v = nd(rng);

    const Stats s_moe = bench([&] { moe.forward(tok.data(), out.data()); }, 200);
    printf("\nMoE  d_model=%d experts=%d top_k=%d  kernel=%s\n", d_model, num_experts, top_k,
           moe.kernel.name);
    printf("  same token:     %.3f ms/token (median)   routed to experts %d, %d   out[0]=%.6f  out[15]=%.6f\n",
           s_moe.median, moe.idx[0], moe.idx[1], out[0], out[15]);

    // Repeating one token keeps the same two experts hot in cache. A stream
    // of different tokens spreads routing over all experts, as decoding does.
    std::vector<std::vector<float>> toks(256, std::vector<float>(size_t(d_model)));
    for (auto& t : toks)
        for (auto& v : t) v = nd(rng);
    size_t next_tok = 0;
    const Stats s_vary = bench([&] { moe.forward(toks[next_tok++ % toks.size()].data(), out.data()); }, 512);
    printf("  varying tokens: %.3f ms/token (median)\n", s_vary.median);

    return g_failures == 0 ? 0 : 1;
}
