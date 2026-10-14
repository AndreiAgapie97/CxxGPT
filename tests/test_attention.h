// test_attention.h — the GPU's fused attention kernels (src/attention.cuh),
// run on the CPU through tests/cuda_emu.h, against ops.h's attention in
// double precision: the output, the per-row log-sum-exp the backward pass
// relies on, and the gradients of q, k and v.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "cuda_emu.h"
#include "../src/attention.cuh"
#include "../src/ops.h"
#include "test_util.h"

namespace attention_test {

// max |got - want| / max |want| over the columns [c0, c1) of rows of width w.
inline double rel_err(const std::vector<float>& got, const std::vector<double>& want, int w, int c0, int c1) {
    if (got.size() != want.size()) return INFINITY;
    double diff = 0, mx = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        const int c = int(i % size_t(w));
        if (c < c0 || c >= c1) continue;
        if (!std::isfinite(got[i]) || !std::isfinite(want[i])) return INFINITY;
        diff = std::max(diff, std::abs(got[i] - want[i]));
        mx = std::max(mx, std::abs(want[i]));
    }
    return mx > 0 ? diff / mx : diff == 0 ? 0 : INFINITY;
}

inline void check(int B, int NH, int HS, int Tn, uint64_t seed) {
    const int C = NH * HS, N = B * Tn;
    Rng rng(seed);
    std::vector<double> qkv(size_t(N) * 3 * C), dout(size_t(N) * C), dqkv(size_t(N) * 3 * C);
    for (auto& x : qkv) x = rng.normal();
    for (auto& x : dout) x = rng.normal();
    for (auto& x : dqkv) x = 0.1 * rng.normal();  // both sides accumulate into it
    const std::vector<float> qkv_f(qkv.begin(), qkv.end()), dout_f(dout.begin(), dout.end());
    std::vector<float> dqkv_f(dqkv.begin(), dqkv.end());

    std::vector<double> out(size_t(N) * C), att(size_t(B) * NH * Tn * Tn), lse(size_t(B) * NH * Tn);
    attention_forward(out.data(), att.data(), qkv.data(), B, Tn, C, NH);
    attention_backward(dqkv.data(), dout.data(), att.data(), qkv.data(), B, Tn, C, NH);
    const double scale = 1.0 / std::sqrt(double(HS));
    for (int b = 0; b < B; ++b)
        for (int h = 0; h < NH; ++h)
            for (int t = 0; t < Tn; ++t) {
                std::vector<double> s(size_t(t) + 1);
                for (int j = 0; j <= t; ++j) {
                    double d = 0;
                    for (int i = 0; i < HS; ++i)
                        d += qkv[(size_t(b) * Tn + t) * 3 * C + h * HS + i] *
                             qkv[(size_t(b) * Tn + j) * 3 * C + C + h * HS + i];
                    s[size_t(j)] = d * scale;
                }
                const double mx = *std::max_element(s.begin(), s.end());
                double sum = 0;
                for (double x : s) sum += std::exp(x - mx);
                lse[(size_t(b) * NH + h) * Tn + t] = mx + std::log(sum);
            }

    const float fscale = 1.0f / std::sqrt(float(HS));
    const int rows = B * NH * Tn;
    std::vector<float> out_f(out.size()), lse_f(lse.size()), dsum_f(lse.size());
    const emu::Dim3 grid{unsigned((Tn + attn::kBlockRows - 1) / attn::kBlockRows), unsigned(NH), unsigned(B)};
    emu::launch(grid, attn::kThreads, attn::fwd_smem(HS) * sizeof(float), [&] {
        attn::fwd(qkv_f.data(), out_f.data(), lse_f.data(), Tn, C, NH, HS, fscale);
    });
    emu::launch({unsigned((rows + attn::kWarps - 1) / attn::kWarps)}, attn::kThreads, 0, [&] {
        attn::bwd_dsum(out_f.data(), dout_f.data(), dsum_f.data(), rows, Tn, C, NH, HS);
    });
    emu::launch(grid, attn::kThreads, attn::dq_smem(HS) * sizeof(float), [&] {
        attn::bwd_dq(qkv_f.data(), dout_f.data(), lse_f.data(), dsum_f.data(), dqkv_f.data(), Tn, C, NH, HS,
                     fscale);
    });
    emu::launch(grid, attn::kThreads, attn::dkv_smem(HS) * sizeof(float), [&] {
        attn::bwd_dkv(qkv_f.data(), dout_f.data(), lse_f.data(), dsum_f.data(), dqkv_f.data(), Tn, C, NH, HS,
                      fscale);
    });

    const double e_out = rel_err(out_f, out, C, 0, C), e_lse = rel_err(lse_f, lse, 1, 0, 1);
    const double e_q = rel_err(dqkv_f, dqkv, 3 * C, 0, C), e_k = rel_err(dqkv_f, dqkv, 3 * C, C, 2 * C);
    const double e_v = rel_err(dqkv_f, dqkv, 3 * C, 2 * C, 3 * C);
    std::printf("  fused attention, %d x %d heads of %d, %d tokens: out %.1e, lse %.1e, dq %.1e, dk %.1e, dv %.1e\n",
                B, NH, HS, Tn, e_out, e_lse, e_q, e_k, e_v);
    EXPECT(e_out < 1e-5 && e_lse < 1e-5, "fused attention forward: out %g, lse %g", e_out, e_lse);
    EXPECT(e_q < 1e-4 && e_k < 1e-4 && e_v < 1e-4, "fused attention backward: dq %g, dk %g, dv %g", e_q, e_k,
           e_v);
}

}  // namespace attention_test

inline void test_attention() {
    EXPECT(std::isinf(attention_test::rel_err({NAN}, {1.0}, 1, 0, 1)), "attention error metric ignored NaN");
    EXPECT(std::isinf(attention_test::rel_err({1.0f}, {INFINITY}, 1, 0, 1)), "attention error metric ignored infinity");
    EXPECT(attention_test::rel_err({0.0f}, {0.0}, 1, 0, 1) == 0, "zero reference error is not zero");
    using attention_test::check;
    check(2, 2, 16, 5, 1);    // shorter than a tile
    check(2, 3, 16, 33, 2);   // one row past a tile
    check(1, 2, 64, 70, 3);   // the night-1 head size, three tiles
    check(2, 4, 8, 24, 4);    // the GPU tests' sizes
    check(1, 1, 128, 40, 5);  // the largest head size
}
