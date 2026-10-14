// test_attention.h — the GPU's fused attention kernels (src/attention.cuh),
// run on the CPU through tests/cuda_emu.h, against ops.h's attention in
// double precision (attention_ref.h): the output, the per-row log-sum-exp
// the backward pass relies on, and the gradients of q, k and v.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "cuda_emu.h"
#include "../src/attention.cuh"
#include "attention_ref.h"
#include "test_util.h"

namespace attention_test {

inline void check(int B, int NH, int HS, int Tn, uint64_t seed) {
    const auto k = attention_ref::make_case(B, NH, HS, Tn, seed);
    const int C = k.C;
    const float fscale = 1.0f / std::sqrt(float(HS));
    const int rows = B * NH * Tn;
    std::vector<float> out_f(k.out.size()), lse_f(k.lse.size()), dsum_f(k.lse.size()), dqkv_f = k.dqkv0;
    const emu::Dim3 grid{unsigned((Tn + attn::kBlockRows - 1) / attn::kBlockRows), unsigned(NH), unsigned(B)};
    emu::launch(grid, attn::kThreads, attn::fwd_smem(HS) * sizeof(float), [&] {
        attn::fwd(k.qkv.data(), out_f.data(), lse_f.data(), Tn, C, NH, HS, fscale);
    });
    emu::launch({unsigned((rows + attn::kWarps - 1) / attn::kWarps)}, attn::kThreads, 0, [&] {
        attn::bwd_dsum(out_f.data(), k.dout.data(), dsum_f.data(), rows, Tn, C, NH, HS);
    });
    emu::launch(grid, attn::kThreads, attn::dq_smem(HS) * sizeof(float), [&] {
        attn::bwd_dq(k.qkv.data(), k.dout.data(), lse_f.data(), dsum_f.data(), dqkv_f.data(), Tn, C, NH, HS,
                     fscale);
    });
    emu::launch(grid, attn::kThreads, attn::dkv_smem(HS) * sizeof(float), [&] {
        attn::bwd_dkv(k.qkv.data(), k.dout.data(), lse_f.data(), dsum_f.data(), dqkv_f.data(), Tn, C, NH, HS,
                      fscale);
    });

    const auto e = attention_ref::errors(k, out_f, lse_f, dqkv_f);
    std::printf("  fused attention, %d x %d heads of %d, %d tokens: out %.1e, lse %.1e, dq %.1e, dk %.1e, dv %.1e\n",
                B, NH, HS, Tn, e.out, e.lse, e.dq, e.dk, e.dv);
    EXPECT(e.out < 1e-5 && e.lse < 1e-5, "fused attention forward: out %g, lse %g", e.out, e.lse);
    EXPECT(e.dq < 1e-4 && e.dk < 1e-4 && e.dv < 1e-4, "fused attention backward: dq %g, dk %g, dv %g", e.dq, e.dk,
           e.dv);
}

}  // namespace attention_test

inline void test_attention() {
    emu::Block fresh(1,4);
    EXPECT(std::all_of(fresh.smem.begin(),fresh.smem.end(),[](float x) { return std::isnan(x); }),
           "emulator shared memory hides uninitialized reads with zeros");
    using attention_ref::rel_err;
    EXPECT(std::isinf(rel_err({NAN}, {1.0}, 1, 0, 1)), "attention error metric ignored NaN");
    EXPECT(std::isinf(rel_err({1.0f}, {INFINITY}, 1, 0, 1)), "attention error metric ignored infinity");
    EXPECT(rel_err({0.0f}, {0.0}, 1, 0, 1) == 0, "zero reference error is not zero");
    using attention_test::check;
    check(2, 2, 16, 5, 1);    // shorter than a tile
    check(2, 3, 16, 33, 2);   // one row past a tile
    check(1, 2, 64, 70, 3);   // the night-1 head size, three tiles
    check(2, 4, 8, 24, 4);    // the GPU tests' sizes
    check(1, 1, 128, 40, 5);  // the largest head size
}
