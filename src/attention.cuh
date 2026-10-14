// attention.cuh — fused causal attention for the GPU trainer (gpu.cu).
//
// The same math as attention_forward / attention_backward in ops.h, in the
// FlashAttention style: the Tn x Tn matrix of attention weights is never
// stored. The forward pass walks the keys tile by tile, keeping for each
// query row a running maximum, a running sum of exponentials and the
// output so far (an "online softmax"), and keeps only one number per row
// for the backward pass: lse = log Σ_s exp(scale · q·k_s). The backward
// pass recomputes the weights from it, p = exp(scale · q·k - lse).
// At the night-1 size (batch 8, 512 tokens, 12 heads, 12 layers) that is
// 2.4 MB instead of 1.2 GB of stored weights.
//
// Layout, as in ops.h: qkv[B·Tn, 3C] holds q, k, v side by side, head h at
// columns h·HS of each; out and dout are [B·Tn, C]; lse and dsum [B, NH, Tn].
//
// Work split: a block of 4 warps takes 16 rows of one (sequence, head), 4
// rows per warp, and walks the other side in tiles of 32 rows held in
// shared memory, one per lane. A lane computes its row's dot product with
// the whole head (so no reduction across lanes); then the lanes share the
// results through warp shuffles and each accumulates the head dimensions
// it owns (d = lane, lane + 32, ...).
//
// The kernels compile for the GPU (nvcc or clang -x cuda) and, through
// tests/cuda_emu.h, as plain C++ on the CPU, where the tests run them.
#pragma once

#include <cmath>
#include <cstddef>

#ifdef __CUDACC__
#define ATT_DEVICE __device__ __forceinline__
#define ATT_GLOBAL __global__
#define ATT_SMEM(name) extern __shared__ float name[]
#define ATT_SYNC() __syncthreads()
#define ATT_TID int(threadIdx.x)
#define ATT_BLOCK blockIdx
#define ATT_UNROLL _Pragma("unroll")
#else
#define ATT_DEVICE inline
#define ATT_GLOBAL inline
#define ATT_SMEM(name) float* name = emu::shared_memory()
#define ATT_SYNC() emu::sync_block()
#define ATT_TID int(emu::thread_idx.x)
#define ATT_BLOCK emu::block_idx
#define ATT_UNROLL
#endif

namespace attn {

constexpr int kLanes = 32;
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * kLanes;   // per block
constexpr int kRows = 4;                    // rows per warp
constexpr int kBlockRows = kWarps * kRows;  // rows per block
constexpr int kTile = kLanes;               // rows of the other side per tile, one per lane
constexpr int kMaxHS = 128;                 // head size limit (registers per lane)
constexpr int kDims = kMaxHS / kLanes;      // head dimensions per lane

// Shared memory per block, in floats. Tiles that lanes read one row each
// use a row stride of HS + 1, so the 32 lanes hit 32 different banks.
inline constexpr int fwd_smem(int HS) { return kBlockRows * HS + kTile * (HS + 1) + kTile * HS; }
inline constexpr int dq_smem(int HS) { return 2 * kBlockRows * HS + 2 * kTile * (HS + 1); }
inline constexpr int dkv_smem(int HS) { return 2 * kBlockRows * HS + 2 * kTile * (HS + 1) + 2 * kTile; }

#ifdef __CUDACC__
ATT_DEVICE float shfl(float v, int lane) { return __shfl_sync(0xffffffffu, v, lane); }
ATT_DEVICE float shfl_xor(float v, int mask) { return __shfl_xor_sync(0xffffffffu, v, mask); }
#else
inline float shfl(float v, int lane) { return emu::shfl(v, lane); }
inline float shfl_xor(float v, int mask) { return emu::shfl(v, int(emu::thread_idx.x % kLanes) ^ mask); }
#endif

ATT_DEVICE float warp_sum(float v) {
    for (int o = kLanes / 2; o > 0; o >>= 1) v += shfl_xor(v, o);
    return v;
}
ATT_DEVICE float warp_max(float v) {
    for (int o = kLanes / 2; o > 0; o >>= 1) v = fmaxf(v, shfl_xor(v, o));
    return v;
}
ATT_DEVICE int imin(int a, int b) { return a < b ? a : b; }
ATT_DEVICE float dot(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

// Forward. Grid (ceil(Tn / 16), NH, B), kThreads threads, fwd_smem(HS) floats.
ATT_GLOBAL void fwd(const float* qkv, float* out, float* lse, int Tn, int C, int NH, int HS, float scale) {
    ATT_SMEM(smem);
    const int b = ATT_BLOCK.z, h = ATT_BLOCK.y, t0 = ATT_BLOCK.x * kBlockRows;
    const int warp = ATT_TID / kLanes, lane = ATT_TID % kLanes, KS = HS + 1;
    float* qs = smem;                  // [kBlockRows][HS], this block's queries
    float* ks = qs + kBlockRows * HS;  // [kTile][KS], a tile of keys
    float* vs = ks + kTile * KS;       // [kTile][HS], and their values
    const size_t row = size_t(3) * C;
    const float* base = qkv + size_t(b) * Tn * row + h * HS;
    for (int i = ATT_TID; i < kBlockRows * HS; i += kThreads) {
        const int t = t0 + i / HS;
        qs[i] = t < Tn ? base[t * row + i % HS] : 0.0f;
    }
    float m[kRows], l[kRows], o[kRows][kDims];
    ATT_UNROLL
    for (int r = 0; r < kRows; ++r) {
        m[r] = -INFINITY;
        l[r] = 0.0f;
        ATT_UNROLL
        for (int i = 0; i < kDims; ++i) o[r][i] = 0.0f;
    }
    const int t_end = imin(t0 + kBlockRows, Tn);  // keys up to the block's last query
    for (int s0 = 0; s0 < t_end; s0 += kTile) {
        ATT_SYNC();  // the previous tile is done with (and qs is in place)
        for (int i = ATT_TID; i < kTile * HS; i += kThreads) {
            const int j = i / HS, d = i % HS, s = s0 + j;
            ks[j * KS + d] = s < Tn ? base[s * row + C + d] : 0.0f;
            vs[i] = s < Tn ? base[s * row + 2 * C + d] : 0.0f;
        }
        ATT_SYNC();
        ATT_UNROLL
        for (int r = 0; r < kRows; ++r) {
            const int t = t0 + warp * kRows + r;
            if (t >= Tn || s0 > t) continue;  // the same for the whole warp
            const int s = s0 + lane;
            const float score = s <= t ? dot(qs + (warp * kRows + r) * HS, ks + lane * KS, HS) * scale : -INFINITY;
            // Online softmax: rescale what came before to the new maximum.
            const float m_new = fmaxf(m[r], warp_max(score));
            const float p = s <= t ? expf(score - m_new) : 0.0f;
            const float corr = expf(m[r] - m_new);
            l[r] = l[r] * corr + warp_sum(p);
            m[r] = m_new;
            ATT_UNROLL
            for (int i = 0; i < kDims; ++i) o[r][i] *= corr;
            const int jn = imin(kTile, t - s0 + 1);  // keys of this tile that t sees
            for (int j = 0; j < jn; ++j) {
                const float pj = shfl(p, j);
                ATT_UNROLL
                for (int i = 0; i < kDims; ++i) {
                    const int d = lane + i * kLanes;
                    if (d < HS) o[r][i] += pj * vs[j * HS + d];
                }
            }
        }
    }
    ATT_UNROLL
    for (int r = 0; r < kRows; ++r) {
        const int t = t0 + warp * kRows + r;
        if (t >= Tn) continue;
        float* y = out + (size_t(b) * Tn + t) * C + h * HS;
        const float inv = 1.0f / l[r];
        ATT_UNROLL
        for (int i = 0; i < kDims; ++i) {
            const int d = lane + i * kLanes;
            if (d < HS) y[d] = o[r][i] * inv;
        }
        if (lane == 0) lse[(size_t(b) * NH + h) * Tn + t] = m[r] + logf(l[r]);
    }
}

// Backward, step 1: dsum = Σ_d dout · out per row, which equals
// Σ_s p · (dout · v_s), the term the softmax backward subtracts.
// One warp per row: grid ceil(B·NH·Tn / kWarps), kThreads threads.
ATT_GLOBAL void bwd_dsum(const float* out, const float* dout, float* dsum, int rows, int Tn, int C, int NH,
                         int HS) {
    const int r = ATT_BLOCK.x * kWarps + ATT_TID / kLanes, lane = ATT_TID % kLanes;
    if (r >= rows) return;  // the whole warp
    const int t = r % Tn, h = (r / Tn) % NH, b = r / Tn / NH;
    const size_t at = (size_t(b) * Tn + t) * C + h * HS;
    float s = 0.0f;
    for (int d = lane; d < HS; d += kLanes) s += out[at + d] * dout[at + d];
    s = warp_sum(s);
    if (lane == 0) dsum[r] = s;
}

// Backward, step 2: dq += Σ_s ds · k_s, with ds = p · (dout·v_s - dsum) ·
// scale. Walks the keys like fwd. Grid (ceil(Tn / 16), NH, B), dq_smem(HS).
ATT_GLOBAL void bwd_dq(const float* qkv, const float* dout, const float* lse, const float* dsum, float* dqkv,
                       int Tn, int C, int NH, int HS, float scale) {
    ATT_SMEM(smem);
    const int b = ATT_BLOCK.z, h = ATT_BLOCK.y, t0 = ATT_BLOCK.x * kBlockRows;
    const int warp = ATT_TID / kLanes, lane = ATT_TID % kLanes, KS = HS + 1;
    float* qs = smem;                   // [kBlockRows][HS]
    float* dos = qs + kBlockRows * HS;  // [kBlockRows][HS], dout of the same rows
    float* ks = dos + kBlockRows * HS;  // [kTile][KS]
    float* vs = ks + kTile * KS;        // [kTile][KS]
    const size_t row = size_t(3) * C;
    const float* base = qkv + size_t(b) * Tn * row + h * HS;
    const float* dbase = dout + size_t(b) * Tn * C + h * HS;
    for (int i = ATT_TID; i < kBlockRows * HS; i += kThreads) {
        const int t = t0 + i / HS, d = i % HS;
        qs[i] = t < Tn ? base[t * row + d] : 0.0f;
        dos[i] = t < Tn ? dbase[size_t(t) * C + d] : 0.0f;
    }
    float L[kRows], D[kRows], dq[kRows][kDims];
    ATT_UNROLL
    for (int r = 0; r < kRows; ++r) {
        const int t = t0 + warp * kRows + r;
        const size_t at = (size_t(b) * NH + h) * Tn + t;
        L[r] = t < Tn ? lse[at] : 0.0f;
        D[r] = t < Tn ? dsum[at] : 0.0f;
        ATT_UNROLL
        for (int i = 0; i < kDims; ++i) dq[r][i] = 0.0f;
    }
    const int t_end = imin(t0 + kBlockRows, Tn);
    for (int s0 = 0; s0 < t_end; s0 += kTile) {
        ATT_SYNC();
        for (int i = ATT_TID; i < kTile * HS; i += kThreads) {
            const int j = i / HS, d = i % HS, s = s0 + j;
            ks[j * KS + d] = s < Tn ? base[s * row + C + d] : 0.0f;
            vs[j * KS + d] = s < Tn ? base[s * row + 2 * C + d] : 0.0f;
        }
        ATT_SYNC();
        ATT_UNROLL
        for (int r = 0; r < kRows; ++r) {
            const int t = t0 + warp * kRows + r;
            if (t >= Tn || s0 > t) continue;
            const int s = s0 + lane;
            float ds = 0.0f;
            if (s <= t) {
                const int q = (warp * kRows + r) * HS;
                const float p = expf(dot(qs + q, ks + lane * KS, HS) * scale - L[r]);
                ds = p * (dot(dos + q, vs + lane * KS, HS) - D[r]) * scale;
            }
            const int jn = imin(kTile, t - s0 + 1);
            for (int j = 0; j < jn; ++j) {
                const float dsj = shfl(ds, j);
                ATT_UNROLL
                for (int i = 0; i < kDims; ++i) {
                    const int d = lane + i * kLanes;
                    if (d < HS) dq[r][i] += dsj * ks[j * KS + d];
                }
            }
        }
    }
    ATT_UNROLL
    for (int r = 0; r < kRows; ++r) {
        const int t = t0 + warp * kRows + r;
        if (t >= Tn) continue;
        float* y = dqkv + (size_t(b) * Tn + t) * row + h * HS;
        ATT_UNROLL
        for (int i = 0; i < kDims; ++i) {
            const int d = lane + i * kLanes;
            if (d < HS) y[d] += dq[r][i];
        }
    }
}

// Backward, step 3: for each key row s, dk += Σ_t ds · q_t and dv += Σ_t p ·
// dout_t over the queries t >= s, walked in tiles. Grid (ceil(Tn / 16), NH,
// B), dkv_smem(HS).
ATT_GLOBAL void bwd_dkv(const float* qkv, const float* dout, const float* lse, const float* dsum, float* dqkv,
                        int Tn, int C, int NH, int HS, float scale) {
    ATT_SMEM(smem);
    const int b = ATT_BLOCK.z, h = ATT_BLOCK.y, s0 = ATT_BLOCK.x * kBlockRows;
    const int warp = ATT_TID / kLanes, lane = ATT_TID % kLanes, KS = HS + 1;
    float* kb = smem;                   // [kBlockRows][HS], this block's keys
    float* vb = kb + kBlockRows * HS;   // [kBlockRows][HS], and values
    float* qt = vb + kBlockRows * HS;   // [kTile][KS], a tile of queries
    float* dot_ = qt + kTile * KS;      // [kTile][KS], their dout
    float* lt = dot_ + kTile * KS;      // [kTile], their lse
    float* dt = lt + kTile;             // [kTile], their dsum
    const size_t row = size_t(3) * C;
    const float* base = qkv + size_t(b) * Tn * row + h * HS;
    const float* dbase = dout + size_t(b) * Tn * C + h * HS;
    const size_t rbase = (size_t(b) * NH + h) * Tn;
    for (int i = ATT_TID; i < kBlockRows * HS; i += kThreads) {
        const int s = s0 + i / HS, d = i % HS;
        kb[i] = s < Tn ? base[s * row + C + d] : 0.0f;
        vb[i] = s < Tn ? base[s * row + 2 * C + d] : 0.0f;
    }
    float dk[kRows][kDims], dv[kRows][kDims];
    ATT_UNROLL
    for (int r = 0; r < kRows; ++r) {
        ATT_UNROLL
        for (int i = 0; i < kDims; ++i) dk[r][i] = dv[r][i] = 0.0f;
    }
    for (int t0 = s0 / kTile * kTile; t0 < Tn; t0 += kTile) {
        ATT_SYNC();
        for (int i = ATT_TID; i < kTile * HS; i += kThreads) {
            const int j = i / HS, d = i % HS, t = t0 + j;
            qt[j * KS + d] = t < Tn ? base[t * row + d] : 0.0f;
            dot_[j * KS + d] = t < Tn ? dbase[size_t(t) * C + d] : 0.0f;
        }
        for (int j = ATT_TID; j < kTile; j += kThreads) {
            lt[j] = t0 + j < Tn ? lse[rbase + t0 + j] : 0.0f;
            dt[j] = t0 + j < Tn ? dsum[rbase + t0 + j] : 0.0f;
        }
        ATT_SYNC();
        ATT_UNROLL
        for (int r = 0; r < kRows; ++r) {
            const int s = s0 + warp * kRows + r;
            if (s >= Tn || t0 + kTile <= s) continue;  // no query of this tile sees s
            const int t = t0 + lane;
            float p = 0.0f, ds = 0.0f;
            if (t < Tn && t >= s) {
                const int k = (warp * kRows + r) * HS;
                p = expf(dot(kb + k, qt + lane * KS, HS) * scale - lt[lane]);
                ds = p * (dot(vb + k, dot_ + lane * KS, HS) - dt[lane]) * scale;
            }
            const int jn = imin(kTile, Tn - t0);
            for (int j = 0; j < jn; ++j) {
                const float pj = shfl(p, j), dsj = shfl(ds, j);
                ATT_UNROLL
                for (int i = 0; i < kDims; ++i) {
                    const int d = lane + i * kLanes;
                    if (d < HS) {
                        dv[r][i] += pj * dot_[j * KS + d];
                        dk[r][i] += dsj * qt[j * KS + d];
                    }
                }
            }
        }
    }
    ATT_UNROLL
    for (int r = 0; r < kRows; ++r) {
        const int s = s0 + warp * kRows + r;
        if (s >= Tn) continue;
        float* y = dqkv + (size_t(b) * Tn + s) * row + h * HS;
        ATT_UNROLL
        for (int i = 0; i < kDims; ++i) {
            const int d = lane + i * kLanes;
            if (d < HS) {
                y[C + d] += dk[r][i];
                y[2 * C + d] += dv[r][i];
            }
        }
    }
}

}  // namespace attn
