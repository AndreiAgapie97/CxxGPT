// attention_tc.cuh — the fused attention of attention.cuh on the tensor
// cores, for training with TF32 (or bf16) matrix products.
//
// Same math, same layout, same interface (out and lse from the forward
// pass; dsum, then dq and dk/dv, in the backward pass), so either set of
// kernels can run a layer. The difference is who multiplies: here every
// product of two tiles (q·kᵀ, p·v, and in the backward pass do·vᵀ, ds·k,
// pᵀ·do, dsᵀ·q) goes to the tensor cores through WMMA, 16 x 16 x 8 at a
// time, with each input rounded to TF32 (11 significant bits) as cuBLAS
// does in the default --matmul tf32. Sums stay fp32.
//
// Work split: a block of 4 warps takes 64 rows (16 per warp) of one
// (sequence, head) and walks the other side in tiles of kTile rows held in
// shared memory. Tensor-core results come out in an unspecified register
// layout, so each warp stores them to its own scratch rows in shared
// memory; the softmax and the output updates then work on those rows, two
// lanes per row (lane 2r and 2r + 1 take row r's even and odd columns).
// Scores are kept in log2 units (scale · log2 e folded in) so exp2f does
// the exponentials; lse is stored in natural log units, as attention.cuh
// stores it.
//
// Testing: these kernels use the tensor cores, so they run only on a GPU;
// tests/test_gpu.cpp checks them against a double-precision reference and
// against the fp32 kernels. The fp32 kernels in attention.cuh stay the
// reference that runs everywhere (on the CPU emulator in make test), and
// still serve --matmul fp32 and head sizes that are not a multiple of 16.
#pragma once

#include <mma.h>

namespace attn_tc {

using namespace nvcuda;

constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr int kWarpRows = 16;                   // rows per warp (one WMMA tile high)
constexpr int kBlockRows = kWarps * kWarpRows;  // rows per block
constexpr int kTile = 32;                       // rows of the other side per tile
constexpr int kMaxHS = 128;

// Row strides in shared memory: a multiple of 4 floats, as WMMA loads
// want, and not of 32, so rows start in different memory banks.
__host__ __device__ constexpr int pad(int n) { return n + 4; }
__host__ __device__ constexpr int cmax(int a, int b) { return a > b ? a : b; }

// Shared memory per block, in floats. Each kernel needs its tiles during
// the walk, and per-warp rows of HS columns before it (staging q or k, v)
// and after it (the results); the two never overlap in time.
constexpr int fwd_smem(int HS) {
    // k and v tiles, plus one scratch row block per warp for the scores and
    // the p·v products.
    return 2 * kTile * pad(HS) + kWarps * kWarpRows * pad(cmax(kTile, HS));
}
constexpr int dq_smem(int HS) {
    const int walk = 2 * kTile * pad(HS) + kWarps * 2 * kWarpRows * pad(kTile);
    return cmax(walk, kWarps * 2 * kWarpRows * pad(HS));
}
constexpr int dkv_smem(int HS) {
    const int walk = 2 * kTile * pad(HS) + 2 * kTile + kWarps * 2 * kWarpRows * pad(kTile);
    return cmax(walk, kWarps * 2 * kWarpRows * pad(HS));
}

constexpr float kLog2e = 1.4426950408889634f;
constexpr float kLn2 = 0.6931471805599453f;

using FragA = wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major>;
using FragBRow = wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major>;
using FragBCol = wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major>;
using Acc = wmma::fragment<wmma::accumulator, 16, 16, 8, float>;

// Loads a fragment and rounds it to TF32, which the tensor cores expect.
template <class Frag>
__device__ __forceinline__ void load(Frag& f, const float* p, int ld) {
    wmma::load_matrix_sync(f, p, ld);
#pragma unroll
    for (int i = 0; i < f.num_elements; ++i) f.x[i] = wmma::__float_to_tf32(f.x[i]);
}

// Rows [r0, r0 + n) of one head's HS columns (src, row stride ld floats)
// into shared memory (row stride pad(HS)); rows at or past Tn as zeros.
// Threads tid, tid + step, ... share the work, four floats at a time.
template <int HS>
__device__ __forceinline__ void load_rows(float* dst, const float* src, size_t ld, int r0, int n, int Tn, int tid,
                                          int step) {
    constexpr int V = HS / 4;
    for (int i = tid; i < n * V; i += step) {
        const int j = i / V, c = i % V * 4, r = r0 + j;
        const float4 x =
            r < Tn ? *reinterpret_cast<const float4*>(src + size_t(r) * ld + c) : make_float4(0.f, 0.f, 0.f, 0.f);
        *reinterpret_cast<float4*>(dst + j * pad(HS) + c) = x;
    }
}

// Forward. Grid (ceil(Tn / 64), NH, B), kThreads threads, fwd_smem(HS) floats.
template <int HS>
__global__ void __launch_bounds__(kThreads) fwd(const float* qkv, float* out, float* lse, int Tn, int C, int NH,
                                                float scale) {
    extern __shared__ __align__(128) float smem[];
    constexpr int S = pad(HS), PS = pad(cmax(kTile, HS)), D = HS / 2;
    const int b = blockIdx.z, h = blockIdx.y;
    const int t0 = (gridDim.x - 1 - blockIdx.x) * kBlockRows;  // the longest rows start first
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    float* ks = smem;                                   // [kTile][S], a tile of keys
    float* vs = ks + kTile * S;                         // [kTile][S], and their values
    float* ps = vs + kTile * S + warp * kWarpRows * PS;  // [16][PS], this warp's scratch rows
    const size_t ld = size_t(3) * C;
    const float* base = qkv + size_t(b) * Tn * ld + h * HS;
    const int q0 = t0 + warp * kWarpRows;

    // This warp's 16 queries, through its scratch rows into fragments.
    load_rows<HS>(ps, base, ld, q0, kWarpRows, Tn, lane, 32);
    __syncwarp();
    FragA qf[HS / 8];
#pragma unroll
    for (int k = 0; k < HS / 8; ++k) load(qf[k], ps + 8 * k, S);  // load_rows' stride
    __syncwarp();

    const float sl = scale * kLog2e;
    const int r = lane / 2, half = lane % 2, t = q0 + r;  // this lane's row
    float m = -INFINITY, l = 0.0f, o[D];                 // o[j]: column 2j + half
#pragma unroll
    for (int j = 0; j < D; ++j) o[j] = 0.0f;
    float* prow = ps + r * PS;

    const int t_end = min(t0 + kBlockRows, Tn);  // keys up to the block's last query
    for (int s0 = 0; s0 < t_end; s0 += kTile) {
        __syncthreads();  // the previous tile is done with
        load_rows<HS>(ks, base + C, ld, s0, kTile, Tn, threadIdx.x, kThreads);
        load_rows<HS>(vs, base + 2 * C, ld, s0, kTile, Tn, threadIdx.x, kThreads);
        __syncthreads();
        if (s0 > q0 + kWarpRows - 1 || q0 >= Tn) continue;  // no row of this warp sees the tile
        // Keys this warp's rows can see, in blocks of 16.
        const int nc = min(kTile, (q0 + kWarpRows - s0 + 15) / 16 * 16);

        // Scores q·kᵀ into the scratch rows.
        for (int n = 0; n < nc / 16; ++n) {
            Acc acc;
            wmma::fill_fragment(acc, 0.0f);
#pragma unroll
            for (int k = 0; k < HS / 8; ++k) {
                FragBCol kf;
                load(kf, ks + n * 16 * S + 8 * k, S);
                wmma::mma_sync(acc, qf[k], kf, acc);
            }
            wmma::store_matrix_sync(ps + n * 16, acc, PS, wmma::mem_row_major);
        }
        __syncwarp();

        // Online softmax on this lane's half of its row: rescale what came
        // before to the new maximum, and turn the scores into weights.
        float mx = -INFINITY;
        for (int c = half; c < nc; c += 2) {
            const float x = s0 + c <= t ? prow[c] * sl : -INFINITY;
            prow[c] = x;
            mx = fmaxf(mx, x);
        }
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        const float m_new = fmaxf(m, mx);
        const float m_use = m_new == -INFINITY ? 0.0f : m_new;  // a row that sees no key yet
        const float corr = exp2f(m - m_use);
        float sum = 0.0f;
        for (int c = half; c < nc; c += 2) {
            const float p = exp2f(prow[c] - m_use);
            prow[c] = p;
            sum += p;
        }
        sum += __shfl_xor_sync(0xffffffffu, sum, 1);
        l = l * corr + sum;
        m = m_new;
#pragma unroll
        for (int j = 0; j < D; ++j) o[j] *= corr;
        __syncwarp();

        // p·v for this tile, then added to the output rows.
        Acc pv[HS / 16];
#pragma unroll
        for (int n = 0; n < HS / 16; ++n) wmma::fill_fragment(pv[n], 0.0f);
        for (int k = 0; k < nc / 8; ++k) {
            FragA pf;
            load(pf, ps + 8 * k, PS);
#pragma unroll
            for (int n = 0; n < HS / 16; ++n) {
                FragBRow vf;
                load(vf, vs + 8 * k * S + 16 * n, S);
                wmma::mma_sync(pv[n], pf, vf, pv[n]);
            }
        }
        __syncwarp();
#pragma unroll
        for (int n = 0; n < HS / 16; ++n) wmma::store_matrix_sync(ps + 16 * n, pv[n], PS, wmma::mem_row_major);
        __syncwarp();
#pragma unroll
        for (int j = 0; j < D; ++j) o[j] += prow[2 * j + half];
    }

    // Normalize, and write the rows back four floats at a time.
    __syncwarp();
    const float inv = 1.0f / l;
#pragma unroll
    for (int j = 0; j < D; ++j) prow[2 * j + half] = o[j] * inv;
    if (half == 0 && t < Tn) lse[(size_t(b) * NH + h) * Tn + t] = (m + log2f(l)) * kLn2;
    __syncwarp();
    constexpr int V = HS / 4;
    for (int i = lane; i < kWarpRows * V; i += 32) {
        const int j = i / V, c = i % V * 4, tt = q0 + j;
        if (tt < Tn)
            *reinterpret_cast<float4*>(out + (size_t(b) * Tn + tt) * C + h * HS + c) =
                *reinterpret_cast<const float4*>(ps + j * PS + c);
    }
}

// Adds a warp's 16 result rows (scratch rows of stride pad(HS)) into rows
// [r0, r0 + 16) of dst (row stride ld floats), skipping rows at or past Tn.
template <int HS>
__device__ __forceinline__ void add_rows(float* dst, size_t ld, const float* src, int r0, int Tn, int lane) {
    constexpr int V = HS / 4;
    for (int i = lane; i < kWarpRows * V; i += 32) {
        const int j = i / V, c = i % V * 4, r = r0 + j;
        if (r >= Tn) continue;
        float4* y = reinterpret_cast<float4*>(dst + size_t(r) * ld + c);
        const float4 a = *y, x = *reinterpret_cast<const float4*>(src + j * pad(HS) + c);
        *y = make_float4(a.x + x.x, a.y + x.y, a.z + x.z, a.w + x.w);
    }
}

// Backward, dq += Σ_s ds · k_s, with ds = p · (dout·v_s - dsum) · scale and
// p = exp(scale · q·k_s - lse). Walks the keys like fwd. Grid (ceil(Tn / 64),
// NH, B), kThreads threads, dq_smem(HS) floats.
template <int HS>
__global__ void __launch_bounds__(kThreads) bwd_dq(const float* qkv, const float* dout, const float* lse,
                                                   const float* dsum, float* dqkv, int Tn, int C, int NH,
                                                   float scale) {
    extern __shared__ __align__(128) float smem[];
    constexpr int S = pad(HS), PS = pad(kTile);
    const int b = blockIdx.z, h = blockIdx.y;
    const int t0 = (gridDim.x - 1 - blockIdx.x) * kBlockRows;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    float* ks = smem;                                          // [kTile][S]
    float* vs = ks + kTile * S;                                // [kTile][S]
    float* ss = vs + kTile * S + warp * 2 * kWarpRows * PS;    // [16][PS]: scores, then ds
    float* dps = ss + kWarpRows * PS;                          // [16][PS]: dout·vᵀ
    const size_t ld = size_t(3) * C;
    const float* base = qkv + size_t(b) * Tn * ld + h * HS;
    const float* dbase = dout + size_t(b) * Tn * C + h * HS;
    const int q0 = t0 + warp * kWarpRows;

    // This warp's queries and their dout, staged through the shared memory
    // (free until the walk starts) into fragments.
    float* stage = smem + warp * 2 * kWarpRows * S;
    load_rows<HS>(stage, base, ld, q0, kWarpRows, Tn, lane, 32);
    load_rows<HS>(stage + kWarpRows * S, dbase, C, q0, kWarpRows, Tn, lane, 32);
    __syncwarp();
    FragA qf[HS / 8], df[HS / 8];
#pragma unroll
    for (int k = 0; k < HS / 8; ++k) {
        load(qf[k], stage + 8 * k, S);
        load(df[k], stage + kWarpRows * S + 8 * k, S);
    }
    Acc dq[HS / 16];
#pragma unroll
    for (int n = 0; n < HS / 16; ++n) wmma::fill_fragment(dq[n], 0.0f);

    const float sl = scale * kLog2e;
    const int r = lane / 2, half = lane % 2, t = q0 + r;
    const size_t at = (size_t(b) * NH + h) * Tn + t;
    const float L = t < Tn ? lse[at] * kLog2e : 0.0f, Dt = t < Tn ? dsum[at] : 0.0f;
    float* srow = ss + r * PS;
    const float* dprow = dps + r * PS;

    const int t_end = min(t0 + kBlockRows, Tn);
    for (int s0 = 0; s0 < t_end; s0 += kTile) {
        __syncthreads();  // the previous tile (or the staging) is done with
        load_rows<HS>(ks, base + C, ld, s0, kTile, Tn, threadIdx.x, kThreads);
        load_rows<HS>(vs, base + 2 * C, ld, s0, kTile, Tn, threadIdx.x, kThreads);
        __syncthreads();
        if (s0 > q0 + kWarpRows - 1 || q0 >= Tn) continue;
        const int nc = min(kTile, q0 + kWarpRows - s0);

        // Scores q·kᵀ and dout·vᵀ into the scratch rows.
        for (int n = 0; n < nc / 16; ++n) {
            Acc sa, da;
            wmma::fill_fragment(sa, 0.0f);
            wmma::fill_fragment(da, 0.0f);
#pragma unroll
            for (int k = 0; k < HS / 8; ++k) {
                FragBCol kf, vf;
                load(kf, ks + n * 16 * S + 8 * k, S);
                load(vf, vs + n * 16 * S + 8 * k, S);
                wmma::mma_sync(sa, qf[k], kf, sa);
                wmma::mma_sync(da, df[k], vf, da);
            }
            wmma::store_matrix_sync(ss + n * 16, sa, PS, wmma::mem_row_major);
            wmma::store_matrix_sync(dps + n * 16, da, PS, wmma::mem_row_major);
        }
        __syncwarp();
        for (int c = half; c < nc; c += 2) {
            const float p = s0 + c <= t ? exp2f(srow[c] * sl - L) : 0.0f;
            srow[c] = p * (dprow[c] - Dt) * scale;
        }
        __syncwarp();
        // dq += ds · k
        for (int k = 0; k < nc / 8; ++k) {
            FragA dsf;
            load(dsf, ss + 8 * k, PS);
#pragma unroll
            for (int n = 0; n < HS / 16; ++n) {
                FragBRow kf;
                load(kf, ks + 8 * k * S + 16 * n, S);
                wmma::mma_sync(dq[n], dsf, kf, dq[n]);
            }
        }
    }

    __syncthreads();  // the walk is over: its shared memory is free
#pragma unroll
    for (int n = 0; n < HS / 16; ++n) wmma::store_matrix_sync(stage + 16 * n, dq[n], S, wmma::mem_row_major);
    __syncwarp();
    add_rows<HS>(dqkv + size_t(b) * Tn * ld + h * HS, ld, stage, q0, Tn, lane);
}

// Backward, for each key row s: dk += Σ_t ds · q_t and dv += Σ_t p · dout_t
// over the queries t >= s, walked in tiles. Each warp works on its 16 keys
// as rows: sᵀ = k·qᵀ and dpᵀ = v·doutᵀ. Grid (ceil(Tn / 64), NH, B),
// kThreads threads, dkv_smem(HS) floats.
template <int HS>
__global__ void __launch_bounds__(kThreads) bwd_dkv(const float* qkv, const float* dout, const float* lse,
                                                    const float* dsum, float* dqkv, int Tn, int C, int NH,
                                                    float scale) {
    extern __shared__ __align__(128) float smem[];
    constexpr int S = pad(HS), PS = pad(kTile);
    const int b = blockIdx.z, h = blockIdx.y, s0 = blockIdx.x * kBlockRows;  // early keys (most work) first
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    float* qs = smem;                                         // [kTile][S], a tile of queries
    float* dos = qs + kTile * S;                              // [kTile][S], their dout
    float* lt = dos + kTile * S;                              // [kTile], their lse (log2 units)
    float* dt = lt + kTile;                                   // [kTile], their dsum
    float* ps = dt + kTile + warp * 2 * kWarpRows * PS;       // [16][PS]: pᵀ
    float* dss = ps + kWarpRows * PS;                         // [16][PS]: dpᵀ, then dsᵀ
    const size_t ld = size_t(3) * C;
    const float* base = qkv + size_t(b) * Tn * ld + h * HS;
    const float* dbase = dout + size_t(b) * Tn * C + h * HS;
    const size_t rbase = (size_t(b) * NH + h) * Tn;
    const int k0 = s0 + warp * kWarpRows;  // this warp's first key

    float* stage = smem + warp * 2 * kWarpRows * S;
    load_rows<HS>(stage, base + C, ld, k0, kWarpRows, Tn, lane, 32);
    load_rows<HS>(stage + kWarpRows * S, base + 2 * C, ld, k0, kWarpRows, Tn, lane, 32);
    __syncwarp();
    FragA kf[HS / 8], vf[HS / 8];
#pragma unroll
    for (int k = 0; k < HS / 8; ++k) {
        load(kf[k], stage + 8 * k, S);
        load(vf[k], stage + kWarpRows * S + 8 * k, S);
    }
    Acc dk[HS / 16], dv[HS / 16];
#pragma unroll
    for (int n = 0; n < HS / 16; ++n) {
        wmma::fill_fragment(dk[n], 0.0f);
        wmma::fill_fragment(dv[n], 0.0f);
    }

    const float sl = scale * kLog2e;
    const int r = lane / 2, half = lane % 2, s = k0 + r;  // this lane's key
    float* prow = ps + r * PS;
    float* dsrow = dss + r * PS;

    for (int t0 = s0 / kTile * kTile; t0 < Tn; t0 += kTile) {
        __syncthreads();
        load_rows<HS>(qs, base, ld, t0, kTile, Tn, threadIdx.x, kThreads);
        load_rows<HS>(dos, dbase, C, t0, kTile, Tn, threadIdx.x, kThreads);
        for (int j = threadIdx.x; j < kTile; j += kThreads) {
            lt[j] = t0 + j < Tn ? lse[rbase + t0 + j] * kLog2e : 0.0f;
            dt[j] = t0 + j < Tn ? dsum[rbase + t0 + j] : 0.0f;
        }
        __syncthreads();
        if (t0 + kTile <= k0 || k0 >= Tn) continue;  // no query of this tile sees these keys

        for (int n = 0; n < kTile / 16; ++n) {
            Acc sa, da;
            wmma::fill_fragment(sa, 0.0f);
            wmma::fill_fragment(da, 0.0f);
#pragma unroll
            for (int k = 0; k < HS / 8; ++k) {
                FragBCol qf, df;
                load(qf, qs + n * 16 * S + 8 * k, S);
                load(df, dos + n * 16 * S + 8 * k, S);
                wmma::mma_sync(sa, kf[k], qf, sa);
                wmma::mma_sync(da, vf[k], df, da);
            }
            wmma::store_matrix_sync(ps + n * 16, sa, PS, wmma::mem_row_major);
            wmma::store_matrix_sync(dss + n * 16, da, PS, wmma::mem_row_major);
        }
        __syncwarp();
        for (int c = half; c < kTile; c += 2) {
            const int t = t0 + c;
            const float p = t >= s && t < Tn ? exp2f(prow[c] * sl - lt[c]) : 0.0f;
            prow[c] = p;
            dsrow[c] = p * (dsrow[c] - dt[c]) * scale;
        }
        __syncwarp();
        // dv += pᵀ · dout, dk += dsᵀ · q
        for (int k = 0; k < kTile / 8; ++k) {
            FragA pf, dsf;
            load(pf, ps + 8 * k, PS);
            load(dsf, dss + 8 * k, PS);
#pragma unroll
            for (int n = 0; n < HS / 16; ++n) {
                FragBRow df, qf;
                load(df, dos + 8 * k * S + 16 * n, S);
                load(qf, qs + 8 * k * S + 16 * n, S);
                wmma::mma_sync(dv[n], pf, df, dv[n]);
                wmma::mma_sync(dk[n], dsf, qf, dk[n]);
            }
        }
    }

    __syncthreads();
#pragma unroll
    for (int n = 0; n < HS / 16; ++n) {
        wmma::store_matrix_sync(stage + 16 * n, dk[n], S, wmma::mem_row_major);
        wmma::store_matrix_sync(stage + kWarpRows * S + 16 * n, dv[n], S, wmma::mem_row_major);
    }
    __syncwarp();
    float* dbase_kv = dqkv + size_t(b) * Tn * ld + h * HS;
    add_rows<HS>(dbase_kv + C, ld, stage, k0, Tn, lane);
    add_rows<HS>(dbase_kv + 2 * C, ld, stage + kWarpRows * S, k0, Tn, lane);
}

// Runs f.template operator()<HS>() for the head size HS (a multiple of 16
// up to kMaxHS); false for other sizes, which the fp32 kernels take.
template <class F>
bool with_head_size(int HS, F&& f) {
    switch (HS) {
        case 16: f.template operator()<16>(); return true;
        case 32: f.template operator()<32>(); return true;
        case 48: f.template operator()<48>(); return true;
        case 64: f.template operator()<64>(); return true;
        case 80: f.template operator()<80>(); return true;
        case 96: f.template operator()<96>(); return true;
        case 112: f.template operator()<112>(); return true;
        case 128: f.template operator()<128>(); return true;
        default: return false;
    }
}

}  // namespace attn_tc
