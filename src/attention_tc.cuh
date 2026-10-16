// attention_tc.cuh — the fused attention of attention.cuh on the tensor
// cores, for training with TF32 (or bf16) matrix products.
//
// Same math and the same layout as attention.cuh: out and lse from the
// forward pass; dsum (attention.cuh's bwd_dsum), then dq and dk/dv in the
// backward pass. Every product of two tiles (q·kᵀ, p·v, and in the backward
// pass dout·vᵀ, ds·k, pᵀ·dout, dsᵀ·q) goes to the tensor cores through the
// mma.sync instruction, 16 x 8 x 8 at a time, with each input rounded to
// TF32 (11 significant bits, cvt.rna: to nearest, as cuBLAS rounds in the
// default --matmul tf32, and as WMMA's __float_to_tf32 did here before).
// Sums stay fp32. Attention keeps these TF32 products under --matmul bf16
// too, and in ternary models (--fp-attn or not).
//
// Work split: a block of 4 warps takes 64 rows (16 per warp) of one
// (sequence, head) and walks the other side in tiles of kTile rows, copied
// into shared memory with cp.async while the warps work on the previous
// tile (two stages). Unlike WMMA, mma.sync says which lane holds which
// result, so everything between two products stays in registers (the
// FlashAttention-2 layout): lane l holds rows g = l / 4 and g + 8 and, in
// each 8-column tile, columns 2q and 2q + 1 (q = l % 4). A row's maximum and
// sum are then two shuffles among the four lanes of a quad. A result such
// as p is the next product's A operand as it is: A wants columns q and q + 4
// where the lane has 2q and 2q + 1, and since the order of the terms in a
// sum does not matter, the B operand's rows are read in the same order
// (2q, 2q + 1) instead (mma_pb).
//
// Scores are kept in log2 units (scale · log2 e folded in), so exp2f does
// the exponentials; lse is stored in natural log units, as attention.cuh
// stores it.
//
// The backward kernels write dq, dk and dv (unlike attention.cuh's, which
// add): every element of dqkv is written exactly once, so the trainer does
// not zero it first, and nothing is read back. On the night model's
// shape (8 x 512 tokens, 12 heads of 64) that is 75 MB less memory traffic
// per layer: on the RTX 4070 laptop (256 GB/s) the backward pass is partly
// limited by memory, not only by the tensor cores.
//
// Testing: these kernels use the tensor cores, so they run only on a GPU;
// tests/test_gpu.cpp checks them against a double-precision reference. The
// fp32 kernels in attention.cuh stay the reference that runs everywhere (on
// the CPU emulator in make test), and still serve --matmul fp32 and head
// sizes that are not a multiple of 16.
#pragma once

namespace attn_tc {

constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr int kWarpRows = 16;                   // rows per warp (one mma tile high)
constexpr int kBlockRows = kWarps * kWarpRows;  // rows per block
constexpr int kTile = 32;                       // rows of the other side per tile
constexpr int kChunk = 16;                      // of which a warp takes this many at a time
constexpr int kStages = 2;                      // tiles in shared memory: one in use, one arriving
constexpr int kMaxHS = 128;
static_assert(kStages >= 2 && kTile % kChunk == 0, "the walk waits for tile i while tile i + 1 arrives");

// Shared-memory row stride, in floats: HS + 4. With HS a multiple of 16,
// rows start 16 bytes apart modulo 128 bytes, so ldmatrix's eight rows fall
// in different memory banks; and rows 2q (q = 0..3) start 8 banks apart, so
// mma_pb's single-value loads (rows 2q and 2q + 1, columns g = 0..7) hit 32
// different banks.
__host__ __device__ constexpr int ld(int HS) { return HS + 4; }

// Shared memory per block, in floats: kStages tiles of two matrices, plus,
// for dk/dv, the tile's lse and dsum.
constexpr int fwd_smem(int HS) { return kStages * 2 * kTile * ld(HS); }
constexpr int dq_smem(int HS) { return kStages * 2 * kTile * ld(HS); }
constexpr int dkv_smem(int HS) { return kStages * (2 * kTile * ld(HS) + 2 * kTile); }

constexpr float kLog2e = 1.4426950408889634f;
constexpr float kLn2 = 0.6931471805599453f;

__device__ __forceinline__ unsigned tf32(float x) {
    unsigned r;
    asm("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(r) : "f"(x));
    return r;
}
// c += a · b for one 16 x 8 x 8 tile: A 16 x 8 (rows g, g + 8; columns q,
// q + 4), B 8 x 8 (rows q, q + 4; column g), C 16 x 8 (rows g, g + 8;
// columns 2q, 2q + 1).
__device__ __forceinline__ void mma(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// ldmatrix moves 16-bit halves: on fp32 rows (8 rows of 4 floats per
// matrix) each lane gets whole floats, row lane / 4, column lane % 4.
__device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], const float* p) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}
// Copies (cp.async) that land in shared memory while the threads go on;
// bytes = 0 writes zeros and reads nothing.
__device__ __forceinline__ void cp16(float* smem, const float* gmem, int bytes) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(bytes));
}
__device__ __forceinline__ void cp4(float* smem, const float* gmem, int bytes) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(s), "l"(gmem), "r"(bytes));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
// Waits until at most N groups of copies are still in flight.
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// Rows [r0, r0 + kTile) of one head's HS columns (src, row stride ldg
// floats) into shared memory (row stride ld(HS)); rows at or past Tn as
// zeros. The block's threads share the work, 16 bytes at a time.
template <int HS>
__device__ __forceinline__ void load_tile(float* dst, const float* src, size_t ldg, int r0, int Tn) {
    constexpr int V = HS / 4;
#pragma unroll
    for (int i = threadIdx.x; i < kTile * V; i += kThreads) {
        const int j = i / V, c = i % V * 4, r = r0 + j;
        cp16(dst + j * ld(HS) + c, r < Tn ? src + size_t(r) * ldg + c : src, r < Tn ? 16 : 0);
    }
}
// kTile floats from src[r0], zeros at or past Tn.
__device__ __forceinline__ void load_vec(float* dst, const float* src, int r0, int Tn) {
    for (int i = threadIdx.x; i < kTile; i += kThreads)
        cp4(dst + i, r0 + i < Tn ? src + r0 + i : src, r0 + i < Tn ? 4 : 0);
}

// A warp's 16 rows from r0 (row stride ldg floats) as A operands, rounded
// to TF32: f[k] holds rows g, g + 8 and columns 8k + q, 8k + q + 4. Rows at
// or past Tn as zeros.
template <int HS>
__device__ __forceinline__ void load_a(unsigned (&f)[HS / 8][4], const float* src, size_t ldg, int r0, int Tn,
                                       int g, int q) {
    const bool lo = r0 + g < Tn, hi = r0 + g + 8 < Tn;
    const float* p0 = src + size_t(r0 + g) * ldg + q;
    const float* p1 = p0 + 8 * ldg;
#pragma unroll
    for (int k = 0; k < HS / 8; ++k) {
        f[k][0] = tf32(lo ? p0[8 * k] : 0.0f);
        f[k][1] = tf32(hi ? p1[8 * k] : 0.0f);
        f[k][2] = tf32(lo ? p0[8 * k + 4] : 0.0f);
        f[k][3] = tf32(hi ? p1[8 * k + 4] : 0.0f);
    }
}

// s = a · bᵀ for the 16 rows of B from n0 (B stored [rows][HS] in shared
// memory): two 8-column tiles, s[0] and s[1]. One ldmatrix gives both
// tiles' B operands for 8 of the HS columns summed over.
template <int HS>
__device__ __forceinline__ void mma_abt(float (&s)[2][4], const unsigned (&a)[HS / 8][4], const float* Bs, int n0,
                                        int lane) {
    const int b_row = (lane % 8) + (lane / 16) * 8, b_col = ((lane / 8) % 2) * 4;
#pragma unroll
    for (int k = 0; k < HS / 8; ++k) {
        unsigned r[4];
        ldsm_x4(r, Bs + (n0 + b_row) * ld(HS) + 8 * k + b_col);
        mma(s[0], a[k], tf32(__uint_as_float(r[0])), tf32(__uint_as_float(r[1])));
        mma(s[1], a[k], tf32(__uint_as_float(r[2])), tf32(__uint_as_float(r[3])));
    }
}

// acc += p · B over B's 16 rows from k0 (B stored [rows][HS] in shared
// memory; acc HS / 8 tiles of 8 columns), p being two result tiles as
// mma_abt leaves them (columns 2q, 2q + 1 of each). Those columns serve as
// the A operand's q and q + 4, so B's rows are read as 2q and 2q + 1.
template <int HS>
__device__ __forceinline__ void mma_pb(float (&acc)[HS / 8][4], const float (&p)[2][4], const float* Bs, int k0,
                                       int g, int q) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const unsigned a[4] = {tf32(p[j][0]), tf32(p[j][2]), tf32(p[j][1]), tf32(p[j][3])};
        const float* b = Bs + (k0 + 8 * j + 2 * q) * ld(HS) + g;
#pragma unroll
        for (int n = 0; n < HS / 8; ++n) mma(acc[n], a, tf32(b[8 * n]), tf32(b[8 * n + ld(HS)]));
    }
}

// Forward. Grid (ceil(Tn / 64), NH, B), kThreads threads, fwd_smem(HS)
// floats.
template <int HS>
__global__ void __launch_bounds__(kThreads) fwd(const float* qkv, float* out, float* lse, int Tn, int C, int NH,
                                                float scale) {
    extern __shared__ __align__(128) float smem[];
    constexpr int LD = ld(HS), ND = HS / 8;
    const int b = blockIdx.z, h = blockIdx.y;
    const int t0 = (gridDim.x - 1 - blockIdx.x) * kBlockRows;  // the longest rows start first
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, q = lane % 4;
    const size_t ldg = size_t(3) * C;
    const float* base = qkv + size_t(b) * Tn * ldg + h * HS;
    const int q0 = t0 + warp * kWarpRows, last = q0 + kWarpRows - 1;  // this warp's rows
    const int t_end = min(t0 + kBlockRows, Tn), tiles = (t_end + kTile - 1) / kTile;  // keys the block sees

    // Tile i's keys and values, in stage i % kStages.
    auto stage = [&](int i) { return smem + (i % kStages) * 2 * kTile * LD; };
    auto issue = [&](int i) {
        if (i < tiles) {
            load_tile<HS>(stage(i), base + C, ldg, i * kTile, Tn);
            load_tile<HS>(stage(i) + kTile * LD, base + 2 * C, ldg, i * kTile, Tn);
        }
        cp_commit();  // a group per tile, empty or not, so cp_wait counts tiles
    };
#pragma unroll
    for (int i = 0; i < kStages - 1; ++i) issue(i);

    unsigned qf[ND][4];
    load_a<HS>(qf, base, ldg, q0, Tn, g, q);
    const float sl = scale * kLog2e;
    float o[ND][4], m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};  // rows g and g + 8
#pragma unroll
    for (int n = 0; n < ND; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    const int row[2] = {q0 + g, q0 + g + 8};

    for (int i = 0; i < tiles; ++i) {
        cp_wait<kStages - 2>();
        __syncthreads();  // tile i is in, and every warp is done with the stage it replaces
        issue(i + kStages - 1);
        const float* ks = stage(i);
        const float* vs = ks + kTile * LD;
        if (q0 >= Tn) continue;
#pragma unroll
        for (int c = 0; c < kTile; c += kChunk) {
            const int k0 = i * kTile + c;
            if (k0 > last) break;  // no row of this warp sees these keys
            float s[2][4] = {};
            mma_abt<HS>(s, qf, ks, c, lane);
            // Scale into log2 units, mask the future, and find each row's
            // maximum over the quad.
            float mx[2] = {-INFINITY, -INFINITY};
            const bool diag = k0 + kChunk - 1 > q0;
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int key = k0 + 8 * j + 2 * q + (e & 1), r = e >> 1;
                    float x = s[j][e] * sl;
                    if (diag && key > row[r]) x = -INFINITY;
                    s[j][e] = x;
                    mx[r] = fmaxf(mx[r], x);
                }
            // Online softmax: rescale what came before to the new maximum.
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
                mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
                const float m_new = fmaxf(m[r], mx[r]);
                const float m_use = m_new == -INFINITY ? 0.0f : m_new;  // a row that sees no key yet
                const float corr = exp2f(m[r] - m_use);
                m[r] = m_new;
                mx[r] = m_use;
                l[r] *= corr;
#pragma unroll
                for (int n = 0; n < ND; ++n) o[n][2 * r] *= corr, o[n][2 * r + 1] *= corr;
            }
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float p = exp2f(s[j][e] - mx[e >> 1]);
                    s[j][e] = p;
                    l[e >> 1] += p;  // this lane's columns only; summed over the quad at the end
                }
            mma_pb<HS>(o, s, vs, c, g, q);
        }
    }

    // Normalize, and write the rows two floats at a time.
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        l[r] += __shfl_xor_sync(0xffffffffu, l[r], 1);
        l[r] += __shfl_xor_sync(0xffffffffu, l[r], 2);
        if (row[r] >= Tn) continue;
        const float inv = 1.0f / l[r];
        float* dst = out + (size_t(b) * Tn + row[r]) * C + h * HS + 2 * q;
#pragma unroll
        for (int n = 0; n < ND; ++n)
            *reinterpret_cast<float2*>(dst + 8 * n) = make_float2(o[n][2 * r] * inv, o[n][2 * r + 1] * inv);
        if (q == 0) lse[(size_t(b) * NH + h) * Tn + row[r]] = (m[r] + log2f(l[r])) * kLn2;
    }
}

// Backward, dq = Σ_s ds · k_s, with ds = p · (dout·v_s - dsum) · scale and
// p = exp(scale · q·k_s - lse). Walks the keys like fwd, and writes dq.
// Grid (ceil(Tn / 64), NH, B), kThreads threads, dq_smem(HS) floats.
template <int HS>
__global__ void __launch_bounds__(kThreads) bwd_dq(const float* qkv, const float* dout, const float* lse,
                                                   const float* dsum, float* dqkv, int Tn, int C, int NH,
                                                   float scale) {
    extern __shared__ __align__(128) float smem[];
    constexpr int LD = ld(HS), ND = HS / 8;
    const int b = blockIdx.z, h = blockIdx.y;
    const int t0 = (gridDim.x - 1 - blockIdx.x) * kBlockRows;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, q = lane % 4;
    const size_t ldg = size_t(3) * C;
    const float* base = qkv + size_t(b) * Tn * ldg + h * HS;
    const float* dbase = dout + size_t(b) * Tn * C + h * HS;
    const int q0 = t0 + warp * kWarpRows, last = q0 + kWarpRows - 1;
    const int t_end = min(t0 + kBlockRows, Tn), tiles = (t_end + kTile - 1) / kTile;

    auto stage = [&](int i) { return smem + (i % kStages) * 2 * kTile * LD; };
    auto issue = [&](int i) {
        if (i < tiles) {
            load_tile<HS>(stage(i), base + C, ldg, i * kTile, Tn);
            load_tile<HS>(stage(i) + kTile * LD, base + 2 * C, ldg, i * kTile, Tn);
        }
        cp_commit();
    };
#pragma unroll
    for (int i = 0; i < kStages - 1; ++i) issue(i);

    unsigned qf[ND][4], df[ND][4];  // this warp's queries and their dout
    load_a<HS>(qf, base, ldg, q0, Tn, g, q);
    load_a<HS>(df, dbase, C, q0, Tn, g, q);
    const float sl = scale * kLog2e;
    const int row[2] = {q0 + g, q0 + g + 8};
    float L[2], D[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const size_t at = (size_t(b) * NH + h) * Tn + row[r];
        L[r] = row[r] < Tn ? lse[at] * kLog2e : 0.0f;
        D[r] = row[r] < Tn ? dsum[at] : 0.0f;
    }
    float dq[ND][4];
#pragma unroll
    for (int n = 0; n < ND; ++n) dq[n][0] = dq[n][1] = dq[n][2] = dq[n][3] = 0.0f;

    for (int i = 0; i < tiles; ++i) {
        cp_wait<kStages - 2>();
        __syncthreads();
        issue(i + kStages - 1);
        const float* ks = stage(i);
        const float* vs = ks + kTile * LD;
        if (q0 >= Tn) continue;
#pragma unroll
        for (int c = 0; c < kTile; c += kChunk) {
            const int k0 = i * kTile + c;
            if (k0 > last) break;
            // Scores q·kᵀ and dout·vᵀ, then ds in place of the scores.
            float s[2][4] = {}, dp[2][4] = {};
            mma_abt<HS>(s, qf, ks, c, lane);
            mma_abt<HS>(dp, df, vs, c, lane);
            const bool diag = k0 + kChunk - 1 > q0;
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int key = k0 + 8 * j + 2 * q + (e & 1), r = e >> 1;
                    const float p = diag && key > row[r] ? 0.0f : exp2f(s[j][e] * sl - L[r]);
                    s[j][e] = p * (dp[j][e] - D[r]) * scale;
                }
            mma_pb<HS>(dq, s, ks, c, g, q);  // dq += ds · k
        }
    }

    float* dst = dqkv + size_t(b) * Tn * ldg + h * HS + 2 * q;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        if (row[r] >= Tn) continue;
#pragma unroll
        for (int n = 0; n < ND; ++n)
            *reinterpret_cast<float2*>(dst + size_t(row[r]) * ldg + 8 * n) = make_float2(dq[n][2 * r], dq[n][2 * r + 1]);
    }
}

// Backward, for each key s: dk = Σ_t ds · q_t and dv = Σ_t p · dout_t over
// the queries t >= s, walked in tiles; writes dk and dv. Each warp takes
// its 16 keys as rows: sᵀ = k·qᵀ and dpᵀ = v·doutᵀ. Grid (ceil(Tn / 64),
// NH, B), kThreads threads, dkv_smem(HS) floats.
template <int HS>
__global__ void __launch_bounds__(kThreads) bwd_dkv(const float* qkv, const float* dout, const float* lse,
                                                    const float* dsum, float* dqkv, int Tn, int C, int NH,
                                                    float scale) {
    extern __shared__ __align__(128) float smem[];
    constexpr int LD = ld(HS), ND = HS / 8, SZ = 2 * kTile * LD + 2 * kTile;
    const int b = blockIdx.z, h = blockIdx.y, s0 = blockIdx.x * kBlockRows;  // early keys (most work) first
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, q = lane % 4;
    const size_t ldg = size_t(3) * C;
    const float* base = qkv + size_t(b) * Tn * ldg + h * HS;
    const float* dbase = dout + size_t(b) * Tn * C + h * HS;
    const size_t rbase = (size_t(b) * NH + h) * Tn;
    const int k0 = s0 + warp * kWarpRows;  // this warp's first key
    const int first = s0 / kTile, tiles = (Tn + kTile - 1) / kTile;

    // Tile i's queries, their dout, lse and dsum, in stage i % kStages.
    auto stage = [&](int i) { return smem + (i % kStages) * SZ; };
    auto issue = [&](int i) {
        if (i < tiles) {
            float* st = stage(i);
            load_tile<HS>(st, base, ldg, i * kTile, Tn);
            load_tile<HS>(st + kTile * LD, dbase, C, i * kTile, Tn);
            load_vec(st + 2 * kTile * LD, lse + rbase, i * kTile, Tn);
            load_vec(st + 2 * kTile * LD + kTile, dsum + rbase, i * kTile, Tn);
        }
        cp_commit();
    };
#pragma unroll
    for (int i = 0; i < kStages - 1; ++i) issue(first + i);

    unsigned kf[ND][4], vf[ND][4];  // this warp's keys and their values
    load_a<HS>(kf, base + C, ldg, k0, Tn, g, q);
    load_a<HS>(vf, base + 2 * C, ldg, k0, Tn, g, q);
    const float sl = scale * kLog2e;
    const int key[2] = {k0 + g, k0 + g + 8};
    float dk[ND][4], dv[ND][4];
#pragma unroll
    for (int n = 0; n < ND; ++n) {
        dk[n][0] = dk[n][1] = dk[n][2] = dk[n][3] = 0.0f;
        dv[n][0] = dv[n][1] = dv[n][2] = dv[n][3] = 0.0f;
    }

    for (int i = first; i < tiles; ++i) {
        cp_wait<kStages - 2>();
        __syncthreads();
        issue(i + kStages - 1);
        const float* qs = stage(i);
        const float* dos = qs + kTile * LD;
        const float* lt = dos + kTile * LD;
        const float* dt = lt + kTile;
        if (k0 >= Tn) continue;
#pragma unroll
        for (int c = 0; c < kTile; c += kChunk) {
            const int t = i * kTile + c;  // the chunk's first query
            if (t + kChunk - 1 < k0 || t >= Tn) continue;  // these queries see none of this warp's keys
            float s[2][4] = {}, dp[2][4] = {};
            mma_abt<HS>(s, kf, qs, c, lane);
            mma_abt<HS>(dp, vf, dos, c, lane);
            const bool diag = t < k0 + kChunk - 1;
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int cc = c + 8 * j + 2 * q + (e & 1), r = e >> 1;  // the query's column in the tile
                    const bool live = !(diag && i * kTile + cc < key[r]) && i * kTile + cc < Tn;
                    const float p = live ? exp2f(s[j][e] * sl - lt[cc] * kLog2e) : 0.0f;
                    s[j][e] = p;
                    dp[j][e] = p * (dp[j][e] - dt[cc]) * scale;  // dsᵀ
                }
            mma_pb<HS>(dv, s, dos, c, g, q);  // dv += pᵀ · dout
            mma_pb<HS>(dk, dp, qs, c, g, q);  // dk += dsᵀ · q
        }
    }

    float* dst = dqkv + size_t(b) * Tn * ldg + h * HS + 2 * q;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        if (key[r] >= Tn) continue;
        float* y = dst + size_t(key[r]) * ldg;
#pragma unroll
        for (int n = 0; n < ND; ++n) {
            *reinterpret_cast<float2*>(y + C + 8 * n) = make_float2(dk[n][2 * r], dk[n][2 * r + 1]);
            *reinterpret_cast<float2*>(y + 2 * C + 8 * n) = make_float2(dv[n][2 * r], dv[n][2 * r + 1]);
        }
    }
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
