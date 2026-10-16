// grouped_gemm.cuh — the MoE experts' matrix products, every expert in one
// launch, with each expert's rows read from GPU memory.
//
// The router decides on the GPU how many rows each expert gets. cuBLAS
// needs those sizes on the host (cublasSgemmGroupedBatched takes host
// arrays; cuBLASLt's grouped mode, which reads them from the GPU, is not
// supported on the RTX 4070), so before this the host waited for the
// router in every layer of every pass, then launched one product per
// expert. Here a block finds its expert and its tile in `starts` (the
// experts' first rows, a prefix sum the GPU computes), so the host only
// queues work.
//
// A group g (an expert) is C_g = beta·C_g + A_g·B_g, row-major, in one of two
// shapes:
// - rows (forward, and the input gradient): A_g and C_g are rows
//   [starts[g], starts[g + 1]) of A and C; B_g = B + g·strideB. M_g is the
//   group's row count, N and K are fixed.
// - reduce (the weight gradient): the group's rows are the reduction:
//   A_g(m, k) and B_g(k, n) read row starts[g] + k; C_g = C + g·strideC. K_g
//   is the group's row count, M and N are fixed.
// TA: A is stored [K][M] (A(m, k) at A[k·lda + m]) instead of [M][K]; TB: B
// is stored [N][K] instead of [K][N].
//
// A rows-mode product can also finish SwiGLU's backward pass in its
// epilogue (Problem::swiglu): the experts' eproj input gradient is SwiGLU's
// output gradient, which is then never stored.
//
// Three kernels: on the tensor cores in bf16 (inputs bf16) and in TF32
// (inputs fp32, rounded to TF32 as cuBLAS's TF32 mode does), in tiles
// (Tiling) whose operands are staged in shared memory with cp.async;
// and a plain fp32 kernel for exact products, which also takes the cases the
// tensor-core kernels cannot (rows not 16-byte aligned), rounding to TF32 or
// reading bf16 itself. Sums are fp32 throughout.
#pragma once

#include <cuda_bf16.h>
#include <mma.h>

#include <cstdint>
#include <type_traits>

namespace gg {

using namespace nvcuda;

struct Problem {
    const void* A = nullptr;
    const void* B = nullptr;
    float* C = nullptr;
    int lda = 0, ldb = 0, ldc = 0;
    long long strideB = 0, strideC = 0;  // per group: B in rows mode, C in reduce mode
    int M = 0, N = 0, K = 0;             // rows mode: N, K; reduce mode: M, N
    const int* starts = nullptr;         // [groups + 1] first rows, on the GPU
    int groups = 0;
    float beta = 0.0f;
    // With launch<..., Fused = true> (rows mode, beta 0): instead of
    // C(m, n) = v, v is the gradient of SwiGLU's output at (m, n). With
    // SwiGLU's input in swiglu (fp32, rows of 2N: gate | up), the gate's
    // and up's gradients go into swiglu_out (bf16, the same layout) at
    // (m, n) and (m, N + n), and C is not written.
    const float* swiglu = nullptr;
    __nv_bfloat16* swiglu_out = nullptr;
};

// swiglu_backward for one element (gpu.cu's swiglu_bwd and the epilogue
// above share it): from the gate g and up u of SwiGLU's input and the
// gradient d of its output, the gradients of g and u.
__device__ __forceinline__ void swiglu_grad(float g, float u, float d, float& dg, float& du) {
    const float sg = 1.0f / (1.0f + expf(-g));
    dg = d * u * sg * (1.0f + g * (1.0f - sg));
    du = d * g * sg;
}

// The group and tile a block works on, and its matrices.
struct Tile {
    const char* A;
    const char* B;
    float* C;
    int M, K, m0, n0;  // M, K: this group's
    int row0;          // rows mode: the group's first row
    bool live;
};

template <bool Reduce, int BM, int BN>
__device__ Tile find_tile(const Problem& p, size_t esz) {
    Tile t{};
    if (Reduce) {
        const int g = blockIdx.z, s = p.starts[g];
        t.A = static_cast<const char*>(p.A) + size_t(s) * p.lda * esz;
        t.B = static_cast<const char*>(p.B) + size_t(s) * p.ldb * esz;
        t.C = p.C + size_t(g) * p.strideC;
        t.M = p.M;
        t.K = p.starts[g + 1] - s;
        t.m0 = blockIdx.x * BM;
        t.n0 = blockIdx.y * BN;
        t.live = t.m0 < t.M;
        return t;
    }
    // Rows mode: blockIdx.x counts (group, row tile) pairs, groups in order.
    int x = blockIdx.x;
    for (int g = 0; g < p.groups; ++g) {
        const int s = p.starts[g], n = p.starts[g + 1] - s, tiles = (n + BM - 1) / BM;
        if (x < tiles) {
            t.A = static_cast<const char*>(p.A) + size_t(s) * p.lda * esz;
            t.B = static_cast<const char*>(p.B) + size_t(g) * p.strideB * esz;
            t.C = p.C + size_t(s) * p.ldc;
            t.M = n;
            t.K = p.K;
            t.row0 = s;
            t.m0 = x * BM;
            t.n0 = blockIdx.y * BN;
            t.live = true;
            return t;
        }
        x -= tiles;
    }
    t.live = false;
    return t;
}

// C(m, n) = beta·C(m, n) + v, inside the group (m < M, n < N). With beta 0 the
// old value is not read (it may be anything, NaN included). Fused: SwiGLU's
// backward of v instead (Problem::swiglu). A compile-time choice: the extra
// code in every kernel's epilogue costs registers, and so speed, even where
// it is not taken.
template <bool Fused>
__device__ __forceinline__ void put(const Problem& p, const Tile& t, int m, int n, float v) {
    if (m >= t.M || n >= p.N) return;
    if constexpr (Fused) {
        const size_t at = size_t(t.row0 + m) * 2 * p.N + n;
        float dg, du;
        swiglu_grad(p.swiglu[at], p.swiglu[at + p.N], v, dg, du);
        p.swiglu_out[at] = __float2bfloat16(dg);
        p.swiglu_out[at + p.N] = __float2bfloat16(du);
        return;
    }
    float& c = t.C[size_t(m) * p.ldc + n];
    c = p.beta == 0.0f ? v : p.beta * c + v;
}

// put of v0, v1 at (m, n), (m, n + 1), n even. Fused, with N even (so every
// pair starts 8 bytes aligned): the gate and up pairs read as float2, the
// gradients written as bf16 pairs, the same values as two put calls.
template <bool Fused>
__device__ __forceinline__ void put2(const Problem& p, const Tile& t, int m, int n, float v0, float v1) {
    if constexpr (Fused) {
        if (m < t.M && n + 1 < p.N && p.N % 2 == 0) {
            const size_t at = size_t(t.row0 + m) * 2 * p.N + n;
            const float2 g = *reinterpret_cast<const float2*>(p.swiglu + at);
            const float2 u = *reinterpret_cast<const float2*>(p.swiglu + at + p.N);
            float dg0, du0, dg1, du1;
            swiglu_grad(g.x, u.x, v0, dg0, du0);
            swiglu_grad(g.y, u.y, v1, dg1, du1);
            *reinterpret_cast<__nv_bfloat162*>(p.swiglu_out + at) = __floats2bfloat162_rn(dg0, dg1);
            *reinterpret_cast<__nv_bfloat162*>(p.swiglu_out + at + p.N) = __floats2bfloat162_rn(du0, du1);
            return;
        }
    }
    put<Fused>(p, t, m, n, v0);
    put<Fused>(p, t, m, n + 1, v1);
}

// ---------------------------------------------------------------------------
// Tensor cores
// ---------------------------------------------------------------------------

template <class T>
struct TC;
template <>
struct TC<__nv_bfloat16> {
    static constexpr int kStep = 16;  // WMMA k
    using FragA = wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major>;
    using FragAT = wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major>;
    using FragB = wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major>;
    using FragBT = wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major>;
    using Acc = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
    template <class F>
    static __device__ void round(F&) {}
};
template <>
struct TC<float> {
    static constexpr int kStep = 8;
    using FragA = wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major>;
    using FragAT = wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::col_major>;
    using FragB = wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major>;
    using FragBT = wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major>;
    using Acc = wmma::fragment<wmma::accumulator, 16, 16, 8, float>;
    template <class F>
    static __device__ void round(F& f) {
#pragma unroll
        for (int i = 0; i < f.num_elements; ++i) f.x[i] = wmma::__float_to_tf32(f.x[i]);
    }
};

// A tensor-core tiling: WarpsM x WarpsN warps, each FM x FN WMMA tiles of
// 16 x 16, so a block computes BM x BN of C; BK columns of A (rows of B)
// per step of the walk along K (BKBytes bytes of each row), Stages of them
// in flight (cp.async).
template <int WarpsM_, int WarpsN_, int FM_, int FN_, int BKBytes, int Stages_>
struct Tiling {
    static constexpr int WarpsM = WarpsM_, WarpsN = WarpsN_, FM = FM_, FN = FN_, Stages = Stages_;
    static constexpr int threads = WarpsM * WarpsN * 32, BM = WarpsM * FM * 16, BN = WarpsN * FN * 16;
    static constexpr int bk_bytes = BKBytes;
};
// Measured on the RTX 4070 laptop GPU on the night model's expert products
// (8 x 512 x top-2 rows per micro-batch, C 768, Fe 1024; scratchpad bench,
// 2026-10-16): 128 x 256 tiles, three stages for the row products; 128 x 128,
// two stages for the weight gradients. All six products of a layer: 4.8 ms
// in bf16 against 5.3 ms for cuBLAS, one product per expert; 9.0 ms
// against 10.0 in TF32.
using RowsTiling = Tiling<2, 4, 4, 4, 64, 3>;
using ReduceTiling = Tiling<2, 4, 4, 2, 64, 2>;
using SmallRowsTiling = Tiling<1, 4, 1, 2, 64, 2>;  // 16 x 128: decoding's few rows per expert
constexpr int kThreads = 256;  // the plain kernel's

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, int bytes) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// Rows [r0, r0 + R) and columns [c0, c0 + Cc) of a row-major matrix g
// (row stride ld) into shared memory s (row stride sld), 16 bytes at a time
// by Threads threads; what lies outside rows < rlim, columns < clim is zero.
template <class T, int R, int Cc, int Threads>
__device__ __forceinline__ void load_tile(T* s, int sld, const T* g, int ld, int r0, int c0, int rlim, int clim) {
    constexpr int kVec = 16 / int(sizeof(T)), kPerRow = Cc / kVec;
#pragma unroll
    for (int i = threadIdx.x; i < R * kPerRow; i += Threads) {
        const int r = i / kPerRow, c = (i % kPerRow) * kVec, gr = r0 + r, gc = c0 + c;
        int bytes = 0;
        const T* src = g;
        if (gr < rlim && gc < clim) {
            bytes = (clim - gc < kVec ? clim - gc : kVec) * int(sizeof(T));
            src = g + size_t(gr) * ld + gc;
        }
        cp_async16(s + r * sld + c, src, bytes);
    }
}

template <class T, class L, bool TA, bool TB>
struct Smem {
    static constexpr int BK = L::bk_bytes / int(sizeof(T)), P = 16 / int(sizeof(T));  // pad rows by 16 bytes
    // A tile [BM][BK] (TA: [BK][BM]); B tile [BK][BN] (TB: [BN][BK]).
    static constexpr int lda = TA ? L::BM + P : BK + P, a_size = TA ? BK * lda : L::BM * lda;
    static constexpr int ldb = TB ? BK + P : L::BN + P, b_size = TB ? L::BN * ldb : BK * ldb;
    static constexpr int stage = a_size + b_size;
    static constexpr size_t bytes = size_t(L::Stages) * stage * sizeof(T);
    static_assert(a_size * sizeof(T) % 32 == 0 && stage * sizeof(T) % 32 == 0, "WMMA needs 32-byte aligned tiles");
    static_assert(bytes >= size_t(L::threads / 32) * 256 * sizeof(float), "room for the warps' scratch");
};

template <class T, class L, bool TA, bool TB, bool Reduce, bool Fused>
__global__ void __launch_bounds__(L::threads) grouped_tc(Problem p) {
    using S = Smem<T, L, TA, TB>;
    using F = TC<T>;
    constexpr int BK = S::BK, FM = L::FM, FN = L::FN;
    const Tile t = find_tile<Reduce, L::BM, L::BN>(p, sizeof(T));
    if (!t.live) return;
    extern __shared__ __align__(32) unsigned char smem_raw[];
    T* smem = reinterpret_cast<T*>(smem_raw);
    const T* A = reinterpret_cast<const T*>(t.A);
    const T* B = reinterpret_cast<const T*>(t.B);
    const int warp = threadIdx.x / 32, wm = warp / L::WarpsN, wn = warp % L::WarpsN;

    auto load_stage = [&](int stage, int k0) {
        T* As = smem + stage * S::stage;
        T* Bs = As + S::a_size;
        if (TA) load_tile<T, BK, L::BM, L::threads>(As, S::lda, A, p.lda, k0, t.m0, t.K, t.M);
        else load_tile<T, L::BM, BK, L::threads>(As, S::lda, A, p.lda, t.m0, k0, t.M, t.K);
        if (TB) load_tile<T, L::BN, BK, L::threads>(Bs, S::ldb, B, p.ldb, t.n0, k0, p.N, t.K);
        else load_tile<T, BK, L::BN, L::threads>(Bs, S::ldb, B, p.ldb, k0, t.n0, t.K, p.N);
    };

    typename F::Acc acc[FM][FN];
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    // Stages - 1 steps ahead. Every iteration commits one group (perhaps
    // empty), so once all but Stages - 2 groups are complete, this step's
    // tiles have arrived.
    const int steps = (t.K + BK - 1) / BK;
#pragma unroll
    for (int s = 0; s < L::Stages - 1; ++s) {
        if (s < steps) load_stage(s, s * BK);
        cp_async_commit();
    }
    for (int s = 0; s < steps; ++s) {
        cp_async_wait<L::Stages - 2>();
        __syncthreads();  // this step's tiles are in; every warp is done with the slot refilled next
        const int next = s + L::Stages - 1;
        if (next < steps) load_stage(next % L::Stages, next * BK);
        cp_async_commit();
        const T* As = smem + (s % L::Stages) * S::stage;
        const T* Bs = As + S::a_size;
#pragma unroll
        for (int kk = 0; kk < BK; kk += F::kStep) {
            typename F::FragB bf[FN];
            typename F::FragBT bft[FN];
#pragma unroll
            for (int j = 0; j < FN; ++j) {
                const int n = (wn * FN + j) * 16;
                if (TB) {
                    wmma::load_matrix_sync(bft[j], Bs + n * S::ldb + kk, S::ldb);
                    F::round(bft[j]);
                } else {
                    wmma::load_matrix_sync(bf[j], Bs + kk * S::ldb + n, S::ldb);
                    F::round(bf[j]);
                }
            }
#pragma unroll
            for (int i = 0; i < FM; ++i) {
                const int m = (wm * FM + i) * 16;
                if (TA) {
                    typename F::FragAT af;
                    wmma::load_matrix_sync(af, As + kk * S::lda + m, S::lda);
                    F::round(af);
#pragma unroll
                    for (int j = 0; j < FN; ++j) {
                        if (TB) wmma::mma_sync(acc[i][j], af, bft[j], acc[i][j]);
                        else wmma::mma_sync(acc[i][j], af, bf[j], acc[i][j]);
                    }
                } else {
                    typename F::FragA af;
                    wmma::load_matrix_sync(af, As + m * S::lda + kk, S::lda);
                    F::round(af);
#pragma unroll
                    for (int j = 0; j < FN; ++j) {
                        if (TB) wmma::mma_sync(acc[i][j], af, bft[j], acc[i][j]);
                        else wmma::mma_sync(acc[i][j], af, bf[j], acc[i][j]);
                    }
                }
            }
        }
    }
    cp_async_wait<0>();
    __syncthreads();  // the tiles' memory becomes the warps' scratch
    // Out through a 16 x 16 scratch per warp.
    float* scratch = reinterpret_cast<float*>(smem_raw) + warp * 256;
    const int lane = threadIdx.x % 32;
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j) {
            wmma::store_matrix_sync(scratch, acc[i][j], 16, wmma::mem_row_major);
            __syncwarp();
            const int m = t.m0 + (wm * FM + i) * 16, n = t.n0 + (wn * FN + j) * 16;
            for (int e = lane; e < 256; e += 32) put<Fused>(p, t, m + e / 16, n + e % 16, scratch[e]);
            __syncwarp();
        }
}

// ---------------------------------------------------------------------------
// bf16 on the tensor cores through mma.sync, with operands loaded by ldmatrix
// ---------------------------------------------------------------------------
//
// The same work as grouped_tc in bf16, with more control: each warp keeps a
// WM x WN block of C in registers and multiplies 16 x 8 x 16 at a time
// (mma.sync m16n8k16, fp32 sums), its operands read from shared memory by
// ldmatrix (.trans for A stored [K][M] and B stored [K][N]). Shared-memory
// rows are padded by 16 bytes, so the eight rows an ldmatrix reads fall in
// different banks.
template <int WarpsM_, int WarpsN_, int WM_, int WN_, int Stages_>
struct MmaTiling {
    static constexpr int WarpsM = WarpsM_, WarpsN = WarpsN_, WM = WM_, WN = WN_, Stages = Stages_;
    static constexpr int threads = WarpsM * WarpsN * 32, BM = WarpsM * WM, BN = WarpsN * WN;
    static constexpr int bk_bytes = 64;          // 32 bf16 per row and step
    static constexpr int MT = WM / 16, NT = WN / 8;  // mma tiles per warp
    static_assert(WM % 16 == 0 && WN % 16 == 0, "warp tiles of whole 16 x 16 pairs");
};

__device__ __forceinline__ unsigned smem_u32(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4_t(unsigned (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma_bf16(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <class L, bool TA, bool TB, bool Reduce, bool Fused>
__global__ void __launch_bounds__(L::threads) grouped_mma(Problem p) {
    using T = __nv_bfloat16;
    using S = Smem<T, L, TA, TB>;
    constexpr int BK = S::BK, MT = L::MT, NT = L::NT;
    const Tile t = find_tile<Reduce, L::BM, L::BN>(p, sizeof(T));
    if (!t.live) return;
    extern __shared__ __align__(32) unsigned char smem_raw[];
    T* smem = reinterpret_cast<T*>(smem_raw);
    const T* A = reinterpret_cast<const T*>(t.A);
    const T* B = reinterpret_cast<const T*>(t.B);
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, wm = warp / L::WarpsN, wn = warp % L::WarpsN;

    auto load_stage = [&](int stage, int k0) {
        T* As = smem + stage * S::stage;
        T* Bs = As + S::a_size;
        if (TA) load_tile<T, BK, L::BM, L::threads>(As, S::lda, A, p.lda, k0, t.m0, t.K, t.M);
        else load_tile<T, L::BM, BK, L::threads>(As, S::lda, A, p.lda, t.m0, k0, t.M, t.K);
        if (TB) load_tile<T, L::BN, BK, L::threads>(Bs, S::ldb, B, p.ldb, t.n0, k0, p.N, t.K);
        else load_tile<T, BK, L::BN, L::threads>(Bs, S::ldb, B, p.ldb, k0, t.n0, t.K, p.N);
    };

    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.0f;

    // Where this lane's ldmatrix rows start, within a 16 x 16 block.
    const int a_row = TA ? (lane % 8) + (lane / 16) * 8 : lane % 16;  // TA: k; else m
    const int a_col = TA ? ((lane / 8) % 2) * 8 : (lane / 16) * 8;    // TA: m; else k
    const int b_row = TB ? (lane % 8) + (lane / 16) * 8 : (lane % 8) + ((lane / 8) % 2) * 8;  // TB: n; else k
    const int b_col = TB ? ((lane / 8) % 2) * 8 : (lane / 16) * 8;                            // TB: k; else n

    const int steps = (t.K + BK - 1) / BK;
#pragma unroll
    for (int s = 0; s < L::Stages - 1; ++s) {
        if (s < steps) load_stage(s, s * BK);
        cp_async_commit();
    }
    for (int s = 0; s < steps; ++s) {
        cp_async_wait<L::Stages - 2>();
        __syncthreads();
        const int next = s + L::Stages - 1;
        if (next < steps) load_stage(next % L::Stages, next * BK);
        cp_async_commit();
        const T* As = smem + (s % L::Stages) * S::stage;
        const T* Bs = As + S::a_size;
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            unsigned b[NT][2];
#pragma unroll
            for (int j = 0; j < NT; j += 2) {
                const int n = wn * L::WN + j * 8;
                unsigned r[4];
                if (TB) ldsm_x4(r, Bs + (n + b_row) * S::ldb + kk + b_col);
                else ldsm_x4_t(r, Bs + (kk + b_row) * S::ldb + n + b_col);
                b[j][0] = r[0], b[j][1] = r[1], b[j + 1][0] = r[2], b[j + 1][1] = r[3];
            }
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int m = wm * L::WM + i * 16;
                unsigned a[4];
                if (TA) ldsm_x4_t(a, As + (kk + a_row) * S::lda + m + a_col);
                else ldsm_x4(a, As + (m + a_row) * S::lda + kk + a_col);
#pragma unroll
                for (int j = 0; j < NT; ++j) mma_bf16(acc[i][j], a, b[j][0], b[j][1]);
            }
        }
    }
    cp_async_wait<0>();
    // c0, c1 at (row, col), (row, col + 1); c2, c3 eight rows below.
    const int row = t.m0 + wm * L::WM + lane / 4, col = t.n0 + wn * L::WN + (lane % 4) * 2;
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int r0 = row + i * 16, c0 = col + j * 8;
            put2<Fused>(p, t, r0, c0, acc[i][j][0], acc[i][j][1]);
            put2<Fused>(p, t, r0 + 8, c0, acc[i][j][2], acc[i][j][3]);
        }
}

template <class L, bool TA, bool TB, bool Reduce, bool Fused>
void launch_mma(const Problem& p, int total_rows) {
    const dim3 grid = Reduce ? dim3((p.M + L::BM - 1) / L::BM, (p.N + L::BN - 1) / L::BN, p.groups)
                             : dim3((total_rows + L::BM - 1) / L::BM + p.groups, (p.N + L::BN - 1) / L::BN);
    if (grid.x == 0 || grid.y == 0 || grid.z == 0) return;
    constexpr size_t bytes = Smem<__nv_bfloat16, L, TA, TB>::bytes;
    static const bool once = (cudaFuncSetAttribute(grouped_mma<L, TA, TB, Reduce, Fused>,
                                                   cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)),
                              true);
    (void)once;
    grouped_mma<L, TA, TB, Reduce, Fused><<<grid, L::threads, bytes>>>(p);
}

// ---------------------------------------------------------------------------
// Plain fp32 (exact products), and the fallback for unaligned rows
// ---------------------------------------------------------------------------

// A plain tiling: SM x SN of C per block of 256 threads, each RM x RN
// results (strided, so neighbouring threads read neighbouring shared
// memory), SK columns of A per step.
template <int SM_, int SN_, int SK_, int RM_, int RN_>
struct PlainTiling {
    static constexpr int SM = SM_, SN = SN_, SK = SK_, RM = RM_, RN = RN_;
    static constexpr int TN = SN / RN;  // threads across
    static_assert((SM / RM) * (SN / RN) == kThreads, "256 threads");
};
using PlainBig = PlainTiling<128, 128, 8, 8, 8>;
using PlainSmall = PlainTiling<32, 64, 16, 2, 4>;  // decoding: a few rows per expert

template <class T>
__device__ __forceinline__ float value(T x, bool tf32) {
    const float f = float(x);
    return tf32 ? wmma::__float_to_tf32(f) : f;
}
template <>
__device__ __forceinline__ float value(__nv_bfloat16 x, bool) {
    return __bfloat162float(x);
}

template <class T, class L, bool TA, bool TB, bool Reduce, bool Fused>
__global__ void __launch_bounds__(kThreads) grouped_simt(Problem p, bool tf32) {
    constexpr int SM = L::SM, SN = L::SN, SK = L::SK, RM = L::RM, RN = L::RN;
    const Tile t = find_tile<Reduce, SM, SN>(p, sizeof(T));
    if (!t.live) return;
    __shared__ float As[SK][SM + 4], Bs[SK][SN + 4];  // As[k][m], Bs[k][n]
    const T* A = reinterpret_cast<const T*>(t.A);
    const T* B = reinterpret_cast<const T*>(t.B);
    const int tm = threadIdx.x / L::TN, tn = threadIdx.x % L::TN;  // results m = tm + (SM / RM) i, n = tn + TN j
    float acc[RM][RN] = {};
    for (int k0 = 0; k0 < t.K; k0 += SK) {
        // Neighbouring threads read neighbouring addresses of A and B as
        // they are stored.
        for (int i = threadIdx.x; i < SK * SM; i += kThreads) {
            const int k = TA ? i / SM : i % SK, m = TA ? i % SM : i / SK, gk = k0 + k, gm = t.m0 + m;
            const bool in = gk < t.K && gm < t.M;
            As[k][m] = in ? value(A[TA ? size_t(gk) * p.lda + gm : size_t(gm) * p.lda + gk], tf32) : 0.0f;
        }
        for (int i = threadIdx.x; i < SK * SN; i += kThreads) {
            const int k = TB ? i % SK : i / SN, n = TB ? i / SK : i % SN, gk = k0 + k, gn = t.n0 + n;
            const bool in = gk < t.K && gn < p.N;
            Bs[k][n] = in ? value(B[TB ? size_t(gn) * p.ldb + gk : size_t(gk) * p.ldb + gn], tf32) : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < SK; ++k) {
            float a[RM], b[RN];
#pragma unroll
            for (int i = 0; i < RM; ++i) a[i] = As[k][tm + (SM / RM) * i];
#pragma unroll
            for (int j = 0; j < RN; ++j) b[j] = Bs[k][tn + L::TN * j];
#pragma unroll
            for (int i = 0; i < RM; ++i)
#pragma unroll
                for (int j = 0; j < RN; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < RM; ++i)
#pragma unroll
        for (int j = 0; j < RN; ++j) put<Fused>(p, t, t.m0 + tm + (SM / RM) * i, t.n0 + tn + L::TN * j, acc[i][j]);
}

// ---------------------------------------------------------------------------
// Launch
// ---------------------------------------------------------------------------

enum class Kind { fp32, tf32, bf16 };

// Whether every row of the operands starts 16 bytes aligned, as the
// tensor-core kernels' cp.async loads need.
inline bool aligned16(const Problem& p, size_t esz, bool reduce) {
    auto ok = [](const void* q) { return reinterpret_cast<uintptr_t>(q) % 16 == 0; };
    const size_t vec = 16 / esz;
    return ok(p.A) && ok(p.B) && p.lda % vec == 0 && p.ldb % vec == 0 &&
           (reduce || p.strideB % static_cast<long long>(vec) == 0);
}

template <class T, class L, bool TA, bool TB, bool Reduce, bool Fused>
void launch_tc(const Problem& p, int total_rows) {
    const dim3 grid = Reduce ? dim3((p.M + L::BM - 1) / L::BM, (p.N + L::BN - 1) / L::BN, p.groups)
                             : dim3((total_rows + L::BM - 1) / L::BM + p.groups, (p.N + L::BN - 1) / L::BN);
    if (grid.x == 0 || grid.y == 0 || grid.z == 0) return;
    constexpr size_t bytes = Smem<T, L, TA, TB>::bytes;
    static const bool once = (cudaFuncSetAttribute(grouped_tc<T, L, TA, TB, Reduce, Fused>,
                                                   cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)),
                              true);
    (void)once;
    grouped_tc<T, L, TA, TB, Reduce, Fused><<<grid, L::threads, bytes>>>(p);
}

template <class T, class L, bool TA, bool TB, bool Reduce, bool Fused>
void launch_plain(const Problem& p, int total_rows, bool tf32) {
    const dim3 grid = Reduce ? dim3((p.M + L::SM - 1) / L::SM, (p.N + L::SN - 1) / L::SN, p.groups)
                             : dim3((total_rows + L::SM - 1) / L::SM + p.groups, (p.N + L::SN - 1) / L::SN);
    if (grid.x == 0 || grid.y == 0 || grid.z == 0) return;
    grouped_simt<T, L, TA, TB, Reduce, Fused><<<grid, kThreads>>>(p, tf32);
}

// One grouped product. total_rows (rows mode): the groups' rows together,
// which bounds the number of row tiles the launch needs. With few rows per
// group (decoding: about B·top-k / experts), tiles of fewer rows waste less.
// Fused: SwiGLU's backward in the epilogue (rows mode, Problem::swiglu set).
template <bool TA, bool TB, bool Reduce, bool Fused = false>
void launch(Kind kind, const Problem& p, int total_rows) {
    static_assert(!(Fused && Reduce), "SwiGLU's backward follows a rows-mode product");
    const bool bf16 = kind == Kind::bf16;
    const size_t esz = bf16 ? 2 : 4;
    const bool few = !Reduce && p.groups > 0 && total_rows / p.groups < 64;
    if (kind != Kind::fp32 && aligned16(p, esz, Reduce)) {
        if (Reduce) {
            if (bf16) launch_tc<__nv_bfloat16, ReduceTiling, TA, TB, Reduce, Fused>(p, total_rows);
            else launch_tc<float, ReduceTiling, TA, TB, Reduce, Fused>(p, total_rows);
        } else if (few) {
            if (bf16) launch_tc<__nv_bfloat16, SmallRowsTiling, TA, TB, Reduce, Fused>(p, total_rows);
            else launch_tc<float, SmallRowsTiling, TA, TB, Reduce, Fused>(p, total_rows);
        } else {
            // bf16, measured in training on the night model (2026-10-16):
            // the forward products on WMMA 128 x 128 tiles, the input
            // gradients through mma.sync on 128 x 256.
            if (!bf16) launch_tc<float, RowsTiling, TA, TB, Reduce, Fused>(p, total_rows);
            else if constexpr (TB) launch_tc<__nv_bfloat16, ReduceTiling, TA, TB, Reduce, Fused>(p, total_rows);
            else launch_mma<MmaTiling<2, 4, 64, 64, 2>, TA, TB, Reduce, Fused>(p, total_rows);
        }
        return;
    }
    const bool tf32 = kind == Kind::tf32;
    if (few) {
        if (bf16) launch_plain<__nv_bfloat16, PlainSmall, TA, TB, Reduce, Fused>(p, total_rows, false);
        else launch_plain<float, PlainSmall, TA, TB, Reduce, Fused>(p, total_rows, tf32);
    } else {
        if (bf16) launch_plain<__nv_bfloat16, PlainBig, TA, TB, Reduce, Fused>(p, total_rows, false);
        else launch_plain<float, PlainBig, TA, TB, Reduce, Fused>(p, total_rows, tf32);
    }
}

}  // namespace gg
