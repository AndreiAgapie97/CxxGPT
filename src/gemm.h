// gemm.h — matrix multiply for training: C (+)= A·B.
//
// A and B are strided views, so one kernel computes all three products a
// linear layer needs, without transposing anything in memory:
//   forward      Y  = X · Wᵀ     (B is W read column-wise)
//   input grad   dX = dY · W
//   weight grad  dW += dYᵀ · X   (A is dY read column-wise)
//
// float runs a packed, register-blocked kernel (the GotoBLAS/BLIS scheme);
// double runs a plain triple loop, which the gradient checks use as ground
// truth.
#pragma once

#include <algorithm>
#include <cstring>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#include "common.h"
#include "thread_pool.h"

// Element (r, c) of a matrix is p[r * rs + c * cs]. A row-major matrix with
// n columns is {p, n, 1}; its transpose is {p, 1, n}.
template <class T>
struct MatView {
    const T* p;
    long rs, cs;
    T at(long r, long c) const { return p[r * rs + c * cs]; }
};

namespace gemm_detail {

// One SIMD register of floats, and the few operations the kernel needs.
#if defined(__AVX512F__)
using vf = __m512;
constexpr int W = 16;
constexpr int MR = 12;  // 12 x 2 accumulators + 2 B loads + 1 broadcast of 32 registers
inline vf vzero() { return _mm512_setzero_ps(); }
inline vf vload(const float* p) { return _mm512_loadu_ps(p); }
inline void vstore(float* p, vf v) { _mm512_storeu_ps(p, v); }
inline vf vbcast(float x) { return _mm512_set1_ps(x); }
inline vf vfma(vf a, vf b, vf c) { return _mm512_fmadd_ps(a, b, c); }
inline vf vadd(vf a, vf b) { return _mm512_add_ps(a, b); }
#elif defined(__AVX__)
using vf = __m256;
constexpr int W = 8;
constexpr int MR = 6;   // 6 x 2 accumulators + 2 B loads + 1 broadcast of 16 registers
inline vf vzero() { return _mm256_setzero_ps(); }
inline vf vload(const float* p) { return _mm256_loadu_ps(p); }
inline void vstore(float* p, vf v) { _mm256_storeu_ps(p, v); }
inline vf vbcast(float x) { return _mm256_set1_ps(x); }
#if defined(__FMA__)
inline vf vfma(vf a, vf b, vf c) { return _mm256_fmadd_ps(a, b, c); }
#else
inline vf vfma(vf a, vf b, vf c) { return _mm256_add_ps(_mm256_mul_ps(a, b), c); }
#endif
inline vf vadd(vf a, vf b) { return _mm256_add_ps(a, b); }
#else
// Portable fallback: a compiler vector type (SSE on plain x86-64).
typedef float vf __attribute__((vector_size(16)));
constexpr int W = 4;
constexpr int MR = 6;
inline vf vzero() { return vf{}; }
inline vf vload(const float* p) { vf v; std::memcpy(&v, p, sizeof v); return v; }
inline void vstore(float* p, vf v) { std::memcpy(p, &v, sizeof v); }
inline vf vbcast(float x) { return vf{} + x; }
inline vf vfma(vf a, vf b, vf c) { return a * b + c; }
inline vf vadd(vf a, vf b) { return a + b; }
#endif

constexpr int NR = 2 * W;     // micro-tile is MR x NR
constexpr int KC = 256;       // depth of one packed block
constexpr int MC = MR * 16;   // rows of A packed at a time (~200 KB for AVX-512: in L2)
constexpr int NC = NR * 128;  // columns of B packed at a time

// C[0:mr, 0:nr] = (overwrite ? 0 : C) + Ap · Bp, where Ap holds kc steps of
// MR values and Bp holds kc steps of NR values (zero-padded past mr / nr).
inline void micro_kernel(int kc, const float* Ap, const float* Bp, float* C, long ldc,
                         int mr, int nr, bool overwrite) {
    vf c0[MR], c1[MR];
    for (int i = 0; i < MR; ++i) c0[i] = c1[i] = vzero();
    for (int k = 0; k < kc; ++k) {
        const vf b0 = vload(Bp), b1 = vload(Bp + W);
#pragma GCC unroll 16
        for (int i = 0; i < MR; ++i) {
            const vf a = vbcast(Ap[i]);
            c0[i] = vfma(a, b0, c0[i]);
            c1[i] = vfma(a, b1, c1[i]);
        }
        Ap += MR;
        Bp += NR;
    }
    if (mr == MR && nr == NR) {
        for (int i = 0; i < MR; ++i) {
            float* c = C + i * ldc;
            if (overwrite) {
                vstore(c, c0[i]);
                vstore(c + W, c1[i]);
            } else {
                vstore(c, vadd(vload(c), c0[i]));
                vstore(c + W, vadd(vload(c + W), c1[i]));
            }
        }
        return;
    }
    alignas(64) float tile[MR * NR];
    for (int i = 0; i < MR; ++i) {
        vstore(tile + i * NR, c0[i]);
        vstore(tile + i * NR + W, c1[i]);
    }
    for (int i = 0; i < mr; ++i)
        for (int j = 0; j < nr; ++j)
            C[i * ldc + j] = (overwrite ? 0.0f : C[i * ldc + j]) + tile[i * NR + j];
}

// Rows [i0, i0+mc) x depth [p0, p0+kc) of A, as micro-panels of MR rows.
inline void pack_a(MatView<float> A, int i0, int mc, int p0, int kc, float* Ap) {
    for (int ir = 0; ir < mc; ir += MR, Ap += MR * kc) {
        const int mr = std::min(MR, mc - ir);
        if (A.cs == 1) {  // rows are contiguous: walk along them
            for (int i = 0; i < MR; ++i) {
                if (i >= mr) {
                    for (int k = 0; k < kc; ++k) Ap[k * MR + i] = 0.0f;
                    continue;
                }
                const float* a = A.p + (i0 + ir + i) * A.rs + p0;
                for (int k = 0; k < kc; ++k) Ap[k * MR + i] = a[k];
            }
        } else {
            for (int k = 0; k < kc; ++k) {
                const float* a = A.p + (p0 + k) * A.cs + (i0 + ir) * A.rs;
                for (int i = 0; i < MR; ++i) Ap[k * MR + i] = i < mr ? a[i * A.rs] : 0.0f;
            }
        }
    }
}

// Depth [p0, p0+kc) x columns [j0 + jr0, j0 + jr1) of B, as micro-panels of NR
// columns. jr0 and jr1 are multiples of NR, relative to the panel start j0.
inline void pack_b(MatView<float> B, int p0, int kc, int j0, int nc, int jr0, int jr1, float* Bp) {
    for (int jr = jr0; jr < jr1; jr += NR) {
        float* bp = Bp + long(jr / NR) * NR * kc;
        const int nr = std::min(NR, nc - jr);
        if (B.rs == 1 && B.cs != 1) {  // columns are contiguous: walk down them
            for (int j = 0; j < NR; ++j) {
                if (j >= nr) {
                    for (int k = 0; k < kc; ++k) bp[k * NR + j] = 0.0f;
                    continue;
                }
                const float* b = B.p + (j0 + jr + j) * B.cs + p0;
                for (int k = 0; k < kc; ++k) bp[k * NR + j] = b[k];
            }
        } else {
            for (int k = 0; k < kc; ++k) {
                const float* b = B.p + (p0 + k) * B.rs + (j0 + jr) * B.cs;
                for (int j = 0; j < NR; ++j) bp[k * NR + j] = j < nr ? b[j * B.cs] : 0.0f;
            }
        }
    }
}

// Rows [0, M) x columns [j0, j1) of C for small M, when A's rows and B's
// columns are both contiguous (Y = X · Wᵀ in decoding): plain dot products,
// four columns at a time, without packing. Packing B would copy the whole
// weight matrix on every call to serve a single row of A.
inline void gemv_rows(int M, int K, MatView<float> A, MatView<float> B, float* C, long ldc,
                      bool accumulate, int j0, int j1) {
    auto hsum = [](vf v) {
        alignas(64) float t[W];
        vstore(t, v);
        float s = 0.0f;
        for (int i = 0; i < W; ++i) s += t[i];
        return s;
    };
    for (int i = 0; i < M; ++i) {
        const float* a = A.p + long(i) * A.rs;
        float* c = C + long(i) * ldc;
        int j = j0;
        for (; j + 4 <= j1; j += 4) {
            const float* b0 = B.p + long(j) * B.cs;
            const float* b1 = b0 + B.cs;
            const float* b2 = b1 + B.cs;
            const float* b3 = b2 + B.cs;
            vf s0 = vzero(), s1 = vzero(), s2 = vzero(), s3 = vzero();
            int k = 0;
            for (; k + W <= K; k += W) {
                const vf x = vload(a + k);
                s0 = vfma(x, vload(b0 + k), s0);
                s1 = vfma(x, vload(b1 + k), s1);
                s2 = vfma(x, vload(b2 + k), s2);
                s3 = vfma(x, vload(b3 + k), s3);
            }
            float r[4] = {hsum(s0), hsum(s1), hsum(s2), hsum(s3)};
            for (; k < K; ++k) {
                r[0] += a[k] * b0[k];
                r[1] += a[k] * b1[k];
                r[2] += a[k] * b2[k];
                r[3] += a[k] * b3[k];
            }
            for (int q = 0; q < 4; ++q) c[j + q] = accumulate ? c[j + q] + r[q] : r[q];
        }
        for (; j < j1; ++j) {
            const float* b = B.p + long(j) * B.cs;
            vf s0 = vzero();
            int k = 0;
            for (; k + W <= K; k += W) s0 = vfma(vload(a + k), vload(b + k), s0);
            float r = hsum(s0);
            for (; k < K; ++k) r += a[k] * b[k];
            c[j] = accumulate ? c[j] + r : r;
        }
    }
}

}  // namespace gemm_detail

// C[M, N] = (accumulate ? C : 0) + A[M, K] · B[K, N]. C is row-major with
// row stride ldc and must not overlap A or B. threaded = false runs on the
// calling thread only, for callers already inside a parallel_for.
inline void gemm(int M, int N, int K, MatView<float> A, MatView<float> B, float* C, long ldc,
                 bool accumulate, bool threaded = true) {
    using namespace gemm_detail;
    if (M <= 0 || N <= 0) return;
    if (K <= 0) {
        if (!accumulate)
            for (int i = 0; i < M; ++i) std::fill(C + i * ldc, C + i * ldc + N, 0.0f);
        return;
    }
    auto run = [&](int total, const std::function<void(int, int)>& fn) {
        if (threaded) pool().parallel_for(total, fn, 1);
        else fn(0, total);
    };
    if (M <= 4 && A.cs == 1 && B.rs == 1) {
        run((N + 15) / 16, [&](int q0, int q1) {
            gemv_rows(M, K, A, B, C, ldc, accumulate, q0 * 16, std::min(N, q1 * 16));
        });
        return;
    }
    // The B panel is shared by all threads; it belongs to the calling thread,
    // which stays blocked in parallel_for while the workers read it.
    static thread_local AlignedBuffer<float> b_buf;
    const int m_panels = (M + MR - 1) / MR;
    for (int j0 = 0; j0 < N; j0 += NC) {
        const int nc = std::min(NC, N - j0);
        const int n_panels = (nc + NR - 1) / NR;
        for (int p0 = 0; p0 < K; p0 += KC) {
            const int kc = std::min(KC, K - p0);
            const bool overwrite = !accumulate && p0 == 0;
            float* Bp = b_buf.get(size_t(n_panels) * NR * kc);
            run(n_panels, [&](int q0, int q1) { pack_b(B, p0, kc, j0, nc, q0 * NR, q1 * NR, Bp); });
            run(m_panels, [&](int q0, int q1) {
                static thread_local AlignedBuffer<float> a_buf;
                for (int i0 = q0 * MR; i0 < std::min(M, q1 * MR); i0 += MC) {
                    const int mc = std::min({MC, M - i0, q1 * MR - i0});
                    float* Ap = a_buf.get(size_t(MC) * kc);
                    pack_a(A, i0, mc, p0, kc, Ap);
                    for (int jr = 0; jr < nc; jr += NR)
                        for (int ir = 0; ir < mc; ir += MR)
                            micro_kernel(kc, Ap + long(ir) * kc, Bp + long(jr) * kc,
                                         C + (i0 + ir) * ldc + j0 + jr, ldc,
                                         std::min(MR, mc - ir), std::min(NR, nc - jr), overwrite);
                }
            });
        }
    }
}

// Double precision: the reference the gradient checks run on. Same contract.
inline void gemm(int M, int N, int K, MatView<double> A, MatView<double> B, double* C, long ldc,
                 bool accumulate, bool /*threaded*/ = true) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            double s = 0.0;
            for (int k = 0; k < K; ++k) s += A.at(i, k) * B.at(k, j);
            C[i * ldc + j] = accumulate ? C[i * ldc + j] + s : s;
        }
}
