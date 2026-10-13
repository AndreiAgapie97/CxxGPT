// ternary.h — inference kernels for ternary weights: two-bitplane storage,
// int8 activations, AVX2 / AVX-512 / VNNI, threaded across output rows; and
// a fused ternary MoE layer built on them.
#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "thread_pool.h"

// =============================================================================
// Constants
// =============================================================================
// [[maybe_unused]]: the SIMD widths are only referenced by the ISA paths a
// given -march compiles in.
namespace cfg {
constexpr int kWeightsPerWord = 32;  // bitplane: 1 weight per mask bit
constexpr int kInterleavedPerWord = 16;  // legacy layout: 2 bits per weight
[[maybe_unused]] constexpr int kAvx2Floats   = 8;
[[maybe_unused]] constexpr int kAvx2Bytes    = 32;
[[maybe_unused]] constexpr int kAvx512Floats = 16;
constexpr int kInt8Max       = 127;  // not 128: negating -128 wraps (_mm*_sign_epi8, 0 - x)
constexpr int kMinRowsPerThread = ThreadPool::kDefaultMinPerThread;
}  // namespace cfg


// =============================================================================
// Layout A: interleaved 2-bit codes (kept as the baseline to beat)
// =============================================================================
struct TernaryMatrixInterleaved {
    int rows, cols;
    float scale;
    std::vector<uint32_t> data;

    TernaryMatrixInterleaved(int r, int c, float s = 1.0f)
        : rows(r), cols(c), scale(s),
          data(size_t(r) * (c / cfg::kInterleavedPerWord), 0) {
        assert(c % cfg::kInterleavedPerWord == 0);
    }
};

inline void gemv_interleaved(const float* __restrict X, const TernaryMatrixInterleaved& W,
                      float* __restrict Y) {
    const int K = W.cols;
    const int blocks = K / cfg::kInterleavedPerWord;
    static constexpr float LUT[4] = {0.0f, 1.0f, -1.0f, 0.0f};

    pool().parallel_for(W.rows, [&](int r0, int r1) {
        for (int r = r0; r < r1; ++r) {
            float sum = 0.0f;
            const uint32_t* __restrict row = &W.data[size_t(r) * blocks];
            for (int b = 0; b < blocks; ++b) {
                const uint32_t packed = row[b];
                const float* __restrict xb = X + b * cfg::kInterleavedPerWord;
                for (int i = 0; i < cfg::kInterleavedPerWord; ++i)
                    sum += xb[i] * LUT[(packed >> (i * 2)) & 0x3u];
            }
            Y[r] = sum * W.scale;
        }
    });
}

// =============================================================================
// Layout B: two bitplanes
// =============================================================================
struct TernaryMatrixBitplane {
    int rows, cols, words_per_row;
    float scale;
    std::vector<uint32_t> pos;  // bit set => weight == +1
    std::vector<uint32_t> neg;  // bit set => weight == -1

    TernaryMatrixBitplane(int r, int c, float s = 1.0f)
        : rows(r), cols(c), words_per_row(c / cfg::kWeightsPerWord), scale(s),
          pos(size_t(r) * (c / cfg::kWeightsPerWord), 0),
          neg(size_t(r) * (c / cfg::kWeightsPerWord), 0) {
        assert(c % cfg::kWeightsPerWord == 0 && "cols must be a multiple of 32");
    }

    // A bit set in both planes is undefined. Assert rather than silently
    // treating it as zero, so a packing bug is loud instead of an accuracy leak.
    void validate() const {
        for (size_t i = 0; i < pos.size(); ++i) assert((pos[i] & neg[i]) == 0);
    }

    // Weight at (r, k): +1, -1 or 0. For tests and references, not hot loops.
    int weight(int r, int k) const {
        const size_t i = size_t(r) * words_per_row + k / cfg::kWeightsPerWord;
        const uint32_t bit = 1u << (k % cfg::kWeightsPerWord);
        return (pos[i] & bit) ? 1 : ((neg[i] & bit) ? -1 : 0);
    }
};

// Kernels compute rows [r0, r1) of Y = W x and do no threading themselves:
// callers split rows across the pool, and may mix several matrices in one
// pool job (see TernaryMoELayer::forward).
using RowsFn = void (*)(const float*, const TernaryMatrixBitplane&, float*, int, int);
using GemvFn = void (*)(const float*, const TernaryMatrixBitplane&, float*);  // whole matrix

// Whole-matrix GEMV from a row kernel, rows split across the pool.
template <RowsFn F>
inline void gemv_threaded(const float* __restrict X, const TernaryMatrixBitplane& W, float* __restrict Y) {
    pool().parallel_for(W.rows, [&](int r0, int r1) { F(X, W, Y, r0, r1); });
}

// --- B1: scalar, branchless -------------------------------------------------
inline void gemv_bitplane_scalar(const float* __restrict X, const TernaryMatrixBitplane& W,
                          float* __restrict Y, int r0, int r1) {
    const int WPR = W.words_per_row;
    for (int r = r0; r < r1; ++r) {
        const uint32_t* __restrict p = &W.pos[size_t(r) * WPR];
        const uint32_t* __restrict n = &W.neg[size_t(r) * WPR];
        float acc_p = 0.0f, acc_n = 0.0f;
        for (int w = 0; w < WPR; ++w) {
            const float* __restrict xw = X + w * cfg::kWeightsPerWord;
            const uint32_t bp = p[w], bn = n[w];
            for (int i = 0; i < cfg::kWeightsPerWord; ++i) {
                acc_p += xw[i] * float((bp >> i) & 1u);
                acc_n += xw[i] * float((bn >> i) & 1u);
            }
        }
        Y[r] = (acc_p - acc_n) * W.scale;
    }
}

// --- B2: sparse (ctz over set bits) -----------------------------------------
// Kept for the record: this is SLOWER than B1 at typical densities. The
// branches are unpredictable and cost more than the skipped work. It only
// pays above roughly 85-90% zeros.
inline void gemv_bitplane_sparse(const float* __restrict X, const TernaryMatrixBitplane& W,
                          float* __restrict Y, int r0, int r1) {
    const int WPR = W.words_per_row;
    for (int r = r0; r < r1; ++r) {
        const uint32_t* __restrict p = &W.pos[size_t(r) * WPR];
        const uint32_t* __restrict n = &W.neg[size_t(r) * WPR];
        float acc_p = 0.0f, acc_n = 0.0f;
        for (int w = 0; w < WPR; ++w) {
            const float* __restrict xw = X + w * cfg::kWeightsPerWord;
            uint32_t bp = p[w];
            while (bp) { acc_p += xw[__builtin_ctz(bp)]; bp &= bp - 1; }
            uint32_t bn = n[w];
            while (bn) { acc_n += xw[__builtin_ctz(bn)]; bn &= bn - 1; }
        }
        Y[r] = (acc_p - acc_n) * W.scale;
    }
}

// --- B3: AVX2, fp32 activations ---------------------------------------------
#if defined(__AVX2__)
inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(lo);
    __m128 s = _mm_add_ps(lo, sh);
    sh = _mm_movehl_ps(sh, s);
    return _mm_cvtss_f32(_mm_add_ss(s, sh));
}

// Expand the low 8 bits of b into eight 32-bit all-ones / all-zeros lanes.
inline __m256 expand8(uint32_t b, __m256i shifts) {
    __m256i v = _mm256_set1_epi32(int(b));
    v = _mm256_srlv_epi32(v, shifts);  // bit i -> lane i, position 0
    v = _mm256_slli_epi32(v, 31);      // -> sign bit
    v = _mm256_srai_epi32(v, 31);      // -> 0xFFFFFFFF or 0
    return _mm256_castsi256_ps(v);
}

inline void gemv_bitplane_avx2(const float* __restrict X, const TernaryMatrixBitplane& W,
                        float* __restrict Y, int r0, int r1) {
    const int WPR = W.words_per_row;
    const __m256i shifts = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

    for (int r = r0; r < r1; ++r) {
        const uint32_t* __restrict p = &W.pos[size_t(r) * WPR];
        const uint32_t* __restrict n = &W.neg[size_t(r) * WPR];
        __m256 acc_p = _mm256_setzero_ps(), acc_n = _mm256_setzero_ps();

        for (int w = 0; w < WPR; ++w) {
            const float* __restrict xw = X + w * cfg::kWeightsPerWord;
            const uint32_t bp = p[w], bn = n[w];
            for (int c = 0; c < cfg::kWeightsPerWord / cfg::kAvx2Floats; ++c) {
                __m256 x = _mm256_loadu_ps(xw + c * cfg::kAvx2Floats);
                acc_p = _mm256_add_ps(acc_p, _mm256_and_ps(x, expand8(bp >> (c * 8), shifts)));
                acc_n = _mm256_add_ps(acc_n, _mm256_and_ps(x, expand8(bn >> (c * 8), shifts)));
            }
        }
        Y[r] = (hsum256(acc_p) - hsum256(acc_n)) * W.scale;
    }
}
#endif

// --- B3b: AVX-512, fp32 activations -----------------------------------------
// The mask registers make expand8() unnecessary: a mask word IS the predicate.
#if defined(__AVX512F__)
inline void gemv_bitplane_avx512(const float* __restrict X, const TernaryMatrixBitplane& W,
                          float* __restrict Y, int r0, int r1) {
    const int WPR = W.words_per_row;
    constexpr int kChunks = cfg::kWeightsPerWord / cfg::kAvx512Floats;  // 2

    for (int r = r0; r < r1; ++r) {
        const uint32_t* __restrict p = &W.pos[size_t(r) * WPR];
        const uint32_t* __restrict n = &W.neg[size_t(r) * WPR];
        __m512 acc_p = _mm512_setzero_ps(), acc_n = _mm512_setzero_ps();

        for (int w = 0; w < WPR; ++w) {
            const float* __restrict xw = X + w * cfg::kWeightsPerWord;
            const uint32_t bp = p[w], bn = n[w];
            for (int c = 0; c < kChunks; ++c) {
                __m512 x = _mm512_loadu_ps(xw + c * cfg::kAvx512Floats);
                acc_p = _mm512_mask_add_ps(acc_p, __mmask16(bp >> (c * 16)), acc_p, x);
                acc_n = _mm512_mask_add_ps(acc_n, __mmask16(bn >> (c * 16)), acc_n, x);
            }
        }
        Y[r] = (_mm512_reduce_add_ps(acc_p) - _mm512_reduce_add_ps(acc_n)) * W.scale;
    }
}
#endif

// =============================================================================
// int8 activations
// =============================================================================
// Symmetric per-vector quantization: q = round(x / scale), scale = amax / 127.
// Rounding is half-to-even in both paths (lrintf in the default rounding
// mode, cvtps2dq under the default MXCSR), so SIMD and scalar agree bit for
// bit; the tests check that.
inline float quantize_int8(const float* __restrict X, int K, int8_t* __restrict out) {
    int k = 0;
    float amax = 0.0f;
    // max(v, m) keeps m when v is NaN, matching std::max(amax, v) below.
#if defined(__AVX512F__)
    __m512 m512 = _mm512_setzero_ps();
    for (; k + cfg::kAvx512Floats <= K; k += cfg::kAvx512Floats)
        m512 = _mm512_max_ps(_mm512_abs_ps(_mm512_loadu_ps(X + k)), m512);
    amax = _mm512_reduce_max_ps(m512);
#elif defined(__AVX2__)
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256 m256 = _mm256_setzero_ps();
    for (; k + cfg::kAvx2Floats <= K; k += cfg::kAvx2Floats)
        m256 = _mm256_max_ps(_mm256_and_ps(_mm256_loadu_ps(X + k), abs_mask), m256);
    __m128 m128 = _mm_max_ps(_mm256_castps256_ps128(m256), _mm256_extractf128_ps(m256, 1));
    m128 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
    m128 = _mm_max_ss(m128, _mm_shuffle_ps(m128, m128, 1));
    amax = _mm_cvtss_f32(m128);
#endif
    for (; k < K; ++k) amax = std::max(amax, std::fabs(X[k]));
    if (amax == 0.0f) { std::memset(out, 0, size_t(K)); return 0.0f; }

    const float scale = amax / float(cfg::kInt8Max);
    const float inv = 1.0f / scale;
    k = 0;
#if defined(__AVX512F__)
    const __m512 vinv = _mm512_set1_ps(inv);
    const __m512i lo512 = _mm512_set1_epi32(-cfg::kInt8Max), hi512 = _mm512_set1_epi32(cfg::kInt8Max);
    for (; k + cfg::kAvx512Floats <= K; k += cfg::kAvx512Floats) {
        __m512i q = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(X + k), vinv));
        q = _mm512_min_epi32(_mm512_max_epi32(q, lo512), hi512);
        _mm_storeu_si128((__m128i*)(out + k), _mm512_cvtepi32_epi8(q));
    }
#elif defined(__AVX2__)
    const __m256 vinv = _mm256_set1_ps(inv);
    const __m256i lo256 = _mm256_set1_epi32(-cfg::kInt8Max), hi256 = _mm256_set1_epi32(cfg::kInt8Max);
    for (; k + cfg::kAvx2Floats <= K; k += cfg::kAvx2Floats) {
        __m256i q = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(X + k), vinv));
        q = _mm256_min_epi32(_mm256_max_epi32(q, lo256), hi256);
        const __m128i w = _mm_packs_epi32(_mm256_castsi256_si128(q), _mm256_extracti128_si256(q, 1));
        _mm_storel_epi64((__m128i*)(out + k), _mm_packs_epi16(w, w));
    }
#endif
    for (; k < K; ++k) {
        const int v = int(std::lrintf(X[k] * inv));
        out[k] = int8_t(std::min(cfg::kInt8Max, std::max(-cfg::kInt8Max, v)));
    }
    return scale;
}

// int8 kernels take activations already quantized by quantize_int8(), so an
// input that feeds several matrices (gate and up, every routed expert) is
// quantized once rather than once per matrix.
using RowsQFn = void (*)(const int8_t*, float, const TernaryMatrixBitplane&, float*, int, int);

inline void gemv_bitplane_int8_scalar(const int8_t* __restrict xq, float x_scale,
                               const TernaryMatrixBitplane& W, float* __restrict Y,
                               int r0, int r1) {
    const int WPR = W.words_per_row;
    const float out_scale = x_scale * W.scale;

    for (int r = r0; r < r1; ++r) {
        const uint32_t* __restrict p = &W.pos[size_t(r) * WPR];
        const uint32_t* __restrict n = &W.neg[size_t(r) * WPR];
        int32_t acc = 0;
        for (int w = 0; w < WPR; ++w) {
            const int8_t* __restrict xw = xq + w * cfg::kWeightsPerWord;
            const uint32_t bp = p[w], bn = n[w];
            for (int i = 0; i < cfg::kWeightsPerWord; ++i)
                acc += int32_t(xw[i]) *
                       (int32_t((bp >> i) & 1u) - int32_t((bn >> i) & 1u));
        }
        Y[r] = float(acc) * out_scale;
    }
}

#if defined(__AVX2__)
// Expand a 32-bit mask into 32 bytes of 0x00 / 0xFF, LSB -> byte 0.
inline __m256i expand32(uint32_t m) {
    const __m256i shuf = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                          2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i bit = _mm256_set1_epi64x(int64_t(0x8040201008040201ULL));
    __m256i v = _mm256_shuffle_epi8(_mm256_set1_epi32(int(m)), shuf);
    return _mm256_cmpeq_epi8(_mm256_and_si256(v, bit), bit);
}

inline int hsum_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, _MM_SHUFFLE(1, 0, 3, 2)));
    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(lo);
}

// One mask word applied to its 32 activations: x, -x, or 0 per byte.
inline __m256i apply_ternary32(uint32_t bp, uint32_t bn, const int8_t* xw) {
    const __m256i one8 = _mm256_set1_epi8(1);
    const __m256i x = _mm256_loadu_si256((const __m256i*)xw);
    const __m256i s = _mm256_sub_epi8(_mm256_and_si256(expand32(bp), one8),
                                      _mm256_and_si256(expand32(bn), one8));
    return _mm256_sign_epi8(x, s);
}

inline void gemv_bitplane_int8_avx2(const int8_t* __restrict xq, float x_scale,
                             const TernaryMatrixBitplane& W, float* __restrict Y,
                             int r0, int r1) {
    const int WPR = W.words_per_row;
    const float out_scale = x_scale * W.scale;

    const __m256i one8 = _mm256_set1_epi8(1);
    const __m256i one16 = _mm256_set1_epi16(1);

    for (int r = r0; r < r1; ++r) {
        const uint32_t* __restrict p = &W.pos[size_t(r) * WPR];
        const uint32_t* __restrict n = &W.neg[size_t(r) * WPR];
        __m256i acc = _mm256_setzero_si256();

        for (int w = 0; w < WPR; ++w) {
            // one 32-bit mask word == one 32-byte vector of activations
            __m256i v = apply_ternary32(p[w], n[w], xq + w * cfg::kAvx2Bytes);
            __m256i s16 = _mm256_maddubs_epi16(one8, v);   // unsigned 1 * signed v
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(s16, one16));
        }
        Y[r] = float(hsum_epi32(acc)) * out_scale;
    }
}
#endif

// --- VNNI -------------------------------------------------------------------
// vpdpbusd does maddubs + madd + add in one instruction. Unlike the add it
// replaces, it sits on the accumulator's dependency chain with several
// cycles of latency, so these kernels split the sum across independent
// accumulators: 2 for 256-bit, 4 for 512-bit. A single accumulator gives
// up roughly half of the 512-bit kernel's gain over B5b.
#if defined(__AVX2__) && \
    (defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__)))
#define TERNARY_HAVE_VNNI256 1

inline __m256i dpbusd256(__m256i acc, __m256i u8, __m256i s8) {
#if defined(__AVXVNNI__)
    return _mm256_dpbusd_avx_epi32(acc, u8, s8);  // VEX: AVX-VNNI without AVX-512
#else
    return _mm256_dpbusd_epi32(acc, u8, s8);      // EVEX: AVX512-VNNI + VL
#endif
}

inline void gemv_bitplane_int8_vnni256(const int8_t* __restrict xq, float x_scale,
                                const TernaryMatrixBitplane& W, float* __restrict Y,
                                int r0, int r1) {
    const int WPR = W.words_per_row;
    const float out_scale = x_scale * W.scale;
    const __m256i one8 = _mm256_set1_epi8(1);
    constexpr int B = cfg::kAvx2Bytes;

    for (int r = r0; r < r1; ++r) {
        const uint32_t* __restrict p = &W.pos[size_t(r) * WPR];
        const uint32_t* __restrict n = &W.neg[size_t(r) * WPR];
        __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256();

        int w = 0;
        for (; w + 2 <= WPR; w += 2) {
            acc0 = dpbusd256(acc0, one8, apply_ternary32(p[w], n[w], xq + w * B));
            acc1 = dpbusd256(acc1, one8, apply_ternary32(p[w + 1], n[w + 1], xq + (w + 1) * B));
        }
        if (w < WPR) acc0 = dpbusd256(acc0, one8, apply_ternary32(p[w], n[w], xq + w * B));
        Y[r] = float(hsum_epi32(_mm256_add_epi32(acc0, acc1))) * out_scale;
    }
}
#endif

#if defined(__AVX512BW__)
// Two mask words applied to their 64 activations: x, -x, or 0 per byte.
// With pair == false only p[0] / n[0] are read and the upper 32 activation
// bytes are neither loaded nor counted.
inline __m512i apply_ternary64(const uint32_t* p, const uint32_t* n,
                                      const int8_t* xw, bool pair) {
    uint64_t kp = p[0], kn = n[0];
    if (pair) {
        kp |= uint64_t(p[1]) << 32;
        kn |= uint64_t(n[1]) << 32;
    }
    const __mmask64 load_mask = pair ? ~0ULL : 0xFFFFFFFFULL;
    const __m512i x = _mm512_maskz_loadu_epi8(load_mask, xw);
    const __m512i v = _mm512_maskz_mov_epi8(__mmask64(kp), x);
    return _mm512_mask_sub_epi8(v, __mmask64(kn), _mm512_setzero_si512(), x);
}

// 64 int8 lanes per iteration => two mask words at a time.
inline void gemv_bitplane_int8_avx512(const int8_t* __restrict xq, float x_scale,
                               const TernaryMatrixBitplane& W, float* __restrict Y,
                               int r0, int r1) {
    const int WPR = W.words_per_row;
    const float out_scale = x_scale * W.scale;

    const __m512i one8 = _mm512_set1_epi8(1);
    const __m512i one16 = _mm512_set1_epi16(1);

    for (int r = r0; r < r1; ++r) {
        const uint32_t* __restrict p = &W.pos[size_t(r) * WPR];
        const uint32_t* __restrict n = &W.neg[size_t(r) * WPR];
        __m512i acc = _mm512_setzero_si512();

        for (int w = 0; w < WPR; w += 2) {
            __m512i v = apply_ternary64(p + w, n + w, xq + w * cfg::kWeightsPerWord,
                                        w + 1 < WPR);
            __m512i s16 = _mm512_maddubs_epi16(one8, v);
            acc = _mm512_add_epi32(acc, _mm512_madd_epi16(s16, one16));
        }
        Y[r] = float(_mm512_reduce_add_epi32(acc)) * out_scale;
    }
}

#if defined(__AVX512VNNI__)
inline void gemv_bitplane_int8_vnni512(const int8_t* __restrict xq, float x_scale,
                                const TernaryMatrixBitplane& W, float* __restrict Y,
                                int r0, int r1) {
    const int WPR = W.words_per_row;
    const float out_scale = x_scale * W.scale;
    const __m512i one8 = _mm512_set1_epi8(1);
    constexpr int B = cfg::kWeightsPerWord;
    constexpr int kAcc = 4;  // each covers one word pair => 8 words per step

    for (int r = r0; r < r1; ++r) {
        const uint32_t* __restrict p = &W.pos[size_t(r) * WPR];
        const uint32_t* __restrict n = &W.neg[size_t(r) * WPR];
        __m512i acc[kAcc];
        for (auto& a : acc) a = _mm512_setzero_si512();

        int w = 0;
        for (; w + 2 * kAcc <= WPR; w += 2 * kAcc)
            for (int a = 0; a < kAcc; ++a) {
                const int wa = w + 2 * a;
                acc[a] = _mm512_dpbusd_epi32(acc[a], one8,
                                             apply_ternary64(p + wa, n + wa, xq + wa * B, true));
            }
        for (; w < WPR; w += 2)
            acc[0] = _mm512_dpbusd_epi32(acc[0], one8,
                                         apply_ternary64(p + w, n + w, xq + w * B, w + 1 < WPR));
        for (int a = 1; a < kAcc; ++a) acc[0] = _mm512_add_epi32(acc[0], acc[a]);
        Y[r] = float(_mm512_reduce_add_epi32(acc[0])) * out_scale;
    }
}
#endif
#endif

// Whole-matrix GEMV with float input for an int8 row kernel: quantize, then
// split rows across the pool. The scratch is per calling thread, so
// concurrent callers never share it.
template <RowsQFn F>
inline void gemv_quantize_then(const float* __restrict X, const TernaryMatrixBitplane& W,
                        float* __restrict Y) {
    thread_local std::vector<int8_t> xq;
    xq.resize(size_t(W.cols));
    const float x_scale = quantize_int8(X, W.cols, xq.data());
    const int8_t* q = xq.data();
    pool().parallel_for(W.rows, [&](int r0, int r1) { F(q, x_scale, W, Y, r0, r1); });
}

// =============================================================================
// fp32 helpers: dot product, SiLU
// =============================================================================
// Without -ffast-math the compiler may not reorder a float sum, so a plain
// dot-product loop runs as one scalar dependency chain. These split it.
#if defined(__AVX2__) && !defined(__AVX512F__)
inline __m256 madd256(__m256 a, __m256 b, __m256 c) {
#if defined(__FMA__)
    return _mm256_fmadd_ps(a, b, c);
#else
    return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#endif
}
#endif

inline float dot_f32(const float* __restrict a, const float* __restrict b, int n) {
    int i = 0;
    float s = 0.0f;
#if defined(__AVX512F__)
    constexpr int V = cfg::kAvx512Floats;
    __m512 acc[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                     _mm512_setzero_ps()};
    for (; i + 4 * V <= n; i += 4 * V)
        for (int j = 0; j < 4; ++j)
            acc[j] = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + j * V), _mm512_loadu_ps(b + i + j * V), acc[j]);
    for (; i + V <= n; i += V)
        acc[0] = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc[0]);
    s = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(acc[0], acc[1]), _mm512_add_ps(acc[2], acc[3])));
#elif defined(__AVX2__)
    constexpr int V = cfg::kAvx2Floats;
    __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                     _mm256_setzero_ps()};
    for (; i + 4 * V <= n; i += 4 * V)
        for (int j = 0; j < 4; ++j)
            acc[j] = madd256(_mm256_loadu_ps(a + i + j * V), _mm256_loadu_ps(b + i + j * V), acc[j]);
    for (; i + V <= n; i += V)
        acc[0] = madd256(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc[0]);
    s = hsum256(_mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3])));
#endif
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

// exp(x) per lane, Cephes expf: x = n ln2 + r with |r| <= ln2/2, exp(r) from a
// degree-6 polynomial, scaled by 2^n. About 1 ulp. Inputs are clamped to
// [-87, 88.3], where 2^n stays a normal float; SiLU only needs exp(-g) to
// saturate, not to reach 0 or inf.
#if defined(__AVX2__)
namespace expf_c {
constexpr float kLo = -87.0f, kHi = 88.3f, kLog2e = 1.44269504088896341f;
constexpr float kC1 = 0.693359375f, kC2 = -2.12194440e-4f;  // ln2 = kC1 + kC2, kC1 exact in 9 bits
constexpr float kP[6] = {1.9875691500e-4f, 1.3981999507e-3f, 8.3334519073e-3f,
                         4.1665795894e-2f, 1.6666665459e-1f, 5.0000001201e-1f};
}  // namespace expf_c
#endif

#if defined(__AVX512F__)
inline __m512 exp512(__m512 x) {
    using namespace expf_c;
    x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(kLo)), _mm512_set1_ps(kHi));
    const __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(kLog2e)),
                                          _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 r = _mm512_fnmadd_ps(n, _mm512_set1_ps(kC1), x);
    r = _mm512_fnmadd_ps(n, _mm512_set1_ps(kC2), r);
    __m512 p = _mm512_set1_ps(kP[0]);
    for (int i = 1; i < 6; ++i) p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(kP[i]));
    p = _mm512_fmadd_ps(p, _mm512_mul_ps(r, r), _mm512_add_ps(r, _mm512_set1_ps(1.0f)));
    return _mm512_scalef_ps(p, n);
}
#elif defined(__AVX2__)
inline __m256 exp256(__m256 x) {
    using namespace expf_c;
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(kLo)), _mm256_set1_ps(kHi));
    const __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(kLog2e)),
                                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_sub_ps(x, _mm256_mul_ps(n, _mm256_set1_ps(kC1)));  // exact: kC1 has 9 bits
    r = _mm256_sub_ps(r, _mm256_mul_ps(n, _mm256_set1_ps(kC2)));
    __m256 p = _mm256_set1_ps(kP[0]);
    for (int i = 1; i < 6; ++i) p = madd256(p, r, _mm256_set1_ps(kP[i]));
    p = madd256(p, _mm256_mul_ps(r, r), _mm256_add_ps(r, _mm256_set1_ps(1.0f)));
    const __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}
#endif

// out[i] = SiLU(g[i]) * u[i] = g[i] * u[i] / (1 + exp(-g[i])).
inline void silu_mul(const float* __restrict g, const float* __restrict u,
                     float* __restrict out, int n) {
    int i = 0;
#if defined(__AVX512F__)
    const __m512 one = _mm512_set1_ps(1.0f);
    for (; i + cfg::kAvx512Floats <= n; i += cfg::kAvx512Floats) {
        const __m512 x = _mm512_loadu_ps(g + i);
        const __m512 e = exp512(_mm512_sub_ps(_mm512_setzero_ps(), x));
        _mm512_storeu_ps(out + i, _mm512_div_ps(_mm512_mul_ps(x, _mm512_loadu_ps(u + i)),
                                                _mm512_add_ps(one, e)));
    }
#elif defined(__AVX2__)
    const __m256 one = _mm256_set1_ps(1.0f);
    for (; i + cfg::kAvx2Floats <= n; i += cfg::kAvx2Floats) {
        const __m256 x = _mm256_loadu_ps(g + i);
        const __m256 e = exp256(_mm256_sub_ps(_mm256_setzero_ps(), x));
        _mm256_storeu_ps(out + i, _mm256_div_ps(_mm256_mul_ps(x, _mm256_loadu_ps(u + i)),
                                                _mm256_add_ps(one, e)));
    }
#endif
    for (; i < n; ++i) out[i] = (g[i] / (1.0f + std::exp(-g[i]))) * u[i];
}

// --- fp32 dense reference ---------------------------------------------------
// The baseline the ternary kernels are compared against, so it gets the same
// SIMD treatment rather than a scalar loop.
inline void gemv_fp32(const float* __restrict X, const float* __restrict W, int M, int K,
               float* __restrict Y) {
    pool().parallel_for(M, [&](int r0, int r1) {
        for (int r = r0; r < r1; ++r) Y[r] = dot_f32(W + size_t(r) * K, X, K);
    });
}

// =============================================================================
// Packing — both layouts filled from the same source weights
// =============================================================================
inline void pack_interleaved(const std::vector<int8_t>& src, TernaryMatrixInterleaved& W) {
    const int K = W.cols, blocks = K / cfg::kInterleavedPerWord;
    std::fill(W.data.begin(), W.data.end(), 0u);
    for (int r = 0; r < W.rows; ++r)
        for (int k = 0; k < K; ++k) {
            const int8_t v = src[size_t(r) * K + k];
            const uint32_t code = (v > 0) ? 1u : (v < 0 ? 2u : 0u);
            W.data[size_t(r) * blocks + k / cfg::kInterleavedPerWord] |=
                code << ((k % cfg::kInterleavedPerWord) * 2);
        }
}

inline void pack_bitplane(const std::vector<int8_t>& src, TernaryMatrixBitplane& W) {
    const int K = W.cols, WPR = W.words_per_row;
    std::fill(W.pos.begin(), W.pos.end(), 0u);
    std::fill(W.neg.begin(), W.neg.end(), 0u);
    for (int r = 0; r < W.rows; ++r)
        for (int k = 0; k < K; ++k) {
            const int8_t v = src[size_t(r) * K + k];
            const size_t idx = size_t(r) * WPR + k / cfg::kWeightsPerWord;
            const uint32_t bit = 1u << (k % cfg::kWeightsPerWord);
            if (v > 0) W.pos[idx] |= bit;
            else if (v < 0) W.neg[idx] |= bit;
        }
    W.validate();
}

inline std::vector<int8_t> random_ternary(int M, int K, double zero_frac, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<int8_t> w(size_t(M) * K);
    for (auto& v : w) {
        const double t = u(rng);
        v = (t < zero_frac) ? 0 : (t < zero_frac + (1.0 - zero_frac) / 2 ? 1 : -1);
    }
    return w;
}

// =============================================================================
// Kernel registry
// =============================================================================
struct Kernel {
    const char* name;
    GemvFn fn;      // whole matrix, float activations; int8 kernels quantize first
    RowsFn rows;    // row range, float activations (fp32 kernels only)
    RowsQFn qrows;  // row range, pre-quantized activations (int8 kernels only)
};

inline std::vector<Kernel> all_kernels() {
    return {
        {"B1: bitplane scalar", gemv_threaded<gemv_bitplane_scalar>,
         gemv_bitplane_scalar, nullptr},
        {"B2: bitplane sparse", gemv_threaded<gemv_bitplane_sparse>,
         gemv_bitplane_sparse, nullptr},
#if defined(__AVX2__)
        {"B3: bitplane AVX2", gemv_threaded<gemv_bitplane_avx2>,
         gemv_bitplane_avx2, nullptr},
#endif
#if defined(__AVX512F__)
        {"B3b: bitplane AVX-512", gemv_threaded<gemv_bitplane_avx512>,
         gemv_bitplane_avx512, nullptr},
#endif
        {"B4: int8 scalar", gemv_quantize_then<gemv_bitplane_int8_scalar>,
         nullptr, gemv_bitplane_int8_scalar},
#if defined(__AVX2__)
        {"B5: int8 AVX2", gemv_quantize_then<gemv_bitplane_int8_avx2>,
         nullptr, gemv_bitplane_int8_avx2},
#endif
#if defined(__AVX512BW__)
        {"B5b: int8 AVX-512", gemv_quantize_then<gemv_bitplane_int8_avx512>,
         nullptr, gemv_bitplane_int8_avx512},
#endif
#if defined(TERNARY_HAVE_VNNI256)
        {"B6: int8 VNNI 256-bit", gemv_quantize_then<gemv_bitplane_int8_vnni256>,
         nullptr, gemv_bitplane_int8_vnni256},
#endif
#if defined(__AVX512BW__) && defined(__AVX512VNNI__)
        {"B6b: int8 VNNI 512-bit", gemv_quantize_then<gemv_bitplane_int8_vnni512>,
         nullptr, gemv_bitplane_int8_vnni512},
#endif
    };
}

// The int8 kernel the MoE layer uses by default: the widest this build has.
inline Kernel default_int8_kernel() {
#if defined(__AVX512BW__) && defined(__AVX512VNNI__)
    const RowsQFn want = gemv_bitplane_int8_vnni512;
#elif defined(__AVX512BW__)
    const RowsQFn want = gemv_bitplane_int8_avx512;
#elif defined(TERNARY_HAVE_VNNI256)
    const RowsQFn want = gemv_bitplane_int8_vnni256;
#elif defined(__AVX2__)
    const RowsQFn want = gemv_bitplane_int8_avx2;
#else
    const RowsQFn want = gemv_bitplane_int8_scalar;
#endif
    for (const auto& k : all_kernels())
        if (k.qrows == want) return k;
    assert(false && "default int8 kernel missing from all_kernels()");
    return all_kernels().front();
}

// =============================================================================
// Expert / MoE
// =============================================================================
struct TernaryExpert {
    int d_model, hidden_dim;
    TernaryMatrixBitplane W_gate, W_up, W_down;
    std::vector<float> gate_out, up_out, inter, down_out;
    std::vector<int8_t> inter_q;
    float inter_scale = 0.0f;

    TernaryExpert(int dim, int hidden, float s)
        : d_model(dim), hidden_dim(hidden),
          W_gate(hidden, dim, s), W_up(hidden, dim, s), W_down(dim, hidden, s),
          gate_out(hidden), up_out(hidden), inter(hidden), down_out(dim), inter_q(hidden) {}

    // Hidden rows [lo, hi): gate and up projections, then SiLU(gate) * up.
    // For an int8 kernel, xq / x_scale hold x already quantized.
    void gate_up(const float* x, const int8_t* xq, float x_scale, const Kernel& k,
                 int lo, int hi) {
        if (k.qrows) {
            k.qrows(xq, x_scale, W_gate, gate_out.data(), lo, hi);
            k.qrows(xq, x_scale, W_up, up_out.data(), lo, hi);
        } else {
            k.rows(x, W_gate, gate_out.data(), lo, hi);
            k.rows(x, W_up, up_out.data(), lo, hi);
        }
        silu_mul(gate_out.data() + lo, up_out.data() + lo, inter.data() + lo, hi - lo);
    }

    // The int8 scale needs the whole hidden vector, so this runs between the
    // gate/up job and the down job.
    void quantize_hidden() {
        inter_scale = quantize_int8(inter.data(), hidden_dim, inter_q.data());
    }

    // Output rows [lo, hi) of the down projection, into down_out.
    void down(const Kernel& k, int lo, int hi) {
        if (k.qrows) k.qrows(inter_q.data(), inter_scale, W_down, down_out.data(), lo, hi);
        else k.rows(inter.data(), W_down, down_out.data(), lo, hi);
    }
};

class TernaryMoELayer {
public:
    int d_model, hidden_dim, num_experts, top_k;
    std::vector<float> W_router;
    std::vector<TernaryExpert> experts;
    Kernel kernel;
    std::vector<float> logits, weights;
    std::vector<int> idx;
    std::vector<int8_t> xq;

    TernaryMoELayer(int dim, int num_exp, int k, const Kernel& kern)
        : d_model(dim), hidden_dim(dim * 4), num_experts(num_exp), top_k(k),
          W_router(size_t(num_exp) * dim, 0.0f), kernel(kern),
          logits(num_exp), weights(k), idx(num_exp), xq(dim) {
        assert(dim % cfg::kWeightsPerWord == 0);
        assert(k > 0 && k <= num_exp);
        experts.reserve(num_exp);
        for (int i = 0; i < num_exp; ++i) experts.emplace_back(dim, hidden_dim, 0.05f);
    }

    // Random router and experts. Router weights are N(0, 1/d_model), so a
    // unit-variance token gets logits of unit variance and a real top-k pick.
    void randomize(double zero_frac, uint32_t seed) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> nd(0.0f, 1.0f / std::sqrt(float(d_model)));
        for (auto& w : W_router) w = nd(rng);
        for (auto& e : experts) {
            pack_bitplane(random_ternary(e.hidden_dim, e.d_model, zero_frac, ++seed), e.W_gate);
            pack_bitplane(random_ternary(e.hidden_dim, e.d_model, zero_frac, ++seed), e.W_up);
            pack_bitplane(random_ternary(e.d_model, e.hidden_dim, zero_frac, ++seed), e.W_down);
        }
    }

    // One token. After routing, the work is two pool jobs rather than three
    // GEMVs per expert: gate + up + SiLU for every routed expert at once
    // (top_k * hidden_dim rows), then every routed expert's down projection
    // and the weighted sum (d_model rows). Not reentrant: uses member scratch.
    void forward(const float* x, float* out) {
        for (int e = 0; e < num_experts; ++e)
            logits[e] = dot_f32(&W_router[size_t(e) * d_model], x, d_model);

        std::iota(idx.begin(), idx.end(), 0);
        std::nth_element(idx.begin(), idx.begin() + (top_k - 1), idx.end(),
                         [&](int a, int b) { return logits[a] > logits[b]; });
        std::sort(idx.begin(), idx.begin() + top_k,
                  [&](int a, int b) { return logits[a] > logits[b]; });

        const float mx = logits[idx[0]];  // max subtraction: exp() overflows without it
        float sum_exp = 0.0f;
        for (int i = 0; i < top_k; ++i) {
            weights[i] = std::exp(logits[idx[i]] - mx);
            sum_exp += weights[i];
        }
        for (int i = 0; i < top_k; ++i) weights[i] /= sum_exp;

        const float x_scale = kernel.qrows ? quantize_int8(x, d_model, xq.data()) : 0.0f;

        const int H = hidden_dim;
        pool().parallel_for(top_k * H, [&](int r0, int r1) {
            while (r0 < r1) {  // a chunk may straddle two experts
                const int i = r0 / H, lo = r0 % H, hi = std::min(H, lo + (r1 - r0));
                experts[size_t(idx[i])].gate_up(x, xq.data(), x_scale, kernel, lo, hi);
                r0 += hi - lo;
            }
        });

        if (kernel.qrows)
            for (int i = 0; i < top_k; ++i) experts[size_t(idx[i])].quantize_hidden();

        pool().parallel_for(d_model, [&](int r0, int r1) {
            for (int i = 0; i < top_k; ++i) experts[size_t(idx[i])].down(kernel, r0, r1);
            for (int d = r0; d < r1; ++d) {
                float s = 0.0f;
                for (int i = 0; i < top_k; ++i) s += weights[i] * experts[size_t(idx[i])].down_out[d];
                out[d] = s;
            }
        });
    }
};
