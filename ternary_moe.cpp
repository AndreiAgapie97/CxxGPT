// ternary_moe.cpp — v2
//
// Ternary MoE inference kernels: two-bitplane weights, int8 activations,
// AVX2 / AVX-512, threaded across output rows.
//
// Build:
//   g++ -O3 -march=native -std=c++17 -pthread ternary_moe.cpp -o ternary_moe
//
// Run:
//   ./ternary_moe          # tests + benchmark
//   ./ternary_moe -t N     # force N threads

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

// =============================================================================
// Constants
// =============================================================================
namespace cfg {
constexpr int kBitsPerWord   = 32;  // bits in the mask word type
constexpr int kWeightsPerWord = 32;  // bitplane: 1 weight per mask bit
constexpr int kInterleavedPerWord = 16;  // legacy layout: 2 bits per weight
constexpr int kAvx2Floats    = 8;
constexpr int kAvx2Bytes     = 32;
constexpr int kAvx512Floats  = 16;
constexpr int kAvx512Bytes   = 64;
constexpr int kInt8Max       = 127;  // not 128: _mm*_sign_epi8 saturates on -128
constexpr int kMinRowsPerThread = 16;
}  // namespace cfg

// =============================================================================
// Thread pool — persistent workers, parallel_for over output rows
// =============================================================================
class ThreadPool {
public:
    explicit ThreadPool(int n) : nthreads_(std::max(1, n)) {
        for (int i = 1; i < nthreads_; ++i)
            workers_.emplace_back([this, i] { worker(i); });
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
            ++generation_;
        }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }

    int size() const { return nthreads_; }

    // fn(row_begin, row_end) is called once per thread with a disjoint range.
    void parallel_for(int total, const std::function<void(int, int)>& fn) {
        if (nthreads_ == 1 || total < nthreads_ * cfg::kMinRowsPerThread) {
            fn(0, total);
            return;
        }
        {
            std::lock_guard<std::mutex> lk(m_);
            job_ = &fn;
            total_ = total;
            pending_ = nthreads_ - 1;
            ++generation_;
        }
        cv_.notify_all();
        run_chunk(0);  // this thread takes chunk 0
        std::unique_lock<std::mutex> lk(m_);
        done_.wait(lk, [this] { return pending_ == 0; });
        job_ = nullptr;
    }

private:
    void run_chunk(int idx) {
        const int per = (total_ + nthreads_ - 1) / nthreads_;
        const int r0 = std::min(idx * per, total_);
        const int r1 = std::min(r0 + per, total_);
        if (r0 < r1) (*job_)(r0, r1);
    }

    void worker(int idx) {
        uint64_t seen = 0;
        for (;;) {
            std::unique_lock<std::mutex> lk(m_);
            cv_.wait(lk, [this, &seen] { return generation_ != seen; });
            seen = generation_;
            if (stop_) return;
            lk.unlock();

            run_chunk(idx);

            lk.lock();
            if (--pending_ == 0) done_.notify_one();
        }
    }

    const int nthreads_;
    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cv_, done_;
    const std::function<void(int, int)>* job_ = nullptr;
    int total_ = 0, pending_ = 0;
    uint64_t generation_ = 0;
    bool stop_ = false;
};

static ThreadPool* g_pool = nullptr;
static inline ThreadPool& pool() { return *g_pool; }

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

void gemv_interleaved(const float* __restrict X, const TernaryMatrixInterleaved& W,
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
};

// --- B1: scalar, branchless -------------------------------------------------
void gemv_bitplane_scalar(const float* __restrict X, const TernaryMatrixBitplane& W,
                          float* __restrict Y) {
    const int WPR = W.words_per_row;
    pool().parallel_for(W.rows, [&](int r0, int r1) {
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
    });
}

// --- B2: sparse (ctz over set bits) -----------------------------------------
// Kept for the record: this is SLOWER than B1 at typical densities. The
// branches are unpredictable and cost more than the skipped work. It only
// pays above roughly 85-90% zeros.
void gemv_bitplane_sparse(const float* __restrict X, const TernaryMatrixBitplane& W,
                          float* __restrict Y) {
    const int WPR = W.words_per_row;
    pool().parallel_for(W.rows, [&](int r0, int r1) {
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
    });
}

// --- B3: AVX2, fp32 activations ---------------------------------------------
#if defined(__AVX2__)
static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(lo);
    __m128 s = _mm_add_ps(lo, sh);
    sh = _mm_movehl_ps(sh, s);
    return _mm_cvtss_f32(_mm_add_ss(s, sh));
}

// Expand the low 8 bits of b into eight 32-bit all-ones / all-zeros lanes.
static inline __m256 expand8(uint32_t b, __m256i shifts) {
    __m256i v = _mm256_set1_epi32(int(b));
    v = _mm256_srlv_epi32(v, shifts);  // bit i -> lane i, position 0
    v = _mm256_slli_epi32(v, 31);      // -> sign bit
    v = _mm256_srai_epi32(v, 31);      // -> 0xFFFFFFFF or 0
    return _mm256_castsi256_ps(v);
}

void gemv_bitplane_avx2(const float* __restrict X, const TernaryMatrixBitplane& W,
                        float* __restrict Y) {
    const int WPR = W.words_per_row;
    const __m256i shifts = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

    pool().parallel_for(W.rows, [&](int r0, int r1) {
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
    });
}
#endif

// --- B3b: AVX-512, fp32 activations -----------------------------------------
// The mask registers make expand8() unnecessary: a mask word IS the predicate.
#if defined(__AVX512F__)
void gemv_bitplane_avx512(const float* __restrict X, const TernaryMatrixBitplane& W,
                          float* __restrict Y) {
    const int WPR = W.words_per_row;
    constexpr int kChunks = cfg::kWeightsPerWord / cfg::kAvx512Floats;  // 2

    pool().parallel_for(W.rows, [&](int r0, int r1) {
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
    });
}
#endif

// =============================================================================
// int8 activations
// =============================================================================
static float quantize_int8(const float* __restrict X, int K, int8_t* __restrict out) {
    float amax = 0.0f;
    for (int k = 0; k < K; ++k) amax = std::max(amax, std::fabs(X[k]));
    if (amax == 0.0f) { std::memset(out, 0, size_t(K)); return 0.0f; }
    const float scale = amax / float(cfg::kInt8Max);
    const float inv = 1.0f / scale;
    for (int k = 0; k < K; ++k) {
        const int v = int(std::lrintf(X[k] * inv));
        out[k] = int8_t(std::min(cfg::kInt8Max, std::max(-cfg::kInt8Max, v)));
    }
    return scale;
}

// Shared scratch for the quantized activations (one row vector, reused).
static std::vector<int8_t> g_xq;

void gemv_bitplane_int8_scalar(const float* __restrict X, const TernaryMatrixBitplane& W,
                               float* __restrict Y) {
    const int K = W.cols, WPR = W.words_per_row;
    g_xq.resize(size_t(K));
    const float out_scale = quantize_int8(X, K, g_xq.data()) * W.scale;
    const int8_t* __restrict xq = g_xq.data();

    pool().parallel_for(W.rows, [&](int r0, int r1) {
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
    });
}

#if defined(__AVX2__)
// Expand a 32-bit mask into 32 bytes of 0x00 / 0xFF, LSB -> byte 0.
static inline __m256i expand32(uint32_t m) {
    const __m256i shuf = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                          2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i bit = _mm256_set1_epi64x(int64_t(0x8040201008040201ULL));
    __m256i v = _mm256_shuffle_epi8(_mm256_set1_epi32(int(m)), shuf);
    return _mm256_cmpeq_epi8(_mm256_and_si256(v, bit), bit);
}

static inline int hsum_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, _MM_SHUFFLE(1, 0, 3, 2)));
    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(lo);
}

void gemv_bitplane_int8_avx2(const float* __restrict X, const TernaryMatrixBitplane& W,
                             float* __restrict Y) {
    const int K = W.cols, WPR = W.words_per_row;
    g_xq.resize(size_t(K));
    const float out_scale = quantize_int8(X, K, g_xq.data()) * W.scale;
    const int8_t* __restrict xq = g_xq.data();

    const __m256i one8 = _mm256_set1_epi8(1);
    const __m256i one16 = _mm256_set1_epi16(1);

    pool().parallel_for(W.rows, [&](int r0, int r1) {
        for (int r = r0; r < r1; ++r) {
            const uint32_t* __restrict p = &W.pos[size_t(r) * WPR];
            const uint32_t* __restrict n = &W.neg[size_t(r) * WPR];
            __m256i acc = _mm256_setzero_si256();

            for (int w = 0; w < WPR; ++w) {
                // one 32-bit mask word == one 32-byte vector of activations
                __m256i x = _mm256_loadu_si256((const __m256i*)(xq + w * cfg::kAvx2Bytes));
                __m256i s = _mm256_sub_epi8(_mm256_and_si256(expand32(p[w]), one8),
                                            _mm256_and_si256(expand32(n[w]), one8));
                __m256i v = _mm256_sign_epi8(x, s);            // x, -x, or 0
                __m256i s16 = _mm256_maddubs_epi16(one8, v);   // unsigned 1 * signed v
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(s16, one16));
            }
            Y[r] = float(hsum_epi32(acc)) * out_scale;
        }
    });
}
#endif

#if defined(__AVX512BW__)
// 64 int8 lanes per iteration => two mask words at a time.
void gemv_bitplane_int8_avx512(const float* __restrict X, const TernaryMatrixBitplane& W,
                               float* __restrict Y) {
    const int K = W.cols, WPR = W.words_per_row;
    g_xq.resize(size_t(K));
    const float out_scale = quantize_int8(X, K, g_xq.data()) * W.scale;
    const int8_t* __restrict xq = g_xq.data();

    const __m512i one8 = _mm512_set1_epi8(1);
    const __m512i one16 = _mm512_set1_epi16(1);
    const __m512i zero = _mm512_setzero_si512();

    pool().parallel_for(W.rows, [&](int r0, int r1) {
        for (int r = r0; r < r1; ++r) {
            const uint32_t* __restrict p = &W.pos[size_t(r) * WPR];
            const uint32_t* __restrict n = &W.neg[size_t(r) * WPR];
            __m512i acc = _mm512_setzero_si512();

            for (int w = 0; w < WPR; w += 2) {
                const bool pair = (w + 1) < WPR;
                uint64_t kp = p[w], kn = n[w];
                if (pair) {
                    kp |= uint64_t(p[w + 1]) << 32;
                    kn |= uint64_t(n[w + 1]) << 32;
                }
                const __mmask64 load_mask = pair ? ~0ULL : 0xFFFFFFFFULL;
                __m512i x = _mm512_maskz_loadu_epi8(load_mask, xq + w * cfg::kWeightsPerWord);

                // v = +x where pos, -x where neg, 0 elsewhere
                __m512i v = _mm512_maskz_mov_epi8(__mmask64(kp), x);
                v = _mm512_mask_sub_epi8(v, __mmask64(kn), zero, x);

                __m512i s16 = _mm512_maddubs_epi16(one8, v);
                acc = _mm512_add_epi32(acc, _mm512_madd_epi16(s16, one16));
            }
            Y[r] = float(_mm512_reduce_add_epi32(acc)) * out_scale;
        }
    });
}
#endif

// --- fp32 dense reference ---------------------------------------------------
void gemv_fp32(const float* __restrict X, const float* __restrict W, int M, int K,
               float* __restrict Y) {
    pool().parallel_for(M, [&](int r0, int r1) {
        for (int r = r0; r < r1; ++r) {
            const float* __restrict row = W + size_t(r) * K;
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) sum += row[k] * X[k];
            Y[r] = sum;
        }
    });
}

// =============================================================================
// Packing — both layouts filled from the same source weights
// =============================================================================
void pack_interleaved(const std::vector<int8_t>& src, TernaryMatrixInterleaved& W) {
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

void pack_bitplane(const std::vector<int8_t>& src, TernaryMatrixBitplane& W) {
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

std::vector<int8_t> random_ternary(int M, int K, double zero_frac, uint32_t seed) {
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
// Expert / MoE
// =============================================================================
using GemvFn = void (*)(const float*, const TernaryMatrixBitplane&, float*);

struct TernaryExpert {
    int d_model, hidden_dim;
    TernaryMatrixBitplane W_gate, W_up, W_down;
    std::vector<float> gate_out, up_out, inter;

    TernaryExpert(int dim, int hidden, float s)
        : d_model(dim), hidden_dim(hidden),
          W_gate(hidden, dim, s), W_up(hidden, dim, s), W_down(dim, hidden, s),
          gate_out(hidden), up_out(hidden), inter(hidden) {}

    void forward(const float* x, float* out, GemvFn gemv) {
        gemv(x, W_gate, gate_out.data());
        gemv(x, W_up, up_out.data());
        for (int i = 0; i < hidden_dim; ++i) {
            const float g = gate_out[i];
            inter[i] = (g / (1.0f + std::exp(-g))) * up_out[i];
        }
        gemv(inter.data(), W_down, out);
    }
};

class TernaryMoELayer {
public:
    int d_model, num_experts, top_k;
    std::vector<float> W_router;
    std::vector<TernaryExpert> experts;
    GemvFn gemv;
    std::vector<float> logits, weights, expert_out;
    std::vector<int> idx;

    TernaryMoELayer(int dim, int num_exp, int k, GemvFn fn)
        : d_model(dim), num_experts(num_exp), top_k(k),
          W_router(size_t(num_exp) * dim, 0.1f), gemv(fn),
          logits(num_exp), weights(k), expert_out(dim), idx(num_exp) {
        assert(dim % cfg::kWeightsPerWord == 0);
        assert(k > 0 && k <= num_exp);
        experts.reserve(num_exp);
        for (int i = 0; i < num_exp; ++i) experts.emplace_back(dim, dim * 4, 0.05f);
    }

    void forward(const float* x, float* out) {
        for (int e = 0; e < num_experts; ++e) {
            const float* wr = &W_router[size_t(e) * d_model];
            float s = 0.0f;
            for (int d = 0; d < d_model; ++d) s += wr[d] * x[d];
            logits[e] = s;
        }

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

        std::fill(out, out + d_model, 0.0f);
        for (int i = 0; i < top_k; ++i) {
            experts[idx[i]].forward(x, expert_out.data(), gemv);
            const float w = weights[i];
            for (int d = 0; d < d_model; ++d) out[d] += w * expert_out[d];
        }
    }
};

// =============================================================================
// Kernel registry
// =============================================================================
struct Kernel {
    const char* name;
    GemvFn fn;
    bool quantized;  // int8 activations => compare with relative error
};

static std::vector<Kernel> all_kernels() {
    std::vector<Kernel> v = {
        {"A: interleaved (via B)", nullptr, false},  // placeholder, handled separately
        {"B1: bitplane scalar", gemv_bitplane_scalar, false},
        {"B2: bitplane sparse", gemv_bitplane_sparse, false},
#if defined(__AVX2__)
        {"B3: bitplane AVX2", gemv_bitplane_avx2, false},
#endif
#if defined(__AVX512F__)
        {"B3b: bitplane AVX-512", gemv_bitplane_avx512, false},
#endif
        {"B4: int8 scalar", gemv_bitplane_int8_scalar, true},
#if defined(__AVX2__)
        {"B5: int8 AVX2", gemv_bitplane_int8_avx2, true},
#endif
#if defined(__AVX512BW__)
        {"B5b: int8 AVX-512", gemv_bitplane_int8_avx512, true},
#endif
    };
    v.erase(v.begin());  // drop the placeholder
    return v;
}

// =============================================================================
// Correctness tests
// =============================================================================
static int g_failures = 0;

static void check(bool ok, const char* what) {
    if (!ok) { printf("  FAIL: %s\n", what); ++g_failures; }
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
        for (int k = 0; k < K; ++k) {
            const uint32_t bit = 1u << (k % cfg::kWeightsPerWord);
            const size_t i = size_t(r) * Wb.words_per_row + k / cfg::kWeightsPerWord;
            const int got = (Wb.pos[i] & bit) ? 1 : ((Wb.neg[i] & bit) ? -1 : 0);
            if (got != src[size_t(r) * K + k]) {
                printf("  FAIL: %s pack round-trip at (%d,%d)\n", label, r, k);
                ++g_failures;
                return;
            }
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

    for (const auto& k : all_kernels()) {
        std::vector<float> y(size_t(M), NAN);
        k.fn(X.data(), Wb, y.data());
        float m = 0.0f, rmax = 0.0f;
        for (int r = 0; r < M; ++r) {
            m = std::max(m, std::fabs(y[r] - ref[r]));
            rmax = std::max(rmax, std::fabs(ref[r]));
        }
        // int8 quantization error scales with the largest activation
        const float tol = k.quantized ? 0.05f * std::max(rmax, 1e-6f) + 1e-4f : 1e-3f;
        char buf[160];
        snprintf(buf, sizeof buf, "%s / %s (maxdiff %.2e, tol %.2e)", label, k.name, m, tol);
        check(m <= tol, buf);
    }
}

static void run_tests() {
    printf("Correctness tests\n");
    constexpr int M = 8, K = 64;

    std::mt19937 rng(4242);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> X(K);
    for (auto& v : X) v = nd(rng);

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
    {   // saturating activations: check the -128 clamp
        std::vector<float> Xs(K, 0.0f);
        Xs[0] = -1.0f;
        for (int k = 1; k < K; ++k) Xs[k] = -1.0f;
        run_case("uniform -1 activations", random_ternary(M, K, 0.5, 11), Xs, M, K);
    }
    for (double zf : {0.0, 0.5, 0.9, 0.99})
        run_case("random", random_ternary(M, K, zf, unsigned(zf * 1000) + 1), X, M, K);

    // odd row counts and multiple words per row
    run_case("M=1 K=256", random_ternary(1, 256, 0.6, 77),
             [] { std::vector<float> v(256); std::mt19937 r(9);
                  std::normal_distribution<float> d(0, 1);
                  for (auto& x : v) x = d(r); return v; }(), 1, 256);
    run_case("M=33 K=96", random_ternary(33, 96, 0.6, 78),
             [] { std::vector<float> v(96); std::mt19937 r(10);
                  std::normal_distribution<float> d(0, 1);
                  for (auto& x : v) x = d(r); return v; }(), 33, 96);

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

float rel_err(const std::vector<float>& ref, const std::vector<float>& got) {
    double se = 0.0, sr = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = double(ref[i]) - double(got[i]);
        se += d * d;
        sr += double(ref[i]) * double(ref[i]);
    }
    return float(std::sqrt(se / std::max(sr, 1e-30)));
}

int main(int argc, char** argv) {
    int nthreads = int(std::thread::hardware_concurrency());
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], "-t") == 0) nthreads = std::atoi(argv[i + 1]);
    if (nthreads < 1) nthreads = 1;

    ThreadPool p(nthreads);
    g_pool = &p;

    run_tests();

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
#if defined(__AVX512BW__)
    GemvFn fn = gemv_bitplane_int8_avx512;
#elif defined(__AVX2__)
    GemvFn fn = gemv_bitplane_int8_avx2;
#else
    GemvFn fn = gemv_bitplane_int8_scalar;
#endif
    TernaryMoELayer moe(d_model, num_experts, top_k, fn);

    uint32_t seed = 99;
    for (auto& e : moe.experts) {
        pack_bitplane(random_ternary(e.hidden_dim, e.d_model, zero_frac, seed++), e.W_gate);
        pack_bitplane(random_ternary(e.hidden_dim, e.d_model, zero_frac, seed++), e.W_up);
        pack_bitplane(random_ternary(e.d_model, e.hidden_dim, zero_frac, seed++), e.W_down);
    }

    std::vector<float> tok(d_model), out(d_model);
    for (auto& v : tok) v = nd(rng);

    const Stats s_moe = bench([&] { moe.forward(tok.data(), out.data()); }, 200);
    printf("\nMoE  d_model=%d experts=%d top_k=%d\n", d_model, num_experts, top_k);
    printf("  %.3f ms/token (median)   out[0]=%.6f  out[15]=%.6f\n",
           s_moe.median, out[0], out[15]);

    return g_failures == 0 ? 0 : 1;
}
