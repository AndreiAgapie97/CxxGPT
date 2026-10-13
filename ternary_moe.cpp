// ternary_moe.cpp — v3
//
// Ternary MoE inference kernels: two-bitplane weights, int8 activations,
// AVX2 / AVX-512 / VNNI, threaded across output rows.
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
constexpr int kInt8Max       = 127;  // not 128: negating -128 wraps (_mm*_sign_epi8, 0 - x)
constexpr int kMinRowsPerThread = 16;
}  // namespace cfg

// =============================================================================
// Thread pool — persistent workers, parallel_for over output rows
// =============================================================================
// A condition-variable handoff costs tens of microseconds per job (futex
// wake + scheduler), which is longer than a whole expert-sized GEMV. So the
// workers spin on an atomic between jobs and only fall back to sleeping
// after kSpinMicros without work.
static inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#endif
}

class ThreadPool {
public:
    explicit ThreadPool(int n) : nthreads_(std::max(1, n)) {
        for (int i = 1; i < nthreads_; ++i) workers_.emplace_back([this] { worker(); });
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_.store(true, std::memory_order_relaxed);
        }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }

    int size() const { return nthreads_; }

    // Calls fn(row_begin, row_end) on disjoint ranges that together cover
    // [0, total), spread across the pool's threads and the caller. Ranges are
    // claimed dynamically, so a slow or preempted thread just takes fewer.
    // Safe to call from several threads (jobs run one at a time), but not
    // reentrant: fn must not call parallel_for itself.
    void parallel_for(int total, const std::function<void(int, int)>& fn) {
        if (nthreads_ == 1 || total < nthreads_ * cfg::kMinRowsPerThread) {
            fn(0, total);
            return;
        }
        std::lock_guard<std::mutex> one_job(call_m_);
        // Close the previous job first. A worker can still hold its exhausted
        // cursor (gen, old_total); compared against the new total, that would
        // look like unclaimed rows, and its CAS would succeed until the cursor
        // changes. total_ is written last with release, so a worker that reads
        // the new total also sees the close and its claim fails.
        cursor_.store((uint64_t(gen_) << 32) | kClosed, std::memory_order_relaxed);
        job_.store(&fn, std::memory_order_relaxed);
        chunk_.store(std::max(cfg::kMinRowsPerThread, total / (nthreads_ * kChunksPerThread)),
                     std::memory_order_relaxed);
        done_.store(0, std::memory_order_relaxed);
        total_.store(total, std::memory_order_release);
        const uint32_t gen = ++gen_;
        bool wake;
        {   // under m_ so a worker about to sleep sees either this job or the notify
            std::lock_guard<std::mutex> lk(m_);
            cursor_.store(uint64_t(gen) << 32, std::memory_order_release);
            wake = sleepers_ > 0;
        }
        if (wake) cv_.notify_all();

        work(gen);
        for (unsigned spins = 0;; ++spins) {
            const int done = done_.load(std::memory_order_acquire);
            assert(done <= total && "a row was claimed twice");
            if (done == total) break;
            if (spins < 1u << 20) cpu_relax();
            else std::this_thread::yield();  // a worker got preempted mid-chunk
        }
    }

private:
    static constexpr int kChunksPerThread = 2;  // each claim costs ~0.5 us of cache traffic
    static constexpr uint32_t kClosed = 0x7FFFFFFF;  // cursor row: no job open
public:
    static constexpr int kSpinMicros = 1000;

private:

    // Claim and run chunks of job `gen` until none are left. The cursor packs
    // (generation << 32 | next row), so a claim made by a worker that woke up
    // late for an already-finished job fails instead of touching the next one.
    void work(uint32_t gen) {
        uint64_t c = cursor_.load(std::memory_order_acquire);
        for (;;) {
            if (uint32_t(c >> 32) != gen) return;
            const int total = total_.load(std::memory_order_acquire);
            const int r0 = int(uint32_t(c));
            if (r0 >= total) return;
            const int r1 = std::min(r0 + chunk_.load(std::memory_order_relaxed), total);
            if (!cursor_.compare_exchange_weak(c, (uint64_t(gen) << 32) | uint32_t(r1),
                                               std::memory_order_acquire))
                continue;
            // The claim succeeded, so job `gen` is still running and job_ is its.
            (*job_.load(std::memory_order_relaxed))(r0, r1);
            done_.fetch_add(r1 - r0, std::memory_order_release);
        }
    }

    uint32_t current_gen() const {
        return uint32_t(cursor_.load(std::memory_order_acquire) >> 32);
    }

    // Spin until a job newer than `seen` is posted (or stop), then sleep.
    uint32_t wait_for_job(uint32_t seen) {
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::microseconds(kSpinMicros);
        for (unsigned i = 1;; ++i) {
            if (current_gen() != seen || stop_.load(std::memory_order_relaxed))
                return current_gen();
            cpu_relax();
            if (i % 256 == 0 && std::chrono::steady_clock::now() > deadline) break;
        }
        std::unique_lock<std::mutex> lk(m_);
        ++sleepers_;
        cv_.wait(lk, [&] { return current_gen() != seen || stop_.load(std::memory_order_relaxed); });
        --sleepers_;
        return current_gen();
    }

    void worker() {
        uint32_t seen = 0;
        for (;;) {
            seen = wait_for_job(seen);
            if (stop_.load(std::memory_order_relaxed)) return;
            work(seen);
        }
    }

    const int nthreads_;
    std::vector<std::thread> workers_;
    std::mutex call_m_;  // held for a whole job
    uint32_t gen_ = 0;   // guarded by call_m_

    // Job description: written before the cursor is published, read by
    // workers after they see it. Atomic only because a late worker may read
    // them while the next job is being set up (and then discard them).
    std::atomic<const std::function<void(int, int)>*> job_{nullptr};
    std::atomic<int> total_{0}, chunk_{1};

    alignas(64) std::atomic<uint64_t> cursor_{0};
    alignas(64) std::atomic<int> done_{0};  // rows finished in the current job

    alignas(64) std::mutex m_;  // sleeping only
    std::condition_variable cv_;
    int sleepers_ = 0;          // guarded by m_
    std::atomic<bool> stop_{false};
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

    // Weight at (r, k): +1, -1 or 0. For tests and references, not hot loops.
    int weight(int r, int k) const {
        const size_t i = size_t(r) * words_per_row + k / cfg::kWeightsPerWord;
        const uint32_t bit = 1u << (k % cfg::kWeightsPerWord);
        return (pos[i] & bit) ? 1 : ((neg[i] & bit) ? -1 : 0);
    }
};

using GemvFn = void (*)(const float*, const TernaryMatrixBitplane&, float*);

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

// int8 kernels take activations already quantized by quantize_int8(), so an
// input that feeds several matrices (gate and up, every routed expert) is
// quantized once rather than once per matrix.
using GemvQFn = void (*)(const int8_t*, float, const TernaryMatrixBitplane&, float*);

void gemv_bitplane_int8_scalar(const int8_t* __restrict xq, float x_scale,
                               const TernaryMatrixBitplane& W, float* __restrict Y) {
    const int WPR = W.words_per_row;
    const float out_scale = x_scale * W.scale;

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

// One mask word applied to its 32 activations: x, -x, or 0 per byte.
static inline __m256i apply_ternary32(uint32_t bp, uint32_t bn, const int8_t* xw) {
    const __m256i one8 = _mm256_set1_epi8(1);
    const __m256i x = _mm256_loadu_si256((const __m256i*)xw);
    const __m256i s = _mm256_sub_epi8(_mm256_and_si256(expand32(bp), one8),
                                      _mm256_and_si256(expand32(bn), one8));
    return _mm256_sign_epi8(x, s);
}

void gemv_bitplane_int8_avx2(const int8_t* __restrict xq, float x_scale,
                             const TernaryMatrixBitplane& W, float* __restrict Y) {
    const int WPR = W.words_per_row;
    const float out_scale = x_scale * W.scale;

    const __m256i one8 = _mm256_set1_epi8(1);
    const __m256i one16 = _mm256_set1_epi16(1);

    pool().parallel_for(W.rows, [&](int r0, int r1) {
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
    });
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

static inline __m256i dpbusd256(__m256i acc, __m256i u8, __m256i s8) {
#if defined(__AVXVNNI__)
    return _mm256_dpbusd_avx_epi32(acc, u8, s8);  // VEX: AVX-VNNI without AVX-512
#else
    return _mm256_dpbusd_epi32(acc, u8, s8);      // EVEX: AVX512-VNNI + VL
#endif
}

void gemv_bitplane_int8_vnni256(const int8_t* __restrict xq, float x_scale,
                                const TernaryMatrixBitplane& W, float* __restrict Y) {
    const int WPR = W.words_per_row;
    const float out_scale = x_scale * W.scale;
    const __m256i one8 = _mm256_set1_epi8(1);
    constexpr int B = cfg::kAvx2Bytes;

    pool().parallel_for(W.rows, [&](int r0, int r1) {
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
    });
}
#endif

#if defined(__AVX512BW__)
// Two mask words applied to their 64 activations: x, -x, or 0 per byte.
// With pair == false only p[0] / n[0] are read and the upper 32 activation
// bytes are neither loaded nor counted.
static inline __m512i apply_ternary64(const uint32_t* p, const uint32_t* n,
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
void gemv_bitplane_int8_avx512(const int8_t* __restrict xq, float x_scale,
                               const TernaryMatrixBitplane& W, float* __restrict Y) {
    const int WPR = W.words_per_row;
    const float out_scale = x_scale * W.scale;

    const __m512i one8 = _mm512_set1_epi8(1);
    const __m512i one16 = _mm512_set1_epi16(1);

    pool().parallel_for(W.rows, [&](int r0, int r1) {
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
    });
}

#if defined(__AVX512VNNI__)
void gemv_bitplane_int8_vnni512(const int8_t* __restrict xq, float x_scale,
                                const TernaryMatrixBitplane& W, float* __restrict Y) {
    const int WPR = W.words_per_row;
    const float out_scale = x_scale * W.scale;
    const __m512i one8 = _mm512_set1_epi8(1);
    constexpr int B = cfg::kWeightsPerWord;
    constexpr int kAcc = 4;  // each covers one word pair => 8 words per step

    pool().parallel_for(W.rows, [&](int r0, int r1) {
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
    });
}
#endif
#endif

// Float-input adapter for an int8 kernel: quantize, then run Q. The scratch
// is per calling thread, so concurrent callers never share it.
template <GemvQFn Q>
void gemv_quantize_then(const float* __restrict X, const TernaryMatrixBitplane& W,
                        float* __restrict Y) {
    thread_local std::vector<int8_t> xq;
    xq.resize(size_t(W.cols));
    const float x_scale = quantize_int8(X, W.cols, xq.data());
    Q(xq.data(), x_scale, W, Y);
}

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
// Kernel registry
// =============================================================================
struct Kernel {
    const char* name;
    GemvFn fn;    // float activations in; int8 kernels quantize first
    GemvQFn qfn;  // non-null => int8 kernel, callable on pre-quantized input
};

static std::vector<Kernel> all_kernels() {
    return {
        {"B1: bitplane scalar", gemv_bitplane_scalar, nullptr},
        {"B2: bitplane sparse", gemv_bitplane_sparse, nullptr},
#if defined(__AVX2__)
        {"B3: bitplane AVX2", gemv_bitplane_avx2, nullptr},
#endif
#if defined(__AVX512F__)
        {"B3b: bitplane AVX-512", gemv_bitplane_avx512, nullptr},
#endif
        {"B4: int8 scalar", gemv_quantize_then<gemv_bitplane_int8_scalar>,
         gemv_bitplane_int8_scalar},
#if defined(__AVX2__)
        {"B5: int8 AVX2", gemv_quantize_then<gemv_bitplane_int8_avx2>,
         gemv_bitplane_int8_avx2},
#endif
#if defined(__AVX512BW__)
        {"B5b: int8 AVX-512", gemv_quantize_then<gemv_bitplane_int8_avx512>,
         gemv_bitplane_int8_avx512},
#endif
#if defined(TERNARY_HAVE_VNNI256)
        {"B6: int8 VNNI 256-bit", gemv_quantize_then<gemv_bitplane_int8_vnni256>,
         gemv_bitplane_int8_vnni256},
#endif
#if defined(__AVX512BW__) && defined(__AVX512VNNI__)
        {"B6b: int8 VNNI 512-bit", gemv_quantize_then<gemv_bitplane_int8_vnni512>,
         gemv_bitplane_int8_vnni512},
#endif
    };
}

// The int8 kernel the MoE layer uses by default: the widest this build has.
static Kernel default_int8_kernel() {
#if defined(__AVX512BW__) && defined(__AVX512VNNI__)
    const GemvQFn want = gemv_bitplane_int8_vnni512;
#elif defined(__AVX512BW__)
    const GemvQFn want = gemv_bitplane_int8_avx512;
#elif defined(TERNARY_HAVE_VNNI256)
    const GemvQFn want = gemv_bitplane_int8_vnni256;
#elif defined(__AVX2__)
    const GemvQFn want = gemv_bitplane_int8_avx2;
#else
    const GemvQFn want = gemv_bitplane_int8_scalar;
#endif
    for (const auto& k : all_kernels())
        if (k.qfn == want) return k;
    assert(false && "default int8 kernel missing from all_kernels()");
    return all_kernels().front();
}

// =============================================================================
// Expert / MoE
// =============================================================================
struct TernaryExpert {
    int d_model, hidden_dim;
    TernaryMatrixBitplane W_gate, W_up, W_down;
    std::vector<float> gate_out, up_out, inter;

    TernaryExpert(int dim, int hidden, float s)
        : d_model(dim), hidden_dim(hidden),
          W_gate(hidden, dim, s), W_up(hidden, dim, s), W_down(dim, hidden, s),
          gate_out(hidden), up_out(hidden), inter(hidden) {}

    // For an int8 kernel, xq / x_scale must hold x already quantized: the
    // caller does that once per token for all of the token's experts.
    void forward(const float* x, const int8_t* xq, float x_scale, float* out,
                 const Kernel& k) {
        if (k.qfn) {
            k.qfn(xq, x_scale, W_gate, gate_out.data());
            k.qfn(xq, x_scale, W_up, up_out.data());
        } else {
            k.fn(x, W_gate, gate_out.data());
            k.fn(x, W_up, up_out.data());
        }
        for (int i = 0; i < hidden_dim; ++i) {
            const float g = gate_out[i];
            inter[i] = (g / (1.0f + std::exp(-g))) * up_out[i];
        }
        k.fn(inter.data(), W_down, out);  // single consumer: fn quantizes if needed
    }
};

class TernaryMoELayer {
public:
    int d_model, num_experts, top_k;
    std::vector<float> W_router;
    std::vector<TernaryExpert> experts;
    Kernel kernel;
    std::vector<float> logits, weights, expert_out;
    std::vector<int> idx;
    std::vector<int8_t> xq;

    TernaryMoELayer(int dim, int num_exp, int k, const Kernel& kern)
        : d_model(dim), num_experts(num_exp), top_k(k),
          W_router(size_t(num_exp) * dim, 0.0f), kernel(kern),
          logits(num_exp), weights(k), expert_out(dim), idx(num_exp), xq(dim) {
        assert(dim % cfg::kWeightsPerWord == 0);
        assert(k > 0 && k <= num_exp);
        experts.reserve(num_exp);
        for (int i = 0; i < num_exp; ++i) experts.emplace_back(dim, dim * 4, 0.05f);
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

        const float x_scale = kernel.qfn ? quantize_int8(x, d_model, xq.data()) : 0.0f;

        std::fill(out, out + d_model, 0.0f);
        for (int i = 0; i < top_k; ++i) {
            experts[idx[i]].forward(x, xq.data(), x_scale, expert_out.data(), kernel);
            const float w = weights[i];
            for (int d = 0; d < d_model; ++d) out[d] += w * expert_out[d];
        }
    }
};

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

    for (const auto& k : all_kernels()) {
        std::vector<float> y(size_t(M), NAN);
        k.fn(X.data(), Wb, y.data());
        float m = 0.0f, rmax = 0.0f;
        for (int r = 0; r < M; ++r) {
            m = std::max(m, std::fabs(y[r] - ref[r]));
            rmax = std::max(rmax, std::fabs(ref[r]));
        }
        // int8 quantization error scales with the largest activation
        const float tol = k.qfn ? 0.05f * std::max(rmax, 1e-6f) + 1e-4f : 1e-3f;
        char buf[160];
        snprintf(buf, sizeof buf, "%s / %s (maxdiff %.2e, tol %.2e)", label, k.name, m, tol);
        check(m <= tol, buf);
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

    const std::vector<float> Z(64, 0.0f);
    std::vector<int8_t> qz(Z.size(), 1);
    const float sz = quantize_int8(Z.data(), int(Z.size()), qz.data());
    check(sz == 0.0f && std::all_of(qz.begin(), qz.end(), [](int8_t v) { return v == 0; }),
          "quantizer: all-zero input");
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
            const float tol = k.qfn ? 5e-2f : 1e-5f;
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
    TernaryMoELayer moe(d_model, num_experts, top_k, default_int8_kernel());
    moe.randomize(zero_frac, 99);

    std::vector<float> tok(d_model), out(d_model);
    for (auto& v : tok) v = nd(rng);

    const Stats s_moe = bench([&] { moe.forward(tok.data(), out.data()); }, 200);
    printf("\nMoE  d_model=%d experts=%d top_k=%d  kernel=%s\n", d_model, num_experts, top_k,
           moe.kernel.name);
    printf("  %.3f ms/token (median)   routed to experts %d, %d   out[0]=%.6f  out[15]=%.6f\n",
           s_moe.median, moe.idx[0], moe.idx[1], out[0], out[15]);

    return g_failures == 0 ? 0 : 1;
}
