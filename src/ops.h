// ops.h — the layers of a transformer, each as a forward and a backward
// function, in the style of llm.c.
//
// Conventions:
// - Everything is templated on T. float trains; double runs the gradient
//   checks, where finite differences need the extra precision.
// - Activations are row-major [N, C], where N = batch * sequence length.
// - A linear layer's weight is [out, in], and y = x · Wᵀ.
// - Backward functions ACCUMULATE (+=) into every gradient they produce. The
//   caller zeroes gradient buffers before a backward pass. That makes
//   residual connections and tied weights free: two paths just add.
#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

#include "gemm.h"
#include "profile.h"
#include "thread_pool.h"

// Runs fn over row ranges of [0, n) on the pool.
inline void for_rows(int n, const std::function<void(int, int)>& fn, int min_per_thread = 4) {
    pool().parallel_for(n, fn, min_per_thread);
}

// Dot product with 8 independent partial sums, so the compiler vectorizes
// it without having to reorder a floating-point reduction itself.
template <class T>
inline T dot(const T* a, const T* b, int n) {
    T acc[8] = {};
    int i = 0;
    for (; i + 8 <= n; i += 8)
        for (int j = 0; j < 8; ++j) acc[j] += a[i + j] * b[i + j];
    T s = 0;
    for (; i < n; ++i) s += a[i] * b[i];
    return s + ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
}

// exp() in a form the compiler vectorizes: round x / ln2 to an integer k
// with the magic-number trick, reduce r = x - k ln2 (Cody-Waite, two
// constants), evaluate the Cephes polynomial for e^r, then add k to the
// exponent bits. Within 2 ulp of std::exp; clamps instead of overflowing
// (e^-87.3 is the smallest normal float). double keeps std::exp, so the
// gradient checks are unaffected.
inline float fast_exp(float in) {
    // std::max(lo, NaN) returns lo, so the float-to-int conversion below is
    // always defined; NaN is passed through at the end. No early return: a
    // branch here stops GCC from vectorizing the loops that call this.
    const float x = std::min(std::max(-87.33654f, in), 88.37626f);
    const float k = (x * 1.44269504088896341f + 12582912.0f) - 12582912.0f;
    float r = x - k * 0.693359375f;
    r = r - k * -2.12194440e-4f;
    float p = 1.9875691500e-4f;
    p = p * r + 1.3981999507e-3f;
    p = p * r + 8.3334519073e-3f;
    p = p * r + 4.1665795894e-2f;
    p = p * r + 1.6666665459e-1f;
    p = p * r + 5.0000001201e-1f;
    p = p * r * r + r + 1.0f;
    const float result = std::bit_cast<float>(std::bit_cast<int32_t>(p) + (static_cast<int32_t>(k) << 23));
    return in == in ? result : in;  // in != in only for NaN
}
inline double fast_exp(double x) { return std::exp(x); }

// Zeroes n elements, spread over the pool (memory bandwidth scales with cores).
template <class T>
void zero_fill(T* p, size_t n) {
    PROFILE("zero_fill");
    const int blocks = int((n + 16383) / 16384);
    pool().parallel_for(blocks, [&](int b0, int b1) {
        std::fill(p + size_t(b0) * 16384, p + std::min(n, size_t(b1) * 16384), T(0));
    }, 1);
}

// ---------------------------------------------------------------------------
// Linear (no bias): y[N, OC] = x[N, IC] · w[OC, IC]ᵀ
// ---------------------------------------------------------------------------
template <class T>
void linear_forward(T* y, const T* x, const T* w, int N, int IC, int OC) {
    PROFILE("linear_fwd");
    gemm(N, OC, IC, MatView<T>{x, IC, 1}, MatView<T>{w, 1, IC}, y, OC, false);
}

// dx += dy · w (skipped if dx is null), dw += dyᵀ · x.
template <class T>
void linear_backward(T* dx, T* dw, const T* dy, const T* x, const T* w, int N, int IC, int OC) {
    PROFILE("linear_bwd");
    if (dx) gemm(N, IC, OC, MatView<T>{dy, OC, 1}, MatView<T>{w, IC, 1}, dx, IC, true);
    gemm(OC, IC, N, MatView<T>{dy, 1, OC}, MatView<T>{x, IC, 1}, dw, IC, true);
}

// ---------------------------------------------------------------------------
// Token embedding: out[n] = wte[tokens[n]]
// ---------------------------------------------------------------------------
template <class T>
void embedding_forward(T* out, const int* tokens, const T* wte, int N, int C) {
    PROFILE("embedding_fwd");
    for_rows(N, [&](int n0, int n1) {
        for (int n = n0; n < n1; ++n)
            std::copy(wte + int64_t(tokens[n]) * C, wte + int64_t(tokens[n] + 1) * C, out + int64_t(n) * C);
    });
}

// Serial: two positions with the same token add into the same row.
template <class T>
void embedding_backward(T* dwte, const T* dout, const int* tokens, int N, int C) {
    PROFILE("embedding_bwd");
    for (int n = 0; n < N; ++n) {
        T* d = dwte + int64_t(tokens[n]) * C;
        const T* g = dout + int64_t(n) * C;
        for (int c = 0; c < C; ++c) d[c] += g[c];
    }
}

// ---------------------------------------------------------------------------
// RMSNorm: out = x / sqrt(mean(x²) + eps) * w. rstd[N] is saved for backward.
// ---------------------------------------------------------------------------
constexpr double kRmsEps = 1e-5;

template <class T>
void rmsnorm_forward(T* out, T* rstd, const T* x, const T* w, int N, int C) {
    PROFILE("rmsnorm_fwd");
    for_rows(N, [&](int n0, int n1) {
        for (int n = n0; n < n1; ++n) {
            const T* xn = x + int64_t(n) * C;
            const T r = T(1) / std::sqrt(dot(xn, xn, C) / T(C) + T(kRmsEps));
            rstd[n] = r;
            T* o = out + int64_t(n) * C;
            for (int c = 0; c < C; ++c) o[c] = xn[c] * r * w[c];
        }
    });
}

// With r = rstd and g_i = dout_i * w_i:
//   dx_i = r * g_i - r³ / C * x_i * Σ_j g_j x_j,   dw_i = Σ_n dout_i * x_i * r.
template <class T>
void rmsnorm_backward(T* dx, T* dw, const T* dout, const T* x, const T* w, const T* rstd,
                      int N, int C) {
    PROFILE("rmsnorm_bwd");
    for_rows(N, [&](int n0, int n1) {
        for (int n = n0; n < n1; ++n) {
            const T* xn = x + int64_t(n) * C;
            const T* dn = dout + int64_t(n) * C;
            const T r = rstd[n];
            T gx = 0;
            for (int c = 0; c < C; ++c) gx += dn[c] * w[c] * xn[c];
            const T k = r * r * r * gx / T(C);
            T* dxn = dx + int64_t(n) * C;
            for (int c = 0; c < C; ++c) dxn[c] += r * dn[c] * w[c] - k * xn[c];
        }
    });
    // dw sums over rows: each fixed block of rows sums into its own partial
    // (same blocks whatever the thread count, so results are reproducible),
    // then the partials are added up.
    constexpr int kRowsPerBlock = 64;
    const int blocks = (N + kRowsPerBlock - 1) / kRowsPerBlock;
    // thread_local belongs to the calling thread: take the pointer here,
    // since inside the lambda the name would refer to each worker's copy.
    static thread_local std::vector<T> partial_buf;
    partial_buf.assign(size_t(blocks) * C, T(0));
    T* partial = partial_buf.data();
    pool().parallel_for(blocks, [&](int b0, int b1) {
        for (int b = b0; b < b1; ++b) {
            T* acc = partial + size_t(b) * C;
            for (int n = b * kRowsPerBlock; n < std::min(N, (b + 1) * kRowsPerBlock); ++n) {
                const T* xn = x + int64_t(n) * C;
                const T* dn = dout + int64_t(n) * C;
                const T r = rstd[n];
                for (int c = 0; c < C; ++c) acc[c] += dn[c] * xn[c] * r;
            }
        }
    }, 1);
    for (int b = 0; b < blocks; ++b)
        for (int c = 0; c < C; ++c) dw[c] += partial[size_t(b) * C + c];
}

// ---------------------------------------------------------------------------
// Rotary position embedding, applied in place to the q and k parts of
// qkv[B*T, 3C]. Each head's HS dimensions are split in halves; pair
// (i, i + HS/2) is rotated by angle t * base^(-2i/HS). Backward rotates the
// gradient by the opposite angle.
// ---------------------------------------------------------------------------
template <class T>
struct RopeTable {
    int T_max = 0, half = 0;
    std::vector<T> cos, sin;  // [T_max, half]

    RopeTable() = default;
    RopeTable(int t_max, int head_size, double base = 10000.0)
        : T_max(t_max), half(head_size / 2), cos(size_t(t_max) * half), sin(size_t(t_max) * half) {
        for (int t = 0; t < t_max; ++t)
            for (int i = 0; i < half; ++i) {
                const double a = t * std::pow(base, -2.0 * i / head_size);
                cos[size_t(t) * half + i] = T(std::cos(a));
                sin[size_t(t) * half + i] = T(std::sin(a));
            }
    }
};

// Row n of qkv sits at position n % Tn, or at `fixed_pos` for every row if
// that is >= 0 (decoding one token per sequence).
template <class T>
void rope_apply(T* qkv, const RopeTable<T>& rope, int B, int Tn, int C, int NH, bool inverse,
                int fixed_pos = -1) {
    PROFILE("rope");
    const int HS = C / NH, half = HS / 2;
    for_rows(B * Tn, [&](int n0, int n1) {
        for (int n = n0; n < n1; ++n) {
            const int t = fixed_pos >= 0 ? fixed_pos : n % Tn;
            const T* cs = &rope.cos[size_t(t) * half];
            const T* sn = &rope.sin[size_t(t) * half];
            for (int part = 0; part < 2; ++part)  // q, then k
                for (int h = 0; h < NH; ++h) {
                    T* x = qkv + int64_t(n) * 3 * C + part * C + h * HS;
                    for (int i = 0; i < half; ++i) {
                        const T x0 = x[i], x1 = x[i + half];
                        const T s = inverse ? -sn[i] : sn[i];
                        x[i] = x0 * cs[i] - x1 * s;
                        x[i + half] = x0 * s + x1 * cs[i];
                    }
                }
        }
    });
}

template <class T>
void rope_forward(T* qkv, const RopeTable<T>& rope, int B, int Tn, int C, int NH) {
    rope_apply(qkv, rope, B, Tn, C, NH, false);
}

// In place on dqkv: the gradient w.r.t. the rotated q, k becomes the gradient
// w.r.t. the unrotated ones (a rotation's inverse is its transpose).
template <class T>
void rope_backward(T* dqkv, const RopeTable<T>& rope, int B, int Tn, int C, int NH) {
    rope_apply(dqkv, rope, B, Tn, C, NH, true);
}

// ---------------------------------------------------------------------------
// Causal multi-head attention. qkv[B*T, 3C] holds q, k, v side by side; head
// h uses columns [h*HS, (h+1)*HS) of each. out[B*T, C]. att[B, NH, T, T]
// keeps the softmax probabilities (zero above the diagonal) for backward.
//
// Each (batch, head) pair is one task on the pool and does its products with
// the single-threaded gemm. Computing the full T x T score matrix wastes the
// masked half, but the packed kernel is still far faster than dot loops.
// ---------------------------------------------------------------------------
template <class T>
void attention_forward(T* out, T* att, const T* qkv, int B, int Tn, int C, int NH) {
    PROFILE("attention_fwd");
    const int HS = C / NH;
    const T scale = T(1) / std::sqrt(T(HS));
    pool().parallel_for(B * NH, [&](int j0, int j1) {
        for (int j = j0; j < j1; ++j) {
            const int b = j / NH, h = j % NH;
            const T* q = qkv + int64_t(b) * Tn * 3 * C + h * HS;
            const T* k = q + C;
            const T* v = q + 2 * C;
            T* P = att + int64_t(j) * Tn * Tn;
            // P = Q · Kᵀ, then a scaled, causally masked softmax per row.
            gemm(Tn, Tn, HS, MatView<T>{q, 3 * C, 1}, MatView<T>{k, 1, 3 * C}, P, Tn, false, false);
            for (int t = 0; t < Tn; ++t) {
                T* p = P + int64_t(t) * Tn;
                T mx = -std::numeric_limits<T>::infinity();
                for (int s = 0; s <= t; ++s) mx = std::max(mx, p[s]);
                T sum = 0;
                for (int s = 0; s <= t; ++s) {
                    p[s] = fast_exp((p[s] - mx) * scale);
                    sum += p[s];
                }
                const T inv = T(1) / sum;
                for (int s = 0; s <= t; ++s) p[s] *= inv;
                for (int s = t + 1; s < Tn; ++s) p[s] = 0;
            }
            // out = P · V
            gemm(Tn, HS, Tn, MatView<T>{P, Tn, 1}, MatView<T>{v, 3 * C, 1},
                 out + int64_t(b) * Tn * C + h * HS, C, false, false);
        }
    }, 1);
}

// dqkv += gradient w.r.t. q, k, v. Heads write disjoint columns, so the
// (batch, head) tasks never touch the same memory.
template <class T>
void attention_backward(T* dqkv, const T* dout, const T* att, const T* qkv, int B, int Tn, int C,
                        int NH) {
    PROFILE("attention_bwd");
    const int HS = C / NH;
    const T scale = T(1) / std::sqrt(T(HS));
    pool().parallel_for(B * NH, [&](int j0, int j1) {
        static thread_local std::vector<T> scratch;
        scratch.resize(size_t(Tn) * Tn);
        T* dS = scratch.data();
        for (int j = j0; j < j1; ++j) {
            const int b = j / NH, h = j % NH;
            const int64_t base = int64_t(b) * Tn * 3 * C + h * HS;
            const T* q = qkv + base;
            const T* k = q + C;
            const T* v = q + 2 * C;
            T* dq = dqkv + base;
            T* dk = dq + C;
            T* dv = dq + 2 * C;
            const T* P = att + int64_t(j) * Tn * Tn;
            const T* dO = dout + int64_t(b) * Tn * C + h * HS;
            // dP = dO · Vᵀ
            gemm(Tn, Tn, HS, MatView<T>{dO, C, 1}, MatView<T>{v, 1, 3 * C}, dS, Tn, false, false);
            // Softmax backward: dS = P ⊙ (dP - Σ P ⊙ dP), times the score scale.
            // P is zero above the diagonal, so dS is too.
            for (int t = 0; t < Tn; ++t) {
                const T* p = P + int64_t(t) * Tn;
                T* d = dS + int64_t(t) * Tn;
                T pd = 0;
                for (int s = 0; s <= t; ++s) pd += p[s] * d[s];
                for (int s = 0; s <= t; ++s) d[s] = p[s] * (d[s] - pd) * scale;
                for (int s = t + 1; s < Tn; ++s) d[s] = 0;
            }
            // dQ += dS · K,  dK += dSᵀ · Q,  dV += Pᵀ · dO
            gemm(Tn, HS, Tn, MatView<T>{dS, Tn, 1}, MatView<T>{k, 3 * C, 1}, dq, 3 * C, true, false);
            gemm(Tn, HS, Tn, MatView<T>{dS, 1, Tn}, MatView<T>{q, 3 * C, 1}, dk, 3 * C, true, false);
            gemm(Tn, HS, Tn, MatView<T>{P, 1, Tn}, MatView<T>{dO, C, 1}, dv, 3 * C, true, false);
        }
    }, 1);
}

// ---------------------------------------------------------------------------
// SwiGLU: inp[N, 2F] = [gate | up], out[N, F] = silu(gate) * up,
// silu(g) = g * sigmoid(g).
// ---------------------------------------------------------------------------
template <class T>
void swiglu_forward(T* out, const T* inp, int N, int F) {
    PROFILE("swiglu_fwd");
    for_rows(N, [&](int n0, int n1) {
        for (int n = n0; n < n1; ++n) {
            const T* g = inp + int64_t(n) * 2 * F;
            const T* u = g + F;
            T* o = out + int64_t(n) * F;
            for (int i = 0; i < F; ++i) o[i] = g[i] / (T(1) + fast_exp(-g[i])) * u[i];
        }
    });
}

// silu'(g) = sigmoid(g) * (1 + g * (1 - sigmoid(g)))
template <class T>
void swiglu_backward(T* dinp, const T* dout, const T* inp, int N, int F) {
    PROFILE("swiglu_bwd");
    for_rows(N, [&](int n0, int n1) {
        for (int n = n0; n < n1; ++n) {
            const T* g = inp + int64_t(n) * 2 * F;
            const T* u = g + F;
            const T* d = dout + int64_t(n) * F;
            T* dg = dinp + int64_t(n) * 2 * F;
            T* du = dg + F;
            for (int i = 0; i < F; ++i) {
                const T sg = T(1) / (T(1) + fast_exp(-g[i]));
                dg[i] += d[i] * u[i] * sg * (T(1) + g[i] * (T(1) - sg));
                du[i] += d[i] * g[i] * sg;
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Elementwise helpers
// ---------------------------------------------------------------------------
template <class T>
void add_forward(T* out, const T* a, const T* b, int64_t n) {
    PROFILE("residual_add");
    const int blocks = int((n + 16383) / 16384);
    pool().parallel_for(blocks, [&](int b0, int b1) {
        const int64_t i1 = std::min(n, int64_t(b1) * 16384);
        for (int64_t i = int64_t(b0) * 16384; i < i1; ++i) out[i] = a[i] + b[i];
    }, 1);
}

template <class T>
void add_into(T* dst, const T* src, int64_t n) {
    for (int64_t i = 0; i < n; ++i) dst[i] += src[i];
}

// ---------------------------------------------------------------------------
// Softmax cross-entropy. logits[N, V] are replaced by probabilities in place.
// losses[n] = -log p[target]; a target of -1 is ignored (loss 0, no
// gradient). Returns the mean loss over counted positions.
// ---------------------------------------------------------------------------
template <class T>
double cross_entropy_forward(T* probs, T* losses, const int* targets, int N, int V) {
    PROFILE("xent_fwd");
    for (int n = 0; n < N; ++n) CHECK(targets[n] >= -1 && targets[n] < V, "target %d out of range", targets[n]);
    for_rows(N, [&](int n0, int n1) {
        for (int n = n0; n < n1; ++n) {
            T* p = probs + int64_t(n) * V;
            T mx = p[0];
            for (int i = 1; i < V; ++i) mx = std::max(mx, p[i]);
            // Save the target logit before replacing logits by probabilities.
            // log(sum exp(z - max(z))) + max(z) - z_target stays meaningful
            // even when p_target underflows. A clamped -log(p_target) would
            // flatten the loss while backward still returns p - onehot.
            const T target_gap = targets[n] < 0 ? T(0) : mx - p[targets[n]];
            T sum = 0;
            for (int i = 0; i < V; ++i) {
                p[i] = fast_exp(p[i] - mx);
                sum += p[i];
            }
            const T inv = T(1) / sum;
            for (int i = 0; i < V; ++i) p[i] *= inv;
            losses[n] = targets[n] < 0 ? T(0) : target_gap + std::log(sum);
        }
    });
    double total = 0;
    int count = 0;
    for (int n = 0; n < N; ++n)
        if (targets[n] >= 0) total += double(losses[n]), ++count;
    return count ? total / count : 0.0;
}

// ---------------------------------------------------------------------------
// Unlikelihood (#34): besides the right next token, some rows name wrong
// ones whose probability training should push down (for loops: the tokens
// that would repeat a run already written). Row n's negative tokens are
// tokens[offsets[n] .. offsets[n + 1]); their loss is -log(1 - p_c) each,
// times alpha, added to the row's cross-entropy.
// ---------------------------------------------------------------------------
struct NegativeRows {
    const int* offsets = nullptr;  // N + 1 entries
    const int* tokens = nullptr;
    double alpha = 0;
    bool on() const { return offsets && alpha > 0; }
};

// 1 - p_c below this counts as this in the loss, which stays finite (at
// most 69 per negative) when the other probabilities all underflow.
constexpr double kUnlikelihoodFloor = 1e-30;

// 1 - p[c] for one row of probabilities, summed from the other tokens: when
// p[c] is close to 1, 1 - p[c] would lose every digit.
template <class T>
double one_minus_prob(const T* p, int V, int c) {
    double q = 0;
    for (int i = 0; i < V; ++i)
        if (i != c) q += double(p[i]);
    return q;
}

// The sum over rows and their negatives of -log(1 - p_c), on probabilities
// from cross_entropy_forward (alpha not applied). A negative must be a valid
// token on a counted row, never the row's target, and listed once.
template <class T>
double unlikelihood_forward(const T* probs, const int* targets, NegativeRows neg, int N, int V) {
    PROFILE("unlikelihood_fwd");
    for (int n = 0; n < N; ++n)
        for (int k = neg.offsets[n]; k < neg.offsets[n + 1]; ++k)
            CHECK(neg.tokens[k] >= 0 && neg.tokens[k] < V && targets[n] >= 0 && neg.tokens[k] != targets[n],
                  "negative token %d is not a wrong token of a counted row", neg.tokens[k]);
    for (int n = 0; n < N; ++n)
        for (int k = neg.offsets[n]; k < neg.offsets[n + 1]; ++k)
            for (int k2 = neg.offsets[n]; k2 < k; ++k2)
                CHECK(neg.tokens[k2] != neg.tokens[k], "negative token %d appears twice in a row", neg.tokens[k]);
    std::vector<double> rows(size_t(N), 0.0);
    for_rows(N, [&](int n0, int n1) {
        for (int n = n0; n < n1; ++n)
            for (int k = neg.offsets[n]; k < neg.offsets[n + 1]; ++k)
                rows[size_t(n)] -= std::log(std::max(one_minus_prob(probs + int64_t(n) * V, V, neg.tokens[k]),
                                                     kUnlikelihoodFloor));
    });
    double total = 0;
    for (double r : rows) total += r;
    return total;
}

// dlogits = (p - onehot(target)) * dloss for counted rows, 0 for ignored
// ones. Overwrites (not accumulates), and may alias probs.
// With negatives, each row also gets alpha * dloss times the gradient of its
// unlikelihood terms. For one negative c, with q = 1 - p_c:
//   d/dz_j [-log q] = (p_c / q) * (delta_jc - p_j),
// which is p_c at j = c (q cancels exactly, so it is written that way) and
// -(p_c / q) * p_j elsewhere; these sum to 0, and none exceeds 1 even when
// q is tiny, since p_j <= q. The floor only touches the loss value. If q is
// 0 (in float, when every other probability underflows), the push on c
// could not be balanced by the others, so that negative adds no gradient.
template <class T>
void cross_entropy_backward(T* dlogits, const T* probs, const int* targets, int N, int V, T dloss,
                            NegativeRows neg) {
    PROFILE("xent_bwd");
    for (int n = 0; n < N; ++n) CHECK(targets[n] >= -1 && targets[n] < V, "target %d out of range", targets[n]);
    const bool ul = neg.on();
    for_rows(N, [&](int n0, int n1) {
        std::vector<double> w;  // p_c / (1 - p_c) for this row's negatives
        for (int n = n0; n < n1; ++n) {
            const T* p = probs + int64_t(n) * V;
            T* d = dlogits + int64_t(n) * V;
            if (targets[n] < 0) {
                std::fill(d, d + V, T(0));
                continue;
            }
            const int k0 = ul ? neg.offsets[n] : 0, k1 = ul ? neg.offsets[n + 1] : 0;
            if (k0 == k1) {
                for (int i = 0; i < V; ++i) d[i] = p[i] * dloss;
                d[targets[n]] -= dloss;
                continue;
            }
            // Read the whole row before writing it: d may alias p.
            w.clear();
            double sum_w = 0;
            for (int k = k0; k < k1; ++k) {
                const int c = neg.tokens[k];
                const double q = one_minus_prob(p, V, c);
                w.push_back(q > 0 ? double(p[c]) / q : 0.0);
                sum_w += w.back();
            }
            std::vector<double> own(size_t(k1 - k0));
            for (int k = k0; k < k1; ++k) own[size_t(k - k0)] = double(p[neg.tokens[k]]);
            const double rest = 1 - neg.alpha * sum_w;
            for (int i = 0; i < V; ++i) d[i] = T(double(p[i]) * rest * double(dloss));
            for (int k = k0; k < k1; ++k) {
                const double pc = own[size_t(k - k0)], wc = w[size_t(k - k0)];
                const double self = wc > 0 ? neg.alpha * pc : 0.0;  // (p_c / q) * (1 - p_c), with 1 - p_c = q
                d[neg.tokens[k]] = T((pc * (1 - neg.alpha * (sum_w - wc)) + self) * double(dloss));
            }
            d[targets[n]] -= dloss;
        }
    });
}

template <class T>
void cross_entropy_backward(T* dlogits, const T* probs, const int* targets, int N, int V, T dloss) {
    cross_entropy_backward(dlogits, probs, targets, N, V, dloss, NegativeRows{});
}

// ---------------------------------------------------------------------------
// BitLinear quantizers (BitNet b1.58). Training keeps fp32 master weights;
// the forward pass uses quantized copies, and backward passes gradients
// straight through the rounding (the straight-through estimator).
// ---------------------------------------------------------------------------

// Ternary weights: gamma = mean |w|, w_q = clip(round(w / gamma), -1, 1) * gamma.
// The inference kernels store exactly these values: the sign pattern as
// bitplanes and gamma as the matrix scale.
template <class T>
T ternary_quantize(T* wq, const T* w, int64_t n) {
    PROFILE("ternary_quant");
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) s += std::abs(double(w[i]));
    const T gamma = T(s / double(n)) + T(1e-8);
    for (int64_t i = 0; i < n; ++i) {
        const T r = std::nearbyint(w[i] / gamma);
        wq[i] = std::clamp(r, T(-1), T(1)) * gamma;
    }
    return gamma;
}

// int8 activations, one scale per row (token): scale = max|x| / 127,
// x_q = round(x / scale) * scale, with exactly the arithmetic of the
// inference kernels' quantize_int8 (multiply by 1 / scale, round half to
// even, clamp to [-127, 127]), so a trained model sees the same int8 levels
// in CPU training and deployment. GPU TF32/bf16 GEMMs round these inputs
// further; this input-arithmetic contract does not promise identical GPU products.
template <class T>
void act_quantize(T* xq, const T* x, int N, int C) {
    PROFILE("act_quant");
    for_rows(N, [&](int n0, int n1) {
        for (int n = n0; n < n1; ++n) {
            const T* xn = x + int64_t(n) * C;
            T* q = xq + int64_t(n) * C;
            T amax = 0;
            for (int c = 0; c < C; ++c) amax = std::max(amax, std::abs(xn[c]));
            if (amax == T(0)) {
                std::fill(q, q + C, T(0));
                continue;
            }
            const T scale = amax / T(127);
            if (scale == T(0)) { std::fill(q, q + C, T(0)); continue; }
            if (scale <= T(1) / std::numeric_limits<T>::max()) {
                // Separate loops keep vectorization from speculating an
                // unused overflowing reciprocal on the tiny-scale path.
                for (int c = 0; c < C; ++c) {
                    const T rounded = std::clamp(std::nearbyint(xn[c] / scale), T(-127), T(127));
                    // int8 zero dequantizes as +0; nearbyint can produce -0.
                    q[c] = (rounded == T(0) ? T(0) : rounded) * scale;
                }
                continue;
            }
            const T inv = T(1) / scale;
            for (int c = 0; c < C; ++c) {
                const T rounded = std::clamp(std::nearbyint(xn[c] * inv), T(-127), T(127));
                // int8 zero dequantizes as +0; nearbyint can produce -0.
                q[c] = (rounded == T(0) ? T(0) : rounded) * scale;
            }
        }
    });
}

// BitLinear with already-quantized weights. Caller owns xq[N, IC] scratch;
// master weights are quantized once per optimizer update by the model.
template <class T>
void bitlinear_forward(T* y, const T* x, const T* wq, T* xq, int N, int IC, int OC) {
    act_quantize(xq, x, N, IC);
    linear_forward(y, xq, wq, N, IC, OC);
}

// Identity straight-through estimator (STE): dx += dy Wq and
// dW_master += dy^T xq. Scales and rounding decisions are detached.
// This is the derivative of a local surrogate with quantizer Jacobians I,
// not the exact derivative of the discrete forward function.
template <class T>
void bitlinear_backward(T* dx, T* dw, const T* dy, const T* x, const T* wq, T* xq,
                        int N, int IC, int OC) {
    act_quantize(xq, x, N, IC);
    linear_backward(dx, dw, dy, xq, wq, N, IC, OC);
}
