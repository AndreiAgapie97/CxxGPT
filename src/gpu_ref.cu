// gpu_ref.cu — the double-precision reference model (gpu_ref.h). Simple on
// purpose: one thread per output value, plain loops, no shared memory, no
// tricks; the same formulas as the CPU engine had (model.h, ops.h, train.h).
#include "gpu_ref.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <utility>

#include "common.h"

#define REF_CHECK(call)                                                                                     \
    do {                                                                                                    \
        const cudaError_t err_ = (call);                                                                    \
        if (err_ != cudaSuccess) {                                                                          \
            std::fprintf(stderr, "%s:%d: CUDA error: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_));  \
            std::exit(1);                                                                                   \
        }                                                                                                   \
    } while (0)
#define REF_LAUNCH() REF_CHECK(cudaGetLastError())

namespace {

constexpr int kThreads = 128;
int grid_for(size_t n) { return int(std::max<size_t>(1, (n + kThreads - 1) / kThreads)); }

// A device array that grows only.
template <class T>
struct Buf {
    T* p = nullptr;
    size_t n = 0;
    Buf() = default;
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
    Buf(Buf&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr, o.n = 0; }
    ~Buf() {
        if (p) cudaFree(p);
    }
    T* need(size_t count) {
        if (count <= n && p) return p;
        if (p) cudaFree(p);
        REF_CHECK(cudaMalloc(&p, std::max<size_t>(count, 1) * sizeof(T)));
        n = std::max<size_t>(count, 1);
        return p;
    }
    void up(const T* h, size_t count) {
        need(count);
        REF_CHECK(cudaMemcpy(p, h, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    void down(T* h, size_t count) const { REF_CHECK(cudaMemcpy(h, p, count * sizeof(T), cudaMemcpyDeviceToHost)); }
    void zero(size_t count) {
        need(count);
        REF_CHECK(cudaMemset(p, 0, count * sizeof(T)));
    }
};

// ---- Elementwise and row kernels ----

__global__ void k_embedding(double* x, const int* tok, const double* wte, int N, int C) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(N) * C) return;
    x[i] = wte[size_t(tok[i / C]) * C + i % C];
}

// out = x / sqrt(mean(x²) + 1e-5) * w, row by row.
__global__ void k_rmsnorm(double* out, const double* x, const double* w, int N, int C) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const double* xn = x + size_t(n) * C;
    double ss = 0;
    for (int c = 0; c < C; ++c) ss += xn[c] * xn[c];
    const double r = 1.0 / sqrt(ss / C + 1e-5);
    for (int c = 0; c < C; ++c) out[size_t(n) * C + c] = xn[c] * r * w[c];
}

__global__ void k_add(double* out, const double* a, const double* b, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

// y[n, o] = Σ_c x[xrow(n), c] · w[wsel(n), o, c]: x rows through xrow (or n),
// the weight matrix through wsel (an expert per row; or the one matrix).
__global__ void k_linear(double* y, const double* x, const int* xrow, const double* w, const int* wsel,
                         size_t wstride, int N, int IC, int OC) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(N) * OC) return;
    const int n = int(i / OC), o = int(i % OC);
    const double* xn = x + size_t(xrow ? xrow[n] : n) * IC;
    const double* wo = w + (wsel ? size_t(wsel[n]) * wstride : 0) + size_t(o) * IC;
    double s = 0;
    for (int c = 0; c < IC; ++c) s += xn[c] * wo[c];
    y[i] = s;
}

// int8 activations, one scale per row (ops.h's act_quantize in double):
// scale = max|x| / 127, round(x · (1 / scale)) half to even, clamped to
// [-127, 127], times scale; an all-zero row stays zero.
__global__ void k_act_quant(double* q, const double* x, int N, int C) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const double* xn = x + size_t(n) * C;
    double amax = 0;
    for (int c = 0; c < C; ++c) amax = fmax(amax, fabs(xn[c]));
    const double scale = amax / 127.0;
    for (int c = 0; c < C; ++c) {
        double r = 0;
        if (scale > 0) r = fmin(127.0, fmax(-127.0, rint(xn[c] * (1.0 / scale))));
        q[size_t(n) * C + c] = r == 0 ? 0.0 : r * scale;
    }
}

// Ternary weights of one copy (ternary_quantize in double): gamma = mean |w|
// + 1e-8, clip(round(w / gamma), -1, 1) · gamma. One thread: tests are small.
__global__ void k_ternary(double* wq, const double* w, size_t n) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    double s = 0;
    for (size_t i = 0; i < n; ++i) s += fabs(w[i]);
    const double gamma = s / double(n) + 1e-8;
    for (size_t i = 0; i < n; ++i) wq[i] = fmin(1.0, fmax(-1.0, rint(w[i] / gamma))) * gamma;
}

// The same in float, as the CPU engine's float model computed it.
__global__ void k_ternary_float(float* wq, const float* w, size_t n) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    double s = 0;
    for (size_t i = 0; i < n; ++i) s += fabs(double(w[i]));
    const float gamma = float(s / double(n)) + 1e-8f;
    for (size_t i = 0; i < n; ++i) wq[i] = fminf(1.0f, fmaxf(-1.0f, rintf(w[i] / gamma))) * gamma;
}

// a = b + c, or with c null a = b (offsets: frozen inputs).
__global__ void k_plus(double* a, const double* b, const double* c, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) a[i] = b[i] + (c ? c[i] : 0.0);
}
__global__ void k_minus(double* a, const double* b, const double* c, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) a[i] = b[i] - c[i];
}

// RoPE on q and k of row n (position n % Tn): pairs (i, i + HS/2) rotated by
// t · 10000^(-2i / HS).
__global__ void k_rope(double* qkv, int N, int Tn, int C, int NH) {
    const int HS = C / NH, half = HS / 2;
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(N) * 2 * NH * half) return;
    const int k = int(i % half), h = int(i / half % NH), part = int(i / half / NH % 2), n = int(i / half / NH / 2);
    const int t = n % Tn;
    const double a = t * pow(10000.0, -2.0 * k / HS), cs = cos(a), sn = sin(a);
    double* x = qkv + size_t(n) * 3 * C + part * C + h * HS;
    const double x0 = x[k], x1 = x[k + half];
    x[k] = x0 * cs - x1 * sn;
    x[k + half] = x0 * sn + x1 * cs;
}

// Causal attention: out[n, h·HS + d] = Σ_{s<=t} softmax_s(q_t · k_s / sqrt(HS)) v_s[d],
// one thread per (row, head); lse (optional) the scaled scores' log-sum-exp.
__global__ void k_attention(double* out, double* lse, const double* qkv, int B, int Tn, int C, int NH) {
    const int HS = C / NH;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * Tn * NH) return;
    const int h = i % NH, n = i / NH, b = n / Tn, t = n % Tn;
    const double scale = 1.0 / sqrt(double(HS));
    const double* q = qkv + size_t(n) * 3 * C + h * HS;
    auto score = [&](int s) {
        const double* k = qkv + (size_t(b) * Tn + s) * 3 * C + C + h * HS;
        double d = 0;
        for (int j = 0; j < HS; ++j) d += q[j] * k[j];
        return d * scale;
    };
    double mx = -INFINITY;
    for (int s = 0; s <= t; ++s) mx = fmax(mx, score(s));
    double sum = 0;
    for (int s = 0; s <= t; ++s) sum += exp(score(s) - mx);
    if (lse) lse[(size_t(b) * NH + h) * Tn + t] = mx + log(sum);
    for (int d = 0; d < HS; ++d) {
        double y = 0;
        for (int s = 0; s <= t; ++s) y += exp(score(s) - mx) * qkv[(size_t(b) * Tn + s) * 3 * C + 2 * C + h * HS + d];
        out[size_t(n) * C + h * HS + d] = y / sum;
    }
}

// The attention backward pass, in two passes. First, per (query row, head):
// the scaled scores' max and exp-sum, and Σ_s P_ts dP_ts, where P is the
// softmax and dP_ts = dout_t · v_s. stats[i] = {max, sum, Σ P dP}.
__global__ void k_attention_stats(double* stats, const double* dout, const double* qkv, int B, int Tn, int C, int NH) {
    const int HS = C / NH;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * Tn * NH) return;
    const int h = i % NH, n = i / NH, b = n / Tn, t = n % Tn;
    const double scale = 1.0 / sqrt(double(HS));
    const double* q = qkv + size_t(n) * 3 * C + h * HS;
    const double* d = dout + size_t(n) * C + h * HS;
    auto score = [&](int s) {
        const double* k = qkv + (size_t(b) * Tn + s) * 3 * C + C + h * HS;
        double x = 0;
        for (int j = 0; j < HS; ++j) x += q[j] * k[j];
        return x * scale;
    };
    double mx = -INFINITY;
    for (int s = 0; s <= t; ++s) mx = fmax(mx, score(s));
    double sum = 0;
    for (int s = 0; s <= t; ++s) sum += exp(score(s) - mx);
    double pdp = 0;
    for (int s = 0; s <= t; ++s) {
        const double* v = qkv + (size_t(b) * Tn + s) * 3 * C + 2 * C + h * HS;
        double dp = 0;
        for (int j = 0; j < HS; ++j) dp += d[j] * v[j];
        pdp += exp(score(s) - mx) / sum * dp;
    }
    stats[size_t(i) * 3] = mx;
    stats[size_t(i) * 3 + 1] = sum;
    stats[size_t(i) * 3 + 2] = pdp;
}

// Then per (row r, head): with dS_ts = P_ts (dP_ts - Σ_u P_tu dP_tu),
// dq_r += scale Σ_s dS_rs k_s; as a key and value, row r gathers from the
// query rows t >= r: dk_r += scale Σ_t dS_tr q_t, dv_r += Σ_t P_tr dout_t.
__global__ void k_attention_bwd(double* dqkv, const double* dout, const double* qkv, const double* stats, int B,
                                int Tn, int C, int NH) {
    const int HS = C / NH;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * Tn * NH) return;
    const int h = i % NH, n = i / NH, b = n / Tn, r = n % Tn;
    const double scale = 1.0 / sqrt(double(HS));
    auto q = [&](int t) { return qkv + (size_t(b) * Tn + t) * 3 * C + h * HS; };
    auto d = [&](int t) { return dout + (size_t(b) * Tn + t) * C + h * HS; };
    auto st = [&](int t) { return stats + ((size_t(b) * Tn + t) * NH + h) * 3; };
    // P_ts and dS_ts.
    auto weights = [&](int t, int s, double* ds) {
        double x = 0, dp = 0;
        for (int j = 0; j < HS; ++j) x += q(t)[j] * q(s)[C + j], dp += d(t)[j] * q(s)[2 * C + j];
        const double p = exp(x * scale - st(t)[0]) / st(t)[1];
        *ds = p * (dp - st(t)[2]);
        return p;
    };
    double* g = dqkv + (size_t(b) * Tn + r) * 3 * C + h * HS;
    for (int s = 0; s <= r; ++s) {
        double ds;
        weights(r, s, &ds);
        for (int j = 0; j < HS; ++j) g[j] += scale * ds * q(s)[C + j];
    }
    for (int t = r; t < Tn; ++t) {
        double ds;
        const double p = weights(t, r, &ds);
        for (int j = 0; j < HS; ++j) {
            g[C + j] += scale * ds * q(t)[j];
            g[2 * C + j] += p * d(t)[j];
        }
    }
}

// SwiGLU: out[n, i] = g / (1 + e^-g) · u for g, u the row's halves.
__global__ void k_swiglu(double* out, const double* inp, int N, int F) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(N) * F) return;
    const size_t n = i / F, j = i % F;
    const double g = inp[n * 2 * F + j], u = inp[n * 2 * F + F + j];
    out[i] = g / (1.0 + exp(-g)) * u;
}

// Routing for each token: the top K experts by logit + bias (the lowest
// index first on a tie), gates a softmax over their unbiased logits; each
// choice also counted in load[e]. With fixed, the choices already in sel
// are kept (Quant::frozen: the recorded ones).
__global__ void k_route(int* sel, double* gate, unsigned long long* load, const double* rl, const double* bias,
                        int N, int E, int K, bool fixed) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const double* r = rl + size_t(n) * E;
    int* s = sel + size_t(n) * K;
    for (int i = 0; i < K; ++i) {
        if (fixed) {
            atomicAdd(load + s[i], 1ull);
            continue;
        }
        int best = -1;
        for (int e = 0; e < E; ++e) {
            bool taken = false;
            for (int j = 0; j < i; ++j) taken |= s[j] == e;
            if (!taken && (best < 0 || r[e] + bias[e] > r[best] + bias[best])) best = e;
        }
        s[i] = best;
        atomicAdd(load + best, 1ull);
    }
    double mx = r[s[0]];
    for (int i = 1; i < K; ++i) mx = fmax(mx, r[s[i]]);
    double sum = 0;
    for (int i = 0; i < K; ++i) sum += exp(r[s[i]] - mx);
    for (int i = 0; i < K; ++i) gate[size_t(n) * K + i] = exp(r[s[i]] - mx) / sum;
}

// out[n] += Σ_i gate[n, i] · y[n·K + i] (the experts' outputs, one row per choice).
__global__ void k_combine(double* out, const double* y, const double* gate, int N, int C, int K) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(N) * C) return;
    const size_t n = i / C, c = i % C;
    for (int k = 0; k < K; ++k) out[i] += gate[n * K + k] * y[(n * K + k) * C + c];
}

// Per row: the softmax of the logits into p, and the cross-entropy
// -log p[target] (0 where the target is -1).
__global__ void k_softmax_ce(double* p, double* loss, const double* logits, const int* targets, int N, int V) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const double* z = logits + size_t(n) * V;
    double mx = z[0];
    for (int i = 1; i < V; ++i) mx = fmax(mx, z[i]);
    double sum = 0;
    for (int i = 0; i < V; ++i) sum += exp(z[i] - mx);
    for (int i = 0; i < V; ++i) p[size_t(n) * V + i] = exp(z[i] - mx) / sum;
    loss[n] = targets[n] < 0 ? 0.0 : mx + log(sum) - z[targets[n]];
}

// unlikelihood.h's negatives, one thread per window row, walking it token by
// token as NoRepeat does: the written text since the last BOS or skip, and
// every run of N tokens written since the document's BOS. A position's
// negatives are the tokens that followed its last N - 1 tokens in an
// earlier run, without the target, each once, in order. neg [N·Tn, Tn]: up
// to Tn per position; cnt per position.
__global__ void k_negatives(int* neg, int* cnt, const int* x, const int* y, int B, int Tn, int Nn, int* text,
                            int* runs) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    const int* xb = x + size_t(b) * Tn;
    const int* yb = y + size_t(b) * Tn;
    int* tx = text + size_t(b) * Tn;    // the text since the BOS or the last skip
    int* rn = runs + size_t(b) * Tn;    // the runs since the BOS: positions in tx' history where each ended
    int* hist = runs + size_t(B) * Tn + size_t(b) * Tn * Nn;  // each run's N tokens
    int len = 0, nruns = 0;
    for (int t = 0; t < Tn; ++t) {
        if (xb[t] == kBos) {
            len = 0;
            nruns = 0;
        } else if (t == 0 || yb[t - 1] < 0) {
            len = 0;  // skip: the text starts again, the runs stay
        } else {
            tx[len++] = xb[t];
            if (len >= Nn) {  // a run of N tokens just completed
                for (int j = 0; j < Nn; ++j) hist[size_t(nruns) * Nn + j] = tx[len - Nn + j];
                rn[nruns++] = t;
            }
        }
        int* out = neg + (size_t(b) * Tn + t) * Tn;
        int n = 0;
        if (yb[t] >= 0 && len >= Nn - 1)
            for (int r = 0; r < nruns; ++r) {
                const int* run = hist + size_t(r) * Nn;
                bool same = true;
                for (int j = 0; j < Nn - 1 && same; ++j) same = run[j] == tx[len - Nn + 1 + j];
                const int c = run[Nn - 1];
                if (!same || c == yb[t]) continue;
                bool listed = false;
                for (int k = 0; k < n && !listed; ++k) listed = out[k] == c;
                if (!listed) out[n++] = c;
            }
        cnt[size_t(b) * Tn + t] = n;
    }
}

// The unlikelihood term of each row: Σ over its negatives c of
// -log(max(1 - p_c, 1e-30)), with 1 - p_c summed from the other tokens.
__global__ void k_ul_rows(double* ul, const double* p, const int* neg, const int* cnt, int N, int Tn, int V) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    double s = 0;
    for (int k = 0; k < cnt[n]; ++k) {
        const int c = neg[size_t(n) * Tn + k];
        double q = 0;
        for (int i = 0; i < V; ++i)
            if (i != c) q += p[size_t(n) * V + i];
        s -= log(fmax(q, 1e-30));
    }
    ul[n] = s;
}

// ---- Optimizer pieces ----

// C[m, n] = Σ_k A[m, k] B[k, n] (row-major, plain).
__global__ void k_matmul(double* Cm, const double* A, const double* Bm, int M, int Nc, int K) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(M) * Nc) return;
    const size_t m = i / Nc, n = i % Nc;
    double s = 0;
    for (int k = 0; k < K; ++k) s += A[m * K + k] * Bm[size_t(k) * Nc + n];
    Cm[i] = s;
}
__global__ void k_transpose(double* T_, const double* A, int M, int Nc) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(M) * Nc) return;
    T_[(i % Nc) * M + i / Nc] = A[i];
}
__global__ void k_axpby(double* out, double a, const double* x, double b, const double* y, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a * x[i] + b * y[i];
}
__global__ void k_scale(double* x, double s, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

__global__ void k_adamw(double* p, const double* g, double* m, double* v, size_t n, double lr, double b1, double b2,
                        double c1, double c2, double eps, double wd) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    m[i] = b1 * m[i] + (1 - b1) * g[i];
    v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
    const double upd = (m[i] * c1) / (sqrt(v[i] * c2) + eps);
    p[i] -= lr * (upd + wd * p[i]);
}

// Muon's momentum for one r x c block: mom = mu mom + (1 - mu) g, and
// u = (1 - mu) g + mu mom (Nesterov).
__global__ void k_muon_mom(double* u, double* mom, const double* g, size_t n, double mu) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    mom[i] = mu * mom[i] + (1 - mu) * g[i];
    u[i] = (1 - mu) * g[i] + mu * mom[i];
}
__global__ void k_muon_apply(double* p, const double* X, size_t n, double lr, double scale, double wd) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) p[i] -= lr * (scale * X[i] + wd * p[i]);
}

__global__ void k_sum_squares(double* out, const float* x, size_t n) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    double s = 0;
    for (size_t i = 0; i < n; ++i) s += double(x[i]) * double(x[i]);
    *out = s;
}

}  // namespace

// ---------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------

struct GpuReference::Impl {
    Config cfg;
    std::vector<ParamInfo> infos;
    size_t n_params = 0;
    Buf<double> params, wq, wq_off, bias;
    std::map<std::string, size_t> offset;  // tensor -> offset
    // Quantization offsets per projection site (Quant::record, frozen).
    std::map<std::string, Buf<double>> act_off;
    bool w_recorded = false;
    // Each layer's expert choices (record; frozen keeps them: a choice that
    // flips between w - h v and w + h v is a step, not a gradient).
    std::map<int, std::vector<int>> routes;
    // Activations of the last forward.
    Buf<double> x0, h, ln, qkv, att, tmp, resid, fc, sw, rl, gate, ey, efc, esw, xq, logits, probs, losses;
    Buf<int> tok, tgt, sel, xrow, wsel, neg, cnt, text, runs;
    Buf<unsigned long long> load;
    Buf<double> ul_rows;
    int N_last = 0;

    Impl(const Config& c, const std::vector<ParamInfo>& in) : cfg(c), infos(in) {
        for (const auto& p : infos) {
            n_params = std::max(n_params, p.offset + p.size);
            offset[p.name] = p.offset;
        }
        params.zero(n_params);
        wq.zero(n_params);
        wq_off.zero(n_params);
        const int E = std::max(1, cfg.n_experts);
        bias.zero(size_t(cfg.n_layers) * E);
    }

    size_t off(const std::string& name) const {
        const auto it = offset.find(name);
        CHECK(it != offset.end(), "reference: no tensor %s", name.c_str());
        return it->second;
    }

    // The weights the projections use: ternary copies for the ternary
    // tensors (record: their offsets kept; frozen: w plus those offsets).
    void prepare_weights(Quant quant) {
        REF_CHECK(cudaMemcpy(wq.p, params.p, n_params * sizeof(double), cudaMemcpyDeviceToDevice));
        for (const auto& p : infos) {
            if (!p.ternary) continue;
            if (quant == Quant::frozen) {
                CHECK(w_recorded, "reference: Quant::frozen without a Quant::record forward first");
                k_plus<<<grid_for(p.size), kThreads>>>(wq.p + p.offset, params.p + p.offset, wq_off.p + p.offset, p.size);
                REF_LAUNCH();
                continue;
            }
            const size_t per = p.size / p.count;
            for (int c = 0; c < p.count; ++c) k_ternary<<<1, 1>>>(wq.p + p.offset + c * per, params.p + p.offset + c * per, per);
            REF_LAUNCH();
            if (quant == Quant::record) {
                k_minus<<<grid_for(p.size), kThreads>>>(wq_off.p + p.offset, wq.p + p.offset, params.p + p.offset, p.size);
                REF_LAUNCH();
            }
        }
        if (quant == Quant::record) w_recorded = true;
    }

    // A projection's input as the product reads it: for a BitLinear (q), the
    // int8-rounded activations (record: offsets kept under `site`; frozen:
    // x plus them).
    const double* proj_input(const double* x, int N, int IC, bool q, Quant quant, const std::string& site) {
        if (!q) return x;
        double* out = xq.need(size_t(N) * IC);
        const size_t n = size_t(N) * IC;
        if (quant == Quant::frozen) {
            const auto it = act_off.find(site);
            CHECK(it != act_off.end(), "reference: no recorded offsets for %s", site.c_str());
            k_plus<<<grid_for(n), kThreads>>>(out, x, it->second.p, n);
        } else {
            k_act_quant<<<grid_for(size_t(N)), kThreads>>>(out, x, N, IC);
            if (quant == Quant::record) {
                Buf<double>& o = act_off[site];
                o.need(n);
                k_minus<<<grid_for(n), kThreads>>>(o.p, out, x, n);
            }
        }
        REF_LAUNCH();
        return out;
    }

    void linear(double* y, const double* x, size_t woff, int N, int IC, int OC, bool q,
                const int* xr = nullptr, const int* ws = nullptr, size_t wstride = 0) {
        const double* w = (q ? wq.p : params.p) + woff;
        k_linear<<<grid_for(size_t(N) * OC), kThreads>>>(y, x, xr, w, ws, wstride, N, IC, OC);
        REF_LAUNCH();
    }

    Result forward(const int* tokens_h, const int* targets_h, int B, int Tn, double alpha, int ngram, Quant quant) {
        const int N = B * Tn, C = cfg.d_model, V = cfg.vocab, NH = cfg.n_heads, L = cfg.n_layers;
        const int F = cfg.always_hidden(), Fe = cfg.ehidden(), E = cfg.n_experts, K = cfg.top_k;
        for (int n = 0; n < N; ++n) CHECK(tokens_h[n] >= 0 && tokens_h[n] < V, "token %d out of range", tokens_h[n]);
        N_last = N;
        tok.up(tokens_h, N);
        prepare_weights(quant);
        const bool qa = cfg.ternary && !cfg.fp_attention, qf = cfg.ternary;
        Result res;
        res.expert_load.assign(L, std::vector<double>(std::max(1, E), 0.0));
        x0.need(size_t(N) * C);
        k_embedding<<<grid_for(size_t(N) * C), kThreads>>>(x0.p, tok.p, params.p + off("wte"), N, C);
        REF_LAUNCH();
        h.need(size_t(N) * C);
        REF_CHECK(cudaMemcpy(h.p, x0.p, size_t(N) * C * sizeof(double), cudaMemcpyDeviceToDevice));
        for (int l = 0; l < L; ++l) {
            const std::string p = "l" + std::to_string(l) + ".";
            ln.need(size_t(N) * C);
            k_rmsnorm<<<grid_for(N), kThreads>>>(ln.p, h.p, params.p + off(p + "ln1"), N, C);
            REF_LAUNCH();
            qkv.need(size_t(N) * 3 * C);
            linear(qkv.p, proj_input(ln.p, N, C, qa, quant, p + "wqkv"), off(p + "wqkv"), N, C, 3 * C, qa);
            if (cfg.rope_layer(l)) {
                k_rope<<<grid_for(size_t(N) * C), kThreads>>>(qkv.p, N, Tn, C, NH);
                REF_LAUNCH();
            }
            att.need(size_t(N) * C);
            k_attention<<<grid_for(size_t(N) * NH), kThreads>>>(att.p, nullptr, qkv.p, B, Tn, C, NH);
            REF_LAUNCH();
            tmp.need(size_t(N) * C);
            linear(tmp.p, proj_input(att.p, N, C, qa, quant, p + "wo"), off(p + "wo"), N, C, C, qa);
            resid.need(size_t(N) * C);
            k_add<<<grid_for(size_t(N) * C), kThreads>>>(resid.p, h.p, tmp.p, size_t(N) * C);
            REF_LAUNCH();
            k_rmsnorm<<<grid_for(N), kThreads>>>(ln.p, resid.p, params.p + off(p + "ln2"), N, C);
            REF_LAUNCH();
            // The FFN into tmp: the dense part (or the shared experts), then the routed experts.
            REF_CHECK(cudaMemset(tmp.p, 0, size_t(N) * C * sizeof(double)));
            if (F > 0) {
                fc.need(size_t(N) * 2 * F);
                sw.need(size_t(N) * F);
                linear(fc.p, proj_input(ln.p, N, C, qf, quant, p + "wfc"), off(p + "wfc"), N, C, 2 * F, qf);
                k_swiglu<<<grid_for(size_t(N) * F), kThreads>>>(sw.p, fc.p, N, F);
                REF_LAUNCH();
                linear(tmp.p, proj_input(sw.p, N, F, qf, quant, p + "wproj"), off(p + "wproj"), N, F, C, qf);
            }
            if (cfg.moe()) {
                const int NK = N * K;
                rl.need(size_t(N) * E);
                linear(rl.p, ln.p, off(p + "router"), N, C, E, false);
                sel.need(NK);
                gate.need(NK);
                load.zero(E);
                const bool fixed = quant == Quant::frozen;
                if (fixed) {
                    const auto it = routes.find(l);
                    CHECK(it != routes.end() && it->second.size() == size_t(NK), "reference: no recorded routes");
                    sel.up(it->second.data(), NK);
                }
                k_route<<<grid_for(N), kThreads>>>(sel.p, gate.p, load.p, rl.p, bias.p + size_t(l) * E, N, E, K, fixed);
                REF_LAUNCH();
                std::vector<unsigned long long> ld(E);
                load.down(ld.data(), E);
                for (int e = 0; e < E; ++e) res.expert_load[l][e] = double(ld[e]);
                // One row per choice (token n·K + i): its token, its expert.
                std::vector<int> sel_h(NK), xr(NK);
                sel.down(sel_h.data(), NK);
                if (quant == Quant::record) routes[l] = sel_h;
                for (int j = 0; j < NK; ++j) xr[j] = j / K;
                xrow.up(xr.data(), NK);
                wsel.up(sel_h.data(), NK);
                efc.need(size_t(NK) * 2 * Fe);
                esw.need(size_t(NK) * Fe);
                ey.need(size_t(NK) * C);
                // The experts' input is the token's row: quantized per token.
                const double* xin = proj_input(ln.p, N, C, qf, quant, p + "efc");
                linear(efc.p, xin, off(p + "efc"), NK, C, 2 * Fe, qf, xrow.p, wsel.p, size_t(2) * Fe * C);
                k_swiglu<<<grid_for(size_t(NK) * Fe), kThreads>>>(esw.p, efc.p, NK, Fe);
                REF_LAUNCH();
                const double* sin_ = proj_input(esw.p, NK, Fe, qf, quant, p + "eproj");
                linear(ey.p, sin_, off(p + "eproj"), NK, Fe, C, qf, nullptr, wsel.p, size_t(C) * Fe);
                k_combine<<<grid_for(size_t(N) * C), kThreads>>>(tmp.p, ey.p, gate.p, N, C, K);
                REF_LAUNCH();
            }
            k_add<<<grid_for(size_t(N) * C), kThreads>>>(h.p, resid.p, tmp.p, size_t(N) * C);
            REF_LAUNCH();
        }
        k_rmsnorm<<<grid_for(N), kThreads>>>(ln.p, h.p, params.p + off("lnf"), N, C);
        REF_LAUNCH();
        logits.need(size_t(N) * V);
        linear(logits.p, ln.p, off("wte"), N, C, V, false);  // tied head
        if (!targets_h) return res;
        tgt.up(targets_h, N);
        probs.need(size_t(N) * V);
        losses.need(N);
        k_softmax_ce<<<grid_for(N), kThreads>>>(probs.p, losses.p, logits.p, tgt.p, N, V);
        REF_LAUNCH();
        std::vector<double> loss_h(N);
        losses.down(loss_h.data(), N);
        int valid = 0;
        double total = 0;
        for (int n = 0; n < N; ++n)
            if (targets_h[n] >= 0) total += loss_h[n], ++valid;
        res.ce = valid ? total / valid : 0.0;
        res.loss = res.ce;
        if (alpha <= 0 || valid == 0) return res;
        neg.need(size_t(N) * Tn);
        cnt.need(N);
        text.need(size_t(B) * Tn);
        runs.need(size_t(B) * Tn + size_t(B) * Tn * ngram);
        k_negatives<<<grid_for(B), kThreads>>>(neg.p, cnt.p, tok.p, tgt.p, B, Tn, ngram, text.p, runs.p);
        REF_LAUNCH();
        ul_rows.need(N);
        k_ul_rows<<<grid_for(N), kThreads>>>(ul_rows.p, probs.p, neg.p, cnt.p, N, Tn, V);
        REF_LAUNCH();
        std::vector<double> ul_h(N);
        std::vector<int> cnt_h(N);
        ul_rows.down(ul_h.data(), N);
        cnt.down(cnt_h.data(), N);
        double ul = 0;
        for (int n = 0; n < N; ++n) {
            ul += ul_h[n];
            res.fired += cnt_h[n] > 0;
            res.negatives += size_t(cnt_h[n]);
        }
        res.ul = ul / valid;
        res.loss = res.ce + alpha * res.ul;
        return res;
    }
};

GpuReference::GpuReference(const Config& cfg, const std::vector<ParamInfo>& infos)
    : impl_(std::make_unique<Impl>(cfg, infos)) {}
GpuReference::~GpuReference() = default;
size_t GpuReference::num_params() const { return impl_->n_params; }
void GpuReference::set_params(const std::vector<double>& p) {
    CHECK(p.size() == impl_->n_params, "reference: %zu params for %zu", p.size(), impl_->n_params);
    impl_->params.up(p.data(), p.size());
}
void GpuReference::set_params(const float* p) {
    set_params(std::vector<double>(p, p + impl_->n_params));
}
void GpuReference::set_router_bias(const std::vector<std::vector<float>>& bias) {
    const int E = std::max(1, impl_->cfg.n_experts);
    std::vector<double> flat;
    for (const auto& b : bias) {
        CHECK(int(b.size()) == E, "reference: router bias for %zu experts", b.size());
        flat.insert(flat.end(), b.begin(), b.end());
    }
    CHECK(flat.size() == size_t(impl_->cfg.n_layers) * E, "reference: router bias for %zu layers", bias.size());
    impl_->bias.up(flat.data(), flat.size());
}
GpuReference::Result GpuReference::forward(const int* tokens, const int* targets, int B, int Tn, double ul_alpha,
                                           int ul_ngram, Quant quant) {
    return impl_->forward(tokens, targets, B, Tn, ul_alpha, ul_ngram, quant);
}
std::vector<double> GpuReference::logits() const {
    std::vector<double> out(size_t(impl_->N_last) * impl_->cfg.vocab);
    impl_->logits.down(out.data(), out.size());
    return out;
}

// ---------------------------------------------------------------------------
// The optimizer
// ---------------------------------------------------------------------------

void reference_optimizer_step(const ReferenceOptimizer& opt, const std::vector<ParamInfo>& infos, double lr, long step,
                              std::vector<double>& params, const std::vector<double>& grads, std::vector<double>& m,
                              std::vector<double>& v) {
    const size_t n = params.size();
    CHECK(grads.size() == n && m.size() == n && v.size() == n, "reference optimizer: sizes differ");
    Buf<double> P, G, Mm, Vv;
    P.up(params.data(), n);
    G.up(grads.data(), n);
    Mm.up(m.data(), n);
    Vv.up(v.data(), n);
    const double c1 = 1.0 / (1.0 - std::pow(opt.beta1, double(step))), c2 = 1.0 / (1.0 - std::pow(opt.beta2, double(step)));
    for (const auto& info : infos) {
        const double wd = info.decay ? opt.weight_decay : 0.0;
        if (!(opt.muon && muon_matrix(info))) {
            k_adamw<<<grid_for(info.size), kThreads>>>(P.p + info.offset, G.p + info.offset, Mm.p + info.offset,
                                                       Vv.p + info.offset, info.size, lr, opt.beta1, opt.beta2, c1, c2,
                                                       opt.eps, wd);
            REF_LAUNCH();
            continue;
        }
        // Muon, block by block: every expert copy, every fused part.
        const int parts = muon_parts(info), r = info.rows / parts, c = info.cols;
        const bool tr = r > c;
        const int mr = tr ? c : r, nc = tr ? r : c;
        const size_t per_copy = info.size / info.count, rc = size_t(r) * c;
        Buf<double> U, X, A, Bm, Y;
        for (int copy = 0; copy < info.count; ++copy)
            for (int k = 0; k < parts; ++k) {
                const size_t o = info.offset + copy * per_copy + size_t(k) * rc;
                U.need(rc);
                X.need(rc);
                k_muon_mom<<<grid_for(rc), kThreads>>>(U.p, Mm.p + o, G.p + o, rc, opt.muon_momentum);
                REF_LAUNCH();
                if (tr) k_transpose<<<grid_for(rc), kThreads>>>(X.p, U.p, r, c);
                else REF_CHECK(cudaMemcpy(X.p, U.p, rc * sizeof(double), cudaMemcpyDeviceToDevice));
                REF_LAUNCH();
                // Newton–Schulz on X [mr, nc]: X / (||X|| + 1e-7), then
                // A = X Xᵀ, B = b A + c A², X = a X + B X, five times.
                std::vector<double> xh(rc);
                X.down(xh.data(), rc);
                double ss = 0;
                for (double xv : xh) ss += xv * xv;
                k_scale<<<grid_for(rc), kThreads>>>(X.p, 1.0 / (std::sqrt(ss) + 1e-7), rc);
                REF_LAUNCH();
                const double a = 3.4445, b = -4.7750, cc = 2.0315;
                Buf<double> XT;
                A.need(size_t(mr) * mr);
                Bm.need(size_t(mr) * mr);
                Y.need(rc);
                XT.need(rc);
                for (int s = 0; s < opt.muon_steps; ++s) {
                    k_transpose<<<grid_for(rc), kThreads>>>(XT.p, X.p, mr, nc);
                    k_matmul<<<grid_for(size_t(mr) * mr), kThreads>>>(A.p, X.p, XT.p, mr, mr, nc);
                    k_matmul<<<grid_for(size_t(mr) * mr), kThreads>>>(Bm.p, A.p, A.p, mr, mr, mr);
                    k_axpby<<<grid_for(size_t(mr) * mr), kThreads>>>(Bm.p, b, A.p, cc, Bm.p, size_t(mr) * mr);
                    k_matmul<<<grid_for(rc), kThreads>>>(Y.p, Bm.p, X.p, mr, nc, mr);
                    k_axpby<<<grid_for(rc), kThreads>>>(X.p, a, X.p, 1.0, Y.p, rc);
                    REF_LAUNCH();
                }
                if (tr) {
                    k_transpose<<<grid_for(rc), kThreads>>>(U.p, X.p, mr, nc);  // back to [r, c]
                    REF_LAUNCH();
                } else {
                    REF_CHECK(cudaMemcpy(U.p, X.p, rc * sizeof(double), cudaMemcpyDeviceToDevice));
                }
                k_muon_apply<<<grid_for(rc), kThreads>>>(P.p + o, U.p, rc, lr, 0.2 * std::sqrt(double(std::max(r, c))), wd);
                REF_LAUNCH();
            }
    }
    P.down(params.data(), n);
    Mm.down(m.data(), n);
    Vv.down(v.data(), n);
}

// ---------------------------------------------------------------------------
// Attention, ternary codes, sums
// ---------------------------------------------------------------------------

void reference_attention(const std::vector<double>& qkv, const std::vector<double>& dout, std::vector<double>& out,
                         std::vector<double>& lse, std::vector<double>& dqkv, int B, int Tn, int NH, int HS) {
    const int C = NH * HS;
    const size_t N = size_t(B) * Tn;
    Buf<double> q, d, o, l, dq;
    q.up(qkv.data(), N * 3 * C);
    d.up(dout.data(), N * C);
    dq.up(dqkv.data(), N * 3 * C);
    o.need(N * C);
    l.need(N * NH);
    Buf<double> st;
    st.need(N * NH * 3);
    k_attention<<<grid_for(N * NH), kThreads>>>(o.p, l.p, q.p, B, Tn, C, NH);
    k_attention_stats<<<grid_for(N * NH), kThreads>>>(st.p, d.p, q.p, B, Tn, C, NH);
    k_attention_bwd<<<grid_for(N * NH), kThreads>>>(dq.p, d.p, q.p, st.p, B, Tn, C, NH);
    REF_LAUNCH();
    out.resize(N * C);
    lse.resize(N * NH);
    o.down(out.data(), out.size());
    l.down(lse.data(), lse.size());
    dq.down(dqkv.data(), dqkv.size());
}

void reference_ternary(const float* w, float* wq, size_t n) {
    Buf<float> a, b;
    a.up(w, n);
    b.need(n);
    k_ternary_float<<<1, 1>>>(b.p, a.p, n);
    REF_LAUNCH();
    b.down(wq, n);
}

double reference_sum_squares(const std::vector<float>& x) {
    Buf<float> a;
    Buf<double> s;
    a.up(x.data(), x.size());
    s.need(1);
    k_sum_squares<<<1, 1>>>(s.p, a.p, x.size());
    REF_LAUNCH();
    double out = 0;
    s.down(&out, 1);
    return out;
}
