// gpu.cu — the CUDA implementation of GpuTrainer (see gpu.h).
//
// The model is Model<float>'s, op for op; each kernel names the CPU
// function it mirrors (ops.h, model.h), and tests/test_gpu.cpp checks them
// against each other. Row-wise operations (norms, softmaxes, quantizers)
// run one warp per row; elementwise ones one thread per element. Matrix
// products go to cuBLAS. Everything runs in order on the default stream.
// The host waits on the device twice per forward pass: for the losses, and
// once per MoE layer for the expert counts, since cuBLAS needs each
// expert's row count on the host.
#include "gpu.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        const cudaError_t err_ = (call);                                                       \
        if (err_ != cudaSuccess) {                                                             \
            std::fprintf(stderr, "%s:%d: CUDA error: %s\n", __FILE__, __LINE__,                \
                         cudaGetErrorString(err_));                                            \
            std::exit(1);                                                                      \
        }                                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                                     \
    do {                                                                                       \
        const cublasStatus_t st_ = (call);                                                     \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                                    \
            std::fprintf(stderr, "%s:%d: cuBLAS error: %s\n", __FILE__, __LINE__,              \
                         cublasGetStatusString(st_));                                          \
            std::exit(1);                                                                      \
        }                                                                                      \
    } while (0)

// After every kernel launch: reports bad launch configurations at once.
#define LAUNCH_CHECK() CUDA_CHECK(cudaGetLastError())

namespace {

constexpr int kWarp = 32;
constexpr int kBlock = 256;                     // threads per block
constexpr int kRowsPerBlock = kBlock / kWarp;   // for one-warp-per-row kernels
constexpr int kPartials = 64;                   // blocks in a two-stage reduction

int blocks_for(size_t n, int per_block = kBlock) { return int((n + per_block - 1) / per_block); }

size_t g_device_bytes = 0;  // held by all DevBufs

// A device array. resize() does not keep the contents.
template <class T>
struct DevBuf {
    T* p = nullptr;
    size_t n = 0;

    DevBuf() = default;
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    DevBuf(DevBuf&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr, o.n = 0; }
    DevBuf& operator=(DevBuf&& o) noexcept {
        std::swap(p, o.p);
        std::swap(n, o.n);
        return *this;
    }
    ~DevBuf() { release(); }

    void release() {
        if (p) {
            cudaFree(p);
            g_device_bytes -= n * sizeof(T);
        }
        p = nullptr;
        n = 0;
    }
    void resize(size_t count) {
        if (count <= n && p) return;
        release();
        CUDA_CHECK(cudaMalloc(&p, std::max<size_t>(count, 1) * sizeof(T)));
        n = std::max<size_t>(count, 1);
        g_device_bytes += n * sizeof(T);
    }
    void zero(size_t count) { CUDA_CHECK(cudaMemsetAsync(p, 0, count * sizeof(T))); }
    void zero() { zero(n); }
};

template <class T>
void to_device(T* dst, const T* src, size_t n) {
    CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyHostToDevice));
}
template <class T>
void to_host(T* dst, const T* src, size_t n) {
    CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyDeviceToHost));
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

template <class T>
__device__ T warp_sum(T v) {
    for (int o = kWarp / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ float warp_max(float v) {
    for (int o = kWarp / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// Sum (or max) over the block; every thread gets the result. sh holds at
// least kWarp + 1 values.
template <class T>
__device__ T block_reduce(T v, T* sh, bool is_max) {
    const int lane = threadIdx.x % kWarp, wid = threadIdx.x / kWarp, nw = blockDim.x / kWarp;
    v = is_max ? T(warp_max(float(v))) : warp_sum(v);
    __syncthreads();  // sh may still be in use by a previous call
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    if (wid == 0) {
        T w = lane < nw ? sh[lane] : (is_max ? T(-INFINITY) : T(0));
        w = is_max ? T(warp_max(float(w))) : warp_sum(w);
        if (lane == 0) sh[kWarp] = w;
    }
    __syncthreads();
    return sh[kWarp];
}

// partial[blockIdx.x] = Σ x² (or Σ |x| with abs_only) over a grid-stride
// slice of x[0, n), in double. blockIdx.y selects one of several arrays:
// x = base + off[y], n = len[y], partials at partial + y * gridDim.x.
__global__ void partial_sums(double* partial, const float* base, const size_t* off, const size_t* len,
                             bool abs_only) {
    __shared__ double sh[kWarp + 1];
    const float* x = base + off[blockIdx.y];
    const size_t n = len[blockIdx.y];
    double s = 0.0;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        s += abs_only ? fabs(double(x[i])) : double(x[i]) * double(x[i]);
    s = block_reduce(s, sh, false);
    if (threadIdx.x == 0) partial[size_t(blockIdx.y) * gridDim.x + blockIdx.x] = s;
}

// ---------------------------------------------------------------------------
// Elementwise and row-wise kernels (ops.h)
// ---------------------------------------------------------------------------

// embedding_forward: out[n] = wte[tokens[n]]
__global__ void embedding_fwd(float* out, const int* tokens, const float* wte, int N, int C) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(N) * C) return;
    const size_t n = i / C, c = i % C;
    out[i] = wte[size_t(tokens[n]) * C + c];
}

// embedding_backward: dwte[tokens[n]] += dout[n]
__global__ void embedding_bwd(float* dwte, const float* dout, const int* tokens, int N, int C) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(N) * C) return;
    const size_t n = i / C, c = i % C;
    atomicAdd(&dwte[size_t(tokens[n]) * C + c], dout[i]);
}

// rmsnorm_forward, one warp per row.
__global__ void rmsnorm_fwd(float* out, float* rstd, const float* x, const float* w, int N, int C) {
    const int row = blockIdx.x * kRowsPerBlock + threadIdx.x / kWarp, lane = threadIdx.x % kWarp;
    if (row >= N) return;
    const float* xn = x + size_t(row) * C;
    float ss = 0.0f;
    for (int c = lane; c < C; c += kWarp) ss += xn[c] * xn[c];
    ss = warp_sum(ss);
    const float r = 1.0f / sqrtf(ss / float(C) + 1e-5f);
    if (lane == 0) rstd[row] = r;
    float* o = out + size_t(row) * C;
    for (int c = lane; c < C; c += kWarp) o[c] = xn[c] * r * w[c];
}

// rmsnorm_backward, the dx part: dx_i += r g_i w_i - r³/C x_i Σ_j g_j w_j x_j.
__global__ void rmsnorm_bwd_dx(float* dx, const float* dout, const float* x, const float* w, const float* rstd,
                               int N, int C) {
    const int row = blockIdx.x * kRowsPerBlock + threadIdx.x / kWarp, lane = threadIdx.x % kWarp;
    if (row >= N) return;
    const float* xn = x + size_t(row) * C;
    const float* dn = dout + size_t(row) * C;
    float gx = 0.0f;
    for (int c = lane; c < C; c += kWarp) gx += dn[c] * w[c] * xn[c];
    gx = warp_sum(gx);
    const float r = rstd[row], k = r * r * r * gx / float(C);
    float* dxn = dx + size_t(row) * C;
    for (int c = lane; c < C; c += kWarp) dxn[c] += r * dn[c] * w[c] - k * xn[c];
}

// rmsnorm_backward, the dw part, stage 1: block b sums rows [b*R, (b+1)*R).
__global__ void rmsnorm_bwd_dw_partial(float* partial, const float* dout, const float* x, const float* rstd,
                                       int N, int C, int R) {
    const int n0 = blockIdx.x * R, n1 = min(N, n0 + R);
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float acc = 0.0f;
        for (int n = n0; n < n1; ++n) acc += dout[size_t(n) * C + c] * x[size_t(n) * C + c] * rstd[n];
        partial[size_t(blockIdx.x) * C + c] = acc;
    }
}

// Stage 2: dw[c] += Σ_b partial[b][c].
__global__ void add_partials(float* dw, const float* partial, int blocks, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float s = dw[c];
    for (int b = 0; b < blocks; ++b) s += partial[size_t(b) * C + c];
    dw[c] = s;
}

// rope_apply on q and k of qkv[N, 3C]; row n is at position n % Tn.
__global__ void rope(float* qkv, const float* cos_t, const float* sin_t, int N, int Tn, int C, int NH, int half,
                     bool inverse) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(N) * 2 * NH * half) return;
    const int k = int(i % half);
    size_t r = i / half;
    const int h = int(r % NH);
    r /= NH;
    const int part = int(r % 2);
    const int n = int(r / 2), t = n % Tn;
    float* x = qkv + size_t(n) * 3 * C + part * C + h * 2 * half;
    const float cs = cos_t[size_t(t) * half + k];
    const float sn = inverse ? -sin_t[size_t(t) * half + k] : sin_t[size_t(t) * half + k];
    const float x0 = x[k], x1 = x[k + half];
    x[k] = x0 * cs - x1 * sn;
    x[k + half] = x0 * sn + x1 * cs;
}

// attention_forward's softmax: each row of att[B, NH, Tn, Tn] holds raw
// scores; becomes softmax(scale * scores) over s <= t, and 0 above.
__global__ void attn_softmax_fwd(float* att, int rows, int Tn, float scale) {
    const int row = blockIdx.x * kRowsPerBlock + threadIdx.x / kWarp, lane = threadIdx.x % kWarp;
    if (row >= rows) return;
    const int t = row % Tn;
    float* p = att + size_t(row) * Tn;
    float mx = -INFINITY;
    for (int s = lane; s <= t; s += kWarp) mx = fmaxf(mx, p[s]);
    mx = warp_max(mx);
    float sum = 0.0f;
    for (int s = lane; s <= t; s += kWarp) {
        const float e = expf((p[s] - mx) * scale);
        p[s] = e;
        sum += e;
    }
    sum = warp_sum(sum);
    const float inv = 1.0f / sum;
    for (int s = lane; s < Tn; s += kWarp) p[s] = s <= t ? p[s] * inv : 0.0f;
}

// attention_backward's softmax part: dS = P ⊙ (dP - Σ P ⊙ dP) * scale.
__global__ void attn_softmax_bwd(float* dS, const float* att, int rows, int Tn, float scale) {
    const int row = blockIdx.x * kRowsPerBlock + threadIdx.x / kWarp, lane = threadIdx.x % kWarp;
    if (row >= rows) return;
    const int t = row % Tn;
    const float* p = att + size_t(row) * Tn;
    float* d = dS + size_t(row) * Tn;
    float pd = 0.0f;
    for (int s = lane; s <= t; s += kWarp) pd += p[s] * d[s];
    pd = warp_sum(pd);
    for (int s = lane; s < Tn; s += kWarp) d[s] = s <= t ? p[s] * (d[s] - pd) * scale : 0.0f;
}

// swiglu_forward: inp[N, 2F] = [gate | up], out = silu(gate) * up.
__global__ void swiglu_fwd(float* out, const float* inp, int N, int F) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(N) * F) return;
    const size_t n = i / F, j = i % F;
    const float g = inp[n * 2 * F + j], u = inp[n * 2 * F + F + j];
    out[i] = g / (1.0f + expf(-g)) * u;
}

// swiglu_backward (accumulates into dinp).
__global__ void swiglu_bwd(float* dinp, const float* dout, const float* inp, int N, int F) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(N) * F) return;
    const size_t n = i / F, j = i % F;
    const float g = inp[n * 2 * F + j], u = inp[n * 2 * F + F + j], d = dout[i];
    const float sg = 1.0f / (1.0f + expf(-g));
    dinp[n * 2 * F + j] += d * u * sg * (1.0f + g * (1.0f - sg));
    dinp[n * 2 * F + F + j] += d * g * sg;
}

__global__ void add_fwd(float* out, const float* a, const float* b, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

// cross_entropy_forward, one block per row: logits become probabilities.
__global__ void xent_fwd(float* probs, float* losses, const int* targets, int V) {
    __shared__ float sh[kWarp + 1];
    const int n = blockIdx.x;
    float* p = probs + size_t(n) * V;
    float mx = -INFINITY;
    for (int i = threadIdx.x; i < V; i += blockDim.x) mx = fmaxf(mx, p[i]);
    mx = block_reduce(mx, sh, true);
    float sum = 0.0f;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        const float e = expf(p[i] - mx);
        p[i] = e;
        sum += e;
    }
    sum = block_reduce(sum, sh, false);
    const float inv = 1.0f / sum;
    for (int i = threadIdx.x; i < V; i += blockDim.x) p[i] *= inv;
    __syncthreads();  // p[target] may have been written by another thread
    if (threadIdx.x == 0) {
        const int t = targets[n];
        losses[n] = t < 0 ? 0.0f : -logf(fmaxf(p[t], FLT_MIN));
    }
}

// cross_entropy_backward, in place: d = (p - onehot(target)) * dloss.
__global__ void xent_bwd(float* d, const int* targets, int N, int V, float dloss) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(N) * V) return;
    const int n = int(i / V), v = int(i % V), t = targets[n];
    if (t < 0) {
        d[i] = 0.0f;
        return;
    }
    d[i] = v == t ? d[i] * dloss - dloss : d[i] * dloss;
}

// ternary_quantize, stage 2: gamma[y] = mean |w| + 1e-8 from the partials.
__global__ void ternary_gamma(float* gamma, const double* partial, const size_t* len, int parts) {
    if (threadIdx.x != 0) return;
    double s = 0.0;
    for (int b = 0; b < parts; ++b) s += partial[size_t(blockIdx.x) * parts + b];
    gamma[blockIdx.x] = float(s / double(len[blockIdx.x])) + 1e-8f;
}

// ternary_quantize, stage 3: wq = clamp(round(w / gamma), -1, 1) * gamma.
__global__ void ternary_apply(float* qbase, const float* base, const size_t* off, const size_t* len,
                              const float* gamma) {
    const float* w = base + off[blockIdx.y];
    float* wq = qbase + off[blockIdx.y];
    const float g = gamma[blockIdx.y];
    const size_t n = len[blockIdx.y];
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        wq[i] = fminf(fmaxf(rintf(w[i] / g), -1.0f), 1.0f) * g;
}

// act_quantize, one warp per row: x -> round(x / scale) * scale, scale = max|x| / 127.
__global__ void act_quant(float* xq, const float* x, int N, int C) {
    const int row = blockIdx.x * kRowsPerBlock + threadIdx.x / kWarp, lane = threadIdx.x % kWarp;
    if (row >= N) return;
    const float* xn = x + size_t(row) * C;
    float* q = xq + size_t(row) * C;
    float amax = 0.0f;
    for (int c = lane; c < C; c += kWarp) amax = fmaxf(amax, fabsf(xn[c]));
    amax = warp_max(amax);
    if (amax == 0.0f) {
        for (int c = lane; c < C; c += kWarp) q[c] = 0.0f;
        return;
    }
    const float scale = amax / 127.0f, inv = 1.0f / scale;
    for (int c = lane; c < C; c += kWarp) q[c] = fminf(fmaxf(rintf(xn[c] * inv), -127.0f), 127.0f) * scale;
}

// AdamW::update for one tensor, with the gradient scaled by gscale (clipping).
__global__ void adamw(float* p, const float* g, float* m, float* v, size_t n, float lr, float b1, float b2,
                      float c1, float c2, float eps, float wd, float gscale) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        const float gi = g[i] * gscale;
        m[i] = b1 * m[i] + (1.0f - b1) * gi;
        v[i] = b2 * v[i] + (1.0f - b2) * gi * gi;
        const float upd = (m[i] * c1) / (sqrtf(v[i] * c2) + eps);
        p[i] -= lr * (upd + wd * p[i]);
    }
}

// Muon (AdamW::muon_update), for `batch` blocks of r x c at stride r * c:
// Nesterov momentum into mom, and the update copied to X in the
// orientation Newton–Schulz works on (transposed when r > c).
__global__ void muon_momentum(float* X, float* mom, const float* g, int batch, int r, int c, float mu,
                              float gscale) {
    const size_t rc = size_t(r) * c, n = rc * batch;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        const float gi = g[i] * gscale;
        mom[i] = mu * mom[i] + (1.0f - mu) * gi;
        const size_t k = i % rc, row = k / c, col = k % c;
        X[i - k + (r > c ? col * r + row : k)] = (1.0f - mu) * gi + mu * mom[i];
    }
}

// newton_schulz's first step: each len-long matrix of X divided by its
// Frobenius norm (+ 1e-7). One thread block per matrix.
__global__ void frobenius_normalize(float* X, size_t len) {
    __shared__ double sh[kWarp + 1];
    float* x = X + blockIdx.x * len;
    double s = 0.0;
    for (size_t i = threadIdx.x; i < len; i += blockDim.x) s += double(x[i]) * double(x[i]);
    s = block_reduce(s, sh, false);
    const float inv = float(1.0 / (sqrt(s) + 1e-7));
    for (size_t i = threadIdx.x; i < len; i += blockDim.x) x[i] *= inv;
}

// out = a x + b y, elementwise; out may be x or y.
__global__ void axpby(float* out, float a, const float* x, float b, const float* y, size_t n) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        out[i] = a * x[i] + b * y[i];
}

// The Muon step itself: p -= lr (scale X + wd p), X read back in p's layout.
__global__ void muon_apply(float* p, const float* X, int batch, int r, int c, float lr, float scale, float wd) {
    const size_t rc = size_t(r) * c, n = rc * batch;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        const size_t k = i % rc, row = k / c, col = k % c;
        p[i] -= lr * (scale * X[i - k + (r > c ? col * r + row : k)] + wd * p[i]);
    }
}

// ---------------------------------------------------------------------------
// Mixture-of-Experts kernels (Model::moe_forward / moe_backward)
// ---------------------------------------------------------------------------

// One thread per token: top-k experts by biased logit, gates = softmax of
// the chosen unbiased logits, and per-expert counts.
__global__ void moe_route(int* sel, float* gate, int* counts, const float* rlogits, const float* bias, int N,
                          int E, int K) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float* r = rlogits + size_t(n) * E;
    int* s = sel + size_t(n) * K;
    for (int i = 0; i < K; ++i) {
        int best = -1;
        for (int e = 0; e < E; ++e) {
            bool taken = false;
            for (int j = 0; j < i; ++j) taken |= s[j] == e;
            if (!taken && (best < 0 || r[e] + bias[e] > r[best] + bias[best])) best = e;
        }
        s[i] = best;
        atomicAdd(&counts[best], 1);
    }
    float mx = r[s[0]];
    for (int i = 1; i < K; ++i) mx = fmaxf(mx, r[s[i]]);
    float sum = 0.0f;
    float* g = gate + size_t(n) * K;
    for (int i = 0; i < K; ++i) sum += g[i] = expf(r[s[i]] - mx);
    for (int i = 0; i < K; ++i) g[i] /= sum;
}

// Groups the N*K assignments by expert, keeping assignment order within an
// expert (the CPU's counting sort). One block per expert scans all
// assignments in chunks; a warp ballot plus per-warp totals give each
// selected assignment its rank.
__global__ void moe_order(int* order, int* row_of, const int* sel, const int* expert_start, int NK) {
    __shared__ int warp_total[kWarp];
    __shared__ int base;
    const int e = blockIdx.x, lane = threadIdx.x % kWarp, wid = threadIdx.x / kWarp;
    const int nw = blockDim.x / kWarp;
    if (threadIdx.x == 0) base = expert_start[e];
    __syncthreads();
    for (int j0 = 0; j0 < NK; j0 += blockDim.x) {
        const int j = j0 + threadIdx.x;
        const bool mine = j < NK && sel[j] == e;
        const unsigned mask = __ballot_sync(0xffffffffu, mine);
        if (lane == 0) warp_total[wid] = __popc(mask);
        __syncthreads();
        int before = 0, total = 0;
        for (int w = 0; w < nw; ++w) {
            before += w < wid ? warp_total[w] : 0;
            total += warp_total[w];
        }
        if (mine) {
            const int row = base + before + __popc(mask & ((1u << lane) - 1u));
            order[row] = j;
            row_of[j] = row;
        }
        __syncthreads();  // everyone has read base and warp_total
        if (threadIdx.x == 0) base += total;
        __syncthreads();
    }
}

// xg[r] = x[token of assignment order[r]]
__global__ void moe_gather(float* xg, const float* x, const int* order, int NK, int C, int K) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(NK) * C) return;
    const size_t r = i / C, c = i % C;
    xg[i] = x[size_t(order[r] / K) * C + c];
}

// out[n] += Σ_i gate[n, i] * ey[row_of[n, i]]
__global__ void moe_combine(float* out, const float* ey, const float* gate, const int* row_of, int N, int C,
                            int K) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(N) * C) return;
    const size_t n = i / C, c = i % C;
    float acc = out[i];
    for (int k = 0; k < K; ++k) acc += gate[n * K + k] * ey[size_t(row_of[n * K + k]) * C + c];
    out[i] = acc;
}

// One warp per assignment j: dey[row] = gate * dout[n], dgate[j] = <dout[n], ey[row]>.
__global__ void moe_bwd_combine(float* dey, float* dgate, const float* dout, const float* ey, const float* gate,
                                const int* row_of, int NK, int C, int K) {
    const int j = blockIdx.x * kRowsPerBlock + threadIdx.x / kWarp, lane = threadIdx.x % kWarp;
    if (j >= NK) return;
    const size_t n = size_t(j / K), row = size_t(row_of[j]);
    const float g = gate[j];
    float dot = 0.0f;
    for (int c = lane; c < C; c += kWarp) {
        const float d = dout[n * C + c];
        dey[row * C + c] = g * d;
        dot += d * ey[row * C + c];
    }
    dot = warp_sum(dot);
    if (lane == 0) dgate[j] = dot;
}

// dx[n] += Σ_i dxg[row_of[n, i]]
__global__ void moe_scatter(float* dx, const float* dxg, const int* row_of, int N, int C, int K) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(N) * C) return;
    const size_t n = i / C, c = i % C;
    float acc = dx[i];
    for (int k = 0; k < K; ++k) acc += dxg[size_t(row_of[n * K + k]) * C + c];
    dx[i] = acc;
}

// Router logits' gradient through the gate softmax: d r_s = g_s (dg_s - Σ g dg).
// drl must be zero beforehand (only the chosen experts get a gradient).
__global__ void moe_router_grad(float* drl, const float* gate, const float* dgate, const int* sel, int N, int E,
                                int K) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float* g = gate + size_t(n) * K;
    const float* dg = dgate + size_t(n) * K;
    float s = 0.0f;
    for (int i = 0; i < K; ++i) s += g[i] * dg[i];
    for (int i = 0; i < K; ++i) drl[size_t(n) * E + sel[size_t(n) * K + i]] = g[i] * (dg[i] - s);
}

}  // namespace

// ---------------------------------------------------------------------------
// GpuTrainer
// ---------------------------------------------------------------------------

struct GpuTrainer::Impl {
    Config cfg;
    std::vector<ParamInfo> infos;
    size_t n_params = 0;
    cublasHandle_t blas = nullptr;

    DevBuf<float> params, grads, qparams, m, v;
    long adam_step = 0;
    // v, AdamW's second moment, has room only for the tensors AdamW updates
    // (with Muon on, its matrices need none: about 1 GB at 250 M
    // parameters). v_at[i] is where tensor i's part starts, or kNoV.
    static constexpr size_t kNoV = ~size_t(0);
    std::vector<size_t> v_at;
    // Muon (set_muon): settings and Newton–Schulz work space.
    bool muon = false;
    float muon_mu = 0.95f;
    int muon_ns = 5;
    DevBuf<float> ns_x, ns_y, ns_a, ns_b;
    bool weights_dirty = true;

    // Ternary tensors, one segment per (tensor, expert copy).
    int n_seg = 0;
    std::vector<size_t> seg_len_h;
    DevBuf<size_t> seg_off, seg_len;
    DevBuf<double> seg_partial;
    DevBuf<float> seg_gamma;
    // All parameters as one segment, for the gradient norm.
    DevBuf<size_t> all_off, all_len;
    DevBuf<double> norm_partial;

    struct LayerOffsets {
        size_t ln1 = 0, wqkv = 0, wo = 0, ln2 = 0, wfc = 0, wproj = 0, router = 0, efc = 0, eproj = 0;
    };
    size_t wte = 0, lnf = 0;
    std::vector<LayerOffsets> layer;

    DevBuf<float> rope_cos, rope_sin;
    DevBuf<float> router_bias;  // [layer, expert]
    std::vector<std::vector<double>> expert_load;

    struct LayerActs {
        DevBuf<float> ln1, rstd1, qkv, att, atty, resid1, ln2, rstd2, resid2, fc, sw;
        DevBuf<float> rlogits, gate, xg, efc, esw, ey;
        DevBuf<int> sel, order, row_of;
        std::vector<int> expert_start;  // host copy, [E + 1]
    };
    std::vector<LayerActs> acts;
    DevBuf<float> x0, lnf_out, rstdf, logits, losses, xq, out_tmp;
    DevBuf<float> dresid, dC, dqkv, datty, dfc, dsw, dey, dxg, drl, dS, dgate, dw_partial;
    DevBuf<int> tokens, targets, counts, expert_start_d;
    int B_ = 0, T_ = 0, B_cur = 0, T_cur = 0, n_valid = 0;
    bool has_targets = false;
    std::vector<float> losses_h;
    std::vector<int> counts_h;

    Impl(const Config& c, const std::vector<ParamInfo>& in, bool tf32) : cfg(c), infos(in) {
        cfg.validate();
        CHECK(!cfg.fp_attention, "full-precision attention (--fp-attn) is not on the GPU yet");
        int devices = 0;
        const cudaError_t err = cudaGetDeviceCount(&devices);
        CHECK(err == cudaSuccess && devices > 0, "no CUDA GPU available (%s)",
              err == cudaSuccess ? "no devices" : cudaGetErrorString(err));
        for (const auto& p : infos) n_params = std::max(n_params, p.offset + p.size);
        CUBLAS_CHECK(cublasCreate(&blas));
        CUBLAS_CHECK(cublasSetMathMode(blas, tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH));
        params.resize(n_params);
        grads.resize(n_params);
        if (cfg.ternary) qparams.resize(n_params);  // read only through W(), for ternary models
        m.resize(n_params);
        grads.zero();
        m.zero();
        layout_v();

        // Offsets by name, as Model's layout builds them.
        auto off = [&](const std::string& name, bool required) -> size_t {
            for (const auto& p : infos)
                if (p.name == name) return p.offset;
            CHECK(!required, "parameter %s missing", name.c_str());
            return 0;
        };
        wte = off("wte", true);
        lnf = off("lnf", true);
        layer.resize(cfg.n_layers);
        for (int l = 0; l < cfg.n_layers; ++l) {
            const std::string p = "l" + std::to_string(l) + ".";
            auto& o = layer[l];
            o.ln1 = off(p + "ln1", true);
            o.wqkv = off(p + "wqkv", true);
            o.wo = off(p + "wo", true);
            o.ln2 = off(p + "ln2", true);
            o.wfc = off(p + "wfc", cfg.always_hidden() > 0);
            o.wproj = off(p + "wproj", cfg.always_hidden() > 0);
            o.router = off(p + "router", cfg.moe());
            o.efc = off(p + "efc", cfg.moe());
            o.eproj = off(p + "eproj", cfg.moe());
        }

        std::vector<size_t> seg_off_h;
        for (const auto& p : infos) {
            if (!p.ternary) continue;
            const size_t per = p.size / p.count;
            for (int c = 0; c < p.count; ++c) {
                seg_off_h.push_back(p.offset + c * per);
                seg_len_h.push_back(per);
            }
        }
        n_seg = int(seg_off_h.size());
        if (n_seg > 0) {
            seg_off.resize(n_seg);
            seg_len.resize(n_seg);
            seg_partial.resize(size_t(n_seg) * kPartials);
            seg_gamma.resize(n_seg);
            to_device(seg_off.p, seg_off_h.data(), n_seg);
            to_device(seg_len.p, seg_len_h.data(), n_seg);
        }
        const size_t zero = 0;
        all_off.resize(1);
        all_len.resize(1);
        norm_partial.resize(kPartials);
        to_device(all_off.p, &zero, 1);
        to_device(all_len.p, &n_params, 1);

        // RoPE tables, computed exactly as RopeTable<float> does.
        const int half = cfg.head_size() / 2;
        std::vector<float> cs(size_t(cfg.seq_len) * half), sn(cs.size());
        for (int t = 0; t < cfg.seq_len; ++t)
            for (int i = 0; i < half; ++i) {
                const double a = t * std::pow(10000.0, -2.0 * i / cfg.head_size());
                cs[size_t(t) * half + i] = float(std::cos(a));
                sn[size_t(t) * half + i] = float(std::sin(a));
            }
        rope_cos.resize(cs.size());
        rope_sin.resize(sn.size());
        to_device(rope_cos.p, cs.data(), cs.size());
        to_device(rope_sin.p, sn.data(), sn.size());

        const int E = std::max(1, cfg.n_experts);
        router_bias.resize(size_t(cfg.n_layers) * E);
        router_bias.zero();
        expert_load.assign(cfg.n_layers, std::vector<double>(E, 0.0));
        counts.resize(E);
        expert_start_d.resize(E + 1);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    ~Impl() {
        if (blas) cublasDestroy(blas);
    }

    float* P(size_t o) { return params.p + o; }
    float* G(size_t o) { return grads.p + o; }
    const float* W(size_t o) const { return cfg.ternary ? qparams.p + o : params.p + o; }

    // Row-major C[M, N] = op(A)[M, K] · op(B)[K, N] + beta * C. With ta, A is
    // stored as [K, M] (row stride lda); likewise B with tb. cuBLAS is
    // column-major, where a row-major matrix reads as its transpose, so this
    // computes Cᵀ = op(B)ᵀ · op(A)ᵀ.
    void gemm(bool ta, bool tb, int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C,
              int ldc, float beta) {
        if (M == 0 || N == 0) return;
        const float one = 1.0f;
        CUBLAS_CHECK(cublasSgemm(blas, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K,
                                 &one, B, ldb, A, lda, &beta, C, ldc));
    }

    // The same, for `batch` independent products at fixed strides.
    void gemm_batched(bool ta, bool tb, int M, int N, int K, const float* A, int lda, long long sa,
                      const float* B, int ldb, long long sb, float* C, int ldc, long long sc, float beta,
                      int batch) {
        const float one = 1.0f;
        CUBLAS_CHECK(cublasSgemmStridedBatched(blas, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N,
                                               N, M, K, &one, B, ldb, sb, A, lda, sa, &beta, C, ldc, sc, batch));
    }

    // linear_forward: y[N, OC] = x[N, IC] · w[OC, IC]ᵀ
    void linear_forward(float* y, const float* x, const float* w, int N, int IC, int OC) {
        gemm(false, true, N, OC, IC, x, IC, w, IC, y, OC, 0.0f);
    }
    // linear_backward: dx += dy · w (if dx), dw += dyᵀ · x
    void linear_backward(float* dx, float* dw, const float* dy, const float* x, const float* w, int N, int IC,
                         int OC) {
        if (dx) gemm(false, false, N, IC, OC, dy, OC, w, IC, dx, IC, 1.0f);
        gemm(true, false, OC, IC, N, dy, OC, x, IC, dw, IC, 1.0f);
    }

    void act_quantize(float* xq_, const float* x, int N, int C) {
        if (N == 0) return;
        act_quant<<<blocks_for(N, kRowsPerBlock), kBlock>>>(xq_, x, N, C);
        LAUNCH_CHECK();
    }

    // Model::proj_forward / proj_backward: BitLinear when ternary.
    void proj_forward(float* y, const float* x, size_t woff, int N, int IC, int OC) {
        if (cfg.ternary) {
            act_quantize(xq.p, x, N, IC);
            x = xq.p;
        }
        linear_forward(y, x, W(woff), N, IC, OC);
    }
    void proj_backward(float* dx, const float* dy, const float* x, size_t woff, int N, int IC, int OC) {
        if (cfg.ternary) {
            act_quantize(xq.p, x, N, IC);
            x = xq.p;
        }
        linear_backward(dx, G(woff), dy, x, W(woff), N, IC, OC);
    }

    void rmsnorm_forward(float* out, float* rstd, const float* x, const float* w, int N, int C) {
        rmsnorm_fwd<<<blocks_for(N, kRowsPerBlock), kBlock>>>(out, rstd, x, w, N, C);
        LAUNCH_CHECK();
    }
    void rmsnorm_backward(float* dx, float* dw, const float* dout, const float* x, const float* w,
                          const float* rstd, int N, int C) {
        rmsnorm_bwd_dx<<<blocks_for(N, kRowsPerBlock), kBlock>>>(dx, dout, x, w, rstd, N, C);
        LAUNCH_CHECK();
        constexpr int R = 64;
        const int nb = (N + R - 1) / R;
        rmsnorm_bwd_dw_partial<<<nb, kBlock>>>(dw_partial.p, dout, x, rstd, N, C, R);
        LAUNCH_CHECK();
        add_partials<<<blocks_for(C), kBlock>>>(dw, dw_partial.p, nb, C);
        LAUNCH_CHECK();
    }

    void rope_apply(float* qkv, int B, int Tn, bool inverse) {
        const int C = cfg.d_model, NH = cfg.n_heads, half = cfg.head_size() / 2;
        const size_t n = size_t(B) * Tn * 2 * NH * half;
        rope<<<blocks_for(n), kBlock>>>(qkv, rope_cos.p, rope_sin.p, B * Tn, Tn, C, NH, half, inverse);
        LAUNCH_CHECK();
    }

    void add(float* out, const float* a, const float* b, size_t n) {
        add_fwd<<<blocks_for(n), kBlock>>>(out, a, b, n);
        LAUNCH_CHECK();
    }

    // attention_forward: per sequence, the heads as one strided batch.
    void attention_forward(float* out, float* att, const float* qkv, int B, int Tn) {
        const int C = cfg.d_model, NH = cfg.n_heads, HS = cfg.head_size();
        const long long TT = (long long)Tn * Tn;
        for (int b = 0; b < B; ++b) {
            const float* q = qkv + size_t(b) * Tn * 3 * C;
            float* S = att + size_t(b) * NH * TT;
            // S = Q · Kᵀ for every head
            gemm_batched(false, true, Tn, Tn, HS, q, 3 * C, HS, q + C, 3 * C, HS, S, Tn, TT, 0.0f, NH);
        }
        const int rows = B * NH * Tn;
        attn_softmax_fwd<<<blocks_for(rows, kRowsPerBlock), kBlock>>>(att, rows, Tn,
                                                                      1.0f / std::sqrt(float(HS)));
        LAUNCH_CHECK();
        for (int b = 0; b < B; ++b) {
            const float* vv = qkv + size_t(b) * Tn * 3 * C + 2 * C;
            // out = P · V
            gemm_batched(false, false, Tn, HS, Tn, att + size_t(b) * NH * TT, Tn, TT, vv, 3 * C, HS,
                         out + size_t(b) * Tn * C, C, HS, 0.0f, NH);
        }
    }

    // attention_backward (accumulates into dqkv).
    void attention_backward(float* dqkv_, const float* dout, const float* att, const float* qkv, int B, int Tn) {
        const int C = cfg.d_model, NH = cfg.n_heads, HS = cfg.head_size();
        const long long TT = (long long)Tn * Tn;
        for (int b = 0; b < B; ++b) {
            const float* vv = qkv + size_t(b) * Tn * 3 * C + 2 * C;
            // dP = dO · Vᵀ
            gemm_batched(false, true, Tn, Tn, HS, dout + size_t(b) * Tn * C, C, HS, vv, 3 * C, HS,
                         dS.p + size_t(b) * NH * TT, Tn, TT, 0.0f, NH);
        }
        const int rows = B * NH * Tn;
        attn_softmax_bwd<<<blocks_for(rows, kRowsPerBlock), kBlock>>>(dS.p, att, rows, Tn,
                                                                      1.0f / std::sqrt(float(HS)));
        LAUNCH_CHECK();
        for (int b = 0; b < B; ++b) {
            const float* q = qkv + size_t(b) * Tn * 3 * C;
            float* dq = dqkv_ + size_t(b) * Tn * 3 * C;
            const float* dSb = dS.p + size_t(b) * NH * TT;
            const float* Pb = att + size_t(b) * NH * TT;
            const float* dOb = dout + size_t(b) * Tn * C;
            // dQ += dS · K,  dK += dSᵀ · Q,  dV += Pᵀ · dO
            gemm_batched(false, false, Tn, HS, Tn, dSb, Tn, TT, q + C, 3 * C, HS, dq, 3 * C, HS, 1.0f, NH);
            gemm_batched(true, false, Tn, HS, Tn, dSb, Tn, TT, q, 3 * C, HS, dq + C, 3 * C, HS, 1.0f, NH);
            gemm_batched(true, false, Tn, HS, Tn, Pb, Tn, TT, dOb, C, HS, dq + 2 * C, 3 * C, HS, 1.0f, NH);
        }
    }

    void swiglu_forward(float* out, const float* inp, int N, int F) {
        if (N == 0) return;
        swiglu_fwd<<<blocks_for(size_t(N) * F), kBlock>>>(out, inp, N, F);
        LAUNCH_CHECK();
    }
    void swiglu_backward(float* dinp, const float* dout, const float* inp, int N, int F) {
        if (N == 0) return;
        swiglu_bwd<<<blocks_for(size_t(N) * F), kBlock>>>(dinp, dout, inp, N, F);
        LAUNCH_CHECK();
    }

    void quantize_weights() {
        if (n_seg > 0) {
            partial_sums<<<dim3(kPartials, n_seg), kBlock>>>(seg_partial.p, params.p, seg_off.p, seg_len.p, true);
            LAUNCH_CHECK();
            ternary_gamma<<<n_seg, kWarp>>>(seg_gamma.p, seg_partial.p, seg_len.p, kPartials);
            LAUNCH_CHECK();
            ternary_apply<<<dim3(kPartials, n_seg), kBlock>>>(qparams.p, params.p, seg_off.p, seg_len.p,
                                                             seg_gamma.p);
            LAUNCH_CHECK();
        }
        weights_dirty = false;
    }

    void allocate(int B, int Tn) {
        CHECK(Tn <= cfg.seq_len, "sequence length %d > seq_len %d", Tn, cfg.seq_len);
        if (B * Tn <= B_ * T_ && Tn <= T_ && B <= B_) return;
        B_ = B;
        T_ = Tn;
        const size_t N = size_t(B) * Tn, C = cfg.d_model, V = cfg.vocab;
        const size_t F = cfg.always_hidden(), E = cfg.n_experts, K = cfg.top_k;
        const size_t Fe = cfg.ehidden(), NK = cfg.moe() ? N * K : 0;
        const size_t att = size_t(B) * cfg.n_heads * Tn * Tn;
        acts.resize(cfg.n_layers);
        for (auto& a : acts) {
            a.ln1.resize(N * C); a.rstd1.resize(N); a.qkv.resize(N * 3 * C);
            a.att.resize(att); a.atty.resize(N * C);
            a.resid1.resize(N * C); a.ln2.resize(N * C); a.rstd2.resize(N);
            a.resid2.resize(N * C); a.fc.resize(N * 2 * F); a.sw.resize(N * F);
            if (cfg.moe()) {
                a.rlogits.resize(N * E); a.sel.resize(N * K); a.gate.resize(N * K);
                a.order.resize(NK); a.row_of.resize(NK);
                a.xg.resize(NK * C); a.efc.resize(NK * 2 * Fe); a.esw.resize(NK * Fe);
                a.ey.resize(NK * C);
            }
        }
        x0.resize(N * C); lnf_out.resize(N * C); rstdf.resize(N);
        logits.resize(N * V); losses.resize(N);
        tokens.resize(N); targets.resize(N);
        xq.resize(std::max(N, NK) * std::max({C, F, Fe}));
        out_tmp.resize(N * C);
        dresid.resize(N * C); dC.resize(N * C); dqkv.resize(N * 3 * C); datty.resize(N * C);
        dfc.resize(std::max(N * 2 * F, NK * 2 * Fe)); dsw.resize(std::max(N * F, NK * Fe));
        dS.resize(att);
        dw_partial.resize((N + 63) / 64 * C);
        if (cfg.moe()) { dey.resize(NK * C); dxg.resize(NK * C); drl.resize(N * E); dgate.resize(NK); }
        losses_h.resize(N);
    }

    void ffn_forward(int l, float* out, int N, bool training) {
        auto& a = acts[l];
        const auto& o = layer[l];
        const int C = cfg.d_model, F = cfg.always_hidden();
        if (F > 0) {
            proj_forward(a.fc.p, a.ln2.p, o.wfc, N, C, 2 * F);
            swiglu_forward(a.sw.p, a.fc.p, N, F);
            proj_forward(out, a.sw.p, o.wproj, N, F, C);
        } else {
            CUDA_CHECK(cudaMemsetAsync(out, 0, size_t(N) * C * sizeof(float)));
        }
        if (cfg.moe()) moe_forward(l, out, N, training);
    }

    void ffn_backward(int l, float* dx, const float* dout, int N) {
        auto& a = acts[l];
        const auto& o = layer[l];
        const int C = cfg.d_model, F = cfg.always_hidden();
        if (F > 0) {
            dsw.zero(size_t(N) * F);
            proj_backward(dsw.p, dout, a.sw.p, o.wproj, N, F, C);
            dfc.zero(size_t(N) * 2 * F);
            swiglu_backward(dfc.p, dsw.p, a.fc.p, N, F);
            proj_backward(dx, dfc.p, a.ln2.p, o.wfc, N, C, 2 * F);
        }
        if (cfg.moe()) moe_backward(l, dx, dout, N);
    }

    void moe_forward(int l, float* out, int N, bool training) {
        auto& a = acts[l];
        const auto& o = layer[l];
        const int C = cfg.d_model, E = cfg.n_experts, K = cfg.top_k, Fe = cfg.ehidden(), NK = N * K;
        linear_forward(a.rlogits.p, a.ln2.p, P(o.router), N, C, E);
        counts.zero(E);
        moe_route<<<blocks_for(N), kBlock>>>(a.sel.p, a.gate.p, counts.p, a.rlogits.p,
                                             router_bias.p + size_t(l) * E, N, E, K);
        LAUNCH_CHECK();
        counts_h.resize(E);
        to_host(counts_h.data(), counts.p, E);  // waits for the routing
        if (training)
            for (int e = 0; e < E; ++e) expert_load[l][e] += counts_h[e];
        a.expert_start.assign(E + 1, 0);
        for (int e = 0; e < E; ++e) a.expert_start[e + 1] = a.expert_start[e] + counts_h[e];
        to_device(expert_start_d.p, a.expert_start.data(), E + 1);
        moe_order<<<E, 1024>>>(a.order.p, a.row_of.p, a.sel.p, expert_start_d.p, NK);
        LAUNCH_CHECK();
        moe_gather<<<blocks_for(size_t(NK) * C), kBlock>>>(a.xg.p, a.ln2.p, a.order.p, NK, C, K);
        LAUNCH_CHECK();
        const size_t efc_per = size_t(2) * Fe * C, eproj_per = size_t(C) * Fe;
        for (int e = 0; e < E; ++e) {
            const int r0 = a.expert_start[e], n = a.expert_start[e + 1] - r0;
            if (n == 0) continue;
            proj_forward(a.efc.p + size_t(r0) * 2 * Fe, a.xg.p + size_t(r0) * C, o.efc + e * efc_per, n, C, 2 * Fe);
            swiglu_forward(a.esw.p + size_t(r0) * Fe, a.efc.p + size_t(r0) * 2 * Fe, n, Fe);
            proj_forward(a.ey.p + size_t(r0) * C, a.esw.p + size_t(r0) * Fe, o.eproj + e * eproj_per, n, Fe, C);
        }
        moe_combine<<<blocks_for(size_t(N) * C), kBlock>>>(out, a.ey.p, a.gate.p, a.row_of.p, N, C, K);
        LAUNCH_CHECK();
    }

    void moe_backward(int l, float* dx, const float* dout, int N) {
        auto& a = acts[l];
        const auto& o = layer[l];
        const int C = cfg.d_model, E = cfg.n_experts, K = cfg.top_k, Fe = cfg.ehidden(), NK = N * K;
        moe_bwd_combine<<<blocks_for(NK, kRowsPerBlock), kBlock>>>(dey.p, dgate.p, dout, a.ey.p, a.gate.p,
                                                                  a.row_of.p, NK, C, K);
        LAUNCH_CHECK();
        dxg.zero(size_t(NK) * C);
        const size_t efc_per = size_t(2) * Fe * C, eproj_per = size_t(C) * Fe;
        for (int e = 0; e < E; ++e) {
            const int r0 = a.expert_start[e], n = a.expert_start[e + 1] - r0;
            if (n == 0) continue;
            dsw.zero(size_t(n) * Fe);
            proj_backward(dsw.p, dey.p + size_t(r0) * C, a.esw.p + size_t(r0) * Fe, o.eproj + e * eproj_per, n, Fe, C);
            dfc.zero(size_t(n) * 2 * Fe);
            swiglu_backward(dfc.p, dsw.p, a.efc.p + size_t(r0) * 2 * Fe, n, Fe);
            proj_backward(dxg.p + size_t(r0) * C, dfc.p, a.xg.p + size_t(r0) * C, o.efc + e * efc_per, n, C, 2 * Fe);
        }
        moe_scatter<<<blocks_for(size_t(N) * C), kBlock>>>(dx, dxg.p, a.row_of.p, N, C, K);
        LAUNCH_CHECK();
        drl.zero(size_t(N) * E);
        moe_router_grad<<<blocks_for(N), kBlock>>>(drl.p, a.gate.p, dgate.p, a.sel.p, N, E, K);
        LAUNCH_CHECK();
        linear_backward(dx, G(o.router), drl.p, a.ln2.p, P(o.router), N, C, E);
    }

    double forward(const int* tokens_h, const int* targets_h, int B, int Tn, bool training) {
        CHECK(targets_h, "GpuTrainer::forward needs targets");
        allocate(B, Tn);
        B_cur = B;
        T_cur = Tn;
        const int N = B * Tn, C = cfg.d_model, V = cfg.vocab;
        for (int n = 0; n < N; ++n) CHECK(tokens_h[n] >= 0 && tokens_h[n] < V, "token %d out of range", tokens_h[n]);
        if (weights_dirty) quantize_weights();
        to_device(tokens.p, tokens_h, N);
        to_device(targets.p, targets_h, N);

        embedding_fwd<<<blocks_for(size_t(N) * C), kBlock>>>(x0.p, tokens.p, P(wte), N, C);
        LAUNCH_CHECK();
        const float* x = x0.p;
        for (int l = 0; l < cfg.n_layers; ++l) {
            auto& a = acts[l];
            const auto& o = layer[l];
            rmsnorm_forward(a.ln1.p, a.rstd1.p, x, P(o.ln1), N, C);
            proj_forward(a.qkv.p, a.ln1.p, o.wqkv, N, C, 3 * C);
            rope_apply(a.qkv.p, B, Tn, false);
            attention_forward(a.atty.p, a.att.p, a.qkv.p, B, Tn);
            proj_forward(out_tmp.p, a.atty.p, o.wo, N, C, C);
            add(a.resid1.p, x, out_tmp.p, size_t(N) * C);
            rmsnorm_forward(a.ln2.p, a.rstd2.p, a.resid1.p, P(o.ln2), N, C);
            ffn_forward(l, out_tmp.p, N, training);
            add(a.resid2.p, a.resid1.p, out_tmp.p, size_t(N) * C);
            x = a.resid2.p;
        }
        rmsnorm_forward(lnf_out.p, rstdf.p, x, P(lnf), N, C);
        linear_forward(logits.p, lnf_out.p, P(wte), N, C, V);  // tied head
        xent_fwd<<<N, kBlock>>>(logits.p, losses.p, targets.p, V);
        LAUNCH_CHECK();
        to_host(losses_h.data(), losses.p, N);
        has_targets = true;
        n_valid = 0;
        double total = 0.0;
        for (int n = 0; n < N; ++n)
            if (targets_h[n] >= 0) total += losses_h[n], ++n_valid;
        return n_valid ? total / n_valid : 0.0;
    }

    void backward(double loss_scale) {
        CHECK(has_targets, "backward() needs a forward() first");
        const int B = B_cur, Tn = T_cur, N = B * Tn, C = cfg.d_model, V = cfg.vocab;
        if (n_valid == 0) return;
        xent_bwd<<<blocks_for(size_t(N) * V), kBlock>>>(logits.p, targets.p, N, V, float(loss_scale / n_valid));
        LAUNCH_CHECK();
        const float* dlogits = logits.p;
        const float* xL = acts.back().resid2.p;
        dC.zero(size_t(N) * C);
        linear_backward(dC.p, G(wte), dlogits, lnf_out.p, P(wte), N, C, V);
        dresid.zero(size_t(N) * C);
        rmsnorm_backward(dresid.p, G(lnf), dC.p, xL, P(lnf), rstdf.p, N, C);
        for (int l = cfg.n_layers - 1; l >= 0; --l) {
            auto& a = acts[l];
            const auto& o = layer[l];
            const float* x_in = l ? acts[l - 1].resid2.p : x0.p;
            dC.zero(size_t(N) * C);
            ffn_backward(l, dC.p, dresid.p, N);
            rmsnorm_backward(dresid.p, G(o.ln2), dC.p, a.resid1.p, P(o.ln2), a.rstd2.p, N, C);
            datty.zero(size_t(N) * C);
            proj_backward(datty.p, dresid.p, a.atty.p, o.wo, N, C, C);
            dqkv.zero(size_t(N) * 3 * C);
            attention_backward(dqkv.p, datty.p, a.att.p, a.qkv.p, B, Tn);
            rope_apply(dqkv.p, B, Tn, true);
            dC.zero(size_t(N) * C);
            proj_backward(dC.p, dqkv.p, a.ln1.p, o.wqkv, N, C, 3 * C);
            rmsnorm_backward(dresid.p, G(o.ln1), dC.p, x_in, P(o.ln1), a.rstd1.p, N, C);
        }
        embedding_bwd<<<blocks_for(size_t(N) * C), kBlock>>>(G(wte), dresid.p, tokens.p, N, C);
        LAUNCH_CHECK();
    }

    double grad_norm() {
        partial_sums<<<dim3(kPartials, 1), kBlock>>>(norm_partial.p, grads.p, all_off.p, all_len.p, false);
        LAUNCH_CHECK();
        double part[kPartials];
        to_host(part, norm_partial.p, kPartials);
        double s = 0.0;
        for (double p : part) s += p;
        return std::sqrt(s);
    }

    void adamw_step(double lr, double beta1, double beta2, double eps, double weight_decay, double grad_scale) {
        ++adam_step;
        const float c1 = float(1.0 / (1.0 - std::pow(beta1, double(adam_step))));
        const float c2 = float(1.0 / (1.0 - std::pow(beta2, double(adam_step))));
        for (size_t i = 0; i < infos.size(); ++i) {
            const ParamInfo& info = infos[i];
            const float wd = info.decay ? float(weight_decay) : 0.0f;
            if (v_at[i] == kNoV) {
                muon_step(info, float(lr), wd, float(grad_scale));
                continue;
            }
            const int grid = std::min(blocks_for(info.size), 4096);
            adamw<<<grid, kBlock>>>(params.p + info.offset, grads.p + info.offset, m.p + info.offset,
                                    v.p + v_at[i], info.size, float(lr), float(beta1), float(beta2), c1, c2,
                                    float(eps), wd, float(grad_scale));
            LAUNCH_CHECK();
        }
        weights_dirty = true;
    }

    // Muon blocks of one tensor: every (expert copy, fused part), each r x c.
    struct MuonShape {
        int batch, r, c;
        int mrows() const { return std::min(r, c); }  // Newton–Schulz runs on the wide orientation
        int ncols() const { return std::max(r, c); }
    };
    static MuonShape muon_shape(const ParamInfo& info) {
        const int parts = muon_parts(info);
        return {info.count * parts, info.rows / parts, info.cols};
    }

    // Lays v out for the current optimizer, keeping what it holds.
    void layout_v() {
        std::vector<size_t> at(infos.size(), kNoV);
        size_t n = 0;
        for (size_t i = 0; i < infos.size(); ++i)
            if (!(muon && muon_matrix(infos[i]))) at[i] = n, n += infos[i].size;
        DevBuf<float> nv;
        nv.resize(n);
        nv.zero();
        for (size_t i = 0; i < v_at.size(); ++i)
            if (at[i] != kNoV && v_at[i] != kNoV)
                CUDA_CHECK(cudaMemcpy(nv.p + at[i], v.p + v_at[i], infos[i].size * sizeof(float),
                                      cudaMemcpyDeviceToDevice));
        v = std::move(nv);  // the old buffer goes with nv
        v_at = std::move(at);
    }

    void set_muon(bool on, double momentum, int ns_steps) {
        muon = on;
        muon_mu = float(momentum);
        muon_ns = ns_steps;
        layout_v();
        if (!on) return;
        for (const auto& info : infos) {
            if (!muon_matrix(info)) continue;
            const MuonShape sh = muon_shape(info);
            const size_t mm = size_t(sh.mrows()) * sh.mrows();
            ns_x.resize(info.size);
            ns_y.resize(info.size);
            ns_a.resize(mm * sh.batch);
            ns_b.resize(mm * sh.batch);
        }
    }

    // AdamW::muon_update for one tensor, all its blocks batched through
    // cuBLAS; newton_schulz (train.h) step for step.
    void muon_step(const ParamInfo& info, float lr, float wd, float gscale) {
        const MuonShape sh = muon_shape(info);
        const int mr = sh.mrows(), nc = sh.ncols(), batch = sh.batch;
        const long long mn = (long long)mr * nc, mm = (long long)mr * mr;
        const int grid = std::min(blocks_for(info.size), 4096);
        float *X = ns_x.p, *Y = ns_y.p, *A = ns_a.p, *B = ns_b.p;
        muon_momentum<<<grid, kBlock>>>(X, m.p + info.offset, grads.p + info.offset, batch, sh.r, sh.c, muon_mu,
                                        gscale);
        frobenius_normalize<<<batch, kBlock>>>(X, size_t(mn));
        LAUNCH_CHECK();
        const float a = 3.4445f, b = -4.7750f, c = 2.0315f;
        for (int s = 0; s < muon_ns; ++s) {
            gemm_batched(false, true, mr, mr, nc, X, nc, mn, X, nc, mn, A, mr, mm, 0.0f, batch);   // A = X Xᵀ
            gemm_batched(false, false, mr, mr, mr, A, mr, mm, A, mr, mm, B, mr, mm, 0.0f, batch);  // B = A²
            axpby<<<std::min(blocks_for(size_t(mm) * batch), 4096), kBlock>>>(B, b, A, c, B, size_t(mm) * batch);
            gemm_batched(false, false, mr, nc, mr, B, mr, mm, X, nc, mn, Y, nc, mn, 0.0f, batch);  // Y = B X
            axpby<<<grid, kBlock>>>(X, a, X, 1.0f, Y, info.size);
            LAUNCH_CHECK();
        }
        muon_apply<<<grid, kBlock>>>(params.p + info.offset, X, batch, sh.r, sh.c, lr,
                                     0.2f * std::sqrt(float(nc)), wd);
        LAUNCH_CHECK();
    }
};

GpuTrainer::GpuTrainer(const Config& cfg, const std::vector<ParamInfo>& infos, bool tf32)
    : impl_(std::make_unique<Impl>(cfg, infos, tf32)) {}

GpuTrainer::~GpuTrainer() = default;

void GpuTrainer::set_params(const float* p) {
    to_device(impl_->params.p, p, impl_->n_params);
    impl_->weights_dirty = true;
}
void GpuTrainer::get_params(float* p) const { to_host(p, impl_->params.p, impl_->n_params); }
void GpuTrainer::get_grads(float* g) const { to_host(g, impl_->grads.p, impl_->n_params); }

void GpuTrainer::set_adam_state(long step, const float* m, const float* v) {
    impl_->adam_step = step;
    if (m) to_device(impl_->m.p, m, impl_->n_params);
    else impl_->m.zero();
    impl_->v.zero();
    if (!v) return;
    for (size_t i = 0; i < impl_->infos.size(); ++i)
        if (impl_->v_at[i] != Impl::kNoV)
            to_device(impl_->v.p + impl_->v_at[i], v + impl_->infos[i].offset, impl_->infos[i].size);
}
void GpuTrainer::get_adam_state(long* step, float* m, float* v) const {
    if (step) *step = impl_->adam_step;
    if (m) to_host(m, impl_->m.p, impl_->n_params);
    if (!v) return;
    std::fill(v, v + impl_->n_params, 0.0f);  // Muon's matrices have none, as on the CPU
    for (size_t i = 0; i < impl_->infos.size(); ++i)
        if (impl_->v_at[i] != Impl::kNoV)
            to_host(v + impl_->infos[i].offset, impl_->v.p + impl_->v_at[i], impl_->infos[i].size);
}

void GpuTrainer::set_router_bias(const std::vector<std::vector<float>>& bias) {
    const int E = std::max(1, impl_->cfg.n_experts);
    CHECK(int(bias.size()) == impl_->cfg.n_layers, "router bias for %zu layers", bias.size());
    std::vector<float> flat;
    for (const auto& b : bias) {
        CHECK(int(b.size()) == E, "router bias for %zu experts", b.size());
        flat.insert(flat.end(), b.begin(), b.end());
    }
    to_device(impl_->router_bias.p, flat.data(), flat.size());
}

double GpuTrainer::forward(const int* tokens, const int* targets, int B, int Tn, bool training) {
    return impl_->forward(tokens, targets, B, Tn, training);
}
void GpuTrainer::backward(double loss_scale) { impl_->backward(loss_scale); }
void GpuTrainer::zero_grad() { impl_->grads.zero(); }
double GpuTrainer::grad_norm() { return impl_->grad_norm(); }
void GpuTrainer::adamw_step(double lr, double beta1, double beta2, double eps, double weight_decay,
                            double grad_scale) {
    impl_->adamw_step(lr, beta1, beta2, eps, weight_decay, grad_scale);
}

void GpuTrainer::set_muon(bool on, double momentum, int ns_steps) { impl_->set_muon(on, momentum, ns_steps); }

std::vector<std::vector<double>> GpuTrainer::take_expert_load() {
    std::vector<std::vector<double>> out = impl_->expert_load;
    for (auto& l : impl_->expert_load) std::fill(l.begin(), l.end(), 0.0);
    return out;
}

size_t GpuTrainer::num_params() const { return impl_->n_params; }
size_t GpuTrainer::device_bytes() const { return g_device_bytes; }

std::string GpuTrainer::device_name() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    const unsigned long long tenths = prop.totalGlobalMem / 100000000ull;  // in units of 0.1 GB
    return std::string(prop.name) + " (" + std::to_string(tenths / 10) + "." + std::to_string(tenths % 10) +
           " GB, compute " + std::to_string(prop.major) + "." + std::to_string(prop.minor) + ")";
}
