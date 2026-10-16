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
#include <cstddef>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "attention.cuh"
#include "attention_tc.cuh"
#include "grouped_gemm.cuh"
#include "unlikelihood.h"

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

// rmsnorm_forward, one warp per row, into out and/or outh (bf16; either may
// be null).
__global__ void rmsnorm_fwd(float* out, __nv_bfloat16* outh, float* rstd, const float* x, const float* w, int N,
                            int C) {
    const int row = blockIdx.x * kRowsPerBlock + threadIdx.x / kWarp, lane = threadIdx.x % kWarp;
    if (row >= N) return;
    const float* xn = x + size_t(row) * C;
    float ss = 0.0f;
    for (int c = lane; c < C; c += kWarp) ss += xn[c] * xn[c];
    ss = warp_sum(ss);
    const float r = 1.0f / sqrtf(ss / float(C) + 1e-5f);
    if (lane == 0) rstd[row] = r;
    const size_t at = size_t(row) * C;
    for (int c = lane; c < C; c += kWarp) {
        const float y = xn[c] * r * w[c];
        if (out) out[at + c] = y;
        if (outh) outh[at + c] = __float2bfloat16(y);
    }
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

// rope_apply on q and k of qkv[N, 3C]; row n is at position n % Tn + pos0
// (pos0: incremental decoding's position).
__global__ void rope(float* qkv, const float* cos_t, const float* sin_t, int N, int Tn, int C, int NH, int half,
                     bool inverse, int pos0) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(N) * 2 * NH * half) return;
    const int k = int(i % half);
    size_t r = i / half;
    const int h = int(r % NH);
    r /= NH;
    const int part = int(r % 2);
    const int n = int(r / 2), t = n % Tn + pos0;
    float* x = qkv + size_t(n) * 3 * C + part * C + h * 2 * half;
    const float cs = cos_t[size_t(t) * half + k];
    const float sn = inverse ? -sin_t[size_t(t) * half + k] : sin_t[size_t(t) * half + k];
    const float x0 = x[k], x1 = x[k + half];
    x[k] = x0 * cs - x1 * sn;
    x[k + half] = x0 * sn + x1 * cs;
}

// Incremental decoding (Model::decode_step): the k and v of each sequence's
// new token, from qkv[B, 3C], into the caches kc, vc [B, S, C] at pos.
__global__ void kv_store(float* kc, float* vc, const float* qkv, int B, int S, int C, int pos) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(B) * C) return;
    const size_t b = i / C, c = i % C, at = (b * S + pos) * C + c;
    kc[at] = qkv[b * 3 * C + C + c];
    vc[at] = qkv[b * 3 * C + 2 * C + c];
}

// Incremental decoding's attention: block b * NH + h is head h of sequence
// b, whose new query (in qkv[B, 3C]) attends to its cached keys and values
// at positions start[b]..pos (left padding comes before start[b]). The
// scores live in dynamic shared memory, pos + 1 floats.
__global__ void decode_attn(float* out, const float* qkv, const float* kc, const float* vc, const int* start,
                            int S, int C, int NH, int HS, int pos, float scale) {
    extern __shared__ float p[];
    __shared__ float red[kWarp + 1];
    const int b = blockIdx.x / NH, h = blockIdx.x % NH, s0 = start[b];
    // A padding position (before the sequence starts) attends to nothing:
    // zeros, as on the CPU, rather than 0/0, which could reach the router.
    if (pos < s0) {
        for (int i = threadIdx.x; i < HS; i += blockDim.x) out[size_t(b) * C + h * HS + i] = 0.0f;
        return;
    }
    const int lane = threadIdx.x % kWarp, wid = threadIdx.x / kWarp, nw = blockDim.x / kWarp;
    const float* q = qkv + size_t(b) * 3 * C + h * HS;
    const size_t base = size_t(b) * S * C + size_t(h) * HS;
    // One warp per cached position: its score q·k.
    for (int s = s0 + wid; s <= pos; s += nw) {
        const float* k = kc + base + size_t(s) * C;
        float d = 0.0f;
        for (int i = lane; i < HS; i += kWarp) d += q[i] * k[i];
        d = warp_sum(d);
        if (lane == 0) p[s] = d * scale;
    }
    __syncthreads();
    float mx = -INFINITY;
    for (int s = s0 + threadIdx.x; s <= pos; s += blockDim.x) mx = fmaxf(mx, p[s]);
    mx = block_reduce(mx, red, true);
    float sum = 0.0f;
    for (int s = s0 + threadIdx.x; s <= pos; s += blockDim.x) sum += p[s] = expf(p[s] - mx);
    sum = block_reduce(sum, red, false);  // its barriers also make every p[s] visible
    const float inv = 1.0f / sum;
    for (int i = threadIdx.x; i < HS; i += blockDim.x) {
        float y = 0.0f;
        for (int s = s0; s <= pos; ++s) y += p[s] * vc[base + size_t(s) * C + i];
        out[size_t(b) * C + h * HS + i] = y * inv;
    }
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

// swiglu_forward: inp[N, 2F] = [gate | up], out = silu(gate) * up, in fp32
// (out) or bf16 (outh); the other is null.
__global__ void swiglu_fwd(float* out, __nv_bfloat16* outh, const float* inp, int N, int F) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(N) * F) return;
    const size_t n = i / F, j = i % F;
    const float g = inp[n * 2 * F + j], u = inp[n * 2 * F + F + j];
    const float y = g / (1.0f + expf(-g)) * u;
    if (out) out[i] = y;
    else outh[i] = __float2bfloat16(y);
}

// swiglu_backward, written (not added) into dinp, or into dinph in bf16.
__global__ void swiglu_bwd(float* dinp, __nv_bfloat16* dinph, const float* dout, const float* inp, int N, int F) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(N) * F) return;
    const size_t n = i / F, j = i % F;
    const float g = inp[n * 2 * F + j], u = inp[n * 2 * F + F + j], d = dout[i];
    const float sg = 1.0f / (1.0f + expf(-g));
    const float dg = d * u * sg * (1.0f + g * (1.0f - sg)), du = d * g * sg;
    if (dinp) {
        dinp[n * 2 * F + j] = dg;
        dinp[n * 2 * F + F + j] = du;
    } else {
        dinph[n * 2 * F + j] = __float2bfloat16(dg);
        dinph[n * 2 * F + F + j] = __float2bfloat16(du);
    }
}

__global__ void add_fwd(float* out, const float* a, const float* b, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

// cross_entropy_forward, one block per row: logits become probabilities.
__global__ void xent_fwd(float* probs, float* losses, const int* targets, int V) {
    __shared__ float sh[kWarp + 1];
    __shared__ float target_gap;
    const int n = blockIdx.x;
    float* p = probs + size_t(n) * V;
    float mx = -INFINITY;
    for (int i = threadIdx.x; i < V; i += blockDim.x) mx = fmaxf(mx, p[i]);
    mx = block_reduce(mx, sh, true);
    // Capture the target logit before any thread replaces logits by exp.
    // The barrier makes this read safe even when another warp owns p[t].
    if (threadIdx.x == 0) {
        const int t = targets[n];
        target_gap = t < 0 ? 0.0f : mx - p[t];
    }
    __syncthreads();
    float sum = 0.0f;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        const float e = expf(p[i] - mx);
        p[i] = e;
        sum += e;
    }
    sum = block_reduce(sum, sh, false);
    const float inv = 1.0f / sum;
    for (int i = threadIdx.x; i < V; i += blockDim.x) p[i] *= inv;
    __syncthreads();
    if (threadIdx.x == 0) {
        const int t = targets[n];
        losses[n] = t < 0 ? 0.0f : target_gap + logf(sum);
    }
}

// cross_entropy_backward: d = (p - onehot(target)) * dloss, from the
// probabilities in d, in place, or into dh in bf16 when dh is not null.
// With unlikelihood, p is first scaled by row_scale[n] = 1 - alpha * W (W:
// the row's sum of p_c / (1 - p_c)), and ul_fix then writes the negatives'
// own entries.
__global__ void xent_bwd(float* d, __nv_bfloat16* dh, const int* targets, int N, int V, float dloss,
                         const float* row_scale) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(N) * V) return;
    const int n = int(i / V), v = int(i % V), t = targets[n];
    const float p = row_scale ? d[i] * row_scale[n] : d[i];
    const float g = t < 0 ? 0.0f : v == t ? p * dloss - dloss : p * dloss;
    if (dh) dh[i] = __float2bfloat16(g);
    else d[i] = g;
}

// unlikelihood_forward (ops.h), one block per row, on the probabilities
// xent_fwd left: for each negative c, 1 - p_c summed in double from the
// other tokens, its loss (floored), and what the backward pass needs:
// p_c and w = p_c / (1 - p_c) per negative; per row the sum of w, the
// scale 1 - alpha * W for xent_bwd, and the row's loss.
__global__ void ul_fwd(const float* probs, const int* off, const int* tok, float* neg_p, double* neg_w,
                       double* row_w, float* row_scale, double* row_ul, double alpha, int V) {
    __shared__ double sh[kWarp + 1];
    const int n = blockIdx.x;
    const float* p = probs + size_t(n) * V;
    double W = 0.0, ul = 0.0;
    for (int k = off[n]; k < off[n + 1]; ++k) {
        const int c = tok[k];
        double q = 0.0;
        for (int i = threadIdx.x; i < V; i += blockDim.x)
            if (i != c) q += double(p[i]);
        q = block_reduce(q, sh, false);
        const double pc = double(p[c]), w = q > 0 ? pc / q : 0.0;
        W += w;
        ul -= log(fmax(q, kUnlikelihoodFloor));
        if (threadIdx.x == 0) neg_p[k] = p[c], neg_w[k] = w;
    }
    if (threadIdx.x == 0) {
        row_w[n] = W;
        // At most -FLT_MAX: p_j <= 1 - p_c keeps p_j * scale small, but an
        // infinite scale would turn p_j = 0 into NaN.
        row_scale[n] = float(fmax(1.0 - alpha * W, -double(FLT_MAX)));
        row_ul[n] = ul;
    }
}

// The negatives' own gradient entries, after xent_bwd, one thread per
// negative (cross_entropy_backward in ops.h): p_c (1 - alpha (W - w_c)) +
// alpha p_c, the second term only where 1 - p_c > 0.
__global__ void ul_fix(float* d, __nv_bfloat16* dh, const int* tok, const int* row, const float* neg_p,
                       const double* neg_w, const double* row_w, int K, int V, double alpha, float dloss) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    const int n = row[k];
    const double pc = neg_p[k], wc = neg_w[k];
    const double self = wc > 0 ? alpha * pc : 0.0;
    const float g = float((pc * (1 - alpha * (row_w[n] - wc)) + self) * double(dloss));
    const size_t i = size_t(n) * V + tok[k];
    if (dh) dh[i] = __float2bfloat16(g);
    else d[i] = g;
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

// fp32 to bf16 (round to nearest even), for the operands of bf16 products.
__global__ void to_bf16(__nv_bfloat16* out, const float* in, size_t n) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        out[i] = __float2bfloat16(in[i]);
}

// ---------------------------------------------------------------------------
// The end of a training step (finish_step): sums, checks and balancing that
// the host used to do on numbers copied back
// ---------------------------------------------------------------------------

// What a training step leaves on the GPU for the host to read once, in one
// copy, at its end (finish_step).
struct StepState {
    double ce = 0, ul = 0;  // the step's summed cross-entropy and unlikelihood term, over its trained targets
    double norm = 0;        // the gradient's L2 norm
    float scale = 1.0f;     // what clipping multiplies the gradient by (1: no clipping)
    int ok = 0;             // norm and loss finite: the update runs
    double load_ratio = 1.0;  // the busiest expert's share against an even split, worst layer (max_load_ratio)
};

// *sum += Σ losses[n] over the rows with a target (targets[n] >= 0), in
// double. One block, so the order of the sum is the same every time.
__global__ void sum_trained(double* sum, const float* losses, const int* targets, int N) {
    __shared__ double sh[kWarp + 1];
    double s = 0.0;
    for (int n = threadIdx.x; n < N; n += blockDim.x)
        if (targets[n] >= 0) s += double(losses[n]);
    s = block_reduce(s, sh, false);
    if (threadIdx.x == 0) *sum += s;
}

// *sum += Σ x[n]. One block.
__global__ void sum_all(double* sum, const double* x, int N) {
    __shared__ double sh[kWarp + 1];
    double s = 0.0;
    for (int n = threadIdx.x; n < N; n += blockDim.x) s += x[n];
    s = block_reduce(s, sh, false);
    if (threadIdx.x == 0) *sum += s;
}

// The gradient norm from partial_sums' n partials, summed in order as the
// host did, whether the step may update, and the clipping scale, as
// GpuBackend::step computed them: max_norm / norm, rounded to float, when
// the norm is over max_norm (> 0). One thread.
__global__ void step_check(StepState* s, const double* partial, int n, double max_norm) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += partial[i];
    const double norm = sqrt(sum);
    s->norm = norm;
    s->ok = isfinite(norm) && isfinite(s->ce);
    s->scale = max_norm > 0 && norm > max_norm ? float(max_norm / norm) : 1.0f;
}

// Model::max_load_ratio and update_router_bias on the step's expert counts
// ([layer, expert]), exactly as the host did it: the busiest expert's ratio
// into s->load_ratio, each bias +rate below its layer's mean and -rate
// above. The counts then go into the interval's (for --load-log) and are
// cleared. One thread: layers x experts is a few hundred numbers.
__global__ void balance_router(StepState* s, float* bias, unsigned long long* step, unsigned long long* interval,
                               int L, int E, float rate) {
    double worst = 1.0;
    for (int l = 0; l < L; ++l) {
        unsigned long long* c = step + size_t(l) * E;
        double total = 0.0, busiest = 0.0;
        for (int e = 0; e < E; ++e) total += double(c[e]), busiest = fmax(busiest, double(c[e]));
        if (total > 0) worst = fmax(worst, busiest * E / total);
        const double mean = total / E;
        for (int e = 0; e < E; ++e) {
            if (double(c[e]) < mean) bias[size_t(l) * E + e] += rate;
            else if (double(c[e]) > mean) bias[size_t(l) * E + e] -= rate;
            interval[size_t(l) * E + e] += c[e];
            c[e] = 0;
        }
    }
    s->load_ratio = worst;
}

// The experts' first rows in the grouped order (moe_order): starts[0] = 0,
// starts[e + 1] = starts[e] + counts[e]. One thread.
__global__ void expert_starts(int* starts, const int* counts, int E) {
    starts[0] = 0;
    for (int e = 0; e < E; ++e) starts[e + 1] = starts[e] + counts[e];
}

// counts[e] tokens went to expert e in this layer: add them to the step's.
__global__ void add_counts(unsigned long long* step, const int* counts, int E) {
    for (int e = threadIdx.x; e < E; e += blockDim.x) step[e] += (unsigned long long)counts[e];
}

// AdamW::update for one tensor, with the gradient scaled by *gscale (clipping).
__global__ void adamw(float* p, const float* g, float* m, float* v, size_t n, float lr, float b1, float b2,
                      float c1, float c2, float eps, float wd, const float* gscale_p) {
    const float gscale = *gscale_p;
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
                              const float* gscale_p) {
    const float gscale = *gscale_p;
    const size_t rc = size_t(r) * c, n = rc * batch;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        const float gi = g[i] * gscale;
        mom[i] = mu * mom[i] + (1.0f - mu) * gi;
        const size_t k = i % rc, row = k / c, col = k % c;
        X[i - k + (r > c ? col * r + row : k)] = (1.0f - mu) * gi + mu * mom[i];
    }
}

// newton_schulz's first step: each len-long matrix of X divided by its
// Frobenius norm (+ 1e-7), also into Xh (bf16) unless null. One thread
// block per matrix.
__global__ void frobenius_normalize(float* X, __nv_bfloat16* Xh, size_t len) {
    __shared__ double sh[kWarp + 1];
    float* x = X + blockIdx.x * len;
    double s = 0.0;
    for (size_t i = threadIdx.x; i < len; i += blockDim.x) s += double(x[i]) * double(x[i]);
    s = block_reduce(s, sh, false);
    const float inv = float(1.0 / (sqrt(s) + 1e-7));
    for (size_t i = threadIdx.x; i < len; i += blockDim.x) {
        x[i] *= inv;
        if (Xh) Xh[blockIdx.x * len + i] = __float2bfloat16(x[i]);
    }
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
// selected assignment its rank. Expert e's rows start after the counts of
// the experts before it.
__global__ void moe_order(int* order, int* row_of, const int* sel, const int* counts, int NK) {
    __shared__ int warp_total[kWarp];
    __shared__ int base;
    const int e = blockIdx.x, lane = threadIdx.x % kWarp, wid = threadIdx.x / kWarp;
    const int nw = blockDim.x / kWarp;
    if (threadIdx.x == 0) {
        base = 0;
        for (int i = 0; i < e; ++i) base += counts[i];
    }
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

// xg[r] = x[token of assignment order[r]], in fp32 or bf16.
template <class T>
__global__ void moe_gather(T* xg, const T* x, const int* order, int NK, int C, int K) {
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

// One warp per assignment j: dey[row] = gate * dout[n] (or deyh, in bf16),
// dgate[j] = <dout[n], ey[row]>.
__global__ void moe_bwd_combine(float* dey, __nv_bfloat16* deyh, float* dgate, const float* dout, const float* ey,
                                const float* gate, const int* row_of, int NK, int C, int K) {
    const int j = blockIdx.x * kRowsPerBlock + threadIdx.x / kWarp, lane = threadIdx.x % kWarp;
    if (j >= NK) return;
    const size_t n = size_t(j / K), row = size_t(row_of[j]);
    const float g = gate[j];
    float dot = 0.0f;
    for (int c = lane; c < C; c += kWarp) {
        const float d = dout[n * C + c];
        if (dey) dey[row * C + c] = g * d;
        else deyh[row * C + c] = __float2bfloat16(g * d);
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

// Fused attention, shared by GpuTrainer and gpu_fused_attention (tests):
// the tensor-core kernels (attention_tc.cuh) when tc, else the fp32 ones
// (attention.cuh). qkv [B·Tn, 3C], out [B·Tn, C], lse and dsum [B, NH, Tn].

// Lets the kernels for this head size use the shared memory they need
// (above 48 KB, CUDA wants to be told first).
void fused_attention_prepare(int HS, bool tc) {
    auto allow = [](const void* kernel, int floats) {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        floats * int(sizeof(float))));
    };
    if (tc) {
        attn_tc::with_head_size(HS, [&]<int H>() {
            allow((const void*)attn_tc::fwd<H>, attn_tc::fwd_smem(H));
            allow((const void*)attn_tc::bwd_dq<H>, attn_tc::dq_smem(H));
            allow((const void*)attn_tc::bwd_dkv<H>, attn_tc::dkv_smem(H));
        });
    } else {
        allow((const void*)attn::fwd, attn::fwd_smem(HS));
        allow((const void*)attn::bwd_dq, attn::dq_smem(HS));
        allow((const void*)attn::bwd_dkv, attn::dkv_smem(HS));
    }
}

void fused_attention_forward(bool tc, const float* qkv, float* out, float* lse, int B, int Tn, int C, int NH,
                             int HS) {
    const float scale = 1.0f / std::sqrt(float(HS));
    if (tc) {
        const dim3 grid((Tn + attn_tc::kBlockRows - 1) / attn_tc::kBlockRows, NH, B);
        attn_tc::with_head_size(HS, [&]<int H>() {
            attn_tc::fwd<H><<<grid, attn_tc::kThreads, attn_tc::fwd_smem(H) * sizeof(float)>>>(qkv, out, lse, Tn, C,
                                                                                            NH, scale);
        });
    } else {
        const dim3 grid((Tn + attn::kBlockRows - 1) / attn::kBlockRows, NH, B);
        attn::fwd<<<grid, attn::kThreads, attn::fwd_smem(HS) * sizeof(float)>>>(qkv, out, lse, Tn, C, NH, HS,
                                                                                scale);
    }
    LAUNCH_CHECK();
}

// dqkv += the backward of out (given dout); dsum is scratch, B·NH·Tn floats.
void fused_attention_backward(bool tc, const float* qkv, const float* out, const float* dout, const float* lse,
                              float* dsum, float* dqkv, int B, int Tn, int C, int NH, int HS) {
    const float scale = 1.0f / std::sqrt(float(HS));
    const int rows = B * NH * Tn;
    attn::bwd_dsum<<<(rows + attn::kWarps - 1) / attn::kWarps, attn::kThreads>>>(out, dout, dsum, rows, Tn, C, NH,
                                                                                 HS);
    LAUNCH_CHECK();
    if (tc) {
        const dim3 grid((Tn + attn_tc::kBlockRows - 1) / attn_tc::kBlockRows, NH, B);
        attn_tc::with_head_size(HS, [&]<int H>() {
            attn_tc::bwd_dq<H><<<grid, attn_tc::kThreads, attn_tc::dq_smem(H) * sizeof(float)>>>(
                qkv, dout, lse, dsum, dqkv, Tn, C, NH, scale);
            LAUNCH_CHECK();
            attn_tc::bwd_dkv<H><<<grid, attn_tc::kThreads, attn_tc::dkv_smem(H) * sizeof(float)>>>(
                qkv, dout, lse, dsum, dqkv, Tn, C, NH, scale);
        });
    } else {
        const dim3 grid((Tn + attn::kBlockRows - 1) / attn::kBlockRows, NH, B);
        attn::bwd_dq<<<grid, attn::kThreads, attn::dq_smem(HS) * sizeof(float)>>>(qkv, dout, lse, dsum, dqkv, Tn, C,
                                                                                  NH, HS, scale);
        LAUNCH_CHECK();
        attn::bwd_dkv<<<grid, attn::kThreads, attn::dkv_smem(HS) * sizeof(float)>>>(qkv, dout, lse, dsum, dqkv, Tn,
                                                                                    C, NH, HS, scale);
    }
    LAUNCH_CHECK();
}

// A product's input: fp32, or (with bf16 products) bf16 already.
struct Operand {
    const float* f = nullptr;
    const __nv_bfloat16* h = nullptr;
    Operand(const float* p) : f(p) {}
    Operand(const __nv_bfloat16* p) : h(p) {}
    Operand operator+(size_t o) const { return f ? Operand(f + o) : Operand(h + o); }
};

}  // namespace

// ---------------------------------------------------------------------------
// GpuTrainer
// ---------------------------------------------------------------------------

struct GpuTrainer::Impl {
    Config cfg;
    std::vector<ParamInfo> infos;
    size_t n_params = 0;
    cublasHandle_t blas = nullptr;
    cublasComputeType_t compute = CUBLAS_COMPUTE_32F_FAST_TF32;  // how fp32 products multiply (Matmul)
    // Matmul::bf16: the model's projections (linear_forward/backward) take
    // bf16 operands: a bf16 copy of the weights (wbf, refreshed with the
    // ternary copy) and inputs written in bf16 (LayerActs), or rounded at
    // the product (op_x, op_dy); sums and results stay fp32, as does
    // everything else.
    bool bf16_ops = false;

    DevBuf<float> params, grads, qparams, m, v;
    DevBuf<__nv_bfloat16> wbf, op_x, op_dy;
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
    DevBuf<__nv_bfloat16> ns_xh, ns_ah;  // with bf16 products: X and A as their inputs
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
    // Tokens routed to each expert [layer, expert]: in this step (cleared
    // by balancing or take_expert_load) and since the last load-log line.
    DevBuf<unsigned long long> step_load, interval_load;
    // The step's sums, checks and clip scale (StepState), and its host copy.
    DevBuf<StepState> state;
    DevBuf<double> fwd_sums;  // forward()'s own cross-entropy and term sums, outside a step's
    StepState* state_h = nullptr;  // pinned

    // Token ids go up from pinned host memory without waiting, so the host
    // can queue the next micro-batch while the GPU still works on this one.
    // The slots take turns; before refilling one, the host waits for its
    // last copy to finish.
    struct Upload {
        int* h = nullptr;
        size_t cap = 0;
        cudaEvent_t done = nullptr;
    };
    static constexpr int kUploads = 4;
    Upload uploads[kUploads];
    int upload_turn = 0;

    void upload(int* dst, const int* src, size_t n) {
        Upload& u = uploads[upload_turn];
        upload_turn = (upload_turn + 1) % kUploads;
        CUDA_CHECK(cudaEventSynchronize(u.done));
        if (u.cap < n) {
            if (u.h) CUDA_CHECK(cudaFreeHost(u.h));
            CUDA_CHECK(cudaMallocHost(&u.h, n * sizeof(int)));
            u.cap = n;
        }
        std::copy(src, src + n, u.h);
        CUDA_CHECK(cudaMemcpyAsync(dst, u.h, n * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(u.done));
    }
    // The address of one StepState field on the GPU, for kernels to read.
    template <class T>
    T* state_field(size_t offset) { return reinterpret_cast<T*>(reinterpret_cast<char*>(state.p) + offset); }
    const float* grad_scale() { return state_field<float>(offsetof(StepState, scale)); }

    // Profiling (set_profile): GPU time per named region of the step, from
    // CUDA events recorded around it, so timing never makes the GPU wait.
    bool profiling = false;
    struct Mark {
        const char* name;
        cudaEvent_t start, stop;
    };
    std::vector<Mark> marks;
    std::vector<cudaEvent_t> spare_events;

    cudaEvent_t record_event() {
        cudaEvent_t e;
        if (spare_events.empty()) {
            CUDA_CHECK(cudaEventCreate(&e));
        } else {
            e = spare_events.back();
            spare_events.pop_back();
        }
        CUDA_CHECK(cudaEventRecord(e));
        return e;
    }
    // Region r(*this, "name"): the rest of the enclosing block counts as
    // "name". Regions must not nest.
    struct Region {
        Impl& impl;
        const char* name;
        cudaEvent_t start = nullptr;
        Region(Impl& i, const char* n) : impl(i), name(n) {
            if (impl.profiling) start = impl.record_event();
        }
        ~Region() {
            if (start) impl.marks.push_back({name, start, impl.record_event()});
        }
    };

    std::vector<std::pair<std::string, double>> take_profile() {
        std::vector<std::pair<std::string, double>> out;
        if (marks.empty()) return out;
        CUDA_CHECK(cudaDeviceSynchronize());
        for (const auto& mk : marks) {
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, mk.start, mk.stop));
            auto it = std::find_if(out.begin(), out.end(), [&](const auto& e) { return e.first == mk.name; });
            if (it == out.end()) out.emplace_back(mk.name, 1e-3 * ms);
            else it->second += 1e-3 * ms;
            spare_events.push_back(mk.start);
            spare_events.push_back(mk.stop);
        }
        marks.clear();
        return out;
    }

    // Attention: fused kernels (attention_tc.cuh, attention.cuh), which keep
    // one number per row (lse), or cuBLAS products through the full matrix
    // of weights (att).
    bool fused_attn = false;
    bool tc_attn = false;  // the fused kernels on the tensor cores (attention_tc.cuh)

    // Activations that are only ever products' inputs (ln1, sw, xg, esw,
    // lnf_out, and in the backward pass dfc, dey, dlogits) are kept in bf16
    // instead (the ...h buffers) with bf16 products; ln2 and atty in both,
    // since the router and the attention backward read them in fp32.
    struct LayerActs {
        DevBuf<float> ln1, rstd1, qkv, att, lse, atty, resid1, ln2, rstd2, resid2, fc, sw;
        DevBuf<float> rlogits, gate, xg, efc, esw, ey;
        DevBuf<__nv_bfloat16> ln1h, attyh, ln2h, swh, xgh, eswh;
        DevBuf<int> sel, order, row_of;
        DevBuf<int> starts;  // the experts' first rows [E + 1] (expert_starts)
        std::vector<int> expert_start;  // their host copy, for the per-expert products
    };
    std::vector<LayerActs> acts;
    DevBuf<float> x0, lnf_out, rstdf, logits, losses, xq, out_tmp;
    DevBuf<float> dresid, dC, dqkv, datty, dfc, dsw, dey, dxg, drl, dS, dsum, dgate, dw_partial;
    DevBuf<__nv_bfloat16> lnfh, dfch, deyh, dlogitsh;
    DevBuf<int> tokens, targets, counts;
    // The experts' products: one grouped launch per projection, without the
    // host (grouped_gemm.cuh; set_grouped_experts), or cuBLAS per expert,
    // sized by the router's counts copied to the host (pinned memory, so
    // the copy runs behind other work) once counts_ready has passed. The
    // default stays cuBLAS until the grouped kernels are as fast (#49).
    bool grouped_experts = false;
    int* counts_h = nullptr;
    cudaEvent_t counts_ready = nullptr;
    int B_ = 0, T_ = 0, B_cur = 0, T_cur = 0, n_valid = 0;
    bool has_targets = false;
    // Unlikelihood (#34), as Model::unlikelihood: the negatives are found on
    // the host (unlikelihood.h) in each training forward, the term computed
    // here (ul_fwd, ul_fix). ul_active: the last forward had negatives.
    double ul_alpha = 0;
    int ul_ngram = 8;
    RepeatNegatives negatives_h;
    std::vector<int> neg_row_h;
    DevBuf<int> neg_off, neg_tok, neg_row;
    DevBuf<float> neg_p, row_scale;
    DevBuf<double> neg_w, row_w, row_ul;
    bool ul_active = false;
    GpuTrainer::ForwardParts parts;
    // Incremental decoding (decode_begin, decode_step): the k and v caches
    // [layer][B, S, C], each sequence's first real position, and the logits
    // copied back for the host to pick the next token. The caches live in
    // the gradient buffer when they fit there: between training steps it
    // holds nothing (each step starts with zero_grad()), and training leaves
    // little other memory free. grads_scratch says the gradients were
    // overwritten; zero_grad() clears it. kv_own: the caches' own memory
    // when the gradient buffer is too small.
    std::vector<float*> kcache, vcache;
    DevBuf<float> kv_own;
    DevBuf<int> kv_start;
    int kv_B = 0;
    bool grads_scratch = false;
    std::vector<float> logits_h;

    Impl(const Config& c, const std::vector<ParamInfo>& in, Matmul matmul, bool fused)
        : cfg(c), infos(in), fused_attn(fused) {
        cfg.validate();
        int devices = 0;
        const cudaError_t err = cudaGetDeviceCount(&devices);
        CHECK(err == cudaSuccess && devices > 0, "no CUDA GPU available (%s)",
              err == cudaSuccess ? "no devices" : cudaGetErrorString(err));
        // While the host waits for the GPU (expert counts, losses, logits),
        // it sleeps instead of spinning, CUDA's default, which kept a whole
        // CPU core busy all night. It must be set before the GPU is first
        // used; for a later trainer in the same process it is already set.
        if (cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync) != cudaSuccess) (void)cudaGetLastError();
        for (const auto& p : infos) n_params = std::max(n_params, p.offset + p.size);
        CUBLAS_CHECK(cublasCreate(&blas));
        // With bf16 products, the products left in fp32 (cuBLAS attention's)
        // use TF32.
        compute = matmul == Matmul::fp32 ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_FAST_TF32;
        bf16_ops = matmul == Matmul::bf16;
        CHECK(!bf16_ops || !cfg.ternary,
              "--matmul bf16 does not take ternary models yet (their quantized weights and activations are not "
              "exact in bf16); use --matmul tf32");
        if (fused_attn) {
            const int HS = cfg.head_size();
            CHECK(HS <= attn::kMaxHS, "fused attention takes heads of up to %d (this model: %d); use --no-fused-attn",
                  attn::kMaxHS, HS);
            // With TF32 or bf16 matmuls, attention multiplies on the tensor
            // cores too; exact fp32 keeps the fp32 kernels.
            tc_attn = fused_attention_on_tensor_cores(matmul, HS);
            fused_attention_prepare(HS, tc_attn);
        }
        params.resize(n_params);
        grads.resize(n_params);
        if (cfg.ternary) qparams.resize(n_params);  // read only through W(), for ternary models
        if (bf16_ops) wbf.resize(n_params);
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
        step_load.resize(size_t(cfg.n_layers) * E);
        interval_load.resize(size_t(cfg.n_layers) * E);
        step_load.zero();
        interval_load.zero();
        counts.resize(E);
        CUDA_CHECK(cudaMallocHost(&counts_h, E * sizeof(int)));
        CUDA_CHECK(cudaEventCreateWithFlags(&counts_ready, cudaEventDisableTiming));
        state.resize(1);
        const StepState fresh;
        to_device(state.p, &fresh, 1);
        fwd_sums.resize(2);
        CUDA_CHECK(cudaMallocHost(&state_h, sizeof(StepState)));
        for (Upload& u : uploads) {
            CUDA_CHECK(cudaEventCreateWithFlags(&u.done, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventRecord(u.done));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    ~Impl() {
        if (counts_h) cudaFreeHost(counts_h);
        if (counts_ready) cudaEventDestroy(counts_ready);
        if (state_h) cudaFreeHost(state_h);
        for (Upload& u : uploads) {
            if (u.done) cudaEventSynchronize(u.done), cudaEventDestroy(u.done);
            if (u.h) cudaFreeHost(u.h);
        }
        if (blas) cublasDestroy(blas);
        for (const auto& mk : marks) {
            cudaEventDestroy(mk.start);
            cudaEventDestroy(mk.stop);
        }
        for (cudaEvent_t e : spare_events) cudaEventDestroy(e);
    }

    float* P(size_t o) { return params.p + o; }
    float* G(size_t o) { return grads.p + o; }
    // Weights as the products use them: their ternary copy when q.
    const float* W(size_t o, bool q) const { return q ? qparams.p + o : params.p + o; }
    bool attn_ternary() const { return cfg.ternary && !cfg.fp_attention; }

    // Row-major C[M, N] = op(A)[M, K] · op(B)[K, N] + beta * C. With ta, A is
    // stored as [K, M] (row stride lda); likewise B with tb. cuBLAS is
    // column-major, where a row-major matrix reads as its transpose, so this
    // computes Cᵀ = op(B)ᵀ · op(A)ᵀ. A and B are fp32, or both bf16 (bf16
    // products: fp32 sums, then C stays fp32 either way).
    void gemm(bool ta, bool tb, int M, int N, int K, const void* A, int lda, const void* B, int ldb, float* C,
              int ldc, float beta, bool bf16 = false) {
        if (M == 0 || N == 0) return;
        const float one = 1.0f;
        const cudaDataType_t type = bf16 ? CUDA_R_16BF : CUDA_R_32F;
        CUBLAS_CHECK(cublasGemmEx(blas, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &one,
                                  B, type, ldb, A, type, lda, &beta, C, CUDA_R_32F, ldc,
                                  bf16 ? CUBLAS_COMPUTE_32F : compute, CUBLAS_GEMM_DEFAULT));
    }

    // The same, for `batch` independent products at fixed strides.
    void gemm_batched(bool ta, bool tb, int M, int N, int K, const float* A, int lda, long long sa,
                      const float* B, int ldb, long long sb, float* C, int ldc, long long sc, float beta,
                      int batch) {
        const float one = 1.0f;
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(blas, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N,
                                                N, M, K, &one, B, CUDA_R_32F, ldb, sb, A, CUDA_R_32F, lda, sa, &beta,
                                                C, CUDA_R_32F, ldc, sc, batch, compute, CUBLAS_GEMM_DEFAULT));
    }
    // C = alpha op(A) op(B) + beta C, batched, from bf16 inputs (fp32 sums).
    void gemm_batched_bf16(bool ta, bool tb, int M, int N, int K, const __nv_bfloat16* A, int lda, long long sa,
                           const __nv_bfloat16* B, int ldb, long long sb, float* C, int ldc, long long sc,
                           float alpha, float beta, int batch) {
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(blas, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N,
                                                N, M, K, &alpha, B, CUDA_R_16BF, ldb, sb, A, CUDA_R_16BF, lda, sa,
                                                &beta, C, CUDA_R_32F, ldc, sc, batch, CUBLAS_COMPUTE_32F,
                                                CUBLAS_GEMM_DEFAULT));
    }

    // With bf16 products: the bf16 copy of weights w (a pointer into params).
    const __nv_bfloat16* WB(const float* w) const { return wbf.p + (w - params.p); }
    // An activation kept in fp32 (f) or, with bf16 products, in bf16 (h).
    Operand pick(const DevBuf<float>& f, const DevBuf<__nv_bfloat16>& h) const {
        return bf16_ops ? Operand(h.p) : Operand(f.p);
    }
    // x as a bf16 product's input: as it is, or n floats rounded into the
    // operand buffer op (grown on first use).
    const __nv_bfloat16* as_bf16(Operand x, DevBuf<__nv_bfloat16>& op, size_t n) {
        if (x.h) return x.h;
        op.resize(n);
        to_bf16<<<std::min(blocks_for(n), 4096), kBlock>>>(op.p, x.f, n);
        LAUNCH_CHECK();
        return op.p;
    }

    // linear_forward: y[N, OC] = x[N, IC] · w[OC, IC]ᵀ. bf16 products when
    // bf16_ops, unless low = false (the router: its top-k choice should not
    // flip on rounding).
    void linear_forward(float* y, Operand x, const float* w, int N, int IC, int OC, bool low = true) {
        if (bf16_ops && low) {
            gemm(false, true, N, OC, IC, as_bf16(x, op_x, size_t(N) * IC), IC, WB(w), IC, y, OC, 0.0f, true);
            return;
        }
        CHECK(x.f, "an fp32 product got a bf16 input");
        gemm(false, true, N, OC, IC, x.f, IC, w, IC, y, OC, 0.0f);
    }
    // linear_backward: dx += dy · w (if dx), dw += dyᵀ · x
    void linear_backward(float* dx, float* dw, Operand dy, Operand x, const float* w, int N, int IC, int OC,
                         bool low = true) {
        if (bf16_ops && low) {
            const auto* dyb = as_bf16(dy, op_dy, size_t(N) * OC);
            if (dx) gemm(false, false, N, IC, OC, dyb, OC, WB(w), IC, dx, IC, 1.0f, true);
            gemm(true, false, OC, IC, N, dyb, OC, as_bf16(x, op_x, size_t(N) * IC), IC, dw, IC, 1.0f, true);
            return;
        }
        CHECK(x.f && dy.f, "an fp32 product got a bf16 input");
        if (dx) gemm(false, false, N, IC, OC, dy.f, OC, w, IC, dx, IC, 1.0f);
        gemm(true, false, OC, IC, N, dy.f, OC, x.f, IC, dw, IC, 1.0f);
    }

    void act_quantize(float* xq_, const float* x, int N, int C) {
        if (N == 0) return;
        act_quant<<<blocks_for(N, kRowsPerBlock), kBlock>>>(xq_, x, N, C);
        LAUNCH_CHECK();
    }

    // Model::proj_forward / proj_backward: BitLinear when q (ternary weights,
    // int8 activations).
    void proj_forward(float* y, Operand x, size_t woff, int N, int IC, int OC, bool q) {
        Region r(*this, "matmul fwd");
        if (q) {  // never with bf16 products
            act_quantize(xq.p, x.f, N, IC);
            x = Operand(xq.p);
        }
        linear_forward(y, x, W(woff, q), N, IC, OC);
    }
    void proj_backward(float* dx, Operand dy, Operand x, size_t woff, int N, int IC, int OC, bool q) {
        Region r(*this, "matmul bwd");
        if (q) {
            act_quantize(xq.p, x.f, N, IC);
            x = Operand(xq.p);
        }
        linear_backward(dx, G(woff), dy, x, W(woff, q), N, IC, OC);
    }

    // Into out, outh (bf16) or both; the other may be null.
    void rmsnorm_forward(float* out, __nv_bfloat16* outh, float* rstd, const float* x, const float* w, int N, int C) {
        Region r(*this, "norms, rope, adds");
        rmsnorm_fwd<<<blocks_for(N, kRowsPerBlock), kBlock>>>(out, outh, rstd, x, w, N, C);
        LAUNCH_CHECK();
    }
    void rmsnorm_backward(float* dx, float* dw, const float* dout, const float* x, const float* w,
                          const float* rstd, int N, int C) {
        Region r(*this, "norms, rope, adds");
        rmsnorm_bwd_dx<<<blocks_for(N, kRowsPerBlock), kBlock>>>(dx, dout, x, w, rstd, N, C);
        LAUNCH_CHECK();
        constexpr int R = 64;
        const int nb = (N + R - 1) / R;
        rmsnorm_bwd_dw_partial<<<nb, kBlock>>>(dw_partial.p, dout, x, rstd, N, C, R);
        LAUNCH_CHECK();
        add_partials<<<blocks_for(C), kBlock>>>(dw, dw_partial.p, nb, C);
        LAUNCH_CHECK();
    }

    void rope_apply(float* qkv, int B, int Tn, bool inverse, int pos0 = 0) {
        const int C = cfg.d_model, NH = cfg.n_heads, half = cfg.head_size() / 2;
        const size_t n = size_t(B) * Tn * 2 * NH * half;
        Region r(*this, "norms, rope, adds");
        rope<<<blocks_for(n), kBlock>>>(qkv, rope_cos.p, rope_sin.p, B * Tn, Tn, C, NH, half, inverse, pos0);
        LAUNCH_CHECK();
    }

    void add(float* out, const float* a, const float* b, size_t n) {
        Region r(*this, "norms, rope, adds");
        add_fwd<<<blocks_for(n), kBlock>>>(out, a, b, n);
        LAUNCH_CHECK();
    }

    // A bf16 copy of an activation that is also needed in fp32.
    void to_bf16_region(__nv_bfloat16* out, const float* in, size_t n) {
        Region r(*this, "bf16 copies");
        to_bf16<<<std::min(blocks_for(n), 4096), kBlock>>>(out, in, n);
        LAUNCH_CHECK();
    }

    // Attention of layer activations a: a.atty from a.qkv.
    void attention_forward(LayerActs& a, int B, int Tn) {
        Region r(*this, "attention fwd");
        const int C = cfg.d_model, NH = cfg.n_heads, HS = cfg.head_size();
        const float scale = 1.0f / std::sqrt(float(HS));
        if (fused_attn) {
            fused_attention_forward(tc_attn, a.qkv.p, a.atty.p, a.lse.p, B, Tn, C, NH, HS);
            return;
        }
        // Without: per sequence, the heads as one strided batch of cuBLAS
        // products, through the full matrix of weights.
        float* out = a.atty.p;
        float* att = a.att.p;
        const float* qkv = a.qkv.p;
        const long long TT = (long long)Tn * Tn;
        for (int b = 0; b < B; ++b) {
            const float* q = qkv + size_t(b) * Tn * 3 * C;
            float* S = att + size_t(b) * NH * TT;
            // S = Q · Kᵀ for every head
            gemm_batched(false, true, Tn, Tn, HS, q, 3 * C, HS, q + C, 3 * C, HS, S, Tn, TT, 0.0f, NH);
        }
        const int rows = B * NH * Tn;
        attn_softmax_fwd<<<blocks_for(rows, kRowsPerBlock), kBlock>>>(att, rows, Tn, scale);
        LAUNCH_CHECK();
        for (int b = 0; b < B; ++b) {
            const float* vv = qkv + size_t(b) * Tn * 3 * C + 2 * C;
            // out = P · V
            gemm_batched(false, false, Tn, HS, Tn, att + size_t(b) * NH * TT, Tn, TT, vv, 3 * C, HS,
                         out + size_t(b) * Tn * C, C, HS, 0.0f, NH);
        }
    }

    // attention_forward's backward: accumulates into dqkv_, given dout = d(a.atty).
    void attention_backward(float* dqkv_, const float* dout, const LayerActs& a, int B, int Tn) {
        Region r(*this, "attention bwd");
        const int C = cfg.d_model, NH = cfg.n_heads, HS = cfg.head_size();
        const float scale = 1.0f / std::sqrt(float(HS));
        if (fused_attn) {
            fused_attention_backward(tc_attn, a.qkv.p, a.atty.p, dout, a.lse.p, dsum.p, dqkv_, B, Tn, C, NH, HS);
            return;
        }
        const float* att = a.att.p;
        const float* qkv = a.qkv.p;
        const long long TT = (long long)Tn * Tn;
        for (int b = 0; b < B; ++b) {
            const float* vv = qkv + size_t(b) * Tn * 3 * C + 2 * C;
            // dP = dO · Vᵀ
            gemm_batched(false, true, Tn, Tn, HS, dout + size_t(b) * Tn * C, C, HS, vv, 3 * C, HS,
                         dS.p + size_t(b) * NH * TT, Tn, TT, 0.0f, NH);
        }
        const int rows = B * NH * Tn;
        attn_softmax_bwd<<<blocks_for(rows, kRowsPerBlock), kBlock>>>(dS.p, att, rows, Tn, scale);
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

    // Into out, or outh in bf16 (the other null).
    void swiglu_forward(float* out, __nv_bfloat16* outh, const float* inp, int N, int F) {
        if (N == 0) return;
        Region r(*this, "swiglu");
        swiglu_fwd<<<blocks_for(size_t(N) * F), kBlock>>>(out, outh, inp, N, F);
        LAUNCH_CHECK();
    }
    // Writes dinp (or dinph in bf16), rather than adding to it.
    void swiglu_backward(float* dinp, __nv_bfloat16* dinph, const float* dout, const float* inp, int N, int F) {
        if (N == 0) return;
        Region r(*this, "swiglu");
        swiglu_bwd<<<blocks_for(size_t(N) * F), kBlock>>>(dinp, dinph, dout, inp, N, F);
        LAUNCH_CHECK();
    }

    // The weights as the products use them, after each change: the ternary
    // copy, and with bf16 products the bf16 copy.
    void quantize_weights() {
        {
            Region r(*this, "ternary weights");
            if (n_seg > 0) {
                partial_sums<<<dim3(kPartials, n_seg), kBlock>>>(seg_partial.p, params.p, seg_off.p, seg_len.p,
                                                                 true);
                LAUNCH_CHECK();
                ternary_gamma<<<n_seg, kWarp>>>(seg_gamma.p, seg_partial.p, seg_len.p, kPartials);
                LAUNCH_CHECK();
                ternary_apply<<<dim3(kPartials, n_seg), kBlock>>>(qparams.p, params.p, seg_off.p, seg_len.p,
                                                                 seg_gamma.p);
                LAUNCH_CHECK();
            }
        }
        if (bf16_ops) {
            Region r(*this, "bf16 weights");
            to_bf16<<<std::min(blocks_for(n_params), 4096), kBlock>>>(wbf.p, params.p, n_params);
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
        const size_t att = size_t(B) * cfg.n_heads * Tn * Tn, rows = size_t(B) * cfg.n_heads * Tn;
        acts.resize(cfg.n_layers);
        // An activation that only feeds products: fp32, or bf16 with bf16 products.
        auto act = [&](DevBuf<float>& f, DevBuf<__nv_bfloat16>& h, size_t n) {
            if (bf16_ops) h.resize(n);
            else f.resize(n);
        };
        for (auto& a : acts) {
            act(a.ln1, a.ln1h, N * C); a.rstd1.resize(N); a.qkv.resize(N * 3 * C);
            if (fused_attn) a.lse.resize(rows); else a.att.resize(att);
            a.atty.resize(N * C);
            if (bf16_ops) a.attyh.resize(N * C);
            a.resid1.resize(N * C); a.ln2.resize(N * C); a.rstd2.resize(N);
            if (bf16_ops) a.ln2h.resize(N * C);
            a.resid2.resize(N * C); a.fc.resize(N * 2 * F); act(a.sw, a.swh, N * F);
            if (cfg.moe()) {
                a.rlogits.resize(N * E); a.sel.resize(N * K); a.gate.resize(N * K);
                a.order.resize(NK); a.row_of.resize(NK);
                act(a.xg, a.xgh, NK * C); a.efc.resize(NK * 2 * Fe); act(a.esw, a.eswh, NK * Fe);
                a.starts.resize(E + 1);
                a.ey.resize(NK * C);
            }
        }
        x0.resize(N * C); act(lnf_out, lnfh, N * C); rstdf.resize(N);
        logits.resize(N * V); losses.resize(N);
        tokens.resize(N); targets.resize(N);
        xq.resize(std::max(N, NK) * std::max({C, F, Fe}));
        out_tmp.resize(N * C);
        dresid.resize(N * C); dC.resize(N * C); dqkv.resize(N * 3 * C); datty.resize(N * C);
        act(dfc, dfch, std::max(N * 2 * F, NK * 2 * Fe)); dsw.resize(std::max(N * F, NK * Fe));
        if (fused_attn) dsum.resize(rows); else dS.resize(att);
        dw_partial.resize((N + 63) / 64 * C);
        if (cfg.moe()) { act(dey, deyh, NK * C); dxg.resize(NK * C); drl.resize(N * E); dgate.resize(NK); }
        if (bf16_ops) dlogitsh.resize(N * V);
    }

    void ffn_forward(int l, float* out, int N, bool training) {
        auto& a = acts[l];
        const auto& o = layer[l];
        const int C = cfg.d_model, F = cfg.always_hidden();
        if (F > 0) {
            proj_forward(a.fc.p, pick(a.ln2, a.ln2h), o.wfc, N, C, 2 * F, cfg.ternary);
            swiglu_forward(a.sw.p, a.swh.p, a.fc.p, N, F);
            proj_forward(out, pick(a.sw, a.swh), o.wproj, N, F, C, cfg.ternary);
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
            proj_backward(dsw.p, dout, pick(a.sw, a.swh), o.wproj, N, F, C, cfg.ternary);
            swiglu_backward(dfc.p, dfch.p, dsw.p, a.fc.p, N, F);
            proj_backward(dx, pick(dfc, dfch), pick(a.ln2, a.ln2h), o.wfc, N, C, 2 * F, cfg.ternary);
        }
        if (cfg.moe()) moe_backward(l, dx, dout, N);
    }

    // The experts' products for all experts at once (grouped_gemm.cuh), as
    // proj_forward and proj_backward do them for one expert's rows: expert
    // e's weights at woff + e * per, its rows from starts[e].
    gg::Kind grouped_kind() const {
        return bf16_ops ? gg::Kind::bf16 : compute == CUBLAS_COMPUTE_32F ? gg::Kind::fp32 : gg::Kind::tf32;
    }
    // y[rows of e, OC] = x[rows of e, IC] · w_eᵀ
    void experts_forward(float* y, Operand x, size_t woff, size_t per, const int* starts, int NK, int IC, int OC,
                         bool q) {
        Region r(*this, "matmul fwd");
        if (q) {  // never with bf16 products
            act_quantize(xq.p, x.f, NK, IC);
            x = Operand(xq.p);
        }
        gg::Problem p;
        if (bf16_ops) {
            p.A = as_bf16(x, op_x, size_t(NK) * IC);
            p.B = WB(W(woff, q));
        } else {
            CHECK(x.f, "an fp32 product got a bf16 input");
            p.A = x.f;
            p.B = W(woff, q);
        }
        p.C = y;
        p.lda = IC, p.ldb = IC, p.ldc = OC;
        p.strideB = (long long)per;
        p.N = OC, p.K = IC;
        p.starts = starts, p.groups = cfg.n_experts;
        p.beta = 0.0f;
        gg::launch<false, true, false>(grouped_kind(), p, NK);
        LAUNCH_CHECK();
    }
    // dx[rows of e, IC] = dy · w_e (written, not added), and the weights'
    // gradient: dw_e += dyᵀ · x over e's rows.
    void experts_backward(float* dx, Operand dy, Operand x, size_t woff, size_t per, const int* starts, int NK,
                          int IC, int OC, bool q) {
        Region r(*this, "matmul bwd");
        if (q) {
            act_quantize(xq.p, x.f, NK, IC);
            x = Operand(xq.p);
        }
        gg::Problem d, g;
        if (bf16_ops) {
            d.A = g.A = as_bf16(dy, op_dy, size_t(NK) * OC);
            d.B = WB(W(woff, q));
            g.B = as_bf16(x, op_x, size_t(NK) * IC);
        } else {
            CHECK(x.f && dy.f, "an fp32 product got a bf16 input");
            d.A = g.A = dy.f;
            d.B = W(woff, q);
            g.B = x.f;
        }
        d.C = dx;
        d.lda = OC, d.ldb = IC, d.ldc = IC;
        d.strideB = (long long)per;
        d.N = IC, d.K = OC;
        d.starts = starts, d.groups = cfg.n_experts;
        d.beta = 0.0f;
        gg::launch<false, false, false>(grouped_kind(), d, NK);
        LAUNCH_CHECK();
        g.C = G(woff);
        g.lda = OC, g.ldb = IC, g.ldc = IC;
        g.strideC = (long long)per;
        g.M = OC, g.N = IC;
        g.starts = starts, g.groups = cfg.n_experts;
        g.beta = 1.0f;
        gg::launch<true, false, true>(grouped_kind(), g, NK);
        LAUNCH_CHECK();
    }

    // Routing, then every expert's rows through its two projections. The
    // router's counts never leave the GPU: the grouped products find each
    // expert's rows in a.starts.
    void moe_forward(int l, float* out, int N, bool training) {
        auto& a = acts[l];
        const auto& o = layer[l];
        const int C = cfg.d_model, E = cfg.n_experts, K = cfg.top_k, Fe = cfg.ehidden(), NK = N * K;
        std::optional<Region> routing(std::in_place, *this, "moe routing");
        linear_forward(a.rlogits.p, a.ln2.p, P(o.router), N, C, E, false);
        counts.zero(E);
        moe_route<<<blocks_for(N), kBlock>>>(a.sel.p, a.gate.p, counts.p, a.rlogits.p,
                                             router_bias.p + size_t(l) * E, N, E, K);
        expert_starts<<<1, 1>>>(a.starts.p, counts.p, E);
        LAUNCH_CHECK();
        if (!grouped_experts) {
            // The host needs the counts to size the experts' products. The
            // copy goes first and the grouping of rows behind it, so the
            // GPU groups while the host waits for the copy.
            CUDA_CHECK(cudaMemcpyAsync(counts_h, counts.p, E * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaEventRecord(counts_ready));
        }
        moe_order<<<E, 1024>>>(a.order.p, a.row_of.p, a.sel.p, counts.p, NK);
        LAUNCH_CHECK();
        if (bf16_ops)
            moe_gather<<<blocks_for(size_t(NK) * C), kBlock>>>(a.xgh.p, a.ln2h.p, a.order.p, NK, C, K);
        else
            moe_gather<<<blocks_for(size_t(NK) * C), kBlock>>>(a.xg.p, a.ln2.p, a.order.p, NK, C, K);
        LAUNCH_CHECK();
        if (training) {
            add_counts<<<1, kWarp>>>(step_load.p + size_t(l) * E, counts.p, E);
            LAUNCH_CHECK();
        }
        const size_t efc_per = size_t(2) * Fe * C, eproj_per = size_t(C) * Fe;
        if (grouped_experts) {
            routing.reset();
            experts_forward(a.efc.p, pick(a.xg, a.xgh), o.efc, efc_per, a.starts.p, NK, C, 2 * Fe, cfg.ternary);
            swiglu_forward(bf16_ops ? nullptr : a.esw.p, bf16_ops ? a.eswh.p : nullptr, a.efc.p, NK, Fe);
            experts_forward(a.ey.p, pick(a.esw, a.eswh), o.eproj, eproj_per, a.starts.p, NK, Fe, C, cfg.ternary);
        } else {
            CUDA_CHECK(cudaEventSynchronize(counts_ready));
            a.expert_start.assign(E + 1, 0);
            for (int e = 0; e < E; ++e) a.expert_start[e + 1] = a.expert_start[e] + counts_h[e];
            routing.reset();
            const Operand xg = pick(a.xg, a.xgh), esw = pick(a.esw, a.eswh);
            for (int e = 0; e < E; ++e) {
                const int r0 = a.expert_start[e], n = a.expert_start[e + 1] - r0;
                if (n == 0) continue;
                proj_forward(a.efc.p + size_t(r0) * 2 * Fe, xg + size_t(r0) * C, o.efc + e * efc_per, n, C, 2 * Fe,
                             cfg.ternary);
                swiglu_forward(bf16_ops ? nullptr : a.esw.p + size_t(r0) * Fe,
                               bf16_ops ? a.eswh.p + size_t(r0) * Fe : nullptr, a.efc.p + size_t(r0) * 2 * Fe, n, Fe);
                proj_forward(a.ey.p + size_t(r0) * C, esw + size_t(r0) * Fe, o.eproj + e * eproj_per, n, Fe, C,
                             cfg.ternary);
            }
        }
        Region combine(*this, "moe routing");
        moe_combine<<<blocks_for(size_t(N) * C), kBlock>>>(out, a.ey.p, a.gate.p, a.row_of.p, N, C, K);
        LAUNCH_CHECK();
    }

    void moe_backward(int l, float* dx, const float* dout, int N) {
        auto& a = acts[l];
        const auto& o = layer[l];
        const int C = cfg.d_model, E = cfg.n_experts, K = cfg.top_k, Fe = cfg.ehidden(), NK = N * K;
        {
            Region r(*this, "moe routing");
            moe_bwd_combine<<<blocks_for(NK, kRowsPerBlock), kBlock>>>(dey.p, deyh.p, dgate.p, dout, a.ey.p,
                                                                      a.gate.p, a.row_of.p, NK, C, K);
            LAUNCH_CHECK();
        }
        const size_t efc_per = size_t(2) * Fe * C, eproj_per = size_t(C) * Fe;
        if (grouped_experts) {
            experts_backward(dsw.p, pick(dey, deyh), pick(a.esw, a.eswh), o.eproj, eproj_per, a.starts.p, NK, Fe, C,
                             cfg.ternary);
            swiglu_backward(dfc.p, dfch.p, dsw.p, a.efc.p, NK, Fe);
            experts_backward(dxg.p, pick(dfc, dfch), pick(a.xg, a.xgh), o.efc, efc_per, a.starts.p, NK, C, 2 * Fe,
                             cfg.ternary);
        } else {
            dxg.zero(size_t(NK) * C);
            const Operand dy = pick(dey, deyh), xg = pick(a.xg, a.xgh), esw = pick(a.esw, a.eswh);
            for (int e = 0; e < E; ++e) {
                const int r0 = a.expert_start[e], n = a.expert_start[e + 1] - r0;
                if (n == 0) continue;
                dsw.zero(size_t(n) * Fe);
                proj_backward(dsw.p, dy + size_t(r0) * C, esw + size_t(r0) * Fe, o.eproj + e * eproj_per, n, Fe, C,
                              cfg.ternary);
                swiglu_backward(dfc.p, dfch.p, dsw.p, a.efc.p + size_t(r0) * 2 * Fe, n, Fe);
                proj_backward(dxg.p + size_t(r0) * C, pick(dfc, dfch), xg + size_t(r0) * C, o.efc + e * efc_per, n,
                              C, 2 * Fe, cfg.ternary);
            }
        }
        Region r(*this, "moe routing");
        moe_scatter<<<blocks_for(size_t(N) * C), kBlock>>>(dx, dxg.p, a.row_of.p, N, C, K);
        LAUNCH_CHECK();
        drl.zero(size_t(N) * E);
        moe_router_grad<<<blocks_for(N), kBlock>>>(drl.p, a.gate.p, dgate.p, a.sel.p, N, E, K);
        LAUNCH_CHECK();
        linear_backward(dx, G(o.router), drl.p, a.ln2.p, P(o.router), N, C, E, false);
    }

    // A forward pass whose loss stays on the GPU: the cross-entropy summed
    // over the trained targets is added to *ce_sum, and with unlikelihood
    // training the term's sum to *ul_sum (device addresses). The host only
    // counts the targets it sent (n_valid, for backward's scale).
    void forward_sums(const int* tokens_h, const int* targets_h, int B, int Tn, bool training, double* ce_sum,
                      double* ul_sum) {
        CHECK(targets_h, "GpuTrainer::forward needs targets");
        release_decode();  // training never runs with less memory than without decoding
        allocate(B, Tn);
        B_cur = B;
        T_cur = Tn;
        const int N = B * Tn, C = cfg.d_model, V = cfg.vocab;
        for (int n = 0; n < N; ++n) CHECK(tokens_h[n] >= 0 && tokens_h[n] < V, "token %d out of range", tokens_h[n]);
        if (weights_dirty) quantize_weights();
        upload(tokens.p, tokens_h, N);
        upload(targets.p, targets_h, N);

        embedding_fwd<<<blocks_for(size_t(N) * C), kBlock>>>(x0.p, tokens.p, P(wte), N, C);
        LAUNCH_CHECK();
        const float* x = x0.p;
        for (int l = 0; l < cfg.n_layers; ++l) {
            auto& a = acts[l];
            const auto& o = layer[l];
            rmsnorm_forward(a.ln1.p, a.ln1h.p, a.rstd1.p, x, P(o.ln1), N, C);
            proj_forward(a.qkv.p, pick(a.ln1, a.ln1h), o.wqkv, N, C, 3 * C, attn_ternary());
            if (cfg.rope_layer(l)) rope_apply(a.qkv.p, B, Tn, false);
            attention_forward(a, B, Tn);
            if (bf16_ops) to_bf16_region(a.attyh.p, a.atty.p, size_t(N) * C);
            proj_forward(out_tmp.p, pick(a.atty, a.attyh), o.wo, N, C, C, attn_ternary());
            add(a.resid1.p, x, out_tmp.p, size_t(N) * C);
            rmsnorm_forward(a.ln2.p, a.ln2h.p, a.rstd2.p, a.resid1.p, P(o.ln2), N, C);
            ffn_forward(l, out_tmp.p, N, training);
            add(a.resid2.p, a.resid1.p, out_tmp.p, size_t(N) * C);
            x = a.resid2.p;
        }
        rmsnorm_forward(lnf_out.p, lnfh.p, rstdf.p, x, P(lnf), N, C);
        {
            Region r(*this, "output layer and loss");
            linear_forward(logits.p, pick(lnf_out, lnfh), P(wte), N, C, V);  // tied head
            xent_fwd<<<N, kBlock>>>(logits.p, losses.p, targets.p, V);
            LAUNCH_CHECK();
            sum_trained<<<1, 1024>>>(ce_sum, losses.p, targets.p, N);
            LAUNCH_CHECK();
        }
        has_targets = true;
        n_valid = 0;
        for (int n = 0; n < N; ++n) n_valid += targets_h[n] >= 0;
        parts = {};
        ul_active = false;
        if (!training || ul_alpha <= 0 || n_valid == 0) return;
        unlikelihood_forward(tokens_h, targets_h, B, Tn, ul_sum);
    }

    // forward_sums on forward()'s own sums, copied back: the mean
    // cross-entropy (+ alpha times the term in training), as Model::forward.
    double forward(const int* tokens_h, const int* targets_h, int B, int Tn, bool training) {
        fwd_sums.zero();
        forward_sums(tokens_h, targets_h, B, Tn, training, fwd_sums.p, fwd_sums.p + 1);
        double sums[2];
        to_host(sums, fwd_sums.p, 2);
        parts.ce = n_valid ? sums[0] / n_valid : 0.0;
        if (!ul_active) return parts.ce;
        parts.ul = sums[1] / n_valid;
        return parts.ce + ul_alpha * parts.ul;
    }

    void unlikelihood_forward(const int* tokens_h, const int* targets_h, int B, int Tn, double* ul_sum) {
        const int N = B * Tn, V = cfg.vocab;
        find_repeat_negatives(negatives_h, tokens_h, targets_h, B, Tn, ul_ngram);
        const size_t K = negatives_h.tokens.size();
        parts.fired = negatives_h.fired;
        parts.negatives = K;
        if (K == 0) return;  // the gradient is the cross-entropy's
        check_negatives(negatives_h.rows(ul_alpha), targets_h, N, V);
        neg_row_h.resize(K);
        for (int n = 0; n < N; ++n)
            for (int k = negatives_h.offsets[size_t(n)]; k < negatives_h.offsets[size_t(n) + 1]; ++k)
                neg_row_h[size_t(k)] = n;
        neg_off.resize(size_t(N) + 1);
        neg_tok.resize(K);
        neg_row.resize(K);
        neg_p.resize(K);
        neg_w.resize(K);
        row_w.resize(N);
        row_scale.resize(N);
        row_ul.resize(N);
        to_device(neg_off.p, negatives_h.offsets.data(), size_t(N) + 1);
        to_device(neg_tok.p, negatives_h.tokens.data(), K);
        to_device(neg_row.p, neg_row_h.data(), K);
        {
            Region r(*this, "output layer and loss");
            ul_fwd<<<N, kBlock>>>(logits.p, neg_off.p, neg_tok.p, neg_p.p, neg_w.p, row_w.p, row_scale.p,
                                  row_ul.p, ul_alpha, V);
            LAUNCH_CHECK();
            sum_all<<<1, 1024>>>(ul_sum, row_ul.p, N);
            LAUNCH_CHECK();
        }
        ul_active = true;
    }

    // ---- Incremental decoding, as Model::decode_begin / decode_step ----
    //
    // One token per sequence per step, all at the same position, sequences
    // left-padded to start[b]. A step runs forward()'s layers on B rows,
    // with attention over the caches instead of the whole sequence.

    // The caches' size for one sequence: k and v, every layer, the whole context.
    size_t decode_bytes_per_sequence() const {
        return size_t(2) * cfg.n_layers * cfg.seq_len * cfg.d_model * sizeof(float);
    }

    void release_decode() {
        kv_own.release();
        kcache.clear();
        vcache.clear();
        kv_B = 0;
    }

    // Before reading the gradients: decoding must not have overwritten them.
    void check_grads(const char* what) const {
        CHECK(!grads_scratch, "%s after decoding, which used the gradient buffer: zero_grad() first", what);
    }

    void decode_begin(int B, const std::vector<int>& start) {
        CHECK(B > 0 && int(start.size()) == B, "decode_begin: %zu starts for %d sequences", start.size(), B);
        // Decoding uses the first B rows of the training buffers; allocate()
        // for B rows would free and rebuild them all (B > B_ there).
        if (size_t(B_) * T_ == 0) allocate(B, 1);
        CHECK(size_t(B) <= size_t(B_) * T_, "decode batch %d > the %d rows allocated", B, B_ * T_);
        CHECK(size_t(cfg.seq_len) * sizeof(float) <= 48 * 1024, "decode attention keeps a context of %d scores in "
              "shared memory: at most 12288", cfg.seq_len);
        has_targets = false;  // decoding overwrites the activations backward() would read
        const size_t per = size_t(B) * cfg.seq_len * cfg.d_model, need = 2 * size_t(cfg.n_layers) * per;
        float* base = nullptr;
        if (need <= grads.n) {
            base = grads.p;
            grads_scratch = true;
        } else {
            kv_own.resize(need);  // grows only: one exam's generate() calls share it
            base = kv_own.p;
        }
        kcache.resize(cfg.n_layers);
        vcache.resize(cfg.n_layers);
        for (int l = 0; l < cfg.n_layers; ++l) {
            kcache[l] = base + size_t(2 * l) * per;
            vcache[l] = base + size_t(2 * l + 1) * per;
        }
        kv_start.resize(B);
        to_device(kv_start.p, start.data(), B);
        kv_B = B;
        logits_h.resize(size_t(B) * cfg.vocab);
    }

    const float* decode_step(const int* tokens_h, int pos) {
        const int B = kv_B, N = B, C = cfg.d_model, V = cfg.vocab, NH = cfg.n_heads, HS = cfg.head_size();
        const int S = cfg.seq_len;
        CHECK(B > 0, "decode_step without decode_begin");
        CHECK(pos >= 0 && pos < S, "decode position %d outside the context of %d", pos, S);
        for (int n = 0; n < N; ++n) CHECK(tokens_h[n] >= 0 && tokens_h[n] < V, "token %d out of range", tokens_h[n]);
        if (weights_dirty) quantize_weights();
        upload(tokens.p, tokens_h, N);
        embedding_fwd<<<blocks_for(size_t(N) * C), kBlock>>>(x0.p, tokens.p, P(wte), N, C);
        LAUNCH_CHECK();
        const float scale = 1.0f / std::sqrt(float(HS));
        const float* x = x0.p;
        for (int l = 0; l < cfg.n_layers; ++l) {
            auto& a = acts[l];
            const auto& o = layer[l];
            rmsnorm_forward(a.ln1.p, a.ln1h.p, a.rstd1.p, x, P(o.ln1), N, C);
            proj_forward(a.qkv.p, pick(a.ln1, a.ln1h), o.wqkv, N, C, 3 * C, attn_ternary());
            if (cfg.rope_layer(l)) rope_apply(a.qkv.p, B, 1, false, pos);
            {
                Region r(*this, "attention fwd");
                kv_store<<<blocks_for(size_t(N) * C), kBlock>>>(kcache[l], vcache[l], a.qkv.p, N, S, C, pos);
                LAUNCH_CHECK();
                decode_attn<<<N * NH, kBlock, size_t(pos + 1) * sizeof(float)>>>(
                    a.atty.p, a.qkv.p, kcache[l], vcache[l], kv_start.p, S, C, NH, HS, pos, scale);
                LAUNCH_CHECK();
            }
            if (bf16_ops) to_bf16_region(a.attyh.p, a.atty.p, size_t(N) * C);
            proj_forward(out_tmp.p, pick(a.atty, a.attyh), o.wo, N, C, C, attn_ternary());
            add(a.resid1.p, x, out_tmp.p, size_t(N) * C);
            rmsnorm_forward(a.ln2.p, a.ln2h.p, a.rstd2.p, a.resid1.p, P(o.ln2), N, C);
            ffn_forward(l, out_tmp.p, N, false);
            add(a.resid2.p, a.resid1.p, out_tmp.p, size_t(N) * C);
            x = a.resid2.p;
        }
        rmsnorm_forward(lnf_out.p, lnfh.p, rstdf.p, x, P(lnf), N, C);
        {
            Region r(*this, "output layer and loss");
            linear_forward(logits.p, pick(lnf_out, lnfh), P(wte), N, C, V);  // tied head
        }
        to_host(logits_h.data(), logits.p, size_t(N) * V);
        return logits_h.data();
    }

    void backward(double loss_scale) {
        CHECK(has_targets, "backward() needs a forward() first");
        check_grads("backward()");
        const int B = B_cur, Tn = T_cur, N = B * Tn, C = cfg.d_model, V = cfg.vocab;
        if (n_valid == 0) return;
        const float* xL = acts.back().resid2.p;
        dC.zero(size_t(N) * C);
        {
            Region r(*this, "output layer and loss");
            const float dloss = float(loss_scale / n_valid);
            xent_bwd<<<blocks_for(size_t(N) * V), kBlock>>>(logits.p, dlogitsh.p, targets.p, N, V, dloss,
                                                            ul_active ? row_scale.p : nullptr);
            LAUNCH_CHECK();
            if (ul_active) {
                const int K = int(negatives_h.tokens.size());
                ul_fix<<<blocks_for(size_t(K)), kBlock>>>(logits.p, dlogitsh.p, neg_tok.p, neg_row.p, neg_p.p,
                                                          neg_w.p, row_w.p, K, V, ul_alpha, dloss);
                LAUNCH_CHECK();
                ul_active = false;
            }
            linear_backward(dC.p, G(wte), pick(logits, dlogitsh), pick(lnf_out, lnfh), P(wte), N, C, V);
        }
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
            proj_backward(datty.p, dresid.p, pick(a.atty, a.attyh), o.wo, N, C, C, attn_ternary());
            dqkv.zero(size_t(N) * 3 * C);
            attention_backward(dqkv.p, datty.p, a, B, Tn);
            if (cfg.rope_layer(l)) rope_apply(dqkv.p, B, Tn, true);
            dC.zero(size_t(N) * C);
            proj_backward(dC.p, dqkv.p, pick(a.ln1, a.ln1h), o.wqkv, N, C, 3 * C, attn_ternary());
            rmsnorm_backward(dresid.p, G(o.ln1), dC.p, x_in, P(o.ln1), a.rstd1.p, N, C);
        }
        embedding_bwd<<<blocks_for(size_t(N) * C), kBlock>>>(G(wte), dresid.p, tokens.p, N, C);
        LAUNCH_CHECK();
    }

    // The gradient norm, the update check and the clip scale into state
    // (step_check), all on the GPU.
    void check_step(double max_norm) {
        check_grads("grad_norm()");
        Region r(*this, "gradient norm");
        partial_sums<<<dim3(kPartials, 1), kBlock>>>(norm_partial.p, grads.p, all_off.p, all_len.p, false);
        step_check<<<1, 1>>>(state.p, norm_partial.p, kPartials, max_norm);
        LAUNCH_CHECK();
    }

    double grad_norm() {
        check_step(0.0);
        double norm = 0.0;
        to_host(&norm, state_field<double>(offsetof(StepState, norm)), 1);
        return norm;
    }

    // A training step's start: its loss sums back to zero (zero_grad).
    void begin_step() {
        static_assert(offsetof(StepState, ul) == offsetof(StepState, ce) + sizeof(double), "ce and ul adjacent");
        CUDA_CHECK(cudaMemsetAsync(state_field<double>(offsetof(StepState, ce)), 0, 2 * sizeof(double)));
    }

    // The step's end, as GpuBackend::step and Backend::balance did it on the
    // host: the checks on the GPU, then one copy of them back. With a
    // finite norm and loss, the optimizer step (gradient times the clip
    // scale, read on the GPU) and the router biases nudged from the step's
    // expert loads; otherwise the weights stay and those loads are dropped.
    GpuTrainer::StepStats finish_step(double lr, double beta1, double beta2, double eps, double weight_decay,
                                      double max_norm, double bias_rate) {
        check_step(max_norm);
        CUDA_CHECK(cudaMemcpy(state_h, state.p, sizeof(StepState), cudaMemcpyDeviceToHost));
        GpuTrainer::StepStats out;
        out.ce_sum = state_h->ce;
        out.ul_sum = state_h->ul;
        out.norm = state_h->norm;
        out.ok = state_h->ok != 0;
        const int E = std::max(1, cfg.n_experts);
        if (!out.ok) {
            step_load.zero();
            return out;
        }
        optimizer_step(lr, beta1, beta2, eps, weight_decay);
        if (cfg.moe()) {
            balance_router<<<1, 1>>>(state.p, router_bias.p, step_load.p, interval_load.p, cfg.n_layers, E,
                                     float(bias_rate));
            LAUNCH_CHECK();
        }
        return out;
    }

    void adamw_step(double lr, double beta1, double beta2, double eps, double weight_decay, double grad_scale) {
        const float s = float(grad_scale);
        CUDA_CHECK(cudaMemcpy(state_field<float>(offsetof(StepState, scale)), &s, sizeof(s), cudaMemcpyHostToDevice));
        optimizer_step(lr, beta1, beta2, eps, weight_decay);
    }

    // AdamW (and Muon) with the gradient times the clip scale in state.
    void optimizer_step(double lr, double beta1, double beta2, double eps, double weight_decay) {
        check_grads("adamw_step()");
        ++adam_step;
        const float c1 = float(1.0 / (1.0 - std::pow(beta1, double(adam_step))));
        const float c2 = float(1.0 / (1.0 - std::pow(beta2, double(adam_step))));
        for (size_t i = 0; i < infos.size(); ++i) {
            const ParamInfo& info = infos[i];
            const float wd = info.decay ? float(weight_decay) : 0.0f;
            if (v_at[i] == kNoV) {
                muon_step(info, float(lr), wd);
                continue;
            }
            const int grid = std::min(blocks_for(info.size), 4096);
            Region r(*this, "adamw");
            adamw<<<grid, kBlock>>>(params.p + info.offset, grads.p + info.offset, m.p + info.offset,
                                    v.p + v_at[i], info.size, float(lr), float(beta1), float(beta2), c1, c2,
                                    float(eps), wd, grad_scale());
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
            ns_a.resize(mm * sh.batch);
            if (bf16_ops) {
                ns_xh.resize(info.size);
                ns_ah.resize(mm * sh.batch);
            } else {
                ns_y.resize(info.size);
                ns_b.resize(mm * sh.batch);
            }
        }
    }

    // AdamW::muon_update for one tensor, all its blocks batched through
    // cuBLAS; newton_schulz (train.h) step for step. With bf16 products the
    // products take bf16 copies of X and A (fp32 sums, X itself stays fp32),
    // and cuBLAS adds the b A and a X terms in.
    void muon_step(const ParamInfo& info, float lr, float wd) {
        Region r(*this, "muon");
        const MuonShape sh = muon_shape(info);
        const int mr = sh.mrows(), nc = sh.ncols(), batch = sh.batch;
        const long long mn = (long long)mr * nc, mm = (long long)mr * mr;
        const int grid = std::min(blocks_for(info.size), 4096);
        const int grid_a = std::min(blocks_for(size_t(mm) * batch), 4096);
        float *X = ns_x.p, *Y = ns_y.p, *A = ns_a.p, *B = ns_b.p;
        __nv_bfloat16 *Xh = ns_xh.p, *Ah = ns_ah.p;
        muon_momentum<<<grid, kBlock>>>(X, m.p + info.offset, grads.p + info.offset, batch, sh.r, sh.c, muon_mu,
                                        grad_scale());
        frobenius_normalize<<<batch, kBlock>>>(X, bf16_ops ? Xh : nullptr, size_t(mn));
        LAUNCH_CHECK();
        const float a = 3.4445f, b = -4.7750f, c = 2.0315f;
        for (int s = 0; bf16_ops && s < muon_ns; ++s) {
            gemm_batched_bf16(false, true, mr, mr, nc, Xh, nc, mn, Xh, nc, mn, A, mr, mm, 1.0f, 0.0f, batch);  // A = X Xᵀ
            to_bf16<<<grid_a, kBlock>>>(Ah, A, size_t(mm) * batch);
            gemm_batched_bf16(false, false, mr, mr, mr, Ah, mr, mm, Ah, mr, mm, A, mr, mm, c, b, batch);  // A = b A + c A²
            to_bf16<<<grid_a, kBlock>>>(Ah, A, size_t(mm) * batch);
            gemm_batched_bf16(false, false, mr, nc, mr, Ah, mr, mm, Xh, nc, mn, X, nc, mn, 1.0f, a, batch);  // X = a X + A X
            if (s + 1 < muon_ns) to_bf16<<<grid, kBlock>>>(Xh, X, info.size);
            LAUNCH_CHECK();
        }
        for (int s = 0; !bf16_ops && s < muon_ns; ++s) {
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

GpuTrainer::GpuTrainer(const Config& cfg, const std::vector<ParamInfo>& infos, Matmul matmul, bool fused_attention)
    : impl_(std::make_unique<Impl>(cfg, infos, matmul, fused_attention)) {}

GpuTrainer::~GpuTrainer() = default;

void GpuTrainer::set_params(const float* p) {
    to_device(impl_->params.p, p, impl_->n_params);
    impl_->weights_dirty = true;
}
void GpuTrainer::get_params(float* p) const { to_host(p, impl_->params.p, impl_->n_params); }
void GpuTrainer::get_grads(float* g) const {
    impl_->check_grads("get_grads()");
    to_host(g, impl_->grads.p, impl_->n_params);
}
void GpuTrainer::get_ternary_weights(float* q) {
    CHECK(impl_->cfg.ternary, "get_ternary_weights() needs a ternary model");
    if (impl_->weights_dirty) impl_->quantize_weights();
    to_host(q, impl_->qparams.p, impl_->n_params);
}

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
void GpuTrainer::set_unlikelihood(double alpha, int ngram) {
    CHECK(alpha >= 0 && ngram >= 2, "unlikelihood: alpha %g, ngram %d", alpha, ngram);
    impl_->ul_alpha = alpha;
    impl_->ul_ngram = ngram;
}
GpuTrainer::ForwardParts GpuTrainer::last_parts() const { return impl_->parts; }
void GpuTrainer::zero_grad() {
    impl_->grads.zero();
    impl_->grads_scratch = false;
    impl_->begin_step();
}
void GpuTrainer::forward_train(const int* tokens, const int* targets, int B, int Tn) {
    Impl& m = *impl_;
    m.forward_sums(tokens, targets, B, Tn, true, m.state_field<double>(offsetof(StepState, ce)),
                   m.state_field<double>(offsetof(StepState, ul)));
}
GpuTrainer::StepStats GpuTrainer::finish_step(double lr, double beta1, double beta2, double eps,
                                              double weight_decay, double max_norm, double bias_rate) {
    return impl_->finish_step(lr, beta1, beta2, eps, weight_decay, max_norm, bias_rate);
}

void GpuTrainer::decode_begin(int B, const std::vector<int>& start) { impl_->decode_begin(B, start); }
const float* GpuTrainer::decode_step(const int* tokens, int pos) { return impl_->decode_step(tokens, pos); }
int GpuTrainer::decode_capacity(int max_batch) const {
    const Impl& m = *impl_;
    const size_t per = m.decode_bytes_per_sequence();
    // In the gradient buffer (decode_begin), or else in free memory: keep a
    // margin there for cuBLAS and the buffers a first decode allocates when
    // nothing is allocated yet. Caches already held count as free.
    const size_t in_grads = m.grads.n * sizeof(float) / per;
    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    constexpr size_t kMargin = size_t(256) << 20;
    const size_t avail = free_b + m.kv_own.n * sizeof(float);
    size_t fit = std::max(in_grads, avail > kMargin ? (avail - kMargin) / per : 0);
    if (size_t(m.B_) * m.T_ > 0) fit = std::min(fit, size_t(m.B_) * m.T_);  // decode uses the training rows
    return int(std::min(fit, size_t(std::max(0, max_batch))));
}
size_t GpuTrainer::decode_bytes_per_sequence() const { return impl_->decode_bytes_per_sequence(); }
size_t GpuTrainer::free_device_bytes() {
    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    return free_b;
}
double GpuTrainer::grad_norm() { return impl_->grad_norm(); }
void GpuTrainer::adamw_step(double lr, double beta1, double beta2, double eps, double weight_decay,
                            double grad_scale) {
    impl_->adamw_step(lr, beta1, beta2, eps, weight_decay, grad_scale);
}

void GpuTrainer::set_muon(bool on, double momentum, int ns_steps) { impl_->set_muon(on, momentum, ns_steps); }

// [layer][expert] doubles from a device array of counts, which is then cleared.
static std::vector<std::vector<double>> take_counts(DevBuf<unsigned long long>& counts, int L, int E) {
    std::vector<unsigned long long> flat(size_t(L) * E);
    to_host(flat.data(), counts.p, flat.size());
    counts.zero();
    std::vector<std::vector<double>> out(L, std::vector<double>(E));
    for (int l = 0; l < L; ++l)
        for (int e = 0; e < E; ++e) out[l][e] = double(flat[size_t(l) * E + e]);
    return out;
}
std::vector<std::vector<double>> GpuTrainer::take_expert_load() {
    return take_counts(impl_->step_load, impl_->cfg.n_layers, std::max(1, impl_->cfg.n_experts));
}
std::vector<std::vector<double>> GpuTrainer::take_interval_load() {
    return take_counts(impl_->interval_load, impl_->cfg.n_layers, std::max(1, impl_->cfg.n_experts));
}
double GpuTrainer::last_load_ratio() const {
    double r = 1.0;
    to_host(&r, impl_->state_field<double>(offsetof(StepState, load_ratio)), 1);
    return r;
}
std::vector<std::vector<float>> GpuTrainer::get_router_bias() const {
    const int L = impl_->cfg.n_layers, E = std::max(1, impl_->cfg.n_experts);
    std::vector<float> flat(size_t(L) * E);
    to_host(flat.data(), impl_->router_bias.p, flat.size());
    std::vector<std::vector<float>> out(L);
    for (int l = 0; l < L; ++l) out[l].assign(flat.begin() + size_t(l) * E, flat.begin() + size_t(l + 1) * E);
    return out;
}

size_t GpuTrainer::num_params() const { return impl_->n_params; }
size_t GpuTrainer::device_bytes() const { return g_device_bytes; }

void GpuTrainer::set_profile(bool on) { impl_->profiling = on; }
void GpuTrainer::set_grouped_experts(bool on) { impl_->grouped_experts = on; }
std::vector<std::pair<std::string, double>> GpuTrainer::take_profile() { return impl_->take_profile(); }

bool fused_attention_fits(int head_size) { return head_size <= attn::kMaxHS; }

bool fused_attention_on_tensor_cores(Matmul matmul, int head_size) {
    return matmul != Matmul::fp32 && head_size % 16 == 0 && head_size <= attn_tc::kMaxHS;
}

void gpu_fused_attention(const float* qkv, const float* dout, float* out, float* lse, float* dqkv, int B, int Tn,
                         int NH, int HS, bool tensor_cores) {
    CHECK(HS <= attn::kMaxHS && (!tensor_cores || fused_attention_on_tensor_cores(Matmul::tf32, HS)),
          "no fused attention kernels for heads of %d", HS);
    const int C = NH * HS;
    const size_t N = size_t(B) * Tn, rows = size_t(B) * NH * Tn;
    DevBuf<float> d_qkv, d_dout, d_out, d_lse, d_dsum, d_dqkv;
    auto up = [](DevBuf<float>& d, const float* h, size_t n) {
        d.resize(n);
        CUDA_CHECK(cudaMemcpy(d.p, h, n * sizeof(float), cudaMemcpyHostToDevice));
    };
    auto down = [](float* h, const DevBuf<float>& d, size_t n) {
        CUDA_CHECK(cudaMemcpy(h, d.p, n * sizeof(float), cudaMemcpyDeviceToHost));
    };
    up(d_qkv, qkv, N * 3 * C);
    up(d_dout, dout, N * C);
    up(d_dqkv, dqkv, N * 3 * C);
    d_out.resize(N * C);
    d_lse.resize(rows);
    d_dsum.resize(rows);
    fused_attention_prepare(HS, tensor_cores);
    fused_attention_forward(tensor_cores, d_qkv.p, d_out.p, d_lse.p, B, Tn, C, NH, HS);
    fused_attention_backward(tensor_cores, d_qkv.p, d_out.p, d_dout.p, d_lse.p, d_dsum.p, d_dqkv.p, B, Tn, C, NH, HS);
    down(out, d_out, N * C);
    down(lse, d_lse, rows);
    down(dqkv, d_dqkv, N * 3 * C);
}

void gpu_grouped_product(Matmul matmul, GroupedOp op, const std::vector<float>& A, const std::vector<float>& B,
                         std::vector<float>& C, const std::vector<int>& starts, int M, int N, int K, int lda, int ldb,
                         int ldc, long long stride, float beta, int offset) {
    const bool bf16 = matmul == Matmul::bf16;
    // The operands, offset elements into their buffers (offset 1 makes the
    // rows unaligned, for the fallback kernel); in bf16 for bf16 products.
    DevBuf<float> fa, fb, fc;
    DevBuf<__nv_bfloat16> ha, hb;
    DevBuf<int> ds;
    auto up = [&](DevBuf<float>& f, DevBuf<__nv_bfloat16>& h, const std::vector<float>& x) -> const void* {
        f.resize(x.size() + offset);
        to_device(f.p + offset, x.data(), x.size());
        if (!bf16) return f.p + offset;
        h.resize(x.size() + offset);
        if (x.empty()) return h.p + offset;
        to_bf16<<<std::min(blocks_for(x.size()), 4096), kBlock>>>(h.p + offset, f.p + offset, x.size());
        LAUNCH_CHECK();
        return h.p + offset;
    };
    gg::Problem p;
    p.A = up(fa, ha, A);
    p.B = up(fb, hb, B);
    fc.resize(C.size());
    to_device(fc.p, C.data(), C.size());
    ds.resize(starts.size());
    to_device(ds.p, starts.data(), starts.size());
    p.C = fc.p;
    p.lda = lda, p.ldb = ldb, p.ldc = ldc;
    p.M = M, p.N = N, p.K = K;
    p.starts = ds.p;
    p.groups = int(starts.size()) - 1;
    p.beta = beta;
    const gg::Kind kind = bf16 ? gg::Kind::bf16 : matmul == Matmul::tf32 ? gg::Kind::tf32 : gg::Kind::fp32;
    const int rows = starts.back();
    if (op == GroupedOp::weight_grad) {
        p.strideC = stride;
        gg::launch<true, false, true>(kind, p, rows);
    } else {
        p.strideB = stride;
        if (op == GroupedOp::forward) gg::launch<false, true, false>(kind, p, rows);
        else gg::launch<false, false, false>(kind, p, rows);
    }
    LAUNCH_CHECK();
    to_host(C.data(), fc.p, C.size());
}

std::string GpuTrainer::device_name() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    const unsigned long long tenths = prop.totalGlobalMem / 100000000ull;  // in units of 0.1 GB
    return std::string(prop.name) + " (" + std::to_string(tenths / 10) + "." + std::to_string(tenths % 10) +
           " GB, compute " + std::to_string(prop.major) + "." + std::to_string(prop.minor) + ")";
}
