// attention_ref.h — causal attention in double precision, the reference for
// the fused attention kernels: on the CPU emulator (test_attention.h) and
// on the GPU (test_gpu.cpp). Random inputs, ops.h's attention forward and
// backward, and the per-row log-sum-exp the kernels keep for their
// backward pass.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "../src/ops.h"

namespace attention_ref {

// max |got - want| / max |want| over the columns [c0, c1) of rows of width w.
inline double rel_err(const std::vector<float>& got, const std::vector<double>& want, int w, int c0, int c1) {
    if (got.size() != want.size()) return INFINITY;
    double diff = 0, mx = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        const int c = int(i % size_t(w));
        if (c < c0 || c >= c1) continue;
        if (!std::isfinite(got[i]) || !std::isfinite(want[i])) return INFINITY;
        diff = std::max(diff, std::abs(got[i] - want[i]));
        mx = std::max(mx, std::abs(want[i]));
    }
    return mx > 0 ? diff / mx : diff == 0 ? 0 : INFINITY;
}

// One random case and its exact results. The float inputs are what the
// kernels get; dqkv starts nonzero, since the kernels add into it.
struct Case {
    int B, NH, HS, Tn, C;
    std::vector<float> qkv, dout, dqkv0;          // inputs
    std::vector<double> out, lse, dqkv;           // reference results
};

// A case's random inputs, rounded to float: the exact results are for the
// inputs the kernels get. dqkv is left at dqkv0, for the backward to add to.
inline Case make_inputs(int B, int NH, int HS, int Tn, uint64_t seed) {
    Case k{B, NH, HS, Tn, NH * HS, {}, {}, {}, {}, {}, {}};
    const int C = k.C, N = B * Tn;
    Rng rng(seed);
    k.qkv.resize(size_t(N) * 3 * C);
    k.dout.resize(size_t(N) * C);
    k.dqkv0.resize(size_t(N) * 3 * C);
    for (auto& x : k.qkv) x = float(rng.normal());
    for (auto& x : k.dout) x = float(rng.normal());
    for (auto& x : k.dqkv0) x = float(0.1 * rng.normal());
    k.dqkv.assign(k.dqkv0.begin(), k.dqkv0.end());
    return k;
}

// The same case with its exact results from ops.h in double.
inline Case make_case(int B, int NH, int HS, int Tn, uint64_t seed) {
    Case k = make_inputs(B, NH, HS, Tn, seed);
    const int C = k.C, N = B * Tn;
    const std::vector<double> qkv(k.qkv.begin(), k.qkv.end()), dout(k.dout.begin(), k.dout.end());
    k.out.resize(size_t(N) * C);
    std::vector<double> att(size_t(B) * NH * Tn * Tn);
    attention_forward(k.out.data(), att.data(), qkv.data(), B, Tn, C, NH);
    attention_backward(k.dqkv.data(), dout.data(), att.data(), qkv.data(), B, Tn, C, NH);
    k.lse.resize(size_t(B) * NH * Tn);
    const double scale = 1.0 / std::sqrt(double(HS));
    for (int b = 0; b < B; ++b)
        for (int h = 0; h < NH; ++h)
            for (int t = 0; t < Tn; ++t) {
                std::vector<double> s(size_t(t) + 1);
                for (int j = 0; j <= t; ++j) {
                    double d = 0;
                    for (int i = 0; i < HS; ++i)
                        d += qkv[(size_t(b) * Tn + t) * 3 * C + h * HS + i] *
                             qkv[(size_t(b) * Tn + j) * 3 * C + C + h * HS + i];
                    s[size_t(j)] = d * scale;
                }
                const double mx = *std::max_element(s.begin(), s.end());
                double sum = 0;
                for (double x : s) sum += std::exp(x - mx);
                k.lse[(size_t(b) * NH + h) * Tn + t] = mx + std::log(sum);
            }
    return k;
}

// The kernels' errors against the reference.
struct Errors {
    double out, lse, dq, dk, dv;
};

inline Errors errors(const Case& k, const std::vector<float>& out, const std::vector<float>& lse,
                     const std::vector<float>& dqkv) {
    const int C = k.C;
    return {rel_err(out, k.out, C, 0, C), rel_err(lse, k.lse, 1, 0, 1), rel_err(dqkv, k.dqkv, 3 * C, 0, C),
            rel_err(dqkv, k.dqkv, 3 * C, C, 2 * C), rel_err(dqkv, k.dqkv, 3 * C, 2 * C, 3 * C)};
}

}  // namespace attention_ref
