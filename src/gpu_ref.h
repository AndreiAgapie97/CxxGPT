// gpu_ref.h — the model in double precision on the GPU, written for
// clarity, not speed (gpu_ref.cu): the reference the tests check the fast
// GPU code (gpu.cu) against, so that no CPU engine is needed for it.
//
// Every kernel computes one output value per thread with plain loops, and
// shares no code with gpu.cu's kernels: an error in one is not repeated in
// the other. The reference only runs forward. Gradients are checked by
// finite differences of its loss: for a random direction v, the change of
// the loss along v (the loss at w + h v minus at w - h v, over 2h) must
// equal the gradient's projection g · v. In double precision that holds to
// about 1e-9, far below the float code's differences, and over several
// random directions it measures the same relative error as comparing whole
// gradients.
//
// Ternary BitLinear layers train with the straight-through estimator: the
// gradient skips the rounding. A finite difference of the real forward would
// see the rounding instead, so Quant::record keeps every quantization offset
// (q(x) - x for each projection's input, q(W) - W for the weights) at the
// starting weights, and Quant::frozen adds those offsets instead of
// quantizing. The frozen loss equals the real one at the starting weights,
// and its derivative is exactly the straight-through gradient. The same
// goes for MoE routing: record keeps each token's chosen experts, and frozen
// reuses them (the training gradient treats the choice as fixed too).
//
// This header is plain C++ (no CUDA headers), like gpu.h.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "config.h"

class GpuReference {
public:
    GpuReference(const Config& cfg, const std::vector<ParamInfo>& infos);
    ~GpuReference();
    GpuReference(const GpuReference&) = delete;
    GpuReference& operator=(const GpuReference&) = delete;

    size_t num_params() const;
    void set_params(const std::vector<double>& params);
    void set_params(const float* params);
    void set_router_bias(const std::vector<std::vector<float>>& bias);  // [layer][expert]

    enum class Quant { exact, record, frozen };
    struct Result {
        double loss = 0;  // ce + alpha · ul
        double ce = 0;    // mean cross-entropy over the targets >= 0
        double ul = 0;    // the unlikelihood term's mean over those targets (alpha not applied)
        size_t fired = 0, negatives = 0;               // positions with a negative, negatives
        std::vector<std::vector<double>> expert_load;  // [layer][expert] tokens routed
    };
    // Model::forward(tokens, targets, B, Tn, training): with ul_alpha > 0,
    // the unlikelihood term over the negatives of unlikelihood.h's
    // definition (found here independently). targets may be null: logits only.
    Result forward(const int* tokens, const int* targets, int B, int Tn, double ul_alpha = 0, int ul_ngram = 8,
                   Quant quant = Quant::exact);
    // The last forward's logits [B·Tn, vocab], before the softmax.
    std::vector<double> logits() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// AdamW::update (and with muon, its Muon steps for the muon_matrix tensors:
// Nesterov momentum in m, Newton–Schulz, 0.2 sqrt(max(rows, cols))), in
// double on the GPU, on a gradient already multiplied by the clip scale.
// step is the step number after this update (1 for the first).
struct ReferenceOptimizer {
    double beta1 = 0.9, beta2 = 0.95, eps = 1e-8, weight_decay = 0.1;
    bool muon = false;
    double muon_momentum = 0.95;
    int muon_steps = 5;
};
void reference_optimizer_step(const ReferenceOptimizer& opt, const std::vector<ParamInfo>& infos, double lr, long step,
                              std::vector<double>& params, const std::vector<double>& grads, std::vector<double>& m,
                              std::vector<double>& v);

// Causal attention in double for qkv [B·Tn, 3·NH·HS]: out [B·Tn, NH·HS], the
// rows' log-sum-exp of the scaled scores lse [B, NH, Tn], and dqkv += the
// backward pass of dout. For the fused attention kernels' tests.
void reference_attention(const std::vector<double>& qkv, const std::vector<double>& dout, std::vector<double>& out,
                         std::vector<double>& lse, std::vector<double>& dqkv, int B, int Tn, int NH, int HS);

// The ternary weights of one tensor copy, in float exactly as the CPU engine
// computed them (ops.h's ternary_quantize: gamma = mean |w| + 1e-8, then
// clip(round(w / gamma), -1, 1) · gamma), for the ternary codes test (#40).
void reference_ternary(const float* w, float* wq, size_t n);

// The sum of squares of x, in double: for gradient norms.
double reference_sum_squares(const std::vector<float>& x);
