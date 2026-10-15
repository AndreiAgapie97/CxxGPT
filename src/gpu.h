// gpu.h — training on an NVIDIA GPU (implemented in gpu.cu).
//
// GpuTrainer runs the same model as Model<float> (model.h): same Config,
// same flat parameter layout, same forward and backward math, so a
// checkpoint moves freely between CPU and GPU, and the CPU's evaluation,
// sampling and ternary inference engine all work on GPU-trained weights.
//
// The weights, the optimizer state and the gradients are fp32. Matrix
// products go through cuBLAS with fp32 outputs and sums; Matmul says how
// their inputs are multiplied:
// - fp32: exactly, which the tests use to compare against the CPU engine;
// - tf32 (GpuTrainer's default; train --gpu's for ternary models): on the
//   tensor cores, each input rounded to 11 significant bits;
// - bf16 (train --gpu's default): the model's projections take inputs
//   stored in bf16 (8 significant bits): a bf16 copy of the weights, refreshed after each
//   step, and activations written in bf16. The tensor cores multiply those
//   at twice the TF32 rate (1.45x the tokens/s of tf32 on the night-1
//   model). Muon's products take bf16 inputs too; the router and
//   attention stay as with tf32. Not for ternary models yet.
//
// Fused attention (fused_attention) never stores the Tn x Tn matrix of
// attention weights. With TF32 or bf16 matmuls it multiplies on the tensor
// cores (attention_tc.cuh); with exact fp32, or a head size that is not a
// multiple of 16, it runs the fp32 kernels (attention.cuh). train --gpu
// uses it by default; GpuTrainer's own default is cuBLAS products through
// the full matrix, the second reference the tests compare it with.
//
// This header is plain C++, so the rest of the code (built by g++ or
// clang++) can use the GPU without including any CUDA header.
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "config.h"

enum class Matmul { fp32, tf32, bf16 };

class GpuTrainer {
public:
    GpuTrainer(const Config& cfg, const std::vector<ParamInfo>& infos, Matmul matmul = Matmul::tf32,
               bool fused_attention = false);
    // Preserve the earlier precision-selector API and its cuBLAS attention.
    GpuTrainer(const Config& cfg, const std::vector<ParamInfo>& infos, bool tf32)
        : GpuTrainer(cfg, infos, tf32 ? Matmul::tf32 : Matmul::fp32, false) {}
    ~GpuTrainer();
    GpuTrainer(const GpuTrainer&) = delete;
    GpuTrainer& operator=(const GpuTrainer&) = delete;

    // Copies between host and device. Arrays have num_params() floats.
    void set_params(const float* params);
    void get_params(float* params) const;
    void get_grads(float* grads) const;
    // AdamW moments and step count. m and v may be null (a fresh optimizer).
    void set_adam_state(long step, const float* m, const float* v);
    void get_adam_state(long* step, float* m, float* v) const;
    void set_router_bias(const std::vector<std::vector<float>>& bias);  // [layer][expert]

    // Mean cross-entropy over the targets that are not -1, as Model::forward.
    // training = true counts expert loads for the router bias.
    double forward(const int* tokens, const int* targets, int B, int Tn, bool training = false);
    // Accumulates d(loss * loss_scale) into the gradients, as Model::backward.
    void backward(double loss_scale = 1.0);
    void zero_grad();
    // Global L2 norm of the gradients.
    double grad_norm();
    // One AdamW step (same update as AdamW::update in train.h), with every
    // gradient multiplied by grad_scale first (gradient clipping).
    void adamw_step(double lr, double beta1, double beta2, double eps, double weight_decay,
                    double grad_scale);
    // With Muon on, adamw_step gives the Muon matrices (muon_matrix in
    // config.h) Muon steps instead, as AdamW::update with muon = true; their
    // momentum lives in the AdamW m. Off by default.
    void set_muon(bool on, double momentum = 0.95, int ns_steps = 5);
    // Tokens routed to each expert since the last call, [layer][expert].
    std::vector<std::vector<double>> take_expert_load();

    // Incremental decoding with a KV cache on the GPU, as Model::decode_begin
    // and decode_step (generate.h runs it through GpuDecoder, below): B
    // sequences, one token each per step, sequence b's real tokens starting
    // at start[b]. Products are as in training (Matmul), attention over the
    // cache in fp32. The caches go in the gradient buffer when they fit
    // (between steps it holds nothing: each step starts with zero_grad(),
    // and until then backward, grad_norm, adamw_step and get_grads refuse
    // to run), else in memory of their own, freed by the next forward().
    void decode_begin(int B, const std::vector<int>& start);
    // Feeds tokens[b] at position pos (0, 1, 2, ... in order); returns the
    // logits for the next token, [B, vocab], in host memory, valid until the
    // next call.
    const float* decode_step(const int* tokens, int pos);
    // How many sequences' caches fit (in the gradient buffer or the GPU
    // memory free now), at most max_batch; 0 if not even one does.
    int decode_capacity(int max_batch) const;
    size_t decode_bytes_per_sequence() const;  // the caches of one sequence
    static size_t free_device_bytes();         // as the GPU reports it now

    // Profiling, off by default: the GPU time of each part of the step
    // (attention, matrix products, MoE routing, optimizer, ...), measured
    // with CUDA events. take_profile returns seconds per part since its last
    // call.
    void set_profile(bool on);
    std::vector<std::pair<std::string, double>> take_profile();

    size_t num_params() const;
    size_t device_bytes() const;  // memory this trainer holds on the GPU
    static std::string device_name();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// A GpuTrainer as generate() and eval_reasoning() (generate.h) take a model:
// its Config and the two decoding calls. So the evaluation code is the same
// on the CPU's Model and on the GPU.
struct GpuDecoder {
    Config cfg;
    GpuTrainer& gpu;
    void decode_begin(int B, const std::vector<int>& start) { gpu.decode_begin(B, start); }
    const float* decode_step(const int* tokens, int pos) { return gpu.decode_step(tokens, pos); }
};

// Whether the fused attention kernels take this head size (up to 128).
bool fused_attention_fits(int head_size);

// Whether fused attention runs on the tensor cores (attention_tc.cuh): with
// TF32 or bf16 matmuls, for head sizes that are a multiple of 16 up to 128.
// Otherwise it runs attention.cuh's fp32 kernels.
bool fused_attention_on_tensor_cores(Matmul matmul, int head_size);

// The fused attention kernels on their own, on host arrays (for tests):
// out and lse from qkv [B·Tn, 3·NH·HS], then dqkv += the backward pass of
// dout, as GpuTrainer runs them. tensor_cores picks attention_tc.cuh's
// kernels over attention.cuh's.
void gpu_fused_attention(const float* qkv, const float* dout, float* out, float* lse, float* dqkv, int B, int Tn,
                         int NH, int HS, bool tensor_cores);
