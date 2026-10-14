// gpu.h — training on an NVIDIA GPU (implemented in gpu.cu).
//
// GpuTrainer runs the same model as Model<float> (model.h): same Config,
// same flat parameter layout, same forward and backward math, so a
// checkpoint moves freely between CPU and GPU, and the CPU's evaluation,
// sampling and ternary inference engine all work on GPU-trained weights.
//
// Everything is stored in fp32. Matrix products go through cuBLAS, with fp32
// inputs, outputs and sums; Matmul says how the inputs are multiplied:
// - fp32: exactly, which the tests use to compare against the CPU engine;
// - tf32 (the default): on the tensor cores, each input rounded to 11
//   significant bits, the usual trade for training speed;
// - bf16: on the tensor cores at twice tf32's rate (RTX 40xx), each input
//   rounded to 8 significant bits. The rounding happens inside cuBLAS, so
//   whether it is faster overall is for the GPU to show (scripts/probe.sh).
//
// Fused attention (attention.cuh) is opt-in until NVIDIA runtime validation.
// It never stores the Tn x Tn matrix of attention weights; the default
// retains cuBLAS products through that matrix and supplies a second reference.
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
