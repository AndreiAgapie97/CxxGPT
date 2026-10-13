// gpu.h — training on an NVIDIA GPU (implemented in gpu.cu).
//
// GpuTrainer runs the same model as Model<float> (model.h): same Config,
// same flat parameter layout, same forward and backward math, so a
// checkpoint moves freely between CPU and GPU, and the CPU's evaluation,
// sampling and ternary inference engine all work on GPU-trained weights.
//
// Everything stays in fp32. Matrix products go through cuBLAS; with tf32 on
// (the default) they use TF32 tensor cores, about 10 significant bits per
// product, which is the usual trade for training speed. tf32 = false gives
// plain fp32, which the tests use to compare against the CPU engine.
//
// This header is plain C++, so the rest of the code (built by g++ or
// clang++) can use the GPU without including any CUDA header.
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "config.h"

class GpuTrainer {
public:
    GpuTrainer(const Config& cfg, const std::vector<ParamInfo>& infos, bool tf32 = true);
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

    size_t num_params() const;
    size_t device_bytes() const;  // memory this trainer holds on the GPU
    static std::string device_name();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
