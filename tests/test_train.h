// Optimizer and data-boundary tests, independent of model gradient formulas.
#pragma once

#include "../src/train.h"
#include "test_util.h"

inline Config training_test_config(bool ternary = false) {
    Config c;
    c.vocab = 13; c.seq_len = 8; c.d_model = 8; c.n_heads = 2;
    c.n_layers = 1; c.ffn_hidden = 12; c.ternary = ternary;
    return c;
}

inline void test_clipping_reference() {
    Model<float> model(training_test_config());
    for (float magnitude : {1.0f, 1e20f, 1e-30f}) {
        for (size_t i = 0; i < model.grads.size(); ++i)
            model.grads[i] = (i % 2 ? -magnitude : magnitude);
        const auto before = model.grads;
        double sq = 0;
        for (float g : before) sq += double(g) * double(g);
        const double ref = std::sqrt(sq);
        const double limit = ref / 2;
        const double got = clip_grad_norm(model, limit);
        EXPECT(std::isfinite(got) && std::abs(got / ref - 1) < 1e-12,
               "gradient norm %g vs double reference %g", got, ref);
        for (size_t i = 0; i < before.size(); ++i)
            EXPECT(std::abs(double(model.grads[i]) / before[i] - 0.5) < 1e-6,
                   "clipping magnitude %g lost direction at %zu", double(magnitude), i);
    }
    model.grads.assign(model.grads.size(), 3.0f);
    const auto before = model.grads;
    clip_grad_norm(model, 0);  // disabled
    EXPECT(model.grads == before, "disabled clipping changed gradients");
    model.zero_grad();
    EXPECT(clip_grad_norm(model, 1) == 0, "zero gradient has nonzero norm");
}

inline void test_adamw_reference() {
    Model<float> model(training_test_config(true));
    model.init(91);
    std::vector<double> p(model.params.begin(), model.params.end());
    std::vector<double> m(p.size(), 0), v(p.size(), 0);
    AdamW opt;
    opt.weight_decay = 0.17;
    for (int step = 1; step <= 4; ++step) {
        const double lr = 0.003 / step;
        for (size_t i = 0; i < p.size(); ++i) model.grads[i] = float(std::sin(double(i + step)) * 0.3);
        for (const auto& info : model.infos)
            for (size_t i = info.offset; i < info.offset + info.size; ++i) {
                const double g = model.grads[i];
                m[i] = opt.beta1 * m[i] + (1 - opt.beta1) * g;
                v[i] = opt.beta2 * v[i] + (1 - opt.beta2) * g * g;
                const double mh = m[i] / (1 - std::pow(opt.beta1, step));
                const double vh = v[i] / (1 - std::pow(opt.beta2, step));
                p[i] -= lr * (mh / (std::sqrt(vh) + opt.eps) + (info.decay ? opt.weight_decay * p[i] : 0));
            }
        model.weights_dirty = false;
        opt.update(model, lr);
        EXPECT(opt.step == step && model.weights_dirty, "optimizer step or quantization invalidation failed");
        EXPECT(max_abs_diff(model.params.data(), p.data(), p.size()) < 3e-7,
               "AdamW parameters disagree with double reference");
        EXPECT(max_abs_diff(opt.m.data(), m.data(), p.size()) < 3e-8 &&
               max_abs_diff(opt.v.data(), v.data(), p.size()) < 3e-8,
               "AdamW moments disagree with double reference");
    }
}

inline void test_sampling_endpoints() {
    TokenData data;
    data.vocab = 13;
    data.tokens = {0, 1, 2, 3, 4, 5, 6, 7};
    data.mask = {1, 1, 1, 0, 1, 1, 1, 1};
    Rng rng(123);
    int inp[3], tgt[3];
    bool first = false, last = false;
    for (int i = 0; i < 200; ++i) {
        data.sample_batch(rng, 1, 3, 1, 7, inp, tgt);
        const int start = inp[0];
        first |= start == 1;
        last |= start == 3;
        EXPECT(start >= 1 && start <= 3, "sample outside requested range: %d", start);
        for (int t = 0; t < 3; ++t)
            EXPECT(inp[t] == start + t && tgt[t] == (data.mask[start + t + 1] ? start + t + 1 : -1),
                   "shifted target/mask mismatch");
    }
    EXPECT(first && last, "sampler excluded a valid endpoint: first=%d last=%d", first, last);
    // Exactly one window, with a nonzero lower bound.
    data.sample_batch(rng, 1, 3, 2, 6, inp, tgt);
    EXPECT(inp[0] == 2 && inp[2] == 4 && tgt[0] == -1 && tgt[2] == 5,
           "single-window range not sampled correctly");
}

inline double toy_training_step(Model<float>& model, AdamW& opt, Rng& rng) {
    constexpr int B = 2, Tn = 8;
    int inp[B * Tn], tgt[B * Tn];
    for (int b = 0; b < B; ++b) {
        const int start = int(rng.below(model.cfg.vocab));
        for (int t = 0; t < Tn; ++t) {
            inp[b * Tn + t] = (start + t) % model.cfg.vocab;
            tgt[b * Tn + t] = (start + t + 1) % model.cfg.vocab;
        }
    }
    model.zero_grad();
    const double loss = model.forward(inp, tgt, B, Tn, true);
    model.backward();
    const double norm = clip_grad_norm(model, 1.0);
    EXPECT(std::isfinite(loss) && std::isfinite(norm), "training produced non-finite loss/norm");
    opt.update(model, 0.01);
    model.update_router_bias(0.001);
    return loss;
}

inline void test_ternary_learning_and_resume() {
    Config cfg = training_test_config(true);
    cfg.n_experts = 4; cfg.top_k = 2; cfg.expert_hidden = 6; cfg.n_shared = 1;
    Model<float> model(cfg);
    model.init(18);
    AdamW opt;
    Rng rng(19);
    const double initial = toy_training_step(model, opt, rng);
    for (int i = 1; i < 30; ++i) toy_training_step(model, opt, rng);
    const double final = toy_training_step(model, opt, rng);
    EXPECT(final < 0.8 * initial, "ternary MoE did not learn: initial %g final %g", initial, final);
    model.quantize_weights();
    const auto quantized_before = model.qparams;
    const auto master_before = model.params;

    // Same model/optimizer settings: save moments, RNG and balancing biases,
    // continue, reload into a fresh model, and take exactly the same step.
    const std::string path = "/tmp/cxxgpt_test_train_" +
                             std::to_string(uint64_t(now_seconds() * 1e9)) + ".ckpt";
    save_checkpoint(path, model, opt, rng);
    const Config loaded_cfg = peek_checkpoint_config(path);
    Model<float> restored(loaded_cfg);
    AdamW restored_opt;
    Rng restored_rng(0);
    load_checkpoint(path, restored, &restored_opt, &restored_rng);
    std::remove(path.c_str());
    EXPECT(restored.params == model.params && restored.router_bias == model.router_bias &&
           restored_opt.m == opt.m && restored_opt.v == opt.v && restored_opt.step == opt.step,
           "checkpoint did not restore training state");
    for (int i = 0; i < 4; ++i) EXPECT(restored_rng.state()[i] == rng.state()[i], "RNG state changed");
    const double la = toy_training_step(model, opt, rng);
    const double lb = toy_training_step(restored, restored_opt, restored_rng);
    EXPECT(la == lb && restored.params == model.params && restored_opt.m == opt.m &&
           restored_opt.v == opt.v && restored.router_bias == model.router_bias,
           "resumed training differs from uninterrupted training");
    model.quantize_weights();
    EXPECT(model.params != master_before && model.qparams != quantized_before,
           "optimizer did not update master weights and their quantized copies");
    std::printf("  ternary MoE learned: %.3f -> %.3f; checkpoint continuation identical\n", initial, final);
}

inline void test_train() {
    test_clipping_reference();
    test_adamw_reference();
    test_sampling_endpoints();
    test_ternary_learning_and_resume();
}
