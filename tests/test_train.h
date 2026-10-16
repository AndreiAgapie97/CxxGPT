// Optimizer and data-boundary tests, independent of model gradient formulas.
#pragma once

#include "../src/train.h"
#include "../src/train_control.h"
#include "test_util.h"
#include "test_accumulation.h"

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
    const std::string path = test_path("train-resume.ckpt");
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

// Warmup-stable-decay: before the cooldown the rate does not depend on the
// run's length (what lets a run go on, or a cooldown branch off it), and the
// cooldown runs from max_lr down to min_frac * max_lr without going back up.
inline void test_wsd_schedule() {
    const double lr = 1e-3, min_frac = 0.1;
    EXPECT(lr_wsd(0, 10, 1000, 200, lr, min_frac) == lr_at(0, 10, 1000, lr, min_frac) &&
               lr_wsd(9, 10, 1000, 200, lr, min_frac) == lr,
           "WSD warmup differs from the cosine schedule's");
    for (int64_t step : {10, 400, 799})
        for (int64_t total : {1000, 5000, 1000000})
            EXPECT(lr_wsd(step, 10, total, 200, lr, min_frac) == lr, "step %ld of %ld: rate %g before the cooldown",
                   long(step), long(total), lr_wsd(step, 10, total, 200, lr, min_frac));
    EXPECT(lr_wsd(800, 10, 1000, 200, lr, min_frac) == lr, "the cooldown does not start at max_lr");
    double prev = lr;
    for (int64_t step = 801; step < 1000; ++step) {
        const double x = lr_wsd(step, 10, 1000, 200, lr, min_frac);
        EXPECT(x < prev && x > lr * min_frac, "cooldown step %ld: %g after %g", long(step), x, prev);
        prev = x;
    }
    // Half the cooldown done: 1 - sqrt(1/2) of the way down.
    const double half = lr * (min_frac + (1 - min_frac) * (1 - std::sqrt(0.5)));
    EXPECT(std::abs(lr_wsd(900, 10, 1000, 200, lr, min_frac) / half - 1) < 1e-15, "cooldown midpoint");
    EXPECT(lr_wsd(1000, 10, 1000, 200, lr, min_frac) == lr * min_frac &&
               lr_wsd(5000, 10, 1000, 200, lr, min_frac) == lr * min_frac,
           "after the last step the rate is not min_frac * max_lr");
    // A branched cooldown: the checkpoint at step 3000, resumed with --steps
    // 3500 --decay-steps 500, starts at max_lr and decays from its first step.
    EXPECT(lr_wsd(3000, 10, 3500, 500, lr, min_frac) == lr && lr_wsd(3001, 10, 3500, 500, lr, min_frac) < lr,
           "a branched cooldown does not start at the checkpoint's step");
    // No room after the warmup: the cooldown starts right after it.
    EXPECT(lr_wsd(10, 10, 20, 50, lr, min_frac) == lr && lr_wsd(11, 10, 20, 50, lr, min_frac) < lr,
           "a cooldown longer than the run");
    // Without a cooldown, flat until the last step.
    EXPECT(lr_wsd(19, 10, 20, 0, lr, min_frac) == lr, "zero decay steps");
}

inline void test_control_file() {
    ControlCommands c = parse_control("# a comment\n\n  lr-scale 0.25 \r\nsave\neval\nreason\nstop\n");
    EXPECT(c.errors.empty() && c.lr_scale == 0.25 && c.save && c.eval && c.reason && c.stop,
           "control commands not read (%zu errors)", c.errors.size());
    c = parse_control("lr-scale\nlr-scale 0\nlr-scale -1\nlr-scale 1e999\nlr-scale 0.5x\nsave now\nsavee\nstop 1 2\n");
    EXPECT(c.errors.size() == 8 && c.lr_scale == 1 && !c.save && !c.stop, "bad control lines accepted (%zu errors)",
           c.errors.size());
    // Windows PowerShell 5's `"lr-scale 2" > FILE`: UTF-16 with a byte-order mark.
    std::string utf16 = "\xFF\xFE";
    for (char ch : std::string("lr-scale 2\r\nsave\r\n")) utf16 += std::string{ch, '\0'};
    c = parse_control(utf16);
    EXPECT(c.errors.empty() && c.lr_scale == 2 && c.save, "UTF-16 control file not read");
    c = parse_control("\xEF\xBB\xBF" "stop");  // UTF-8 byte-order mark, no final newline
    EXPECT(c.errors.empty() && c.stop, "UTF-8 control file with a byte-order mark not read");

    // The file, with the clock passed in: a change counts once it has stayed
    // the same for ControlFile::kSettleSeconds (0.5 s).
    const std::string path = test_path("train.control");
    std::filesystem::remove(path);
    ControlFile file(path);
    EXPECT(!file.read_now(), "a missing control file has commands");
    EXPECT(!file.poll(0) && !file.poll(1), "a missing control file reads as a change");
    auto write = [&](const char* text) { std::ofstream(path, std::ios::binary) << text; };
    write("lr-scale 0.5\nsave\n");
    EXPECT(!file.poll(2) && !file.poll(2.4), "a control file acted on before it settled");
    auto got = file.poll(2.6);
    EXPECT(got && got->lr_scale == 0.5 && got->save, "a new control file was not read");
    EXPECT(!file.poll(2.7) && !file.poll(9), "an unchanged control file reads again");
    // A shell's `>`: the file is empty for a moment, then written. The empty
    // moment never counts, so lr-scale does not go back to 1 in between.
    write("");
    EXPECT(!file.poll(10), "an emptied control file counted at once");
    write("lr-scale 0.5\nstop\n");
    EXPECT(!file.poll(10.1) && !file.poll(10.5), "a rewritten control file counted before it settled");
    got = file.poll(10.7);
    EXPECT(got && got->lr_scale == 0.5 && got->stop && !got->save, "a changed control file was not read");
    std::filesystem::remove(path);
    EXPECT(!file.poll(11), "a deleted control file counted at once");
    got = file.poll(11.6);
    EXPECT(got && got->lr_scale == 1 && !got->stop && got->errors.empty(), "a deleted control file is not empty");
    EXPECT(!file.poll(12), "a deleted control file reads twice");
    // Already there at the start: read at once, and not again until it changes.
    write("lr-scale 2\nstop\n");
    ControlFile at_start(path);
    got = at_start.read_now();
    EXPECT(got && got->lr_scale == 2 && got->stop, "a control file there at the start was not read");
    EXPECT(!at_start.poll(0) && !at_start.poll(5), "a control file there at the start counts again");
    std::filesystem::remove(path);
}

inline void test_train() {
    test_wsd_schedule();
    test_control_file();
    test_masked_accumulation();
    test_clipping_reference();
    test_adamw_reference();
    test_sampling_endpoints();
    test_ternary_learning_and_resume();
}
