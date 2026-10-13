// test_gpu.cpp — the CUDA trainer (src/gpu.cu) against the CPU engine.
// Needs an NVIDIA GPU: make gpu-test.
//
// For each architecture, both sides start from the same weights and train
// a few steps on the same batches; the losses, every parameter tensor's
// gradient, the gradient norm and the updated weights must agree. With
// exact fp32 matmuls the only differences are summation order; ternary
// models also see rare int8 rounding flips, so their tolerance is looser.
// Muon runs the same comparison on the dense and MoE models. Some runs clip
// the gradient (to half its norm), as training with --clip does. A last check
// trains on the GPU alone, with TF32, until a toy pattern is learned.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../src/gpu.h"
#include "../src/model.h"
#include "../src/train.h"
#include "test_util.h"

namespace {

// ||a - b|| / ||b|| (0 if both are zero).
double rel_l2(const float* a, const float* b, size_t n) {
    double d = 0, r = 0;
    for (size_t i = 0; i < n; ++i) {
        d += (double(a[i]) - b[i]) * (double(a[i]) - b[i]);
        r += double(b[i]) * b[i];
    }
    return r > 0 ? std::sqrt(d / r) : std::sqrt(d);
}

void random_batch(Rng& rng, int V, int N, std::vector<int>& tok, std::vector<int>& tgt) {
    tok.resize(N);
    tgt.resize(N);
    for (int n = 0; n < N; ++n) {
        tok[n] = int(rng.below(V));
        tgt[n] = rng.below(5) == 0 ? -1 : int(rng.below(V));
    }
}

// clip > 0: clip the gradient to that fraction of its norm, scaling it on the
// GPU the way GpuBackend does.
void compare(const char* name, const Config& cfg, double tol_loss, double tol_grad, double tol_param,
             bool muon = false, double clip = 0) {
    Model<float> cpu(cfg);
    cpu.init(5);
    Rng rng(6);
    for (auto& layer : cpu.router_bias)
        for (auto& b : layer) b = float(0.01 * rng.normal());
    AdamW opt;
    opt.muon = muon;
    GpuTrainer gpu(cfg, cpu.infos, /*tf32=*/false);
    gpu.set_params(cpu.params.data());
    gpu.set_adam_state(0, nullptr, nullptr);
    gpu.set_router_bias(cpu.router_bias);
    gpu.set_muon(muon, opt.muon_momentum, opt.muon_steps);

    const int B = 3, Tn = cfg.seq_len, N = B * Tn;
    std::vector<int> tok, tgt;
    std::vector<float> g(cpu.num_params()), p(cpu.num_params());
    double worst_loss = 0, worst_grad = 0, worst_param = 0;
    std::string worst_tensor;
    for (int step = 0; step < 3; ++step) {
        random_batch(rng, cfg.vocab, N, tok, tgt);
        cpu.zero_grad();
        gpu.zero_grad();
        const double lc = cpu.forward(tok.data(), tgt.data(), B, Tn, true);
        const double lg = gpu.forward(tok.data(), tgt.data(), B, Tn, true);
        worst_loss = std::max(worst_loss, std::abs(lc - lg) / std::abs(lc));
        cpu.backward(1.0);
        gpu.backward(1.0);
        gpu.get_grads(g.data());
        for (const auto& info : cpu.infos) {
            const double d = rel_l2(&g[info.offset], &cpu.grads[info.offset], info.size);
            if (d > worst_grad) worst_grad = d, worst_tensor = info.name;
        }
        const double nc = clip_grad_norm(cpu, 0.0), ng = gpu.grad_norm();
        EXPECT(std::abs(nc - ng) <= tol_grad * nc, "%s: grad norm %g (gpu) vs %g (cpu)", name, ng, nc);
        double scale = 1.0;
        if (clip > 0) {
            clip_grad_norm(cpu, clip * nc);
            scale = double(float(clip * nc / ng));
        }
        opt.update(cpu, 1e-2);
        gpu.adamw_step(1e-2, opt.beta1, opt.beta2, opt.eps, opt.weight_decay, scale);
        gpu.get_params(p.data());
        worst_param = std::max(worst_param, rel_l2(p.data(), cpu.params.data(), p.size()));
    }
    // The optimizer state matches the CPU's, and survives a trip through a
    // second trainer, loaded as GpuBackend does (state first, then Muon).
    std::vector<float> gm(p.size()), gv(p.size()), gm2(p.size()), gv2(p.size());
    long gstep = 0, gstep2 = 0;
    gpu.get_adam_state(&gstep, gm.data(), gv.data());
    const double dm = rel_l2(gm.data(), opt.m.data(), gm.size()), dv = rel_l2(gv.data(), opt.v.data(), gv.size());
    EXPECT(gstep == opt.step && dm <= 3 * tol_grad && dv <= 3 * tol_grad,
           "%s: optimizer state differs from the CPU's (step %ld vs %ld, m by %g, v by %g)", name, gstep, opt.step,
           dm, dv);
    GpuTrainer gpu2(cfg, cpu.infos, /*tf32=*/false);
    gpu2.set_adam_state(gstep, gm.data(), gv.data());
    gpu2.set_muon(muon, opt.muon_momentum, opt.muon_steps);
    gpu2.get_adam_state(&gstep2, gm2.data(), gv2.data());
    EXPECT(gstep2 == gstep && gm2 == gm && gv2 == gv, "%s: optimizer state changed in a round trip", name);

    const auto load = gpu.take_expert_load();
    if (cfg.moe() && tol_grad < 1e-3)
        EXPECT(load == cpu.expert_load, "%s: expert loads differ (routing changed)", name);
    std::printf("  %-14s loss %.1e, grads %.1e (worst %s), params %.1e\n", name, worst_loss, worst_grad,
                worst_tensor.c_str(), worst_param);
    EXPECT(worst_loss <= tol_loss, "%s: loss differs by %g", name, worst_loss);
    EXPECT(worst_grad <= tol_grad, "%s: gradient of %s differs by %g", name, worst_tensor.c_str(), worst_grad);
    EXPECT(worst_param <= tol_param, "%s: weights after the optimizer step differ by %g", name, worst_param);
}

// The GPU alone, with TF32, learns a repeating sequence.
void learns_pattern(bool muon) {
    Config cfg;
    cfg.vocab = 64;
    cfg.seq_len = 32;
    cfg.d_model = 64;
    cfg.n_layers = 2;
    cfg.n_heads = 2;
    cfg.n_experts = 4;
    cfg.top_k = 2;
    cfg.ternary = true;
    Model<float> host(cfg);
    host.init(7);
    GpuTrainer gpu(cfg, host.infos, /*tf32=*/true);
    gpu.set_params(host.params.data());
    gpu.set_muon(muon);
    const int B = 8, N = B * cfg.seq_len;
    std::vector<int> tok(N), tgt(N);
    for (int n = 0; n < N; ++n) {
        tok[n] = (n * 7) % cfg.vocab;
        tgt[n] = ((n + 1) * 7) % cfg.vocab;
    }
    double first = 0, last = 0;
    for (int step = 0; step < 150; ++step) {
        gpu.zero_grad();
        last = gpu.forward(tok.data(), tgt.data(), B, cfg.seq_len, true);
        if (step == 0) first = last;
        gpu.backward(1.0);
        const double norm = gpu.grad_norm();
        gpu.adamw_step(3e-3, 0.9, 0.95, 1e-8, 0.0, norm > 1.0 ? 1.0 / norm : 1.0);
    }
    const char* opt = muon ? "Muon" : "AdamW";
    std::printf("  learns pattern (%s): loss %.3f -> %.3f\n", opt, first, last);
    EXPECT(last < 0.1 * first, "GPU training with %s did not learn the pattern: %.3f -> %.3f", opt, first, last);
}

}  // namespace

int main() {
    ThreadPool tp(4);
    g_pool = &tp;
    std::printf("%s\n", GpuTrainer::device_name().c_str());

    Config dense;
    dense.vocab = 61;
    dense.seq_len = 24;
    dense.d_model = 64;
    dense.n_layers = 2;
    dense.n_heads = 4;
    compare("dense", dense, 1e-5, 1e-4, 1e-4);

    Config tern = dense;
    tern.ternary = true;
    compare("ternary", tern, 1e-3, 3e-2, 1e-2);

    Config moe = dense;
    moe.n_experts = 4;
    moe.top_k = 2;
    moe.n_shared = 1;
    moe.expert_hidden = 32;
    compare("moe+shared", moe, 1e-5, 1e-4, 1e-4);

    Config moe1 = moe;
    moe1.top_k = 1;
    moe1.n_shared = 0;
    compare("moe top-1", moe1, 1e-5, 1e-4, 1e-4, false, 0.5);

    Config ours = moe;
    ours.ternary = true;
    compare("ternary moe", ours, 1e-3, 3e-2, 1e-2);

    // Muon moves the weights about as much as the gradients differ, ten
    // times more than AdamW does, hence the looser weight tolerance.
    compare("dense muon", dense, 1e-5, 1e-4, 1e-3, true);
    compare("moe muon", moe, 1e-5, 1e-4, 1e-3, true, 0.5);

    learns_pattern(false);
    learns_pattern(true);
    if (g_failures) return 1;
    std::printf("all passed\n");
    return 0;
}
