// test_gpu.cpp — the CUDA trainer (src/gpu.cu) against the CPU engine.
// Needs an NVIDIA GPU: make gpu-test.
//
// For each architecture, both sides start from the same weights and train
// a few steps on the same batches; the losses, every parameter tensor's
// gradient, the gradient norm and the updated weights must agree. With
// exact fp32 matmuls the only differences are summation order; ternary
// models also see rare int8 rounding flips, so their tolerance is looser.
// Muon runs the same comparison on the dense and MoE models. Some runs clip
// the gradient (to half its norm), as training with --clip does. Attention
// runs on the fused kernels (attention.cuh), except in one run that checks
// the cuBLAS path (--no-fused-attn). Two runs check --fp-attn and
// --nope-every. One run multiplies in bf16, against looser tolerances, and
// two in TF32, where attention runs on cuBLAS and on the tensor-core fused
// kernels (attention_tc.cuh). Before all that, both sets of fused kernels
// are checked alone against a double-precision reference. A last check
// trains on the GPU alone, with TF32 and with bf16, until a toy pattern is
// learned.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../src/generate.h"
#include "../src/gpu.h"
#include "../src/model.h"
#include "../src/train.h"
#include "test_util.h"
#include "attention_ref.h"
#include "extreme_loss_fixture.h"
#include "test_accumulation.h"

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

struct Run {
    bool muon = false;
    // > 0: clip the gradient to that fraction of its norm, scaling it on
    // the GPU the way GpuBackend does.
    double clip = 0;
    bool fused = true;  // the GPU's fused attention kernels
    Matmul matmul = Matmul::fp32;
};

void compare(const char* name, const Config& cfg, double tol_loss, double tol_grad, double tol_param, Run run = {}) {
    const bool muon = run.muon;
    const double clip = run.clip;
    Model<float> cpu(cfg);
    cpu.init(5);
    Rng rng(6);
    for (auto& layer : cpu.router_bias)
        for (auto& b : layer) b = float(0.01 * rng.normal());
    AdamW opt;
    opt.muon = muon;
    GpuTrainer gpu(cfg, cpu.infos, run.matmul, run.fused);
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
    GpuTrainer gpu2(cfg, cpu.infos, run.matmul, run.fused);
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

// The production accumulation helper must preserve the CPU macro-batch
// objective on CUDA too. Runtime validation requires physical GPU hardware.
void masked_accumulation() {
    const auto group = masked_accumulation_fixture();
  for (bool fused : {false, true}) {
    for (int variant = 0; variant < 8; ++variant) {
        Config cfg;
        cfg.vocab = 13; cfg.seq_len = 8; cfg.d_model = 32;
        cfg.n_heads = 2; cfg.n_layers = 1; cfg.ffn_hidden = 48;
        cfg.ternary = variant % 2 || variant == 4;
        cfg.fp_attention = variant == 4 || variant == 5;
        if (variant >= 4) cfg.n_layers = 2;
        if (variant >= 5) cfg.nope_every = variant == 6 ? 1 : 2;
        if (variant == 2 || variant == 3 || variant == 5 || variant == 7) {
            cfg.n_experts = 4; cfg.top_k = 2; cfg.expert_hidden = 24; cfg.n_shared = 1;
        }
        Model<float> cpu(cfg);
        cpu.init(57);
        GpuTrainer gpu(cfg, cpu.infos, Matmul::fp32, fused);
        gpu.set_params(cpu.params.data());
        cpu.zero_grad();
        const double lc = cpu.forward(group.inputs.data(), group.targets.data(), 4, 8, true);
        cpu.backward();
        const double lg = accumulated_backward(gpu, group);
        std::vector<float> grad(cpu.num_params());
        gpu.get_grads(grad.data());
        for (float g : grad) EXPECT(std::isfinite(g), "non-finite accumulated GPU gradient");
        const double tol = cfg.ternary ? 3e-2 : 1e-4;
        EXPECT(std::abs(lc - lg) < (cfg.ternary ? 1e-3 : 1e-5) * std::abs(lc),
               "variant %d accumulated GPU loss %g vs CPU %g", variant, lg, lc);
        double worst = 0;
        for (const auto& p : cpu.infos) {
            const double d = rel_l2(grad.data() + p.offset, cpu.grads.data() + p.offset, p.size);
            EXPECT(std::isfinite(d) && d < tol, "variant %d accumulated GPU tensor %s error %g",
                   variant, p.name.c_str(), d);
            worst = std::max(worst, d);
        }
        std::printf("  variant %d masked GPU accumulation, fused=%d: loss difference %.1e gradient %.1e\n",
                    variant, int(fused), std::abs(lc - lg), worst);
        auto empty = group;
        std::fill(empty.targets.begin(), empty.targets.end(), -1); empty.refresh_counts();
        EXPECT(accumulated_backward(gpu, empty) == 0, "empty GPU accumulated loss nonzero");
        gpu.get_grads(grad.data());
        for (float g : grad) EXPECT(g == 0, "empty GPU group retained stale gradient");
    }
  }
}

// The GPU alone learns a repeating sequence, with a ternary MoE model
// (full precision with bf16 products, which do not take ternary models).
void learns_pattern(bool muon, Matmul matmul = Matmul::tf32, bool fused = false) {
    Config cfg;
    cfg.vocab = 64;
    cfg.seq_len = 32;
    cfg.d_model = 64;
    cfg.n_layers = 2;
    cfg.n_heads = 2;
    cfg.n_experts = 4;
    cfg.top_k = 2;
    cfg.ternary = matmul != Matmul::bf16;
    Model<float> host(cfg);
    host.init(7);
    GpuTrainer gpu(cfg, host.infos, matmul, fused);
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
    const char* mm = matmul == Matmul::bf16 ? "bf16" : matmul == Matmul::fp32 ? "fp32" : "tf32";
    std::printf("  learns pattern (%s, %s%s): loss %.3f -> %.3f\n", opt, mm, fused ? ", fused attention" : "", first,
                last);
    EXPECT(last < 0.1 * first, "GPU training with %s did not learn the pattern: %.3f -> %.3f", opt, first, last);
}

// The fused attention kernels alone, against the double-precision reference
// (attention_ref.h): attention.cuh's fp32 kernels, and attention_tc.cuh's
// tensor-core ones, whose inputs are rounded to TF32 as cuBLAS rounds them:
// 11 significant bits, about 5e-4 apart, which is the size of their errors
// (2e-4 to 1e-3 on the RTX 4070), hence the bound of 3e-3.
void fused_kernels() {
    struct Shape {
        int B, NH, HS, Tn;
    };
    const Shape shapes[] = {
        {2, 2, 16, 5},    // shorter than a tile
        {2, 3, 16, 33},   // one row past a tile
        {1, 2, 64, 70},   // two blocks of rows, the second mostly padding
        {2, 2, 64, 512},  // the night-1 shape
        {2, 2, 32, 100},
        {1, 2, 48, 130},
        {1, 1, 128, 40},  // the largest head size
    };
    uint64_t seed = 11;
    for (const Shape& s : shapes) {
        const auto k = attention_ref::make_case(s.B, s.NH, s.HS, s.Tn, seed++);
        for (bool tc : {false, true}) {
            std::vector<float> out(k.out.size()), lse(k.lse.size()), dqkv = k.dqkv0;
            gpu_fused_attention(k.qkv.data(), k.dout.data(), out.data(), lse.data(), dqkv.data(), s.B, s.Tn, s.NH,
                                s.HS, tc);
            const auto e = attention_ref::errors(k, out, lse, dqkv);
            std::printf("  fused %s, %d x %d heads of %d, %d tokens: out %.1e, lse %.1e, dq %.1e, dk %.1e, dv %.1e\n",
                        tc ? "tensor-core" : "fp32", s.B, s.NH, s.HS, s.Tn, e.out, e.lse, e.dq, e.dk, e.dv);
            const double tol_out = tc ? 3e-3 : 1e-5, tol_grad = tc ? 3e-3 : 1e-4;
            EXPECT(e.out < tol_out && e.lse < tol_out, "fused %s forward: out %g, lse %g", tc ? "tc" : "fp32", e.out,
                   e.lse);
            EXPECT(e.dq < tol_grad && e.dk < tol_grad && e.dv < tol_grad, "fused %s backward: dq %g, dk %g, dv %g",
                   tc ? "tc" : "fp32", e.dq, e.dk, e.dv);
        }
    }
}

// Zero block projections make the residual stream equal the embedding.
// Constant +/-1000 embedding rows produce logits near +/-32000. The two
// counted losses have a double reference near 64000 and 32000, even though
// both target probabilities underflow. The middle row is ignored.
void extreme_cross_entropy() {
    ExtremeLossFixture fixture;
    auto& cpu = fixture.model;
    const auto& cfg = cpu.cfg;
    const ParamInfo& embedding = *cpu.find("wte");
    GpuTrainer gpu(cfg, cpu.infos, /*tf32=*/false);
    gpu.set_params(cpu.params.data());
    const int* tok = fixture.tokens.data();
    const int* tgt = fixture.targets.data();
    cpu.zero_grad(); gpu.zero_grad();
    const double lc = cpu.forward(tok, tgt, 1, 3), lg = gpu.forward(tok, tgt, 1, 3);
    const double ref = fixture.reference_loss();
    EXPECT(std::isfinite(lc) && std::abs(lc - ref) < 0.05, "extreme CPU loss %g vs %g", lc, ref);
    EXPECT(std::isfinite(lg) && std::abs(lg - ref) < 0.05, "extreme GPU loss %g vs %g", lg, ref);
    cpu.backward(); gpu.backward(1.0);
    std::vector<float> grad(cpu.num_params());
    gpu.get_grads(grad.data());
    for (float g : grad) EXPECT(std::isfinite(g), "non-finite extreme GPU gradient");
    for (int i = 0; i < cfg.d_model; ++i)
        EXPECT(std::abs(grad[embedding.offset + cfg.d_model + i] + 0.5f) < 1e-5f,
               "underflowed target lost its output-weight gradient");
    EXPECT(rel_l2(grad.data(), cpu.grads.data(), grad.size()) < 1e-4,
           "extreme loss gradients disagree between CPU and GPU");
    const int ignored[] = {-1, -1, -1};
    gpu.zero_grad();
    EXPECT(gpu.forward(tok, ignored, 1, 3) == 0, "all-ignored GPU loss nonzero");
    gpu.backward(1.0); gpu.get_grads(grad.data());
    for (float g : grad) EXPECT(g == 0, "all-ignored GPU gradient nonzero");
    std::printf("  extreme cross-entropy: CPU %.1f GPU %.1f reference %.1f\n", lc, lg, ref);
}

// Profiling names the parts of a step, with times from CUDA events.
void profile_regions() {
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
    host.init(8);
    GpuTrainer gpu(cfg, host.infos);
    gpu.set_params(host.params.data());
    gpu.set_profile(true);
    gpu.set_muon(true);
    const int B = 2, N = B * cfg.seq_len;
    std::vector<int> tok(N), tgt(N);
    for (int n = 0; n < N; ++n) tok[n] = tgt[n] = n % cfg.vocab;
    gpu.zero_grad();
    gpu.forward(tok.data(), tgt.data(), B, cfg.seq_len, true);
    gpu.backward(1.0);
    gpu.adamw_step(1e-3, 0.9, 0.95, 1e-8, 0.0, 1.0);
    const auto prof = gpu.take_profile();
    double total = 0;
    std::string names;
    for (const auto& [name, s] : prof) {
        EXPECT(s >= 0 && s < 10, "profile: %s took %g s", name.c_str(), s);
        total += s;
        names += " [" + name + "]";
    }
    for (const char* want : {"matmul fwd", "matmul bwd", "attention fwd", "attention bwd", "moe routing",
                             "output layer and loss", "muon", "adamw", "ternary weights"})
        EXPECT(names.find("[" + std::string(want) + "]") != std::string::npos, "profile has no '%s' part", want);
    EXPECT(gpu.take_profile().empty(), "take_profile did not start over");
    std::printf("  profile: %zu parts, %.2f ms in all\n", prof.size(), total * 1e3);
}

// Incremental decoding on the GPU (decode_begin / decode_step, what training
// takes its reasoning exams with) against the CPU's Model: the same tokens
// into both, prompts of 9, 1 and 5 tokens (left-padded), every step's logits
// compared, to the end of the context; then generate() on both, and a
// training forward() after decoding, which must see the same loss and
// memory as before it (the caches are freed, the training buffers kept).
void decode(const char* name, const Config& cfg, double tol, Matmul matmul = Matmul::fp32) {
    Model<float> cpu(cfg);
    cpu.init(9);
    Rng rng(10);
    for (auto& layer : cpu.router_bias)
        for (auto& b : layer) b = float(0.01 * rng.normal());
    GpuTrainer gpu(cfg, cpu.infos, matmul, true);
    gpu.set_params(cpu.params.data());
    gpu.set_router_bias(cpu.router_bias);
    EXPECT(gpu.decode_capacity(64) >= 3, "%s: decode capacity %d", name, gpu.decode_capacity(64));
    const std::vector<int> lens = {9, 1, 5};
    const int B = int(lens.size()), L = 9, V = cfg.vocab;
    std::vector<int> start(B);
    std::vector<std::vector<int>> prompts(B);
    for (int b = 0; b < B; ++b) {
        start[b] = L - lens[b];
        for (int i = 0; i < lens[b]; ++i) prompts[b].push_back(int(rng.below(V)));
    }
    // Twice: with nothing allocated yet (as eval-reasoning starts), then
    // after a training forward() (as the exams during training run).
    std::vector<int> tok(size_t(B) * cfg.seq_len), tgt(tok.size());
    for (size_t i = 0; i < tok.size(); ++i) tok[i] = tgt[i] = int(rng.below(V));
    double worst = 0, loss_before = 0;
    size_t bytes_before = 0;
    for (int round = 0; round < 2; ++round) {
        if (round == 1) {
            loss_before = gpu.forward(tok.data(), tgt.data(), B, cfg.seq_len, false);
            bytes_before = gpu.device_bytes();
        }
        cpu.decode_begin(B, start);
        gpu.decode_begin(B, start);
        std::vector<int> cur(B);
        for (int pos = 0; pos < cfg.seq_len; ++pos) {
            for (int b = 0; b < B; ++b)
                if (pos < L) cur[b] = pos < start[b] ? prompts[b][0] : prompts[b][pos - start[b]];
            const float* lc = cpu.decode_step(cur.data(), pos);
            const float* lg = gpu.decode_step(cur.data(), pos);
            for (int b = 0; b < B; ++b) {
                const float* rc = lc + size_t(b) * V;
                if (pos >= start[b]) worst = std::max(worst, rel_l2(lg + size_t(b) * V, rc, size_t(V)));
                cur[b] = int(std::max_element(rc, rc + V) - rc);  // the CPU's greedy choice feeds both
            }
        }
    }
    const double loss_after = gpu.forward(tok.data(), tgt.data(), B, cfg.seq_len, false);
    EXPECT(loss_after == loss_before && gpu.device_bytes() == bytes_before,
           "%s: training after decoding saw loss %.9g and %zu bytes, before %.9g and %zu", name, loss_after,
           gpu.device_bytes(), loss_before, bytes_before);
    // The whole generation loop, greedy: the same tokens, with exact products.
    auto never = [](const std::vector<int>&) { return false; };
    GpuDecoder dec{cfg, gpu};
    const auto gc = generate(cpu, prompts, cfg.seq_len, never);
    const auto gg = generate(dec, prompts, cfg.seq_len, never);
    if (matmul == Matmul::fp32) EXPECT(gc == gg, "%s: generate() picked different tokens", name);
    // The caches lived in the gradient buffer; zero_grad() gives it back.
    gpu.zero_grad();
    std::vector<float> g(cpu.num_params(), 1.0f);
    gpu.get_grads(g.data());
    EXPECT(std::all_of(g.begin(), g.end(), [](float x) { return x == 0.0f; }),
           "%s: zero_grad() after decoding left gradients", name);
    std::printf("  decode %-17s logits %.1e\n", name, worst);
    EXPECT(worst <= tol, "%s: decode logits differ by %g", name, worst);
}

}  // namespace

int main() {
    ThreadPool tp(4);
    g_pool = &tp;
    std::printf("%s\n", GpuTrainer::device_name().c_str());
    extreme_cross_entropy();
    masked_accumulation();
    fused_kernels();

    Config dense;
    dense.vocab = 61;
    dense.seq_len = 24;
    dense.d_model = 64;
    dense.n_layers = 2;
    dense.n_heads = 4;
    compare("dense", dense, 1e-5, 1e-4, 1e-4);
    compare("dense, cuBLAS attention", dense, 1e-5, 1e-4, 1e-4, {.fused = false});
    // bf16 products keep 8 significant bits of each input, about 4e-3 apart,
    // so a looser check that nothing else differs. The bounds here and for
    // the MoE runs below are twice what the RTX 4070 measured; the worst
    // gradients are a norm weight's and the router's, which sum many
    // nearly cancelling terms (TF32, 3 bits finer, is about 4x closer).
    compare("dense bf16", dense, 3e-4, 3e-1, 5e-2, {.matmul = Matmul::bf16});

    // Fused attention over several tiles of rows (70 tokens), and at its
    // largest head size, which needs more than 48 KB of shared memory.
    Config wide = dense;
    wide.seq_len = 70;
    wide.d_model = 128;
    wide.n_heads = 2;
    compare("heads of 64", wide, 1e-5, 1e-4, 1e-4);
    wide.seq_len = 40;
    wide.n_heads = 1;
    compare("head of 128", wide, 1e-5, 1e-4, 1e-4);

    // TF32, as training runs: attention through cuBLAS, and fused on the
    // tensor cores. Both round the same inputs the same way, so the fused
    // kernels must stay as close to the CPU as cuBLAS does. The bounds are
    // twice what cuBLAS measured on the RTX 4070 (loss 1.0e-4, gradients
    // 6.1e-2 at worst, on a norm weight, weights 7.7e-3).
    wide.seq_len = 70;
    wide.n_heads = 2;
    compare("tf32, cuBLAS attention", wide, 2e-4, 1.5e-1, 2e-2, {.fused = false, .matmul = Matmul::tf32});
    compare("tf32, fused attention", wide, 2e-4, 1.5e-1, 2e-2, {.matmul = Matmul::tf32});

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
    compare("moe top-1", moe1, 1e-5, 1e-4, 1e-4, {.clip = 0.5});
    // The experts' products in bf16 (each a slice of the rows), against the
    // same model in TF32. Measured: TF32 loss 1.2e-4, gradients 1.0e-1,
    // weights 7.9e-3; bf16 5.9e-4, 4.4e-1 (the router), 3.7e-2; with Muon
    // 2.3e-5, 2.3e-1, 4.1e-3.
    compare("moe tf32", moe, 3e-4, 2e-1, 2e-2, {.matmul = Matmul::tf32});
    compare("moe bf16", moe, 1.2e-3, 9e-1, 8e-2, {.matmul = Matmul::bf16});
    compare("moe muon bf16", moe, 5e-5, 5e-1, 1e-2, {.muon = true, .matmul = Matmul::bf16});

    Config ours = moe;
    ours.ternary = true;
    compare("ternary moe", ours, 1e-3, 3e-2, 1e-2);

    // Ternary FFN with full-precision attention (--fp-attn), and layers
    // without RoPE (--nope-every 2: the second of every two).
    Config fpa = ours;
    fpa.fp_attention = true;
    compare("ternary moe, fp attention", fpa, 1e-3, 3e-2, 1e-2);
    Config nope = dense;
    nope.nope_every = 2;
    compare("dense, half NoPE", nope, 1e-5, 1e-4, 1e-4);

    // Muon moves the weights about as much as the gradients differ, ten
    // times more than AdamW does, hence the looser weight tolerance.
    compare("dense muon", dense, 1e-5, 1e-4, 1e-3, {.muon = true});
    compare("moe muon", moe, 1e-5, 1e-4, 1e-3, {.muon = true, .clip = 0.5});

    // Decoding: exact products (summation order and expf against the CPU's
    // fast_exp are the only differences), MoE routing, layers without RoPE,
    // ternary weights, and bf16 products as the nights run. The RTX 4070
    // measured 1.6e-7 to 2.2e-7 with exact products and 1.4e-2 with bf16;
    // the bounds are 5x and 2x that (ternary 1e-3: an int8 rounding flip
    // is allowed, as in compare()).
    decode("dense", dense, 1e-6);
    decode("moe+shared", moe, 1e-6);
    decode("half NoPE", nope, 1e-6);
    decode("ternary moe", ours, 1e-3);
    decode("moe bf16", moe, 3e-2, Matmul::bf16);

    learns_pattern(false);
    learns_pattern(true);
    learns_pattern(false, Matmul::bf16);
    learns_pattern(true, Matmul::bf16);
    learns_pattern(false, Matmul::tf32, true);
    profile_regions();
    if (g_failures) return 1;
    std::printf("all passed\n");
    return 0;
}
