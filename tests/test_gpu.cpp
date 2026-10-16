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
// are checked alone against a double-precision reference. Then decoding:
// the GPU's KV-cache decoder against the CPU's, and eval_reasoning() on it
// (eval-reasoning --gpu) against the CPU, greedy, with a vote and with the
// calculator. Unlikelihood training (#34) runs the same comparison with
// the term on, plus a saturated row and a term that never fires. The GPU's
// ternary codes must be the CPU's, code for code (#40). A last
// check trains on the GPU alone, with TF32 and with bf16, until a toy
// pattern is learned.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>
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
#include "test_ternary_codes.h"

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
    bool grouped = false;  // the experts' grouped products (set_grouped_experts, #49)
    // > 0: unlikelihood training (#34) on both sides, on batches whose
    // tokens come from 4 values, so runs repeat and the term fires.
    double ul_alpha = 0;
    int ul_ngram = 2;
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
    gpu.set_grouped_experts(run.grouped);
    gpu.set_params(cpu.params.data());
    gpu.set_adam_state(0, nullptr, nullptr);
    gpu.set_router_bias(cpu.router_bias);
    gpu.set_muon(muon, opt.muon_momentum, opt.muon_steps);
    cpu.unlikelihood = {run.ul_alpha, run.ul_ngram};
    gpu.set_unlikelihood(run.ul_alpha, run.ul_ngram);

    const int B = 3, Tn = cfg.seq_len, N = B * Tn;
    std::vector<int> tok, tgt;
    std::vector<float> g(cpu.num_params()), p(cpu.num_params());
    double worst_loss = 0, worst_grad = 0, worst_param = 0, worst_ul = 0;
    size_t fired = 0;
    std::string worst_tensor;
    for (int step = 0; step < 3; ++step) {
        random_batch(rng, cfg.vocab, N, tok, tgt);
        if (run.ul_alpha > 0)
            for (int& t : tok) t %= 4;
        cpu.zero_grad();
        gpu.zero_grad();
        const double lc = cpu.forward(tok.data(), tgt.data(), B, Tn, true);
        const double lg = gpu.forward(tok.data(), tgt.data(), B, Tn, true);
        worst_loss = std::max(worst_loss, std::abs(lc - lg) / std::abs(lc));
        if (run.ul_alpha > 0) {
            const GpuTrainer::ForwardParts f = gpu.last_parts();
            EXPECT(f.fired == cpu.last_fired && f.negatives == cpu.last_negatives,
                   "%s: the GPU found %zu negatives at %zu positions, the CPU %zu at %zu", name, f.negatives,
                   f.fired, cpu.last_negatives, cpu.last_fired);
            worst_ul = std::max(worst_ul, std::abs(f.ul - cpu.last_ul) / std::max(cpu.last_ul, 1e-30));
            fired += cpu.last_fired;
        }
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
    if (run.ul_alpha > 0) {
        std::printf("  %-14s unlikelihood term %.1e apart, fired at %zu positions\n", "", worst_ul, fired);
        EXPECT(fired > 0, "%s: the unlikelihood term never fired", name);
        EXPECT(worst_ul <= tol_grad, "%s: the unlikelihood term differs by %g", name, worst_ul);
    }
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

// Ternary codes (#40): the GPU quantizes with its own kernels (absolute
// sums in double, one scale per tensor or expert, rintf), which must give
// every weight the CPU's code, at the thresholds too, and the CPU's scale
// to within an ulp; then again on weights the GPU itself trained.
void ternary_codes_on_gpu(const Config& cfg) {
    Model<float> cpu(cfg);
    cpu.init(3);
    Rng rng(4);
    for (auto& w : cpu.params) w = float(rng.normal() * 0.3);
    std::vector<std::pair<size_t, std::vector<size_t>>> planted;
    for (const ParamInfo& p : cpu.infos)
        if (p.ternary) {
            const size_t offset = p.offset + (p.count - 1) * (p.size / p.count);  // the last copy
            planted.push_back({offset, plant_threshold_cases(&cpu.params[offset], p.size / p.count)});
            break;
        }
    CHECK(!planted.empty(), "no ternary tensor");
    GpuTrainer gpu(cfg, cpu.infos, Matmul::fp32, true);
    std::vector<float> q(cpu.num_params());
    // Codes that differ, and values more than an ulp apart.
    auto mismatches = [&](size_t* values) {
        size_t codes = 0;
        *values = 0;
        for (const ParamInfo& p : cpu.infos)
            if (p.ternary)
                for (size_t i = p.offset; i < p.offset + p.size; ++i) {
                    codes += ternary_sign(q[i]) != ternary_sign(cpu.qparams[i]);
                    *values += std::abs(q[i] - cpu.qparams[i]) > 1.2e-7f * std::abs(cpu.qparams[i]);
                }
        return codes;
    };
    gpu.set_params(cpu.params.data());
    gpu.get_ternary_weights(q.data());
    cpu.weights_dirty = true;
    cpu.quantize_weights();
    size_t values = 0;
    const size_t codes = mismatches(&values);
    EXPECT(codes == 0 && values == 0, "ternary codes: %zu codes and %zu values differ from the CPU's", codes,
           values);
    const auto cases = threshold_cases();
    for (size_t c = 0; c < cases.size(); ++c) {
        const float wq = q[planted[0].first + planted[0].second[c]];
        EXPECT(ternary_sign(wq) == cases[c].second, "GPU: w = %.9g got code %d, expected %d", cases[c].first,
               ternary_sign(wq), cases[c].second);
    }

    // A few AdamW steps on the GPU, then its own weights quantized on the CPU.
    std::vector<int> tok, tgt;
    for (int step = 0; step < 3; ++step) {
        random_batch(rng, cfg.vocab, 2 * cfg.seq_len, tok, tgt);
        gpu.zero_grad();
        gpu.forward(tok.data(), tgt.data(), 2, cfg.seq_len, true);
        gpu.backward(1.0);
        gpu.adamw_step(1e-3, 0.9, 0.95, 1e-8, 0.1, 1.0);
    }
    gpu.get_params(cpu.params.data());
    // NaN weights would give code 0 on both sides and pass the comparison.
    const size_t non_finite =
        size_t(std::count_if(cpu.params.begin(), cpu.params.end(), [](float w) { return !std::isfinite(w); }));
    EXPECT(non_finite == 0, "ternary codes: %zu weights are not finite after 3 GPU steps", non_finite);
    gpu.get_ternary_weights(q.data());
    cpu.weights_dirty = true;
    cpu.quantize_weights();
    const size_t trained = mismatches(&values);
    EXPECT(trained == 0 && values == 0, "ternary codes after training: %zu codes and %zu values differ",
           trained, values);
    std::printf("  ternary codes: GPU = CPU for every weight, before and after 3 GPU steps\n");
}

// Unlikelihood with a saturated softmax: the fixture's logits are about
// +-3e7, so the negative (token 0, repeating "0 0") has p = 1 and every
// other probability underflows. Both sides give the floor's loss; the CPU's
// exponential stops short of 0 where the GPU's reaches it, so only the
// GPU's gradient being finite is checked. Then a term that never fires
// (runs longer than the batch) must leave the GPU's gradient as it was, up
// to rounding: the token embedding's gradient is summed with atomic adds,
// whose order (and so its last bits) changes from run to run.
void unlikelihood_edges() {
    ExtremeLossFixture fixture;
    auto& cpu = fixture.model;
    const auto& cfg = cpu.cfg;
    GpuTrainer gpu(cfg, cpu.infos, /*tf32=*/false);
    gpu.set_params(cpu.params.data());
    const int tgt[] = {1, 1, 2};
    cpu.unlikelihood = {1.0, 2};
    gpu.set_unlikelihood(1.0, 2);
    cpu.zero_grad();
    gpu.zero_grad();
    const double lc = cpu.forward(fixture.tokens.data(), tgt, 1, 3, true);
    const double lg = gpu.forward(fixture.tokens.data(), tgt, 1, 3, true);
    const GpuTrainer::ForwardParts f = gpu.last_parts();
    const double floor_term = -std::log(kUnlikelihoodFloor) / 3;  // one negative, 3 targets
    EXPECT(f.negatives == 1 && cpu.last_negatives == 1, "saturated: %zu GPU, %zu CPU negatives", f.negatives,
           cpu.last_negatives);
    EXPECT(std::abs(f.ul - floor_term) < 1e-9 && std::abs(cpu.last_ul - floor_term) < 1e-9,
           "saturated: term %g (GPU), %g (CPU), floor %g", f.ul, cpu.last_ul, floor_term);
    EXPECT(std::abs(lc - lg) <= 1e-5 * std::abs(lc), "saturated: loss %g (GPU) vs %g (CPU)", lg, lc);
    gpu.backward(1.0);
    std::vector<float> grad(cpu.num_params());
    gpu.get_grads(grad.data());
    bool finite = true;
    for (float g : grad) finite = finite && std::isfinite(g);
    EXPECT(finite, "saturated: non-finite GPU gradient");

    Config dense;
    dense.vocab = 61;
    dense.seq_len = 24;
    dense.d_model = 64;
    dense.n_layers = 2;
    dense.n_heads = 4;
    Model<float> m(dense);
    m.init(5);
    Rng rng(9);
    std::vector<int> tok, t;
    random_batch(rng, dense.vocab, 3 * dense.seq_len, tok, t);
    std::vector<float> plain(m.num_params()), quiet(m.num_params());
    for (const bool on : {false, true}) {
        GpuTrainer g(dense, m.infos, Matmul::fp32, true);
        g.set_params(m.params.data());
        if (on) g.set_unlikelihood(1.0, 50);
        g.zero_grad();
        g.forward(tok.data(), t.data(), 3, dense.seq_len, true);
        EXPECT(!on || g.last_parts().negatives == 0, "runs of 50 tokens repeated in a batch of 24");
        g.backward(1.0);
        g.get_grads(on ? quiet.data() : plain.data());
    }
    double diff = 0.0, scale = 0.0;
    for (size_t i = 0; i < plain.size(); ++i) {
        diff = std::max(diff, double(std::abs(quiet[i] - plain[i])));
        scale = std::max(scale, double(std::abs(plain[i])));
    }
    EXPECT(diff <= 1e-6 * scale, "a term that never fires changed the GPU's gradient by %g (largest %g)", diff,
           scale);
    std::printf("  unlikelihood: saturated term %.1f on both, quiet term changes the gradient by %.1e (of %.1e)\n",
                f.ul, diff, scale);
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
    // With the no-repeat block (eval-reasoning --no-repeat), too.
    const auto bc = generate(cpu, prompts, cfg.seq_len, never, 0.0, nullptr, nullptr, calc::Mode::none, nullptr,
                             nullptr, 3);
    const auto bg = generate(dec, prompts, cfg.seq_len, never, 0.0, nullptr, nullptr, calc::Mode::none, nullptr,
                             nullptr, 3);
    if (matmul == Matmul::fp32) EXPECT(bc == bg, "%s: generate() with --no-repeat 3 picked different tokens", name);
    // The caches lived in the gradient buffer; zero_grad() gives it back.
    gpu.zero_grad();
    std::vector<float> g(cpu.num_params(), 1.0f);
    gpu.get_grads(g.data());
    EXPECT(std::all_of(g.begin(), g.end(), [](float x) { return x == 0.0f; }),
           "%s: zero_grad() after decoding left gradients", name);
    std::printf("  decode %-17s logits %.1e\n", name, worst);
    EXPECT(worst <= tol, "%s: decode logits differ by %g", name, worst);
}

// eval-reasoning --gpu: eval_reasoning() on GpuDecoder (exact fp32 products)
// against the CPU's Model, greedy, with 4 sampled tries and a vote, and
// with the calculator. A freshly initialized model would never finish an
// answer or call the calculator, so a tiny byte-level MoE model first
// learns, on the GPU, to answer "Q: 12*34" with "<<12*34=408>>408\n#### 408":
// a calculator note (calc-notes mode finishes it) and an answer line. Then
// both sides must give the same EvalResult: every level's scores, the first
// tries' texts, the shown failures and the calculator calls, with different
// batches (12 answers at a time on the CPU, 8 on the GPU), as the CLI runs.
void eval_on_gpu() {
    Config cfg;
    cfg.vocab = Tokenizer{}.vocab_size();
    cfg.seq_len = 64;
    cfg.d_model = 64;
    cfg.n_layers = 2;
    cfg.n_heads = 4;
    cfg.n_experts = 4;
    cfg.top_k = 2;
    cfg.expert_hidden = 64;
    Model<float> cpu(cfg);
    cpu.init(11);
    GpuTrainer gpu(cfg, cpu.infos, Matmul::fp32, false);
    gpu.set_params(cpu.params.data());
    Rng rng(12);
    auto problem = [&](int digits) {
        const int lo = digits == 1 ? 2 : 10, hi = digits == 1 ? 9 : 49;
        const int x = lo + int(rng.below(uint64_t(hi - lo + 1))), y = lo + int(rng.below(uint64_t(hi - lo + 1)));
        const std::string q = std::to_string(x) + "*" + std::to_string(y), c = std::to_string(x * y);
        return Problem{"mul", digits, q, "<<" + q + "=" + c + ">>" + c, c};
    };
    // Training: sequences of [BOS] "Q: ...\nA: ...\n#### c\n", padded.
    const Tokenizer tok;
    const int B = 32, Tn = cfg.seq_len;
    std::vector<int> inp(size_t(B) * Tn), tgt(inp.size());
    double first = 0, last = 0;
    for (int step = 0; step < 400; ++step) {
        std::fill(inp.begin(), inp.end(), 0);
        std::fill(tgt.begin(), tgt.end(), -1);
        for (int b = 0; b < B; ++b) {
            const Problem p = problem(1 + b % 2);
            std::vector<int> t = tok.encode(p.prompt() + p.completion());
            t.insert(t.begin(), kBos);
            for (size_t i = 0; i < t.size() && int(i) < Tn; ++i) {
                inp[size_t(b) * Tn + i] = t[i];
                if (i + 1 < t.size()) tgt[size_t(b) * Tn + i] = t[i + 1];
            }
        }
        gpu.zero_grad();
        last = gpu.forward(inp.data(), tgt.data(), B, Tn, true);
        if (step == 0) first = last;
        gpu.backward(1.0);
        const double norm = gpu.grad_norm();
        gpu.adamw_step(3e-3, 0.9, 0.95, 1e-8, 0.0, norm > 1.0 ? 1.0 / norm : 1.0);
    }
    gpu.get_params(cpu.params.data());
    cpu.weights_dirty = true;
    for (auto& layer : cpu.router_bias)
        for (auto& b : layer) b = float(0.01 * rng.normal());
    gpu.set_router_bias(cpu.router_bias);
    std::vector<Problem> test;
    for (int i = 0; i < 24; ++i) test.push_back(problem(1 + i % 2));

    GpuDecoder dec{cfg, gpu};
    std::vector<std::string> plain;  // the greedy completions without the block
    auto check = [&](const char* name, const VoteOpts& vote, calc::Mode tool) {
        const EvalResult c = eval_reasoning(cpu, test, tok, 12, 40, 8, true, vote, tool);
        const EvalResult g = eval_reasoning(dec, test, tok, 8, 40, 8, true, vote, tool);
        bool same = c.correct == g.correct && c.total == g.total && c.completions == g.completions &&
                    c.tool_calls.fired == g.tool_calls.fired && c.tool_calls.malformed == g.tool_calls.malformed &&
                    c.by_task.size() == g.by_task.size() && c.failures.size() == g.failures.size();
        for (auto ic = c.by_task.begin(), ig = g.by_task.begin(); same && ic != c.by_task.end(); ++ic, ++ig) {
            const LevelScore &x = ic->second, &y = ig->second;
            same = ic->first == ig->first && x.total == y.total && x.first_correct == y.first_correct &&
                   x.vote_correct == y.vote_correct && x.any_correct == y.any_correct &&
                   x.distinct_answers == y.distinct_answers;
        }
        for (size_t i = 0; same && i < c.failures.size(); ++i) same = c.failures[i].second == g.failures[i].second;
        const size_t finished = size_t(std::count_if(c.completions.begin(), c.completions.end(), completion_done));
        std::printf("  eval on the GPU, %-15s %2d / %d right, %zu / %zu finished, %zu calculator calls\n", name,
                    c.correct, c.total, finished, c.completions.size(), c.tool_calls.fired);
        EXPECT(same, "eval on the GPU, %s: the EvalResult differs from the CPU's (%d / %d right against %d)", name,
               g.correct, g.total, c.correct);
        if (vote.tries == 1 && vote.no_repeat == 0 && tool == calc::Mode::none) plain = c.completions;
        // The comparison means something only if answers were finished, or
        // (no-repeat) if the block changed them.
        if (vote.no_repeat == 0)
            EXPECT(finished * 2 > c.completions.size(), "eval on the GPU, %s: only %zu of %zu answers finished", name,
                   finished, c.completions.size());
        else
            EXPECT(c.completions != plain, "eval on the GPU, %s: the block never fired", name);
        if (tool != calc::Mode::none) EXPECT(c.tool_calls.fired > 0, "eval on the GPU, %s: no calculator calls", name);
    };
    std::printf("  eval on the GPU: trained loss %.3f -> %.3f\n", first, last);
    VoteOpts votes;
    votes.tries = 4;
    votes.temperature = 0.7;
    votes.seed = 3;
    check("greedy", {}, calc::Mode::none);
    check("maj@4, temp 0.7", votes, calc::Mode::none);
    check("greedy, calc", {}, calc::Mode::notes);
    check("maj@4, calc", votes, calc::Mode::notes);
    // The no-repeat block, chosen on the GPU (no_repeat_mark), at a run
    // length that fires on these answers.
    VoteOpts block;
    block.no_repeat = 3;
    check("greedy, no-rep 3", block, calc::Mode::none);
    votes.no_repeat = 3;
    check("maj@4, no-rep 3", votes, calc::Mode::notes);

    // One generator shared by all sequences in slot order (sample's), the
    // same tokens and the same generator state afterwards.
    std::vector<std::vector<int>> prompts;
    for (int i = 0; i < 5; ++i) prompts.push_back(tok.encode(test[size_t(i)].prompt()));
    Rng rc(21), rg(21);
    const auto never = [](const std::vector<int>&) { return false; };
    const auto oc = generate(cpu, prompts, 30, never, 1.0, &rc);
    const auto og = generate(dec, prompts, 30, never, 1.0, &rg);
    EXPECT(oc == og && rc.next() == rg.next(), "shared-generator sampling differs between the CPU and the GPU");
}

}  // namespace

// x rounded to bf16 (to nearest, ties to even), as __float2bfloat16 does.
double to_bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    u &= 0xffff0000u;
    float y;
    std::memcpy(&y, &u, 4);
    return y;
}

// The MoE experts' grouped products (grouped_gemm.cuh, #49) against a
// double-precision product of the inputs as the kernels read them (rounded
// to bf16 for bf16 products; TF32 rounding is left in the tolerance). All
// three shapes the layers use, groups that are empty, tiny, uneven and
// larger than a tile, sizes off the tiles, and unaligned rows (offset 1, or
// a K not a multiple of 8), which the plain fp32 kernel takes. With beta 0
// the old C is not read: it starts as NaN there.
void grouped_products() {
    struct Case {
        const char* name;
        std::vector<int> rows;  // per group
        int M, N, K;            // M: weight_grad's rows of C_g
    };
    const Case cases[] = {
        {"uneven", {37, 0, 130, 5, 1, 64}, 24, 48, 40},
        {"tiles", {0, 300, 0, 129}, 136, 136, 72},
        {"odd K", {17, 3, 0}, 9, 20, 37},
        {"wide", {200, 0, 70}, 136, 300, 24},  // more than one 128 x 256 tile across
        {"all empty", {0, 0, 0}, 16, 32, 16},
    };
    Rng rng(9);
    double worst[3] = {0, 0, 0};
    const Matmul kinds[3] = {Matmul::fp32, Matmul::tf32, Matmul::bf16};
    const double tol[3] = {1e-6, 3e-3, 1e-5};
    for (int k = 0; k < 3; ++k)
        for (const Case& c : cases)
            for (int op = 0; op < 3; ++op)
                for (int offset : {0, 1}) {
                    std::vector<int> starts{0};
                    for (int r : c.rows) starts.push_back(starts.back() + r);
                    const int R = starts.back(), G = int(c.rows.size());
                    const GroupedOp gop = GroupedOp(op);
                    const bool wg = gop == GroupedOp::weight_grad;
                    // Shapes: A [R][K] (weight_grad: [R][M]); B per group [N][K]
                    // (forward), [K][N] (input_grad), or [R][N] (weight_grad).
                    const int lda = wg ? c.M : c.K, ldb = gop == GroupedOp::forward ? c.K : c.N, ldc = c.N;
                    const long long stride = wg ? (long long)c.M * c.N : (long long)c.N * c.K;
                    std::vector<float> A(size_t(R) * lda), B(wg ? size_t(R) * c.N : size_t(G) * stride);
                    for (float& x : A) x = float(rng.normal());
                    for (float& x : B) x = float(rng.normal());
                    const float beta = wg ? 1.0f : 0.0f;
                    std::vector<float> C(wg ? size_t(G) * stride : size_t(R) * c.N);
                    for (float& x : C) x = wg ? float(rng.normal()) : std::numeric_limits<float>::quiet_NaN();
                    const std::vector<float> C0 = C;
                    gpu_grouped_product(kinds[k], gop, A, B, C, starts, c.M, c.N, c.K, lda, ldb, ldc, stride, beta,
                                        offset);
                    auto in = [&](float x) { return kinds[k] == Matmul::bf16 ? to_bf16(x) : double(x); };
                    std::vector<float> want(C.size());
                    for (int g = 0; g < G; ++g) {
                        if (wg) {
                            for (int m = 0; m < c.M; ++m)
                                for (int n = 0; n < c.N; ++n) {
                                    double s = beta * double(C0[size_t(g) * stride + size_t(m) * ldc + n]);
                                    for (int r = starts[g]; r < starts[g + 1]; ++r)
                                        s += in(A[size_t(r) * lda + m]) * in(B[size_t(r) * ldb + n]);
                                    want[size_t(g) * stride + size_t(m) * ldc + n] = float(s);
                                }
                            continue;
                        }
                        for (int r = starts[g]; r < starts[g + 1]; ++r)
                            for (int n = 0; n < c.N; ++n) {
                                double s = 0;
                                for (int kk = 0; kk < c.K; ++kk) {
                                    const float b = gop == GroupedOp::forward ? B[size_t(g) * stride + size_t(n) * ldb + kk]
                                                                              : B[size_t(g) * stride + size_t(kk) * ldb + n];
                                    s += in(A[size_t(r) * lda + kk]) * in(b);
                                }
                                want[size_t(r) * ldc + n] = float(s);
                            }
                    }
                    const double d = C.empty() ? 0.0 : rel_l2(C.data(), want.data(), C.size());
                    bool finite = true;
                    for (float x : C) finite &= std::isfinite(x);
                    worst[k] = std::max(worst[k], d);
                    EXPECT(finite && d <= tol[k], "grouped %s, %s, op %d, offset %d: %g apart%s", c.name,
                           k == 0 ? "fp32" : k == 1 ? "tf32" : "bf16", op, offset, d, finite ? "" : ", not finite");
                }
    std::printf("  grouped expert products: fp32 %.1e, tf32 %.1e, bf16 %.1e apart\n", worst[0], worst[1], worst[2]);
}

// The step on the GPU (forward_train, finish_step: train --gpu since #50)
// against the same step driven from the host as GpuBackend did it before:
// the norm, the clip scale and the update check computed on the host from
// numbers copied back, the router biases nudged on the host
// (Model::update_router_bias) and sent back. Two trainers from the same
// weights, on the same batches of two micro-batches each. Gradients add up
// with atomics in a varying order, so the two agree to rounding, not bit
// for bit. Then a weight set to NaN: the GPU skips the update, keeps every
// weight and the optimizer step, and drops the step's expert loads.
void step_on_gpu(const char* name, const Config& cfg, bool muon, double max_norm) {
    Model<float> host(cfg);  // the weights, and the host side's biases and loads
    host.init(7);
    Rng rng(8);
    for (auto& layer : host.router_bias)
        for (auto& b : layer) b = float(0.01 * rng.normal());
    AdamW opt;
    GpuTrainer old_way(cfg, host.infos, Matmul::fp32, true), new_way(cfg, host.infos, Matmul::fp32, true);
    for (GpuTrainer* g : {&old_way, &new_way}) {
        g->set_params(host.params.data());
        g->set_adam_state(0, nullptr, nullptr);
        g->set_router_bias(host.router_bias);
        g->set_muon(muon, opt.muon_momentum, opt.muon_steps);
    }
    const int B = 2, Tn = cfg.seq_len, micros = 2;
    const double lr = 1e-2, rate = 1e-2;
    std::vector<float> pa(host.num_params()), pb(pa.size());
    std::vector<std::vector<double>> interval(cfg.n_layers, std::vector<double>(std::max(1, cfg.n_experts), 0.0));
    double worst_loss = 0, worst_norm = 0, worst_param = 0;
    bool biases_equal = true, ratios_equal = true;
    for (int step = 0; step < 4; ++step) {
        std::vector<std::vector<int>> tok(micros), tgt(micros);
        size_t valid[micros], total = 0;
        for (int k = 0; k < micros; ++k) {
            random_batch(rng, cfg.vocab, B * Tn, tok[k], tgt[k]);
            valid[k] = count_targets(tgt[k].data(), tgt[k].size());
            total += valid[k];
        }
        // The host-driven step.
        old_way.zero_grad();
        TokenLossMean loss;
        for (int k = 0; k < micros; ++k) {
            loss.add(old_way.forward(tok[k].data(), tgt[k].data(), B, Tn, true), valid[k]);
            old_way.backward(double(valid[k]) / double(total));
        }
        const double norm = old_way.grad_norm();
        const double scale = max_norm > 0 && norm > max_norm ? double(float(max_norm / norm)) : 1.0;
        old_way.adamw_step(lr, opt.beta1, opt.beta2, opt.eps, opt.weight_decay, scale);
        const auto loads = old_way.take_expert_load();
        for (size_t l = 0; l < loads.size() && cfg.moe(); ++l)
            for (size_t e = 0; e < loads[l].size(); ++e) {
                host.expert_load[l][e] = loads[l][e];
                interval[l][e] += loads[l][e];
            }
        const double ratio = host.max_load_ratio();
        host.update_router_bias(rate);
        old_way.set_router_bias(host.router_bias);
        // The step on the GPU.
        new_way.zero_grad();
        for (int k = 0; k < micros; ++k) {
            new_way.forward_train(tok[k].data(), tgt[k].data(), B, Tn);
            new_way.backward(double(valid[k]) / double(total));
        }
        const GpuTrainer::StepStats s =
            new_way.finish_step(lr, opt.beta1, opt.beta2, opt.eps, opt.weight_decay, max_norm, rate);
        EXPECT(s.ok, "%s: a finite step was skipped", name);
        worst_loss = std::max(worst_loss, std::abs(s.ce_sum / double(total) - loss.value()) / loss.value());
        worst_norm = std::max(worst_norm, std::abs(s.norm - norm) / norm);
        old_way.get_params(pa.data());
        new_way.get_params(pb.data());
        worst_param = std::max(worst_param, rel_l2(pb.data(), pa.data(), pa.size()));
        if (cfg.moe()) {
            biases_equal &= new_way.get_router_bias() == host.router_bias;
            ratios_equal &= new_way.last_load_ratio() == ratio;
        }
    }
    if (cfg.moe())
        EXPECT(new_way.take_interval_load() == interval, "%s: the interval's expert loads differ", name);
    EXPECT(biases_equal && ratios_equal, "%s: router biases or load ratios differ from the host's", name);
    std::printf("  %-16s loss %.1e, norm %.1e, params %.1e%s\n", name, worst_loss, worst_norm, worst_param,
                cfg.moe() ? ", biases and loads exact" : "");
    EXPECT(worst_loss <= 1e-6 && worst_norm <= 1e-5 && worst_param <= 1e-5,
           "%s: the GPU step differs from the host-driven one", name);

    // A NaN weight: nothing changes but the step's loads, which go.
    new_way.get_params(pb.data());
    pb[0] = std::numeric_limits<float>::quiet_NaN();
    new_way.set_params(pb.data());
    long step_before = 0, step_after = 0;
    std::vector<float> m(pb.size()), v(pb.size()), after(pb.size());
    new_way.get_adam_state(&step_before, m.data(), v.data());
    const auto bias_before = new_way.get_router_bias();
    std::vector<int> tok, tgt;
    random_batch(rng, cfg.vocab, B * Tn, tok, tgt);
    new_way.zero_grad();
    new_way.forward_train(tok.data(), tgt.data(), B, Tn);
    new_way.backward(1.0);
    const GpuTrainer::StepStats s =
        new_way.finish_step(lr, opt.beta1, opt.beta2, opt.eps, opt.weight_decay, max_norm, rate);
    new_way.get_params(after.data());
    new_way.get_adam_state(&step_after, m.data(), v.data());
    bool load_dropped = true;
    for (const auto& layer : new_way.take_expert_load())
        for (double c : layer) load_dropped &= c == 0;
    EXPECT(!s.ok && std::memcmp(after.data(), pb.data(), pb.size() * sizeof(float)) == 0 &&
               step_after == step_before && new_way.get_router_bias() == bias_before && load_dropped,
           "%s: a non-finite step changed the weights, the optimizer step, the biases or kept its loads", name);
}

int main() {
    ThreadPool tp(4);
    g_pool = &tp;
    std::printf("%s\n", GpuTrainer::device_name().c_str());
    extreme_cross_entropy();
    masked_accumulation();
    fused_kernels();
    grouped_products();

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
    ternary_codes_on_gpu(ours);
    Config nope = dense;
    nope.nope_every = 2;
    compare("dense, half NoPE", nope, 1e-5, 1e-4, 1e-4);

    // Muon moves the weights about as much as the gradients differ, ten
    // times more than AdamW does, hence the looser weight tolerance.
    compare("dense muon", dense, 1e-5, 1e-4, 1e-3, {.muon = true});
    compare("moe muon", moe, 1e-5, 1e-4, 1e-3, {.muon = true, .clip = 0.5});

    // The experts' grouped products (#49) in the same runs: exact, TF32,
    // bf16, ternary, Muon, top-1 and unlikelihood.
    compare("moe+shared, grouped", moe, 1e-5, 1e-4, 1e-4, {.grouped = true});
    compare("moe top-1, grouped", moe1, 1e-5, 1e-4, 1e-4, {.clip = 0.5, .grouped = true});
    compare("moe tf32, grouped", moe, 3e-4, 2e-1, 2e-2, {.matmul = Matmul::tf32, .grouped = true});
    compare("moe bf16, grouped", moe, 1.2e-3, 9e-1, 8e-2, {.matmul = Matmul::bf16, .grouped = true});
    compare("moe muon bf16, grouped", moe, 5e-5, 5e-1, 1e-2, {.muon = true, .matmul = Matmul::bf16, .grouped = true});
    compare("ternary moe, grouped", ours, 1e-3, 3e-2, 1e-2, {.grouped = true});
    compare("moe ul, grouped", moe, 1e-5, 1e-4, 1e-4, {.grouped = true, .ul_alpha = 0.5, .ul_ngram = 3});

    // The whole step on the GPU (#50), plain and clipped, AdamW and Muon.
    step_on_gpu("dense step", dense, false, 0.0);
    step_on_gpu("moe step, clipped", moe, false, 0.05);
    step_on_gpu("moe muon step", moe, true, 0.05);

    // Unlikelihood (#34): the term on both sides, exact products and bf16
    // (whose gradients go through the bf16 copy of the logits' gradient).
    unlikelihood_edges();
    compare("dense ul", dense, 1e-5, 1e-4, 1e-4, {.ul_alpha = 0.5});
    compare("moe ul", moe, 1e-5, 1e-4, 1e-4, {.ul_alpha = 0.5, .ul_ngram = 3});
    compare("dense ul bf16", dense, 3e-4, 3e-1, 5e-2, {.matmul = Matmul::bf16, .ul_alpha = 0.5});

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
    eval_on_gpu();

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
