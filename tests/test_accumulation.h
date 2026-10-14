// A partition of a batch must preserve the global masked-token objective.
#pragma once
#include "../src/train.h"
#include "test_util.h"

inline AccumulationBatch masked_accumulation_fixture() {
    AccumulationBatch g(1, 8, 4);
    const int counts[] = {0, 1, 3, 8};
    for (int m = 0; m < g.micros; ++m)
        for (int t = 0; t < g.seq; ++t) {
            g.input(m)[t] = (3 * m + t) % 13;
            g.target(m)[t] = t < counts[m] ? (7 * m + 2 * t + 1) % 13 : -1;
        }
    g.refresh_counts();
    return g;
}

template<class T>
inline double accumulation_difference(const std::vector<T>& a, const std::vector<T>& b) {
    EXPECT(a.size() == b.size(), "different accumulation tensor sizes");
    if (a.size() != b.size()) return INFINITY;
    double d = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return INFINITY;
        d = std::max(d, std::abs(double(a[i]) - double(b[i])));
    }
    return d;
}

inline void test_masked_accumulation() {
    auto group = masked_accumulation_fixture();
    EXPECT(group.total_valid == 12 && group.valid == std::vector<size_t>({0, 1, 3, 8}),
           "incorrect fixture target counts");
    TokenLossMean mean;
    mean.add(NAN, 0); mean.add(10, 1); mean.add(2, 3);
    EXPECT(mean.value() == 4 && mean.count == 4, "token-weighted metric disagrees with scalar reference");
    for (int variant = 0; variant < 8; ++variant) {
        Config cfg;
        cfg.vocab = 13; cfg.seq_len = 8; cfg.d_model = 8;
        cfg.n_heads = 2; cfg.n_layers = 1; cfg.ffn_hidden = 12;
        cfg.ternary = variant % 2 || variant == 4;
        cfg.fp_attention = variant == 4 || variant == 5;
        if (variant >= 4) cfg.n_layers = 2;
        if (variant >= 5) cfg.nope_every = variant == 6 ? 1 : 2;
        if (variant == 2 || variant == 3 || variant == 5 || variant == 7) {
            cfg.n_experts = 4; cfg.top_k = 2; cfg.expert_hidden = 6; cfg.n_shared = 1;
        }
        Model<double> full(cfg), micro(cfg);
        full.init(57); micro.params = full.params;
        full.zero_grad();
        const double lf = full.forward(group.inputs.data(), group.targets.data(), 4, 8, true);
        full.backward();
        const double lm = accumulated_backward(micro, group);
        const double gd = accumulation_difference(full.grads, micro.grads);
        EXPECT(std::abs(lf - lm) < 1e-12 && gd < 1e-12,
               "variant %d double partition changed loss/gradient: %g / %g", variant, lf - lm, gd);
        EXPECT(full.expert_load == micro.expert_load, "partition changed routing counts");
        if (variant == 0) {
            const auto analytic = micro.grads;
            Rng rng(111);
            auto loss = [&] { return full.forward(group.inputs.data(), group.targets.data(), 4, 8, false); };
            const double error = grad_error(full.params, analytic, loss, rng, 200);
            EXPECT(error < 1e-6, "accumulated gradient differs from global-loss finite differences: %g", error);
            std::printf("  masked global-loss finite-difference error %.1e\n", error);
            // This fixture must actually detect the former equal-micro mean.
            micro.zero_grad();
            double old_loss = 0;
            for (int m = 0; m < group.micros; ++m) {
                old_loss += micro.forward(group.input(m), group.target(m), 1, 8, true) / 4;
                micro.backward(0.25);
            }
            EXPECT(std::abs(old_loss - lm) > 0.1 && accumulation_difference(analytic, micro.grads) > 0.01,
                   "regression fixture does not distinguish equal-micro weighting");
        }
        Model<float> ff(cfg), fm(cfg);
        ff.init(57); fm.params = ff.params;
        AdamW of, om;
        double worst_grad = 0, worst_param = 0, worst_moment = 0;
        for (int step = 0; step < 3; ++step) {
            ff.zero_grad();
            const double a = ff.forward(group.inputs.data(), group.targets.data(), 4, 8, true);
            ff.backward();
            const double b = accumulated_backward(fm, group);
            worst_grad = std::max(worst_grad, accumulation_difference(ff.grads, fm.grads));
            EXPECT(std::abs(a - b) < 2e-6, "variant %d float loss partition mismatch", variant);
            const double nf = clip_grad_norm(ff, 0.1), nm = clip_grad_norm(fm, 0.1);
            EXPECT(std::abs(nf - nm) < 2e-5, "partition changed clipping norm");
            of.update(ff, 0.001); om.update(fm, 0.001);
            ff.update_router_bias(0.001); fm.update_router_bias(0.001);
            EXPECT(ff.router_bias == fm.router_bias, "partition changed router balancing");
            worst_param = std::max(worst_param, accumulation_difference(ff.params, fm.params));
            worst_moment = std::max({worst_moment, accumulation_difference(of.m, om.m),
                                    accumulation_difference(of.v, om.v)});
        }
        EXPECT(worst_grad < 2e-5 && worst_param < 2e-6 && worst_moment < 2e-7,
               "variant %d float equivalence failed: grad %g param %g moment %g",
               variant, worst_grad, worst_param, worst_moment);
        std::printf("  variant %d masked accumulation: double grad %.1e; float grad %.1e param %.1e moment %.1e\n",
                    variant, gd, worst_grad, worst_param, worst_moment);
        auto empty = group;
        std::fill(empty.targets.begin(), empty.targets.end(), -1); empty.refresh_counts();
        fm.grads.assign(fm.grads.size(), 1);
        EXPECT(accumulated_backward(fm, empty) == 0, "empty accumulated loss nonzero");
        for (float g : fm.grads) EXPECT(g == 0, "empty group retained stale gradients");
        for (const auto& load : fm.expert_load)
            for (double x : load) EXPECT(x == 0, "empty group added router loads");
    }
    TokenData data;
    data.vocab = 13;
    for (int i = 0; i < 30; ++i) { data.tokens.push_back(i % 13); data.mask.push_back(i % 3 != 0); }
    Rng a(99), b(99);
    AccumulationBatch staged(2, 8, 3);
    staged.sample(data, a, 1, 29);
    std::vector<int> inp(48), tgt(48);
    data.sample_batch(b, 6, 8, 1, 29, inp.data(), tgt.data());
    EXPECT(inp == staged.inputs && tgt == staged.targets, "staging changed sample order");
    for (int i = 0; i < 4; ++i) EXPECT(a.state()[i] == b.state()[i], "staging changed sampler RNG");
}
