// test_model.h — whole-model gradient checks and the BitLinear pieces.
#pragma once

#include "../src/generate.h"
#include "../src/infer.h"
#include "../src/model.h"
#include "test_util.h"
#include "test_ste.h"

// Random parameters with enough spread that the loss is far from flat and
// MoE routing decisions are far from ties (a tie flipping under a 1e-5
// nudge would break the finite difference, not the gradient).
inline void randomize_for_gradcheck(Model<double>& m, Rng& rng) {
    for (const auto& p : m.infos)
        for (size_t i = 0; i < p.size; ++i)
            m.params[p.offset + i] = p.rows == 1 ? 1.0 + 0.3 * rng.normal() : 0.4 * rng.normal();
    m.weights_dirty = true;
}

inline void model_gradcheck(const char* what, Config cfg, uint64_t seed) {
    Rng rng(seed);
    Model<double> m(cfg);
    randomize_for_gradcheck(m, rng);
    const int B = 2, Tn = 5, N = B * Tn;
    std::vector<int> tokens(N), targets(N);
    for (int n = 0; n < N; ++n) {
        tokens[n] = int(rng.below(cfg.vocab));
        targets[n] = int(rng.below(cfg.vocab));
    }
    targets[3] = -1;  // an ignored position
    m.zero_grad();
    m.forward(tokens.data(), targets.data(), B, Tn);
    m.backward();
    const std::vector<double> analytic = m.grads;
    auto loss = [&] { return m.forward(tokens.data(), targets.data(), B, Tn); };
    // Check every tensor separately, so a failure names the culprit.
    double worst = 0.0;
    for (const auto& p : m.infos) {
        std::vector<double> x(m.params.begin() + p.offset, m.params.begin() + p.offset + p.size);
        std::vector<double> g(analytic.begin() + p.offset, analytic.begin() + p.offset + p.size);
        auto loss_p = [&] {
            std::copy(x.begin(), x.end(), m.params.begin() + p.offset);
            return loss();
        };
        const double e = grad_error(x, g, loss_p, rng, 60);
        std::copy(x.begin(), x.end(), m.params.begin() + p.offset);
        worst = std::max(worst, e);
        EXPECT(e < 1e-6, "%s: gradient of %s off by %g", what, p.name.c_str(), e);
    }
    std::printf("  %s: %zu params in %zu tensors, worst gradient error %.1e\n", what,
                m.num_params(), m.infos.size(), worst);
}

inline void test_model_gradients() {
    Config dense;
    dense.vocab = 11;
    dense.seq_len = 8;
    dense.d_model = 8;
    dense.n_layers = 2;
    dense.n_heads = 2;
    dense.ffn_hidden = 12;
    model_gradcheck("dense", dense, 21);

    Config nope = dense;  // layer 1 without RoPE
    nope.nope_every = 2;
    model_gradcheck("dense, half NoPE", nope, 24);

    Config moe = dense;
    moe.n_experts = 4;
    moe.top_k = 2;
    moe.expert_hidden = 6;
    moe.n_shared = 1;
    model_gradcheck("moe+shared", moe, 22);

    moe.n_shared = 0;
    moe.top_k = 1;
    model_gradcheck("moe top-1", moe, 23);
}

// The float model must compute the same thing as the double one.
inline void test_model_float_vs_double() {
    Config cfg;
    cfg.vocab = 37;
    cfg.seq_len = 24;
    cfg.d_model = 48;
    cfg.n_layers = 2;
    cfg.n_heads = 3;
    cfg.n_experts = 4;
    cfg.top_k = 2;
    cfg.n_shared = 1;
    cfg.expert_hidden = 40;
    Model<float> mf(cfg);
    Model<double> md(cfg);
    mf.init(5);
    std::copy(mf.params.begin(), mf.params.end(), md.params.begin());
    Rng rng(6);
    const int B = 3, Tn = 24, N = B * Tn;
    std::vector<int> tokens(N), targets(N);
    for (int n = 0; n < N; ++n) tokens[n] = int(rng.below(cfg.vocab)), targets[n] = int(rng.below(cfg.vocab));
    mf.zero_grad();
    md.zero_grad();
    const double lf = mf.forward(tokens.data(), targets.data(), B, Tn);
    const double ld = md.forward(tokens.data(), targets.data(), B, Tn);
    mf.backward();
    md.backward();
    EXPECT(std::abs(lf - ld) < 1e-5 * ld, "float loss %.8f vs double %.8f", lf, ld);
    const double eg = max_abs_diff(mf.grads.data(), md.grads.data(), md.grads.size()) /
                      max_abs(md.grads.data(), md.grads.size());
    EXPECT(eg < 1e-4, "float vs double gradients differ by %g", eg);
    // A fresh model predicts roughly uniformly: loss ~ log(vocab).
    EXPECT(std::abs(ld - std::log(double(cfg.vocab))) < 0.1, "initial loss %g vs log V %g", ld,
           std::log(double(cfg.vocab)));
}

inline void test_bitlinear() {
    Rng rng(31);
    // Ternary weights: values in {-gamma, 0, +gamma}, gamma = mean |w|.
    auto w = random_vec<double>(1000, rng);
    std::vector<double> wq(w.size());
    const double gamma = ternary_quantize(wq.data(), w.data(), long(w.size()));
    double mean_abs = 0;
    for (double x : w) mean_abs += std::abs(x);
    mean_abs /= double(w.size());
    EXPECT(std::abs(gamma - mean_abs) < 1e-7, "gamma %g vs mean |w| %g", gamma, mean_abs);
    int counts[3] = {0, 0, 0};
    for (size_t i = 0; i < w.size(); ++i) {
        const double r = wq[i] / gamma;
        const bool ok = std::abs(r) < 1e-12 || std::abs(std::abs(r) - 1) < 1e-12;
        EXPECT(ok, "quantized weight %g is not in {-g, 0, g}", wq[i]);
        if (!ok) break;
        ++counts[int(std::lround(r)) + 1];
        // round(w / gamma) clipped to [-1, 1].
        const double expect = std::clamp(std::nearbyint(w[i] / gamma), -1.0, 1.0);
        EXPECT(std::lround(r) == std::lround(expect), "w=%g quantized to %g", w[i], r);
    }
    EXPECT(counts[0] > 200 && counts[1] > 200 && counts[2] > 200, "odd ternary split %d/%d/%d",
           counts[0], counts[1], counts[2]);

    // int8 activations: per-row max maps to ±127; error at most half a step.
    const int N = 4, C = 50;
    auto x = random_vec<double>(N * C, rng);
    std::vector<double> xq(x.size());
    act_quantize(xq.data(), x.data(), N, C);
    for (int n = 0; n < N; ++n) {
        double amax = 0;
        for (int c = 0; c < C; ++c) amax = std::max(amax, std::abs(x[n * C + c]));
        const double step = amax / 127;
        for (int c = 0; c < C; ++c) {
            const double q = xq[n * C + c] / step;
            EXPECT(std::abs(q - std::nearbyint(q)) < 1e-9 && std::abs(q) <= 127 + 1e-9, "bad level %g", q);
            EXPECT(std::abs(xq[n * C + c] - x[n * C + c]) <= step / 2 + 1e-12, "act error too big");
        }
    }

    // A whole ternary model: quantized copies, a real gradient.
    Config cfg;
    cfg.vocab = 19;
    cfg.seq_len = 8;
    cfg.d_model = 16;
    cfg.n_layers = 1;
    cfg.n_heads = 2;
    cfg.ffn_hidden = 24;
    cfg.ternary = true;
    Model<double> m(cfg);
    m.init(7);
    const int B = 1, Tn = 8;
    std::vector<int> tok(Tn), tgt(Tn);
    for (int i = 0; i < Tn; ++i) tok[i] = int(rng.below(cfg.vocab)), tgt[i] = int(rng.below(cfg.vocab));
    m.zero_grad();
    const double lq = m.forward(tok.data(), tgt.data(), B, Tn);
    m.backward();
    for (const auto& p : m.infos) {
        if (!p.ternary) continue;
        // Every ternary tensor's quantized copy has at most 3 distinct values.
        std::vector<double> vals(m.qparams.begin() + p.offset, m.qparams.begin() + p.offset + p.size);
        std::sort(vals.begin(), vals.end());
        vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
        EXPECT(vals.size() <= 3, "%s has %zu distinct quantized values", p.name.c_str(), vals.size());
    }
    // Quantization is really applied: the same weights in full precision
    // give a different loss.
    Config fp = cfg;
    fp.ternary = false;
    Model<double> mfp(fp);
    mfp.params = m.params;
    const double lfp = mfp.forward(tok.data(), tgt.data(), B, Tn);
    EXPECT(std::abs(lq - lfp) > 1e-6, "ternary loss %g equals full-precision loss %g", lq, lfp);
    double gnorm = 0;
    for (double g : m.grads) gnorm += g * g;
    EXPECT(gnorm > 0 && std::isfinite(gnorm), "ternary model has no gradient");
}

// --fp-attn: only the FFN is ternary. With the FFN's weights at zero, its
// output is zero however it is quantized, so the model must compute exactly
// what the full-precision model computes, gradients included, while the
// all-ternary model (whose attention is quantized) computes something else.
inline void test_fp_attention() {
    Config cfg;
    cfg.vocab = 23;
    cfg.seq_len = 8;
    cfg.d_model = 16;
    cfg.n_layers = 2;
    cfg.n_heads = 2;
    cfg.ffn_hidden = 24;
    cfg.ternary = cfg.fp_attention = true;
    Config fp = cfg, all = cfg;
    fp.ternary = fp.fp_attention = false;
    all.fp_attention = false;
    Model<double> m(cfg), mfp(fp), mall(all);
    m.init(5);
    auto is_ffn = [](const ParamInfo& p) { return p.name.ends_with("wfc") || p.name.ends_with("wproj"); };
    for (const auto& p : m.infos) {
        EXPECT(p.ternary == is_ffn(p), "%s: ternary is %d with --fp-attn", p.name.c_str(), int(p.ternary));
        if (is_ffn(p)) std::fill_n(&m.params[p.offset], p.size, 0.0);
    }
    mfp.params = mall.params = m.params;
    Rng rng(6);
    const int B = 2, Tn = 8, N = B * Tn;
    std::vector<int> tok(N), tgt(N);
    for (int n = 0; n < N; ++n) tok[n] = int(rng.below(cfg.vocab)), tgt[n] = int(rng.below(cfg.vocab));
    m.zero_grad();
    mfp.zero_grad();
    const double l = m.forward(tok.data(), tgt.data(), B, Tn), lfp = mfp.forward(tok.data(), tgt.data(), B, Tn);
    m.backward();
    mfp.backward();
    EXPECT(std::abs(l - lfp) <= 1e-12 * lfp, "--fp-attn loss %.15g, full precision %.15g", l, lfp);
    for (const auto& p : m.infos) {
        // The FFN's gradients see quantized inputs; ln2 only feeds the FFN.
        if (is_ffn(p) || p.name.ends_with("ln2")) continue;
        double diff = 0, mx = 0;
        for (size_t i = p.offset; i < p.offset + p.size; ++i) {
            diff = std::max(diff, std::abs(m.grads[i] - mfp.grads[i]));
            mx = std::max(mx, std::abs(mfp.grads[i]));
        }
        EXPECT(mx > 0 && diff <= 1e-12 * mx, "%s: --fp-attn gradient differs by %g (max %g)", p.name.c_str(),
               diff, mx);
    }
    const double lall = mall.forward(tok.data(), tgt.data(), B, Tn);
    EXPECT(std::abs(lall - lfp) > 1e-6, "all-ternary loss %g equals full-precision loss %g", lall, lfp);
}

// KV-cache decoding of left-padded sequences must give the logits the full
// forward pass gives each sequence on its own.
inline void test_decode_matches_forward() {
    const char* names[] = {"dense", "ternary MoE", "ternary MoE, fp attention", "dense, half NoPE"};
    for (int variant = 0; variant < 4; ++variant) {
        Config cfg;
        cfg.vocab = 29;
        cfg.seq_len = 16;
        cfg.d_model = 32;
        cfg.n_layers = 2;
        cfg.n_heads = 4;
        cfg.nope_every = variant == 3 ? 2 : 0;
        if (variant == 1 || variant == 2) {  // our architecture: ternary MoE with a shared expert
            cfg.n_experts = 4;
            cfg.top_k = 2;
            cfg.n_shared = 1;
            cfg.expert_hidden = 24;
            cfg.ternary = true;
            cfg.fp_attention = variant == 2;
        }
        Model<float> m(cfg);
        m.init(3);
        Rng rng(4);
        const std::vector<int> lens = {5, 9, 1};
        const int B = 3, L = 9, extra = 4;
        std::vector<std::vector<int>> seqs(B);
        std::vector<int> start(B);
        for (int b = 0; b < B; ++b) {
            for (int i = 0; i < lens[b] + extra; ++i) seqs[b].push_back(int(rng.below(cfg.vocab)));
            start[b] = L - lens[b];
        }
        // decoded[b][t]: logits after feeding token t of sequence b
        std::vector<std::vector<std::vector<float>>> decoded(B);
        m.decode_begin(B, start);
        for (int pos = 0; pos < L + extra; ++pos) {
            std::vector<int> cur(B);
            for (int b = 0; b < B; ++b) cur[b] = pos < start[b] ? 0 : seqs[b][pos - start[b]];
            const float* lg = m.decode_step(cur.data(), pos);
            for (int b = 0; b < B; ++b)
                if (pos >= start[b])
                    decoded[b].emplace_back(lg + size_t(b) * cfg.vocab, lg + size_t(b + 1) * cfg.vocab);
        }
        double worst = 0;
        for (int b = 0; b < B; ++b) {
            const int len = int(seqs[b].size());
            m.forward(seqs[b].data(), nullptr, 1, len);
            for (int t = 0; t < len; ++t) {
                const float* ref = m.output_row(t);
                double mx = 0, diff = 0;
                for (int v = 0; v < cfg.vocab; ++v) {
                    mx = std::max(mx, std::abs(double(ref[v])));
                    diff = std::max(diff, std::abs(double(ref[v]) - decoded[b][t][v]));
                }
                worst = std::max(worst, diff / mx);
            }
        }
        EXPECT(worst < 1e-4, "decode vs forward (%s): relative difference %g", names[variant], worst);
    }
}

// Int8 rows (the engine's embedding and full-precision attention): each
// weight within half a step of the original, and dot_f32_i8 (SIMD) against
// a double sum.
inline void test_int8_rows() {
    Rng rng(41);
    for (const int cols : {32, 70, 256}) {
        const int rows = 5;
        std::vector<float> w(size_t(rows) * cols), x(cols);
        for (auto& v : w) v = float(rng.normal());
        for (auto& v : x) v = float(rng.normal());
        w[3] = 0.0f;
        std::fill(w.begin() + cols, w.begin() + 2 * cols, 0.0f);  // an all-zero row
        const Int8Rows m = quantize_rows_int8(w.data(), rows, cols);
        double worst_w = 0, worst_dot = 0;
        for (int r = 0; r < rows; ++r) {
            double exact = 0, mag = 0;
            for (int c = 0; c < cols; ++c) {
                const size_t i = size_t(r) * cols + c;
                const double deq = double(m.scale[size_t(r)]) * m.q[i];
                if (m.scale[size_t(r)] > 0) worst_w = std::max(worst_w, std::abs(deq - w[i]) / m.scale[size_t(r)]);
                else EXPECT(m.q[i] == 0, "int8 rows: an all-zero row gave a nonzero weight");
                exact += double(x[size_t(c)]) * m.q[i];
                mag += std::abs(double(x[size_t(c)]) * m.q[i]);
            }
            const double got = dot_f32_i8(x.data(), &m.q[size_t(r) * cols], cols);
            if (mag > 0) worst_dot = std::max(worst_dot, std::abs(got - exact) / mag);
        }
        EXPECT(worst_w <= 0.5 + 1e-4, "int8 rows (%d columns): a weight is %g steps off", cols, worst_w);
        EXPECT(worst_dot < 1e-6, "dot_f32_i8 (%d columns) differs from the double sum by %g", cols, worst_dot);
    }
}

// A ternary model on the bitplane int8 kernels computes what the training
// forward pass computes (which fake-quantizes with the same arithmetic).
// The embedding (and, with --fp-attn, attention) runs on int8 rows: the
// reference model gets the same rounded weights.
inline void test_ternary_engine() {
    const char* names[] = {"dense", "MoE", "MoE, half NoPE", "MoE, fp attention",
                           "MoE, no shared, biased", "MoE, no shared, biased, fp attention"};
    for (int variant = 0; variant < 6; ++variant) {
        Config cfg;
        cfg.vocab = 40;
        cfg.seq_len = 24;
        cfg.d_model = 64;
        cfg.n_layers = 2;
        cfg.n_heads = 4;
        cfg.ternary = true;
        if (variant >= 1) {
            cfg.n_experts = 4;
            cfg.top_k = 2;
            cfg.n_shared = 1;
            cfg.expert_hidden = 64;
        }
        if (variant == 2) cfg.nope_every = 2;  // layer 1 without RoPE
        if (variant == 3 || variant == 5) cfg.fp_attention = true;
        if (variant >= 4) cfg.n_shared = 0;
        Model<float> m(cfg);
        m.init(17);
        if (variant >= 4) for (auto& bias : m.router_bias) bias = {0.5f, -0.4f, 0.2f, -0.9f};
        // Larger weights than the init, so activations are far from zero
        // and the comparison is meaningful.
        Rng rng(18);
        for (auto& w : m.params) w = float(rng.normal() * 0.3);
        m.weights_dirty = true;
        round_to_engine_int8(m);
        TernaryEngine eng(m);
        std::vector<int> toks(cfg.seq_len);
        for (auto& t : toks) t = int(rng.below(cfg.vocab));
        m.decode_begin(1, {0});
        // Training and inference sum in different orders, so an activation
        // within ~1e-5 of an int8 rounding midpoint can round to the
        // neighbouring level in one and not the other. That is rare and
        // small; everything else must agree to float rounding.
        double worst = 0;
        std::vector<double> errs;
        for (int pos = 0; pos < cfg.seq_len; ++pos) {
            const float* ref = m.decode_step(&toks[pos], pos);
            const float* got = eng.step(toks[pos]);
            double mx = 0, diff = 0;
            for (int v = 0; v < cfg.vocab; ++v) {
                mx = std::max(mx, std::abs(double(ref[v])));
                diff = std::max(diff, std::abs(double(ref[v]) - got[v]));
            }
            worst = std::max(worst, diff / mx);
            errs.push_back(diff / mx);
        }
        std::sort(errs.begin(), errs.end());
        const double median = errs[errs.size() / 2];
        EXPECT(median < 1e-5 && worst < 2e-2, "ternary engine vs training forward (%s): median %g, worst %g",
               names[variant], median, worst);
        // Bitplanes: 2 bits per ternary weight, versus 32 in training.
        size_t ternary_weights = 0;
        for (const auto& p : m.infos) ternary_weights += p.ternary ? p.size : 0;
        const size_t float_bytes = (m.num_params() - ternary_weights) * 4;
        EXPECT(eng.weight_bytes() < float_bytes + ternary_weights / 4 + 4096, "weight bytes %zu", eng.weight_bytes());
    }
}

// The reasoning eval scores every problem, including one whose prompt does
// not fit in the context (wrong, without stopping the run), and gives the
// same answers however the problems are batched, as long as the context
// leaves every batch room for max_new tokens.
inline void test_eval_reasoning() {
    Config cfg;
    cfg.vocab = 256;  // byte-level
    cfg.seq_len = 128;
    cfg.d_model = 32;
    cfg.n_layers = 2;
    cfg.n_heads = 2;
    Model<float> m(cfg);
    m.init(9);
    Rng rng(10);
    std::vector<Problem> ps;
    for (int i = 0; i < 7; ++i) ps.push_back(make_problem(i % 2 ? "add" : "chain", 1 + i % 3, rng));
    Problem big = ps[0];
    big.question = std::string(130, '7') + "+1";
    ps.insert(ps.begin() + 3, big);
    const EvalResult one = eval_reasoning(m, ps, {}, 1, 20, 100);
    const EvalResult three = eval_reasoning(m, ps, {}, 3, 20, 100);
    size_t over = 0;
    for (const auto& p : ps) over += p.completion().size() > 20;  // independent byte tokenizer count
    EXPECT(one.max_new == 20 && one.gold_over_cap == over && three.gold_over_cap == over,
           "gold generation-cap count depends on batching or misses a completion");
    EXPECT(one.total == 8 && three.total == 8, "eval scored %d and %d of 8 problems", one.total, three.total);
    bool too_long = false;
    for (const auto& [p, text] : three.failures)
        if (p.question == big.question) too_long = text.find("does not fit") != std::string::npos;
    EXPECT(too_long, "the over-long prompt was not reported as not fitting");
    bool same = one.correct == three.correct && one.failures.size() == three.failures.size();
    for (size_t i = 0; same && i < one.failures.size(); ++i)
        same = one.failures[i].second == three.failures[i].second;
    EXPECT(same, "eval answers depend on the batch size");
}

// Missing biases change the baseline's selected experts even though all
// matrix weights are identical. Compare decoding against an explicit clone,
// with a zero-bias negative control that must produce different logits.
inline void test_decode_baseline_routing_state() {
  for (bool hybrid : {false, true}) {
    Config cfg; cfg.vocab = 32; cfg.seq_len = 8; cfg.d_model = 32;
    cfg.n_heads = 2; cfg.n_layers = 1; cfg.n_experts = 4;
    cfg.top_k = 2; cfg.expert_hidden = 32; cfg.ternary = true;
    cfg.fp_attention = hybrid;
    cfg.nope_every = hybrid ? 1 : 0;
    Model<float> source(cfg); source.init(7);
    source.router_bias[0] = {1000, -1000, -1000, 1000};
    auto baseline = make_fp32_decode_baseline(source);
    Config fcfg = cfg; fcfg.ternary = fcfg.fp_attention = false;
    Model<float> reference(fcfg), zero_bias(fcfg);
    for (const auto& p : source.infos)
        for (size_t i = p.offset; i < p.offset + p.size; ++i)
            reference.params[i] = zero_bias.params[i] = p.ternary ? source.qparams[i] : source.params[i];
    reference.router_bias = source.router_bias;
    baseline.decode_begin(1, {0}); reference.decode_begin(1, {0}); zero_bias.decode_begin(1, {0});
    double missing_bias_difference = 0;
    for (int pos = 0; pos < 8; ++pos) {
        const int token = (3 * pos + 1) % cfg.vocab;
        const float* a = baseline.decode_step(&token, pos);
        const float* b = reference.decode_step(&token, pos);
        const float* z = zero_bias.decode_step(&token, pos);
        for (int v = 0; v < cfg.vocab; ++v) {
            EXPECT(std::isfinite(a[v]) && a[v] == b[v], "baseline did not preserve routed reference logits");
            missing_bias_difference = std::max(missing_bias_difference, std::abs(double(a[v]) - z[v]));
        }
    }
    EXPECT(missing_bias_difference > 1e-6, "routing-state fixture cannot detect missing biases");
  }
}

// generate gives exactly max_new tokens, or as many as the context allows
// (seq_len - L + 1 for prompts of up to L tokens), and stops each sequence
// where done says.
inline void test_generate_length() {
    Config cfg;
    cfg.vocab = 64;
    cfg.seq_len = 16;
    cfg.d_model = 16;
    cfg.n_layers = 1;
    cfg.n_heads = 2;
    Model<float> m(cfg);
    m.init(11);
    const std::vector<std::vector<int>> prompts = {{1, 2, 3}, {1, 4, 5, 6, 7}};
    auto never = [](const std::vector<int>&) { return false; };
    for (int max_new : {1, 7}) {
        const auto out = generate(m, prompts, max_new, never);
        EXPECT(out[0].size() == size_t(max_new) && out[1].size() == size_t(max_new),
               "asked for %d tokens, got %zu and %zu", max_new, out[0].size(), out[1].size());
    }
    const auto capped = generate(m, prompts, 100, never);  // L = 5: room for 16 - 5 + 1
    EXPECT(capped[0].size() == 12 && capped[1].size() == 12, "context-capped generation gave %zu and %zu tokens",
           capped[0].size(), capped[1].size());
    const auto stopped = generate(m, prompts, 7, [](const std::vector<int>& g) { return g.size() == 2; });
    EXPECT(stopped[0].size() == 2 && stopped[1].size() == 2, "done did not stop the sequences");
}


// A zero-attention one-layer model permits an independent FFN calculation:
// selection is argtopk(r+b), but the two mixture weights are softmax(r_sel).
inline void test_router_selection_and_gates() {
    Config cfg; cfg.vocab = 11; cfg.seq_len = 4; cfg.d_model = 8; cfg.n_heads = 2;
    cfg.n_layers = 1; cfg.n_experts = 4; cfg.top_k = 2; cfg.expert_hidden = 6;
    Model<double> m(cfg); m.init(7);
    for (const auto& p : m.infos) if (p.name.ends_with(".wqkv") || p.name.ends_with(".wo"))
        std::fill_n(m.params.begin() + p.offset, p.size, 0.0);
    auto w = [&](const char* name) { return m.params.data() + m.find(name)->offset; };
    const int C = cfg.d_model, F = cfg.expert_hidden, token = 1;
    std::vector<double> x(w("wte") + token*C, w("wte") + (token+1)*C), h(C), r(4);
    auto norm = [&](const std::vector<double>& in, const double* weight) {
        double ss = 0; for (double v : in) ss += v*v;
        std::vector<double> out(C);
        for (int i = 0; i < C; ++i) out[i] = in[i] / std::sqrt(ss/C + 1e-5) * weight[i];
        return out;
    };
    h = norm(x, w("l0.ln2"));
    for (int e = 0; e < 4; ++e) for (int j = 0; j < C; ++j) r[e] += h[j] * w("l0.router")[e*C+j];
    for (const auto& bias : std::vector<std::vector<double>>{{8,-8,-7,7}, {7,-8,-7,8}}) {
        m.router_bias[0] = bias;
        m.expert_load[0].assign(4, 0);
        std::vector<int> order{0,1,2,3};
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return r[a]+bias[a] > r[b]+bias[b]; });
        auto reference = [&](bool biased_gate) {
            double g[2], sum = 0;
            for (int k = 0; k < 2; ++k) sum += g[k] = std::exp(r[order[k]] + (biased_gate ? bias[order[k]] : 0));
            std::vector<double> residual = x;
            for (int k = 0; k < 2; ++k) {
                const int e = order[k]; std::vector<double> sw(F);
                for (int f = 0; f < F; ++f) {
                    double gate = 0, up = 0;
                    for (int j = 0; j < C; ++j) {
                        gate += h[j] * w("l0.efc")[(e*2*F+f)*C+j];
                        up += h[j] * w("l0.efc")[(e*2*F+F+f)*C+j];
                    }
                    sw[f] = gate/(1+std::exp(-gate))*up;
                }
                for (int i = 0; i < C; ++i) for (int f = 0; f < F; ++f)
                    residual[i] += g[k]/sum * sw[f] * w("l0.eproj")[(e*C+i)*F+f];
            }
            const auto final = norm(residual, w("lnf"));
            std::vector<double> logits(cfg.vocab, 0);
            for (int v = 0; v < cfg.vocab; ++v) for (int j = 0; j < C; ++j)
                logits[v] += final[j]*w("wte")[v*C+j];
            return logits;
        };
        const auto expected = reference(false), wrong = reference(true);
        m.forward(&token, nullptr, 1, 1, true);
        for (int e = 0; e < 4; ++e)
            EXPECT(m.expert_load[0][e] == (e==order[0] || e==order[1] ? 1 : 0), "selection is not argtopk(r+b)");
        double negative_difference = 0;
        for (int v = 0; v < cfg.vocab; ++v) {
            EXPECT(std::abs(m.output_row(0)[v]-expected[v]) < 1e-12, "gate is not softmax of raw selected logits");
            negative_difference = std::max(negative_difference, std::abs(wrong[v]-expected[v]));
        }
        EXPECT(negative_difference > 1e-6, "distinct-bias fixture cannot detect biased gates");
    }
}

inline void test_router_balancing() {
    Config cfg; cfg.vocab = 11; cfg.seq_len = 4; cfg.d_model = 8; cfg.n_heads = 2;
    cfg.n_layers = 2; cfg.n_experts = 4; cfg.top_k = 1; cfg.expert_hidden = 6;
    Model<double> m(cfg);
    m.expert_load = {{3,1,0,0}, {0,2,1,1}};
    EXPECT(m.max_load_ratio() == 3, "max_load_ratio is not the busiest layer's normalized load");
    m.update_router_bias(0.125);
    // Mean is one: the equal-load expert is unchanged, not increased.
    EXPECT(m.router_bias[0] == std::vector<double>({-0.125,0,0.125,0.125}) &&
           m.router_bias[1] == std::vector<double>({0.125,-0.125,0,0}), "balancing bias has wrong sign or tie policy");
    for (const auto& load : m.expert_load) EXPECT(load == std::vector<double>(4,0), "balance did not reset loads");
    EXPECT(m.max_load_ratio() == 1, "empty loads do not report neutral balance");
    const auto bias = m.router_bias;
    m.expert_load = {{1,1,1,1}, {0,0,0,0}}; m.update_router_bias(0.125);
    EXPECT(m.router_bias == bias, "balanced or empty loads changed bias");
    m.expert_load[0] = {4,0,0,0}; m.update_router_bias(0);
    EXPECT(m.router_bias == bias && m.expert_load[0] == std::vector<double>(4,0), "zero-rate discard changed bias/kept loads");
}

inline void test_activation_int8_contract() {
    Rng rng(78);
    for (int C : {1,7,8,9,15,16,17,31,32,33,127}) for (int mode = 0; mode < 5; ++mode) {
        const int N = 3; std::vector<float> x(N*C), got(N*C);
        std::vector<int8_t> q(C);
        for (auto& v : x) v = float(rng.normal());
        if (mode == 1) std::fill(x.begin(), x.end(), 0);
        if (mode == 2) for (int i = 0; i < N*C; ++i) x[i] = (i % 255 - 127)*0.5f;
        if (mode == 3) for (auto& v : x) v *= 1e-10f;
        if (mode == 4) for (auto& v : x) v *= 1e20f;
        act_quantize(got.data(), x.data(), N, C);
        for (int n = 0; n < N; ++n) {
            const float scale = quantize_int8(x.data()+n*C, C, q.data());
            for (int j = 0; j < C; ++j)
                EXPECT(got[n*C+j] == scale*float(q[j]), "training/packed int8 levels differ at C=%d mode=%d", C, mode);
        }
    }
}

inline void test_model() {
    test_activation_int8_contract();
    test_router_balancing();
    test_router_selection_and_gates();
    test_decode_baseline_routing_state();
    test_bitlinear_ste();
    test_generate_length();
    test_eval_reasoning();
    test_int8_rows();
    test_ternary_engine();
    test_decode_matches_forward();
    test_fp_attention();
    test_model_gradients();
    test_model_float_vs_double();
    test_bitlinear();
}
