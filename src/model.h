// model.h — the transformer: configuration, parameters, forward and backward.
//
// Architecture (a modern decoder-only LM, sized by Config):
//   token embedding (tied with the output head)
//   n_layers x [ RMSNorm -> attention (fused QKV, RoPE, causal) -> residual
//                RMSNorm -> FFN -> residual ]
//   RMSNorm -> logits
// The FFN is either a dense SwiGLU, or a Mixture-of-Experts of SwiGLU
// experts with top-k routing (optionally plus always-on shared experts).
// With ternary = true, every projection inside the blocks is a BitLinear:
// ternary weights and int8 activations in the forward pass, which is what
// the inference kernels run.
#pragma once

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include "common.h"
#include "ops.h"

struct Config {
    int vocab = 256;
    int seq_len = 256;  // longest sequence the model handles
    int d_model = 256;
    int n_layers = 4;
    int n_heads = 4;
    int ffn_hidden = 0;  // dense FFN hidden size; 0 picks ~8/3 d_model (rounded to 32)
    // Mixture-of-Experts: n_experts = 0 means a dense FFN.
    int n_experts = 0;
    int top_k = 2;
    int expert_hidden = 0;  // per expert; 0 picks ffn_hidden / top_k
    int n_shared = 0;       // always-on shared experts, each of expert_hidden
    bool ternary = false;   // BitLinear projections in the blocks

    int head_size() const { return d_model / n_heads; }
    int dense_hidden() const {
        return ffn_hidden ? ffn_hidden : std::max(32, (d_model * 8 / 3 + 31) / 32 * 32);
    }
    int ehidden() const { return expert_hidden ? expert_hidden : std::max(8, dense_hidden() / top_k); }
    bool moe() const { return n_experts > 0; }
    // Hidden size of the always-on FFN part: the dense FFN, or the shared experts.
    int always_hidden() const { return moe() ? n_shared * ehidden() : dense_hidden(); }

    void validate() const {
        CHECK(vocab > 0 && seq_len > 0 && n_layers > 0, "bad sizes");
        CHECK(d_model % n_heads == 0, "d_model %d not divisible by n_heads %d", d_model, n_heads);
        CHECK(head_size() % 2 == 0, "head size must be even for RoPE");
        CHECK(!moe() || (top_k >= 1 && top_k <= n_experts), "top_k must be in [1, n_experts]");
    }
};

// One named tensor inside the flat parameter array.
struct ParamInfo {
    std::string name;
    size_t offset, size;
    int rows, cols;    // matrices; norms have rows = 1
    bool decay;        // AdamW weight decay (matrices, not norms)
    bool ternary;      // quantized in the forward pass
    int count = 1;     // stacked copies (experts); each copy quantized separately
};

template <class T>
class Model {
public:
    Config cfg;
    std::vector<ParamInfo> infos;
    std::vector<T> params, grads;
    // Quantized copies of the ternary tensors (same offsets); refreshed by
    // forward() whenever the weights have changed.
    std::vector<T> qparams;
    bool weights_dirty = true;

    // Router bias for load balancing without an auxiliary loss
    // (DeepSeek-V3): added to the logits when choosing experts, never when
    // weighting them, and nudged after each step toward an even load.
    std::vector<std::vector<T>> router_bias;   // [layer][expert]
    std::vector<std::vector<double>> expert_load;  // tokens routed, since last reset

    explicit Model(const Config& c) : cfg(c) {
        cfg.validate();
        build_layout();
        params.assign(n_params_, T(0));
        grads.assign(n_params_, T(0));
        qparams.assign(n_params_, T(0));
        rope = RopeTable<T>(cfg.seq_len, cfg.head_size());
        router_bias.assign(cfg.n_layers, std::vector<T>(std::max(1, cfg.n_experts), T(0)));
        expert_load.assign(cfg.n_layers, std::vector<double>(std::max(1, cfg.n_experts), 0.0));
    }

    size_t num_params() const { return n_params_; }

    // Parameters a token actually uses: with MoE, only top_k of the experts.
    size_t active_params() const {
        if (!cfg.moe()) return n_params_;
        const size_t per_expert = size_t(3) * cfg.ehidden() * cfg.d_model;
        return n_params_ - size_t(cfg.n_layers) * (cfg.n_experts - cfg.top_k) * per_expert;
    }

    // N(0, 0.02) matrices, with the projections that write into the residual
    // stream scaled down by sqrt(2 * n_layers); norms start at 1.
    void init(uint64_t seed) {
        Rng rng(seed);
        const double resid_scale = 1.0 / std::sqrt(2.0 * cfg.n_layers);
        for (const auto& p : infos) {
            T* w = &params[p.offset];
            if (p.rows == 1) {
                std::fill(w, w + p.size, T(1));
                continue;
            }
            const bool resid = p.name.ends_with("wo") || p.name.ends_with("proj");
            const double sd = 0.02 * (resid ? resid_scale : 1.0);
            for (size_t i = 0; i < p.size; ++i) w[i] = T(rng.normal() * sd);
        }
        weights_dirty = true;
    }

    // Activation buffers for batches up to B x Tn.
    void allocate(int B, int Tn) {
        CHECK(Tn <= cfg.seq_len, "sequence length %d > seq_len %d", Tn, cfg.seq_len);
        if (B * Tn <= B_ * T_ && Tn <= T_ && B <= B_) return;
        B_ = B;
        T_ = Tn;
        const size_t N = size_t(B) * Tn, C = cfg.d_model, V = cfg.vocab;
        const size_t F = cfg.always_hidden(), E = cfg.n_experts, K = cfg.top_k;
        const size_t Fe = cfg.ehidden(), NK = cfg.moe() ? N * K : 0;
        acts.resize(cfg.n_layers);
        for (auto& a : acts) {
            a.ln1.resize(N * C); a.rstd1.resize(N); a.qkv.resize(N * 3 * C);
            a.att.resize(size_t(B) * cfg.n_heads * Tn * Tn); a.atty.resize(N * C);
            a.resid1.resize(N * C); a.ln2.resize(N * C); a.rstd2.resize(N);
            a.resid2.resize(N * C); a.fc.resize(N * 2 * F); a.sw.resize(N * F);
            if (cfg.moe()) {
                a.rlogits.resize(N * E); a.sel.resize(N * K); a.gate.resize(N * K);
                a.order.resize(NK); a.row_of.resize(NK); a.expert_start.resize(E + 1);
                a.xg.resize(NK * C); a.efc.resize(NK * 2 * Fe); a.esw.resize(NK * Fe);
                a.ey.resize(NK * C);
            }
        }
        x0.resize(N * C); lnf.resize(N * C); rstdf.resize(N);
        logits.resize(N * V); losses.resize(N);
        tokens_.resize(N); targets_.resize(N);
        const size_t widest = std::max({C, F, Fe});
        xq.resize(std::max(N, NK) * widest);
        out_tmp.resize(N * C);
        // backward scratch
        dresid.resize(N * C); dC.resize(N * C); dqkv.resize(N * 3 * C); datty.resize(N * C);
        dfc.resize(std::max(N * 2 * F, NK * 2 * Fe)); dsw.resize(std::max(N * F, NK * Fe));
        if (cfg.moe()) { dey.resize(NK * C); dxg.resize(NK * C); drl.resize(N * E); dffn.resize(N * C); }
    }

    // Runs the model on tokens[B * Tn]. With targets (-1 = ignore), returns
    // the mean cross-entropy and keeps what backward() needs; logits then
    // hold probabilities. Without targets, logits hold raw logits; returns 0.
    double forward(const int* tokens, const int* targets, int B, int Tn, bool training = false) {
        allocate(B, Tn);
        B_cur = B;
        T_cur = Tn;
        const int N = B * Tn, C = cfg.d_model, V = cfg.vocab;
        if (weights_dirty) quantize_weights();
        std::copy(tokens, tokens + N, tokens_.begin());
        for (int n = 0; n < N; ++n)
            CHECK(tokens[n] >= 0 && tokens[n] < V, "token %d out of range", tokens[n]);

        embedding_forward(x0.data(), tokens, P(wte_), N, C);
        const T* x = x0.data();
        for (int l = 0; l < cfg.n_layers; ++l) {
            auto& a = acts[l];
            const auto& o = layer_[l];
            rmsnorm_forward(a.ln1.data(), a.rstd1.data(), x, P(o.ln1), N, C);
            proj_forward(a.qkv.data(), a.ln1.data(), o.wqkv, N, C, 3 * C);
            rope_forward(a.qkv.data(), rope, B, Tn, C, cfg.n_heads);
            attention_forward(a.atty.data(), a.att.data(), a.qkv.data(), B, Tn, C, cfg.n_heads);
            proj_forward(out_tmp.data(), a.atty.data(), o.wo, N, C, C);
            add_forward(a.resid1.data(), x, out_tmp.data(), long(N) * C);
            rmsnorm_forward(a.ln2.data(), a.rstd2.data(), a.resid1.data(), P(o.ln2), N, C);
            ffn_forward(l, out_tmp.data(), N, training);
            add_forward(a.resid2.data(), a.resid1.data(), out_tmp.data(), long(N) * C);
            x = a.resid2.data();
        }
        rmsnorm_forward(lnf.data(), rstdf.data(), x, P(lnf_), N, C);
        linear_forward(logits.data(), lnf.data(), P(wte_), N, C, V);  // tied head
        if (!targets) {
            has_targets_ = false;
            return 0.0;
        }
        has_targets_ = true;
        std::copy(targets, targets + N, targets_.begin());
        n_valid_ = 0;
        for (int n = 0; n < N; ++n) n_valid_ += targets[n] >= 0;
        return cross_entropy_forward(logits.data(), losses.data(), targets, N, V);
    }

    // Accumulates d(loss * loss_scale) into grads, for the last forward().
    // Call zero_grad() first (once per optimizer step, so that gradient
    // accumulation over micro-batches is just several backward calls).
    void backward(double loss_scale = 1.0) {
        CHECK(has_targets_, "backward() needs a forward() with targets");
        const int B = B_cur, Tn = T_cur, N = B * Tn, C = cfg.d_model, V = cfg.vocab;
        if (n_valid_ == 0) return;
        T* dlogits = logits.data();  // in place: probabilities become gradients
        cross_entropy_backward(dlogits, logits.data(), targets_.data(), N, V,
                               T(loss_scale / n_valid_));
        const T* xL = cfg.n_layers ? acts.back().resid2.data() : x0.data();
        zero(dC, size_t(N) * C);
        linear_backward(dC.data(), G(wte_), dlogits, lnf.data(), P(wte_), N, C, V);
        zero(dresid, size_t(N) * C);
        rmsnorm_backward(dresid.data(), G(lnf_), dC.data(), xL, P(lnf_), rstdf.data(), N, C);
        for (int l = cfg.n_layers - 1; l >= 0; --l) {
            auto& a = acts[l];
            const auto& o = layer_[l];
            const T* x_in = l ? acts[l - 1].resid2.data() : x0.data();
            // resid2 = resid1 + ffn(rmsnorm(resid1)); dresid carries d resid2.
            zero(dC, size_t(N) * C);
            ffn_backward(l, dC.data(), dresid.data(), N);
            rmsnorm_backward(dresid.data(), G(o.ln2), dC.data(), a.resid1.data(), P(o.ln2),
                             a.rstd2.data(), N, C);
            // resid1 = x_in + wo(attention(rope(wqkv(rmsnorm(x_in)))))
            zero(datty, size_t(N) * C);
            proj_backward(datty.data(), dresid.data(), a.atty.data(), o.wo, N, C, C);
            zero(dqkv, size_t(N) * 3 * C);
            attention_backward(dqkv.data(), datty.data(), a.att.data(), a.qkv.data(), B, Tn, C,
                               cfg.n_heads);
            rope_backward(dqkv.data(), rope, B, Tn, C, cfg.n_heads);
            zero(dC, size_t(N) * C);
            proj_backward(dC.data(), dqkv.data(), a.ln1.data(), o.wqkv, N, C, 3 * C);
            rmsnorm_backward(dresid.data(), G(o.ln1), dC.data(), x_in, P(o.ln1), a.rstd1.data(), N, C);
        }
        embedding_backward(G(wte_), dresid.data(), tokens_.data(), N, C);
    }

    void zero_grad() { zero_fill(grads.data(), grads.size()); }

    // Nudges each router bias toward an even load: +rate for experts that
    // got fewer than average tokens since the last call, -rate for more.
    void update_router_bias(double rate) {
        if (!cfg.moe()) return;
        for (int l = 0; l < cfg.n_layers; ++l) {
            auto& load = expert_load[l];
            const double mean = std::accumulate(load.begin(), load.end(), 0.0) / cfg.n_experts;
            for (int e = 0; e < cfg.n_experts; ++e) {
                if (load[e] < mean) router_bias[l][e] += T(rate);
                else if (load[e] > mean) router_bias[l][e] -= T(rate);
            }
            std::fill(load.begin(), load.end(), 0.0);
        }
    }

    // Busiest expert's share relative to an even split (1.0 = perfectly
    // balanced), worst layer, over the loads accumulated since the last reset.
    double max_load_ratio() const {
        double worst = 1.0;
        for (const auto& load : expert_load) {
            const double total = std::accumulate(load.begin(), load.end(), 0.0);
            if (total <= 0) continue;
            worst = std::max(worst, *std::max_element(load.begin(), load.end()) * load.size() / total);
        }
        return worst;
    }

    const ParamInfo* find(const std::string& name) const {
        for (const auto& p : infos)
            if (p.name == name) return &p;
        return nullptr;
    }

    // Probabilities (after a forward with targets) or logits (without).
    const T* output_row(int n) const { return &logits[size_t(n) * cfg.vocab]; }

    // ---- Incremental decoding with a KV cache ----
    //
    // Generates for B sequences at once, one token per sequence per step,
    // all at the same position. Sequences of different lengths are padded
    // on the left; start[b] is the first real position of sequence b, and
    // attention never looks before it. RoPE scores depend only on relative
    // positions, so a left-padded sequence computes exactly what it would
    // unpadded.
    void decode_begin(int B, const std::vector<int>& start) {
        CHECK(int(start.size()) == B, "decode_begin: %zu starts for %d sequences", start.size(), B);
        allocate(B, 1);
        has_targets_ = false;
        kv_B_ = B;
        kv_start_ = start;
        const size_t per_layer = size_t(B) * cfg.seq_len * cfg.d_model;
        kcache_.resize(cfg.n_layers);
        vcache_.resize(cfg.n_layers);
        for (int l = 0; l < cfg.n_layers; ++l) {
            kcache_[l].resize(per_layer);
            vcache_[l].resize(per_layer);
        }
    }

    // Feeds tokens[b] at position pos (0, 1, 2, ... in order) and returns the
    // logits for the next token, [B, vocab].
    const T* decode_step(const int* tokens, int pos) {
        CHECK(pos < cfg.seq_len, "decode position %d >= seq_len %d", pos, cfg.seq_len);
        const int B = kv_B_, N = B, C = cfg.d_model, V = cfg.vocab, NH = cfg.n_heads;
        const int HS = C / NH, S = cfg.seq_len;
        if (weights_dirty) quantize_weights();
        for (int n = 0; n < N; ++n) CHECK(tokens[n] >= 0 && tokens[n] < V, "token %d out of range", tokens[n]);
        embedding_forward(x0.data(), tokens, P(wte_), N, C);
        T* x = x0.data();  // the residual stream, updated in place
        const T scale = T(1) / std::sqrt(T(HS));
        for (int l = 0; l < cfg.n_layers; ++l) {
            auto& a = acts[l];
            const auto& o = layer_[l];
            rmsnorm_forward(a.ln1.data(), a.rstd1.data(), x, P(o.ln1), N, C);
            proj_forward(a.qkv.data(), a.ln1.data(), o.wqkv, N, C, 3 * C);
            rope_apply(a.qkv.data(), rope, B, 1, C, NH, false, pos);
            T* kc = kcache_[l].data();
            T* vc = vcache_[l].data();
            for (int b = 0; b < B; ++b) {
                const T* row = &a.qkv[size_t(b) * 3 * C];
                std::copy(row + C, row + 2 * C, kc + (size_t(b) * S + pos) * C);
                std::copy(row + 2 * C, row + 3 * C, vc + (size_t(b) * S + pos) * C);
            }
            pool().parallel_for(B * NH, [&](int j0, int j1) {
                static thread_local std::vector<T> p;
                p.resize(size_t(S));
                for (int j = j0; j < j1; ++j) {
                    const int b = j / NH, h = j % NH, s0 = kv_start_[b];
                    const T* q = &a.qkv[size_t(b) * 3 * C + h * HS];
                    T mx = -std::numeric_limits<T>::infinity();
                    for (int s = s0; s <= pos; ++s) {
                        p[s] = dot(q, kc + (size_t(b) * S + s) * C + h * HS, HS) * scale;
                        mx = std::max(mx, p[s]);
                    }
                    T sum = 0;
                    for (int s = s0; s <= pos; ++s) sum += p[s] = fast_exp(p[s] - mx);
                    T* y = &a.atty[size_t(b) * C + h * HS];
                    std::fill(y, y + HS, T(0));
                    for (int s = s0; s <= pos; ++s) {
                        const T w = p[s] / sum;
                        const T* v = vc + (size_t(b) * S + s) * C + h * HS;
                        for (int i = 0; i < HS; ++i) y[i] += w * v[i];
                    }
                }
            }, 1);
            proj_forward(out_tmp.data(), a.atty.data(), o.wo, N, C, C);
            add_into(x, out_tmp.data(), long(N) * C);
            rmsnorm_forward(a.ln2.data(), a.rstd2.data(), x, P(o.ln2), N, C);
            ffn_forward(l, out_tmp.data(), N, false);
            add_into(x, out_tmp.data(), long(N) * C);
        }
        rmsnorm_forward(lnf.data(), rstdf.data(), x, P(lnf_), N, C);
        linear_forward(logits.data(), lnf.data(), P(wte_), N, C, V);
        return logits.data();
    }

    // Re-quantize after the weights changed (the optimizer sets the flag).
    void quantize_weights() {
        for (const auto& p : infos) {
            if (!p.ternary) continue;
            const size_t per = p.size / p.count;
            for (int c = 0; c < p.count; ++c)
                ternary_quantize(&qparams[p.offset + c * per], &params[p.offset + c * per], long(per));
        }
        weights_dirty = false;
    }

private:
    struct LayerOffsets {
        size_t ln1, wqkv, wo, ln2, wfc, wproj;  // wfc/wproj: dense FFN or shared experts
        size_t router, efc, eproj;              // MoE
    };

    struct LayerActs {
        std::vector<T> ln1, rstd1, qkv, att, atty, resid1, ln2, rstd2, resid2, fc, sw;
        // MoE: routing and the per-assignment rows, grouped by expert.
        std::vector<T> rlogits, gate, xg, efc, esw, ey;
        std::vector<int> sel, order, row_of, expert_start;
    };

    size_t add_param(const std::string& name, int rows, int cols, bool decay, bool ternary,
                     int count = 1) {
        const size_t off = n_params_;
        const size_t size = size_t(rows) * cols * count;
        infos.push_back({name, off, size, rows, cols, decay, ternary, count});
        n_params_ += size;
        return off;
    }

    void build_layout() {
        const int C = cfg.d_model, V = cfg.vocab, F = cfg.always_hidden(), Fe = cfg.ehidden();
        const bool q = cfg.ternary;
        wte_ = add_param("wte", V, C, true, false);
        layer_.resize(cfg.n_layers);
        for (int l = 0; l < cfg.n_layers; ++l) {
            const std::string p = "l" + std::to_string(l) + ".";
            auto& o = layer_[l];
            o.ln1 = add_param(p + "ln1", 1, C, false, false);
            o.wqkv = add_param(p + "wqkv", 3 * C, C, true, q);
            o.wo = add_param(p + "wo", C, C, true, q);
            o.ln2 = add_param(p + "ln2", 1, C, false, false);
            if (F > 0) {
                o.wfc = add_param(p + "wfc", 2 * F, C, true, q);
                o.wproj = add_param(p + "wproj", C, F, true, q);
            }
            if (cfg.moe()) {
                o.router = add_param(p + "router", cfg.n_experts, C, false, false);
                o.efc = add_param(p + "efc", 2 * Fe, C, true, q, cfg.n_experts);
                o.eproj = add_param(p + "eproj", C, Fe, true, q, cfg.n_experts);
            }
        }
        lnf_ = add_param("lnf", 1, C, false, false);
    }

    T* P(size_t off) { return &params[off]; }
    T* G(size_t off) { return &grads[off]; }
    // The weights a projection multiplies by: quantized copies if ternary.
    const T* W(size_t off) const { return cfg.ternary ? &qparams[off] : &params[off]; }

    static void zero(std::vector<T>& v, size_t n) { zero_fill(v.data(), n); }

    // y = x · Wᵀ, or for BitLinear y = q8(x) · ternary(W)ᵀ.
    void proj_forward(T* y, const T* x, size_t woff, int N, int IC, int OC) {
        if (cfg.ternary) {
            act_quantize(xq.data(), x, N, IC);
            x = xq.data();
        }
        linear_forward(y, x, W(woff), N, IC, OC);
    }

    // dx += dy · W, dW += dyᵀ · x. For BitLinear, the straight-through
    // estimator: the same products with the quantized x and W (x is
    // re-quantized here instead of being stored), with gradients landing on
    // the full-precision master weights and on x unchanged.
    void proj_backward(T* dx, const T* dy, const T* x, size_t woff, int N, int IC, int OC) {
        if (cfg.ternary) {
            act_quantize(xq.data(), x, N, IC);
            x = xq.data();
        }
        linear_backward(dx, G(woff), dy, x, W(woff), N, IC, OC);
    }

    // out[N, C] = FFN(ln2) for layer l.
    void ffn_forward(int l, T* out, int N, bool training) {
        auto& a = acts[l];
        const auto& o = layer_[l];
        const int C = cfg.d_model, F = cfg.always_hidden();
        if (F > 0) {  // dense FFN, or the shared experts
            proj_forward(a.fc.data(), a.ln2.data(), o.wfc, N, C, 2 * F);
            swiglu_forward(a.sw.data(), a.fc.data(), N, F);
            proj_forward(out, a.sw.data(), o.wproj, N, F, C);
        } else {
            std::fill(out, out + size_t(N) * C, T(0));
        }
        if (cfg.moe()) moe_forward(l, out, N, training);
    }

    void ffn_backward(int l, T* dx, const T* dout, int N) {
        auto& a = acts[l];
        const auto& o = layer_[l];
        const int C = cfg.d_model, F = cfg.always_hidden();
        if (F > 0) {
            zero(dsw, size_t(N) * F);
            proj_backward(dsw.data(), dout, a.sw.data(), o.wproj, N, F, C);
            zero(dfc, size_t(N) * 2 * F);
            swiglu_backward(dfc.data(), dsw.data(), a.fc.data(), N, F);
            proj_backward(dx, dfc.data(), a.ln2.data(), o.wfc, N, C, 2 * F);
        }
        if (cfg.moe()) moe_backward(l, dx, dout, N);
    }

    // Routed experts: out += Σ_i gate_i * expert_{sel_i}(x) over the top_k
    // experts chosen for each token.
    void moe_forward(int l, T* out, int N, bool training) {
        auto& a = acts[l];
        const auto& o = layer_[l];
        PROFILE("moe_fwd_total");
        const int C = cfg.d_model, E = cfg.n_experts, K = cfg.top_k, Fe = cfg.ehidden();
        const T* x = a.ln2.data();
        linear_forward(a.rlogits.data(), x, P(o.router), N, C, E);
        // Top-k by biased logit; gates are a softmax over the chosen experts'
        // unbiased logits.
        const T* bias = router_bias[l].data();
        std::vector<int>& cnt = count_scratch_;
        cnt.assign(E, 0);
        for (int n = 0; n < N; ++n) {
            const T* r = &a.rlogits[size_t(n) * E];
            int* s = &a.sel[size_t(n) * K];
            for (int i = 0; i < K; ++i) {
                int best = -1;
                for (int e = 0; e < E; ++e) {
                    bool taken = false;
                    for (int j = 0; j < i; ++j) taken |= s[j] == e;
                    if (!taken && (best < 0 || r[e] + bias[e] > r[best] + bias[best])) best = e;
                }
                s[i] = best;
                ++cnt[best];
            }
            T mx = r[s[0]];
            for (int i = 1; i < K; ++i) mx = std::max(mx, r[s[i]]);
            T sum = 0;
            T* g = &a.gate[size_t(n) * K];
            for (int i = 0; i < K; ++i) sum += g[i] = std::exp(r[s[i]] - mx);
            for (int i = 0; i < K; ++i) g[i] /= sum;
        }
        if (training)
            for (int e = 0; e < E; ++e) expert_load[l][e] += cnt[e];
        // Group the N*K assignments by expert (a counting sort).
        a.expert_start[0] = 0;
        for (int e = 0; e < E; ++e) a.expert_start[e + 1] = a.expert_start[e] + cnt[e];
        std::vector<int>& fill = fill_scratch_;
        fill.assign(a.expert_start.begin(), a.expert_start.end() - 1);
        for (int j = 0; j < N * K; ++j) {
            const int row = fill[a.sel[j]]++;
            a.order[row] = j;     // row -> assignment (token j / K, slot j % K)
            a.row_of[j] = row;
        }
        // Gather each assignment's input row.
        for_rows(N * K, [&](int r0, int r1) {
            for (int r = r0; r < r1; ++r) {
                const T* src = x + size_t(a.order[r] / K) * C;
                std::copy(src, src + C, &a.xg[size_t(r) * C]);
            }
        });
        const size_t efc_per = size_t(2) * Fe * C, eproj_per = size_t(C) * Fe;
        for (int e = 0; e < E; ++e) {
            const int r0 = a.expert_start[e], n = a.expert_start[e + 1] - r0;
            if (n == 0) continue;
            proj_forward(&a.efc[size_t(r0) * 2 * Fe], &a.xg[size_t(r0) * C], o.efc + e * efc_per, n, C, 2 * Fe);
            swiglu_forward(&a.esw[size_t(r0) * Fe], &a.efc[size_t(r0) * 2 * Fe], n, Fe);
            proj_forward(&a.ey[size_t(r0) * C], &a.esw[size_t(r0) * Fe], o.eproj + e * eproj_per, n, Fe, C);
        }
        // Weighted sum back into token order.
        for_rows(N, [&](int n0, int n1) {
            for (int n = n0; n < n1; ++n) {
                T* dst = out + size_t(n) * C;
                for (int i = 0; i < K; ++i) {
                    const T g = a.gate[size_t(n) * K + i];
                    const T* y = &a.ey[size_t(a.row_of[size_t(n) * K + i]) * C];
                    for (int c = 0; c < C; ++c) dst[c] += g * y[c];
                }
            }
        });
    }

    void moe_backward(int l, T* dx, const T* dout, int N) {
        auto& a = acts[l];
        const auto& o = layer_[l];
        const int C = cfg.d_model, E = cfg.n_experts, K = cfg.top_k, Fe = cfg.ehidden();
        // Through the weighted sum: d ey = gate * dout; d gate = <dout, ey>.
        std::vector<T>& dgate = dgate_scratch_;
        dgate.assign(size_t(N) * K, T(0));
        for_rows(N, [&](int n0, int n1) {
            for (int n = n0; n < n1; ++n) {
                const T* d = dout + size_t(n) * C;
                for (int i = 0; i < K; ++i) {
                    const size_t j = size_t(n) * K + i, row = size_t(a.row_of[j]);
                    const T g = a.gate[j];
                    T* dy = &dey[row * C];
                    for (int c = 0; c < C; ++c) dy[c] = g * d[c];
                    dgate[j] = dot(d, &a.ey[row * C], C);
                }
            }
        });
        // Each expert, on its own rows.
        zero(dxg, size_t(N) * K * C);
        const size_t efc_per = size_t(2) * Fe * C, eproj_per = size_t(C) * Fe;
        for (int e = 0; e < E; ++e) {
            const int r0 = a.expert_start[e], n = a.expert_start[e + 1] - r0;
            if (n == 0) continue;
            zero(dsw, size_t(n) * Fe);
            proj_backward(dsw.data(), &dey[size_t(r0) * C], &a.esw[size_t(r0) * Fe],
                          o.eproj + e * eproj_per, n, Fe, C);
            zero(dfc, size_t(n) * 2 * Fe);
            swiglu_backward(dfc.data(), dsw.data(), &a.efc[size_t(r0) * 2 * Fe], n, Fe);
            proj_backward(&dxg[size_t(r0) * C], dfc.data(), &a.xg[size_t(r0) * C],
                          o.efc + e * efc_per, n, C, 2 * Fe);
        }
        // Scatter the gathered input gradients back to their tokens.
        for_rows(N, [&](int n0, int n1) {
            for (int n = n0; n < n1; ++n)
                for (int i = 0; i < K; ++i)
                    add_into(dx + size_t(n) * C, &dxg[size_t(a.row_of[size_t(n) * K + i]) * C], C);
        });
        // Router: softmax over the chosen logits, d r_s = g_s (dg_s - Σ g dg).
        zero(drl, size_t(N) * E);
        for (int n = 0; n < N; ++n) {
            const T* g = &a.gate[size_t(n) * K];
            const T* dg = &dgate[size_t(n) * K];
            T s = 0;
            for (int i = 0; i < K; ++i) s += g[i] * dg[i];
            for (int i = 0; i < K; ++i) drl[size_t(n) * E + a.sel[size_t(n) * K + i]] = g[i] * (dg[i] - s);
        }
        linear_backward(dx, G(o.router), drl.data(), a.ln2.data(), P(o.router), N, C, E);
    }

    size_t n_params_ = 0;
    size_t wte_ = 0, lnf_ = 0;
    std::vector<LayerOffsets> layer_;
    RopeTable<T> rope;

    int B_ = 0, T_ = 0;          // allocated
    int B_cur = 0, T_cur = 0;    // last forward
    bool has_targets_ = false;
    int n_valid_ = 0;
    std::vector<int> tokens_, targets_;
    std::vector<LayerActs> acts;
    std::vector<T> x0, lnf, rstdf, logits, losses, xq, out_tmp;
    std::vector<T> dresid, dC, dqkv, datty, dfc, dsw, dey, dxg, drl, dffn;
    std::vector<int> count_scratch_, fill_scratch_;
    std::vector<T> dgate_scratch_;
    int kv_B_ = 0;
    std::vector<int> kv_start_;
    std::vector<std::vector<T>> kcache_, vcache_;  // [layer][B, seq_len, C]
};
