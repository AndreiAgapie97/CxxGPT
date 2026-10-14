// infer.h — a trained ternary model running on the inference kernels.
//
// TernaryEngine takes a Model<float> trained with ternary = true and packs
// every BitLinear weight into two bitplanes (1.58 bits of information, 2
// bits of storage per weight) with its absmean scale. Decoding then runs the
// int8 kernels from ternary.h: activations quantized once per projection
// input, integer dot products, one float multiply per output.
//
// The embedding, the output head, norms and the MoE router stay in float,
// as in training. One sequence at a time, with a KV cache.
#pragma once

#include <cmath>
#include <limits>
#include <vector>

#include "model.h"
#include "ternary.h"

class TernaryEngine {
public:
    explicit TernaryEngine(const Model<float>& m) : cfg_(m.cfg), kernel_(default_int8_kernel()) {
        CHECK(cfg_.ternary, "TernaryEngine needs a model trained with --ternary");
        CHECK(!cfg_.fp_attention, "TernaryEngine runs all-ternary models, not --fp-attn ones (yet)");
        CHECK(cfg_.top_k <= 64, "top_k above 64 is not supported");
        CHECK(cfg_.d_model % 32 == 0, "d_model must be a multiple of 32 for the bitplane kernels");
        CHECK(cfg_.always_hidden() % 32 == 0 && cfg_.ehidden() % 32 == 0,
              "FFN hidden sizes must be multiples of 32 for the bitplane kernels");
        const int C = cfg_.d_model, F = cfg_.always_hidden(), Fe = cfg_.ehidden();
        auto tensor = [&](const std::string& name) -> const ParamInfo& {
            const ParamInfo* p = m.find(name);
            CHECK(p, "missing tensor %s", name.c_str());
            return *p;
        };
        auto copy = [&](const ParamInfo& p) {
            return std::vector<float>(m.params.begin() + p.offset, m.params.begin() + p.offset + p.size);
        };
        wte_ = copy(tensor("wte"));
        lnf_ = copy(tensor("lnf"));
        layers_.resize(cfg_.n_layers);
        for (int l = 0; l < cfg_.n_layers; ++l) {
            const std::string p = "l" + std::to_string(l) + ".";
            auto& L = layers_[l];
            L.ln1 = copy(tensor(p + "ln1"));
            L.ln2 = copy(tensor(p + "ln2"));
            L.wqkv = pack(m, tensor(p + "wqkv"), 0);
            L.wo = pack(m, tensor(p + "wo"), 0);
            if (F > 0) {
                L.wfc = pack(m, tensor(p + "wfc"), 0);
                L.wproj = pack(m, tensor(p + "wproj"), 0);
            }
            if (cfg_.moe()) {
                L.router = copy(tensor(p + "router"));
                L.bias = m.router_bias[l];
                for (int e = 0; e < cfg_.n_experts; ++e) {
                    L.efc.push_back(pack(m, tensor(p + "efc"), e));
                    L.eproj.push_back(pack(m, tensor(p + "eproj"), e));
                }
            }
        }
        rope_ = RopeTable<float>(cfg_.seq_len, cfg_.head_size());
        const int S = cfg_.seq_len;
        kcache_.assign(cfg_.n_layers, std::vector<float>(size_t(S) * C));
        vcache_.assign(cfg_.n_layers, std::vector<float>(size_t(S) * C));
        x_.resize(C); h_.resize(C); qkv_.resize(3 * C); att_.resize(C); tmp_.resize(C);
        xq_.resize(std::max({C, F, Fe})); scores_.resize(S);
        fc_.resize(size_t(2) * std::max(F, 1) + size_t(cfg_.top_k) * 2 * Fe);
        sw_.resize(size_t(std::max(F, 1)) + size_t(cfg_.top_k) * Fe);
        swq_.resize(sw_.size());
        sw_scale_.resize(cfg_.top_k + 1);
        down_.resize(size_t(cfg_.top_k) * C);
        logits_.resize(cfg_.vocab);
        rlogits_.resize(std::max(1, cfg_.n_experts));
        Fe_ = Fe;
        F_ = F;
    }

    void reset() { pos_ = 0; }
    int position() const { return pos_; }

    // Bytes of weight storage: bitplanes for the ternary matrices, float for
    // the rest.
    size_t weight_bytes() const {
        size_t b = (wte_.size() + lnf_.size()) * 4;
        auto bp = [](const TernaryMatrixBitplane& W) { return (W.pos.size() + W.neg.size()) * 4 + 4; };
        for (const auto& L : layers_) {
            b += (L.ln1.size() + L.ln2.size() + L.router.size()) * 4 + bp(L.wqkv) + bp(L.wo);
            if (F_ > 0) b += bp(L.wfc) + bp(L.wproj);
            for (const auto& W : L.efc) b += bp(W);
            for (const auto& W : L.eproj) b += bp(W);
        }
        return b;
    }

    // Feeds one token at the next position; returns logits [vocab].
    const float* step(int token) {
        const int C = cfg_.d_model, NH = cfg_.n_heads, HS = C / NH, S = cfg_.seq_len, V = cfg_.vocab;
        CHECK(pos_ < S, "sequence longer than seq_len %d", S);
        CHECK(token >= 0 && token < V, "token %d out of range", token);
        std::copy(&wte_[size_t(token) * C], &wte_[size_t(token + 1) * C], x_.begin());
        const float scale = 1.0f / std::sqrt(float(HS));
        for (int l = 0; l < cfg_.n_layers; ++l) {
            auto& L = layers_[l];
            rmsnorm(h_.data(), x_.data(), L.ln1.data(), C);
            gemv(L.wqkv, h_.data(), qkv_.data());
            if (cfg_.rope_layer(l)) rope_apply(qkv_.data(), rope_, 1, 1, C, NH, false, pos_);
            float* kc = kcache_[l].data();
            float* vc = vcache_[l].data();
            std::copy(&qkv_[C], &qkv_[2 * C], kc + size_t(pos_) * C);
            std::copy(&qkv_[2 * C], &qkv_[3 * C], vc + size_t(pos_) * C);
            for (int h = 0; h < NH; ++h) {
                const float* q = &qkv_[h * HS];
                float mx = -std::numeric_limits<float>::infinity();
                for (int s = 0; s <= pos_; ++s) {
                    scores_[s] = dot(q, kc + size_t(s) * C + h * HS, HS) * scale;
                    mx = std::max(mx, scores_[s]);
                }
                float sum = 0;
                for (int s = 0; s <= pos_; ++s) sum += scores_[s] = fast_exp(scores_[s] - mx);
                float* y = &att_[h * HS];
                std::fill(y, y + HS, 0.0f);
                for (int s = 0; s <= pos_; ++s) {
                    const float w = scores_[s] / sum;
                    const float* v = vc + size_t(s) * C + h * HS;
                    for (int i = 0; i < HS; ++i) y[i] += w * v[i];
                }
            }
            gemv(L.wo, att_.data(), tmp_.data());
            for (int c = 0; c < C; ++c) x_[c] += tmp_[c];
            rmsnorm(h_.data(), x_.data(), L.ln2.data(), C);
            ffn(L, h_.data(), tmp_.data());
            for (int c = 0; c < C; ++c) x_[c] += tmp_[c];
        }
        rmsnorm(h_.data(), x_.data(), lnf_.data(), C);
        pool().parallel_for(V, [&](int v0, int v1) {
            for (int v = v0; v < v1; ++v) logits_[v] = dot(h_.data(), &wte_[size_t(v) * C], C);
        });
        ++pos_;
        return logits_.data();
    }

private:
    struct Layer {
        std::vector<float> ln1, ln2, router, bias;
        TernaryMatrixBitplane wqkv{32, 32}, wo{32, 32}, wfc{32, 32}, wproj{32, 32};
        std::vector<TernaryMatrixBitplane> efc, eproj;
    };

    // Copy `index` of a (possibly stacked) ternary tensor into bitplanes,
    // with the same absmean scale the training forward pass used.
    static TernaryMatrixBitplane pack(const Model<float>& m, const ParamInfo& p, int index) {
        const size_t per = p.size / p.count;
        const float* w = &m.params[p.offset + index * per];
        std::vector<float> wq(per);
        const float gamma = ternary_quantize(wq.data(), w, long(per));
        std::vector<int8_t> t(per);
        for (size_t i = 0; i < per; ++i) t[i] = int8_t(std::lround(wq[i] / gamma));
        TernaryMatrixBitplane W(p.rows, p.cols, gamma);
        pack_bitplane(t, W);
        return W;
    }

    static void rmsnorm(float* out, const float* x, const float* w, int C) {
        const float r = 1.0f / std::sqrt(dot(x, x, C) / float(C) + float(kRmsEps));
        for (int c = 0; c < C; ++c) out[c] = x[c] * r * w[c];
    }

    // y = W x with x quantized to int8 first.
    void gemv(const TernaryMatrixBitplane& W, const float* x, float* y) {
        const float s = quantize_int8(x, W.cols, xq_.data());
        pool().parallel_for(W.rows, [&](int r0, int r1) { kernel_.qrows(xq_.data(), s, W, y, r0, r1); });
    }

    // FFN for one token: shared/dense part plus routed experts, as two pool
    // jobs (all gate/up rows; then all down rows and the weighted sum).
    void ffn(const Layer& L, const float* h, float* out) {
        const int C = cfg_.d_model, F = F_, Fe = Fe_, K = cfg_.moe() ? cfg_.top_k : 0;
        // Routing: top-k by biased logit, softmax over the chosen logits.
        int sel[64];
        float gate[64];
        if (K > 0) {
            const int E = cfg_.n_experts;
            for (int e = 0; e < E; ++e) rlogits_[e] = dot(h, &L.router[size_t(e) * C], C);
            for (int i = 0; i < K; ++i) {
                int best = -1;
                for (int e = 0; e < E; ++e) {
                    bool taken = false;
                    for (int j = 0; j < i; ++j) taken |= sel[j] == e;
                    if (!taken && (best < 0 || rlogits_[e] + L.bias[e] > rlogits_[best] + L.bias[best])) best = e;
                }
                sel[i] = best;
            }
            float mx = rlogits_[sel[0]];
            for (int i = 1; i < K; ++i) mx = std::max(mx, rlogits_[sel[i]]);
            float sum = 0;
            for (int i = 0; i < K; ++i) sum += gate[i] = std::exp(rlogits_[sel[i]] - mx);
            for (int i = 0; i < K; ++i) gate[i] /= sum;
        }
        // Job 1: gate and up rows of the shared FFN (F) and of each routed
        // expert (Fe), all from the same quantized input.
        const float hs = quantize_int8(h, C, xq_.data());
        const int rows1 = 2 * F + K * 2 * Fe;
        pool().parallel_for(rows1, [&](int r0, int r1) {
            while (r0 < r1) {
                if (r0 < 2 * F) {
                    const int hi = std::min(r1, 2 * F);
                    kernel_.qrows(xq_.data(), hs, L.wfc, fc_.data(), r0, hi);
                    r0 = hi;
                } else {
                    const int i = (r0 - 2 * F) / (2 * Fe), lo = (r0 - 2 * F) % (2 * Fe);
                    const int hi = std::min(2 * Fe, lo + (r1 - r0));
                    kernel_.qrows(xq_.data(), hs, L.efc[sel[i]], &fc_[2 * F + size_t(i) * 2 * Fe], lo, hi);
                    r0 += hi - lo;
                }
            }
        });
        // SwiGLU per block, then quantize each block's hidden vector.
        auto swiglu = [&](const float* fc, int n, float* sw, int8_t* q) {
            for (int j = 0; j < n; ++j) sw[j] = fc[j] / (1.0f + fast_exp(-fc[j])) * fc[n + j];
            return quantize_int8(sw, n, q);
        };
        if (F > 0) sw_scale_[K] = swiglu(fc_.data(), F, sw_.data(), swq_.data());
        for (int i = 0; i < K; ++i)
            sw_scale_[i] = swiglu(&fc_[2 * F + size_t(i) * 2 * Fe], Fe, &sw_[F + size_t(i) * Fe],
                                  &swq_[F + size_t(i) * Fe]);
        // Job 2: every down projection, rows [r0, r1) of the output, summed.
        pool().parallel_for(C, [&](int r0, int r1) {
            if (F > 0) kernel_.qrows(swq_.data(), sw_scale_[K], L.wproj, out, r0, r1);
            else std::fill(out + r0, out + r1, 0.0f);
            for (int i = 0; i < K; ++i) {
                float* d = &down_[size_t(i) * C];  // rows [r0, r1) are this thread's
                kernel_.qrows(&swq_[F + size_t(i) * Fe], sw_scale_[i], L.eproj[sel[i]], d, r0, r1);
                for (int r = r0; r < r1; ++r) out[r] += gate[i] * d[r];
            }
        });
    }

    Config cfg_;
    Kernel kernel_;
    std::vector<Layer> layers_;
    std::vector<float> wte_, lnf_;
    RopeTable<float> rope_;
    std::vector<std::vector<float>> kcache_, vcache_;
    std::vector<float> x_, h_, qkv_, att_, tmp_, scores_, fc_, sw_, down_, logits_, rlogits_;
    std::vector<int8_t> xq_, swq_;
    std::vector<float> sw_scale_;
    int pos_ = 0, F_ = 0, Fe_ = 0;
};
