// config.h — the model's shape (Config) and its parameter layout entries
// (ParamInfo). Kept free of the CPU kernels so the CUDA backend can include
// it.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <string>

#include "common.h"

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
    // Default expert size: the dense FFN split across the top_k experts a
    // token uses (same active compute), rounded up to a multiple of 32 as
    // the inference kernels' bitplanes need.
    int ehidden() const {
        return expert_hidden ? expert_hidden : std::max(32, (dense_hidden() / top_k + 31) / 32 * 32);
    }
    bool moe() const { return n_experts > 0; }
    // Hidden size of the always-on FFN part: the dense FFN, or the shared experts.
    int always_hidden() const { return moe() ? n_shared * ehidden() : dense_hidden(); }

    void validate() const {
        CHECK(vocab > 0 && seq_len > 0 && n_layers > 0, "bad sizes");
        // Validate divisors before head_size()/ehidden() or tensor allocation.
        CHECK(d_model > 0 && n_heads > 0, "d_model and n_heads must be positive");
        CHECK(ffn_hidden >= 0 && expert_hidden >= 0, "hidden sizes must be nonnegative (0 = auto)");
        CHECK(n_experts >= 0 && n_shared >= 0, "expert counts must be nonnegative");
        CHECK(top_k >= 1, "top_k must be positive");
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

// The matrices Muon updates: every weight matrix inside the blocks.
// Embeddings, norms and routers stay on AdamW.
inline bool muon_matrix(const ParamInfo& p) {
    return p.rows > 1 && p.cols > 1 && p.decay && p.name != "wte";
}

// Fused projections are orthogonalized part by part: [q | k | v] and [gate | up].
inline int muon_parts(const ParamInfo& p) {
    auto ends = [&](const char* s) {
        const size_t n = std::strlen(s);
        return p.name.size() >= n && p.name.compare(p.name.size() - n, n, s) == 0;
    };
    return ends("wqkv") ? 3 : ends("wfc") || ends("efc") ? 2 : 1;
}
