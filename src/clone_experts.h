// clone_experts.h — repair saved MoE experts without changing training.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "train.h"

struct ExpertClone {
    int layer, from, to;
};

// Bias records the balancing nudges, not actual token counts. With only
// that history available, repair the largest-bias expert(s) from the
// smallest-bias one. Equal biases give no evidence for a repair.
inline std::vector<ExpertClone> automatic_expert_clones(const Model<float>& model) {
    CHECK(model.cfg.moe(), "clone-experts needs an MoE checkpoint");
    std::vector<ExpertClone> edits;
    for (int l = 0; l < model.cfg.n_layers; ++l) {
        const auto& bias = model.router_bias[l];
        for (float b : bias) CHECK(std::isfinite(b), "non-finite router bias");
        const auto [low, high] = std::minmax_element(bias.begin(), bias.end());
        if (*low == *high) continue;
        const int from = int(low - bias.begin());
        for (int e = 0; e < model.cfg.n_experts; ++e)
            if (bias[e] == *high) edits.push_back({l, from, e});
    }
    return edits;
}

inline void clone_expert(Model<float>& model, AdamW& opt, const ExpertClone& edit,
                         double noise, Rng& rng, float mean_bias) {
    CHECK(model.cfg.moe(), "clone-experts needs an MoE checkpoint");
    CHECK(edit.layer >= 0 && edit.layer < model.cfg.n_layers, "layer index out of range");
    CHECK(edit.from >= 0 && edit.from < model.cfg.n_experts &&
          edit.to >= 0 && edit.to < model.cfg.n_experts && edit.from != edit.to,
          "expert indices must be in range and different");
    CHECK(std::isfinite(noise) && noise >= 0, "--noise must be finite and nonnegative");
    CHECK(opt.m.empty() ? opt.v.empty() : opt.m.size() == model.params.size() && opt.v.size() == opt.m.size(),
          "invalid checkpoint optimizer size");
    const std::string prefix = "l" + std::to_string(edit.layer) + ".";
    for (const char* name : {"efc", "eproj", "router"}) {
        const auto found = std::find_if(model.infos.begin(), model.infos.end(),
            [&](const ParamInfo& p) { return p.name == prefix + name; });
        CHECK(found != model.infos.end(), "missing expert parameter %s%s", prefix.c_str(), name);
        const size_t size = found->size / size_t(model.cfg.n_experts);
        const size_t from = found->offset + size_t(edit.from) * size;
        const size_t to = found->offset + size_t(edit.to) * size;
        double square = 0;
        for (size_t i = 0; i < size; ++i) {
            const double w = model.params[from + i];
            CHECK(std::isfinite(w), "non-finite source expert weight");
            square += w * w;
        }
        // Relative size means standard deviation / source slice RMS.
        // A zero slice stays zero; --noise 0 is a bit-for-bit copy.
        const double sigma = noise * std::sqrt(square / double(size));
        CHECK(std::isfinite(sigma), "--noise is too large for this expert");
        for (size_t i = 0; i < size; ++i) {
            const float w = model.params[from + i];
            const float value = sigma == 0 ? w : float(double(w) + sigma * rng.normal());
            CHECK(std::isfinite(value), "noise produced a non-finite weight");
            model.params[to + i] = value;
        }
        if (!opt.m.empty()) {
            for (size_t offset : {from, to}) {
                std::fill_n(opt.m.begin() + offset, size, 0.0f);
                std::fill_n(opt.v.begin() + offset, size, 0.0f);
            }
        }
    }
    model.router_bias[edit.layer][edit.from] = mean_bias;
    model.router_bias[edit.layer][edit.to] = mean_bias;
    model.weights_dirty = true;
}

inline void repair_experts(Model<float>& model, AdamW& opt, const std::vector<ExpertClone>& edits,
                           double noise, uint64_t seed, bool report = false) {
    CHECK(std::isfinite(noise) && noise >= 0, "--noise must be finite and nonnegative");
    // Compute all means before any edits, so multiple targets in a layer
    // receive the same original mean rather than a moving one.
    std::vector<float> means;
    for (const auto& bias : model.router_bias) {
        double sum = 0;
        for (float b : bias) {
            CHECK(std::isfinite(b), "non-finite router bias");
            sum += b;
        }
        means.push_back(float(sum / double(bias.size())));
    }
    Rng rng(seed);  // deliberately separate from the saved data sampler
    for (const auto& edit : edits) {
        CHECK(edit.layer >= 0 && edit.layer < model.cfg.n_layers, "layer index out of range");
        clone_expert(model, opt, edit, noise, rng, means[edit.layer]);
        if (report) std::printf("layer %d: expert %d -> expert %d; noise %g; both biases reset to %g\n",
                                edit.layer, edit.from, edit.to, noise, double(means[edit.layer]));
    }
}
