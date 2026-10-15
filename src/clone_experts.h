// clone_experts.h — repair saved MoE experts without changing training.
#pragma once

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "load_log.h"
#include "train.h"

struct ExpertClone {
    int layer, from, to;
};

// Each layer's expert shares, averaged over its last `window` rows of a load
// log (`train --load-log`, night.sh's runs/$RUN.load.csv). The router biases
// cannot say whether an expert is dead: they only record past nudges, and
// they differ in every layer, healthy or not. The log counts tokens.
// After a "# resumed at step N" line, the steps past N were trained again,
// so the earlier rows past step N are dropped first (docs/LAPTOP.md).
inline std::vector<std::vector<double>> recent_expert_shares(const std::string& path, const Config& cfg,
                                                             int window) {
    CHECK(cfg.moe(), "clone-experts needs an MoE checkpoint");
    CHECK(window > 0, "--window must be positive");
    std::ifstream in(path);
    CHECK(in, "cannot open load log %s", path.c_str());
    std::string line;
    const auto next_line = [&] {
        if (!std::getline(in, line)) return false;
        if (!line.empty() && line.back() == '\r') line.pop_back();  // a file copied from Windows
        return true;
    };
    CHECK(next_line(), "%s is empty", path.c_str());
    CHECK(line == ExpertLoadLog::header(cfg.n_experts),
          "%s's header does not match this checkpoint's %d experts", path.c_str(), cfg.n_experts);
    struct Row { long step; int layer; std::vector<double> shares; };
    std::vector<Row> rows;
    for (int number = 2; next_line(); ++number) {
        if (line.empty()) continue;
        const char* p = line.c_str();
        char* end = nullptr;
        errno = 0;
        if (line.rfind("# resumed at step ", 0) == 0) {
            p += 18;
            const long resumed = std::strtol(p, &end, 10);
            CHECK(errno == 0 && end != p && *end == 0, "%s line %d: bad resume line", path.c_str(), number);
            std::erase_if(rows, [&](const Row& r) { return r.step > resumed; });
            continue;
        }
        Row row;
        row.step = std::strtol(p, &end, 10);
        CHECK(errno == 0 && end != p && *end == ',', "%s line %d: bad step", path.c_str(), number);
        p = end + 1;
        const long layer = std::strtol(p, &end, 10);
        CHECK(errno == 0 && end != p && *end == ',' && layer >= 0 && layer < cfg.n_layers,
              "%s line %d: bad layer for this checkpoint", path.c_str(), number);
        row.layer = int(layer);
        for (int e = 0; e <= cfg.n_experts; ++e) {  // the shares, then max_load_ratio
            p = end + 1;
            const double value = std::strtod(p, &end);
            CHECK(errno == 0 && end != p && *end == (e < cfg.n_experts ? ',' : '\0') && std::isfinite(value),
                  "%s line %d: expected %d shares and a ratio", path.c_str(), number, cfg.n_experts);
            if (e < cfg.n_experts) {
                CHECK(value >= 0 && value <= 1, "%s line %d: a share outside 0-1", path.c_str(), number);
                row.shares.push_back(value);
            }
        }
        rows.push_back(std::move(row));
    }
    std::vector<std::vector<double>> shares(cfg.n_layers, std::vector<double>(cfg.n_experts, 0.0));
    for (int l = 0; l < cfg.n_layers; ++l) {
        int used = 0;
        for (auto r = rows.rbegin(); r != rows.rend() && used < window; ++r) {
            if (r->layer != l) continue;
            for (int e = 0; e < cfg.n_experts; ++e) shares[l][e] += r->shares[e];
            ++used;
        }
        CHECK(used > 0, "%s has no rows for layer %d", path.c_str(), l);
        for (double& s : shares[l]) s /= used;
    }
    return shares;
}

// In each layer, every expert with a share under `below` (a fraction: 0.01
// is 1%, as in the load log's warning line) gets a copy of the layer's
// busiest expert (on a tie, the smaller index). Exactly `below` is not
// starved, as in the warning. A layer with no starved expert is left alone.
inline std::vector<ExpertClone> starved_expert_clones(const std::vector<std::vector<double>>& shares,
                                                      double below) {
    std::vector<ExpertClone> edits;
    for (int l = 0; l < int(shares.size()); ++l) {
        const auto& s = shares[l];
        const int from = int(std::max_element(s.begin(), s.end()) - s.begin());
        for (int e = 0; e < int(s.size()); ++e)
            if (s[e] < below && e != from) edits.push_back({l, from, e});
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
