// load_log.h — expert assignments accumulated between training log lines.
#pragma once

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "model.h"

class ExpertLoadLog {
    std::vector<std::vector<double>> counts_;

public:
    explicit ExpertLoadLog(const Config& cfg)
        : counts_(cfg.moe() ? cfg.n_layers : 0, std::vector<double>(cfg.n_experts, 0.0)) {}

    static std::string header(int experts) {
        std::string result = "step,layer";
        for (int e = 0; e < experts; ++e) result += ",expert_" + std::to_string(e);
        return result + ",max_load_ratio";
    }

    void add(const std::vector<std::vector<double>>& loads) {
        for (size_t l = 0; l < counts_.size(); ++l)
            for (size_t e = 0; e < counts_[l].size(); ++e) counts_[l][e] += loads[l][e];
    }

    // A token routed to K experts contributes K assignments. Dividing by
    // their total makes the expert shares sum to one, including for top-k.
    // Empty intervals (dense models or no successful steps) write no rows.
    // Experts under 1% of their layer's assignments go on one warning line
    // per interval, in layer order, e.g.
    //   under 1%: layer 1 expert 3 (0.0%), layer 2 expert 5 (0.4%)
    // After kMaxWarnings experts it only says how many more there are.
    static constexpr int kMaxWarnings = 12;
    void write(long step, FILE* csv, FILE* warnings) {
        std::string starving;  // the warning line's list of experts
        int n_starving = 0;
        for (size_t l = 0; l < counts_.size(); ++l) {
            auto& counts = counts_[l];
            const double total = std::accumulate(counts.begin(), counts.end(), 0.0);
            if (total == 0) continue;
            if (csv) {
                std::fprintf(csv, "%ld,%zu", step, l);
                for (double count : counts) std::fprintf(csv, ",%.17g", count / total);
                const double ratio = *std::max_element(counts.begin(), counts.end()) * counts.size() / total;
                std::fprintf(csv, ",%.17g\n", ratio);
            }
            for (size_t e = 0; e < counts.size(); ++e) {
                if (counts[e] * 100 >= total) continue;  // at least 1%
                ++n_starving;
                if (n_starving > kMaxWarnings) continue;
                char pair[64];
                std::snprintf(pair, sizeof(pair), "%slayer %zu expert %zu (%.1f%%)",
                              n_starving == 1 ? "" : ", ", l, e, counts[e] * 100 / total);
                starving += pair;
            }
            std::fill(counts.begin(), counts.end(), 0.0);
        }
        if (csv) CHECK(std::fflush(csv) == 0 && !std::ferror(csv), "cannot write expert load log");
        if (warnings && n_starving > 0) {
            if (n_starving > kMaxWarnings)
                starving += ", and " + std::to_string(n_starving - kMaxWarnings) + " more";
            std::fprintf(warnings, "under 1%%: %s\n", starving.c_str());
            std::fflush(warnings);
        }
    }
};

// Both backends use this after collecting their counts in Model::expert_load.
// Balancing clears those counts each step, so save them first. A skipped
// update passes no interval: its assignments are discarded as before.
inline double balance_with_load_log(Model<float>& model, double rate, ExpertLoadLog* interval) {
    const double load = model.max_load_ratio();
    if (interval) interval->add(model.expert_load);
    model.update_router_bias(rate);
    return load;
}
