// Independent router choices and CSV checks for interval load logging.
#pragma once

#include <sstream>

#include "../src/load_log.h"
#include "test_util.h"

inline Model<float> load_log_fixture() {
    Config cfg;
    cfg.vocab = 256; cfg.seq_len = 8; cfg.d_model = 8; cfg.n_heads = 2;
    cfg.n_layers = 2; cfg.n_experts = 4; cfg.top_k = 2; cfg.expert_hidden = 6;
    Model<float> model(cfg);
    model.init(37);
    // Each layer leaves the token embedding unchanged. Its router still
    // makes real, token-dependent choices, independently in each layer.
    for (const auto& p : model.infos)
        if (p.name.ends_with(".wqkv") || p.name.ends_with(".wo") || p.name.ends_with(".eproj"))
            std::fill_n(model.params.begin() + p.offset, p.size, 0.0f);
    model.router_bias[1][3] = -10;  // this expert cannot be selected
    return model;
}

// Recompute scores from embeddings and weights in double precision; do not
// read Model::expert_load, its selected experts, or its router-score buffer.
inline std::vector<std::vector<double>> reference_load_counts(const Model<float>& model,
                                                             const std::vector<int>& tokens) {
    const auto& cfg = model.cfg;
    const int C = cfg.d_model, E = cfg.n_experts;
    std::vector<std::vector<double>> result(cfg.n_layers, std::vector<double>(E, 0));
    const float* embedding = model.params.data() + model.find("wte")->offset;
    for (int l = 0; l < cfg.n_layers; ++l) {
        const std::string prefix = "l" + std::to_string(l);
        const float* norm = model.params.data() + model.find(prefix + ".ln2")->offset;
        const float* router = model.params.data() + model.find(prefix + ".router")->offset;
        for (int token : tokens) {
            const float* x = embedding + token * C;
            double squares = 0;
            for (int j = 0; j < C; ++j) squares += double(x[j]) * x[j];
            std::vector<double> scores(E);
            for (int e = 0; e < E; ++e) {
                scores[e] = model.router_bias[l][e];
                for (int j = 0; j < C; ++j)
                    scores[e] += double(x[j]) / std::sqrt(squares / C + 1e-5) * norm[j] * router[e * C + j];
            }
            std::vector<int> order(E);
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });
            for (int k = 0; k < cfg.top_k; ++k) ++result[l][order[k]];
        }
    }
    return result;
}

inline std::string load_log_text(FILE* file) {
    std::rewind(file);
    std::string text;
    char buffer[1024];
    while (std::fgets(buffer, sizeof(buffer), file)) text += buffer;
    return text;
}

inline void check_load_log_rows(std::istream& csv, long step,
                                const std::vector<std::vector<double>>& counts) {
    for (size_t l = 0; l < counts.size(); ++l) {
        std::string row;
        EXPECT(bool(std::getline(csv, row)), "missing layer %zu CSV row", l);
        std::replace(row.begin(), row.end(), ',', ' ');
        std::istringstream fields(row);
        long actual_step = -1;
        size_t actual_layer = counts.size();
        fields >> actual_step >> actual_layer;
        EXPECT(actual_step == step && actual_layer == l, "wrong step/layer in load log");
        double total = 0, maximum = 0;
        for (double count : counts[l]) { total += count; maximum = std::max(maximum, count); }
        double sum = 0;
        for (double count : counts[l]) {
            double share = -1;
            fields >> share;
            EXPECT(std::abs(share - count / total) < 1e-14, "logged share differs from independent router choices");
            sum += share;
        }
        double ratio = -1;
        fields >> ratio;
        EXPECT(std::abs(sum - 1) < 1e-14, "layer shares do not sum to one");
        EXPECT(std::abs(ratio - maximum / total * counts[l].size()) < 1e-14, "wrong layer load ratio");
        std::string extra;
        EXPECT(!bool(fields >> extra), "extra load log fields");
    }
}

inline void test_load_log() {
    auto model = load_log_fixture();
    ExpertLoadLog interval(model.cfg);
    FILE* csv = std::tmpfile();
    FILE* warnings = std::tmpfile();
    CHECK(csv && warnings, "cannot create load log test files");
    EXPECT(ExpertLoadLog::header(4) == "step,layer,expert_0,expert_1,expert_2,expert_3,max_load_ratio",
           "unexpected CSV header");
    std::vector<int> all;
    // Different batch sizes distinguish assignment totals from means of
    // per-step shares, and two forwards model gradient accumulation.
    for (const std::vector<int>& tokens : {std::vector<int>{1, 2, 3, 4, 5, 6}, {7, 8, 9, 10, 11}}) {
        model.forward(tokens.data(), nullptr, 1, int(tokens.size()), true);
        all.insert(all.end(), tokens.begin(), tokens.end());
    }
    balance_with_load_log(model, 0, &interval);
    const int ignored = 200;
    model.forward(&ignored, nullptr, 1, 1, true);
    balance_with_load_log(model, 0, nullptr);  // skipped update must not leak
    model.forward(&ignored, nullptr, 1, 1, false);  // evaluation must not leak
    const std::vector<int> next{20, 21, 22};
    model.forward(next.data(), nullptr, 1, int(next.size()), true);
    balance_with_load_log(model, 0, &interval);
    all.insert(all.end(), next.begin(), next.end());
    interval.write(2, csv, warnings);
    std::istringstream rows(load_log_text(csv));
    check_load_log_rows(rows, 2, reference_load_counts(model, all));
    EXPECT(load_log_text(warnings).find("layer 1: expert 3 got 0.0% of tokens") != std::string::npos,
           "dead expert did not trigger a warning");
    for (const auto& layer : model.expert_load)
        for (double count : layer) EXPECT(count == 0, "logging changed the per-step load reset");
    const long previous_bytes = std::ftell(csv);
    interval.write(2, csv, nullptr);
    EXPECT(std::ftell(csv) == previous_bytes, "empty interval wrote extra rows");
    model.forward(next.data(), nullptr, 1, int(next.size()), true);
    balance_with_load_log(model, 0, &interval);
    interval.write(3, csv, nullptr);
    std::istringstream updated(load_log_text(csv));
    check_load_log_rows(updated, 2, reference_load_counts(model, all));
    check_load_log_rows(updated, 3, reference_load_counts(model, next));
    std::fclose(csv);
    std::fclose(warnings);

    Config cfg = model.cfg;
    cfg.n_layers = 1; cfg.n_experts = 2;
    ExpertLoadLog threshold(cfg);
    warnings = std::tmpfile();
    CHECK(warnings, "cannot create warning test file");
    threshold.add({{1, 99}});
    threshold.write(1, nullptr, warnings);
    EXPECT(load_log_text(warnings).empty(), "exactly 1%% triggered a warning");
    threshold.add({{2, 998}});
    threshold.write(2, nullptr, warnings);
    EXPECT(load_log_text(warnings) == "layer 0: expert 0 got 0.2% of tokens\n", "wrong fractional warning");
    std::fclose(warnings);
}
