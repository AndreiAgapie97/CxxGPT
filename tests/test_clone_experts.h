// Independently reconstruct every edited checkpoint value, including noise.
#pragma once

#include "../src/clone_experts.h"
#include "test_util.h"

inline Config clone_test_config() {
    Config c;
    c.vocab = 13; c.seq_len = 4; c.d_model = 8; c.n_heads = 2; c.n_layers = 2;
    c.n_experts = 4; c.top_k = 1; c.expert_hidden = 6; c.n_shared = 1;
    return c;
}

inline void clone_test_fixture(Model<float>& model, AdamW& opt) {
    model.init(17);
    model.router_bias[0] = {-1, 0.25f, 2, -0.25f};
    model.router_bias[1] = {-2, 1, 0, 1};
    opt.step = 7;
    opt.m.resize(model.num_params()); opt.v.resize(model.num_params());
    for (size_t i = 0; i < model.num_params(); ++i) {
        opt.m[i] = float(i % 17 + 1) / 32;
        opt.v[i] = float(i % 11 + 1) / 64;
    }
}

inline bool clone_same_bits(float a, float b) {
    return std::memcmp(&a, &b, sizeof(float)) == 0;
}

inline void clone_reference(const Model<float>& before, const AdamW& old,
                            const Model<float>& after, const AdamW& now,
                            const std::vector<ExpertClone>& edits, double noise, uint64_t seed) {
    auto parameters = before.params, m = old.m, v = old.v;
    auto biases = before.router_bias;
    Rng random(seed);
    for (const auto& edit : edits) {
        // Derive the rows from tensor dimensions, independently of the
        // production slice size calculation and its copying loop.
        for (const char* suffix : {"efc", "eproj", "router"}) {
            const std::string name = "l" + std::to_string(edit.layer) + "." + suffix;
            for (const auto& info : before.infos) {
                if (info.name != name) continue;
                const size_t rows = std::string(suffix) == "router" ? 1 : size_t(info.rows);
                std::vector<float> source;
                for (size_t row = 0; row < rows; ++row)
                    for (int col = 0; col < info.cols; ++col)
                        source.push_back(before.params[info.offset + (size_t(edit.from) * rows + row) * info.cols + col]);
                long double squares = 0;
                for (float x : source) squares += static_cast<long double>(x) * x;
                const long double sigma = noise * std::sqrt(squares / source.size());
                for (size_t row = 0; row < rows; ++row) {
                    for (int col = 0; col < info.cols; ++col) {
                        const size_t index = row * info.cols + col;
                        long double delta = 0;
                        if (sigma != 0) {
                            // Independent Box-Muller expression from the
                            // random uniforms; never call Rng::normal here.
                            double u = random.uniform();
                            while (u == 0) u = random.uniform();
                            const long double angle = 6.283185307179586L * random.uniform();
                            delta = sigma * std::sqrt(-2.0L * std::log(static_cast<long double>(u))) * std::cos(angle);
                        }
                        const size_t dst = info.offset + (size_t(edit.to) * rows + row) * info.cols + col;
                        parameters[dst] = float(source[index] + delta);
                        if (!m.empty()) {
                            const size_t src = info.offset + (size_t(edit.from) * rows + row) * info.cols + col;
                            m[src] = m[dst] = v[src] = v[dst] = 0;
                        }
                    }
                }
            }
        }
        long double total = 0;
        for (float b : before.router_bias[edit.layer]) total += b;
        biases[edit.layer][edit.from] = biases[edit.layer][edit.to] = float(total / before.cfg.n_experts);
    }
    EXPECT(after.params.size() == parameters.size(), "parameter size changed");
    for (size_t i = 0; i < parameters.size(); ++i) {
        const bool untouched = clone_same_bits(parameters[i], before.params[i]);
        EXPECT(noise == 0 ? clone_same_bits(after.params[i], parameters[i]) :
               untouched ? clone_same_bits(after.params[i], before.params[i]) :
               std::abs(double(after.params[i]) - parameters[i]) < 2e-8,
               "copy/noise or unrelated parameter changed at %zu", i);
    }
    EXPECT(now.m.size() == m.size() && now.v.size() == v.size(), "optimizer arrays changed size");
    for (size_t i = 0; i < m.size(); ++i)
        EXPECT(clone_same_bits(now.m[i], m[i]) && clone_same_bits(now.v[i], v[i]),
               "optimizer changed outside the two expert slices at %zu", i);
    for (size_t l = 0; l < biases.size(); ++l)
        for (size_t e = 0; e < biases[l].size(); ++e)
            EXPECT(clone_same_bits(after.router_bias[l][e], biases[l][e]), "wrong reset bias at %zu/%zu", l, e);
    EXPECT(now.step == old.step && now.muon == old.muon, "optimizer step/kind changed");
}

inline void test_clone_experts() {
    for (int kind = 0; kind < 3; ++kind) {
        Model<float> original(clone_test_config()); AdamW old;
        clone_test_fixture(original, old);
        old.muon = kind == 2;
        if (kind == 0) { old.m.clear(); old.v.clear(); }
        Rng sampler(93); sampler.next();
        const std::string input = test_path("clone-in-" + std::to_string(kind) + ".ckpt");
        const std::string output = test_path("clone-out-" + std::to_string(kind) + ".ckpt");
        save_checkpoint(input, original, old, sampler);
        for (double noise : {0.0, 0.01}) {
            Model<float> model(clone_test_config()); AdamW opt; Rng loaded;
            if (kind == 0) { opt.muon = true; opt.m = {1}; opt.v = {2}; }
            load_checkpoint(input, model, &opt, &loaded, true);
            const std::vector<ExpertClone> edits = {{0, 0, 2}};
            repair_experts(model, opt, edits, noise, 31);
            clone_reference(original, old, model, opt, edits, noise, 31);
            save_checkpoint(output, model, opt, loaded);
            Model<float> restored(clone_test_config()); AdamW state; Rng rng;
            load_checkpoint(output, restored, &state, &rng, true);
            clone_reference(original, old, restored, state, edits, noise, 31);
            for (int i = 0; i < 4; ++i) EXPECT(rng.state()[i] == sampler.state()[i], "repair changed saved sampler RNG");
            Model<float> repeated = original; AdamW repeated_opt = old;
            repair_experts(repeated, repeated_opt, edits, noise, 31);
            EXPECT(repeated.params == model.params, "same seed did not produce identical noise");
        }
    }
    Model<float> model(clone_test_config()); AdamW opt;
    clone_test_fixture(model, opt);
    const auto edits = automatic_expert_clones(model);
    EXPECT(edits.size() == 3, "auto did not select the tied highest biases");
    if (edits.size() == 3) {
        EXPECT(edits[0].layer == 0 && edits[0].from == 0 && edits[0].to == 2 &&
               edits[1].layer == 1 && edits[1].from == 0 && edits[1].to == 1 &&
               edits[2].layer == 1 && edits[2].from == 0 && edits[2].to == 3,
               "auto chose wrong busy/starved expert");
        const auto before = model; const auto old = opt;
        repair_experts(model, opt, edits, 0.01, 13);
        clone_reference(before, old, model, opt, edits, 0.01, 13);
    }
    for (auto& bias : model.router_bias) std::fill(bias.begin(), bias.end(), 0.25f);
    EXPECT(automatic_expert_clones(model).empty(), "balanced layer was edited");
    model.router_bias[0] = {-1, -1, 2, 0};
    const auto tied = automatic_expert_clones(model);
    EXPECT(tied.size() == 1 && tied[0].from == 0 && tied[0].to == 2, "source ties must use lowest expert index");

    // Clone with no noise: the CPU's deterministic top-1 tie choice
    // breaks the equality during a real backward/optimizer step.
    Model<float> train(clone_test_config()); train.init(17);
    AdamW optimizer;
    for (auto& bias : train.router_bias) bias = {0, -100, 0, -100};
    repair_experts(train, optimizer, {{0, 0, 2}}, 0, 1);
    const std::string path = test_path("clone-step.ckpt"); Rng rng(4);
    save_checkpoint(path, train, optimizer, rng);
    Model<float> ready(clone_test_config());
    load_checkpoint(path, ready, &optimizer, &rng, true);
    const int tokens[] = {1, 2, 3, 4}, targets[] = {2, 3, 4, 5};
    ready.zero_grad();
    const float loss = ready.forward(tokens, targets, 1, 4, true);
    ready.backward(); optimizer.update(ready, 0.001);
    EXPECT(std::isfinite(loss), "repaired checkpoint failed CPU step");
    bool diverged = false;
    for (const auto& p : ready.infos) if (p.name == "l0.efc") {
        const size_t per = size_t(p.rows) * p.cols;
        for (size_t i = 0; i < per; ++i) diverged |= ready.params[p.offset + i] != ready.params[p.offset + 2 * per + i];
    }
    EXPECT(diverged, "cloned experts did not diverge after CPU step");
}
