// Analytic whole-model reference shared by CPU and GPU regression tests.
#pragma once

#include <array>

#include "../src/model.h"

struct ExtremeLossFixture {
    static constexpr float amplitude = 1000;
    static Config config() {
        Config c;
        c.vocab = 3; c.seq_len = 3; c.d_model = 32;
        c.n_layers = 1; c.n_heads = 4; c.ffn_hidden = 64;
        return c;
    }
    Model<float> model{config()};
    std::array<int, 3> tokens{0, 0, 0}, targets{1, -1, 2};

    ExtremeLossFixture() {
        for (const auto& p : model.infos)
            if (p.rows == 1) std::fill_n(&model.params[p.offset], p.size, 1.0f);
        const auto& p = *model.find("wte");
        std::fill_n(&model.params[p.offset], model.cfg.d_model, amplitude);
        std::fill_n(&model.params[p.offset + model.cfg.d_model], model.cfg.d_model, -amplitude);
    }

    // Zero block projections preserve the input embedding. Final RMSNorm
    // gives logits [z, -z, 0], z = C*a^2/sqrt(a^2+eps). Counted losses
    // are approximately 2z and z (the exp(-z) terms underflow), so mean=1.5z.
    double reference_loss() const {
        const double a = amplitude;
        return 1.5 * model.cfg.d_model * a * a / std::sqrt(a * a + kRmsEps);
    }
};
