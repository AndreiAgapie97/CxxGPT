// Regression tests for losses far from the small random gradient-check regime.
#pragma once

#include "../src/ops.h"
#include "test_util.h"
#include "extreme_loss_fixture.h"

template <class T>
void test_cross_entropy_extreme() {
    const int targets[] = {1, 0, -1};
    const std::vector<T> logits = {T(0), T(-1000), T(-2000),
                                   T(10000), T(10000), T(10000),
                                   T(0), T(-1000), T(-2000)};
    auto probs = logits;
    std::vector<T> losses(3), grad(9);
    const double loss = cross_entropy_forward(probs.data(), losses.data(), targets, 3, 3);
    const double ref = (1000.0 + std::log(3.0)) / 2;
    EXPECT(std::isfinite(loss) && std::abs(loss - ref) < 5e-5,
           "extreme cross-entropy %.12g vs log-sum-exp %.12g", loss, ref);
    EXPECT(losses[2] == T(0), "ignored loss is nonzero");
    cross_entropy_backward(grad.data(), probs.data(), targets, 3, 3, T(0.5));
    EXPECT(std::abs(double(grad[0]) - 0.5) < 1e-6 && grad[1] == T(-0.5),
           "confident wrong prediction has incorrect gradients");
    for (int n = 0; n < 3; ++n) {
        double sum = 0, dsum = 0;
        for (int v = 0; v < 3; ++v) {
            EXPECT(std::isfinite(probs[n * 3 + v]) && std::isfinite(grad[n * 3 + v]),
                   "non-finite probability or gradient");
            sum += probs[n * 3 + v];
            dsum += grad[n * 3 + v];
        }
        EXPECT(std::abs(sum - 1) < 1e-6 && std::abs(dsum) < 1e-6,
               "probability sum %g, gradient sum %g", sum, dsum);
    }

    // Perturb the losing logit while its probability remains underflowed.
    // The loss must still change, consistently with p - onehot backward.
    auto loss_at = [&](double delta) {
        auto p = logits;
        p[1] += T(delta);
        return cross_entropy_forward(p.data(), losses.data(), targets, 3, 3);
    };
    const double numeric = (loss_at(0.25) - loss_at(-0.25)) / 0.5;
    EXPECT(std::abs(numeric - double(grad[1])) < 1e-6,
           "saturated-loss finite difference %g vs analytic %g", numeric, double(grad[1]));

    int ignored[] = {-1, -1, -1};
    probs = logits;
    EXPECT(cross_entropy_forward(probs.data(), losses.data(), ignored, 3, 3) == 0,
           "all-ignored loss is nonzero");
    cross_entropy_backward(grad.data(), probs.data(), ignored, 3, 3, T(1));
    for (T g : grad) EXPECT(g == T(0), "all-ignored gradient is nonzero");
}

inline void test_numerics() {
    test_cross_entropy_extreme<float>();
    test_cross_entropy_extreme<double>();
    Rng rng(75);
    std::vector<double> x = {1}, gradient = {std::numeric_limits<double>::quiet_NaN()};
    EXPECT(std::isinf(grad_error(x, gradient, [&] { return x[0]; }, rng)),
           "gradient checker silently accepted NaN analytic gradient");
    gradient[0] = 1;
    EXPECT(std::isinf(grad_error(x, gradient, [] { return std::numeric_limits<double>::quiet_NaN(); }, rng)),
           "gradient checker silently accepted NaN loss");
    const float bad = std::numeric_limits<float>::quiet_NaN();
    EXPECT(std::isinf(max_abs_diff(&bad, x.data(), 1)), "comparison silently accepted NaN");
    ExtremeLossFixture f;
    auto& m = f.model;
    m.zero_grad();
    const double l = m.forward(f.tokens.data(), f.targets.data(), 1, 3);
    EXPECT(std::isfinite(l) && std::abs(l - f.reference_loss()) < 0.05,
           "whole-model extreme loss %g vs double reference %g", l, f.reference_loss());
    m.backward();
    for (float g : m.grads) EXPECT(std::isfinite(g), "whole-model extreme gradient non-finite");
    const auto& embedding = *m.find("wte");
    for (int i = 0; i < m.cfg.d_model; ++i)
        EXPECT(std::abs(m.grads[embedding.offset + m.cfg.d_model + i] + 0.5f) < 1e-5f,
               "whole-model underflowed target lost output-weight gradient");
}

// Sum the reward-weighted score-function gradient over all actions. It must
// equal the finite-difference gradient of -E[reward], independently of an
// action-independent baseline. This validates the sign and weighting used
// by the contextual-bandit experiment, without stochastic tolerances.
inline void test_policy_gradient_expectation() {
    Rng rng(74);
    std::vector<double> logits = {-4, -1, 2, 0};
    const std::vector<double> reward = {0, 0.3, 1, -0.2};
    auto expected_loss = [&] {
        const double mx = *std::max_element(logits.begin(), logits.end());
        double sum = 0, weighted = 0;
        for (size_t i = 0; i < logits.size(); ++i) {
            const double e = std::exp(logits[i] - mx);
            sum += e; weighted += e * reward[i];
        }
        return -weighted / sum;
    };
    for (double baseline : {-2.0, 0.0, 0.4, 10.0}) {
        std::vector<double> expected_grad(logits.size(), 0), grad(logits.size());
        for (int action = 0; action < int(logits.size()); ++action) {
            auto p = logits;
            double loss;
            cross_entropy_forward(p.data(), &loss, &action, 1, int(p.size()));
            cross_entropy_backward(grad.data(), p.data(), &action, 1, int(p.size()),
                                   reward[action] - baseline);
            for (size_t i = 0; i < grad.size(); ++i) expected_grad[i] += p[action] * grad[i];
        }
        const double error = grad_error(logits, expected_grad, expected_loss, rng);
        EXPECT(error < 1e-7, "policy gradient expectation with baseline %g: error %g", baseline, error);
    }
}
