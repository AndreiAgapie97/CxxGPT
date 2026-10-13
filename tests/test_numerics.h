// Regression tests for losses far from the small random gradient-check regime.
#pragma once

#include "../src/ops.h"
#include "test_util.h"

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
}
