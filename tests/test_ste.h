// Validate the chosen identity STE without pretending rounding is smooth.
#pragma once

#include "../src/ops.h"
#include "test_util.h"

inline void test_bitlinear_ste() {
    Rng rng(73);
    constexpr int N = 5, IC = 7, OC = 6;
    auto x = random_vec<double>(N * IC, rng), w = random_vec<double>(OC * IC, rng);
    auto dy = random_vec<double>(N * OC, rng);
    // Include a zero activation row, which must produce zero output and dW.
    std::fill(x.begin(), x.begin() + IC, 0.0);
    const auto x0 = x, w0 = w;
    std::vector<double> xq(x.size()), wq(w.size()), scratch(x.size()), y(N * OC);
    act_quantize(xq.data(), x.data(), N, IC);
    ternary_quantize(wq.data(), w.data(), long(w.size()));
    bitlinear_forward(y.data(), x.data(), wq.data(), scratch.data(), N, IC, OC);
    for (int n = 0; n < N; ++n)
        for (int o = 0; o < OC; ++o) {
            double ref = 0;
            for (int i = 0; i < IC; ++i) ref += xq[n * IC + i] * wq[o * IC + i];
            EXPECT(std::abs(y[n * OC + o] - ref) < 1e-12, "BitLinear forward reference mismatch");
        }
    std::vector<double> dx(x.size(), 0), dw(w.size(), 0);
    bitlinear_backward(dx.data(), dw.data(), dy.data(), x.data(), wq.data(), scratch.data(), N, IC, OC);

    // Surrogate Q_x(x) = xq_at_base + (x-x_base), and likewise for W.
    // Its forward value equals the quantized layer at the base point, but
    // its Jacobians for quantizers are exactly I, as prescribed by the STE.
    auto surrogate_loss = [&] {
        double loss = 0;
        for (int n = 0; n < N; ++n)
            for (int o = 0; o < OC; ++o)
                for (int i = 0; i < IC; ++i) {
                    const int xi = n * IC + i, wi = o * IC + i;
                    loss += dy[n * OC + o] * (xq[xi] + x[xi] - x0[xi]) *
                                              (wq[wi] + w[wi] - w0[wi]);
                }
        return loss;
    };
    const double ex = grad_error(x, dx, surrogate_loss, rng);
    const double ew = grad_error(w, dw, surrogate_loss, rng);
    EXPECT(ex < 1e-7 && ew < 1e-7, "STE surrogate gradients: dx %g, dw %g", ex, ew);
    const auto dx_once = dx, dw_once = dw;
    bitlinear_backward(dx.data(), dw.data(), dy.data(), x.data(), wq.data(), scratch.data(), N, IC, OC);
    for (size_t i = 0; i < dx.size(); ++i)
        EXPECT(std::abs(dx[i] - 2 * dx_once[i]) < 1e-12, "STE did not accumulate dx");
    for (size_t i = 0; i < dw.size(); ++i)
        EXPECT(std::abs(dw[i] - 2 * dw_once[i]) < 1e-12, "STE did not accumulate master-weight gradient");
    std::printf("  BitLinear identity STE: surrogate errors dx %.1e, dw %.1e\n", ex, ew);
}
