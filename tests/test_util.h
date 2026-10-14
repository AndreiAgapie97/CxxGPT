// test_util.h — a minimal test harness and gradient checker.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <filesystem>
#include <chrono>
#include <limits>
#include <vector>

#include "../src/common.h"

inline int g_failures = 0;

#define EXPECT(cond, ...)                                                   \
    do {                                                                    \
        if (!(cond)) {                                                      \
            ++g_failures;                                                   \
            std::printf("  FAIL %s:%d: %s\n    ", __FILE__, __LINE__, #cond); \
            std::printf(__VA_ARGS__);                                       \
            std::printf("\n");                                              \
        }                                                                   \
    } while (0)

template <class T>
std::vector<T> random_vec(size_t n, Rng& rng, double sd = 1.0) {
    std::vector<T> v(n);
    for (auto& x : v) x = T(rng.normal() * sd);
    return v;
}

// Compares an analytic gradient of a scalar loss with central finite
// differences, in double precision. Checks every element, or `max_checks`
// random ones for big tensors. Returns the largest error relative to the
// largest gradient magnitude, which is what a wrong formula shows up in.
inline double grad_error(std::vector<double>& x, const std::vector<double>& analytic,
                         const std::function<double()>& loss, Rng& rng, int max_checks = 400,
                         double eps = 1e-5) {
    // std::max can silently discard NaN operands. A numerical checker must
    // fail on non-finite values instead of reporting an error of zero.
    if (analytic.size() != x.size()) return std::numeric_limits<double>::infinity();
    std::vector<size_t> idx;
    if (x.size() <= size_t(max_checks)) {
        for (size_t i = 0; i < x.size(); ++i) idx.push_back(i);
    } else {
        for (int i = 0; i < max_checks; ++i) idx.push_back(rng.below(x.size()));
    }
    double max_diff = 0.0, max_grad = 1e-12;
    for (double g : analytic) {
        if (!std::isfinite(g)) return std::numeric_limits<double>::infinity();
        max_grad = std::max(max_grad, std::abs(g));
    }
    for (size_t i : idx) {
        const double saved = x[i];
        x[i] = saved + eps;
        const double lp = loss();
        x[i] = saved - eps;
        const double lm = loss();
        x[i] = saved;
        if (!std::isfinite(lp) || !std::isfinite(lm)) return std::numeric_limits<double>::infinity();
        const double numeric = (lp - lm) / (2 * eps);
        if (!std::isfinite(numeric)) return std::numeric_limits<double>::infinity();
        max_diff = std::max(max_diff, std::abs(numeric - analytic[i]));
    }
    return max_diff / max_grad;
}

// Σ out_i * g_i: a scalar loss whose gradient w.r.t. out is g.
inline double project(const std::vector<double>& out, const std::vector<double>& g) {
    double s = 0.0;
    for (size_t i = 0; i < out.size(); ++i) s += out[i] * g[i];
    return s;
}

inline double max_abs_diff(const float* a, const double* b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return std::numeric_limits<double>::infinity();
        m = std::max(m, std::abs(double(a[i]) - b[i]));
    }
    return m;
}

inline double max_abs(const double* a, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) m = std::max(m, std::abs(a[i]));
    return m;
}

// Atomically reserve a process-local test workspace; parallel suites cannot
// share files even if their clocks/counters happen to match.
inline std::string test_path(const std::string& name) {
    struct Workspace {
        std::filesystem::path dir;
        Workspace() {
            const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
            for (int i = 0; i < 100; ++i) {
                dir = std::filesystem::temp_directory_path() / ("cxxgpt-tests-" + std::to_string(stamp) + "-" + std::to_string(i));
                std::error_code ec;
                if (std::filesystem::create_directory(dir, ec)) return;
                CHECK(!ec, "cannot reserve test directory: %s", ec.message().c_str());
            }
            CHECK(false, "cannot reserve unique test directory");
        }
        ~Workspace() { std::error_code ec; std::filesystem::remove_all(dir, ec); }
    };
    static Workspace workspace;
    return (workspace.dir / name).string();
}
