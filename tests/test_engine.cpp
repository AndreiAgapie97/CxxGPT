// test_engine.cpp — tests for the training engine. Exit status 0 iff all pass.
//   ./test_engine [-t threads]
#include <cstdlib>
#include <cstring>
#include <thread>

#include "test_model.h"
#include "test_memory.h"
#include "test_ops.h"
#include "test_reasoning.h"
#include "test_reasoning_metrics.h"
#include "test_tokenizer.h"
#include "test_data.h"
#include "test_numerics.h"
#include "test_train.h"
#include "test_optim.h"
#include "test_reasoning_traces.h"

int main(int argc, char** argv) {
    int threads = int(std::thread::hardware_concurrency());
    for (int i = 1; i < argc; ++i)
        if (!std::strcmp(argv[i], "-t") && i + 1 < argc) threads = std::atoi(argv[++i]);
    ThreadPool tp(threads);
    g_pool = &tp;

    struct { const char* name; void (*fn)(); } suites[] = {
        {"ops", test_ops},
        {"model", test_model},
        {"memory", test_memory},
        {"reasoning", test_reasoning},
        {"reasoning metrics", test_reasoning_metrics},
        {"tokenizer", test_tokenizer},
        {"data", test_data},
        {"numerics", test_numerics},
        {"policy gradient", test_policy_gradient_expectation},
        {"train", test_train},
        {"optim", test_optim},
        {"generated traces", test_reasoning_traces},
    };
    for (auto& s : suites) {
        const int before = g_failures;
        const double t0 = now_seconds();
        std::printf("%s\n", s.name);
        s.fn();
        std::printf("  %s (%.2f s)\n", g_failures == before ? "ok" : "FAILED", now_seconds() - t0);
    }
    std::printf(g_failures ? "%d failure(s)\n" : "all passed\n", g_failures);
    return g_failures ? 1 : 0;
}
