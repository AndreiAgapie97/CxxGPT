// Compile from the repository root:
// g++ -O3 -march=native -std=c++20 -pthread -Wall -Wextra -Werror docs/results/2026-10-13/storage-audit.cpp -o /tmp/storage-audit
// /tmp/storage-audit CHECKPOINT [CHECKPOINT...]
#include "../../../src/infer.h"
#include "../../../src/train.h"

int main(int argc, char** argv) {
    CHECK(argc > 1, "provide a trained ternary checkpoint");
    ThreadPool tp(3); g_pool = &tp;
    std::printf("checkpoint,params,active_params,training_array_bytes,fp32_weight_bytes,inference_weight_bytes\n");
    for (int i = 1; i < argc; ++i) {
        Model<float> model(peek_checkpoint_config(argv[i]));
        AdamW opt;
        load_checkpoint(argv[i], model, &opt, nullptr);
        TernaryEngine engine(model);
        const size_t arrays = model.params.size() + model.grads.size() + model.qparams.size() + opt.m.size() + opt.v.size();
        std::printf("%s,%zu,%zu,%zu,%zu,%zu\n", argv[i], model.num_params(), model.active_params(),
                    arrays * sizeof(float), model.num_params() * sizeof(float), engine.weight_bytes());
    }
}
