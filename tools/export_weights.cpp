// Export a native inference checkpoint, retaining all weights, routing biases,
// configuration, step and RNG exactly, while omitting optimizer moments.
#include "../src/train.h"
int main(int argc, char** argv) {
    CHECK(argc == 3, "usage: export-weights INPUT.ckpt OUTPUT.ckpt");
    CHECK(std::string(argv[1]) != argv[2], "input and output must differ");
    ThreadPool pool(2); g_pool = &pool;
    Model<float> model(peek_checkpoint_config(argv[1]));
    Rng rng(1); load_checkpoint(argv[1], model, nullptr, &rng);
    FILE* f = std::fopen(argv[1], "rb"); CHECK(f, "cannot open input");
    char magic[8]; CHECK(std::fread(magic, 1, 8, f) == 8, "truncated header");
    read_config(f); AdamW opt;
    CHECK(std::fread(&opt.step, 8, 1, f) == 1, "truncated step"); std::fclose(f);
    save_checkpoint(argv[2], model, opt, rng);
    Model<float> restored(peek_checkpoint_config(argv[2])); Rng round_rng(1); AdamW round_opt;
    load_checkpoint(argv[2], restored, &round_opt, &round_rng, true);
    CHECK(model.params == restored.params && model.router_bias == restored.router_bias,
          "export changed weights or routing biases");
    CHECK(opt.step == round_opt.step && std::equal(rng.state(), rng.state() + 4, round_rng.state()),
          "export changed step or RNG");
    CHECK(round_opt.m.empty() && round_opt.v.empty(), "export retained optimizer moments");
    std::printf("BIT-EXACT INFERENCE EXPORT: %zu params, step %ld; use --init for new training\n",
                model.params.size(), opt.step);
}
