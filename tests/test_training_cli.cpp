// Native fixture generation and checkpoint comparison for the real trainer.
#include <filesystem>
#include "test_accumulation.h"

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "generate") {
        const std::filesystem::path dir(argv[2]);
        TokenData data;
        data.vocab = 13;
        for (int i = 0; i < 13; ++i) data.tokens.push_back(i);
        data.mask.assign(13, 0);
        for (int i : {1, 10, 11, 12}) data.mask[i] = 1;
        data.save((dir / "masked.tok").string());
        data.mask.assign(13, 0); data.mask[12] = 1;
        data.save((dir / "sparse.tok").string());
        data.mask.assign(13, 0);
        data.save((dir / "ignored.tok").string());
        return 0;
    }
    if (argc == 4 && std::string(argv[1]) == "compare") {
        Model<float> a(peek_checkpoint_config(argv[2])), b(peek_checkpoint_config(argv[3]));
        AdamW oa, ob;
        Rng ra(0), rb(0);
        load_checkpoint(argv[2], a, &oa, &ra); load_checkpoint(argv[3], b, &ob, &rb);
        EXPECT(oa.step == 4 && ob.step == 4, "CLI did not complete four optimizer updates");
        const double param = accumulation_difference(a.params, b.params);
        const double moment = std::max(accumulation_difference(oa.m, ob.m), accumulation_difference(oa.v, ob.v));
        EXPECT(param < 2e-6 && moment < 2e-7, "CLI partition changed training: params %g moments %g", param, moment);
        EXPECT(a.router_bias == b.router_bias, "CLI partition changed balancing");
        for (int i = 0; i < 4; ++i) EXPECT(ra.state()[i] == rb.state()[i], "CLI partition changed sampler state");
        std::printf("  CLI checkpoint partition: params %.1e moments %.1e; four updates, RNG matched\n", param, moment);
        return g_failures ? 1 : 0;
    }
    std::fprintf(stderr, "usage: test_training_cli generate DIR | compare MACRO MICRO\n");
    return 1;
}
