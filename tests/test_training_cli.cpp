// Native fixture generation and checkpoint comparison for the real trainer.
#include <filesystem>
#include <set>
#include "test_accumulation.h"
#include "../src/reasoning.h"

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
        Config cfg;
        cfg.vocab = 256; cfg.seq_len = 8; cfg.d_model = 8;
        cfg.n_layers = 1; cfg.n_heads = 2; cfg.ffn_hidden = 12;
        Model<float> unstable(cfg);
        unstable.params.assign(unstable.params.size(), 1e20f);
        save_checkpoint((dir / "unstable.ckpt").string(), unstable, AdamW{}, Rng(2));
        cfg.seq_len = 32;
        Model<float> audit(cfg); audit.init(7);
        save_checkpoint((dir / "audit.ckpt").string(), audit, AdamW{}, Rng(2));
        Problem p; p.task = "add"; p.question = "1+1"; p.steps = "1+1=2"; p.answer = "2";
        save_problems((dir / "audit.txt").string(), {p});
        Rng probe_rng(20261014);
        const Problem candidate = make_problem("add", 1, probe_rng);
        const std::string text = std::string(1, char(kBos)) + candidate.prompt() + candidate.completion();
        TokenData source; source.vocab = 256;
        for (size_t i = 0; i < text.size(); ++i) {
            source.tokens.push_back(uint8_t(text[i]));
            source.mask.push_back(i >= 1 + candidate.prompt().size());
        }
        source.save((dir / "targets.tok").string());
        return 0;
    }
    if (argc == 5 && std::string(argv[1]) == "targets") {
        const auto source = TokenData::load(argv[2]), final = TokenData::load(argv[3]);
        Rng rng(20261014); const auto excluded = make_problem("add", 1, rng);
        EXPECT(source.tokens == final.tokens && source.vocab == final.vocab, "mask preparation changed tokens");
        EXPECT(size_t(std::count(final.mask.begin(), final.mask.end(), uint8_t(1))) == excluded.answer.size(),
               "final mask did not label exactly the answer digits");
        for (size_t i = 0; i < final.mask.size(); ++i)
            EXPECT(!final.mask[i] || source.mask[i], "final mask enabled a previously ignored token");
        const auto probe = load_problems(argv[4]); std::set<std::string> unique;
        EXPECT(probe.size() == 18, "probe does not have three questions in six cells");
        for (const auto& p : probe) {
            EXPECT(p.question != excluded.question && p.question != "1+1", "held-out probe leaked a known question");
            EXPECT(unique.insert(p.question).second, "duplicate probe question");
        }
        return g_failures ? 1 : 0;
    }
    if (argc == 3 && std::string(argv[1]) == "truncate") {
        Model<float> model(peek_checkpoint_config(argv[2]));
        size_t bytes = 1; // one byte of the last optimizer float, plus following biases
        for (const auto& bias : model.router_bias) bytes += bias.size() * sizeof(float);
        std::filesystem::resize_file(argv[2], std::filesystem::file_size(argv[2]) - bytes);
        return 0;
    }
    if (argc == 4 && std::string(argv[1]) == "same-sampler") {
        Model<float> a(peek_checkpoint_config(argv[2])), b(peek_checkpoint_config(argv[3]));
        Rng ra(0), rb(0);
        load_checkpoint(argv[2], a, nullptr, &ra); load_checkpoint(argv[3], b, nullptr, &rb);
        for (int i = 0; i < 4; ++i) EXPECT(ra.state()[i] == rb.state()[i], "experiment arms consumed different samples");
        return g_failures ? 1 : 0;
    }
    if (argc == 4 && std::string(argv[1]) == "fresh") {
        Model<float> a(peek_checkpoint_config(argv[2])), b(peek_checkpoint_config(argv[3]));
        AdamW opt;
        Rng rng(0), expected(2);
        load_checkpoint(argv[2], a, nullptr, nullptr); load_checkpoint(argv[3], b, &opt, &rng);
        EXPECT(a.params == b.params && a.router_bias == b.router_bias, "warm start changed weights/biases");
        EXPECT(opt.step == 0 && opt.m.empty() && opt.v.empty(), "warm start retained optimizer state");
        for (int i = 0; i < 4; ++i) EXPECT(rng.state()[i] == expected.state()[i], "warm start retained data RNG");
        return g_failures ? 1 : 0;
    }
    if (argc == 4 && std::string(argv[1]) == "bad-save") {
        Model<float> model(peek_checkpoint_config(argv[2]));
        AdamW opt; Rng rng(0);
        load_checkpoint(argv[2], model, &opt, &rng);
        const std::string kind(argv[3]);
        if (kind == "params") model.params[0] = NAN;
        else if (kind == "m") opt.m[0] = INFINITY;
        else if (kind == "v") opt.v[0] = NAN;
        else if (kind == "short-v") opt.v.pop_back();
        else return 2;
        save_checkpoint(argv[2], model, opt, rng); // must fail before replacing the original
        return 2;
    }
    if ((argc == 4 || argc == 5) && std::string(argv[1]) == "compare") {
        Model<float> a(peek_checkpoint_config(argv[2])), b(peek_checkpoint_config(argv[3]));
        AdamW oa, ob;
        oa.muon = ob.muon = argc == 5 && std::string(argv[4]) == "--muon";
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
    std::fprintf(stderr, "usage: test_training_cli generate DIR | compare MACRO MICRO [--muon]\n");
    return 1;
}
