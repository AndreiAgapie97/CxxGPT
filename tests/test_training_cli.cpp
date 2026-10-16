// Native fixture generation and checkpoint comparison for the real trainer.
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include "test_accumulation.h"
#include "test_clone_experts.h"
#include "test_load_log.h"
#include "../src/reasoning.h"
#include "../src/tokenizer.h"

int main(int argc, char** argv) {
    if (argc == 5 && std::string(argv[1]) == "load-log") {
        Model<float> model(peek_checkpoint_config(argv[2]));
        load_checkpoint(argv[2], model, nullptr, nullptr);
        const TokenData data = TokenData::load(argv[3]);
        std::vector<int> inputs(8), targets(8), first, last;
        // Rebuild by hand the batches test_cli.sh's `train --steps 3 --accum 2
        // --batch 1 --seq 8` drew. This mirrors cmd_train in src/cxxgpt.cpp
        // (data_rng is Rng(seed + 1), seed 1 by default; every step calls
        // group.sample over the whole file) and AccumulationBatch::sample in
        // src/train.h (one sample_batch per micro-batch). If sampling changes
        // there, change it here too: the rows below would no longer match.
        Rng rng(2);
        for (int step = 1; step <= 3; ++step) {
            for (int micro = 0; micro < 2; ++micro) {
                data.sample_batch(rng, 1, 8, 0, data.tokens.size(), inputs.data(), targets.data());
                auto& expected = step <= 2 ? first : last;
                expected.insert(expected.end(), inputs.begin(), inputs.end());
            }
        }
        std::ifstream csv(argv[4]);
        std::string header;
        std::getline(csv, header);
        EXPECT(header == ExpertLoadLog::header(4), "missing load log header");
        check_load_log_rows(csv, 2, reference_load_counts(model, first));
        check_load_log_rows(csv, 3, reference_load_counts(model, last));
        EXPECT(!std::getline(csv, header), "unexpected load log rows");
        return g_failures ? 1 : 0;
    }
    if (argc == 3 && std::string(argv[1]) == "clone-fixture") {
        Model<float> model(clone_test_config()); AdamW opt; Rng rng(93);
        clone_test_fixture(model, opt); rng.next();
        const std::string dir = argv[2];
        save_checkpoint(dir + "/clone-adam.ckpt", model, opt, rng);
        opt.muon = true;
        save_checkpoint(dir + "/clone-muon.ckpt", model, opt, rng);
        opt.m.clear(); opt.v.clear();
        save_checkpoint(dir + "/clone-weights.ckpt", model, opt, rng);
        clone_test_load_log(dir + "/clone.load.csv");
        return 0;
    }
    if (argc == 6 && std::string(argv[1]) == "clone-reference") {
        Model<float> before(peek_checkpoint_config(argv[2])), after(peek_checkpoint_config(argv[3]));
        AdamW old, now; Rng a, b;
        load_checkpoint(argv[2], before, &old, &a, true);
        load_checkpoint(argv[3], after, &now, &b, true);
        // auto: what --auto must pick from clone_test_load_log; none: nothing.
        const std::string mode = argv[4];
        const std::vector<ExpertClone> edits = mode == "auto" ?
            std::vector<ExpertClone>{{0, 0, 2}, {1, 0, 1}, {1, 0, 3}} :
            mode == "none" ? std::vector<ExpertClone>{} : std::vector<ExpertClone>{{0, 0, 2}};
        clone_reference(before, old, after, now, edits, std::stod(argv[5]), 31);
        for (int i = 0; i < 4; ++i) EXPECT(a.state()[i] == b.state()[i], "CLI clone changed sampler state");
        return g_failures ? 1 : 0;
    }
    if (argc == 5 && std::string(argv[1]) == "bpe-stream-reference") {
        Tokenizer tok; tok.load(argv[2]);
        std::ifstream input(argv[3],std::ios::binary); CHECK(input, "cannot open whole-text reference");
        const std::string text(std::istreambuf_iterator<char>{input},{});
        CHECK(!input.bad(), "cannot read whole-text reference");
        const auto whole = tok.encode(text);
        const auto streamed = TokenData::load(argv[4]);
        EXPECT(streamed.vocab==uint32_t(tok.vocab_size()), "streamed BPE vocabulary header differs");
        EXPECT(tok.num_merges()>0 && whole.size()<text.size()/2, "stream fixture did not exercise learned merges");
        EXPECT(std::equal(whole.begin(),whole.end(),streamed.tokens.begin(),streamed.tokens.end()),
               "streamed BPE token ids differ from whole-text encoding");
        return g_failures ? 1 : 0;
    }

    if (argc == 2 && std::string(argv[1]) == "bad-token-save") {
        TokenData data; data.vocab = 256; data.tokens.assign(10000,1); data.mask.assign(10000,1);
        data.save("/dev/full");
        return 2;
    }

    if (argc == 4 && std::string(argv[1]) == "clean-bos") {
        const auto text=TokenData::load(argv[2]), qa=TokenData::load(argv[3]);
        EXPECT(text.tokens==std::vector<uint16_t>({97,98}), "raw text BOS remained");
        EXPECT(std::count(qa.tokens.begin(),qa.tokens.end(),uint16_t(kBos))==1,
               "QA source injected extra document BOS tokens");
        return g_failures ? 1 : 0;
    }

    if (argc == 3 && std::string(argv[1]) == "numeric-labels") {
        const auto ps=load_problems(argv[2]);
        EXPECT(ps.size()==3 && ps[0].answer=="9007199254740993" && ps[1].answer=="1200" && ps[2].answer=="-0.0012",
               "numeric QA labels lost exact decimal spelling");
        return g_failures ? 1 : 0;
    }

    if (argc == 2 && std::string(argv[1]) == "bad-tokenizer-vocab") {
        Tokenizer tok; tok.train("a a",65537);
        return 2;
    }

    if (argc == 3 && std::string(argv[1]) == "repeat-backward") {
        ThreadPool tp(1); g_pool = &tp;
        Config cfg; cfg.vocab = 3; cfg.seq_len = 2; cfg.d_model = 4; cfg.n_heads = 1; cfg.n_layers = 1;
        Model<float> m(cfg); m.init(7);
        const int token = 1, target = std::stoi(argv[2]);
        m.forward(&token, &target, 1, 1); m.zero_grad(); m.backward(); m.backward();
        return 2;
    }

    if (argc == 3 && std::string(argv[1]) == "bad-target") {
        float logits[] = {1,2,3}, loss = 0, gradient[3];
        const int target = std::stoi(argv[2]);
        cross_entropy_forward(logits, &loss, &target, 1, 3);
        cross_entropy_backward(gradient, logits, &target, 1, 3, 1.0f);
        return 2;  // bad target must fail before indexing, valid ignore must return loss zero
    }
    if (argc == 3 && std::string(argv[1]) == "bad-target-backward") {
        const float probs[] = {0.2f,0.3f,0.5f}; float gradient[3];
        const int target = std::stoi(argv[2]);
        cross_entropy_backward(gradient, probs, &target, 1, 3, 1.0f);
        return 2;
    }

    // gen-word (word) and gen-math (math) write the same two files.
    if (argc == 3 && (std::string(argv[1]) == "word" || std::string(argv[1]) == "math")) {
        const bool math = std::string(argv[1]) == "math";
        const std::string prefix(argv[2]);
        const auto ps = load_problems(prefix + ".txt");
        EXPECT(ps.size() == 28, "wrong problem count");
        std::string expected_text;
        std::vector<uint16_t> expected_tokens;
        for (const auto& p : ps) {
            EXPECT(math ? (p.task == "equations" || p.task == "fractions") && p.difficulty >= 1
                        : p.task == "word" && p.difficulty >= 2 && p.difficulty <= 8, "bad problem metadata");
            const std::string document = p.prompt() + p.completion();
            expected_text += document + "<|endoftext|>\n";
            expected_tokens.push_back(kBos);
            for (unsigned char byte : trim(document)) expected_tokens.push_back(byte);
        }
        std::ifstream text(prefix + ".text", std::ios::binary);
        const std::string actual_text(std::istreambuf_iterator<char>(text), {});
        EXPECT(actual_text == expected_text, "word text differs from the loaded problem file");
        const auto tokens = TokenData::load(prefix + ".tok");
        EXPECT(tokens.tokens == expected_tokens, "text2tok did not preserve the word documents");
        return g_failures ? 1 : 0;
    }
    if (argc == 3 && std::string(argv[1]) == "curriculum") {
        const std::string dir(argv[2]);
        const auto control = load_problems(dir + "/control.txt"), expanded = load_problems(dir + "/expanded.txt");
        const auto probe = load_problems(dir + "/probe.txt"), validation = load_problems(dir + "/validation.txt");
        EXPECT(control.size() == 24 && expanded.size() == 24 && probe.size() == 16 && validation.size() == 12,
               "curriculum fixture has unexpected split sizes");
        std::set<std::string> reserved, c, e;
        Rng rng(20261014); reserved.insert(make_problem("add", 1, rng).question); reserved.insert("1+1");
        for (const auto* split : {&probe, &validation})
            for (const auto& p : *split) EXPECT(reserved.insert(p.question).second, "curriculum holdout leaked");
        for (const auto& p : control) {
            EXPECT(p.difficulty <= 2 && !reserved.contains(p.question), "control leaked or changed difficulty support");
            EXPECT(c.insert(p.question).second, "duplicate control question");
        }
        size_t harder = 0, shared = 0;
        for (const auto& p : expanded) {
            EXPECT(!reserved.contains(p.question) && e.insert(p.question).second, "expanded question leaked/duplicated");
            harder += p.difficulty == 3; shared += c.contains(p.question);
        }
        EXPECT(harder == 8 && shared == 16, "paired curriculum did not replace exactly one third of problems");
        return g_failures ? 1 : 0;
    }
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
        auto load_model = load_log_fixture();
        save_checkpoint((dir / "load.ckpt").string(), load_model, AdamW{}, Rng(2));
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
    if ((argc == 4 || argc == 5) && (std::string(argv[1]) == "fresh" || std::string(argv[1]) == "fresh-ternary" ||
                     std::string(argv[1]) == "fresh-hybrid" || std::string(argv[1]) == "fresh-all-ternary" ||
                     std::string(argv[1]) == "fresh-nope")) {
        Model<float> a(peek_checkpoint_config(argv[2])), b(peek_checkpoint_config(argv[3]));
        AdamW opt;
        Rng rng(0), expected(argc == 5 ? uint64_t(std::stoul(argv[4])) + 1 : 2);
        load_checkpoint(argv[2], a, nullptr, nullptr); load_checkpoint(argv[3], b, &opt, &rng);
        EXPECT(a.params == b.params && a.router_bias == b.router_bias, "warm start changed weights/biases");
        if (std::string(argv[1]) == "fresh-ternary" || std::string(argv[1]) == "fresh-hybrid")
            EXPECT(!a.cfg.ternary && b.cfg.ternary && !b.qparams.empty(), "warm start did not convert to ternary");
        if (std::string(argv[1]) == "fresh-hybrid")
            EXPECT(b.cfg.fp_attention, "warm start did not retain full-precision attention");
        if (std::string(argv[1]) == "fresh-all-ternary")
            EXPECT(a.cfg.fp_attention && b.cfg.ternary && !b.cfg.fp_attention,
                   "explicit warm start did not convert hybrid attention to ternary");
        if (std::string(argv[1]) == "fresh-nope")
            EXPECT(a.cfg.nope_every == 0 && b.cfg.nope_every == 1, "explicit warm start ignored NoPE");
        EXPECT(opt.step == 0 && opt.m.empty() && opt.v.empty(), "warm start retained optimizer state");
        for (int i = 0; i < 4; ++i) EXPECT(rng.state()[i] == expected.state()[i], "warm start retained data RNG");
        return g_failures ? 1 : 0;
    }
    if (argc == 4 && std::string(argv[1]) == "different-order") {
        Model<float> a(peek_checkpoint_config(argv[2])), b(peek_checkpoint_config(argv[3]));
        Rng ra(0), rb(0);
        load_checkpoint(argv[2], a, nullptr, &ra); load_checkpoint(argv[3], b, nullptr, &rb);
        TokenData data; data.vocab = 256;
        for (int i = 0; i < 256; ++i) data.tokens.push_back(i);
        std::vector<int> x(32), y(32), u(32), v(32);
        data.sample_batch(ra, 4, 8, 0, data.tokens.size(), x.data(), y.data());
        data.sample_batch(rb, 4, 8, 0, data.tokens.size(), u.data(), v.data());
        EXPECT(x != u, "different INIT seeds replayed identical windows");
        EXPECT(a.params == b.params, "different INIT data seed changed loaded weights");
        return g_failures ? 1 : 0;
    }
    if (argc == 4 && std::string(argv[1]) == "unchanged-zero-step") {
        Model<float> a(peek_checkpoint_config(argv[2])), b(peek_checkpoint_config(argv[3]));
        AdamW opt; Rng rng(0);
        load_checkpoint(argv[2], a, nullptr, nullptr); load_checkpoint(argv[3], b, &opt, &rng);
        EXPECT(a.params == b.params && a.router_bias == b.router_bias, "skipped attempt changed saved weights");
        EXPECT(opt.step == 0 && opt.m.empty() && opt.v.empty(), "skipped attempt advanced optimizer");
        return g_failures ? 1 : 0;
    }
    if (argc == 3 && std::string(argv[1]) == "problems") {
        const auto ps = load_problems(argv[2]);
        EXPECT(ps.size() == 2 && ps[0].question == "q" && ps[0].answer == "14" &&
               ps[1].question == "tail" && ps[1].answer == "1080" && ps[1].difficulty == 2,
               "CRLF or unterminated problem record was lost/corrupted");
        return g_failures ? 1 : 0;
    }
    if (argc == 3 && std::string(argv[1]) == "recover-save") {
        Model<float> m(peek_checkpoint_config(argv[2])); AdamW opt; Rng rng(0);
        load_checkpoint(argv[2], m, &opt, &rng);
        // /dev/full accepts fopen but fails writes/flush: the old file must survive.
        std::filesystem::create_symlink("/dev/full", std::string(argv[2]) + ".tmp");
        EXPECT(!save_checkpoint(argv[2], m, opt, rng, true), "failed write reported success");
        EXPECT(!std::filesystem::exists(std::string(argv[2]) + ".tmp"), "failed temporary file remained");
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
