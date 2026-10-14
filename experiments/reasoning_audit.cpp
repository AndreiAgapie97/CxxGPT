// Audit frozen byte-token checkpoints; never updates their weights.
#include <filesystem>
#include "../src/args.h"
#include "../src/generate.h"
#include "../src/reasoning_metrics.h"
#include "../src/train.h"

int main(int argc, char** argv) {
    Args args(argc, argv, 1);
    const std::string ckpt = args.str("ckpt"), path = args.str("problems"), val_path = args.str("val-data");
    const int threads = int(args.integer("threads", 2)), batch = int(args.integer("batch", 2));
    const int eval_batches = int(args.integer("eval-batches", 5));
    args.done();
    CHECK(!ckpt.empty() && !path.empty() && threads > 0 && batch > 0 && eval_batches > 0, "invalid audit arguments");
    ThreadPool pool_(threads); g_pool = &pool_;
    Model<float> model(peek_checkpoint_config(ckpt));
    CHECK(model.cfg.vocab == 256, "audit supports unpadded byte-vocabulary checkpoints only");
    load_checkpoint(ckpt, model, nullptr, nullptr);
    const auto problems = load_problems(path);
    const auto exact = eval_reasoning(model, problems);
    reasoning_metrics::Score total;
    size_t fitted = 0;
    for (const auto& problem : problems) {
        const auto seq = reasoning_metrics::sequence(problem);
        if (seq.input.size() > size_t(model.cfg.seq_len)) continue;
        ++fitted;
        model.forward(seq.input.data(), nullptr, 1, int(seq.input.size()), false);
        reasoning_metrics::merge(total, reasoning_metrics::score(model.output_row(0), model.cfg.vocab, seq));
    }
    const std::string id = std::filesystem::path(ckpt).stem().string();
    std::printf("checkpoint,metric,tokens,mean_loss,correct,total\n");
    auto print = [&](const char* kind, const reasoning_metrics::Metric& m) {
        std::printf("%s,%s,%zu,%.9f,%zu,%zu\n", id.c_str(), kind, m.tokens, m.mean(), m.correct, m.tokens);
    };
    print("all_completion", total.all);
    for (int r = reasoning_metrics::step_digit; r < reasoning_metrics::roles; ++r)
        print(reasoning_metrics::name(reasoning_metrics::Role(r)), total.by_role[r]);
    std::printf("%s,teacher_coverage,0,0,%zu,%zu\n", id.c_str(), fitted, problems.size());
    std::printf("%s,exact_answer,0,0,%d,%d\n", id.c_str(), exact.correct, exact.total);
    if (!val_path.empty()) {
        const auto val = TokenData::load(val_path);
        Rng rng(12345); TokenLossMean global;
        double legacy = 0;
        std::vector<int> in(size_t(batch) * model.cfg.seq_len), targets(in.size());
        for (int i = 0; i < eval_batches; ++i) {
            val.sample_batch(rng, batch, model.cfg.seq_len, 0, val.tokens.size(), in.data(), targets.data());
            const double loss = model.forward(in.data(), targets.data(), batch, model.cfg.seq_len, false);
            global.add(loss, count_targets(targets.data(), targets.size())); legacy += loss / eval_batches;
        }
        std::printf("%s,sampled_val_token_mean,%zu,%.9f,0,0\n", id.c_str(), global.count, global.value());
        std::printf("%s,sampled_val_legacy_batch_mean,%zu,%.9f,0,0\n", id.c_str(), global.count, legacy);
    }
}
