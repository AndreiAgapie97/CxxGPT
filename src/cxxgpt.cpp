// cxxgpt — train and run the model.
//
//   cxxgpt text2tok --in text.txt --out data.tok          bytes -> token file
//   cxxgpt train --data data.tok [options]                train (or resume)
//   cxxgpt sample --ckpt model.ckpt [--prompt TEXT]       generate text
//
// Run a subcommand with --help for its options.
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <thread>

#include "args.h"
#include "common.h"
#include "model.h"
#include "train.h"

namespace {

volatile std::sig_atomic_t g_stop = 0;
void on_signal(int) { g_stop = 1; }

int threads_arg(Args& a) {
    return int(a.integer("threads", long(std::max(1u, std::thread::hardware_concurrency()))));
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    CHECK(in, "cannot open %s", path.c_str());
    return std::string(std::istreambuf_iterator<char>(in), {});
}

// Model options, shared by every command that builds a fresh model.
Config config_from_args(Args& a, int vocab) {
    Config c;
    c.vocab = int(a.integer("vocab", vocab));
    c.seq_len = int(a.integer("seq", 256));
    c.d_model = int(a.integer("dim", 256));
    c.n_layers = int(a.integer("layers", 4));
    c.n_heads = int(a.integer("heads", 4));
    c.ffn_hidden = int(a.integer("ffn", 0));
    c.n_experts = int(a.integer("experts", 0));
    c.top_k = int(a.integer("topk", 2));
    c.expert_hidden = int(a.integer("ehidden", 0));
    c.n_shared = int(a.integer("shared", 0));
    c.ternary = a.flag("ternary");
    return c;
}

void print_config(const Config& c, const Model<float>& m) {
    std::printf("model: vocab %d, seq %d, dim %d, layers %d, heads %d, ", c.vocab, c.seq_len,
                c.d_model, c.n_layers, c.n_heads);
    if (c.moe())
        std::printf("MoE %d experts (top-%d, hidden %d, %d shared)", c.n_experts, c.top_k,
                    c.ehidden(), c.n_shared);
    else
        std::printf("dense FFN %d", c.dense_hidden());
    std::printf("%s\n", c.ternary ? ", ternary BitLinear" : "");
    std::printf("params: %.3f M total, %.3f M active per token\n", m.num_params() / 1e6,
                m.active_params() / 1e6);
}

int cmd_text2tok(Args& a) {
    const std::string in = a.str("in"), out = a.str("out");
    a.done();
    CHECK(!in.empty() && !out.empty(), "usage: cxxgpt text2tok --in text.txt --out data.tok");
    const std::string text = read_file(in);
    TokenData d;
    d.vocab = 256;
    d.tokens.assign(text.begin(), text.end());
    for (auto& t : d.tokens) t &= 0xFF;  // char may be signed
    d.save(out);
    std::printf("%zu bytes -> %s\n", d.tokens.size(), out.c_str());
    return 0;
}

double eval_loss(Model<float>& model, const TokenData& data, size_t lo, size_t hi, int B, int Tn,
                 int batches) {
    Rng rng(12345);  // the same batches every time, so evals are comparable
    std::vector<int> inp(size_t(B) * Tn), tgt(size_t(B) * Tn);
    double total = 0.0;
    for (int i = 0; i < batches; ++i) {
        data.sample_batch(rng, B, Tn, lo, hi, inp.data(), tgt.data());
        total += model.forward(inp.data(), tgt.data(), B, Tn);
    }
    return total / batches;
}

int cmd_train(Args& a) {
    if (a.flag("help")) {
        std::printf(
            "cxxgpt train --data FILE [options]\n"
            "data:   --val FILE (default: last --val-frac of --data, 0.05)\n"
            "model:  --seq 256 --dim 256 --layers 4 --heads 4 --ffn 0(auto)\n"
            "        --experts 0 --topk 2 --ehidden 0(auto) --shared 0 --ternary\n"
            "train:  --batch 16 --accum 1 --steps 2000 --lr 1e-3 --min-lr-frac 0.1\n"
            "        --warmup 100 --wd 0.1 --clip 1.0 --bias-rate 1e-3 --seed 1\n"
            "output: --out model.ckpt --resume --log FILE.csv --log-every 10\n"
            "        --eval-every 200 --eval-batches 20 --save-every 500\n"
            "        --time-limit MINUTES --threads N\n"
            "Ctrl-C saves a checkpoint and stops; --resume continues from --out.\n");
        return 0;
    }
    ThreadPool tp(threads_arg(a));
    g_pool = &tp;
    const std::string data_path = a.str("data"), val_path = a.str("val");
    const std::string out = a.str("out", "model.ckpt"), log_path = a.str("log");
    CHECK(!data_path.empty(), "--data is required (see cxxgpt train --help)");
    const TokenData data = TokenData::load(data_path);
    TokenData val_data;
    if (!val_path.empty()) val_data = TokenData::load(val_path);
    const double val_frac = a.real("val-frac", 0.05);
    const bool resume = a.flag("resume") && std::filesystem::exists(out);

    Config cfg = config_from_args(a, int(data.vocab));
    if (resume) cfg = peek_checkpoint_config(out);
    const int B = int(a.integer("batch", 16)), accum = int(a.integer("accum", 1));
    const int Tn = std::min(cfg.seq_len, int(a.integer("seq", cfg.seq_len)));
    const long steps = a.integer("steps", 2000), warmup = a.integer("warmup", 100);
    const double lr = a.real("lr", 1e-3), min_frac = a.real("min-lr-frac", 0.1);
    const double clip = a.real("clip", 1.0), bias_rate = a.real("bias-rate", 1e-3);
    const long seed = a.integer("seed", 1);
    const int log_every = int(a.integer("log-every", 10)), eval_every = int(a.integer("eval-every", 200));
    const int eval_batches = int(a.integer("eval-batches", 20)), save_every = int(a.integer("save-every", 500));
    const double time_limit = a.real("time-limit", 0) * 60.0;
    AdamW opt;
    opt.weight_decay = a.real("wd", 0.1);
    a.done();
    CHECK(cfg.vocab >= int(data.vocab), "model vocab %d < data vocab %u", cfg.vocab, data.vocab);

    Model<float> model(cfg);
    model.init(uint64_t(seed));
    Rng data_rng(uint64_t(seed) + 1);
    if (resume) {
        load_checkpoint(out, model, &opt, &data_rng);
        std::printf("resumed %s at step %ld\n", out.c_str(), opt.step);
    }
    print_config(cfg, model);

    // Train on [0, train_end) of data; validate on val_data or the tail.
    size_t train_end = data.tokens.size();
    const TokenData* val = &val_data;
    size_t val_lo = 0, val_hi = val_data.tokens.size();
    if (val_path.empty()) {
        train_end = size_t(double(data.tokens.size()) * (1.0 - val_frac));
        val = &data;
        val_lo = train_end;
        val_hi = data.tokens.size();
    }
    std::printf("data: %zu train tokens, %zu val tokens; %d x %d x %d = %d tokens per step\n",
                train_end, val_hi - val_lo, B, Tn, accum, B * Tn * accum);

    FILE* log = nullptr;
    if (!log_path.empty()) {
        const bool fresh = !std::filesystem::exists(log_path);
        log = std::fopen(log_path.c_str(), "a");
        CHECK(log, "cannot open %s", log_path.c_str());
        if (fresh) std::fprintf(log, "step,loss,val_loss,lr,grad_norm,ms_per_step,load_ratio\n");
    }
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    std::vector<int> inp(size_t(B) * Tn), tgt(size_t(B) * Tn);
    const double t_start = now_seconds();
    double loss_acc = 0, ms_acc = 0;
    int n_acc = 0;
    long step = opt.step;
    for (; step < steps && !g_stop; ++step) {
        const double t0 = now_seconds();
        model.zero_grad();
        double loss = 0.0;
        for (int micro = 0; micro < accum; ++micro) {
            data.sample_batch(data_rng, B, Tn, 0, train_end, inp.data(), tgt.data());
            loss += model.forward(inp.data(), tgt.data(), B, Tn, true) / accum;
            model.backward(1.0 / accum);
        }
        const double gnorm = clip_grad_norm(model, clip);
        const double lr_now = lr_at(step, warmup, steps, lr, min_frac);
        opt.update(model, lr_now);
        const double load = model.max_load_ratio();
        model.update_router_bias(bias_rate);
        const double ms = (now_seconds() - t0) * 1e3;
        loss_acc += loss;
        ms_acc += ms;
        ++n_acc;

        const bool last = step + 1 == steps;
        double vloss = -1;
        if (eval_every > 0 && ((step + 1) % eval_every == 0 || last) && val_hi > val_lo + Tn + 1)
            vloss = eval_loss(model, *val, val_lo, val_hi, B, Tn, eval_batches);
        if ((step + 1) % log_every == 0 || last || vloss >= 0) {
            std::printf("step %6ld | loss %.4f", step + 1, loss_acc / n_acc);
            if (vloss >= 0) std::printf(" | val %.4f", vloss);
            std::printf(" | lr %.2e | gnorm %6.3f | %6.1f ms | %7.0f tok/s", lr_now, gnorm,
                        ms_acc / n_acc, B * Tn * accum / (ms_acc / n_acc) * 1e3);
            if (cfg.moe()) std::printf(" | load %.2f", load);
            std::printf("\n");
            std::fflush(stdout);
            if (log) {
                std::fprintf(log, "%ld,%.5f,%.5f,%.3e,%.4f,%.1f,%.3f\n", step + 1, loss_acc / n_acc,
                             vloss, lr_now, gnorm, ms_acc / n_acc, load);
                std::fflush(log);
            }
            loss_acc = ms_acc = 0;
            n_acc = 0;
        }
        if (save_every > 0 && (step + 1) % save_every == 0) save_checkpoint(out, model, opt, data_rng);
        if (time_limit > 0 && now_seconds() - t_start > time_limit) {
            std::printf("time limit reached\n");
            ++step;
            break;
        }
    }
    save_checkpoint(out, model, opt, data_rng);
    std::printf("saved %s at step %ld%s\n", out.c_str(), opt.step, g_stop ? " (interrupted)" : "");
    if (log) std::fclose(log);
    return 0;
}

// Samples the next token from row `row` of the model's raw logits.
int sample_token(const Model<float>& model, int row, double temperature, Rng& rng) {
    const float* lg = model.output_row(row);
    const int V = model.cfg.vocab;
    int best = 0;
    for (int i = 1; i < V; ++i)
        if (lg[i] > lg[best]) best = i;
    if (temperature <= 0) return best;
    std::vector<double> p(V);
    double sum = 0;
    for (int i = 0; i < V; ++i) sum += p[i] = std::exp((lg[i] - lg[best]) / temperature);
    double u = rng.uniform() * sum;
    for (int i = 0; i < V; ++i)
        if ((u -= p[i]) <= 0) return i;
    return best;
}

int cmd_sample(Args& a) {
    ThreadPool tp(threads_arg(a));
    g_pool = &tp;
    const std::string ckpt = a.str("ckpt", "model.ckpt"), prompt = a.str("prompt", "\n");
    const int n = int(a.integer("n", 500));
    const double temp = a.real("temp", 0.8);
    Rng rng(uint64_t(a.integer("seed", 1)));
    a.done();
    Model<float> model(peek_checkpoint_config(ckpt));
    load_checkpoint(ckpt, model, nullptr, nullptr);
    CHECK(model.cfg.vocab == 256, "sample: only byte-level models for now");
    std::vector<int> toks;
    for (unsigned char ch : prompt) toks.push_back(ch);
    std::fwrite(prompt.data(), 1, prompt.size(), stdout);
    const int S = model.cfg.seq_len;
    for (int i = 0; i < n; ++i) {
        const int len = std::min(int(toks.size()), S);
        model.forward(&toks[toks.size() - len], nullptr, 1, len);
        const int t = sample_token(model, len - 1, temp, rng);
        toks.push_back(t);
        std::fputc(t, stdout);
        std::fflush(stdout);
    }
    std::printf("\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string cmd = argc > 1 ? argv[1] : "";
    Args a(argc, argv, 2);
    if (cmd == "text2tok") return cmd_text2tok(a);
    if (cmd == "train") return cmd_train(a);
    if (cmd == "sample") return cmd_sample(a);
    std::printf("usage: cxxgpt <text2tok|train|sample> [--options]   (--help per command)\n");
    return cmd.empty() || cmd == "help" ? 0 : 1;
}
