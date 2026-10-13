// cxxgpt — train and run the model.
//
//   cxxgpt bpe-train --in text.txt --vocab 4096 --out tok.bpe   learn a BPE tokenizer
//   cxxgpt text2tok --in text.txt --out data.tok [--bpe tok.bpe]  text -> token file
//   cxxgpt train --data data.tok [options]                train (or resume)
//   cxxgpt sample --ckpt model.ckpt [--prompt TEXT] [--bpe tok.bpe]  generate text
//   cxxgpt gen-reasoning --out data/reason                 synthetic reasoning data
//   cxxgpt prep-qa --in gsm8k.jsonl --out data/gsm8k/train  Q/A JSON Lines -> problems + tokens
//   cxxgpt eval-reasoning --ckpt model.ckpt --test FILE    exact-answer accuracy
//   cxxgpt bench-infer [--ckpt model.ckpt | model options]  ternary engine vs fp32 decode
//
// Run a subcommand with --help for its options.
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <thread>

#include "args.h"
#include "common.h"
#include "generate.h"
#include "infer.h"
#include "json.h"
#include "model.h"
#include "reasoning.h"
#include "profile.h"
#include "tokenizer.h"
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

int cmd_bpe_train(Args& a) {
    const std::string in = a.str("in"), out = a.str("out");
    const int vocab = int(a.integer("vocab", 4096));
    const long max_bytes = a.integer("max-bytes", 200L << 20);
    a.done();
    CHECK(!in.empty() && !out.empty(), "usage: cxxgpt bpe-train --in text.txt --vocab N --out tok.bpe");
    CHECK(vocab >= 256 && vocab <= 65536, "--vocab must be in [256, 65536] (tokens are stored as uint16)");
    std::string text = read_file(in);
    if (long(text.size()) > max_bytes) text.resize(size_t(max_bytes));
    const double t0 = now_seconds();
    Tokenizer tok;
    tok.train(text, vocab, true);
    tok.save(out);
    const size_t n = tok.encode(text).size();
    std::printf("%d tokens (%zu merges) in %.1f s -> %s\n", tok.vocab_size(), tok.num_merges(),
                now_seconds() - t0, out.c_str());
    std::printf("%zu bytes -> %zu tokens (%.2f bytes per token)\n", text.size(), n, double(text.size()) / n);
    return 0;
}

int cmd_text2tok(Args& a) {
    const std::string in = a.str("in"), out = a.str("out"), bpe = a.str("bpe");
    a.done();
    CHECK(!in.empty() && !out.empty(), "usage: cxxgpt text2tok --in text.txt --out data.tok [--bpe tok.bpe]");
    const std::string text = read_file(in);
    TokenData d;
    if (bpe.empty()) {
        d.vocab = 256;
        d.tokens.assign(text.begin(), text.end());
        for (auto& t : d.tokens) t &= 0xFF;  // char may be signed
    } else {
        Tokenizer tok;
        tok.load(bpe);
        d.vocab = uint32_t(tok.vocab_size());
        const std::vector<int> ids = tok.encode(text);
        d.tokens.assign(ids.begin(), ids.end());
    }
    d.save(out);
    std::printf("%zu bytes -> %zu tokens (vocab %u) -> %s\n", text.size(), d.tokens.size(), d.vocab, out.c_str());
    return 0;
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    size_t pos = 0;
    for (size_t i; (i = s.find(sep, pos)) != std::string::npos; pos = i + 1) out.push_back(s.substr(pos, i - pos));
    out.push_back(s.substr(pos));
    return out;
}

std::string trim(const std::string& s) {
    const size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    return s.substr(b, s.find_last_not_of(" \t\r\n") - b + 1);
}

// The tokenizer at path, or the byte-level one (no merges) if path is empty.
Tokenizer load_tokenizer(const std::string& path) {
    Tokenizer tok;
    if (!path.empty()) tok.load(path);
    return tok;
}

// BOS and the prompt (not trained on), then the completion (trained on).
void append_problem(TokenData& d, const Problem& p, const Tokenizer& tok) {
    const std::vector<int> prompt = prompt_tokens(p, tok), completion = tok.encode(p.completion());
    for (int t : prompt) d.tokens.push_back(uint16_t(t)), d.mask.push_back(0);
    for (int t : completion) d.tokens.push_back(uint16_t(t)), d.mask.push_back(1);
}

// "a.tok,b.tok*3": the files concatenated, b three times (to weight a small
// dataset up). A file without a mask counts as all trained-on.
TokenData load_token_list(const std::string& spec) {
    std::vector<std::pair<TokenData, int>> parts;
    bool any_mask = false;
    for (const std::string& item : split(spec, ',')) {
        const size_t star = item.rfind('*');
        const int times = star == std::string::npos ? 1 : std::atoi(item.c_str() + star + 1);
        CHECK(times >= 1, "bad repeat count in '%s'", item.c_str());
        parts.emplace_back(TokenData::load(item.substr(0, star)), times);
        any_mask |= !parts.back().first.mask.empty();
    }
    if (parts.size() == 1 && parts[0].second == 1) return std::move(parts[0].first);
    TokenData d;
    for (auto& [part, times] : parts) {
        d.vocab = std::max(d.vocab, part.vocab);
        for (int r = 0; r < times; ++r) {
            d.tokens.insert(d.tokens.end(), part.tokens.begin(), part.tokens.end());
            if (!any_mask) continue;
            if (part.mask.empty()) d.mask.insert(d.mask.end(), part.tokens.size(), 1);
            else d.mask.insert(d.mask.end(), part.mask.begin(), part.mask.end());
        }
    }
    return d;
}

// Training data (masked so only answers are learned), a validation slice,
// and a held-out test set that the training data never contains.
int cmd_gen_reasoning(Args& a) {
    const std::string out = a.str("out", "data/reason");
    const long n_train = a.integer("train", 200000), n_val = a.integer("val", 2000);
    const int per_cell = int(a.integer("test-per-cell", 40));
    const int min_d = int(a.integer("min-diff", 1)), max_d = int(a.integer("max-diff", 3));
    const int test_max_d = int(a.integer("test-max-diff", max_d + 1));
    const std::vector<std::string> tasks = split(a.str("tasks", "chain,add,mul,expr,eq,cmp"), ',');
    Rng rng(uint64_t(a.integer("seed", 1)));
    const Tokenizer tok = load_tokenizer(a.str("bpe"));
    a.done();
    for (const auto& t : tasks) make_problem(t, 1, rng);  // rejects unknown names

    // Test set first: every (task, difficulty) cell, including one level
    // harder than training, to measure generalization beyond it.
    std::vector<Problem> test;
    std::set<std::string> held_out;
    for (const auto& t : tasks)
        for (int d = min_d; d <= test_max_d; ++d)
            for (int i = 0; i < per_cell;) {
                Problem p = make_problem(t, d, rng);
                if (!held_out.insert(p.question).second) continue;  // no duplicates
                test.push_back(std::move(p));
                ++i;
            }
    save_problems(out + ".test.txt", test);

    auto make_tokens = [&](long n, const std::string& path) {
        TokenData d;
        d.vocab = uint32_t(tok.vocab_size());
        long skipped = 0;
        for (long i = 0; i < n;) {
            const std::string& t = tasks[rng.below(tasks.size())];
            const Problem p = make_problem(t, min_d + int(rng.below(uint64_t(max_d - min_d + 1))), rng);
            if (held_out.count(p.question)) {
                ++skipped;
                continue;
            }
            append_problem(d, p, tok);
            ++i;
        }
        d.save(path);
        std::printf("%s: %ld problems, %zu tokens (%ld test duplicates skipped)\n", path.c_str(), n,
                    d.tokens.size(), skipped);
    };
    make_tokens(n_train, out + ".train.tok");
    make_tokens(n_val, out + ".val.tok");
    std::printf("%s.test.txt: %zu problems (%zu tasks x difficulty %d..%d x %d)\n", out.c_str(),
                test.size(), tasks.size(), min_d, test_max_d, per_cell);
    return 0;
}

// Question/answer pairs from a JSON Lines file (GSM8K's format: the answer
// is the worked solution, then "#### " and the final answer) to a problems
// file for eval-reasoning (OUT.txt) and masked training tokens (OUT.tok).
int cmd_prep_qa(Args& a) {
    const std::string in = a.str("in"), out = a.str("out"), task = a.str("task", "qa");
    const std::string q_key = a.str("question-key", "question"), a_key = a.str("answer-key", "answer");
    const bool keep_calc = a.flag("keep-calc");
    const Tokenizer tok = load_tokenizer(a.str("bpe"));
    a.done();
    CHECK(!in.empty() && !out.empty(), "usage: cxxgpt prep-qa --in FILE.jsonl --out PREFIX [--bpe tok.bpe]");
    std::ifstream f(in);
    CHECK(f, "cannot open %s", in.c_str());
    std::vector<Problem> problems;
    long line_no = 0, skipped = 0;
    for (std::string line; std::getline(f, line);) {
        ++line_no;
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        Json j;
        std::string err;
        CHECK(Json::parse(line, j, &err), "%s:%ld: %s", in.c_str(), line_no, err.c_str());
        const Json *q = j.get(q_key), *ans = j.get(a_key);
        CHECK(q && ans && q->type == Json::String && ans->type == Json::String,
              "%s:%ld: missing string fields '%s' and '%s'", in.c_str(), line_no, q_key.c_str(), a_key.c_str());
        const size_t mark = ans->str.rfind("#### ");
        if (mark == std::string::npos) {
            ++skipped;
            continue;
        }
        Problem p;
        p.task = task;
        p.question = trim(q->str);
        p.steps = trim(ans->str.substr(0, mark));
        // "1,080" -> "1080", so the exact-match check ignores digit grouping.
        for (char c : trim(ans->str.substr(mark + 5)))
            if (c != ',') p.answer += c;
        if (!keep_calc) {  // drop calculator annotations: "48/2 = <<48/2=24>>24" -> "48/2 = 24"
            std::string t;
            for (size_t i = 0; i < p.steps.size();) {
                const size_t open = p.steps.find("<<", i), close = open == std::string::npos ? open : p.steps.find(">>", open);
                if (close == std::string::npos) {
                    t += p.steps.substr(i);
                    break;
                }
                t += p.steps.substr(i, open - i);
                i = close + 2;
            }
            p.steps = t;
        }
        problems.push_back(std::move(p));
    }
    save_problems(out + ".txt", problems);
    TokenData d;
    d.vocab = uint32_t(tok.vocab_size());
    for (const Problem& p : problems) append_problem(d, p, tok);
    d.save(out + ".tok");
    std::printf("%s: %zu problems (%ld without '#### ' skipped) -> %s.txt, %s.tok (%zu tokens)\n", in.c_str(),
                problems.size(), skipped, out.c_str(), out.c_str(), d.tokens.size());
    return 0;
}

void print_eval(const EvalResult& r, bool show_failures) {
    std::map<std::string, std::map<int, std::pair<int, int>>> table;
    int min_d = 1 << 30, max_d = -(1 << 30);
    for (const auto& [key, cell] : r.by_task) {
        table[key.first][key.second] = cell;
        min_d = std::min(min_d, key.second);
        max_d = std::max(max_d, key.second);
    }
    std::printf("%-6s", "task");
    for (int d = min_d; d <= max_d; ++d) std::printf("   d=%d", d);
    std::printf("   all\n");
    for (const auto& [task, row] : table) {
        std::printf("%-6s", task.c_str());
        int c = 0, n = 0;
        for (int d = min_d; d <= max_d; ++d) {
            auto it = row.find(d);
            if (it == row.end()) {
                std::printf("      -");
                continue;
            }
            std::printf(" %5.1f%%", 100.0 * it->second.first / it->second.second);
            c += it->second.first;
            n += it->second.second;
        }
        std::printf(" %5.1f%%\n", 100.0 * c / std::max(1, n));
    }
    std::printf("overall %.1f%% (%d / %d)\n", 100.0 * r.correct / std::max(1, r.total), r.correct, r.total);
    if (show_failures)
        for (const auto& [p, text] : r.failures)
            std::printf("--- wrong (%s d=%d): %s\n    expected: %s\n    got:      %s\n", p.task.c_str(),
                        p.difficulty, p.question.c_str(), (p.steps + " #### " + p.answer).c_str(),
                        text.c_str());
}

int cmd_eval_reasoning(Args& a) {
    ThreadPool tp(threads_arg(a));
    g_pool = &tp;
    const std::string ckpt = a.str("ckpt", "model.ckpt"), test_path = a.str("test", "data/reason.test.txt");
    const int batch = int(a.integer("batch", 64)), limit = int(a.integer("limit", 0));
    const bool show = a.flag("show-failures");
    const Tokenizer tok = load_tokenizer(a.str("bpe"));
    a.done();
    Model<float> model(peek_checkpoint_config(ckpt));
    load_checkpoint(ckpt, model, nullptr, nullptr);
    std::vector<Problem> test = load_problems(test_path);
    if (limit > 0 && size_t(limit) < test.size()) {
        // An even sample across the file, which is ordered by task.
        std::vector<Problem> sub;
        for (int i = 0; i < limit; ++i) sub.push_back(test[size_t(i) * test.size() / limit]);
        test = std::move(sub);
    }
    const double t0 = now_seconds();
    const EvalResult r = eval_reasoning(model, test, tok, batch);
    print_eval(r, show);
    std::printf("(%.1f s)\n", now_seconds() - t0);
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
            "cxxgpt train --data FILE[,FILE*N...] [options]  (*N repeats a file N times)\n"
            "data:   --val FILE (default: last --val-frac of --data, 0.05)\n"
            "        --bpe tok.bpe (the tokenizer the data was made with, for --reason-test)\n"
            "model:  --seq 256 --dim 256 --layers 4 --heads 4 --ffn 0(auto)\n"
            "        --experts 0 --topk 2 --ehidden 0(auto) --shared 0 --ternary\n"
            "train:  --batch 16 --accum 1 --steps 2000 --lr 1e-3 --min-lr-frac 0.1\n"
            "        --warmup 100 --wd 0.1 --clip 1.0 --bias-rate 1e-3 --seed 1\n"
            "output: --out model.ckpt --resume --log FILE.csv --log-every 10\n"
            "        --eval-every 200 --eval-batches 20 --save-every 500\n"
            "        --time-limit MINUTES --threads N --profile\n"
            "reason: --reason-test FILE --reason-every STEPS --reason-limit 300\n"
            "Ctrl-C saves a checkpoint and stops; --resume continues from --out.\n");
        return 0;
    }
    ThreadPool tp(threads_arg(a));
    g_pool = &tp;
    const std::string data_path = a.str("data"), val_path = a.str("val");
    const std::string out = a.str("out", "model.ckpt"), log_path = a.str("log");
    CHECK(!data_path.empty(), "--data is required (see cxxgpt train --help)");
    const TokenData data = load_token_list(data_path);
    TokenData val_data;
    if (!val_path.empty()) val_data = load_token_list(val_path);
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
    const std::string reason_path = a.str("reason-test");
    const Tokenizer tok = load_tokenizer(a.str("bpe"));
    const int reason_every = int(a.integer("reason-every", 0)), reason_limit = int(a.integer("reason-limit", 300));
    g_profiler.enabled = a.flag("profile");
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

    // Optional exact-answer reasoning eval during training, on an even
    // sample of the test set.
    std::vector<Problem> reason_test;
    if (!reason_path.empty()) {
        const std::vector<Problem> all = load_problems(reason_path);
        const size_t n = std::min(all.size(), size_t(std::max(1, reason_limit)));
        for (size_t i = 0; i < n; ++i) reason_test.push_back(all[i * all.size() / n]);
    }
    auto reason_eval = [&](long at_step) {
        if (reason_test.empty()) return;
        const double t0 = now_seconds();
        const EvalResult r = eval_reasoning(model, reason_test, tok);
        std::printf("reasoning eval at step %ld (%.1f s):\n", at_step, now_seconds() - t0);
        print_eval(r, false);
        if (log) {
            std::fprintf(log, "# reasoning step %ld: %d/%d\n", at_step, r.correct, r.total);
            std::fflush(log);
        }
    };

    std::vector<int> inp(size_t(B) * Tn), tgt(size_t(B) * Tn);
    const double t_start = now_seconds();
    double loss_acc = 0, ms_acc = 0;
    int n_acc = 0;
    long step = opt.step;
    const long first_step = step;
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
        if (eval_every > 0 && ((step + 1) % eval_every == 0 || last) && val_hi >= val_lo + Tn + 1)
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
        if (reason_every > 0 && (step + 1) % reason_every == 0 && !last) reason_eval(step + 1);
        if (time_limit > 0 && now_seconds() - t_start > time_limit) {
            std::printf("time limit reached\n");
            ++step;
            break;
        }
    }
    if (g_profiler.enabled) g_profiler.report(int(step - first_step));
    reason_eval(step);
    save_checkpoint(out, model, opt, data_rng);
    std::printf("saved %s at step %ld%s\n", out.c_str(), opt.step, g_stop ? " (interrupted)" : "");
    if (log) std::fclose(log);
    return 0;
}

int cmd_sample(Args& a) {
    ThreadPool tp(threads_arg(a));
    g_pool = &tp;
    const std::string ckpt = a.str("ckpt", "model.ckpt"), prompt = a.str("prompt", "\n"), bpe = a.str("bpe");
    const int n = int(a.integer("n", 500));
    const double temp = a.real("temp", 0.8);
    Rng rng(uint64_t(a.integer("seed", 1)));
    a.done();
    Model<float> model(peek_checkpoint_config(ckpt));
    load_checkpoint(ckpt, model, nullptr, nullptr);
    Tokenizer tok;
    if (!bpe.empty()) tok.load(bpe);
    CHECK(model.cfg.vocab >= tok.vocab_size(), "the model's vocab %d is smaller than the tokenizer's %d",
          model.cfg.vocab, tok.vocab_size());
    const std::vector<int> ids = tok.encode(prompt);
    std::fwrite(prompt.data(), 1, prompt.size(), stdout);
    const int room = model.cfg.seq_len - int(ids.size());
    CHECK(room > 0, "prompt longer than the model's seq_len");
    const auto out = generate(model, {ids}, std::min(n, room), [](const std::vector<int>&) { return false; },
                              temp, &rng);
    std::vector<int> gen;
    for (int t : out[0]) gen.push_back(t < tok.vocab_size() ? t : '?');  // padding ids decode as '?'
    const std::string text = tok.decode(gen);
    std::fwrite(text.data(), 1, text.size(), stdout);
    std::printf("\n");
    return 0;
}

// Single-sequence decoding speed and weight memory: the ternary engine
// (bitplanes, int8 kernels) against the same model in fp32 (vectorized,
// same threads). A/B runs are interleaved; medians are reported.
int cmd_bench_infer(Args& a) {
    ThreadPool tp(threads_arg(a));
    g_pool = &tp;
    const std::string ckpt = a.str("ckpt");
    const int tokens = int(a.integer("tokens", 128)), reps = int(a.integer("reps", 7));
    Config cfg = config_from_args(a, 256);
    a.done();
    if (!ckpt.empty()) cfg = peek_checkpoint_config(ckpt);
    if (!cfg.ternary && !ckpt.empty())
        std::printf("note: %s was trained without --ternary; this measures speed only\n", ckpt.c_str());
    cfg.ternary = true;
    Model<float> model(cfg);
    if (ckpt.empty()) model.init(1);
    else load_checkpoint(ckpt, model, nullptr, nullptr);
    print_config(cfg, model);
    CHECK(tokens > 0 && tokens <= cfg.seq_len, "--tokens must be in [1, seq_len]");

    // fp32 baseline: the quantized weights as plain floats, no activation
    // quantization.
    model.quantize_weights();
    Config fcfg = cfg;
    fcfg.ternary = false;
    Model<float> fp32(fcfg);
    for (const ParamInfo& p : model.infos)
        std::copy_n(&(p.ternary ? model.qparams : model.params)[p.offset], p.size, &fp32.params[p.offset]);
    TernaryEngine engine(model);

    Rng rng(7);
    std::vector<int> seq(tokens);
    for (int& t : seq) t = int(rng.below(cfg.vocab));
    std::vector<double> t_fp32, t_tern;
    double agree = 0;
    for (int r = 0; r < reps; ++r) {
        double t0 = now_seconds();
        fp32.decode_begin(1, {0});
        std::vector<int> top_fp32;
        for (int pos = 0; pos < tokens; ++pos) {
            const float* lg = fp32.decode_step(&seq[pos], pos);
            top_fp32.push_back(pick_token(lg, cfg.vocab, 0.0, nullptr));
        }
        t_fp32.push_back(now_seconds() - t0);
        t0 = now_seconds();
        engine.reset();
        int same = 0;
        for (int pos = 0; pos < tokens; ++pos)
            same += pick_token(engine.step(seq[pos]), cfg.vocab, 0.0, nullptr) == top_fp32[pos];
        t_tern.push_back(now_seconds() - t0);
        agree = double(same) / tokens;
    }
    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    const double f = tokens / median(t_fp32), t = tokens / median(t_tern);
    const double fb = double(model.num_params()) * 4, tb = double(engine.weight_bytes());
    std::printf("decode, 1 sequence, %d tokens, median of %d, %d threads:\n", tokens, reps, tp.size());
    std::printf("  fp32     %9.1f tok/s   weights %8.2f MB\n", f, fb / 1e6);
    std::printf("  ternary  %9.1f tok/s   weights %8.2f MB\n", t, tb / 1e6);
    std::printf("  speedup %.2fx, memory %.1fx smaller; same argmax as fp32 on %.0f%% of positions\n",
                t / f, fb / tb, 100 * agree);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string cmd = argc > 1 ? argv[1] : "";
    Args a(argc, argv, 2);
    if (cmd == "bpe-train") return cmd_bpe_train(a);
    if (cmd == "text2tok") return cmd_text2tok(a);
    if (cmd == "train") return cmd_train(a);
    if (cmd == "sample") return cmd_sample(a);
    if (cmd == "gen-reasoning") return cmd_gen_reasoning(a);
    if (cmd == "eval-reasoning") return cmd_eval_reasoning(a);
    if (cmd == "prep-qa") return cmd_prep_qa(a);
    if (cmd == "bench-infer") return cmd_bench_infer(a);
    std::printf("usage: cxxgpt <bpe-train|text2tok|train|sample|gen-reasoning|prep-qa|eval-reasoning|bench-infer> [--options]\n");
    return cmd.empty() || cmd == "help" ? 0 : 1;
}
