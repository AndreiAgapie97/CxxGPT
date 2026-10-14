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
#include <cmath>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <set>
#include <thread>

#include "args.h"
#include "common.h"
#include "generate.h"
#ifdef CXXGPT_CUDA
#include "gpu.h"
#endif
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
    c.fp_attention = a.flag("fp-attn");
    c.nope_every = int(a.integer("nope-every", 0));
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
    std::printf("%s", !c.ternary ? "" : c.fp_attention ? ", ternary BitLinear FFN, full-precision attention"
                                                        : ", ternary BitLinear");
    if (c.nope_every == 1) std::printf(", no RoPE");
    else if (c.nope_every) std::printf(", no RoPE in 1 of every %d layers", c.nope_every);
    std::printf("\n");
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
    if (long(text.size()) > max_bytes) {
        std::printf("learning from the first %ld of %zu bytes (--max-bytes)\n", max_bytes, text.size());
        text.resize(size_t(max_bytes));
    }
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

// Text to a token file. The file streams through in blocks that are
// encoded in parallel, so it can be far bigger than memory. With --doc-sep S
// the text is a series of documents separated by S (TinyStories uses
// "<|endoftext|>"); each document is trimmed and starts with BOS.
int cmd_text2tok(Args& a) {
    const std::string in = a.str("in"), out = a.str("out"), bpe = a.str("bpe"), sep = a.str("doc-sep");
    const int threads = threads_arg(a);
    a.done();
    CHECK(!in.empty() && !out.empty(),
          "usage: cxxgpt text2tok --in text.txt --out data.tok [--bpe tok.bpe] [--doc-sep STR]");
    const Tokenizer tok = load_tokenizer(bpe);
    std::vector<Tokenizer> toks(threads, tok);  // encode() caches chunks, so one per thread
    std::ifstream f(in, std::ios::binary);
    CHECK(f, "cannot open %s", in.c_str());
    TokenWriter w(out, uint32_t(tok.vocab_size()));

    // Cuts are made where encoding in pieces matches encoding the whole:
    // right after a separator, or else at Tokenizer::next_safe_split.
    constexpr size_t kBlock = 8 << 20;
    constexpr size_t npos = std::string::npos;
    auto encode_block = [&](std::string_view blk, const Tokenizer& t, std::vector<uint16_t>& o, long& docs) {
        o.clear();
        if (sep.empty()) {
            for (int id : t.encode(blk)) o.push_back(uint16_t(id));
            return;
        }
        for (size_t p = 0; p <= blk.size();) {
            size_t q = blk.find(sep, p);
            if (q == npos) q = blk.size();
            const std::string doc = trim(std::string(blk.substr(p, q - p)));
            if (!doc.empty()) {
                o.push_back(uint16_t(kBos));
                for (int id : t.encode(doc)) o.push_back(uint16_t(id));
                ++docs;
            }
            p = q + sep.size();
        }
    };
    std::string buf;
    bool eof = false, starved = false;
    size_t bytes = 0;
    long docs = 0;
    const double t0 = now_seconds();
    double last_report = t0;
    std::vector<std::vector<uint16_t>> outs(threads);
    std::vector<long> block_docs(threads);
    while (true) {
        while (!eof && (buf.size() < kBlock * threads || starved)) {
            const size_t old = buf.size();
            buf.resize(old + kBlock);
            f.read(buf.data() + old, std::streamsize(kBlock));
            buf.resize(old + size_t(f.gcount()));
            eof = !f;
            starved = false;
        }
        const std::string_view all(buf);
        std::vector<std::string_view> blocks;
        size_t pos = 0;
        while (pos < all.size() && blocks.size() < size_t(threads)) {
            const size_t target = pos + kBlock;
            size_t end;
            if (target >= all.size()) {
                end = eof ? all.size() : npos;
            } else if (sep.empty()) {
                end = Tokenizer::next_safe_split(all, target);
                if (end == all.size() && !eof) end = npos;
            } else {
                end = all.find(sep, target);
                end = end != npos ? end + sep.size() : eof ? all.size() : npos;
            }
            if (end == npos) break;
            blocks.push_back(all.substr(pos, end - pos));
            pos = end;
        }
        if (blocks.empty()) {
            if (eof) break;
            starved = true;  // no place to cut yet: read more
            continue;
        }
        std::vector<std::thread> pool;
        for (size_t i = 0; i < blocks.size(); ++i) {
            block_docs[i] = 0;
            pool.emplace_back([&, i] { encode_block(blocks[i], toks[i], outs[i], block_docs[i]); });
        }
        for (auto& th : pool) th.join();
        for (size_t i = 0; i < blocks.size(); ++i) {
            w.write(outs[i]);
            docs += block_docs[i];
        }
        bytes += pos;
        buf.erase(0, pos);
        if (now_seconds() - last_report > 10) {
            last_report = now_seconds();
            std::printf("  %.0f MB -> %.1f M tokens (%.1f MB/s)\n", bytes / 1e6, w.count() / 1e6,
                        bytes / 1e6 / (last_report - t0));
            std::fflush(stdout);
        }
    }
    w.close();
    std::printf("%zu bytes -> %llu tokens (vocab %d", bytes, (unsigned long long)w.count(), tok.vocab_size());
    if (!sep.empty()) std::printf(", %ld documents", docs);
    std::printf(") in %.1f s -> %s\n", now_seconds() - t0, out.c_str());
    return 0;
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
    std::vector<std::pair<std::string, int>> files;
    for (const std::string& item : split(spec, ',')) {
        const size_t star = item.rfind('*');
        const int times = star == std::string::npos ? 1 : std::atoi(item.c_str() + star + 1);
        CHECK(times >= 1, "bad repeat count in '%s'", item.c_str());
        files.emplace_back(item.substr(0, star), times);
    }
    return TokenData::load_list(files);
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
    Formats formats;
    formats.chain_plan = a.flag("chain-plan");      // chain solutions look variables up first
    formats.expr_rewrite = a.flag("expr-rewrite");  // expr solutions write what is left after each step
    Rng rng(uint64_t(a.integer("seed", 1)));
    const std::string bpe = a.str("bpe");
    a.done();
    const Tokenizer tok = load_tokenizer(bpe);
    for (const auto& t : tasks) make_problem(t, 1, rng);  // rejects unknown names

    // Test set first: every (task, difficulty) cell, including one level
    // harder than training, to measure generalization beyond it.
    std::vector<Problem> test;
    std::set<std::string> held_out;
    for (const auto& t : tasks)
        for (int d = min_d; d <= test_max_d; ++d)
            for (int i = 0; i < per_cell;) {
                Problem p = make_problem(t, d, rng, formats);
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
            const Problem p =
                make_problem(t, min_d + int(rng.below(uint64_t(max_d - min_d + 1))), rng, formats);
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

// "1,080" -> "1080": commas that group digits go, so the exact-match check
// ignores digit grouping; other commas, as in "(1, 2)", stay.
std::string strip_digit_commas(const std::string& s) {
    auto digit = [&](size_t i) { return i < s.size() && s[i] >= '0' && s[i] <= '9'; };
    std::string o;
    for (size_t i = 0; i < s.size(); ++i) {
        const bool grouping = s[i] == ',' && i > 0 && digit(i - 1) && digit(i + 1) && digit(i + 2) &&
                              digit(i + 3) && !digit(i + 4);
        if (!grouping) o += s[i];
    }
    return o;
}

// A JSON number as an answer: 18 -> "18", 2.5 -> "2.5".
std::string format_number(double v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.15g", v);
    return buf;
}

// Question/answer pairs from a JSON Lines file to a problems file for
// eval-reasoning (OUT.txt) and masked training tokens (OUT.tok). By default
// the answer is GSM8K's: the worked solution, then "#### " and the final
// answer. With --final-key, the final answer is a field of its own and the
// whole answer field is the solution.
int cmd_prep_qa(Args& a) {
    const std::string in = a.str("in"), out = a.str("out"), task = a.str("task", "qa");
    const std::string q_key = a.str("question-key", "question"), a_key = a.str("answer-key", "answer");
    const std::string final_key = a.str("final-key");
    const bool keep_calc = a.flag("keep-calc");
    const std::string bpe = a.str("bpe");
    a.done();
    const Tokenizer tok = load_tokenizer(bpe);
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
        Problem p;
        p.task = task;
        p.question = trim(q->str);
        if (final_key.empty()) {
            const size_t mark = ans->str.rfind("#### ");
            if (mark == std::string::npos) {
                ++skipped;
                continue;
            }
            p.steps = trim(ans->str.substr(0, mark));
            p.answer = strip_digit_commas(trim(ans->str.substr(mark + 5)));
        } else {
            const Json* fin = j.get(final_key);
            CHECK(fin, "%s:%ld: missing field '%s'", in.c_str(), line_no, final_key.c_str());
            p.steps = trim(ans->str);
            if (fin->type == Json::String) p.answer = strip_digit_commas(trim(fin->str));
            else if (fin->type == Json::Number) p.answer = format_number(fin->number);
        }
        // One line only: the format ends the answer at the first newline.
        if (p.answer.empty() || p.answer.find('\n') != std::string::npos || p.question.empty()) {
            ++skipped;
            continue;
        }
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
    std::printf("%s: %zu problems (%ld without a usable answer skipped) -> %s.txt, %s.tok (%zu tokens)\n", in.c_str(),
                problems.size(), skipped, out.c_str(), out.c_str(), d.tokens.size());
    return 0;
}

// The table of accuracy per task and difficulty, then up to show_failures
// of the wrong answers.
void print_eval(const EvalResult& r, size_t show_failures) {
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
    // Multi-line answers (GSM8K's) stay indented under their label.
    auto indent = [](std::string t) {
        for (size_t i = 0; (i = t.find('\n', i)) != std::string::npos; i += 15) t.replace(i, 1, "\n              ");
        return t;
    };
    for (size_t i = 0; i < std::min(show_failures, r.failures.size()); ++i) {
        const auto& [p, text] = r.failures[i];
        std::printf("--- wrong (%s d=%d): %s\n    expected: %s\n    got:      %s\n", p.task.c_str(), p.difficulty,
                    p.question.c_str(), indent(p.steps + " #### " + p.answer).c_str(), indent(text).c_str());
    }
}

int cmd_eval_reasoning(Args& a) {
    ThreadPool tp(threads_arg(a));
    g_pool = &tp;
    const std::string ckpt = a.str("ckpt", "model.ckpt"), test_path = a.str("test", "data/reason.test.txt");
    const int batch = int(a.integer("batch", 64)), limit = int(a.integer("limit", 0));
    const bool show = a.flag("show-failures");
    // As TernaryEngine runs it: the embedding (and --fp-attn attention) in
    // int8 rows.
    const bool as_engine = a.flag("as-engine");
    const std::string bpe = a.str("bpe");
    a.done();
    const Tokenizer tok = load_tokenizer(bpe);
    Model<float> model(peek_checkpoint_config(ckpt));
    load_checkpoint(ckpt, model, nullptr, nullptr);
    if (as_engine) round_to_engine_int8(model);
    std::vector<Problem> test = load_problems(test_path);
    if (limit > 0 && size_t(limit) < test.size()) {
        // An even sample across the file, which is ordered by task.
        std::vector<Problem> sub;
        for (int i = 0; i < limit; ++i) sub.push_back(test[size_t(i) * test.size() / limit]);
        test = std::move(sub);
    }
    const double t0 = now_seconds();
    const EvalResult r = eval_reasoning(model, test, tok, batch);
    print_eval(r, show ? r.failures.size() : 0);
    std::printf("(%.1f s)\n", now_seconds() - t0);
    return 0;
}

// One training step's operations, on the CPU or the GPU.
struct Backend {
    virtual ~Backend() = default;
    virtual double forward(const int* inp, const int* tgt, int B, int Tn, bool training) = 0;
    virtual void backward(double loss_scale) = 0;
    virtual void zero_grad() = 0;
    // Clips the gradient to max_norm (if > 0), then takes an AdamW step;
    // a gradient that is not finite (overflow, NaN) is skipped instead.
    // Returns the gradient norm before clipping.
    virtual double step(double lr, double max_norm) = 0;
    // Updates the router biases from the expert loads since the last call;
    // returns the busiest expert's load ratio (see Model::max_load_ratio).
    virtual double balance(double rate) = 0;
    // Brings the weights and optimizer state into the host Model and AdamW,
    // for checkpoints and evaluations.
    virtual void sync() = 0;
    // Device memory held, for the GPU; 0 on the CPU.
    virtual size_t device_bytes() const { return 0; }
};

struct CpuBackend : Backend {
    Model<float>& model;
    AdamW& opt;
    CpuBackend(Model<float>& m, AdamW& o) : model(m), opt(o) {}
    double forward(const int* inp, const int* tgt, int B, int Tn, bool training) override {
        return model.forward(inp, tgt, B, Tn, training);
    }
    void backward(double loss_scale) override { model.backward(loss_scale); }
    void zero_grad() override { model.zero_grad(); }
    double step(double lr, double max_norm) override {
        const double norm = clip_grad_norm(model, max_norm);
        if (std::isfinite(norm)) opt.update(model, lr);
        return norm;
    }
    double balance(double rate) override {
        const double load = model.max_load_ratio();
        model.update_router_bias(rate);
        return load;
    }
    void sync() override {}
};

#ifdef CXXGPT_CUDA
struct GpuBackend : Backend {
    Model<float>& model;
    AdamW& opt;
    GpuTrainer gpu;
    GpuBackend(Model<float>& m, AdamW& o, Matmul matmul, bool fused_attn)
        : model(m), opt(o), gpu(m.cfg, m.infos, matmul, fused_attn) {
        gpu.set_params(model.params.data());
        const bool has_state = opt.m.size() == model.params.size();
        gpu.set_adam_state(opt.step, has_state ? opt.m.data() : nullptr, has_state ? opt.v.data() : nullptr);
        gpu.set_router_bias(model.router_bias);
        gpu.set_muon(opt.muon, opt.muon_momentum, opt.muon_steps);
        gpu.set_profile(g_profiler.enabled);
    }
    double forward(const int* inp, const int* tgt, int B, int Tn, bool training) override {
        return gpu.forward(inp, tgt, B, Tn, training);
    }
    void backward(double loss_scale) override { gpu.backward(loss_scale); }
    void zero_grad() override { gpu.zero_grad(); }
    double step(double lr, double max_norm) override {
        const double norm = gpu.grad_norm();
        if (!std::isfinite(norm)) return norm;
        const double scale = max_norm > 0 && norm > max_norm ? double(float(max_norm / norm)) : 1.0;
        gpu.adamw_step(lr, opt.beta1, opt.beta2, opt.eps, opt.weight_decay, scale);
        ++opt.step;
        // GPU time per part of the step (evaluation passes included).
        if (g_profiler.enabled)
            for (const auto& [name, seconds] : gpu.take_profile()) g_profiler.add(name.c_str(), seconds);
        return norm;
    }
    double balance(double rate) override {
        const auto loads = gpu.take_expert_load();
        for (size_t l = 0; l < loads.size(); ++l)
            for (size_t e = 0; e < loads[l].size(); ++e) model.expert_load[l][e] += loads[l][e];
        const double load = model.max_load_ratio();
        model.update_router_bias(rate);
        gpu.set_router_bias(model.router_bias);
        return load;
    }
    void sync() override {
        gpu.get_params(model.params.data());
        model.weights_dirty = true;
        opt.m.resize(model.params.size());
        opt.v.resize(model.params.size());
        gpu.get_adam_state(&opt.step, opt.m.data(), opt.v.data());
    }
    size_t device_bytes() const override { return gpu.device_bytes(); }
};
#endif

double eval_loss(Backend& be, const TokenData& data, size_t lo, size_t hi, int B, int Tn, int batches) {
    Rng rng(12345);  // the same batches every time, so evals are comparable
    std::vector<int> inp(size_t(B) * Tn), tgt(size_t(B) * Tn);
    double total = 0.0;
    for (int i = 0; i < batches; ++i) {
        data.sample_batch(rng, B, Tn, lo, hi, inp.data(), tgt.data());
        total += be.forward(inp.data(), tgt.data(), B, Tn, false);
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
            "        --fp-attn (with --ternary: attention stays full precision)\n"
            "        --nope-every 0 (N: every Nth layer has no position encoding; 1: none has)\n"
            "train:  --batch 16 --accum 1 --steps 2000 --lr 1e-3 --min-lr-frac 0.1\n"
            "        --warmup 100 --wd 0.1 --clip 1.0 --bias-rate 1e-3 --seed 1\n"
            "        --muon (Muon instead of AdamW for the matrices in the blocks)\n"
            "start:  --init FILE (FILE's weights and model; new optimizer, schedule and data order;\n"
            "        with --ternary, the weights turn ternary from here on)\n"
            "output: --out model.ckpt --resume --log FILE.csv --log-every 10\n"
            "        --eval-every 200 --eval-batches 20 --save-every 500\n"
            "        --time-limit MINUTES --threads N --profile\n"
            "gpu:    --gpu (train on the GPU; needs cxxgpt-gpu, from make gpu)\n"
            "        --matmul tf32 (tensor cores; fp32: exact, slower; bf16: coarser, maybe faster)\n"
            "        --no-fused-attn (attention through cuBLAS and the full matrix of weights: more memory)\n"
            "reason: --reason-test FILE --reason-every STEPS --reason-limit 300\n"
            "Ctrl-C saves a checkpoint and stops; --resume continues from --out, and\n"
            "takes precedence over --init once --out exists.\n");
        return 0;
    }
    ThreadPool tp(threads_arg(a));
    g_pool = &tp;
    const std::string data_path = a.str("data"), val_path = a.str("val");
    const std::string out = a.str("out", "model.ckpt"), log_path = a.str("log"), init = a.str("init");
    CHECK(!data_path.empty(), "--data is required (see cxxgpt train --help)");
    const TokenData data = load_token_list(data_path);
    TokenData val_data;
    if (!val_path.empty()) val_data = load_token_list(val_path);
    const double val_frac = a.real("val-frac", 0.05);
    const bool resume = a.flag("resume") && std::filesystem::exists(out);

    Config cfg = config_from_args(a, int(data.vocab));
    if (resume) {
        cfg = peek_checkpoint_config(out);
    } else if (!init.empty()) {
        // The model comes from the checkpoint, but --ternary can turn its
        // weights ternary from here on: the same weights, used differently
        // (and --fp-attn keeps attention in full precision).
        const bool to_ternary = cfg.ternary, fp_attn = cfg.fp_attention;
        cfg = peek_checkpoint_config(init);
        cfg.ternary = cfg.ternary || to_ternary;
        cfg.fp_attention = cfg.fp_attention || fp_attn;
    }
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
    const bool use_gpu = a.flag("gpu"), fused_attn = !a.flag("no-fused-attn");
    const std::string matmul = a.str("matmul", "tf32");
    CHECK(matmul == "fp32" || matmul == "tf32" || matmul == "bf16", "--matmul takes fp32, tf32 or bf16");
    AdamW opt;
    opt.weight_decay = a.real("wd", 0.1);
    opt.muon = a.flag("muon");
    a.done();
    CHECK(cfg.vocab >= int(data.vocab), "model vocab %d < data vocab %u", cfg.vocab, data.vocab);

    Model<float> model(cfg);
    model.init(uint64_t(seed));
    Rng data_rng(uint64_t(seed) + 1);
    if (resume) {
        load_checkpoint(out, model, &opt, &data_rng);
        std::printf("resumed %s at step %ld\n", out.c_str(), opt.step);
    } else if (!init.empty()) {
        load_checkpoint(init, model, nullptr, nullptr);
        std::printf("starting from the weights in %s\n", init.c_str());
    }
    print_config(cfg, model);
    std::unique_ptr<Backend> backend;
    if (use_gpu) {
#ifdef CXXGPT_CUDA
        const Matmul mm = matmul == "fp32" ? Matmul::fp32 : matmul == "bf16" ? Matmul::bf16 : Matmul::tf32;
        backend = std::make_unique<GpuBackend>(model, opt, mm, fused_attn);
        std::printf("gpu: %s, %s matmuls, %s attention\n", GpuTrainer::device_name().c_str(), matmul.c_str(),
                    fused_attn ? "fused" : "cuBLAS");
#else
        (void)fused_attn;
        CHECK(false, "this cxxgpt was built without CUDA; use cxxgpt-gpu (make gpu)");
#endif
    } else {
        backend = std::make_unique<CpuBackend>(model, opt);
    }
    Backend& be = *backend;

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
        log = create_file(log_path, "a");
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
        be.sync();
        const double t0 = now_seconds();
        const EvalResult r = eval_reasoning(model, reason_test, tok);
        std::printf("reasoning eval at step %ld (%.1f s):\n", at_step, now_seconds() - t0);
        print_eval(r, 1);  // one wrong answer, to see what the model writes
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
    constexpr int kMaxBadSteps = 10;
    int bad_steps = 0;
    bool diverged = false;
    long saved_step = resume ? first_step : -1;  // what --out holds
    for (; step < steps && !g_stop; ++step) {
        const double t0 = now_seconds();
        be.zero_grad();
        double loss = 0.0;
        for (int micro = 0; micro < accum; ++micro) {
            data.sample_batch(data_rng, B, Tn, 0, train_end, inp.data(), tgt.data());
            loss += be.forward(inp.data(), tgt.data(), B, Tn, true) / accum;
            be.backward(1.0 / accum);
        }
        const double lr_now = lr_at(step, warmup, steps, lr, min_frac);
        const double gnorm = be.step(lr_now, clip);
        // A non-finite gradient leaves the weights untouched (Backend::step). A run
        // of them stops training rather than spending the night on NaNs.
        if (!std::isfinite(gnorm) || !std::isfinite(loss)) {
            std::printf("step %ld: loss %g, gradient norm %g; update skipped\n", step + 1, loss, gnorm);
            if (++bad_steps == kMaxBadSteps) {
                std::printf("stopping: %d diverged steps in a row\n", kMaxBadSteps);
                diverged = true;
                break;
            }
            continue;
        }
        bad_steps = 0;
        const double load = be.balance(bias_rate);
        const double ms = (now_seconds() - t0) * 1e3;
        if (step == first_step && be.device_bytes() > 0)
            std::printf("gpu memory: %.0f MB after the first step\n", double(be.device_bytes()) / (1 << 20));
        loss_acc += loss;
        ms_acc += ms;
        ++n_acc;

        const bool last = step + 1 == steps;
        double vloss = -1;
        if (eval_every > 0 && ((step + 1) % eval_every == 0 || last) && val_hi > val_lo + Tn + 1)
            vloss = eval_loss(be, *val, val_lo, val_hi, B, Tn, eval_batches);
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
        if (save_every > 0 && (step + 1) % save_every == 0) {
            be.sync();
            save_checkpoint(out, model, opt, data_rng);
            saved_step = opt.step;
        }
        if (reason_every > 0 && (step + 1) % reason_every == 0 && !last) reason_eval(step + 1);
        if (time_limit > 0 && now_seconds() - t_start > time_limit) {
            std::printf("time limit reached\n");
            ++step;
            break;
        }
    }
    if (g_profiler.enabled) g_profiler.report(int(step - first_step));
    if (diverged) {
        // The weights are finite, but whatever overflowed is in them by now:
        // the last checkpoint is the one to keep.
        if (saved_step >= 0)
            std::printf("not saved: %s keeps step %ld; resume it with a lower --lr\n", out.c_str(), saved_step);
        else
            std::printf("not saved: no checkpoint yet; start again with a lower --lr\n");
        if (log) std::fclose(log);
        return 1;
    }
    reason_eval(step);
    be.sync();
    save_checkpoint(out, model, opt, data_rng);
    std::printf("saved %s at step %ld%s\n", out.c_str(), opt.step, g_stop ? " (interrupted)" : "");
    if (log) std::fclose(log);
    return 0;
}

int cmd_sample(Args& a) {
    ThreadPool tp(threads_arg(a));
    g_pool = &tp;
    const std::string ckpt = a.str("ckpt", "model.ckpt"), prompt = a.str("prompt"), bpe = a.str("bpe");
    const int n = int(a.integer("n", 500));
    const double temp = a.real("temp", 0.8);
    const bool answer = a.flag("answer");  // stop after the "#### answer" line, as eval-reasoning does
    Rng rng(uint64_t(a.integer("seed", 1)));
    a.done();
    Model<float> model(peek_checkpoint_config(ckpt));
    load_checkpoint(ckpt, model, nullptr, nullptr);
    Tokenizer tok;
    if (!bpe.empty()) tok.load(bpe);
    CHECK(model.cfg.vocab >= tok.vocab_size(), "the model's vocab %d is smaller than the tokenizer's %d",
          model.cfg.vocab, tok.vocab_size());
    // Every training document starts with BOS, so the prompt does too: an
    // empty prompt starts a new document.
    std::vector<int> ids = tok.encode(prompt);
    ids.insert(ids.begin(), kBos);
    CHECK(n > 0, "--n must be at least 1");
    CHECK(int(ids.size()) < model.cfg.seq_len, "the prompt is %zu tokens with the BOS in front; the model's context "
          "holds %d", ids.size(), model.cfg.seq_len);
    std::fwrite(prompt.data(), 1, prompt.size(), stdout);
    // Padding ids (model vocab beyond the tokenizer's) decode as '?'.
    auto text_of = [&](const std::vector<int>& g) {
        std::vector<int> ok;
        for (int t : g) ok.push_back(t < tok.vocab_size() ? t : '?');
        return tok.decode(ok);
    };
    // A BOS would start the next document: this one is over.
    const auto out = generate(
        model, {ids}, n,
        [&](const std::vector<int>& g) { return g.back() == kBos || (answer && completion_done(text_of(g))); },
        temp, &rng);
    std::vector<int> gen = out[0];
    if (!gen.empty() && gen.back() == kBos) gen.pop_back();
    const std::string text = text_of(gen);
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
    int tokens = int(a.integer("tokens", 128));
    const bool tokens_given = a.has("tokens");
    const int reps = int(a.integer("reps", 7));
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
    if (!tokens_given) tokens = std::min(tokens, cfg.seq_len);  // a shorter context: all of it
    CHECK(tokens > 0 && tokens <= cfg.seq_len, "--tokens must be in [1, seq_len]");

    // fp32 baseline: the quantized weights as plain floats, no activation
    // quantization.
    model.quantize_weights();
    Config fcfg = cfg;
    fcfg.ternary = fcfg.fp_attention = false;
    Model<float> fp32(fcfg);
    for (const ParamInfo& p : model.infos)
        std::copy_n(&(p.ternary ? model.qparams : model.params)[p.offset], p.size, &fp32.params[p.offset]);
    TernaryEngine engine(model);

    Rng rng(7);
    std::vector<int> seq(tokens);
    for (int& t : seq) t = int(rng.below(cfg.vocab));
    // What the model computes as trained (ternary weights and int8
    // activations, in fp32 arithmetic): the engine should pick the same
    // next token everywhere. The fp32 baseline skips the int8 rounding, so
    // where the top two tokens are close (random input makes them close
    // often) it can pick the other one.
    std::vector<int> top_model;
    model.decode_begin(1, {0});
    for (int pos = 0; pos < tokens; ++pos)
        top_model.push_back(pick_token(model.decode_step(&seq[pos], pos), cfg.vocab, 0.0, nullptr));
    std::vector<double> t_fp32, t_tern;
    double agree = 0, agree_model = 0;
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
        std::vector<int> top_tern;
        for (int pos = 0; pos < tokens; ++pos)
            top_tern.push_back(pick_token(engine.step(seq[pos]), cfg.vocab, 0.0, nullptr));
        t_tern.push_back(now_seconds() - t0);
        int same = 0, same_model = 0;
        for (int pos = 0; pos < tokens; ++pos) {
            same += top_tern[pos] == top_fp32[pos];
            same_model += top_tern[pos] == top_model[pos];
        }
        agree = double(same) / tokens;
        agree_model = double(same_model) / tokens;
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
    std::printf("  speedup %.2fx, memory %.1fx smaller\n", t / f, fb / tb);
    std::printf("  same next token as the model as trained on %.0f%% of positions (as fp32: %.0f%%)\n",
                100 * agree_model, 100 * agree);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    struct Command {
        const char* name;
        int (*run)(Args&);
        const char* what;
    };
    static const Command commands[] = {
        {"bpe-train", cmd_bpe_train, "learns a BPE tokenizer from a text file"},
        {"text2tok", cmd_text2tok, "encodes text into a token file"},
        {"train", cmd_train, "trains a model, or resumes training"},
        {"sample", cmd_sample, "generates text from a prompt"},
        {"gen-reasoning", cmd_gen_reasoning, "writes the synthetic reasoning curriculum"},
        {"prep-qa", cmd_prep_qa, "turns question/answer JSON Lines into problems and token files"},
        {"eval-reasoning", cmd_eval_reasoning, "exact-answer accuracy on a problems file"},
        {"bench-infer", cmd_bench_infer, "times the ternary inference engine against fp32 decoding"},
    };
    const std::string cmd = argc > 1 ? argv[1] : "";
    Args a(argc, argv, 2);
    for (const Command& c : commands)
        if (cmd == c.name) {
            if (a.help()) std::printf("cxxgpt %s: %s\n", c.name, c.what);
            return c.run(a);
        }
    std::printf("usage: cxxgpt <command> [--options]; cxxgpt <command> --help lists its options\n");
    for (const Command& c : commands) std::printf("  %-15s %s\n", c.name, c.what);
    return cmd.empty() || cmd == "help" || cmd == "--help" ? 0 : 1;
}
