// cxxgpt — train and run the model.
//
//   cxxgpt bpe-train --in text.txt --vocab 4096 --out tok.bpe   learn a BPE tokenizer
//   cxxgpt text2tok --in text.txt --out data.tok [--bpe tok.bpe]  text -> token file
//   cxxgpt train --data data.tok [options]                train (or resume)
//   cxxgpt sample --ckpt model.ckpt [--prompt TEXT] [--bpe tok.bpe]  generate text
//   cxxgpt gen-reasoning --out data/reason                 synthetic reasoning data
//   cxxgpt prep-qa --in gsm8k.jsonl --out data/gsm8k/train  Q/A JSON Lines -> problems + tokens
//   cxxgpt eval-reasoning --ckpt model.ckpt --test FILE    exact-answer accuracy
//       [--tries N | --effort low/medium/high] [--temp T] [--seed S]
//                                      sample N tries per problem; the headline becomes their maj@N vote
//       [--no-repeat N]                never write a run of N tokens twice (also on sample)
//       [--labels FILE.jsonl]          why each first try is wrong, one JSON line per problem
//   cxxgpt self-train --ckpt model.ckpt --problems FILE --out PREFIX  the model's own checked solutions
//       [--tries 8] [--temp 0.7] [--keep K] -> PREFIX.txt, PREFIX.tok (masked), PREFIX.json
//   cxxgpt eval-loss --ckpt model.ckpt --data a.tok,b.tok [--all]  loss on held-out text
//   cxxgpt qa-context --data a.tok,b.tok [--seq 512]   answers trained without their question
//   cxxgpt bench-infer [--ckpt model.ckpt | model options]  ternary engine vs fp32 decode
//
// Run a subcommand with --help for its options.
#include <charconv>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <set>
#include <thread>
#include <utility>

#include "args.h"
#include "clone_experts.h"
#include "failure_labels.h"
#include "common.h"
#include "generate.h"
#ifdef CXXGPT_CUDA
#include "gpu.h"
#endif
#include "infer.h"
#include "json.h"
#include "leak_check.h"
#include "load_log.h"
#include "loss_mask.h"
#include "model.h"
#include "profile.h"
#include "qa_context.h"
#include "reasoning.h"
#include "reason_report.h"
#include "self_train.h"
#include "tokenizer.h"
#include "train.h"
#include "train_control.h"
#include "val_sources.h"
#include "wordgen.h"
#include "mathgen.h"
#include "mathgen_fractions.h"

namespace {

volatile std::sig_atomic_t g_stop = 0;
void on_signal(int) { g_stop = 1; }

int threads_arg(Args& a) {
    const int threads = a.integer_int("threads", long(std::max(1u, std::thread::hardware_concurrency())));
    CHECK(threads > 0, "--threads must be positive");
    return threads;
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    CHECK(in, "cannot open %s", path.c_str());
    return std::string(std::istreambuf_iterator<char>(in), {});
}

// Model options, shared by every command that builds a fresh model.
Config config_from_args(Args& a, int vocab) {
    Config c;
    c.vocab = a.integer_int("vocab", vocab);
    c.seq_len = a.integer_int("seq", 256);
    c.d_model = a.integer_int("dim", 256);
    c.n_layers = a.integer_int("layers", 4);
    c.n_heads = a.integer_int("heads", 4);
    c.ffn_hidden = a.integer_int("ffn", 0);
    c.n_experts = a.integer_int("experts", 0);
    c.top_k = a.integer_int("topk", 2);
    c.expert_hidden = a.integer_int("ehidden", 0);
    c.n_shared = a.integer_int("shared", 0);
    c.ternary = a.flag("ternary");
    c.fp_attention = a.flag("fp-attn");
    c.nope_every = a.integer_int("nope-every", 0);
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

int cmd_clone_experts(Args& a) {
    const std::string in = a.str("ckpt"), out = a.str("out");
    // --auto: clone into every expert under --below percent of its layer,
    // averaged over the last --window rows of the run's --load-log.
    const bool automatic = a.flag("auto");
    const std::string load_log = a.str("load-log");
    const bool has_below = a.has("below"), has_window = a.has("window");
    const double below = a.real("below", 1.0);
    const int window = a.integer_int("window", 10);
    const bool has_layer = a.has("layer"), has_from = a.has("from"), has_to = a.has("to");
    const int layer = a.integer_int("layer", -1), from = a.integer_int("from", -1), to = a.integer_int("to", -1);
    const double noise = a.real("noise", 0.01);
    const long seed = a.integer("seed", 1);
    a.done();
    CHECK(!in.empty() && !out.empty(),
          "usage: cxxgpt clone-experts --ckpt IN --out OUT --auto --load-log RUN.load.csv | --layer L --from A --to B");
    CHECK(automatic ? !(has_layer || has_from || has_to) : has_layer && has_from && has_to,
          "choose either --auto or all of --layer, --from, --to");
    CHECK(automatic == !load_log.empty(), "--auto needs --load-log (and --load-log needs --auto)");
    CHECK(automatic || !(has_below || has_window), "--below and --window go with --auto");
    CHECK(below > 0 && below < 100, "--below is a percentage between 0 and 100");
    CHECK(noise >= 0 && seed >= 0, "--noise and --seed must be nonnegative");
    Model<float> model(peek_checkpoint_config(in));
    CHECK(model.cfg.moe(), "clone-experts needs an MoE checkpoint");
    AdamW opt; Rng data_rng;
    load_checkpoint(in, model, &opt, &data_rng, true);
    std::vector<ExpertClone> edits{{layer, from, to}};
    if (automatic) {
        const auto shares = recent_expert_shares(load_log, model.cfg, window);
        edits = starved_expert_clones(shares, below / 100);
        std::printf("expert shares averaged over each layer's last %d rows of %s\n", window, load_log.c_str());
        for (int l = 0; l < model.cfg.n_layers; ++l) {
            const auto& s = shares[l];
            const int low = int(std::min_element(s.begin(), s.end()) - s.begin());
            const int high = int(std::max_element(s.begin(), s.end()) - s.begin());
            std::printf("layer %d: smallest expert %d (%.2f%%), busiest expert %d (%.1f%%)%s\n", l, low,
                        s[low] * 100, high, s[high] * 100, s[low] * 100 < below ? "" : ": no clone");
        }
    }
    repair_experts(model, opt, edits, noise, uint64_t(seed), true);
    if (edits.empty()) std::printf("no experts cloned: none is under %g%% of its layer\n", below);
    save_checkpoint(out, model, opt, data_rng);
    std::printf("%zu expert(s) cloned -> %s (optimizer step %ld preserved)\n", edits.size(), out.c_str(), opt.step);
    return 0;
}

int cmd_bpe_train(Args& a) {
    const std::string in = a.str("in"), out = a.str("out");
    const int vocab = a.integer_int("vocab", 4096);
    const int64_t max_bytes = a.integer("max-bytes", int64_t(200) << 20);
    a.done();
    CHECK(!in.empty() && !out.empty(), "usage: cxxgpt bpe-train --in text.txt --vocab N --out tok.bpe");
    CHECK(vocab >= 256 && vocab <= 65536, "--vocab must be in [256, 65536] (tokens are stored as uint16)");
    CHECK(max_bytes > 0, "--max-bytes must be positive");
    std::string text = read_file(in);
    // Compared as sizes: a file over 2 GB must not wrap to a negative number.
    if (text.size() > size_t(max_bytes)) {
        std::printf("learning from the first %lld of %zu bytes (--max-bytes)\n", (long long)max_bytes, text.size());
        text.resize(size_t(max_bytes));
    }
    const double t0 = now_seconds();
    Tokenizer tok;
    tok.train(text, vocab, true);
    if (tok.long_chunks() > 0)
        std::printf("warning: %zu chunks longer than %zu bytes were left out of the merges\n", tok.long_chunks(),
                    Tokenizer::kMaxChunkBytes);
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

// At most max_bytes of text, cut so that no UTF-8 character is split: the
// bytes that continue a character look like 10xxxxxx in binary.
std::string utf8_prefix(const std::string& text, size_t max_bytes) {
    size_t cut = std::min(text.size(), max_bytes);
    while (cut > 0 && cut < text.size() && (static_cast<unsigned char>(text[cut]) & 0xC0) == 0x80) --cut;
    return text.substr(0, cut);
}

int cmd_leak_check(Args& a) {
    const std::string questions = a.str("questions"), train = a.str("train");
    a.done();
    CHECK(!questions.empty() && !train.empty(),
          "usage: cxxgpt leak-check --questions QUESTIONS.txt --train A.txt,B.txt,...\n"
          "Questions: prep-qa problem records or one question per line.\n"
          "Finds exact runs of 13 ASCII letter/digit words, ignoring case.");
    const leak_check::Index index(leak_check::read_questions(questions));
    for (const std::string& path : split(train, ',')) {
        std::ifstream in(path, std::ios::binary);
        CHECK(in, "cannot open %s", path.c_str());
        const auto hits = index.scan(in);
        const size_t count = size_t(std::count(hits.begin(), hits.end(), true));
        std::printf("%s: %zu / %zu questions share a 13-word run\n", path.c_str(), count, hits.size());
        size_t shown = 0;
        for (size_t q = 0; q < hits.size() && shown < 3; ++q) {
            if (!hits[q]) continue;
            std::string example = utf8_prefix(index.questions[q], 240);
            const bool cut = example.size() < index.questions[q].size();
            for (char& ch : example) if (ch == '\n' || ch == '\r' || ch == '\t') ch = ' ';
            std::printf("  question %zu: %s%s\n", q + 1, example.c_str(), cut ? "..." : "");
            ++shown;
        }
    }
    return 0;
}

// How often a training window trains an answer without its whole question
// in view (qa_context.h, issue #15), for each token file, with today's
// random windows and with train --sampler qa.
int cmd_qa_context(Args& a) {
    const std::string data = a.str("data");
    const int seq = a.integer_int("seq", 512);
    a.done();
    CHECK(!data.empty() && seq > 0,
          "usage: cxxgpt qa-context --data A.tok,B.tok,... [--seq 512]\n"
          "For every trained (masked-in) token: the share of training windows of --seq tokens\n"
          "that show its whole question, part of it, or none of it, for --sampler random and qa.");
    for (const std::string& path : split(data, ',')) {
        const TokenData d = TokenData::load(path);
        if (d.mask.empty()) {
            std::printf("%s: no mask (every token is trained): no questions to lose\n", path.c_str());
            continue;
        }
        const QaStarts starts(d, seq, kBos);
        const QaContext random = qa_context(d, seq), qa = qa_context(d, seq, &starts);
        std::printf("%s: %llu documents (mean %.0f tokens, %llu longer than %d), %llu trained tokens; "
                    "windows of %d\n",
                    path.c_str(), static_cast<unsigned long long>(random.documents),
                    double(random.doc_tokens) / double(std::max<uint64_t>(1, random.documents)),
                    static_cast<unsigned long long>(random.too_long), seq + 1,
                    static_cast<unsigned long long>(random.trained), seq);
        auto line = [](const char* name, const QaContext& q) {
            auto pct = [&](uint64_t part) { return 100.0 * double(part) / double(std::max<uint64_t>(1, q.pairs)); };
            std::printf("  %-7s whole question in view %.1f%%, part of it %.1f%%, none of it %.1f%%; "
                        "%.1f trained targets per window, %llu trained tokens in no window\n",
                        name, pct(q.whole), pct(q.partial), pct(q.none),
                        double(q.pairs) / double(std::max<uint64_t>(1, q.windows)),
                        static_cast<unsigned long long>(q.unreached));
        };
        line("random:", random);
        line("qa:", qa);
    }
    return 0;
}

// The tokenizer at path, or the byte-level one (no merges) if path is empty.
Tokenizer load_tokenizer(const std::string& path) {
    Tokenizer tok;
    if (!path.empty()) tok.load(path);
    return tok;
}

// Text to a token file. The file streams through in blocks that are
// encoded in parallel, so it can be far bigger than memory. With --doc-sep S
// the text is a series of documents separated by S (the mix's text files
// use "<|endoftext|>"); each document is trimmed and starts with BOS.
// With --loss-markers, text between [[no-loss]] and [[/no-loss]] (planted
// mistakes, loss_mask.h) is kept as context but not trained on, and the
// markers are removed. Without it, the markers are ordinary text: web text
// may contain them by chance, and must not fail or change meaning.
int cmd_text2tok(Args& a) {
    const std::string in = a.str("in"), out = a.str("out"), bpe = a.str("bpe"), sep = a.str("doc-sep");
    const bool markers = a.flag("loss-markers");
    const int threads = threads_arg(a);
    a.done();
    CHECK(!in.empty() && !out.empty(),
          "usage: cxxgpt text2tok --in text.txt --out data.tok [--bpe tok.bpe] [--doc-sep STR] [--loss-markers]");
    const Tokenizer tok = load_tokenizer(bpe);
    std::vector<Tokenizer> toks(threads, tok);  // encode() caches chunks, so one per thread
    std::ifstream f(in, std::ios::binary);
    CHECK(f, "cannot open %s", in.c_str());
    TokenWriter w(out, uint32_t(tok.vocab_size()));

    // Cuts are made where encoding in pieces matches encoding the whole:
    // right after a separator, or else at Tokenizer::next_safe_split.
    constexpr size_t kBlock = 8 << 20;
    constexpr size_t npos = std::string::npos;
    // Without markers the mask stays empty, so the file has no mask, as before.
    auto encode_text = [&](std::string_view text, const Tokenizer& t, std::vector<uint16_t>& o,
                           std::vector<uint8_t>& mask) {
        if (markers) encode_loss_marked(text, t, o, mask);
        else for (int id : t.encode(text)) o.push_back(uint16_t(id));
    };
    auto encode_block = [&](std::string_view blk, const Tokenizer& t, std::vector<uint16_t>& o,
                            std::vector<uint8_t>& mask, long& docs) {
        o.clear();
        mask.clear();
        if (sep.empty()) {
            encode_text(blk, t, o, mask);
            return;
        }
        for (size_t p = 0; p <= blk.size();) {
            size_t q = blk.find(sep, p);
            if (q == npos) q = blk.size();
            const std::string doc = trim(std::string(blk.substr(p, q - p)));
            if (!doc.empty()) {
                o.push_back(uint16_t(kBos));
                if (markers) mask.push_back(1);
                encode_text(doc, t, o, mask);
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
    std::vector<std::vector<uint8_t>> masks(threads);
    while (true) {
        while (!eof && (buf.size() < kBlock * threads || starved)) {
            const size_t old = buf.size();
            buf.resize(old + kBlock);
            f.read(buf.data() + old, std::streamsize(kBlock));
            CHECK(!f.bad() && (!f.fail() || f.eof()), "error reading %s", in.c_str());
            buf.resize(old + size_t(f.gcount()));
            // The BOS byte is reserved for "a document starts here": drop it
            // from the text just read, so the text cannot fake one.
            const auto new_text = buf.begin() + std::ptrdiff_t(old);
            buf.erase(std::remove(new_text, buf.end(), char(kBos)), buf.end());
            eof = !f;
            starved = false;
            CHECK(buf.size() <= kBlock*size_t(threads) + (64u<<20),
                  "no safe text split within buffer limit; prepare shorter lines/documents");
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
                end = Tokenizer::next_safe_split(all, target, pos);  // pos is 0 or an earlier cut
                if (end == all.size() && !eof) end = npos;
            } else {
                end = all.find(sep, target);
                end = end != npos ? end + sep.size() : eof ? all.size() : npos;
            }
            if (markers && end != npos) {
                end = loss_marker_safe_split(all, end, pos);  // never cut a marked span
                CHECK(!eof || end != npos, "unclosed no-loss marker");
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
            pool.emplace_back([&, i] { encode_block(blocks[i], toks[i], outs[i], masks[i], block_docs[i]); });
        }
        for (auto& th : pool) th.join();
        for (size_t i = 0; i < blocks.size(); ++i) {
            w.write(outs[i], masks[i]);
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
    size_t long_chunks = 0;
    for (const Tokenizer& t : toks) long_chunks += t.long_chunks();
    if (long_chunks > 0)
        std::printf("warning: %zu chunks longer than %zu bytes were written as plain bytes, without merges\n",
                    long_chunks, Tokenizer::kMaxChunkBytes);
    return 0;
}

// "a.tok,b.tok*3": the files concatenated, b three times (to weight a small
// dataset up). A file without a mask counts as all trained-on.
TokenData load_token_list(const std::string& spec, bool* repeats = nullptr) {
    if (repeats) *repeats = false;
    std::vector<std::pair<std::string, int>> files;
    for (const std::string& item : split(spec, ',')) {
        const size_t star = item.rfind('*');
        int times = 1;
        if (star != std::string::npos) {
            const char* begin = item.data()+star+1; const char* end = item.data()+item.size();
            const auto parsed = std::from_chars(begin,end,times);
            CHECK(parsed.ec == std::errc{} && parsed.ptr == end, "bad repeat count in '%s'", item.c_str());
        }
        CHECK(times >= 1, "bad repeat count in '%s'", item.c_str());
        if (repeats && times>1) *repeats = true;
        files.emplace_back(item.substr(0, star), times);
    }
    return TokenData::load_list(files);
}

// Training data (masked so only answers are learned), a validation slice,
// and a held-out test set that the training data never contains.
int cmd_gen_reasoning(Args& a) {
    const std::string out = a.str("out", "data/reason");
    const int64_t n_train = a.integer("train", 200000), n_val = a.integer("val", 2000);
    const int per_cell = a.integer_int("test-per-cell", 40);
    const int min_d = a.integer_int("min-diff", 1), max_d = a.integer_int("max-diff", 3);
    const int test_max_d = a.integer_int("test-max-diff", max_d < 5 ? max_d+1 : 5);
    const std::vector<std::string> tasks = split(a.str("tasks", "chain,add,mul,expr,eq,cmp"), ',');
    Formats formats;
    formats.chain_plan = a.flag("chain-plan");      // chain solutions look variables up first
    formats.expr_rewrite = a.flag("expr-rewrite");  // expr solutions write what is left after each step
    const uint64_t seed = uint64_t(a.integer("seed", 1));
    Rng rng(seed), mistake_rng(seed ^ 0xd1b54a32d192ed03ULL);
    const double mistakes = a.real("mistakes", 0);
    const std::string bpe = a.str("bpe");
    a.done();
    const Tokenizer tok = load_tokenizer(bpe);
    CHECK(mistakes >= 0 && mistakes <= 1, "--mistakes must be in [0,1]");
    CHECK(min_d >= 1 && min_d <= max_d && max_d <= 5 && test_max_d >= min_d && test_max_d <= 5,
          "reasoning difficulties must be ordered within [1,5]");
    CHECK(per_cell > 0 && n_train >= 0 && n_val >= 0, "reasoning split counts must be nonnegative; test-per-cell positive");
    for (const auto& t : tasks) make_problem(t, 1, rng);  // rejects unknown names

    // Test set first: every (task, difficulty) cell, including one level
    // harder than training, to measure generalization beyond it.
    std::vector<Problem> test;
    std::set<std::string> held_out;
    for (const auto& t : tasks)
        for (int d = min_d; d <= test_max_d; ++d) {
            int tries = 0;
            for (int i = 0; i < per_cell;) {
                CHECK(++tries <= 1000000, "question domain exhausted while preparing %s d=%d", t.c_str(), d);
                Problem p = make_problem(t, d, rng, formats);
                if (!held_out.insert(p.question).second) continue;  // no duplicates
                test.push_back(std::move(p));
                ++i;
            }
        }
    save_problems(out + ".test.txt", test);

    auto make_tokens = [&](int64_t n, const std::string& path, bool training) {
        FILE* text = training && mistakes > 0 ? create_file(out + ".train.text", "wb") : nullptr;
        CHECK(!training || mistakes == 0 || text, "cannot write reasoning training text");
        TokenData d;
        d.vocab = uint32_t(tok.vocab_size());
        long skipped = 0;
        int tries = 0;
        for (int64_t i = 0; i < n;) {
            CHECK(++tries <= 1000000, "question domain exhausted while preparing training/validation");
            const std::string& t = tasks[rng.below(tasks.size())];
            Problem p =
                make_problem(t, min_d + int(rng.below(uint64_t(max_d - min_d + 1))), rng, formats);
            if (held_out.count(p.question)) {
                ++skipped;
                continue;
            }
            if (training && mistakes > 0 && mistake_rng.uniform() < mistakes) plant_mistake(p, mistake_rng);
            if (text) {
                const std::string doc = p.prompt() + p.completion() + "<|endoftext|>\n";
                CHECK(std::fwrite(doc.data(), 1, doc.size(), text) == doc.size(), "error writing reasoning text");
            }
            append_problem(d, p, tok);
            ++i;
            tries = 0;
        }
        if (text) CHECK(std::fclose(text) == 0, "error closing reasoning text");
        d.save(path);
        std::printf("%s: %lld problems, %zu tokens (%ld test duplicates skipped)\n", path.c_str(), (long long)n,
                    d.tokens.size(), skipped);
    };
    make_tokens(n_train, out + ".train.tok", true);
    make_tokens(n_val, out + ".val.tok", false);
    std::printf("%s.test.txt: %zu problems (%zu tasks x difficulty %d..%d x %d)\n", out.c_str(),
                test.size(), tasks.size(), min_d, test_max_d, per_cell);
    return 0;
}

// Two views of the same generated problems: the existing five-field file
// for evaluation, and Q/A documents for text2tok --doc-sep '<|endoftext|>'.
void write_problem_files(const std::string& out, const std::vector<Problem>& problems) {
    save_problems(out + ".txt", problems);
    const std::string text_path = out + ".text";
    // Binary: text2tok reads the file as bytes, so a Windows "\r\n" would
    // become extra tokens and change the stream for the same seed.
    FILE* text = create_file(text_path, "wb");
    CHECK(text, "cannot write %s", text_path.c_str());
    for (const auto& p : problems)
        CHECK(std::fprintf(text, "%s%s<|endoftext|>\n", p.prompt().c_str(), p.completion().c_str()) >= 0,
              "error writing %s", text_path.c_str());
    CHECK(std::fclose(text) == 0, "error writing %s", text_path.c_str());
}

int cmd_gen_word(Args& a) {
    const std::string out = a.str("out", "data/word");
    const int64_t count = a.integer("count", 1000), seed = a.integer("seed", 1);
    const int64_t min_steps = a.integer("min-steps", 2), max_steps = a.integer("max-steps", 8);
    const std::string split = a.str("split", "train");
    const std::string distractors = a.str("distractors", "on"), notes = a.str("notes", "on");
    a.done();
    CHECK(!out.empty(), "--out must be a nonempty prefix");
    CHECK(count >= 0 && seed >= 0, "--count and --seed must be nonnegative");
    CHECK(min_steps >= 2 && min_steps <= max_steps && max_steps <= 8,
          "steps must satisfy 2 <= min-steps <= max-steps <= 8");
    CHECK(split == "train" || split == "test", "--split must be train or test");
    CHECK(distractors == "on" || distractors == "off", "--distractors must be on or off");
    CHECK(notes == "on" || notes == "off", "--notes must be on or off");
    const WordOptions options{int(min_steps), int(max_steps), distractors == "on", notes == "on", split == "test"};
    const auto problems = make_word_problems(uint64_t(seed), count, options);
    write_problem_files(out, problems);
    std::printf("%s.txt and %s.text: %zu %s word problems, %lld..%lld steps\n", out.c_str(), out.c_str(),
                problems.size(), split.c_str(), (long long)min_steps, (long long)max_steps);
    return 0;
}

// Math problems built from a chosen answer, each solved again by a second
// solver before it is written (src/mathgen.h, src/mathgen_fractions.h). The
// same two files as gen-word.
int cmd_gen_math(Args& a) {
    const std::string topic = a.str("topic", "equations");
    const std::string out = a.str("out", "data/math/" + topic);
    const int64_t count = a.integer("count", 1000), seed = a.integer("seed", 1);
    const std::string split = a.str("split", "train");
    const std::string distractors = a.str("distractors", "on"), notes = a.str("notes", "on");
    // Questions not to repeat, such as GSM8K test and dev (README, gen-math).
    const std::string exclude = a.str("exclude", "");
    a.done();
    CHECK(topic == "equations" || topic == "fractions", "--topic must be equations or fractions");
    CHECK(!out.empty(), "--out must be a nonempty prefix");
    CHECK(count >= 0 && seed >= 0, "--count and --seed must be nonnegative");
    CHECK(split == "train" || split == "test", "--split must be train or test");
    CHECK(distractors == "on" || distractors == "off", "--distractors must be on or off");
    CHECK(notes == "on" || notes == "off", "--notes must be on or off");
    std::unique_ptr<Exclusions> exclusions;
    if (!exclude.empty()) exclusions = std::make_unique<Exclusions>(exclude);
    const MathOptions options{distractors == "on", notes == "on", split == "test", exclusions.get()};
    const auto problems = topic == "equations" ? make_equation_problems(uint64_t(seed), count, options)
                                               : make_fraction_problems(uint64_t(seed), count, options);
    write_problem_files(out, problems);
    std::printf("%s.txt and %s.text: %zu %s %s problems, each solved again by the checker\n", out.c_str(),
                out.c_str(), problems.size(), split.c_str(), topic.c_str());
    if (exclusions)
        std::printf("drawn again: %zu sharing a 13-word run, %zu with the same numbers and answer as one of %zu "
                    "excluded questions\n", exclusions->by_run, exclusions->by_numbers, exclusions->size());
    else
        std::printf("no --exclude: these problems may repeat GSM8K test or dev questions (README, gen-math)\n");
    return 0;
}

// A JSON number as an answer, written out in full from its exact text (no
// rounding through a double): 18 -> "18", 2.50 -> "2.5", 1.2e3 -> "1200",
// -0.00120 -> "-0.0012".
std::string format_number(const Json& value) {
    const std::string& raw = value.str;
    const bool negative = !raw.empty() && raw[0] == '-';
    const size_t begin = negative ? 1 : 0;
    const size_t exponent_at = raw.find_first_of("eE");  // npos if there is none

    // The digits without sign, point or exponent, and where the point goes:
    // "12.5" gives digits "125" and point_at 2.
    std::string digits = raw.substr(begin, exponent_at - begin);
    const size_t dot = digits.find('.');
    long point_at = long(dot == std::string::npos ? digits.size() : dot);
    if (dot != std::string::npos) digits.erase(dot, 1);

    // Leading zeros: "0.0012" gives digits "12" and point_at -2.
    const size_t first = digits.find_first_not_of('0');
    if (first == std::string::npos) return "0";
    digits.erase(0, first);
    point_at -= long(first);

    // An exponent moves the point: "1.2e3" gives point_at 1 + 3 = 4.
    if (exponent_at != std::string::npos) {
        const char* start = raw.data() + exponent_at + 1;
        if (*start == '+') ++start;
        long exponent = 0;
        const auto result = std::from_chars(start, raw.data() + raw.size(), exponent);
        CHECK(result.ec == std::errc{}, "numeric answer exponent out of range");
        point_at += exponent;
    }

    // Trailing zeros after the point carry no value; keep one digit.
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();

    if (point_at <= 0) {
        digits = "0." + std::string(size_t(-point_at), '0') + digits;  // 0.0012
    } else if (size_t(point_at) >= digits.size()) {
        digits.append(size_t(point_at) - digits.size(), '0');  // 1200
    } else {
        digits.insert(size_t(point_at), 1, '.');  // 12.5
    }
    return (negative ? "-" : "") + digits;
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
        if (!Json::parse(line,j,&err)) {
            std::fprintf(stderr, "%s:%ld: skipped malformed JSON: %s\n", in.c_str(),line_no,err.c_str());
            ++skipped; continue;
        }
        const Json *q = j.get(q_key), *ans = j.get(a_key);
        if (!q || !ans || q->type!=Json::String || ans->type!=Json::String) {
            std::fprintf(stderr, "%s:%ld: skipped missing question/answer string fields\n", in.c_str(),line_no);
            ++skipped; continue;
        }
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
            if (!fin) {
                std::fprintf(stderr, "%s:%ld: skipped missing final field '%s'\n", in.c_str(),line_no,final_key.c_str());
                ++skipped; continue;
            }
            p.steps = trim(ans->str);
            if (fin->type == Json::String) p.answer = strip_digit_commas(trim(fin->str));
            else if (fin->type == Json::Number) p.answer = format_number(*fin);
        }
        // Raw control BOS in source prose is reserved; only append_problem inserts it.
        for (auto* text : {&p.question,&p.steps,&p.answer})
            text->erase(std::remove(text->begin(),text->end(),char(kBos)),text->end());
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
    std::map<std::string, std::map<int, LevelScore>> table;
    int min_d = 1 << 30, max_d = -(1 << 30);
    for (const auto& [key, score] : r.by_task) {
        table[key.first][key.second] = score;
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
            std::printf(" %5.1f%%", 100.0 * it->second.vote_correct / it->second.total);
            c += it->second.vote_correct;
            n += it->second.total;
        }
        std::printf(" %5.1f%%\n", 100.0 * c / std::max(1, n));
    }
    // The effort label goes last, so "overall X% (a / b)" reads as before.
    std::printf("overall %.1f%% (%d / %d) (%s)\n", 100.0 * r.correct / std::max(1, r.total), r.correct, r.total,
                r.effort.c_str());
    std::printf("gold completions exceeding --max-new %d: %zu / %d\n", r.max_new, r.gold_over_cap, r.total);
    std::printf("unfinished answers (no \"#### \" line, first tries): %zu / %d\n", r.unfinished, r.total);
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

// Keep both entry points strict: a misspelled mode must not change a score silently.
calc::Mode calculator_mode(Args& a) {
    const std::string mode = a.str("tools", "none");
    CHECK(mode == "none" || mode == "calc-notes" || mode == "calc-plain",
          "--tools must be none, calc-notes or calc-plain");
    return mode == "calc-notes" ? calc::Mode::notes :
           mode == "calc-plain" ? calc::Mode::plain : calc::Mode::none;
}

int cmd_eval_reasoning(Args& a) {
    ThreadPool tp(threads_arg(a));
    g_pool = &tp;
    const std::string ckpt = a.str("ckpt", "model.ckpt"), test_path = a.str("test", "data/reason.test.txt");
    const int batch = a.integer_int("batch", 64), limit = a.integer_int("limit", 0);
    const bool show = a.flag("show-failures");
    const int max_new = a.integer_int("max-new", 200);
    CHECK(max_new > 0, "--max-new must be positive");
    // How many times to sample each problem and take a vote. --effort is a
    // shorthand for a number of tries; --tries wins if both are given.
    VoteOpts vote;
    const std::string effort = a.str("effort", "low");
    CHECK(effort == "low" || effort == "medium" || effort == "high", "--effort must be low, medium or high");
    const int effort_tries = effort == "low" ? 1 : effort == "medium" ? 8 : 32;
    vote.tries = a.has("tries") ? a.integer_int("tries", 1) : effort_tries;
    CHECK(vote.tries > 0, "--tries must be positive");
    vote.temperature = a.real("temp", 0.0);
    CHECK(vote.tries == 1 || vote.temperature > 0, "--temp must be above 0 when sampling more than one try");
    // Each try is seeded from (seed, problem, try): see VoteOpts.
    const long seed = a.integer("seed", 1);
    CHECK(seed >= 0, "--seed must not be negative");
    vote.seed = uint64_t(seed);
    // Never repeat a run of N generated tokens (no_repeat.h, NoRepeat); 0: off.
    vote.no_repeat = a.integer_int("no-repeat", 0);
    CHECK(vote.no_repeat == 0 || vote.no_repeat >= 2, "--no-repeat must be 0 (off) or at least 2");
    // Each first try labelled (correct, unfinished, format, arithmetic,
    // passed_gold, other; failure_labels.h, #16), one JSON line per problem.
    const std::string labels_path = a.str("labels");
    // As TernaryEngine runs it: the embedding (and --fp-attn attention) in
    // int8 rows.
    const bool as_engine = a.flag("as-engine");
    const std::string bpe = a.str("bpe");
    const calc::Mode tool = calculator_mode(a);
    // On the GPU (cxxgpt-gpu): exact fp32 products by default, so the
    // scores are the CPU's; --matmul tf32 or bf16 scores the model as
    // training computes it.
    const bool use_gpu = a.flag("gpu");
    CHECK(use_gpu || !a.has("matmul"), "--matmul needs --gpu");
    const std::string matmul = a.str("matmul", "fp32");
    CHECK(matmul == "fp32" || matmul == "tf32" || matmul == "bf16", "--matmul takes fp32, tf32 or bf16");
    a.done();
#ifndef CXXGPT_CUDA
    CHECK(!use_gpu, "this cxxgpt was built without CUDA; use cxxgpt-gpu (make gpu) for --gpu");
#endif
    const Tokenizer tok = load_tokenizer(bpe);
    Model<float> model(peek_checkpoint_config(ckpt));
    load_checkpoint(ckpt, model, nullptr, nullptr);
    if (as_engine) round_to_engine_int8(model);
#ifdef CXXGPT_CUDA
    // The weights (as rounded by --as-engine) and router biases go to the
    // GPU once; both runs (without and with the calculator) decode there.
    // Each answer's KV cache takes GPU memory, so the batch is as many as
    // fit, up to --batch. With too little room, the CPU runs it instead.
    std::unique_ptr<GpuTrainer> gpu;
    int gpu_batch = 0;
    if (use_gpu) {
        const Matmul mm = matmul == "fp32" ? Matmul::fp32 : matmul == "bf16" ? Matmul::bf16 : Matmul::tf32;
        gpu = std::make_unique<GpuTrainer>(model.cfg, model.infos, mm, false, /*inference=*/true);
        gpu->set_params(model.params.data());
        gpu->set_router_bias(model.router_bias);
        gpu_batch = gpu->decode_capacity(batch);
        // One problem's tries are decoded together, so they must fit at once.
        CHECK(gpu_batch >= vote.tries,
              "the GPU has room for %d answers at a time (%zu MB each, %zu MB free), fewer than --tries %d",
              gpu_batch, gpu->decode_bytes_per_sequence() >> 20, GpuTrainer::free_device_bytes() >> 20, vote.tries);
        std::printf("gpu: %s, %s matmuls, %d answers at a time\n", GpuTrainer::device_name().c_str(),
                    matmul.c_str(), gpu_batch);
    }
#endif
    std::vector<Problem> test = load_problems(test_path);
    if (limit > 0 && size_t(limit) < test.size()) {
        // An even sample across the file, which is ordered by task. Each
        // problem keeps its place in the file as its number, so its tries
        // are seeded the same as without --limit.
        std::vector<Problem> sub;
        for (int i = 0; i < limit; ++i) {
            const size_t place = size_t(i) * test.size() / size_t(limit);
            sub.push_back(test[place]);
            vote.problem_ids.push_back(place);
        }
        test = std::move(sub);
    }
    // The same evaluation on whichever of the two decodes.
    auto evaluate = [&](calc::Mode t) {
#ifdef CXXGPT_CUDA
        if (gpu) {
            GpuDecoder dec{model.cfg, *gpu};
            return eval_reasoning(dec, test, tok, gpu_batch, max_new, 8, !labels_path.empty(), vote, t);
        }
#endif
        return eval_reasoning(model, test, tok, batch, max_new, 8, !labels_path.empty(), vote, t);
    };
    const double t0 = now_seconds();
    // The scores, and with more than one try the effort report: per level
    // (task, difficulty) and overall, the first sampled try's accuracy (a
    // sample, not the greedy score), the vote's (maj@N), any try's (pass@N),
    // and the mean number of distinct answers per problem (finished tries
    // only).
    auto report = [&](const EvalResult& r) {
        print_eval(r, show ? r.failures.size() : 0);
        if (vote.tries == 1) return;
        std::printf("effort: tries %d, temperature %g, seed %ld\n", vote.tries, vote.temperature, seed);
        std::printf("%-6s %2s %4s %17s %7s %7s %8s\n", "task", "d", "n", "first sampled try", "vote", "any",
                    "distinct");
        auto pct = [](int part, int whole) { return 100.0 * part / std::max(1, whole); };
        for (const auto& [key, s] : r.by_task)
            std::printf("%-6s %2d %4d %16.1f%% %6.1f%% %6.1f%% %8.2f\n", key.first.c_str(), key.second, s.total,
                        pct(s.first_correct, s.total), pct(s.vote_correct, s.total), pct(s.any_correct, s.total),
                        double(s.distinct_answers) / std::max(1, s.total));
        const LevelScore all = r.overall();
        std::printf("%-9s %4d %16.1f%% %6.1f%% %6.1f%% %8.2f (first sampled try / vote maj@%d / any try "
                    "pass@%d / mean distinct answers)\n",
                    "overall",
                    all.total, pct(all.first_correct, all.total), pct(all.vote_correct, all.total),
                    pct(all.any_correct, all.total), double(all.distinct_answers) / std::max(1, all.total),
                    vote.tries, vote.tries);
    };
    // With a calculator, the same problems twice: without it, then with it.
    const EvalResult r = evaluate(calc::Mode::none);
    if (tool != calc::Mode::none) std::printf("without tool:\n");
    report(r);
    if (!labels_path.empty()) {
        // The run without the calculator: its results would stand in for the
        // model's own arithmetic. Each problem keeps its place in the file.
        std::vector<size_t> ids = vote.problem_ids;
        for (size_t i = ids.size(); i < test.size(); ++i) ids.push_back(i);
        auto str = [](const std::string& s) { return "\"" + labels::json_escape(s) + "\""; };
        const std::string settings = "{\"ckpt\":" + str(ckpt) + ",\"test\":" + str(test_path) +
                                     ",\"effort\":" + str(r.effort) + ",\"max_new\":" + std::to_string(max_new) +
                                     ",\"tool\":\"none\",\"as_engine\":" + (as_engine ? "true" : "false") +
                                     ",\"gpu\":" + (use_gpu ? "true" : "false") + ",\"matmul\":" + str(matmul) +
                                     ",\"labelled\":\"first try\"}";
        const labels::LabelSummary s = labels::write_labels(labels_path, settings, test, r.completions, ids,
                                                            [&](const std::string& c) { return tok.encode(c); });
        s.print(labels::kRunTokens);
        std::printf("labels: %s\n", labels_path.c_str());
    }
    if (tool != calc::Mode::none) {
        const EvalResult assisted = evaluate(tool);
        std::printf("with calculator:\n");
        report(assisted);
        std::printf("calculator calls: %zu fired, %zu malformed\n",
                    assisted.tool_calls.fired, assisted.tool_calls.malformed);
    }
    std::printf("(%.1f s)\n", now_seconds() - t0);
    return 0;
}

// The model's own checked solutions as training data (self_train.h): every
// problem sampled --tries times, and only the tries with the right answer
// and right arithmetic kept, written as OUT.txt (problems), OUT.tok (masked
// tokens, as prep-qa writes them) and OUT.json (settings and counts). The
// problems must be training problems (GSM8K train, generator train splits),
// never a dev or test set: --exclude drops any that repeat those.
int cmd_self_train(Args& a) {
    ThreadPool tp(threads_arg(a));
    g_pool = &tp;
    const std::string ckpt = a.str("ckpt", "model.ckpt"), problems_path = a.str("problems");
    const std::string out = a.str("out");
    const int batch = a.integer_int("batch", 64), limit = a.integer_int("limit", 0);
    const int max_new = a.integer_int("max-new", 200);
    CHECK(max_new > 0, "--max-new must be positive");
    VoteOpts vote;
    vote.tries = a.integer_int("tries", 8);
    CHECK(vote.tries > 0, "--tries must be positive");
    vote.temperature = a.real("temp", 0.7);
    CHECK(vote.temperature > 0, "--temp must be above 0: greedy tries are all the same");
    // Each try is seeded from (seed, problem, try), as in eval-reasoning.
    const long seed = a.integer("seed", 1);
    CHECK(seed >= 0, "--seed must not be negative");
    vote.seed = uint64_t(seed);
    vote.no_repeat = a.integer_int("no-repeat", 0);
    CHECK(vote.no_repeat == 0 || vote.no_repeat >= 2, "--no-repeat must be 0 (off) or at least 2");
    self_train::Options opts;
    // Kept tries per problem at most (0: every distinct one).
    opts.keep = a.integer_int("keep", 0);
    CHECK(opts.keep >= 0, "--keep must not be negative");
    opts.allow_loops = a.flag("allow-loops");
    // Questions not to train on, such as GSM8K test and dev (gen-math's rule).
    const std::string exclude = a.str("exclude", "");
    // Every try's text and verdict, one JSON line per try, for audits.
    const std::string audit_path = a.str("audit");
    const std::string bpe = a.str("bpe");
    const calc::Mode tool = calculator_mode(a);
    const bool use_gpu = a.flag("gpu");
    CHECK(use_gpu || !a.has("matmul"), "--matmul needs --gpu");
    const std::string matmul = a.str("matmul", "fp32");
    CHECK(matmul == "fp32" || matmul == "tf32" || matmul == "bf16", "--matmul takes fp32, tf32 or bf16");
    a.done();
    CHECK(!problems_path.empty() && !out.empty(),
          "usage: cxxgpt self-train --ckpt model.ckpt --problems FILE.txt --out PREFIX [--tries 8 --temp 0.7]");
#ifndef CXXGPT_CUDA
    CHECK(!use_gpu, "this cxxgpt was built without CUDA; use cxxgpt-gpu (make gpu) for --gpu");
#endif
    const Tokenizer tok = load_tokenizer(bpe);
    Model<float> model(peek_checkpoint_config(ckpt));
    load_checkpoint(ckpt, model, nullptr, nullptr);
#ifdef CXXGPT_CUDA
    std::unique_ptr<GpuTrainer> gpu;
    int gpu_batch = 0;
    if (use_gpu) {
        const Matmul mm = matmul == "fp32" ? Matmul::fp32 : matmul == "bf16" ? Matmul::bf16 : Matmul::tf32;
        gpu = std::make_unique<GpuTrainer>(model.cfg, model.infos, mm, false, /*inference=*/true);
        gpu->set_params(model.params.data());
        gpu->set_router_bias(model.router_bias);
        gpu_batch = gpu->decode_capacity(batch);
        CHECK(gpu_batch >= vote.tries,
              "the GPU has room for %d answers at a time (%zu MB each, %zu MB free), fewer than --tries %d",
              gpu_batch, gpu->decode_bytes_per_sequence() >> 20, GpuTrainer::free_device_bytes() >> 20, vote.tries);
        std::printf("gpu: %s, %s matmuls, %d answers at a time\n", GpuTrainer::device_name().c_str(),
                    matmul.c_str(), gpu_batch);
    }
#endif
    // Each problem keeps its place in the file as its number, so its tries
    // are seeded the same whatever --limit or --exclude leave out.
    const std::vector<Problem> all = load_problems(problems_path);
    std::vector<Problem> problems;
    std::unique_ptr<Exclusions> exclusions;
    if (!exclude.empty()) exclusions = std::make_unique<Exclusions>(exclude);
    size_t excluded = 0;
    const size_t wanted = limit > 0 ? std::min(size_t(limit), all.size()) : all.size();
    for (size_t i = 0; i < wanted; ++i) {
        const size_t place = limit > 0 ? i * all.size() / wanted : i;  // an even sample, as eval-reasoning's
        if (exclusions && exclusions->excludes(all[place], 0)) {
            ++excluded;
            continue;
        }
        problems.push_back(all[place]);
        vote.problem_ids.push_back(place);
    }
    const double t0 = now_seconds();
    calc::Counts calls;
    std::vector<std::vector<std::string>> texts;
#ifdef CXXGPT_CUDA
    if (gpu) {
        GpuDecoder dec{model.cfg, *gpu};
        texts = sample_tries(dec, problems, tok, gpu_batch, max_new, vote, tool, &calls);
    }
#endif
    if (texts.empty() && !problems.empty()) texts = sample_tries(model, problems, tok, batch, max_new, vote, tool, &calls);
    const self_train::Result r = self_train::keep_checked(problems, texts, opts);
    self_train::write(out, r.kept, tok);

    auto pct = [](size_t part, size_t whole) { return 100.0 * double(part) / double(std::max<size_t>(1, whole)); };
    const std::string no_repeat = vote.no_repeat ? ", no-repeat " + std::to_string(vote.no_repeat) : "";
    std::printf("%zu problems (%zu excluded), %d tries each (temp %g, seed %ld%s), %zu tries\n", r.problems,
                excluded, vote.tries, vote.temperature, seed, no_repeat.c_str(), r.tries);
    std::printf("first try right: %.1f%% (%zu); solved by any try (pass@%d): %.1f%% (%zu); "
                "with a kept solution: %.1f%% (%zu)\n",
                pct(r.first_right, r.problems), r.first_right, vote.tries, pct(r.solved, r.problems), r.solved,
                pct(r.solved_kept, r.problems), r.solved_kept);
    std::printf("tries by verdict:");
    for (int v = 0; v < self_train::kVerdicts; ++v)
        std::printf(" %s %zu", self_train::verdict_name(self_train::Verdict(v)), r.by_verdict[v]);
    std::printf("\ncalculations in the right tries' work: %zu (%zu the calculator could not read)\n", r.calcs,
                r.unreadable);
    if (tool != calc::Mode::none)
        std::printf("calculator calls: %zu fired, %zu malformed\n", calls.fired, calls.malformed);
    std::printf("kept %zu solutions -> %s.txt, %s.tok, %s.json (%.1f s)\n", r.kept.size(), out.c_str(),
                out.c_str(), out.c_str(), now_seconds() - t0);

    auto str = [](const std::string& s) { return "\"" + labels::json_escape(s) + "\""; };
    char temp[32];
    std::snprintf(temp, sizeof temp, "%g", vote.temperature);
    std::string json = "{\"ckpt\":" + str(ckpt) + ",\"problems\":" + str(problems_path) + ",\"bpe\":" + str(bpe) +
                       ",\"exclude\":" + str(exclude) + ",\"limit\":" + std::to_string(limit) +
                       ",\"tries\":" + std::to_string(vote.tries) + ",\"temp\":" + std::string(temp) +
                       ",\"seed\":" + std::to_string(seed) + ",\"no_repeat\":" + std::to_string(vote.no_repeat) +
                       ",\"max_new\":" + std::to_string(max_new) + ",\"tools\":" +
                       str(tool == calc::Mode::notes ? "calc-notes" : tool == calc::Mode::plain ? "calc-plain" : "none") +
                       ",\"keep\":" + std::to_string(opts.keep) + ",\"allow_loops\":" +
                       (opts.allow_loops ? "true" : "false") + ",\"gpu\":" + (use_gpu ? "true" : "false") +
                       ",\"matmul\":" + str(matmul) + ",\"problems_tried\":" + std::to_string(r.problems) +
                       ",\"excluded\":" + std::to_string(excluded) + ",\"first_right\":" +
                       std::to_string(r.first_right) + ",\"solved\":" + std::to_string(r.solved) +
                       ",\"solved_kept\":" + std::to_string(r.solved_kept) + ",\"kept\":" +
                       std::to_string(r.kept.size()) + ",\"verdicts\":{";
    for (int v = 0; v < self_train::kVerdicts; ++v)
        json += std::string(v ? "," : "") + str(self_train::verdict_name(self_train::Verdict(v))) + ":" +
                std::to_string(r.by_verdict[v]);
    json += "}}\n";
    FILE* f = create_file(out + ".json", "wb");
    CHECK(f && std::fputs(json.c_str(), f) >= 0 && std::fclose(f) == 0, "cannot write %s.json", out.c_str());
    if (!audit_path.empty()) {
        FILE* log = create_file(audit_path, "wb");
        CHECK(log, "cannot write %s", audit_path.c_str());
        for (size_t i = 0; i < problems.size(); ++i)
            for (size_t t = 0; t < texts[i].size(); ++t)
                CHECK(std::fprintf(log, "{\"problem\":%zu,\"try\":%zu,\"verdict\":%s,\"text\":%s}\n",
                                   vote.problem_ids[i], t, str(self_train::verdict_name(r.verdicts[i][t])).c_str(),
                                   str(texts[i][t]).c_str()) >= 0,
                      "error writing %s", audit_path.c_str());
        CHECK(std::fclose(log) == 0, "error writing %s", audit_path.c_str());
        std::printf("every try: %s\n", audit_path.c_str());
    }
    return 0;
}

// One training step's operations, on the CPU or the GPU.
struct Backend {
    virtual ~Backend() = default;
    // The mean loss of one batch, for validation (eval_loss).
    virtual double forward(const int* inp, const int* tgt, int B, int Tn, bool training) = 0;
    // One training step on a sampled group: the gradient of its mean loss
    // (accumulated over its micro-batches); then, if the loss and the
    // gradient norm are finite, the optimizer step on the gradient clipped
    // to max_norm (if > 0), and the router biases nudged by bias_rate from
    // the step's expert loads, which a non-null interval also collects for
    // --load-log. Otherwise (overflow, NaN) the weights stay as they were
    // and the step's loads are dropped.
    struct StepResult {
        double loss = 0, gnorm = 0;  // gnorm: before clipping
        bool ok = false;             // the update ran
    };
    virtual StepResult train_step(const AccumulationBatch& group, double lr, double max_norm, double bias_rate,
                                  ExpertLoadLog* interval) = 0;
    // The busiest expert's load against an even split (Model::max_load_ratio),
    // worst layer, in the last step that updated; 1 for a dense model.
    virtual double load_ratio() = 0;
    // Expert loads the backend kept for --load-log since the last call
    // (the GPU keeps them there between log lines), into interval.
    virtual void flush_loads(ExpertLoadLog& interval) { (void)interval; }
    // Brings the weights and optimizer state into the host Model and AdamW,
    // for checkpoints and evaluations.
    virtual void sync() = 0;
    // The reasoning exam on the current weights: greedy, exact answers.
    virtual EvalResult reason(const std::vector<Problem>& problems, const Tokenizer& tok, int max_new,
                              bool keep_completions) = 0;
    // Device memory held, for the GPU; 0 on the CPU.
    virtual size_t device_bytes() const { return 0; }
    // Unlikelihood training (#34). While it is on, forward() still returns
    // the cross-entropy alone (the curve stays comparable with runs without
    // it), and the term's own numbers add up in take_unlikelihood() until it
    // is called.
    virtual void set_unlikelihood(double alpha, int ngram) = 0;
    struct UnlikelihoodStats {
        double sum = 0;  // the term (without alpha), summed over trained targets
        size_t targets = 0, fired = 0, negatives = 0;
    };
    UnlikelihoodStats take_unlikelihood() { return std::exchange(ul_, {}); }

protected:
    double ul_alpha_ = 0;
    // A training forward's parts (means over its targets), for forward():
    // adds the term's numbers and returns the cross-entropy.
    double add_unlikelihood(double ce, double ul, size_t fired, size_t negatives, const int* tgt, int N) {
        const size_t targets = count_targets(tgt, size_t(N));
        add_unlikelihood_sums(ul * double(targets), targets, fired, negatives);
        return ce;
    }
    void add_unlikelihood_sums(double sum, size_t targets, size_t fired, size_t negatives) {
        ul_.sum += sum;
        ul_.targets += targets;
        ul_.fired += fired;
        ul_.negatives += negatives;
    }

private:
    UnlikelihoodStats ul_;
};

struct CpuBackend : Backend {
    Model<float>& model;
    AdamW& opt;
    double load_ = 1.0;  // load_ratio()
    CpuBackend(Model<float>& m, AdamW& o) : model(m), opt(o) {}
    double forward(const int* inp, const int* tgt, int B, int Tn, bool training) override {
        const double loss = model.forward(inp, tgt, B, Tn, training);
        if (!training || ul_alpha_ <= 0 || !tgt) return loss;
        return add_unlikelihood(model.last_ce, model.last_ul, model.last_fired, model.last_negatives, tgt, B * Tn);
    }
    void backward(double loss_scale) { model.backward(loss_scale); }
    void set_unlikelihood(double alpha, int ngram) override {
        ul_alpha_ = alpha;
        model.unlikelihood = {alpha, ngram};
    }
    void zero_grad() { model.zero_grad(); }
    // Clips the gradient to max_norm (if > 0), then takes an AdamW step;
    // a gradient that is not finite is skipped instead. Returns the norm
    // before clipping.
    double step(double lr, double max_norm) {
        const double norm = clip_grad_norm(model, max_norm);
        if (std::isfinite(norm)) opt.update(model, lr);
        return norm;
    }
    double balance(double rate, ExpertLoadLog* interval) {
        return balance_with_load_log(model, rate, interval);
    }
    StepResult train_step(const AccumulationBatch& group, double lr, double max_norm, double bias_rate,
                          ExpertLoadLog* interval) override {
        StepResult r;
        r.loss = accumulated_backward(*this, group);
        r.gnorm = std::isfinite(r.loss) ? step(lr, max_norm) : std::numeric_limits<double>::quiet_NaN();
        r.ok = std::isfinite(r.loss) && std::isfinite(r.gnorm);
        if (r.ok) load_ = balance(bias_rate, interval);
        else balance(0, nullptr);  // drop this attempt's loads without changing biases
        return r;
    }
    double load_ratio() override { return load_; }
    void sync() override {}
    EvalResult reason(const std::vector<Problem>& problems, const Tokenizer& tok, int max_new,
                      bool keep_completions) override {
        return eval_reasoning(model, problems, tok, 64, max_new, 8, keep_completions);
    }
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
    void set_unlikelihood(double alpha, int ngram) override {
        ul_alpha_ = alpha;
        gpu.set_unlikelihood(alpha, ngram);
    }
    // The whole step on the GPU (GpuTrainer::finish_step): the losses, the
    // norm, the checks, clipping and the router biases stay there; one
    // small copy at the end brings back what the log and the checks need.
    StepResult train_step(const AccumulationBatch& group, double lr, double max_norm, double bias_rate,
                          ExpertLoadLog* interval) override {
        (void)interval;  // the GPU keeps the loads until flush_loads
        gpu.zero_grad();
        for (int micro = 0; micro < group.micros; ++micro) {
            gpu.forward_train(group.input(micro), group.target(micro), group.batch, group.seq);
            gpu.backward(double(group.valid[micro]) / double(group.total_valid));
        }
        const GpuTrainer::StepStats s =
            gpu.finish_step(lr, opt.beta1, opt.beta2, opt.eps, opt.weight_decay, max_norm, bias_rate);
        if (ul_alpha_ > 0) add_unlikelihood_sums(s.ul_sum, group.total_valid, s.ul_fired, s.ul_negatives);
        if (s.ok) ++opt.step;
        // GPU time per part of the step (evaluation passes included).
        if (g_profiler.enabled)
            for (const auto& [name, seconds] : gpu.take_profile()) g_profiler.add(name.c_str(), seconds);
        return {s.ce_sum / double(group.total_valid), s.norm, s.ok};
    }
    double load_ratio() override { return gpu.last_load_ratio(); }
    void flush_loads(ExpertLoadLog& interval) override {
        if (model.cfg.moe()) interval.add(gpu.take_interval_load());
    }
    void sync() override {
        gpu.get_params(model.params.data());
        model.router_bias = gpu.get_router_bias();  // finish_step changes them on the GPU
        model.weights_dirty = true;
        opt.m.resize(model.params.size());
        opt.v.resize(model.params.size());
        gpu.get_adam_state(&opt.step, opt.m.data(), opt.v.data());
    }
    // On the GPU, in the memory training leaves free: as many answers at a
    // time as their caches fit, up to the CPU's 64 (the batch does not
    // change greedy answers). The GPU exam scores the model as training
    // computes it (bf16 products by default), so a CPU eval-reasoning of
    // the checkpoint can differ by a problem or two.
    EvalResult reason(const std::vector<Problem>& problems, const Tokenizer& tok, int max_new,
                      bool keep_completions) override {
        const int batch = gpu.decode_capacity(64);
        if (batch == 0) {
            std::printf("reasoning eval skipped: the GPU decoder needs %zu MB per answer, and the gradient buffer "
                        "and %zu MB of free GPU memory are not enough\n",
                        gpu.decode_bytes_per_sequence() >> 20, GpuTrainer::free_device_bytes() >> 20);
            return {};
        }
        std::printf("reasoning eval on the GPU, %d answers at a time\n", batch);
        GpuDecoder dec{model.cfg, gpu};
        return eval_reasoning(dec, problems, tok, batch, max_new, 8, keep_completions);
    }
    size_t device_bytes() const override { return gpu.device_bytes(); }
};
#endif

double eval_loss(Backend& be, const TokenData& data, size_t lo, size_t hi, int B, int Tn, int batches) {
    return sampled_loss([&](const int* inp, const int* tgt, int b, int t) { return be.forward(inp, tgt, b, t, false); },
                        data, lo, hi, B, Tn, batches);
}

int cmd_train(Args& a) {
    if (a.flag("help")) {
        std::printf(
            "cxxgpt train --data FILE[,FILE*N...] [options]  (*N repeats a file N times)\n"
            "data:   --val FILE (default: last --val-frac of --data, 0.05)\n"
            "        --bpe tok.bpe (the tokenizer the data was made with, for --reason-test)\n"
            "        --val-sources NAME=FILE,... (each held-out file's loss on its own at every\n"
            "        evaluation and at the start; a file with questions and answers gives its answers'\n"
            "        loss and its final answers' (the tokens after ####, needs --bpe); --val-log FILE.csv)\n"
            "model:  --seq 256 --dim 256 --layers 4 --heads 4 --ffn 0(auto)\n"
            "        --experts 0 --topk 2 --ehidden 0(auto) --shared 0 --ternary\n"
            "        --fp-attn (with --ternary: attention stays full precision; --fp-attn 0 clears it on --init)\n"
            "        --nope-every 0 (N: every Nth layer has no position encoding; 1: none has)\n"
            "train:  --batch 16 --accum 1 --steps 2000 --lr 1e-3 --min-lr-frac 0.1\n"
            "        --warmup 100 --wd 0.1 --clip 1.0 --bias-rate 1e-3 --seed 1\n"
            "        --schedule cosine (cosine: the rate falls from --lr to --lr x --min-lr-frac at --steps;\n"
            "        wsd: flat at --lr, then a cooldown over the last --decay-steps, default a fifth of\n"
            "        --steps; before the cooldown the rate does not depend on --steps, so a run can be\n"
            "        made longer, or a copy of its checkpoint resumed with --steps STEP+D --decay-steps D\n"
            "        to cool down a copy to evaluate, the run's own checkpoint untouched)\n"
            "        --muon (Muon instead of AdamW for the matrices in the blocks)\n"
            "        --sampler random (qa: a window starting inside a question-answer document that fits\n"
            "        in one moves back to its start, so the answer is trained with its question)\n"
            "        --unlikelihood 0 (ALPHA > 0: also push down each token that would repeat a run of\n"
            "        --ul-ngram 8 trained tokens of the same document; #34)\n"
            "start:  --init FILE (FILE's weights and model; new optimizer and schedule; data order from --seed;\n"
            "        with --ternary, the weights turn ternary from here on)\n"
            "output: --out model.ckpt --resume --log FILE.csv --log-every 10\n"
            "        --load-log FILE.csv (per-layer expert shares; layers/experts start at 0)\n"
            "        --eval-every 200 --eval-batches 20 --save-every 500\n"
            "        --time-limit MINUTES --threads N --profile\n"
            "control: --control FILE (default OUT.control): while training runs, write commands into it,\n"
            "        one a line: lr-scale X (the schedule's rate times X, while the line is there, also\n"
            "        after a restart), save, eval, reason, stop (each once a change of the file; never\n"
            "        from a file already there at the start). stop works like Ctrl-C\n"
            "gpu:    --gpu (train on the GPU; needs cxxgpt-gpu, from make gpu)\n"
            "        --matmul bf16 (inputs stored in bf16, the default; ternary models multiply their\n"
            "        codes, exactly; tf32: finer, about 1.45x slower; fp32: exact, slowest)\n"
            "        --no-tf32 is an alias for fp32\n"
            "        attention is fused by default (no stored weight matrix; on the tensor cores\n"
            "        with tf32/bf16); --no-fused-attn uses cuBLAS products instead\n"
            "        MoE experts: with bf16, all experts' products in one grouped launch each (no host\n"
            "        waits); with tf32/fp32, cuBLAS per expert (faster there, sized by counts the host\n"
            "        waits for); --moe-grouped or --moe-cublas picks one for any precision\n"
            "reason: --reason-test FILE --reason-every STEPS --reason-limit 300 --max-new 200\n"
            "        --reason-report PREFIX (optional answers/labels in PREFIX.step-N.jsonl)\n"
            "Ctrl-C saves a checkpoint and stops; --resume continues from --out, and\n"
            "takes precedence over --init once --out exists.\n");
        return 0;
    }
    ThreadPool tp(threads_arg(a));
    g_pool = &tp;
    const std::string data_path = a.str("data"), val_path = a.str("val");
    const std::string out = a.str("out", "model.ckpt"), log_path = a.str("log"), init = a.str("init");
    const std::string load_log_path = a.str("load-log");
    const std::string val_sources_spec = a.str("val-sources"), val_log_path = a.str("val-log");
    CHECK(val_log_path.empty() || !val_sources_spec.empty(), "--val-log needs --val-sources");
    CHECK(!data_path.empty(), "--data is required (see cxxgpt train --help)");
    bool repeated_data = false;
    const TokenData data = load_token_list(data_path,&repeated_data);
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
        const bool switch_attn = a.has("fp-attn"), switch_nope = a.has("nope-every");
        const int nope = cfg.nope_every;
        cfg = peek_checkpoint_config(init);
        cfg.ternary = cfg.ternary || to_ternary;
        if (switch_attn) cfg.fp_attention = fp_attn;
        if (switch_nope) cfg.nope_every = nope;
    }
    const int B = a.integer_int("batch", 16), accum = a.integer_int("accum", 1);
    const int Tn = std::min(cfg.seq_len, a.integer_int("seq", cfg.seq_len));
    const int64_t steps = a.integer("steps", 2000), warmup = a.integer("warmup", 100);
    const double lr = a.real("lr", 1e-3), min_frac = a.real("min-lr-frac", 0.1);
    const std::string schedule = a.str("schedule", "cosine");
    CHECK(schedule == "cosine" || schedule == "wsd", "--schedule takes cosine or wsd");
    CHECK(schedule == "wsd" || !a.has("decay-steps"), "--decay-steps needs --schedule wsd");
    const int64_t room = std::max<int64_t>(0, steps - warmup);  // steps after the warmup
    const int64_t decay_steps = a.integer("decay-steps", room / 5);
    CHECK(decay_steps >= 0 && decay_steps <= room, "--decay-steps must be between 0 and --steps - --warmup (%ld)",
          long(room));
    const auto lr_schedule = [&](int64_t s) {
        return schedule == "wsd" ? lr_wsd(s, warmup, steps, decay_steps, lr, min_frac)
                                 : lr_at(s, warmup, steps, lr, min_frac);
    };
    const std::string control_path = a.str("control", out + ".control");
    const double clip = a.real("clip", 1.0), bias_rate = a.real("bias-rate", 1e-3);
    const int64_t seed = a.integer("seed", 1);
    const int log_every = a.integer_int("log-every", 10), eval_every = a.integer_int("eval-every", 200);
    const int eval_batches = a.integer_int("eval-batches", 20), save_every = a.integer_int("save-every", 500);
    const double time_limit = a.real("time-limit", 0) * 60.0;
    const std::string reason_path = a.str("reason-test");
    const std::string reason_report_prefix = a.str("reason-report");
    CHECK(reason_report_prefix.empty() || !reason_path.empty(), "--reason-report requires --reason-test");
    const Tokenizer tok = load_tokenizer(a.str("bpe"));
    const int reason_every = a.integer_int("reason-every", 0), reason_limit = a.integer_int("reason-limit", 300);
    const int max_new = a.integer_int("max-new", 200);
    CHECK(max_new > 0, "--max-new must be positive");
    g_profiler.enabled = a.flag("profile");
    const bool use_gpu = a.flag("gpu"), request_fused = a.flag("fused-attn"), disable_fused = a.flag("no-fused-attn");
    const bool moe_cublas = a.flag("moe-cublas"), moe_grouped = a.flag("moe-grouped");
    CHECK(!moe_cublas || use_gpu, "--moe-cublas needs --gpu");
    CHECK(!moe_grouped || use_gpu, "--moe-grouped needs --gpu");
    CHECK(!moe_cublas || !moe_grouped, "choose only one of --moe-cublas and --moe-grouped");
    CHECK(!request_fused || !disable_fused, "choose only one of --fused-attn and --no-fused-attn");
    const bool no_tf32 = a.flag("no-tf32");
    // bf16 products by default: 1.45x the tokens/s of tf32 on the night-1
    // model, with the same loss curve. Ternary models too (2026-10-16): they
    // multiply their codes (gpu.cu's proj_forward), 1.45x tf32's tokens/s,
    // the same losses over 100-200 steps.
    const std::string asked = a.str("matmul", no_tf32 ? "fp32" : "bf16");
    CHECK(!no_tf32 || asked == "fp32", "--no-tf32 conflicts with --matmul %s", asked.c_str());
    CHECK(asked == "fp32" || asked == "tf32" || asked == "bf16", "--matmul takes fp32, tf32 or bf16");
    // The CPU engine multiplies in exact fp32 with its own attention: it
    // refuses the GPU's precision and attention options rather than ignore them.
    CHECK(use_gpu || !a.has("matmul") || asked == "fp32", "--matmul %s needs --gpu (the CPU engine computes in fp32)",
          asked.c_str());
    CHECK(use_gpu || (!request_fused && !disable_fused), "--fused-attn and --no-fused-attn need --gpu");
    const std::string matmul = use_gpu ? asked : "fp32";
    AdamW opt;
    opt.weight_decay = a.real("wd", 0.1);
    opt.muon = a.flag("muon");
    const std::string sampler = a.str("sampler", "random");
    CHECK(sampler == "random" || sampler == "qa", "--sampler takes random or qa");
    const double ul_alpha = a.real("unlikelihood", 0);
    const int ul_ngram = a.integer_int("ul-ngram", 8);
    CHECK(std::isfinite(ul_alpha) && ul_alpha >= 0, "--unlikelihood must be 0 (off) or positive");
    CHECK(ul_ngram >= 2, "--ul-ngram must be at least 2");
    a.done();
    CHECK(cfg.vocab >= int(data.vocab), "model vocab %d < data vocab %u", cfg.vocab, data.vocab);
    CHECK(val_path.empty() || cfg.vocab >= int(val_data.vocab),
          "model vocab %d < validation vocab %u", cfg.vocab, val_data.vocab);
    CHECK(reason_path.empty() || cfg.vocab >= tok.vocab_size(),
          "model vocab %d < reasoning tokenizer vocab %d", cfg.vocab, tok.vocab_size());
    CHECK(!cfg.moe() || cfg.top_k > 1,
          "--topk 1 cannot train this router: a one-logit softmax has zero task gradient; use --topk 2 or more");
    CHECK(log_every > 0 && eval_batches > 0, "log interval and evaluation batch count must be positive");
    std::vector<ValSource> val_sources;
    for (const auto& [name, path] : parse_val_sources(val_sources_spec)) {
        ValSource src{name, path, load_token_list(path), {}};
        CHECK(cfg.vocab >= int(src.data.vocab), "--val-sources %s: model vocab %d < its vocab %u", name.c_str(),
              cfg.vocab, src.data.vocab);
        if (!src.data.mask.empty()) {
            // Equal sizes are all this can check: the file must come from --bpe.
            CHECK(tok.vocab_size() == int(src.data.vocab),
                  "--val-sources %s: finding its final answers needs --bpe, the tokenizer it was made with "
                  "(%u tokens; --bpe gives %d)", name.c_str(), src.data.vocab, tok.vocab_size());
            src.final_only = src.data;
            src.final_only.mask = final_answer_mask(src.data, [&](int id) -> const std::string& {
                return tok.token_bytes(id);
            });
        }
        val_sources.push_back(std::move(src));
    }
    AccumulationBatch group(B, Tn, accum);

    Model<float> model(cfg);
    if (!use_gpu) model.init(uint64_t(seed));  // on the GPU, GpuTrainer::init_params below
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
        // Fused attention by default, for the head sizes it takes; cuBLAS
        // with --no-fused-attn (or larger heads).
        const bool fused_attn = request_fused || (!disable_fused && fused_attention_fits(cfg.head_size()));
        auto gpu_backend = std::make_unique<GpuBackend>(model, opt, mm, fused_attn);
        // The experts' products: grouped by default in bf16, where our
        // kernels beat cuBLAS per expert (2341 vs ~2460 ms a step on the
        // night model); cuBLAS per expert in TF32 (ternary models: ~3610-3680
        // vs 3810) and fp32. 2026-10-16, docs/research-plan.md "Speed" item 8.
        const bool grouped = moe_grouped || (!moe_cublas && mm == Matmul::bf16);
        gpu_backend->gpu.set_grouped_experts(grouped);
        if (!resume && init.empty()) {
            // A new model: its weights are made on the GPU, and copied back
            // for the host's checkpoints.
            gpu_backend->gpu.init_params(uint64_t(seed));
            gpu_backend->gpu.get_params(model.params.data());
            model.weights_dirty = true;
        }
        backend = std::move(gpu_backend);
        std::printf("gpu: %s, %s matmuls, %s attention%s\n", GpuTrainer::device_name().c_str(), matmul.c_str(),
                    !fused_attn                                              ? "cuBLAS"
                    : fused_attention_on_tensor_cores(mm, cfg.head_size()) ? "fused tensor-core"
                                                                           : "fused fp32",
                    cfg.n_experts == 0 ? "" : grouped ? ", grouped experts" : ", cuBLAS per expert");
#else
        (void)request_fused, (void)moe_cublas, (void)moe_grouped;
        CHECK(false, "this cxxgpt was built without CUDA; use cxxgpt-gpu (make gpu)");
#endif
    } else {
        backend = std::make_unique<CpuBackend>(model, opt);
    }
    Backend& be = *backend;
    be.set_unlikelihood(ul_alpha, ul_ngram);
    const bool ul_on = ul_alpha > 0;

    // Train on [0, train_end) of data; validate on val_data or the tail.
    size_t train_end = data.tokens.size();
    const TokenData* val = &val_data;
    size_t val_lo = 0, val_hi = val_data.tokens.size();
    if (val_path.empty()) {
        CHECK(val_frac >= 0 && val_frac < 1, "--val-frac must be in [0, 1)");
        if (repeated_data && val_frac>0)
            std::fprintf(stderr, "warning: default validation slices repeated data and can overlap training; use a separate --val file\n");
        train_end = size_t(double(data.tokens.size()) * (1.0 - val_frac));
        val = &data;
        val_lo = train_end;
        val_hi = data.tokens.size();
    }
    if (steps > opt.step) {
        CHECK(train_end >= size_t(Tn) + 1, "training range is shorter than a sequence");
        CHECK(data.mask.empty() || std::any_of(data.mask.begin() + 1, data.mask.begin() + train_end,
                                              [](uint8_t m) { return m != 0; }),
              "training range contains no predictable targets");
    }
    std::printf("data sources: %s\n", data_path.c_str());
    // --sampler qa: training windows start at question-answer documents
    // (QaStarts); validation keeps random windows, comparable across runs.
    std::unique_ptr<QaStarts> qa;
    if (sampler == "qa") {
        qa = std::make_unique<QaStarts>(data, Tn, kBos);
        std::printf("sampler: qa (%zu question-answer documents that fit in a window start their windows)\n",
                    qa->documents());
    }
    if (ul_on)
        std::printf("unlikelihood: alpha %g on tokens that would repeat a run of %d trained tokens "
                    "(validation stays cross-entropy only)\n", ul_alpha, ul_ngram);
    std::printf("data: %zu train tokens, %zu val tokens; %d x %d x %d = %zu tokens per step\n",
                train_end, val_hi - val_lo, B, Tn, accum, group.inputs.size());

    FILE* log = nullptr;
    if (!log_path.empty()) {
        const bool fresh = !std::filesystem::exists(log_path);
        log = create_file(log_path, "a");
        CHECK(log, "cannot open %s", log_path.c_str());
        if (fresh)
            std::fprintf(log, "step,loss,val_loss,lr,grad_norm,ms_per_step,load_ratio%s\n",
                         ul_on ? ",unlikelihood,ul_fired" : "");
        if (resume) std::fprintf(log, "# resumed at step %ld\n", opt.step);
    }
    ExpertLoadLog load_interval(cfg);
    FILE* load_log = nullptr;
    if (!load_log_path.empty() && cfg.moe()) {  // a dense model has no experts: no file
        const std::string header = ExpertLoadLog::header(cfg.n_experts);
        const bool fresh = !std::filesystem::exists(load_log_path) || std::filesystem::file_size(load_log_path) == 0;
        if (!fresh) {
            std::ifstream previous(load_log_path);
            std::string first_line;
            std::getline(previous, first_line);
            CHECK(first_line == header, "expert load log header differs: use a new --load-log file");
        }
        load_log = create_file(load_log_path, "a");
        CHECK(load_log, "cannot open %s", load_log_path.c_str());
        if (fresh) std::fprintf(load_log, "%s\n", header.c_str());
        if (resume) std::fprintf(load_log, "# resumed at step %ld\n", opt.step);
    }
    // The "under 1%" warning lines come with the load log, not without it.
    FILE* const load_warnings = load_log ? stdout : nullptr;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    if (schedule == "wsd")
        std::printf("schedule: wsd, lr %.3g from step %ld, cooling down to %.3g over steps %ld-%ld\n", lr,
                    long(warmup), lr * min_frac, long(std::max(warmup, steps - decay_steps)), long(steps));
    else
        std::printf("schedule: cosine, lr %.3g after %ld warmup steps, down to %.3g at step %ld\n", lr,
                    long(warmup), lr * min_frac, long(steps));
    // A control file already there keeps its lr-scale; its other commands
    // are from before (a stop left from the last run would stop this one).
    ControlFile control(control_path);
    double lr_scale = 1;
    if (const auto c = control.read_now()) {
        lr_scale = c->lr_scale;
        std::printf("control file %s is already there: lr-scale %g holds; the other commands in it wait "
                    "for a change of the file\n", control_path.c_str(), lr_scale);
        for (const auto& e : c->errors) std::printf("control: %s: %s, ignored\n", control_path.c_str(), e.c_str());
    } else {
        std::printf("control file: %s (lr-scale X, save, eval, reason, stop; see train --help)\n",
                    control_path.c_str());
    }
    if (opt.step > 0) std::printf("learning rate at step %ld: %.3g\n", opt.step, lr_scale * lr_schedule(opt.step));

    // Optional exact-answer reasoning eval during training, on an even
    // sample of the test set.
    std::vector<Problem> reason_test;
    std::vector<size_t> reason_ids;
    ReasonReport reason_report;
    if (!reason_path.empty()) {
        const std::vector<Problem> all = load_problems(reason_path);
        const size_t n = std::min(all.size(), size_t(std::max(1, reason_limit)));
        for (size_t i = 0; i < n; ++i) {
            reason_ids.push_back(i * all.size() / n);
            reason_test.push_back(all[reason_ids.back()]);
        }
    }
    auto reason_eval = [&](long at_step) {
        if (reason_test.empty()) return;
        const double t0 = now_seconds();
        const EvalResult r = be.reason(reason_test, tok, max_new, !reason_report_prefix.empty());
        if (r.total == 0) return;  // skipped (no room on the GPU)
        std::printf("reasoning eval at step %ld (%.1f s):\n", at_step, now_seconds() - t0);
        print_eval(r, 1);  // one wrong answer, to see what the model writes
        if (!reason_report_prefix.empty()) {
            auto str = [](const std::string& s) { return "\"" + labels::json_escape(s) + "\""; };
            const std::string settings = "{\"type\":\"settings\",\"step\":" + std::to_string(at_step) +
                ",\"out\":" + str(out) + ",\"test\":" + str(reason_path) + ",\"bpe\":" + str(a.str("bpe")) +
                ",\"effort\":" + str(r.effort) + ",\"max_new\":" + std::to_string(max_new) +
                ",\"gpu_requested\":" + (use_gpu ? "true" : "false") + ",\"matmul\":" + str(matmul) +
                ",\"seed\":" + std::to_string(seed) + "}";
            reason_report.write(reason_report_prefix, at_step, settings, reason_test, r.completions, reason_ids,
                                [&](const std::string& c) { return tok.encode(c); });
        }
        if (log) {
            std::fprintf(log, "# reasoning step %ld: %d/%d\n", at_step, r.correct, r.total);
            std::fflush(log);
        }
    };

    // Each --val-sources file's loss, on one line and in --val-log.
    FILE* val_log = nullptr;
    if (!val_log_path.empty()) {
        const bool fresh = !std::filesystem::exists(val_log_path);
        val_log = create_file(val_log_path, "a");
        CHECK(val_log, "cannot open %s", val_log_path.c_str());
        if (fresh) std::fprintf(val_log, "step,name,part,targets,loss\n");
    }
    auto source_eval = [&](long at_step) {
        if (val_sources.empty()) return;
        const double t0 = now_seconds();
        const auto forward = [&](const int* inp, const int* tgt, int b, int t) {
            return be.forward(inp, tgt, b, t, false);
        };
        std::string line;
        for (const ValSource& src : val_sources) {
            const std::vector<ValSourceLoss> parts = score_val_source(forward, src, B, Tn, eval_batches);
            char buf[160];
            line += line.empty() ? "" : " | ";
            for (size_t i = 0; i < parts.size(); ++i) {
                const ValSourceLoss& p = parts[i];
                if (i == 0) std::snprintf(buf, sizeof buf, "%s %.4f", p.name.c_str(), p.loss);
                else std::snprintf(buf, sizeof buf, " (%s %.4f)", p.part.c_str(), p.loss);
                line += buf;
                if (val_log)
                    std::fprintf(val_log, "%ld,%s,%s,%zu,%.5f\n", at_step, p.name.c_str(), p.part.c_str(), p.targets,
                                 p.loss);
            }
            if (parts.empty()) line += src.name + " (no trained targets)";
        }
        std::printf("val sources at step %ld (%.1f s): %s\n", at_step, now_seconds() - t0, line.c_str());
        std::fflush(stdout);
        if (val_log) std::fflush(val_log);
    };
    if (eval_every > 0 && steps > opt.step) source_eval(opt.step);  // where this start begins

    const double t_start = now_seconds();
    double loss_acc = 0, ms_acc = 0;
    int n_acc = 0;
    Backend::UnlikelihoodStats ul_acc;  // over the same steps as loss_acc
    long step = opt.step;
    const long first_step = step;
    size_t empty_groups = 0;
    constexpr int kMaxBadSteps = 10;
    int bad_steps = 0;
    bool diverged = false;
    long last_validation_step = -1;
    long saved_step = resume ? first_step : -1;  // what --out holds
    for (; step < steps && !g_stop;) {
        // The control file, when it has changed (and settled): acted on
        // before the step.
        if (const auto c = control.poll(now_seconds())) {
            auto note = [&](const std::string& what) {
                std::printf("control: %s at step %ld\n", what.c_str(), step);
                if (log) std::fprintf(log, "# control step %ld: %s\n", step, what.c_str());
            };
            for (const auto& e : c->errors) std::printf("control: %s: %s, ignored\n", control_path.c_str(), e.c_str());
            if (c->lr_scale != lr_scale) {
                lr_scale = c->lr_scale;
                char what[96];
                std::snprintf(what, sizeof what, "lr-scale %g (lr %.3g now)", lr_scale, lr_scale * lr_schedule(step));
                note(what);
            }
            if (c->save) {
                be.sync();
                if (save_checkpoint(out, model, opt, data_rng, true)) saved_step = opt.step;
                note("saved " + out);
            }
            if (c->eval) {
                if (val_hi >= val_lo + Tn + 1) {
                    const double vloss = eval_loss(be, *val, val_lo, val_hi, B, Tn, eval_batches);
                    last_validation_step = step;
                    char what[64];
                    std::snprintf(what, sizeof what, "validation %.5f", vloss);
                    note(what);
                    source_eval(step);
                } else {
                    note("eval skipped: no validation range a window long");
                }
            }
            if (c->reason) {
                if (reason_test.empty()) note("reason skipped: no --reason-test");
                else reason_eval(step);
            }
            if (log) std::fflush(log);
            if (c->stop) {
                note("stop");
                g_stop = 1;
                break;
            }
        }
        const double t0 = now_seconds();
        group.sample(data, data_rng, 0, train_end, qa.get());
        if (!group.total_valid) {
            if (++empty_groups == 1) std::printf("skipping sampled groups with no predictable targets\n");
            if (time_limit > 0 && now_seconds() - t_start > time_limit) {
                std::printf("time limit reached\n");
                break;
            }
            continue;  // no Adam moments, decay, balancing, or schedule step
        }
        const double lr_now = lr_scale * lr_schedule(step);
        const Backend::StepResult r = be.train_step(group, lr_now, clip, bias_rate, &load_interval);
        const Backend::UnlikelihoodStats ul_step = be.take_unlikelihood();  // this attempt's only
        const double loss = r.loss, gnorm = r.gnorm;
        // A non-finite loss or gradient leaves the weights untouched
        // (Backend::train_step). A run of them stops training rather than
        // spending the night on NaNs.
        if (!r.ok) {
            std::printf("step %ld: loss %g, gradient norm %g; update skipped\n", step + 1, loss, gnorm);
            if (++bad_steps == kMaxBadSteps) {
                std::printf("stopping: %d diverged steps in a row\n", kMaxBadSteps);
                diverged = true;
                break;
            }
            if (time_limit > 0 && now_seconds() - t_start > time_limit) {
                std::printf("time limit reached\n");
                break;  // skipped attempts below the streak limit do not imply divergence
            }
            continue;  // failed attempts do not advance optimizer or schedule
        }
        bad_steps = 0;
        const double ms = (now_seconds() - t0) * 1e3;
        if (step == first_step && be.device_bytes() > 0)
            std::printf("gpu memory: %.0f MB after the first step\n", double(be.device_bytes()) / (1 << 20));
        loss_acc += loss;
        ms_acc += ms;
        ++n_acc;
        ul_acc.sum += ul_step.sum;
        ul_acc.targets += ul_step.targets;
        ul_acc.fired += ul_step.fired;
        ul_acc.negatives += ul_step.negatives;

        const bool last = step + 1 == steps;
        double vloss = -1;
        if (eval_every > 0 && ((step + 1) % eval_every == 0 || last) && val_hi >= val_lo + Tn + 1) {
            vloss = eval_loss(be, *val, val_lo, val_hi, B, Tn, eval_batches);
            last_validation_step = step + 1;
        }
        if ((step + 1) % log_every == 0 || last || vloss >= 0) {
            const double load = be.load_ratio();  // read from the GPU only for the log
            std::printf("step %6ld | loss %.4f", step + 1, loss_acc / n_acc);
            if (vloss >= 0) std::printf(" | val %.4f", vloss);
            std::printf(" | lr %.2e | gnorm %6.3f | %6.1f ms | %7.0f tok/s", lr_now, gnorm,
                        ms_acc / n_acc, double(group.inputs.size()) / (ms_acc / n_acc) * 1e3);
            if (cfg.moe()) std::printf(" | load %.2f", load);
            // The term without alpha, per trained target; and the share of
            // targets with a token to push down.
            const double ul_mean = ul_acc.targets ? ul_acc.sum / double(ul_acc.targets) : 0.0;
            const double ul_fired = ul_acc.targets ? double(ul_acc.fired) / double(ul_acc.targets) : 0.0;
            if (ul_on)
                std::printf(" | ul %.4g on %.3f%% of targets (%zu negatives)", ul_mean, 100 * ul_fired,
                            ul_acc.negatives);
            std::printf("\n");
            std::fflush(stdout);
            if (log) {
                std::fprintf(log, "%ld,%.5f,%.5f,%.3e,%.4f,%.1f,%.3f", step + 1, loss_acc / n_acc,
                             vloss, lr_now, gnorm, ms_acc / n_acc, load);
                if (ul_on) std::fprintf(log, ",%.6f,%.6f", ul_mean, ul_fired);
                std::fprintf(log, "\n");
                std::fflush(log);
            }
            loss_acc = ms_acc = 0;
            n_acc = 0;
            ul_acc = {};
        }
        if (vloss >= 0) source_eval(step + 1);  // with the validation, after its log line
        if ((step + 1) % log_every == 0 || last) {
            be.flush_loads(load_interval);
            load_interval.write(step + 1, load_log, load_warnings);
        }
        if (save_every > 0 && (step + 1) % save_every == 0) {
            be.sync();
            if (save_checkpoint(out, model, opt, data_rng, true)) saved_step = opt.step;
        }
        if (reason_every > 0 && (step + 1) % reason_every == 0 && !last) reason_eval(step + 1);
        ++step;
        if (time_limit > 0 && now_seconds() - t_start > time_limit) {
            std::printf("time limit reached\n");
            break;
        }
    }
    // Also preserve a partial interval when a time limit, signal or failed
    // attempt stops the run between ordinary logging steps.
    be.flush_loads(load_interval);
    load_interval.write(opt.step, load_log, load_warnings);
    if (load_log) CHECK(std::fclose(load_log) == 0, "cannot close expert load log");
    if (g_profiler.enabled) g_profiler.report(int(step - first_step));
    if (empty_groups) std::printf("skipped %zu empty sampled groups\n", empty_groups);
    if (diverged) {
        // The weights are finite, but whatever overflowed is in them by now:
        // the last checkpoint is the one to keep.
        if (saved_step >= 0)
            std::printf("not saved: %s keeps step %ld; resume it with a lower --lr\n", out.c_str(), saved_step);
        else
            std::printf("not saved: no checkpoint yet; start again with a lower --lr\n");
        if (log) std::fclose(log);
        if (val_log) std::fclose(val_log);
        return 1;
    }
    // Time-limited runs may end between evaluations. SIGINT still saves promptly;
    // do not repeat a validation already logged for this successful update.
    if (!g_stop && eval_every > 0 && last_validation_step != opt.step && val_hi >= val_lo + Tn + 1) {
        const double vloss = eval_loss(be, *val, val_lo, val_hi, B, Tn, eval_batches);
        std::printf("step %6ld | val %.4f (final validation)\n", opt.step, vloss);
        source_eval(opt.step);
        // Numeric CSV rows keep their seven columns; comments already carry reasoning evals.
        if (log) std::fprintf(log, "# final validation step %ld: %.5f\n", opt.step, vloss);
    }
    be.sync();
    save_checkpoint(out, model, opt, data_rng);
    std::printf("saved %s at step %ld%s\n", out.c_str(), opt.step, g_stop ? " (interrupted)" : "");
    if (!g_stop) reason_eval(step);
    if (log) std::fclose(log);
    if (val_log) std::fclose(val_log);
    return 0;
}

// A checkpoint's loss on held-out token files, one line per file: whether
// training on math kept the model's general text (textbooks-val, wiki-val,
// doab-val, arxiv-val), without training. By default the same random
// windows as train --val, so with the same --batch and --batches (train's
// --eval-batches) the number matches training's last validation; --all
// scores every token once instead.
int cmd_eval_loss(Args& a) {
    if (a.flag("help")) {
        std::printf(
            "cxxgpt eval-loss --ckpt FILE --data a.tok[,b.tok...] [options]\n"
            "        the token files must come from the tokenizer the model was trained with\n"
            "        --batch 8 --batches 20 (night.sh's validation: 160 windows; train's --eval-batches)\n"
            "        --seq (default: the model's context)\n"
            "        --all: every token once instead of random windows (exact, slower on big files)\n"
            "        --gpu (cxxgpt-gpu) --matmul fp32 (tf32, bf16: as training computes it)\n");
        return 0;
    }
    ThreadPool tp(threads_arg(a));
    g_pool = &tp;
    const std::string ckpt = a.str("ckpt", "model.ckpt"), data_arg = a.str("data");
    CHECK(!data_arg.empty(), "usage: cxxgpt eval-loss --ckpt FILE --data a.tok[,b.tok...]");
    // night.sh's validation shape by default; on the 8 GB laptop GPU a
    // batch of 12 or more windows of 512 already spills into system memory
    // and runs about 10x slower.
    const int B = a.integer_int("batch", 8), batches = a.integer_int("batches", 20);
    CHECK(B > 0 && batches > 0, "--batch and --batches must be positive");
    const bool all = a.flag("all");
    const bool use_gpu = a.flag("gpu");
    CHECK(use_gpu || !a.has("matmul"), "--matmul needs --gpu");
    const std::string matmul = a.str("matmul", "fp32");
    CHECK(matmul == "fp32" || matmul == "tf32" || matmul == "bf16", "--matmul takes fp32, tf32 or bf16");
    Model<float> model(peek_checkpoint_config(ckpt));
    const int Tn = a.integer_int("seq", model.cfg.seq_len);
    CHECK(Tn > 0 && Tn <= model.cfg.seq_len, "--seq must be in [1, %d], the model's context", model.cfg.seq_len);
    a.done();
#ifndef CXXGPT_CUDA
    CHECK(!use_gpu, "this cxxgpt was built without CUDA; use cxxgpt-gpu (make gpu) for --gpu");
#endif
    load_checkpoint(ckpt, model, nullptr, nullptr);
    std::function<double(const int*, const int*, int, int)> forward = [&](const int* inp, const int* tgt, int b,
                                                                          int t) {
        return model.forward(inp, tgt, b, t, false);
    };
#ifdef CXXGPT_CUDA
    std::unique_ptr<GpuTrainer> gpu;
    if (use_gpu) {
        const Matmul mm = matmul == "fp32" ? Matmul::fp32 : matmul == "bf16" ? Matmul::bf16 : Matmul::tf32;
        const bool fused = fused_attention_fits(model.cfg.head_size());  // as train's default
        gpu = std::make_unique<GpuTrainer>(model.cfg, model.infos, mm, fused, /*inference=*/true);
        gpu->set_params(model.params.data());
        gpu->set_router_bias(model.router_bias);
        std::printf("gpu: %s, %s matmuls, %s attention\n", GpuTrainer::device_name().c_str(), matmul.c_str(),
                    !fused                                                     ? "cuBLAS"
                    : fused_attention_on_tensor_cores(mm, model.cfg.head_size()) ? "fused tensor-core"
                                                                                 : "fused fp32");
        forward = [&](const int* inp, const int* tgt, int b, int t) { return gpu->forward(inp, tgt, b, t, false); };
    }
#endif
    std::printf("%s: %s, %d x %d tokens a batch\n", ckpt.c_str(),
                all ? "every token once" : (std::to_string(batches) + " batches of random windows").c_str(), B, Tn);
    for (const std::string& path : split(data_arg, ',')) {
        const TokenData data = TokenData::load(path);
        CHECK(int(data.vocab) <= model.cfg.vocab, "%s has vocab %u, more than the model's %d: another tokenizer?",
              path.c_str(), data.vocab, model.cfg.vocab);
        const double t0 = now_seconds();
        size_t targets = 0;
        double loss = 0;
        if (all) {
            loss = full_loss(forward, data, 0, data.tokens.size(), B, Tn, &targets);
        } else {
            CHECK(data.tokens.size() >= size_t(Tn) + 1, "%s has %zu tokens, fewer than a window of %d", path.c_str(),
                  data.tokens.size(), Tn + 1);
            loss = sampled_loss(forward, data, 0, data.tokens.size(), B, Tn, batches);
        }
        std::printf("%s: loss %.4f", path.c_str(), loss);
        if (all) std::printf(" over %zu targets", targets);
        std::printf(" (%zu tokens, %.1f s)\n", data.tokens.size(), now_seconds() - t0);
    }
    return 0;
}

int cmd_sample(Args& a) {
    ThreadPool tp(threads_arg(a));
    g_pool = &tp;
    const std::string ckpt = a.str("ckpt", "model.ckpt"), bpe = a.str("bpe");
    std::string prompt = a.str("prompt");
    prompt.erase(std::remove(prompt.begin(),prompt.end(),char(kBos)),prompt.end());
    const int n = a.integer_int("n", 500);
    const double temp = a.real("temp", 0.8);
    const bool answer = a.flag("answer");  // stop after the "#### answer" line, as eval-reasoning does
    Rng rng(uint64_t(a.integer("seed", 1)));
    const calc::Mode tool = calculator_mode(a);
    const int no_repeat = a.integer_int("no-repeat", 0);  // as for eval-reasoning
    CHECK(no_repeat == 0 || no_repeat >= 2, "--no-repeat must be 0 (off) or at least 2");
    // On the GPU (cxxgpt-gpu): every token chosen there; exact fp32 products
    // by default, as eval-reasoning --gpu.
    const bool use_gpu = a.flag("gpu");
    CHECK(use_gpu || !a.has("matmul"), "--matmul needs --gpu");
    const std::string matmul = a.str("matmul", "fp32");
    CHECK(matmul == "fp32" || matmul == "tf32" || matmul == "bf16", "--matmul takes fp32, tf32 or bf16");
    a.done();
#ifndef CXXGPT_CUDA
    CHECK(!use_gpu, "this cxxgpt was built without CUDA; use cxxgpt-gpu (make gpu) for --gpu");
#endif
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
    calc::Counts counts;
    const auto stop = [&](const std::vector<int>& g) {
        return g.back() == kBos || (answer && completion_done(text_of(g)));
    };
    std::vector<std::vector<int>> out;
#ifdef CXXGPT_CUDA
    if (use_gpu) {
        const Matmul mm = matmul == "fp32" ? Matmul::fp32 : matmul == "bf16" ? Matmul::bf16 : Matmul::tf32;
        GpuTrainer gpu(model.cfg, model.infos, mm, false, /*inference=*/true);
        gpu.set_params(model.params.data());
        gpu.set_router_bias(model.router_bias);
        GpuDecoder dec{model.cfg, gpu};
        out = generate(dec, {ids}, n, stop, temp, &rng, nullptr, tool, &tok, &counts, no_repeat);
    }
#endif
    if (!use_gpu) out = generate(model, {ids}, n, stop, temp, &rng, nullptr, tool, &tok, &counts, no_repeat);
    std::vector<int> gen = out[0];
    if (!gen.empty() && gen.back() == kBos) gen.pop_back();
    const std::string text = text_of(gen);
    std::fwrite(text.data(), 1, text.size(), stdout);
    std::printf("\n");
    if (tool != calc::Mode::none)
        std::fprintf(stderr, "calculator calls: %zu fired, %zu malformed\n", counts.fired, counts.malformed);
    return 0;
}

// Single-sequence decoding speed and weight memory: the ternary engine
// (bitplanes, int8 kernels) against the same model in fp32 (vectorized,
// same threads). A/B runs are interleaved; medians are reported.
int cmd_bench_infer(Args& a) {
    ThreadPool tp(threads_arg(a));
    g_pool = &tp;
    const std::string ckpt = a.str("ckpt");
    if (a.help()) {
        std::printf("cxxgpt bench-infer [--ckpt FILE | model options]\n"
                    "    --tokens 128 --reps 7 --warmup 2 --threads N --log FILE.csv\n"
                    "Alternates fp32/packed run order on the same forced token stream.\n");
        return 0;
    }
    int tokens = a.integer_int("tokens", 128);
    const bool tokens_given = a.has("tokens");
    const int reps = a.integer_int("reps", 7);
    const int warmup = a.integer_int("warmup", 2);
    const std::string log_path = a.str("log");
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
    CHECK(reps > 0 && warmup >= 0, "--reps must be positive and --warmup nonnegative");

    // fp32 baseline: the quantized weights as plain floats, no activation
    // quantization.
    Model<float> fp32 = make_fp32_decode_baseline(model);
    TernaryEngine engine(model);

    Rng rng(7);
    std::vector<int> seq(tokens);
    for (int& t : seq) t = int(rng.below(cfg.vocab));
    // Quantized training-model reference, including converted checkpoints
    // in speed-only mode. The fp32 baseline skips int8 rounding, so
    // where the top two tokens are close (random input makes them close
    // often) it can pick the other one.
    std::vector<int> top_model;
    model.decode_begin(1, {0});
    for (int pos = 0; pos < tokens; ++pos)
        top_model.push_back(pick_token(model.decode_step(&seq[pos], pos), cfg.vocab, 0.0, nullptr));
    std::vector<double> t_fp32, t_tern;
    std::vector<int> top_fp32(tokens), top_tern(tokens);
    auto run_pair = [&](bool fp32_first, double& elapsed_fp32, double& elapsed_tern) {
        auto run_fp32 = [&] {
            const double t0 = now_seconds();
            fp32.decode_begin(1, {0});
            for (int pos = 0; pos < tokens; ++pos)
                top_fp32[pos] = pick_token(fp32.decode_step(&seq[pos], pos), cfg.vocab, 0.0, nullptr);
            elapsed_fp32 = now_seconds() - t0;
        };
        auto run_tern = [&] {
            const double t0 = now_seconds();
            engine.reset();
            for (int pos = 0; pos < tokens; ++pos)
                top_tern[pos] = pick_token(engine.step(seq[pos]), cfg.vocab, 0.0, nullptr);
            elapsed_tern = now_seconds() - t0;
        };
        if (fp32_first) { run_fp32(); run_tern(); }
        else { run_tern(); run_fp32(); }
        int same = 0, same_model = 0;
        for (int pos = 0; pos < tokens; ++pos) {
            same += top_fp32[pos] == top_tern[pos];
            same_model += top_model[pos] == top_tern[pos];
        }
        return std::pair<double, double>{double(same) / tokens, double(same_model) / tokens};
    };
    size_t fp32_weight_bytes = fp32.params.size() * sizeof(float);
    if (cfg.moe())
        for (const auto& bias : fp32.router_bias) fp32_weight_bytes += bias.size() * sizeof(float);
    FILE* log = nullptr;
    if (!log_path.empty()) {
        CHECK(!std::filesystem::exists(log_path), "benchmark log already exists: %s", log_path.c_str());
        log = std::fopen(log_path.c_str(), "w");
        CHECK(log, "cannot write %s", log_path.c_str());
        std::fprintf(log, "trial,fp32_first,tokens,threads,fp32_ms,ternary_ms,argmax_agreement,fp32_weight_bytes,ternary_weight_bytes,ternary_model_agreement\n");
    }
    double agree = 0, agree_model = 0;
    for (int r = 0; r < warmup; ++r) {
        double f, t;
        run_pair(r % 2 == 0, f, t);
    }
    for (int r = 0; r < reps; ++r) {
        double f, t;
        const auto [same, same_model] = run_pair(r % 2 == 0, f, t);
        t_fp32.push_back(f); t_tern.push_back(t);
        agree += same / reps; agree_model += same_model / reps;
        if (log) std::fprintf(log, "%d,%d,%d,%d,%.9f,%.9f,%.9f,%zu,%zu,%.9f\n", r,
                              int(r % 2 == 0), tokens, tp.size(), f * 1e3, t * 1e3, same,
                              fp32_weight_bytes, engine.weight_bytes(), same_model);
    }
    if (log) std::fclose(log);
    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        const size_t i = v.size() / 2;
        return v.size() % 2 ? v[i] : (v[i - 1] + v[i]) / 2;
    };
    const double f = tokens / median(t_fp32), t = tokens / median(t_tern);
    const double fb = double(fp32_weight_bytes), tb = double(engine.weight_bytes());
    std::printf("decode, 1 sequence, %d tokens, median of %d, %d threads (%d warmup pairs, alternating order):\n",
                tokens, reps, tp.size(), warmup);
    std::printf("  fp32     %9.1f tok/s   weights %8.2f MB\n", f, fb / 1e6);
    std::printf("  ternary  %9.1f tok/s   weights %8.2f MB\n", t, tb / 1e6);
    std::printf("  speedup %.2fx, memory %.1fx smaller\n", t / f, fb / tb);
    std::printf("  same next token as the ternary training-model path on %.0f%% of positions (as fp32: %.0f%%)\n",
                100 * agree_model, 100 * agree);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    // Startup/eval/save lines reach tee immediately. Microsoft's C library
    // has no line buffering (_IOLBF means full buffering there, and a size
    // of 0 is an invalid argument that aborts the program), so on Windows
    // stdout is unbuffered instead.
#ifdef _WIN32
    std::setvbuf(stdout, nullptr, _IONBF, 0);
#else
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
#endif
    struct Command {
        const char* name;
        int (*run)(Args&);
        const char* what;
    };
    static const Command commands[] = {
        {"bpe-train", cmd_bpe_train, "learns a BPE tokenizer from a text file"},
        {"text2tok", cmd_text2tok, "encodes text into a token file"},
        {"train", cmd_train, "trains a model, or resumes training"},
        {"clone-experts", cmd_clone_experts, "repairs saved MoE experts by copying a busy expert"},
        {"sample", cmd_sample, "generates text from a prompt"},
        {"gen-reasoning", cmd_gen_reasoning, "writes the synthetic reasoning curriculum"},
        {"gen-word", cmd_gen_word, "writes exact word problems with held-out sentence patterns"},
        {"gen-math", cmd_gen_math, "writes math problems (equations, fractions) that a second solver checks"},
        {"prep-qa", cmd_prep_qa, "turns question/answer JSON Lines into problems and token files"},
        {"eval-reasoning", cmd_eval_reasoning, "exact-answer accuracy on a problems file"},
        {"self-train", cmd_self_train, "keeps the model's own right, checked solutions as training tokens"},
        {"eval-loss", cmd_eval_loss, "a checkpoint's loss on held-out token files"},
        {"bench-infer", cmd_bench_infer, "times the ternary inference engine against fp32 decoding"},
        {"leak-check", cmd_leak_check, "finds questions that share a 13-word run with training files"},
        {"qa-context", cmd_qa_context, "how often training windows cut an answer off from its question"},
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
