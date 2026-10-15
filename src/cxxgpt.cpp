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
#include <charconv>
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
#include "common.h"
#include "generate.h"
#ifdef CXXGPT_CUDA
#include "gpu.h"
#endif
#include "infer.h"
#include "json.h"
#include "loss_mask.h"
#include "model.h"
#include "reasoning.h"
#include "wordgen.h"
#include "profile.h"
#include "tokenizer.h"
#include "train.h"

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

int cmd_bpe_train(Args& a) {
    const std::string in = a.str("in"), out = a.str("out");
    const int vocab = a.integer_int("vocab", 4096);
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

// BOS and the prompt (not trained on), then the completion (trained on).
void append_problem(TokenData& d, const Problem& p, const Tokenizer& tok) {
    const std::vector<int> prompt = prompt_tokens(p, tok);
    for (int t : prompt) d.tokens.push_back(uint16_t(t)), d.mask.push_back(0);
    encode_loss_marked(p.completion(), tok, d.tokens, d.mask);
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
    const long n_train = a.integer("train", 200000), n_val = a.integer("val", 2000);
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

    auto make_tokens = [&](long n, const std::string& path, bool training) {
        FILE* text = training && mistakes > 0 ? create_file(out + ".train.text", "wb") : nullptr;
        CHECK(!training || mistakes == 0 || text, "cannot write reasoning training text");
        TokenData d;
        d.vocab = uint32_t(tok.vocab_size());
        long skipped = 0;
        int tries = 0;
        for (long i = 0; i < n;) {
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
        std::printf("%s: %ld problems, %zu tokens (%ld test duplicates skipped)\n", path.c_str(), n,
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
int cmd_gen_word(Args& a) {
    const std::string out = a.str("out", "data/word");
    const long count = a.integer("count", 1000), seed = a.integer("seed", 1);
    const long min_steps = a.integer("min-steps", 2), max_steps = a.integer("max-steps", 8);
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
    save_problems(out + ".txt", problems);
    const std::string text_path = out + ".text";
    FILE* text = create_file(text_path, "w");
    CHECK(text, "cannot write %s", text_path.c_str());
    for (const auto& p : problems)
        CHECK(std::fprintf(text, "%s%s<|endoftext|>\n", p.prompt().c_str(), p.completion().c_str()) >= 0,
              "error writing %s", text_path.c_str());
    CHECK(std::fclose(text) == 0, "error writing %s", text_path.c_str());
    std::printf("%s.txt and %s.text: %zu %s word problems, %ld..%ld steps\n", out.c_str(), out.c_str(),
                problems.size(), split.c_str(), min_steps, max_steps);
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
    std::printf("gold completions exceeding --max-new %d: %zu / %d\n", r.max_new, r.gold_over_cap, r.total);
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
    const int batch = a.integer_int("batch", 64), limit = a.integer_int("limit", 0);
    const bool show = a.flag("show-failures");
    const int max_new = a.integer_int("max-new", 200);
    CHECK(max_new > 0, "--max-new must be positive");
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
    const EvalResult r = eval_reasoning(model, test, tok, batch, max_new);
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
    TokenLossMean total;
    for (int i = 0; i < batches; ++i) {
        data.sample_batch(rng, B, Tn, lo, hi, inp.data(), tgt.data());
        total.add(be.forward(inp.data(), tgt.data(), B, Tn, false), count_targets(tgt.data(), tgt.size()));
    }
    return total.value();
}

int cmd_train(Args& a) {
    if (a.flag("help")) {
        std::printf(
            "cxxgpt train --data FILE[,FILE*N...] [options]  (*N repeats a file N times)\n"
            "data:   --val FILE (default: last --val-frac of --data, 0.05)\n"
            "        --bpe tok.bpe (the tokenizer the data was made with, for --reason-test)\n"
            "model:  --seq 256 --dim 256 --layers 4 --heads 4 --ffn 0(auto)\n"
            "        --experts 0 --topk 2 --ehidden 0(auto) --shared 0 --ternary\n"
            "        --fp-attn (with --ternary: attention stays full precision; --fp-attn 0 clears it on --init)\n"
            "        --nope-every 0 (N: every Nth layer has no position encoding; 1: none has)\n"
            "train:  --batch 16 --accum 1 --steps 2000 --lr 1e-3 --min-lr-frac 0.1\n"
            "        --warmup 100 --wd 0.1 --clip 1.0 --bias-rate 1e-3 --seed 1\n"
            "        --muon (Muon instead of AdamW for the matrices in the blocks)\n"
            "start:  --init FILE (FILE's weights and model; new optimizer and schedule; data order from --seed;\n"
            "        with --ternary, the weights turn ternary from here on)\n"
            "output: --out model.ckpt --resume --log FILE.csv --log-every 10\n"
            "        --eval-every 200 --eval-batches 20 --save-every 500\n"
            "        --time-limit MINUTES --threads N --profile\n"
            "gpu:    --gpu (train on the GPU; needs cxxgpt-gpu, from make gpu)\n"
            "        --matmul tf32 (tensor cores; fp32: exact, slower; bf16: inputs stored in bf16,\n"
            "        coarser and about 1.4x faster; not for ternary models)\n"
            "        --no-tf32 is an alias for fp32\n"
            "        attention is fused by default (no stored weight matrix; on the tensor cores\n"
            "        with tf32/bf16); --no-fused-attn uses cuBLAS products instead\n"
            "reason: --reason-test FILE --reason-every STEPS --reason-limit 300 --max-new 200\n"
            "Ctrl-C saves a checkpoint and stops; --resume continues from --out, and\n"
            "takes precedence over --init once --out exists.\n");
        return 0;
    }
    ThreadPool tp(threads_arg(a));
    g_pool = &tp;
    const std::string data_path = a.str("data"), val_path = a.str("val");
    const std::string out = a.str("out", "model.ckpt"), log_path = a.str("log"), init = a.str("init");
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
    const long steps = a.integer("steps", 2000), warmup = a.integer("warmup", 100);
    const double lr = a.real("lr", 1e-3), min_frac = a.real("min-lr-frac", 0.1);
    const double clip = a.real("clip", 1.0), bias_rate = a.real("bias-rate", 1e-3);
    const long seed = a.integer("seed", 1);
    const int log_every = a.integer_int("log-every", 10), eval_every = a.integer_int("eval-every", 200);
    const int eval_batches = a.integer_int("eval-batches", 20), save_every = a.integer_int("save-every", 500);
    const double time_limit = a.real("time-limit", 0) * 60.0;
    const std::string reason_path = a.str("reason-test");
    const Tokenizer tok = load_tokenizer(a.str("bpe"));
    const int reason_every = a.integer_int("reason-every", 0), reason_limit = a.integer_int("reason-limit", 300);
    const int max_new = a.integer_int("max-new", 200);
    CHECK(max_new > 0, "--max-new must be positive");
    g_profiler.enabled = a.flag("profile");
    const bool use_gpu = a.flag("gpu"), request_fused = a.flag("fused-attn"), disable_fused = a.flag("no-fused-attn");
    CHECK(!request_fused || !disable_fused, "choose only one of --fused-attn and --no-fused-attn");
    const bool no_tf32 = a.flag("no-tf32");
    const std::string matmul = a.str("matmul", no_tf32 ? "fp32" : "tf32");
    CHECK(!no_tf32 || matmul == "fp32", "--no-tf32 conflicts with --matmul %s", matmul.c_str());
    CHECK(matmul == "fp32" || matmul == "tf32" || matmul == "bf16", "--matmul takes fp32, tf32 or bf16");
    AdamW opt;
    opt.weight_decay = a.real("wd", 0.1);
    opt.muon = a.flag("muon");
    a.done();
    CHECK(cfg.vocab >= int(data.vocab), "model vocab %d < data vocab %u", cfg.vocab, data.vocab);
    CHECK(val_path.empty() || cfg.vocab >= int(val_data.vocab),
          "model vocab %d < validation vocab %u", cfg.vocab, val_data.vocab);
    CHECK(reason_path.empty() || cfg.vocab >= tok.vocab_size(),
          "model vocab %d < reasoning tokenizer vocab %d", cfg.vocab, tok.vocab_size());
    CHECK(!cfg.moe() || cfg.top_k > 1,
          "--topk 1 cannot train this router: a one-logit softmax has zero task gradient; use --topk 2 or more");
    CHECK(log_every > 0 && eval_batches > 0, "log interval and evaluation batch count must be positive");
    AccumulationBatch group(B, Tn, accum);

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
        // Fused attention by default, for the head sizes it takes; cuBLAS
        // with --no-fused-attn (or larger heads).
        const bool fused_attn = request_fused || (!disable_fused && fused_attention_fits(cfg.head_size()));
        backend = std::make_unique<GpuBackend>(model, opt, mm, fused_attn);
        std::printf("gpu: %s, %s matmuls, %s attention\n", GpuTrainer::device_name().c_str(), matmul.c_str(),
                    !fused_attn                                              ? "cuBLAS"
                    : fused_attention_on_tensor_cores(mm, cfg.head_size()) ? "fused tensor-core"
                                                                           : "fused fp32");
#else
        (void)request_fused;
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
    std::printf("data: %zu train tokens, %zu val tokens; %d x %d x %d = %zu tokens per step\n",
                train_end, val_hi - val_lo, B, Tn, accum, group.inputs.size());

    FILE* log = nullptr;
    if (!log_path.empty()) {
        const bool fresh = !std::filesystem::exists(log_path);
        log = create_file(log_path, "a");
        CHECK(log, "cannot open %s", log_path.c_str());
        if (fresh) std::fprintf(log, "step,loss,val_loss,lr,grad_norm,ms_per_step,load_ratio\n");
        if (resume) std::fprintf(log, "# resumed at step %ld\n", opt.step);
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
        const EvalResult r = eval_reasoning(model, reason_test, tok, 64, max_new);
        std::printf("reasoning eval at step %ld (%.1f s):\n", at_step, now_seconds() - t0);
        print_eval(r, 1);  // one wrong answer, to see what the model writes
        if (log) {
            std::fprintf(log, "# reasoning step %ld: %d/%d\n", at_step, r.correct, r.total);
            std::fflush(log);
        }
    };

    const double t_start = now_seconds();
    double loss_acc = 0, ms_acc = 0;
    int n_acc = 0;
    long step = opt.step;
    const long first_step = step;
    size_t empty_groups = 0;
    constexpr int kMaxBadSteps = 10;
    int bad_steps = 0;
    bool diverged = false;
    long last_validation_step = -1;
    long saved_step = resume ? first_step : -1;  // what --out holds
    for (; step < steps && !g_stop;) {
        const double t0 = now_seconds();
        group.sample(data, data_rng, 0, train_end);
        if (!group.total_valid) {
            if (++empty_groups == 1) std::printf("skipping sampled groups with no predictable targets\n");
            if (time_limit > 0 && now_seconds() - t_start > time_limit) {
                std::printf("time limit reached\n");
                break;
            }
            continue;  // no Adam moments, decay, balancing, or schedule step
        }
        const double loss = accumulated_backward(be, group);
        const double lr_now = lr_at(step, warmup, steps, lr, min_frac);
        const double gnorm = std::isfinite(loss) ? be.step(lr_now, clip)
                                               : std::numeric_limits<double>::quiet_NaN();
        // A non-finite gradient leaves the weights untouched (Backend::step). A run
        // of them stops training rather than spending the night on NaNs.
        if (!std::isfinite(gnorm) || !std::isfinite(loss)) {
            std::printf("step %ld: loss %g, gradient norm %g; update skipped\n", step + 1, loss, gnorm);
            be.balance(0);  // discard this attempt's loads without changing biases
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
        const double load = be.balance(bias_rate);
        const double ms = (now_seconds() - t0) * 1e3;
        if (step == first_step && be.device_bytes() > 0)
            std::printf("gpu memory: %.0f MB after the first step\n", double(be.device_bytes()) / (1 << 20));
        loss_acc += loss;
        ms_acc += ms;
        ++n_acc;

        const bool last = step + 1 == steps;
        double vloss = -1;
        if (eval_every > 0 && ((step + 1) % eval_every == 0 || last) && val_hi >= val_lo + Tn + 1) {
            vloss = eval_loss(be, *val, val_lo, val_hi, B, Tn, eval_batches);
            last_validation_step = step + 1;
        }
        if ((step + 1) % log_every == 0 || last || vloss >= 0) {
            std::printf("step %6ld | loss %.4f", step + 1, loss_acc / n_acc);
            if (vloss >= 0) std::printf(" | val %.4f", vloss);
            std::printf(" | lr %.2e | gnorm %6.3f | %6.1f ms | %7.0f tok/s", lr_now, gnorm,
                        ms_acc / n_acc, double(group.inputs.size()) / (ms_acc / n_acc) * 1e3);
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
            if (save_checkpoint(out, model, opt, data_rng, true)) saved_step = opt.step;
        }
        if (reason_every > 0 && (step + 1) % reason_every == 0 && !last) reason_eval(step + 1);
        ++step;
        if (time_limit > 0 && now_seconds() - t_start > time_limit) {
            std::printf("time limit reached\n");
            break;
        }
    }
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
        return 1;
    }
    // Time-limited runs may end between evaluations. SIGINT still saves promptly;
    // do not repeat a validation already logged for this successful update.
    if (!g_stop && eval_every > 0 && last_validation_step != opt.step && val_hi >= val_lo + Tn + 1) {
        const double vloss = eval_loss(be, *val, val_lo, val_hi, B, Tn, eval_batches);
        std::printf("step %6ld | val %.4f (final validation)\n", opt.step, vloss);
        // Numeric CSV rows keep their seven columns; comments already carry reasoning evals.
        if (log) std::fprintf(log, "# final validation step %ld: %.5f\n", opt.step, vloss);
    }
    be.sync();
    save_checkpoint(out, model, opt, data_rng);
    std::printf("saved %s at step %ld%s\n", out.c_str(), opt.step, g_stop ? " (interrupted)" : "");
    if (!g_stop) reason_eval(step);
    if (log) std::fclose(log);
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
    std::setvbuf(stdout, nullptr, _IOLBF, 0);  // startup/eval/save lines reach tee immediately
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
        {"gen-word", cmd_gen_word, "writes exact word problems with held-out sentence patterns"},
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
