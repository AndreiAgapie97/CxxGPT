// One CPU configuration per process; array payload/capacity and Linux RSS.
// Process peak includes model construction and transient packing/cloning.
#include <fstream>
#include <memory>
#include <sstream>
#if defined(__linux__)
#include <sys/resource.h>
#endif
#include "../src/args.h"
#include "../src/infer.h"
#include "../src/train.h"

struct ProcessMemory { size_t rss = 0, peak = 0; };
ProcessMemory process_memory() {
    ProcessMemory out;
#if defined(__linux__)
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream row(line); std::string key; size_t kib;
        if (row >> key >> kib && key == "VmRSS:") out.rss = kib * 1024;
    }
    rusage u{};
    CHECK(getrusage(RUSAGE_SELF, &u) == 0, "getrusage failed");
    out.peak = size_t(u.ru_maxrss) * 1024; // Linux reports KiB
#endif
    return out;
}

int main(int argc, char** argv) {
    Args a(argc, argv, 1);
    const std::string mode = a.str("mode", "train"), ckpt = a.str("ckpt");
    Config cfg;
    cfg.vocab = int(a.integer("vocab", 256)); cfg.d_model = int(a.integer("dim", 64));
    cfg.n_layers = int(a.integer("layers", 2)); cfg.n_heads = int(a.integer("heads", 4));
    cfg.seq_len = int(a.integer("context", 128)); cfg.ffn_hidden = int(a.integer("ffn", 128));
    cfg.n_experts = int(a.integer("experts", 0)); cfg.top_k = int(a.integer("topk", 2));
    cfg.expert_hidden = int(a.integer("ehidden", 64)); cfg.n_shared = int(a.integer("shared", 0));
    cfg.ternary = a.flag("ternary");
    const int B = int(a.integer("batch", 2)), T = int(a.integer("seq", 32));
    const int steps = int(a.integer("steps", 3)), tokens = int(a.integer("tokens", 16));
    const int threads = int(a.integer("threads", 2));
    const bool muon = a.flag("muon");
    a.done();
    CHECK(mode == "train" || mode == "decode" || mode == "fp32" || mode == "packed", "invalid mode");
    if (!ckpt.empty()) cfg = peek_checkpoint_config(ckpt);
    CHECK(B > 0 && T > 0 && T <= cfg.seq_len && steps > 0 && tokens > 0 && tokens <= cfg.seq_len && threads > 0,
          "invalid workload");
    CHECK(mode != "packed" || (B == 1 && cfg.ternary), "packed needs batch one and ternary weights");
    ThreadPool tp(threads); g_pool = &tp;
    AdamW opt; opt.muon = muon;
    std::printf("mode,stage,category,payload_bytes,capacity_bytes,rss_bytes,peak_rss_bytes,params,vocab,dim,layers,context,batch,seq,source_ternary,muon\n");
    auto emit = [&](const char* stage, const char* category, ArrayBytes b, size_t params) {
        const auto p = process_memory();
        std::printf("%s,%s,%s,%zu,%zu,%zu,%zu,%zu,%d,%d,%d,%d,%d,%d,%d,%d\n", mode.c_str(), stage,
                    category, b.used, b.reserved, p.rss, p.peak, params, cfg.vocab, cfg.d_model,
                    cfg.n_layers, cfg.seq_len, B, T, int(cfg.ternary), int(muon));
    };
    emit("startup", "total", {}, 0);
    auto m = std::make_unique<Model<float>>(cfg); m->init(1);
    if (!ckpt.empty()) load_checkpoint(ckpt, *m, nullptr, nullptr);
    const size_t P = m->num_params();
    auto model_report = [&](const char* stage) {
        const auto b = m->memory_bytes();
        ArrayBytes optimizer; optimizer.add(opt.m); optimizer.add(opt.v);
        ArrayBytes total = b.total(); total += optimizer;
        emit(stage, "parameters", b.parameters, P); emit(stage, "gradients", b.gradients, P);
        emit(stage, "quantized", b.quantized, P); emit(stage, "optimizer", optimizer, P);
        emit(stage, "routing", b.routing, P); emit(stage, "rope", b.rope, P);
        emit(stage, "activations", b.activations, P); emit(stage, "backward", b.backward, P);
        emit(stage, "scratch", b.scratch, P); emit(stage, "kv", b.kv, P); emit(stage, "total", total, P);
    };
    model_report("loaded");
    if (mode == "packed") {
        TernaryEngine engine(*m); m.reset(); // release source before ready/work stages
        auto report = [&](const char* stage) {
            const auto b = engine.memory_bytes();
            emit(stage, "weights", b.weights, P); emit(stage, "rope", b.rope, P);
            emit(stage, "kv", b.kv, P); emit(stage, "scratch", b.scratch, P);
            emit(stage, "total", b.total(), P);
            emit(stage, "logical_weights", {engine.weight_bytes(), engine.weight_bytes()}, P);
        };
        report("ready");
        for (int t = 0; t < tokens; ++t) {
            const float* logits = engine.step((t * 7 + 1) % cfg.vocab);
            for (int v = 0; v < cfg.vocab; ++v) CHECK(std::isfinite(logits[v]), "non-finite packed output");
        }
        report("worked");
    } else if (mode != "train") {
        if (mode == "fp32") {
            auto clone = std::make_unique<Model<float>>(make_fp32_decode_baseline(*m));
            m = std::move(clone);
        }
        m->decode_begin(B, std::vector<int>(B, 0)); model_report("ready");
        std::vector<int> input(B);
        for (int t = 0; t < tokens; ++t) {
            for (int b = 0; b < B; ++b) input[b] = (t * 7 + b + 1) % cfg.vocab;
            const float* logits = m->decode_step(input.data(), t);
            for (int v = 0; v < B * cfg.vocab; ++v) CHECK(std::isfinite(logits[v]), "non-finite output");
        }
        model_report("worked");
    } else {
        std::vector<int> input(size_t(B) * T), target(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = int(i % size_t(cfg.vocab)); target[i] = (input[i] + 1) % cfg.vocab;
        }
        m->allocate(B, T); model_report("ready");
        for (int i = 0; i < steps; ++i) {
            m->zero_grad();
            CHECK(std::isfinite(m->forward(input.data(), target.data(), B, T, true)), "non-finite loss");
            m->backward(); CHECK(std::isfinite(clip_grad_norm(*m, 1)), "non-finite gradients");
            opt.update(*m, 0.001); m->update_router_bias(0.001);
        }
        model_report("worked");
    }
}
