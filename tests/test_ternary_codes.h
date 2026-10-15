// Ternary codes from training to inference (issue #40). Training, the
// inference packer and a checkpoint round trip must give every BitLinear
// weight the same code (-1, 0 or +1), including weights exactly at the
// rounding thresholds +-gamma / 2. A bf16 copy of the weights would not:
// rounding them to 8 significant bits moves weights just past a threshold
// back onto it, which is how a fine-tuning step can be lost on export.
// CxxGPT has no bf16 cast on that path (bf16 matmuls refuse ternary models,
// checkpoints keep fp32); the bf16 case here shows what these checks catch.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "../src/infer.h"
#include "../src/train.h"
#include "test_util.h"

// A float rounded to bf16 (round to nearest, ties to even, as
// __float2bfloat16) and back. Finite inputs only.
inline float round_to_bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    u &= 0xFFFF0000u;
    std::memcpy(&x, &u, 4);
    return x;
}

inline int ternary_sign(float wq) { return (wq > 0) - (wq < 0); }

// Weights at and next to the thresholds, each with the code it must get
// when gamma is exactly 1 (threshold 0.5; ties round to even, so 0.5 -> 0,
// and 1.5 -> 2 is clamped to 1).
inline std::vector<std::pair<float, int>> threshold_cases() {
    const float half = 0.5f;
    return {{half, 0},
            {std::nextafter(half, 1.0f), 1},
            {std::nextafter(half, 0.0f), 0},
            {-half, 0},
            {std::nextafter(-half, -1.0f), -1},
            {std::nextafter(-half, 0.0f), 0},
            {1.5f, 1},
            {-1.5f, -1},
            {0.0f, 0},
            {1e-30f, 0},
            {1.0f, 1}};
}

// Fills a segment with +-1 and puts each case at an even index, followed by
// a partner of magnitude 2 - |w|: mean |w| stays 1 to within ~1e-10, so
// gamma = float(mean) + 1e-8 is exactly 1. Returns the cases' indices.
inline std::vector<size_t> plant_threshold_cases(float* w, size_t n) {
    for (size_t i = 0; i < n; ++i) w[i] = i % 3 ? 1.0f : -1.0f;
    std::vector<size_t> at;
    const auto cases = threshold_cases();
    CHECK(2 * cases.size() <= n, "segment too small for the threshold cases");
    for (size_t c = 0; c < cases.size(); ++c) {
        w[2 * c] = cases[c].first;
        w[2 * c + 1] = 2.0f - std::abs(cases[c].first);
        at.push_back(2 * c);
    }
    return at;
}

// Every ternary segment's codes through the packer the engine uses, against
// the training forward's quantized weights: the same code everywhere, and
// for nonzero codes the same value (code times the packer's scale).
// Returns the number of mismatches.
inline size_t packer_mismatches(const Model<float>& m, size_t* checked = nullptr) {
    size_t bad = 0, n = 0;
    for (const ParamInfo& p : m.infos) {
        if (!p.ternary) continue;
        const size_t per = p.size / p.count;
        for (int c = 0; c < p.count; ++c) {
            const TernaryMatrixBitplane W = pack_ternary(m, p, c);
            const float* q = &m.qparams[p.offset + c * per];
            for (int r = 0; r < p.rows; ++r)
                for (int k = 0; k < p.cols; ++k) {
                    const float wq = q[size_t(r) * p.cols + k];
                    const int code = W.weight(r, k);
                    bad += code != ternary_sign(wq) || (code != 0 && float(code) * W.scale != wq);
                    ++n;
                }
        }
    }
    if (checked) *checked = n;
    return bad;
}

// Codes that differ between two models' quantized weights.
inline size_t code_changes(const Model<float>& a, const Model<float>& b) {
    size_t changed = 0;
    for (const ParamInfo& p : a.infos)
        if (p.ternary)
            for (size_t i = p.offset; i < p.offset + p.size; ++i)
                changed += ternary_sign(a.qparams[i]) != ternary_sign(b.qparams[i]);
    return changed;
}

inline void test_ternary_codes() {
    Config cfg;
    cfg.vocab = 40;
    cfg.seq_len = 24;
    cfg.d_model = 64;
    cfg.n_layers = 2;
    cfg.n_heads = 4;
    cfg.ternary = true;  // attention ternary too, so every kind of tensor is covered
    cfg.n_experts = 4;
    cfg.top_k = 2;
    cfg.n_shared = 1;
    cfg.expert_hidden = 64;
    Model<float> m(cfg);
    m.init(3);
    Rng rng(4);
    for (auto& w : m.params) w = float(rng.normal() * 0.3);

    // The threshold cases in a stacked tensor (expert 1 of layer 0) and a
    // plain one, so each copy's own scale is exercised.
    struct Planted {
        size_t offset;
        std::vector<size_t> at;
    };
    std::vector<Planted> planted;
    const std::pair<const char*, int> where[] = {{"l0.efc", 1}, {"l1.wqkv", 0}};
    for (const auto& [name, index] : where) {
        const ParamInfo* p = m.find(name);
        CHECK(p && p->ternary, "missing ternary tensor %s", name);
        const size_t per = p->size / p->count, offset = p->offset + size_t(index) * per;
        planted.push_back({offset, plant_threshold_cases(&m.params[offset], per)});
    }
    m.weights_dirty = true;
    m.quantize_weights();

    // Training: gamma is exactly 1 (a filler weight of +1 quantizes to
    // gamma itself), and each case gets its code.
    const auto cases = threshold_cases();
    for (const Planted& s : planted) {
        const float gamma = m.qparams[s.offset + 2 * cases.size() + 1];  // a +1 filler
        EXPECT(gamma == 1.0f, "the planted segment's gamma is %.9g, not 1", gamma);
        for (size_t c = 0; c < cases.size(); ++c) {
            const float wq = m.qparams[s.offset + s.at[c]];
            EXPECT(ternary_sign(wq) == cases[c].second, "w = %.9g got code %d (wq %.9g), expected %d",
                   cases[c].first, ternary_sign(wq), wq, cases[c].second);
        }
    }

    // The inference packer: the same code for every weight of every tensor.
    size_t checked = 0;
    const size_t bad = packer_mismatches(m, &checked);
    EXPECT(bad == 0 && checked > 0, "packer vs training: %zu of %zu codes differ", bad, checked);

    // A fine-tuning step of one ulp moves 0.5 just past the threshold
    // (code 0 -> +1). It survives a checkpoint, and the packer sees it.
    const size_t nudged = planted[0].offset + planted[0].at[0];
    m.params[nudged] = std::nextafter(0.5f, 1.0f);
    m.weights_dirty = true;
    m.quantize_weights();
    EXPECT(ternary_sign(m.qparams[nudged]) == 1, "the nudged weight's code is %d, not 1",
           ternary_sign(m.qparams[nudged]));
    const std::string path = test_path("ternary-codes.ckpt");
    AdamW opt;
    Rng data_rng(5);
    save_checkpoint(path, m, opt, data_rng);
    Model<float> loaded(peek_checkpoint_config(path));
    load_checkpoint(path, loaded, nullptr, nullptr);
    std::remove(path.c_str());
    loaded.quantize_weights();
    EXPECT(loaded.qparams == m.qparams, "a checkpoint round trip changed the quantized weights");
    EXPECT(code_changes(loaded, m) == 0 && packer_mismatches(loaded) == 0,
           "a checkpoint round trip changed %zu codes", code_changes(loaded, m));

    // What a bf16 copy of the master weights would do: the nudged weight
    // rounds back to 0.5 (code +1 -> 0), so the comparison above, run on
    // such a copy, fails.
    Model<float> bf16(cfg);
    bf16.params = loaded.params;
    for (auto& w : bf16.params) w = round_to_bf16(w);
    bf16.weights_dirty = true;
    bf16.quantize_weights();
    const size_t flipped = code_changes(bf16, m);
    EXPECT(ternary_sign(bf16.qparams[nudged]) == 0 && flipped > 0,
           "bf16: the nudged code is %d, %zu codes changed", ternary_sign(bf16.qparams[nudged]), flipped);
    size_t ternary_weights = 0;
    for (const ParamInfo& p : m.infos) ternary_weights += p.ternary ? p.size : 0;
    std::printf("  ternary codes: %zu checked, threshold cases exact, a bf16 copy would change %zu (%.2f%%)\n",
                checked, flipped, 100.0 * double(flipped) / double(ternary_weights));
}
