#pragma once
#include "../src/infer.h"
#include "test_util.h"

inline void test_memory() {
    // Independent dense shape formula, not a second enumeration of buffers.
    Config cfg; cfg.vocab = 32; cfg.seq_len = 8; cfg.d_model = 32;
    cfg.n_heads = 2; cfg.n_layers = 2; cfg.ffn_hidden = 64;
    Model<float> m(cfg); m.init(1);
    auto b = m.memory_bytes();
    EXPECT(b.parameters.used == m.num_params() * 4 && b.gradients.used == b.parameters.used &&
           b.quantized.used == 0 && m.qparams.empty(), "fp32 model keeps an unused quantized copy");
    EXPECT(b.rope.used == size_t(8 * 16 * 4) && b.routing.used == size_t(2 * (4 + 8)),
           "constructor's rotary or routing storage count is wrong");
    m.allocate(2, 3);
    b = m.memory_bytes();
    const size_t N = 6, C = 32, F = 64, L = 2, V = 32;
    const size_t layer = N * (8 * C + 2 + 3 * F) + 2 * 2 * 3 * 3;
    const size_t global = N * (2 * C + 2 + V);
    EXPECT(b.activations.used == (L * layer + global) * 4 + 2 * N * sizeof(int),
           "dense activation payload disagrees with independent shape formula");
    EXPECT(b.backward.used == N * (6 * C + 3 * F) * 4, "dense backward scratch shape is wrong");
    EXPECT(b.scratch.used == N * (F + C) * 4, "projection scratch shape is wrong");
    m.decode_begin(2, {0, 0}); b = m.memory_bytes();
    EXPECT(b.kv.used == 2 * L * 2 * 8 * C * 4 + 2 * sizeof(int), "KV cache count is wrong");
    const auto retained = b.total();
    m.allocate(1, 1);
    EXPECT(m.memory_bytes().total().used == retained.used,
           "accounting hides retained training buffers after smaller decode allocation");
    EXPECT(retained.reserved >= retained.used, "capacity is below payload");

    cfg.ternary = true;
    Model<float> q(cfg); q.init(1);
    EXPECT(q.memory_bytes().quantized.used == q.num_params() * 4, "ternary model lost quantized copy");
    TernaryEngine packed(q);
    const auto pb = packed.memory_bytes();
    // Two stored bits for each projection; the embedding is int8 rows (one
    // byte per weight, one float scale per row) and the norms stay float.
    const size_t float_weights = (C + L * 2 * C) * 4, int8_embedding = V * C + V * 4;
    const size_t bitplanes = L * (4 * C * C + 3 * F * C) / 4;
    EXPECT(pb.weights.used == float_weights + int8_embedding + bitplanes, "packed dense weight payload is wrong");
    EXPECT(packed.weight_bytes() == pb.weights.used + 4 * L * 4, "packed logical scales missing");
    EXPECT(pb.kv.used == 2 * L * 8 * C * 4 && pb.rope.used == size_t(8 * 16 * 4),
           "packed KV/rotary storage is wrong");
    const auto before = pb.total();
    for (int i = 0; i < 8; ++i) packed.step(i);
    EXPECT(packed.memory_bytes().total().reserved == before.reserved,
           "packed decode grows persistent arrays after construction");

    cfg.n_experts = 4; cfg.top_k = 2; cfg.expert_hidden = 32;
    Model<float> moe(cfg); moe.init(2); TernaryEngine pm(moe);
    const auto mb = pm.memory_bytes();
    // With no shared FFN, its two default 32x32 placeholder planes remain
    // allocated but are excluded from logical active weights. Count both.
    EXPECT(mb.weights.used + 4 * L * (2 + 2 * cfg.n_experts) == pm.weight_bytes() + L * 2 * 256,
           "unused MoE placeholder bitplanes were hidden from payload accounting");
}
