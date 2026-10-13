// train.h — optimizer, learning-rate schedule, token data and checkpoints.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "common.h"
#include "model.h"

// ---------------------------------------------------------------------------
// AdamW with decoupled weight decay on the tensors flagged `decay`.
// ---------------------------------------------------------------------------
struct AdamW {
    double beta1 = 0.9, beta2 = 0.95, eps = 1e-8, weight_decay = 0.1;
    long step = 0;
    std::vector<float> m, v;

    void update(Model<float>& model, double lr) {
        if (m.size() != model.params.size()) {
            m.assign(model.params.size(), 0.0f);
            v.assign(model.params.size(), 0.0f);
        }
        PROFILE("adamw");
        ++step;
        const float b1 = float(beta1), b2 = float(beta2);
        const float c1 = float(1.0 / (1.0 - std::pow(beta1, double(step))));
        const float c2 = float(1.0 / (1.0 - std::pow(beta2, double(step))));
        const float lr_f = float(lr), eps_f = float(eps);
        for (const auto& info : model.infos) {
            const float wd = info.decay ? float(weight_decay) : 0.0f;
            float* p = &model.params[info.offset];
            const float* g = &model.grads[info.offset];
            float* mm = &m[info.offset];
            float* vv = &v[info.offset];
            const int blocks = int((info.size + 4095) / 4096);
            pool().parallel_for(blocks, [&](int b0, int b1_) {
                const size_t i0 = size_t(b0) * 4096, i1 = std::min(info.size, size_t(b1_) * 4096);
                for (size_t i = i0; i < i1; ++i) {
                    mm[i] = b1 * mm[i] + (1 - b1) * g[i];
                    vv[i] = b2 * vv[i] + (1 - b2) * g[i] * g[i];
                    const float upd = (mm[i] * c1) / (std::sqrt(vv[i] * c2) + eps_f);
                    p[i] -= lr_f * (upd + wd * p[i]);
                }
            }, 1);
        }
        model.weights_dirty = true;
    }
};

// Global L2 norm of the gradient; scales it down to max_norm if larger.
// Returns the norm before clipping.
inline double clip_grad_norm(Model<float>& model, double max_norm) {
    PROFILE("clip_grad");
    // Fixed blocks with double partial sums: fast and reproducible.
    const size_t n = model.grads.size(), kBlock = 16384;
    const int blocks = int((n + kBlock - 1) / kBlock);
    std::vector<double> partial(blocks, 0.0);
    const float* g = model.grads.data();
    pool().parallel_for(blocks, [&](int b0, int b1) {
        for (int b = b0; b < b1; ++b) {
            const size_t i0 = size_t(b) * kBlock, i1 = std::min(n, i0 + kBlock);
            double acc[8] = {};
            size_t i = i0;
            for (; i + 8 <= i1; i += 8)
                for (int j = 0; j < 8; ++j) acc[j] += double(g[i + j]) * g[i + j];
            double s = 0.0;
            for (; i < i1; ++i) s += double(g[i]) * g[i];
            for (double a : acc) s += a;
            partial[b] = s;
        }
    }, 1);
    double s = 0.0;
    for (double p : partial) s += p;
    const double norm = std::sqrt(s);
    if (max_norm > 0 && norm > max_norm) {
        const float k = float(max_norm / norm);
        for (float& x : model.grads) x *= k;
    }
    return norm;
}

// Linear warmup, then cosine decay to min_frac * max_lr at total_steps.
inline double lr_at(long step, long warmup, long total_steps, double max_lr, double min_frac) {
    if (step < warmup) return max_lr * double(step + 1) / double(warmup);
    if (step >= total_steps) return max_lr * min_frac;
    const double t = double(step - warmup) / double(std::max(1L, total_steps - warmup));
    return max_lr * (min_frac + (1 - min_frac) * 0.5 * (1 + std::cos(3.141592653589793 * t)));
}

// ---------------------------------------------------------------------------
// Token data: a stream of uint16 token ids, plus an optional per-token mask
// (1 = the model is trained to predict this token, 0 = context only).
//
// File layout: "CXTOKEN1", uint32 vocab, uint32 has_mask, uint64 n,
// then n uint16 tokens, then (if has_mask) n uint8 mask bytes.
// ---------------------------------------------------------------------------
struct TokenData {
    uint32_t vocab = 0;
    std::vector<uint16_t> tokens;
    std::vector<uint8_t> mask;  // empty = train on every token

    void save(const std::string& path) const {
        FILE* f = std::fopen(path.c_str(), "wb");
        CHECK(f, "cannot write %s", path.c_str());
        const uint32_t has_mask = mask.empty() ? 0 : 1;
        const uint64_t n = tokens.size();
        std::fwrite("CXTOKEN1", 1, 8, f);
        std::fwrite(&vocab, 4, 1, f);
        std::fwrite(&has_mask, 4, 1, f);
        std::fwrite(&n, 8, 1, f);
        std::fwrite(tokens.data(), 2, n, f);
        if (has_mask) std::fwrite(mask.data(), 1, n, f);
        CHECK(std::fclose(f) == 0, "error writing %s", path.c_str());
    }

    [[nodiscard]] static TokenData load(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        CHECK(f, "cannot open %s", path.c_str());
        char magic[8];
        TokenData d;
        uint32_t has_mask = 0;
        uint64_t n = 0;
        CHECK(std::fread(magic, 1, 8, f) == 8 && !std::memcmp(magic, "CXTOKEN1", 8),
              "%s is not a token file", path.c_str());
        CHECK(std::fread(&d.vocab, 4, 1, f) == 1 && std::fread(&has_mask, 4, 1, f) == 1 &&
              std::fread(&n, 8, 1, f) == 1, "truncated header in %s", path.c_str());
        d.tokens.resize(n);
        CHECK(std::fread(d.tokens.data(), 2, n, f) == n, "truncated tokens in %s", path.c_str());
        if (has_mask) {
            d.mask.resize(n);
            CHECK(std::fread(d.mask.data(), 1, n, f) == n, "truncated mask in %s", path.c_str());
        }
        std::fclose(f);
        for (uint16_t t : d.tokens) CHECK(t < d.vocab, "token %u >= vocab %u in %s", t, d.vocab, path.c_str());
        return d;
    }

    // B windows of Tn + 1 tokens starting at random offsets in [lo, hi - Tn - 1].
    // Targets that the mask excludes become -1.
    void sample_batch(Rng& rng, int B, int Tn, size_t lo, size_t hi, int* inputs, int* targets) const {
        CHECK(B > 0 && Tn > 0 && lo <= hi && hi <= tokens.size(), "invalid data sampling range or shape");
        CHECK(hi - lo >= size_t(Tn) + 1, "data range of %zu tokens is shorter than a sequence", hi - lo);
        for (int b = 0; b < B; ++b) {
            const size_t s = lo + rng.below(hi - lo - Tn);
            for (int t = 0; t < Tn; ++t) {
                inputs[b * Tn + t] = tokens[s + t];
                const bool train = mask.empty() || mask[s + t + 1];
                targets[b * Tn + t] = train ? tokens[s + t + 1] : -1;
            }
        }
    }
};

// Count loss-bearing positions; the model treats every negative target as
// ignored. Use the same denominator for training and validation metrics.
inline size_t count_targets(const int* targets, size_t n) {
    size_t count = 0;
    for (size_t i = 0; i < n; ++i) count += targets[i] >= 0;
    return count;
}

struct TokenLossMean {
    double sum = 0;
    size_t count = 0;
    void add(double mean_loss, size_t valid) {
        if (!valid) return;
        sum += mean_loss * double(valid);
        count += valid;
    }
    double value() const { return count ? sum / double(count) : 0.0; }
};

// Stage only host token IDs, not activations, so all denominators are known
// before backward. This avoids accumulating large unnormalized gradients.
class AccumulationBatch {
public:
    const int batch, seq, micros;
    const size_t per_micro;
    std::vector<int> inputs, targets;
    std::vector<size_t> valid;
    size_t total_valid = 0;

    AccumulationBatch(int B, int Tn, int accum)
        : batch(B), seq(Tn), micros(accum), per_micro(checked_rows(B, Tn, accum)) {
        CHECK(size_t(accum) <= inputs.max_size() / per_micro, "accumulation buffer is too large");
        inputs.resize(per_micro * size_t(accum));
        targets.resize(inputs.size());
        valid.resize(accum);
    }

    int* input(int micro) { return inputs.data() + size_t(micro) * per_micro; }
    int* target(int micro) { return targets.data() + size_t(micro) * per_micro; }
    const int* input(int micro) const { return inputs.data() + size_t(micro) * per_micro; }
    const int* target(int micro) const { return targets.data() + size_t(micro) * per_micro; }

    void refresh_counts() {
        total_valid = 0;
        for (int micro = 0; micro < micros; ++micro) {
            valid[micro] = count_targets(target(micro), per_micro);
            total_valid += valid[micro];
        }
    }
    void sample(const TokenData& data, Rng& rng, size_t lo, size_t hi) {
        for (int micro = 0; micro < micros; ++micro)
            data.sample_batch(rng, batch, seq, lo, hi, input(micro), target(micro));
        refresh_counts();
    }

private:
    static size_t checked_rows(int B, int Tn, int accum) {
        CHECK(B > 0 && Tn > 0 && accum > 0, "batch, sequence and accumulation must be positive");
        CHECK(B <= std::numeric_limits<int>::max() / Tn, "batch times sequence exceeds engine index range");
        return size_t(B) * Tn;
    }
};

// Forward returns a microbatch mean. Multiplying its backward by n_i / N
// makes each valid token contribute 1/N, including unequal masks. Engine is
// Model<T>, GpuTrainer, or the CLI Backend; all share this interface.
template <class Engine>
double accumulated_backward(Engine& engine, const AccumulationBatch& group) {
    engine.zero_grad();
    if (!group.total_valid) return 0.0;
    TokenLossMean loss;
    for (int micro = 0; micro < group.micros; ++micro) {
        const double l = engine.forward(group.input(micro), group.target(micro), group.batch, group.seq, true);
        loss.add(l, group.valid[micro]);
        engine.backward(double(group.valid[micro]) / double(group.total_valid));
    }
    return loss.value();
}

// Writes a token file (TokenData's format, without a mask) piece by piece,
// for data too big to hold in memory. The count goes into the header last.
class TokenWriter {
public:
    TokenWriter(const std::string& path, uint32_t vocab) : path_(path) {
        f_ = std::fopen(path.c_str(), "wb");
        CHECK(f_, "cannot write %s", path.c_str());
        const uint32_t has_mask = 0;
        const uint64_t n = 0;
        std::fwrite("CXTOKEN1", 1, 8, f_);
        std::fwrite(&vocab, 4, 1, f_);
        std::fwrite(&has_mask, 4, 1, f_);
        std::fwrite(&n, 8, 1, f_);
    }
    ~TokenWriter() { close(); }
    TokenWriter(const TokenWriter&) = delete;
    TokenWriter& operator=(const TokenWriter&) = delete;

    void write(const std::vector<uint16_t>& t) {
        if (t.empty()) return;  // fwrite must not see the null data() of an empty vector
        CHECK(std::fwrite(t.data(), 2, t.size(), f_) == t.size(), "error writing %s", path_.c_str());
        n_ += t.size();
    }
    uint64_t count() const { return n_; }

    void close() {
        if (!f_) return;
        CHECK(std::fseek(f_, 16, SEEK_SET) == 0 && std::fwrite(&n_, 8, 1, f_) == 1, "error writing %s",
              path_.c_str());
        CHECK(std::fclose(f_) == 0, "error writing %s", path_.c_str());
        f_ = nullptr;
    }

private:
    std::string path_;
    FILE* f_ = nullptr;
    uint64_t n_ = 0;
};

// ---------------------------------------------------------------------------
// Checkpoints: config, step, data RNG, parameters, optimizer moments and
// router biases, so training resumes exactly where it stopped.
// ---------------------------------------------------------------------------
constexpr char kCkptMagic[9] = "CXCKPT01";

inline void write_config(FILE* f, const Config& c) {
    const int32_t v[] = {c.vocab, c.seq_len, c.d_model, c.n_layers, c.n_heads, c.ffn_hidden,
                         c.n_experts, c.top_k, c.expert_hidden, c.n_shared, c.ternary ? 1 : 0};
    const int32_t n = int32_t(std::size(v));
    std::fwrite(&n, 4, 1, f);
    std::fwrite(v, 4, n, f);
}

inline Config read_config(FILE* f) {
    int32_t n = 0, v[64] = {};
    CHECK(std::fread(&n, 4, 1, f) == 1 && n > 0 && n <= 64, "bad config in checkpoint");
    CHECK(std::fread(v, 4, n, f) == size_t(n), "truncated config");
    CHECK(n == 11, "checkpoint config has %d fields, expected 11", n);
    Config c;
    c.vocab = v[0]; c.seq_len = v[1]; c.d_model = v[2]; c.n_layers = v[3]; c.n_heads = v[4];
    c.ffn_hidden = v[5]; c.n_experts = v[6]; c.top_k = v[7]; c.expert_hidden = v[8];
    c.n_shared = v[9]; c.ternary = v[10] != 0;
    return c;
}

inline void save_checkpoint(const std::string& path, const Model<float>& model, const AdamW& opt,
                            const Rng& data_rng) {
    const std::string tmp = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    CHECK(f, "cannot write %s", tmp.c_str());
    std::fwrite(kCkptMagic, 1, 8, f);
    write_config(f, model.cfg);
    const int64_t step = opt.step;
    const uint64_t n = model.params.size();
    std::fwrite(&step, 8, 1, f);
    std::fwrite(data_rng.state(), 8, 4, f);
    std::fwrite(&n, 8, 1, f);
    std::fwrite(model.params.data(), 4, n, f);
    const uint32_t has_opt = opt.m.size() == n;
    std::fwrite(&has_opt, 4, 1, f);
    if (has_opt) {
        std::fwrite(opt.m.data(), 4, n, f);
        std::fwrite(opt.v.data(), 4, n, f);
    }
    for (const auto& b : model.router_bias) std::fwrite(b.data(), 4, b.size(), f);
    CHECK(std::fclose(f) == 0, "error writing %s", tmp.c_str());
    // Replace atomically, so an interrupted save never destroys the last good one.
    CHECK(std::rename(tmp.c_str(), path.c_str()) == 0, "cannot rename %s", tmp.c_str());
}

// Reads a checkpoint's config only (to build the model before loading).
inline Config peek_checkpoint_config(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    CHECK(f, "cannot open %s", path.c_str());
    char magic[8];
    CHECK(std::fread(magic, 1, 8, f) == 8 && !std::memcmp(magic, kCkptMagic, 8),
          "%s is not a checkpoint", path.c_str());
    const Config c = read_config(f);
    std::fclose(f);
    return c;
}

inline void load_checkpoint(const std::string& path, Model<float>& model, AdamW* opt, Rng* data_rng) {
    FILE* f = std::fopen(path.c_str(), "rb");
    CHECK(f, "cannot open %s", path.c_str());
    char magic[8];
    CHECK(std::fread(magic, 1, 8, f) == 8 && !std::memcmp(magic, kCkptMagic, 8),
          "%s is not a checkpoint", path.c_str());
    read_config(f);
    int64_t step = 0;
    uint64_t rng_state[4], n = 0;
    uint32_t has_opt = 0;
    CHECK(std::fread(&step, 8, 1, f) == 1 && std::fread(rng_state, 8, 4, f) == 4 &&
          std::fread(&n, 8, 1, f) == 1, "truncated checkpoint");
    CHECK(n == model.params.size(), "checkpoint has %llu params, model %zu",
          (unsigned long long)n, model.params.size());
    CHECK(std::fread(model.params.data(), 4, n, f) == n, "truncated params");
    CHECK(std::fread(&has_opt, 4, 1, f) == 1, "truncated checkpoint");
    if (has_opt) {
        std::vector<float> m(n), v(n);
        CHECK(std::fread(m.data(), 4, n, f) == n && std::fread(v.data(), 4, n, f) == n,
              "truncated optimizer state");
        if (opt) {
            opt->m = std::move(m);
            opt->v = std::move(v);
        }
    }
    for (auto& b : model.router_bias)
        CHECK(std::fread(b.data(), 4, b.size(), f) == b.size(), "truncated router bias");
    std::fclose(f);
    if (opt) opt->step = step;
    if (data_rng) data_rng->set_state(rng_state);
    model.weights_dirty = true;
}
