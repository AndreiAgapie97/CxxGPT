// A small, fully native C++ research probe. No downloaded data or frameworks.
// Task A/B: predict state transitions on an 8-state ring (+1 / +3), with a
// task marker in the context. Continual phases compare forgetting with 50%
// trajectory replay. The optional bandit learns actions from scalar reward.
#include <algorithm>
#include <array>
#include <filesystem>
#include <memory>
#include <numeric>
#include <vector>

#include "../src/args.h"
#include "../src/infer.h"
#include "../src/train.h"

namespace {
constexpr int kStates = 8, kBatch = 2, kSeq = 16;

Config config(int variant) {
    Config c;
    c.vocab = kStates + 2; c.seq_len = kSeq; c.d_model = 32;
    c.n_layers = 1; c.n_heads = 4; c.ffn_hidden = 64;
    c.ternary = variant % 2;
    if (variant >= 2) { c.n_experts = 4; c.top_k = 2; c.expert_hidden = 32; }
    return c;
}

const char* name(int variant) {
    constexpr const char* names[] = {"dense", "dense-ternary", "moe", "moe-ternary"};
    return names[variant];
}

// Generate one trajectory by stepping the environment. Marker 8 or 9 tells
// the model which rule applies; the random initial state is not a target.
void trajectory(int* x, int* y, int start, int task) {
    const int delta = task == 0 ? 1 : 3;
    x[0] = kStates + task; y[0] = -1;
    int state = start;
    for (int t = 1; t < kSeq; ++t) {
        x[t] = state;
        state = (state + delta) % kStates;
        y[t] = state;
    }
}

struct Score { double loss, accuracy; };
struct StoredTrajectory { std::array<int, kSeq> x, y; };
using ReplayBuffer = std::vector<StoredTrajectory>;
constexpr size_t kReplayCapacity = 64;

// Exhaustive evaluation: every starting state, all predictable positions.
Score evaluate(Model<float>& m, int task) {
    std::vector<int> x(kStates * kSeq), y(x.size());
    for (int s = 0; s < kStates; ++s) trajectory(&x[s * kSeq], &y[s * kSeq], s, task);
    const double loss = m.forward(x.data(), y.data(), kStates, kSeq);
    int correct = 0;
    for (int n = 0; n < int(x.size()); ++n) {
        if (y[n] < 0) continue;
        const float* p = m.output_row(n);
        const int pred = int(std::max_element(p, p + m.cfg.vocab) - p);
        correct += pred == y[n];
    }
    return {loss, double(correct) / (kStates * (kSeq - 1))};
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    return n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2;
}

// Fixed update and sample budgets, and identical data RNG seeds for variants.
double train(Model<float>& m, AdamW& opt, uint64_t seed, int steps, int task, bool replay,
             ReplayBuffer* buffer = nullptr) {
    CHECK(!replay || (buffer && !buffer->empty()), "replay needs stored trajectories");
    Rng data(seed), choice(seed + 10000);
    std::array<int, kBatch * kSeq> x{}, y{};
    std::vector<double> ms;
    for (int step = 0; step < steps; ++step) {
        for (int b = 0; b < kBatch; ++b) {
            const int start = int(data.below(kStates));
            if (replay && choice.below(2) == 0) {
                const auto& old = (*buffer)[choice.below(buffer->size())];
                std::copy(old.x.begin(), old.x.end(), &x[b * kSeq]);
                std::copy(old.y.begin(), old.y.end(), &y[b * kSeq]);
            } else {
                trajectory(&x[b * kSeq], &y[b * kSeq], start, task);
                // Store the first 64 observed A trajectories. Phase B never
                // queries the old environment to regenerate a replay sample.
                if (buffer && !replay && task == 0 && buffer->size() < kReplayCapacity) {
                    StoredTrajectory stored;
                    std::copy_n(&x[b * kSeq], kSeq, stored.x.begin());
                    std::copy_n(&y[b * kSeq], kSeq, stored.y.begin());
                    buffer->push_back(stored);
                }
            }
        }
        const double t0 = now_seconds();
        m.zero_grad();
        const double loss = m.forward(x.data(), y.data(), kBatch, kSeq, true);
        m.backward();
        const double norm = clip_grad_norm(m, 1.0);
        CHECK(std::isfinite(loss) && std::isfinite(norm), "non-finite training state");
        opt.update(m, 0.003);
        m.update_router_bias(0.001);
        if (step >= 5 || steps <= 5) ms.push_back((now_seconds() - t0) * 1000);
    }
    return median(ms);
}

// Measure deployed decode: bitplanes for ternary, float for dense weights.
// Discard a warmup trial, then report a median across five trials. Variant
// execution order is rotated across seeds in main, not randomized per token.
double decode_time(Model<float>& m) {
    std::unique_ptr<TernaryEngine> engine;
    if (m.cfg.ternary) engine = std::make_unique<TernaryEngine>(m);
    std::vector<double> us;
    // Reuse the start vector rather than constructing a temporary at each
    // reset (also avoids a GCC 14 -O1/TSan vector-copy overread warning).
    std::vector<int> starts;
    starts.push_back(0);
    volatile float sink = 0;
    for (int trial = 0; trial < 6; ++trial) {
        const double t0 = now_seconds();
        for (int s = 0; s < kStates; ++s) {
            int x[kSeq], y[kSeq];
            trajectory(x, y, s, 0);
            if (engine) engine->reset();
            else m.decode_begin(1, starts);
            for (int t = 0; t < kSeq; ++t) {
                const float* out = engine ? engine->step(x[t]) : m.decode_step(&x[t], t);
                sink = out[0];
            }
        }
        if (trial) us.push_back((now_seconds() - t0) * 1e6 / (kStates * kSeq));
    }
    CHECK(std::isfinite(sink), "decode produced a non-finite logit");
    return median(us);
}

void report(FILE* out, const char* experiment, int variant, int seed, const char* stage,
            Model<float>& m, const AdamW& opt, double ms, bool decode, size_t replay_bytes = 0) {
    const Score a = evaluate(m, 0), b = evaluate(m, 1);
    // Explicitly counts only these five flat arrays, excluding activations,
    // caches, allocator capacity, routing buffers, and object metadata.
    const size_t training_arrays = (m.params.size() + m.grads.size() + m.qparams.size() +
                                   opt.m.size() + opt.v.size()) * sizeof(float);
    const size_t weights = m.cfg.ternary ? TernaryEngine(m).weight_bytes() : m.num_params() * sizeof(float);
    std::fprintf(out, "%s,%s,%d,%s,%.9g,%.9g,%.9g,%.9g,%zu,%zu,%zu,%zu,%.9g,%.9g,%zu\n",
                 experiment, name(variant), seed, stage, a.loss, a.accuracy, b.loss, b.accuracy,
                 m.num_params(), m.active_params(), training_arrays, weights, ms,
                 decode ? decode_time(m) : 0.0, replay_bytes);
    CHECK(std::fflush(out) == 0, "cannot flush results");
    std::printf("%s %s seed=%d %s: A %.4f/%.1f%% B %.4f/%.1f%%\n", experiment, name(variant),
                seed, stage, a.loss, a.accuracy * 100, b.loss, b.accuracy * 100);
    std::fflush(stdout);
}

int sample_action(const std::vector<float>& p, Rng& rng) {
    const double u = rng.uniform();
    double sum = 0;
    for (size_t i = 0; i < p.size(); ++i) {
        sum += p[i];
        if (u < sum) return int(i);
    }
    return int(p.size() - 1);  // rounding of probabilities
}

Score bandit_score(Model<float>& m) {
    int x[kStates], y[kStates];
    for (int s = 0; s < kStates; ++s) { x[s] = s; y[s] = (s + 1) % kStates; }
    const double loss = m.forward(x, y, kStates, 1);
    int correct = 0;
    for (int s = 0; s < kStates; ++s) {
        const float* p = m.output_row(s);
        correct += int(std::max_element(p, p + m.cfg.vocab) - p) == y[s];
    }
    return {loss, double(correct) / kStates};
}

// Contextual bandit: observe a random state, choose an action, receive reward
// one iff action equals next state. No correct-action labels are supplied to
// backward; its target is the sampled action. Cross-entropy is -log pi,
// so multiplying its gradient by +(reward-baseline) implements descent on
// -(reward-baseline) log pi(action|state), hence ascent on expected reward.
void bandit(FILE* out, int variant, int seed, int steps) {
    Model<float> m(config(variant)); m.init(uint64_t(seed));
    AdamW opt;
    Rng rng(uint64_t(seed) + 30000);
    const Score before = bandit_score(m);
    double baseline = 0, rewards = 0;
    constexpr int interactions = 8;
    std::vector<double> ms;
    for (int step = 0; step < steps; ++step) {
        const double t0 = now_seconds();
        m.zero_grad();
        for (int j = 0; j < interactions; ++j) {
            const int state = int(rng.below(kStates));
            m.forward(&state, nullptr, 1, 1);
            const float* logits = m.output_row(0);
            std::vector<float> p(logits, logits + m.cfg.vocab);
            float loss;
            const int ignored = -1;
            cross_entropy_forward(p.data(), &loss, &ignored, 1, m.cfg.vocab);
            const int action = sample_action(p, rng);
            const double reward = action == (state + 1) % kStates ? 1.0 : 0.0;
            m.forward(&state, &action, 1, 1, true);
            m.backward((reward - baseline) / interactions);
            // Baseline is computed before seeing this action's reward.
            baseline += 0.05 * (reward - baseline);
            rewards += reward;
        }
        const double norm = clip_grad_norm(m, 1.0);
        CHECK(std::isfinite(norm), "non-finite bandit gradient");
        opt.update(m, 0.003);
        m.update_router_bias(0.001);
        if (step >= 5 || steps <= 5) ms.push_back((now_seconds() - t0) * 1000);
    }
    const Score after = bandit_score(m);
    // Bandit columns A/B hold before/after policy scores, not trajectory tasks.
    std::fprintf(out, "bandit,%s,%d,reward,%.9g,%.9g,%.9g,%.9g,%zu,%zu,%zu,%zu,%.9g,0,0\n",
                 name(variant), seed, before.loss, before.accuracy, after.loss, after.accuracy,
                 m.num_params(), m.active_params(),
                 (m.params.size() + m.grads.size() + m.qparams.size() + opt.m.size() + opt.v.size()) * sizeof(float),
                 m.cfg.ternary ? TernaryEngine(m).weight_bytes() : m.num_params() * sizeof(float), median(ms));
    CHECK(std::fflush(out) == 0, "cannot flush results");
    std::printf("bandit %s seed=%d: greedy %.1f%% -> %.1f%%, sampled reward %.3f\n", name(variant),
                seed, before.accuracy * 100, after.accuracy * 100, rewards / (steps * interactions));
    std::fflush(stdout);
}
}  // namespace

int main(int argc, char** argv) {
    Args args(argc, argv, 1);
    const int steps = int(args.integer("steps", 200)), seeds = int(args.integer("seeds", 3));
    const int threads = int(args.integer("threads", 3));
    const bool continual = args.flag("continual");
    const int rl_steps = int(args.integer("rl-steps", 0));
    const std::string path = args.str("out", "runs/toy-world.csv");
    args.done();
    CHECK(steps > 0 && seeds > 0 && threads > 0 && rl_steps >= 0, "invalid experiment budget");
    if (!std::filesystem::path(path).parent_path().empty())
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    FILE* out = std::fopen(path.c_str(), "wx");
    CHECK(out, "cannot create %s (choose a new path to preserve previous runs)", path.c_str());
    std::fprintf(out, "experiment,variant,seed,stage,loss_a,accuracy_a,loss_b,accuracy_b,params,active_params,training_array_bytes,inference_weight_bytes,median_step_ms,median_decode_us,replay_bytes\n");
    ThreadPool tp(threads); g_pool = &tp;
    std::printf("steps=%d seeds=%d threads=%d continual=%d rl_steps=%d\n", steps, seeds, threads, continual, rl_steps);
    for (int seed = 1; seed <= seeds; ++seed)
        for (int index = 0; index < 4; ++index) {
            // Rotate execution order across seeds to reduce order bias.
            const int variant = (index + seed - 1) % 4;
            Model<float> m(config(variant)); m.init(uint64_t(seed));
            AdamW opt;
            ReplayBuffer buffer;
            report(out, "baseline", variant, seed, "initial", m, opt, 0, false);
            const double ms = train(m, opt, uint64_t(seed) + 100, steps, 0, false, continual ? &buffer : nullptr);
            report(out, "baseline", variant, seed, "learn_A", m, opt, ms, true);
            if (continual) {
                // Identical starting weights, moments, biases, and budgets.
                Model<float> replay_model = m;
                AdamW replay_opt = opt;
                const double no_ms = train(m, opt, uint64_t(seed) + 200, steps, 1, false);
                report(out, "continual", variant, seed, "B_no_replay", m, opt, no_ms, false);
                const double re_ms = train(replay_model, replay_opt, uint64_t(seed) + 200, steps, 1, true, &buffer);
                report(out, "continual", variant, seed, "B_half_replay", replay_model, replay_opt, re_ms, false,
                       buffer.size() * sizeof(StoredTrajectory));
            }
            if (rl_steps) bandit(out, variant, seed, rl_steps);
        }
    CHECK(std::fclose(out) == 0, "cannot close results");
}
