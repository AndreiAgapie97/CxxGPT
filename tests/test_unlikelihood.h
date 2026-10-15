// test_unlikelihood.h — unlikelihood training (#34): which tokens become
// negatives (unlikelihood.h), checked by hand and against a brute force
// that knows nothing of NoRepeat; the loss and its gradient (ops.h) against
// finite differences in double precision, with a negative near p = 1; and
// the whole model's gradient with the term on.
#pragma once

#include <cstring>
#include <set>

#include "../src/model.h"
#include "../src/unlikelihood.h"
#include "test_model.h"
#include "test_util.h"

namespace {

std::vector<std::set<int>> negatives_by_position(const RepeatNegatives& r, size_t N) {
    std::vector<std::set<int>> out(N);
    for (size_t i = 0; i < N; ++i)
        for (int k = r.offsets[i]; k < r.offsets[i + 1]; ++k) out[i].insert(r.tokens[size_t(k)]);
    return out;
}

// The definition, written out directly for one row: c is a negative at t if
// some run of n text tokens of the same document, ending at or before t, is
// the last n - 1 text tokens up to t followed by c, and c is not the target.
// A text token is a trained one (its position's previous target is not -1)
// that is not a BOS; the first token of the row is not text.
std::vector<std::set<int>> brute_negatives(const std::vector<int>& x, const std::vector<int>& y, int n) {
    const int Tn = int(x.size());
    std::vector<char> text(x.size());
    std::vector<int> doc(x.size());
    int d = 0;
    for (int k = 0; k < Tn; ++k) {
        d += x[size_t(k)] == kBos;
        doc[size_t(k)] = d;
        text[size_t(k)] = k > 0 && y[size_t(k - 1)] >= 0 && x[size_t(k)] != kBos;
    }
    auto all_text = [&](int a, int b) {  // [a, b]
        if (a < 0) return false;
        for (int k = a; k <= b; ++k)
            if (!text[size_t(k)]) return false;
        return true;
    };
    std::vector<std::set<int>> out(x.size());
    for (int t = 0; t < Tn; ++t) {
        if (y[size_t(t)] < 0 || !all_text(t - n + 2, t)) continue;
        for (int i = 0; i + n - 1 <= t; ++i) {
            if (!all_text(i, i + n - 1) || doc[size_t(i)] != doc[size_t(t)]) continue;
            bool same = true;
            for (int k = 0; k < n - 1; ++k) same = same && x[size_t(i + k)] == x[size_t(t - n + 2 + k)];
            const int c = x[size_t(i + n - 1)];
            if (same && c != y[size_t(t)]) out[size_t(t)].insert(c);
        }
    }
    return out;
}

void test_negatives_by_hand() {
    // a b c a b c a b, then d instead of the c the loop would write. With
    // n = 3 the real text repeats at t = 5..7 (the loop's next token is the
    // target), so nothing is pushed down there; at t = 8 the text leaves the
    // loop, and the token that would have continued it (c) is the negative.
    const int a = 5, b = 6, c = 7, dd = 8;
    std::vector<int> x = {9, a, b, c, a, b, c, a, b};
    std::vector<int> y = {a, b, c, a, b, c, a, b, dd};
    RepeatNegatives r;
    find_repeat_negatives(r, x.data(), y.data(), 1, int(x.size()), 3);
    auto got = negatives_by_position(r, x.size());
    for (size_t t = 0; t < x.size(); ++t)
        EXPECT((t == 8) == !got[t].empty(), "a b c loop: position %zu has %zu negatives", t, got[t].size());
    EXPECT(got[8] == std::set<int>{c}, "a b c loop: the negative at the exit is not c");
    EXPECT(r.fired == 1 && r.tokens.size() == 1, "a b c loop: %zu positions fired, %zu negatives", r.fired,
           r.tokens.size());

    // Question a b c, answer a b d. Trained, "a b" was followed by c, so c
    // is a negative where the answer writes d; with the question masked, the
    // answer may copy it, and nothing fires.
    const std::vector<int> xq = {9, a, b, c, a, b}, yq = {a, b, c, a, b, dd};
    find_repeat_negatives(r, xq.data(), yq.data(), 1, int(xq.size()), 3);
    EXPECT(negatives_by_position(r, xq.size())[5] == std::set<int>{c}, "unmasked question: c is not the negative");
    std::vector<int> ym = yq;
    ym[0] = ym[1] = ym[2] = -1;  // tokens 1..3 (the question) untrained
    find_repeat_negatives(r, xq.data(), ym.data(), 1, int(xq.size()), 3);
    EXPECT(r.tokens.empty(), "a copied question gave %zu negatives", r.tokens.size());

    // n = 2: "a" was followed by "b" in the first document. After the BOS,
    // "a" followed by c is new: without the reset, b would be a negative at t = 4.
    const std::vector<int> xb = {9, a, b, kBos, a, c}, yb = {a, b, kBos, a, c, c};
    find_repeat_negatives(r, xb.data(), yb.data(), 1, int(xb.size()), 2);
    EXPECT(r.tokens.empty(), "BOS: %zu negatives from the previous document", r.tokens.size());

    // An untrained target gets no negatives, even inside a loop.
    std::vector<int> yu = y;
    yu[8] = -1;
    find_repeat_negatives(r, x.data(), yu.data(), 1, int(x.size()), 3);
    EXPECT(r.tokens.empty() && r.offsets.size() == x.size() + 1, "an untrained target got negatives");

    // Two rows are independent: the loop in row 0 does not reach row 1.
    std::vector<int> x2 = x, y2 = y;
    x2.insert(x2.end(), {9, a, b, a, b, a, b, a, b});
    y2.insert(y2.end(), {a, b, a, b, a, b, a, b, c});
    find_repeat_negatives(r, x2.data(), y2.data(), 2, 9, 3);
    got = negatives_by_position(r, x2.size());
    // Row 1 (n = 3) loops on a b a b, and its next token is always the
    // target until t = 8, where c leaves the loop and "a b" -> a is the negative.
    EXPECT(got[8] == std::set<int>{c}, "row 0 changed when a second row was added");
    EXPECT(got[9 + 8] == std::set<int>{a}, "row 1: the negative at the exit is not a");
    for (int t = 0; t < 8; ++t) EXPECT(got[size_t(9 + t)].empty(), "row 1: position %d fired", t);
}

void test_negatives_brute_force() {
    Rng rng(34);
    int checked = 0, fired = 0;
    for (int trial = 0; trial < 300; ++trial) {
        const int n = 2 + int(rng.below(4)), Tn = 4 + int(rng.below(40)), B = 1 + int(rng.below(3));
        std::vector<int> x(size_t(B) * Tn), y(x.size());
        for (int b = 0; b < B; ++b) {
            // Tokens 2..4 repeat often; a BOS now and then; masked stretches.
            std::vector<int> stream(size_t(Tn) + 1);
            for (auto& s : stream) s = rng.below(25) == 0 ? kBos : 2 + int(rng.below(3));
            bool masked = false;
            for (int t = 0; t < Tn; ++t) {
                if (rng.below(8) == 0) masked = !masked;
                x[size_t(b) * Tn + t] = stream[size_t(t)];
                y[size_t(b) * Tn + t] = masked ? -1 : stream[size_t(t) + 1];
            }
        }
        RepeatNegatives r;
        find_repeat_negatives(r, x.data(), y.data(), B, Tn, n);
        const auto got = negatives_by_position(r, x.size());
        size_t want_fired = 0;
        for (int b = 0; b < B; ++b) {
            const std::vector<int> xr(x.begin() + b * Tn, x.begin() + (b + 1) * Tn);
            const std::vector<int> yr(y.begin() + b * Tn, y.begin() + (b + 1) * Tn);
            const auto want = brute_negatives(xr, yr, n);
            for (int t = 0; t < Tn; ++t) {
                const bool ok = got[size_t(b) * Tn + t] == want[size_t(t)];
                EXPECT(ok, "trial %d (n %d): row %d position %d differs from the brute force", trial, n, b, t);
                if (!ok) return;
                want_fired += !want[size_t(t)].empty();
                ++checked;
            }
        }
        EXPECT(r.fired == want_fired, "trial %d: %zu positions fired, the brute force says %zu", trial, r.fired,
               want_fired);
        fired += int(want_fired);
        // Each row's negatives are distinct (ops.h checks it too).
        EXPECT(r.tokens.size() == [&] { size_t s = 0; for (auto& g : got) s += g.size(); return s; }(),
               "trial %d: a negative is listed twice", trial);
    }
    EXPECT(fired > 100, "the random rows hardly ever repeat (%d positions fired)", fired);
    std::printf("  negatives: %d positions match the brute force, %d with negatives\n", checked, fired);
}

// -log(1 - p_c) from the logits, in log space: log sum_j e^z_j - log sum_{j != c} e^z_j.
double reference_ul(const double* z, int V, int c) {
    double mx = -1e300;
    for (int i = 0; i < V; ++i) mx = std::max(mx, z[i]);
    double all = 0, rest = 0;
    for (int i = 0; i < V; ++i) {
        all += std::exp(z[i] - mx);
        if (i != c) rest += std::exp(z[i] - mx);
    }
    return std::log(all) - std::log(rest);
}

void test_unlikelihood_gradient() {
    Rng rng(35);
    const int N = 5, V = 7;
    const int targets[N] = {3, 0, -1, 6, 2};
    // Row 0: two negatives. Row 1: counted, none. Row 2: ignored. Row 3:
    // the arg-max is a negative with p close to 1 (1 - p about 2e-9).
    // Row 4: three negatives, one of them the arg-max.
    const std::vector<int> offsets = {0, 2, 2, 2, 3, 6};
    const std::vector<int> negs = {1, 4, 5, 0, 1, 4};
    auto logits = random_vec<double>(N * V, rng, 2.0);
    logits[size_t(3) * V + 5] = 20;
    logits[size_t(4) * V + 1] = 4;
    const double alpha = 0.7;
    const NegativeRows neg{offsets.data(), negs.data(), alpha};
    std::vector<double> probs(N * V), losses(N), d(N * V);
    auto loss = [&] {
        probs = logits;
        const double ce = cross_entropy_forward(probs.data(), losses.data(), targets, N, V);
        return ce + alpha * unlikelihood_forward(probs.data(), targets, neg, N, V) / 4;  // 4 counted rows
    };
    const double l = loss();
    cross_entropy_backward(d.data(), probs.data(), targets, N, V, 1.0 / 4, neg);
    const double e = grad_error(logits, d, loss, rng);
    EXPECT(e < 1e-7, "unlikelihood: gradient err %g", e);

    probs = logits;
    const double ce = cross_entropy_forward(probs.data(), losses.data(), targets, N, V);
    double ul = 0;
    for (int n = 0; n < N; ++n)
        for (int k = offsets[size_t(n)]; k < offsets[size_t(n) + 1]; ++k)
            ul += reference_ul(&logits[size_t(n) * V], V, negs[size_t(k)]);
    EXPECT(std::abs(l - (ce + alpha * ul / 4)) < 1e-10, "unlikelihood value %.12g vs %.12g", l, ce + alpha * ul / 4);
    for (int n = 0; n < N; ++n) {
        double sum = 0;
        for (int i = 0; i < V; ++i) sum += d[size_t(n) * V + i];
        EXPECT(std::abs(sum) < 1e-12, "row %d's gradient sums to %g, not 0", n, sum);
    }
    for (int i = 0; i < V; ++i) EXPECT(d[size_t(2) * V + i] == 0.0, "the ignored row has a gradient");

    // Without negatives (or with alpha 0), exactly the cross-entropy's
    // gradient, bit for bit.
    std::vector<double> plain(N * V);
    probs = logits;
    cross_entropy_forward(probs.data(), losses.data(), targets, N, V);
    cross_entropy_backward(plain.data(), probs.data(), targets, N, V, 1.0 / 4);
    const std::vector<int> no_offsets(N + 1, 0);
    for (const NegativeRows off : {NegativeRows{}, NegativeRows{no_offsets.data(), negs.data(), alpha},
                                   NegativeRows{offsets.data(), negs.data(), 0.0}}) {
        std::vector<double> g(N * V);
        cross_entropy_backward(g.data(), probs.data(), targets, N, V, 1.0 / 4, off);
        EXPECT(std::memcmp(g.data(), plain.data(), g.size() * sizeof(double)) == 0,
               "the term changed the gradient while off");
    }
}

// float: a negative with 1 - p near 1e-13, which 1 - p in float rounds to
// 0, and one where every other probability underflows.
void test_unlikelihood_float() {
    const int V = 6;
    const int target[1] = {2};
    const std::vector<int> offsets = {0, 1};
    const std::vector<int> negs = {4};
    const NegativeRows neg{offsets.data(), negs.data(), 1.0};
    const double gaps[] = {30.0, 200.0};
    for (const double gap : gaps) {
        std::vector<double> zd = {0.5, -1.0, 0.0, 1.0, gap, -2.0};
        std::vector<float> zf(zd.begin(), zd.end());
        std::vector<double> pd = zd, ld(1), gd(V);
        std::vector<float> pf = zf, lf(1), gf(V);
        cross_entropy_forward(pd.data(), ld.data(), target, 1, V);
        cross_entropy_forward(pf.data(), lf.data(), target, 1, V);
        const double ud = unlikelihood_forward(pd.data(), target, neg, 1, V);
        const double uf = unlikelihood_forward(pf.data(), target, neg, 1, V);
        cross_entropy_backward(gd.data(), pd.data(), target, 1, V, 1.0, neg);
        cross_entropy_backward(gf.data(), pf.data(), target, 1, V, 1.0f, neg);
        bool finite = std::isfinite(uf);
        double worst = 0, sum = 0;
        for (int i = 0; i < V; ++i) {
            finite = finite && std::isfinite(gf[size_t(i)]);
            worst = std::max(worst, std::abs(double(gf[size_t(i)]) - gd[size_t(i)]));
            sum += gf[size_t(i)];
        }
        EXPECT(finite, "gap %g: a float loss or gradient is not finite", gap);
        EXPECT(std::abs(sum) < 1e-5, "gap %g: the float row's gradient sums to %g", gap, sum);
        if (gap < 100) {
            EXPECT(worst < 1e-5, "gap %g: float gradient off by %g", gap, worst);
            // The term's gradient on the negative is alpha * p_c, about 1,
            // on top of the cross-entropy's p_c.
            EXPECT(std::abs(gf[4] - 2.0f) < 1e-5f, "gap %g: the negative's gradient is %g, not about 2", gap, gf[4]);
            EXPECT(std::abs(uf - ud) < 1e-4 * ud, "gap %g: float term %.6g vs double %.6g", gap, uf, ud);
            EXPECT(std::abs(ud - reference_ul(zd.data(), V, 4)) < 1e-9 * ud, "gap %g: double term off", gap);
        } else {
            // 1 - p in float is below the floor: the floor's loss, and a
            // gradient no larger than the cross-entropy's plus alpha.
            EXPECT(std::abs(uf + std::log(kUnlikelihoodFloor)) < 1e-6, "gap %g: the term is %g, not the floor's",
                   gap, uf);
            for (int i = 0; i < V; ++i)
                EXPECT(std::abs(gf[size_t(i)]) <= 2.0f + 1e-6f, "gap %g: gradient %g at %d", gap, gf[size_t(i)], i);
        }
    }
    // Every other probability exactly 0 (the forward's exponential never
    // quite gets there, so the row is written by hand): the floor's loss,
    // and only the cross-entropy's gradient, which still sums to 0.
    std::vector<float> p = {0, 0, 0, 0, 1, 0}, g(V);
    const float u = float(unlikelihood_forward(p.data(), target, neg, 1, V));
    cross_entropy_backward(g.data(), p.data(), target, 1, V, 1.0f, neg);
    EXPECT(std::abs(u + std::log(kUnlikelihoodFloor)) < 1e-4, "p = 1 exactly: the term is %g, not the floor's", u);
    const std::vector<float> want = {0, 0, -1, 0, 1, 0};
    EXPECT(g == want, "p = 1 exactly: gradient %g on the negative, %g on the target", g[4], g[2]);
}

void test_unlikelihood_model() {
    Config cfg;
    cfg.vocab = 11;
    cfg.seq_len = 16;
    cfg.d_model = 8;
    cfg.n_layers = 2;
    cfg.n_heads = 2;
    cfg.ffn_hidden = 12;
    for (const bool moe : {false, true}) {
        Config c = cfg;
        if (moe) {
            c.n_experts = 4;
            c.top_k = 2;
            c.expert_hidden = 6;
        }
        Rng rng(36);
        Model<double> m(c);
        randomize_for_gradcheck(m, rng);
        const int B = 2, Tn = 12;
        // Row 0 loops on 2 3 4 and leaves the loop twice; row 1 is random,
        // with a masked stretch.
        std::vector<int> stream0 = {7, 2, 3, 4, 2, 3, 4, 5, 2, 3, 4, 2, 6};
        std::vector<int> tokens(B * Tn), targets(B * Tn);
        for (int t = 0; t < Tn; ++t) {
            tokens[size_t(t)] = stream0[size_t(t)];
            targets[size_t(t)] = stream0[size_t(t) + 1];
            tokens[size_t(Tn + t)] = int(rng.below(c.vocab));
            targets[size_t(Tn + t)] = t >= 3 && t < 6 ? -1 : int(rng.below(c.vocab));
        }
        for (int t = 1; t < Tn; ++t)
            if (targets[size_t(Tn + t - 1)] >= 0) tokens[size_t(Tn + t)] = targets[size_t(Tn + t - 1)];
        const char* what = moe ? "moe" : "dense";

        m.zero_grad();
        const double ce_only = m.forward(tokens.data(), targets.data(), B, Tn, true);
        m.backward();
        const std::vector<double> plain = m.grads;
        m.unlikelihood = {0.5, 3};
        m.zero_grad();
        const double l = m.forward(tokens.data(), targets.data(), B, Tn, true);
        EXPECT(m.last_fired >= 2 && m.last_ul > 0, "%s: the loop fired at %zu positions", what, m.last_fired);
        EXPECT(std::abs(m.last_ce - ce_only) < 1e-14 && std::abs(l - (ce_only + 0.5 * m.last_ul)) < 1e-14,
               "%s: forward returned %.12g, not ce + alpha * ul", what, l);
        m.backward();
        const std::vector<double> analytic = m.grads;
        auto loss = [&] { return m.forward(tokens.data(), targets.data(), B, Tn, true); };
        double worst = 0, moved = 0;
        for (const auto& p : m.infos) {
            std::vector<double> x(m.params.begin() + p.offset, m.params.begin() + p.offset + p.size);
            std::vector<double> g(analytic.begin() + p.offset, analytic.begin() + p.offset + p.size);
            auto loss_p = [&] {
                std::copy(x.begin(), x.end(), m.params.begin() + p.offset);
                return loss();
            };
            const double e = grad_error(x, g, loss_p, rng, 60);
            std::copy(x.begin(), x.end(), m.params.begin() + p.offset);
            worst = std::max(worst, e);
            EXPECT(e < 1e-6, "%s with unlikelihood: gradient of %s off by %g", what, p.name.c_str(), e);
        }
        for (size_t i = 0; i < plain.size(); ++i) moved = std::max(moved, std::abs(analytic[i] - plain[i]));
        EXPECT(moved > 1e-4, "%s: the term did not change the gradient (%g)", what, moved);

        // Validation forwards stay cross-entropy only.
        const double val = m.forward(tokens.data(), targets.data(), B, Tn, false);
        EXPECT(val == ce_only && m.last_ul == 0 && m.last_fired == 0, "%s: a validation forward has the term",
               what);
        // Runs longer than any repeat: on, but nothing fires, so the
        // gradient is the cross-entropy's, bit for bit.
        m.unlikelihood = {0.5, 8};
        m.zero_grad();
        m.forward(tokens.data(), targets.data(), B, Tn, true);
        m.backward();
        EXPECT(m.last_fired == 0 && m.grads == plain, "%s: a term that never fires changed the gradient", what);
        std::printf("  %s with unlikelihood: worst gradient error %.1e\n", what, worst);
    }
}

}  // namespace

inline void test_unlikelihood() {
    test_negatives_by_hand();
    test_negatives_brute_force();
    test_unlikelihood_gradient();
    test_unlikelihood_float();
    test_unlikelihood_model();
}
