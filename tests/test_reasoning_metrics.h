#pragma once
#include "../src/reasoning_metrics.h"
#include "../src/ops.h"
#include "test_util.h"

inline void test_reasoning_metrics() {
    using namespace reasoning_metrics;
    Problem p;
    p.question = "1+1"; p.steps = "1+1=2"; p.answer = "2";
    const auto seq = sequence(p);
    const size_t first = 1 + p.prompt().size() - 1;
    EXPECT(seq.target[first] == '1' && seq.input[0] == kBos && seq.target[first-1] == -1,
           "teacher-forcing shifted the answer or included the prompt");
    constexpr int V = 256;
    std::vector<double> logits(seq.input.size() * V, 0);
    for (size_t i = 0; i < seq.target.size(); ++i)
        if (seq.target[i] >= 0) logits[i * V + seq.target[i]] = seq.role[i] == answer_digit ? -1000 : 3;
    const auto got = score(logits.data(), V, seq);
    const double ordinary = std::log(1 + 255 * std::exp(-3.0));
    const double losing = 1000 + std::log(255.0);
    EXPECT(got.by_role[step_digit].tokens == 3 && got.by_role[step_other].tokens == 2 &&
           got.by_role[delimiter].tokens == 6 && got.by_role[answer_digit].tokens == 1 &&
           got.by_role[answer_other].tokens == 1 && got.all.tokens == 13,
           "role boundaries/counts incorrect");
    EXPECT(got.all.correct == 12 && std::abs(got.all.mean() - (12 * ordinary + losing) / 13) < 1e-12,
           "segmented CE disagrees with independent analytic reference");
    std::vector<double> losses(seq.target.size());
    const double engine = cross_entropy_forward(logits.data(), losses.data(), seq.target.data(), int(seq.target.size()), V);
    EXPECT(std::abs(engine - got.all.mean()) < 1e-12, "segmented stable loss differs from engine loss");
    Score total; merge(total, got); merge(total, got);
    EXPECT(total.all.tokens == 26 && total.all.correct == 24 && total.all.mean() == got.all.mean(),
           "token-weighted diagnostic aggregation failed");
}
