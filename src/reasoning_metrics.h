// Byte-level teacher-forced diagnostics. Roles describe characters, not
// semantic correctness: correct digit tokens can still form a wrong answer.
#pragma once
#include <array>
#include <limits>
#include "model.h"
#include "reasoning.h"

namespace reasoning_metrics {
enum Role { context, step_digit, step_other, delimiter, answer_digit, answer_other, roles };
inline const char* name(Role role) {
    static const char* names[] = {"context", "step_digit", "step_other", "delimiter", "answer_digit", "answer_other"};
    return names[role];
}
struct ByteSequence {
    std::vector<int> input, target;
    std::vector<Role> role;
};
inline ByteSequence sequence(const Problem& p) {
    std::string text(1, char(kBos));
    text += p.prompt();
    const size_t start = text.size(), mark = start + p.steps.size(), answer = mark + 6;
    text += p.completion();
    ByteSequence out;
    for (size_t pos = 0; pos + 1 < text.size(); ++pos) {
        out.input.push_back(int(uint8_t(text[pos])));
        const size_t next = pos + 1;
        const int token = uint8_t(text[next]);
        out.target.push_back(next < start ? -1 : token);
        Role r = context;
        const bool digit = token >= '0' && token <= '9';
        if (next >= start && next < mark) r = digit ? step_digit : step_other;
        else if (next >= mark && next < answer) r = delimiter;
        else if (next >= answer) r = digit ? answer_digit : answer_other;
        out.role.push_back(r);
    }
    return out;
}
struct Metric {
    double loss_sum = 0;
    size_t tokens = 0, correct = 0;
    double mean() const { return tokens ? loss_sum / double(tokens) : 0; }
};
struct Score {
    std::array<Metric, roles> by_role{};
    Metric all;
};
template<class T>
Score score(const T* logits, int vocab, const ByteSequence& seq) {
    CHECK(vocab >= 256 && seq.target.size() == seq.role.size(), "invalid byte diagnostic shape");
    Score out;
    for (size_t row = 0; row < seq.target.size(); ++row) {
        const int target = seq.target[row];
        if (target < 0) continue;
        const T* z = logits + row * size_t(vocab);
        int top = 0;
        for (int v = 0; v < vocab; ++v) {
            CHECK(std::isfinite(z[v]), "non-finite diagnostic logits");
            if (z[v] > z[top]) top = v;
        }
        double sum = 0;
        for (int v = 0; v < vocab; ++v) sum += std::exp(double(z[v]) - double(z[top]));
        const double loss = std::log(sum) + (double(z[top]) - double(z[target]));
        auto add = [&](Metric& m) { m.loss_sum += loss; ++m.tokens; m.correct += top == target; };
        add(out.by_role[seq.role[row]]); add(out.all);
    }
    return out;
}
inline void merge(Score& into, const Score& from) {
    auto add = [](Metric& x, const Metric& y) { x.loss_sum += y.loss_sum; x.tokens += y.tokens; x.correct += y.correct; };
    add(into.all, from.all);
    for (int r = 0; r < roles; ++r) add(into.by_role[r], from.by_role[r]);
}
} // namespace reasoning_metrics
