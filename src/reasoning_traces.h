// Native add/cmp statement checks (tolerating whitespace/leading zeros). These
// check arithmetic and grounding in the question, not arbitrary proof prose.
#pragma once
#include <charconv>
#include <string_view>
#include "reasoning.h"

namespace reasoning_traces {
inline std::string_view trim(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\n' || s.front() == '\r')) s.remove_prefix(1);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\n' || s.back() == '\r')) s.remove_suffix(1);
    return s;
}
inline bool number(std::string_view s, int64_t& v) {
    s = trim(s);
    if (s.empty() || s.front() < '0' || s.front() > '9') return false;
    const auto result = std::from_chars(s.data(), s.data() + s.size(), v);
    return result.ec == std::errc{} && result.ptr == s.data() + s.size();
}
inline bool pair(std::string_view s, char sep, int64_t& a, int64_t& b) {
    const size_t pos = s.find(sep);
    return pos != s.npos && number(s.substr(0, pos), a) && number(s.substr(pos + 1), b);
}
inline std::vector<std::string_view> statements(std::string_view s, char sep) {
    std::vector<std::string_view> out;
    s = trim(s);
    if (s.empty()) return out;
    for (;;) {
        const size_t pos = s.find(sep);
        out.push_back(trim(s.substr(0, pos)));
        if (pos == s.npos) break;
        s.remove_prefix(pos + 1);
    }
    return out;
}
struct Score {
    size_t expected = 0, generated = 0, local_correct = 0, grounded_correct = 0;
    bool all_grounded() const { return expected > 0 && generated == expected && grounded_correct == expected; }
};
inline Score score(const Problem& p, std::string_view completion) {
    const size_t mark = completion.find("\n#### ");
    const auto body = trim(completion.substr(0, mark));
    Score out;
    int64_t a = 0, b = 0;
    if (p.task == "add") {
        CHECK(pair(p.question, '+', a, b), "unsupported addition question");
        const auto steps = statements(body, ',');
        out.generated = steps.size();
        int carry = 0;
        for (size_t i = 0; a > 0 || b > 0; ++i, a /= 10, b /= 10) {
            const int x = int(a % 10), y = int(b % 10), sum = x + y + carry;
            ++out.expected;
            if (i < steps.size()) {
                const auto s = steps[i];
                const size_t plus = s.find('+'), eq = s.find('=');
                int64_t dx = 0, dy = 0, dc = 0, z = 0;
                const bool valid = plus != s.npos && eq != s.npos && plus < eq &&
                    number(s.substr(0, plus), dx) && pair(s.substr(plus + 1, eq - plus - 1), '+', dy, dc) &&
                    number(s.substr(eq + 1), z) && dx <= 9 && dy <= 9 && dc <= 1;
                const bool local = valid && z == dx + dy + dc;
                out.local_correct += local;
                out.grounded_correct += local && ((dx == x && dy == y) || (dx == y && dy == x)) &&
                                        dc == carry && z == sum;
            }
            carry = sum / 10;
        }
        // Extra statements cannot make a full trace valid; they also enter
        // the generated denominator, rather than disappearing from counts.
    } else {
        const std::string_view q(p.question);
        CHECK(p.task == "cmp" && q.starts_with("max(") && q.ends_with(')') &&
              pair(q.substr(4, q.size() - 5), ',', a, b), "unsupported comparison question");
        const auto steps = statements(body, ';');
        out.generated = steps.size();
        const std::string sa = std::to_string(a), sb = std::to_string(b);
        if (sa.size() != sb.size()) {
            out.expected = 1;
            if (!steps.empty()) {
                const size_t sep = steps[0].find(" digits vs ");
                int64_t x = 0, y = 0;
                const bool local = sep != steps[0].npos && number(steps[0].substr(0, sep), x) &&
                    number(steps[0].substr(sep + 11), y) && x > 0 && y > 0 && x != y;
                out.local_correct = local;
                out.grounded_correct = local && size_t(x) == sa.size() && size_t(y) == sb.size();
            }
        } else {
            for (size_t i = 0; i < sa.size(); ++i) {
                ++out.expected;
                const char relation = sa[i] == sb[i] ? '=' : sa[i] < sb[i] ? '<' : '>';
                if (i < steps.size()) {
                    const auto s = steps[i];
                    const bool digits = s.size() == 3 && s[0] >= '0' && s[0] <= '9' && s[2] >= '0' && s[2] <= '9';
                    const char rel = digits ? (s[0] == s[2] ? '=' : s[0] < s[2] ? '<' : '>') : '?';
                    const bool local = digits && s[1] == rel;
                    out.local_correct += local;
                    out.grounded_correct += local && s[0] == sa[i] && s[2] == sb[i] && s[1] == relation;
                }
                if (relation != '=') break;
            }
        }
    }
    return out;
}
}  // namespace reasoning_traces
