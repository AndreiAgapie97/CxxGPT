// calc.h — exact, bounded arithmetic for generated calculator calls.
#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

namespace calc {

// Keep reduced values in signed 64 bits. Wider intermediates let us detect
// overflow before narrowing; unsupported sizes fail rather than round.
struct Rational { int64_t num = 0, den = 1; };

inline std::optional<Rational> reduced(__int128 num, __int128 den) {
    if (!den) return {};
    if (den < 0) num = -num, den = -den;
    __int128 a = num < 0 ? -num : num, b = den;
    while (b) { const auto r = a % b; a = b; b = r; }
    num /= a; den /= a;
    constexpr auto limit = std::numeric_limits<int64_t>::max();
    if (num < -__int128(limit) || num > limit || den > limit) return {};
    return Rational{int64_t(num), int64_t(den)};
}

class Parser {
    std::string_view text;
    size_t pos = 0;
    bool valid = true;
    int depth = 0;
    void space() { while (pos < text.size() && std::string_view(" \t\n\r\f\v").find(text[pos]) != std::string_view::npos) ++pos; }
    Rational combine(Rational a, Rational b, char op) {
        const __int128 an = a.num, ad = a.den, bn = b.num, bd = b.den;
        std::optional<Rational> r;
        if (op == '+') r = reduced(an * bd + bn * ad, ad * bd);
        if (op == '-') r = reduced(an * bd - bn * ad, ad * bd);
        if (op == '*') r = reduced(an * bn, ad * bd);
        if (op == '/') r = reduced(an * bd, ad * bn);
        if (!r) valid = false;
        return r.value_or(Rational{});
    }
    Rational atom() {
        space();
        if (++depth > 128) { valid = false; --depth; return {}; }
        Rational r;
        if (pos < text.size() && (text[pos] == '+' || text[pos] == '-')) {
            const char sign = text[pos++]; r = atom();
            if (sign == '-') r.num = -r.num;
        } else if (pos < text.size() && text[pos] == '(') {
            ++pos; r = expression(); space();
            if (pos == text.size() || text[pos++] != ')') valid = false;
        } else {
            __int128 num = 0, den = 1;
            bool decimal = false, digit = false;
            constexpr auto limit = std::numeric_limits<int64_t>::max();
            while (pos < text.size()) {
                const char c = text[pos];
                if (c == '.' && !decimal) { decimal = true; ++pos; continue; }
                if (c < '0' || c > '9') break;
                digit = true; ++pos; num = num * 10 + c - '0';
                if (decimal) den *= 10;
                if (num > limit || den > limit) { valid = false; break; }
            }
            if (!digit) valid = false;
            const auto value = reduced(num, den);
            if (!value) valid = false;
            r = value.value_or(Rational{});
        }
        --depth; return r;
    }
    Rational product() {
        Rational r = atom(); space();
        while (valid && pos < text.size() && (text[pos] == '*' || text[pos] == '/')) {
            const char op = text[pos++]; r = combine(r, atom(), op); space();
        }
        return r;
    }
    Rational expression() {
        Rational r = product(); space();
        while (valid && pos < text.size() && (text[pos] == '+' || text[pos] == '-')) {
            const char op = text[pos++]; r = combine(r, product(), op); space();
        }
        return r;
    }
public:
    explicit Parser(std::string_view s) : text(s) {}
    std::optional<Rational> parse() {
        if (text.size() > 4096) return {};
        Rational r = expression();
        if (!valid || pos != text.size()) return {};
        return r;
    }
};

inline std::optional<std::string> evaluate(std::string_view expression) {
    const auto r = Parser(expression).parse();
    if (!r) return {};
    int64_t d = r->den;
    while (d % 2 == 0) d /= 2;
    while (d % 5 == 0) d /= 5;
    // A repeating decimal cannot be printed exactly in GSM8K's format.
    if (d != 1) return {};
    const uint64_t n = uint64_t(r->num < 0 ? -r->num : r->num);
    std::string out = (r->num < 0 ? "-" : "") + std::to_string(n / r->den);
    __int128 rest = n % r->den;
    if (rest) out += '.';
    while (rest) { rest *= 10; out += char('0' + rest / r->den); rest %= r->den; }
    return out;
}

enum class Mode { none, notes, plain };
struct Counts { size_t fired = 0, malformed = 0; };

// Called only after a model token, never after our own result tokens.
// nullopt means no call, an empty string means a malformed call.
inline std::optional<std::string> response(std::string_view text, Mode mode) {
    if (mode == Mode::none || text.empty()) return {};
    std::string expr;
    if (mode == Mode::notes) {
        if (text.back() != '=') return {};
        const size_t begin = text.rfind("<<");
        if (begin == std::string_view::npos || text.find(">>", begin) != std::string_view::npos) return {};
        expr = text.substr(begin + 2, text.size() - begin - 3);
    } else {
        if (!text.ends_with(" = ")) return {};
        const size_t end = text.size() - 3;
        size_t begin = end;
        // Only arithmetic bytes, spaces and multiplication aliases belong
        // to the suffix. Never rescue part of an unsupported expression.
        while (begin > 0) {
            const unsigned char c = text[begin - 1];
            if ((c >= '0' && c <= '9') || std::string_view(".+-*/() x\t").find(char(c)) != std::string_view::npos ||
                c == 0xc3 || c == 0x97) --begin;
            else break;
        }
        if (begin == end) return {};
        if (begin && text[begin] != ' ' && text[begin] != '\t' && text[begin - 1] != '\n' &&
            text[begin - 1] != ':' && text[begin - 1] != '$') return std::string{};
        if (begin) {
            const char before = text[begin - 1];
            const auto letter = [](char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); };
            if (letter(before)) {
                // A preceding prose word is fine ("She has 16 * 7"), but
                // do not discard an unsupported word/operator between
                // operands, as in "2 X 3 + 4" or "2 \\times 3 + 4".
                size_t word = begin;
                while (word && letter(text[word - 1])) --word;
                size_t left = word;
                while (left && (text[left - 1] == ' ' || text[left - 1] == '\t')) --left;
                if (left && ((text[left - 1] >= '0' && text[left - 1] <= '9') ||
                             std::string_view(".+-*/()^%,\\=<>_@&").find(text[left - 1]) != std::string_view::npos))
                    return std::string{};
            } else if (std::string_view("\n\r:;$").find(before) == std::string_view::npos) {
                return std::string{};
            }
        }
        expr = text.substr(begin, end - begin);
        const size_t first = expr.find_first_not_of(" \t");
        if (first == std::string::npos) return {};
        if (std::string_view("*/x").find(expr[first]) != std::string_view::npos) return std::string{};
        if (expr[first] == '+' || expr[first] == '-') {
            if (first + 1 == expr.size() || expr[first + 1] == ' ' || expr[first + 1] == '\t') return std::string{};
            // "a -3*2" may be subtraction from a named value. Accept a
            // leading sign only at a clear clause boundary, never after a word.
            if (begin && text[begin - 1] != '\n' && text[begin - 1] != ':' && text[begin - 1] != '$')
                return std::string{};
        }
        for (size_t i = 0; (i = expr.find("×", i)) != std::string::npos;) expr.replace(i, 2, "*");
        for (char& c : expr) if (c == 'x') c = '*';
        // A bare number or a named assignment is not a calculator call.
        if (expr.find_first_of("+-*/") == std::string::npos) return {};
    }
    auto result = evaluate(expr);
    if (!result) return std::string{};
    if (mode == Mode::notes) *result += ">>";
    return result;
}
} // namespace calc
