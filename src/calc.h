// calc.h — exact, bounded arithmetic for generated calculator calls.
#pragma once

#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>

namespace calc {

// A fraction num/den, always in lowest terms, with den > 0. Both parts fit
// in signed 64 bits and num is never INT64_MIN, so -num is always safe.
// Every step is exact: a value that does not fit is rejected, never rounded.
// There is no 128-bit arithmetic: 128-bit division needs runtime helpers
// (__divti3, __modti3) that the Windows (MSVC) runtime does not ship.
struct Rational { int64_t num = 0, den = 1; };

constexpr int64_t kMin = std::numeric_limits<int64_t>::min();

// a * b and a + b, or nothing when the exact result does not fit (INT64_MIN
// counts as not fitting, see Rational).
inline std::optional<int64_t> mul(int64_t a, int64_t b) {
    int64_t r = 0;
    if (__builtin_mul_overflow(a, b, &r) || r == kMin) return {};
    return r;
}
inline std::optional<int64_t> add(int64_t a, int64_t b) {
    int64_t r = 0;
    if (__builtin_add_overflow(a, b, &r) || r == kMin) return {};
    return r;
}

// num/den in lowest terms (den > 0). std::gcd works on |num|.
inline Rational lowest(int64_t num, int64_t den) {
    const int64_t g = std::gcd(num, den);
    return {num / g, den / g};
}

// a * b. Cancelling crosswise first (a.num with b.den, b.num with a.den)
// leaves the result in lowest terms, so the two products are exactly the
// answer's numerator and denominator: one overflows only if the answer
// does not fit.
inline std::optional<Rational> times(Rational a, Rational b) {
    const int64_t g1 = std::gcd(a.num, b.den), g2 = std::gcd(b.num, a.den);
    const auto num = mul(a.num / g1, b.num / g2), den = mul(a.den / g2, b.den / g1);
    if (!num || !den) return {};
    return Rational{*num, *den};
}

// a / b is a times b turned upside down (keeping the sign on top).
inline std::optional<Rational> divide(Rational a, Rational b) {
    if (b.num == 0) return {};
    return times(a, b.num < 0 ? Rational{-b.den, -b.num} : Rational{b.den, b.num});
}

// a + b over the smallest common denominator (Knuth's method). With
// g = gcd(a.den, b.den), the scaled numerators and their sum are the
// answer's numerator times a factor of g. They must fit too, which only
// rejects an answer that would fit for values close to 2^63.
inline std::optional<Rational> plus(Rational a, Rational b) {
    const int64_t g = std::gcd(a.den, b.den);
    const auto left = mul(a.num, b.den / g), right = mul(b.num, a.den / g);
    if (!left || !right) return {};
    const auto sum = add(*left, *right);
    if (!sum) return {};
    // a and b are in lowest terms, so the sum can share a factor only with g.
    const int64_t g2 = std::gcd(*sum, g);
    const auto den = mul(a.den / g, b.den / g2);
    if (!den) return {};
    return Rational{*sum / g2, *den};
}

class Parser {
    std::string_view text;
    size_t pos = 0;
    bool valid = true;
    int depth = 0;
    void space() { while (pos < text.size() && std::string_view(" \t\n\r\f\v").find(text[pos]) != std::string_view::npos) ++pos; }
    Rational combine(Rational a, Rational b, char op) {
        std::optional<Rational> r;
        if (op == '+') r = plus(a, b);
        if (op == '-') r = plus(a, Rational{-b.num, b.den});
        if (op == '*') r = times(a, b);
        if (op == '/') r = divide(a, b);
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
            int64_t num = 0, den = 1;
            bool decimal = false, digit = false;
            while (pos < text.size()) {
                const char c = text[pos];
                if (c == '.' && !decimal) { decimal = true; ++pos; continue; }
                if (c < '0' || c > '9') break;
                digit = true; ++pos;
                // num = num * 10 + digit, and one more decimal place after
                // the point, unless that no longer fits.
                const auto times10 = mul(num, 10);
                const auto next = times10 ? add(*times10, c - '0') : std::optional<int64_t>();
                const auto places = decimal ? mul(den, 10) : std::optional<int64_t>(den);
                if (!next || !places) { valid = false; break; }
                num = *next; den = *places;
            }
            if (!digit) valid = false;
            if (valid) r = lowest(num, den);
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

inline bool is_digit(char c) { return c >= '0' && c <= '9'; }

// The expression without its digit-grouping commas: "24*600,000" is
// "24*600000". Only a strict group counts: a number that starts with 1-3
// digits, the first not 0, with no digit or "." right before it, then one
// or more ",ddd", and no digit right after. Every other comma stays, so
// the parser refuses it: "1,2345", "1,2", "2,3+4" (a list), "1234,567",
// "1.5,000", and "0,125" (a decimal comma in some countries).
// (operations/build_gsm8k_notes.py's ungroup() is the same rule as a regex.)
inline std::string ungroup(std::string_view text) {
    std::string out;
    size_t i = 0;
    while (i < text.size()) {
        const bool starts = is_digit(text[i]) && text[i] != '0' && (i == 0 || (!is_digit(text[i - 1]) && text[i - 1] != '.'));
        if (!starts) { out += text[i++]; continue; }
        size_t end = i;
        while (end < text.size() && is_digit(text[end])) ++end;
        std::string number(text.substr(i, end - i));
        if (end - i <= 3) {
            // A group is "," and three digits, and no digit after them.
            while (end + 3 < text.size() && text[end] == ',' && is_digit(text[end + 1]) && is_digit(text[end + 2]) &&
                   is_digit(text[end + 3]) && (end + 4 == text.size() || !is_digit(text[end + 4]))) {
                number += text.substr(end + 1, 3);
                end += 4;
            }
        }
        out += number;
        i = end;
    }
    return out;
}

inline std::optional<std::string> evaluate(std::string_view expression) {
    const auto r = Parser(ungroup(expression)).parse();
    if (!r) return {};
    int64_t d = r->den;
    while (d % 2 == 0) d /= 2;
    while (d % 5 == 0) d /= 5;
    // A repeating decimal cannot be printed exactly in GSM8K's format.
    if (d != 1) return {};
    const uint64_t n = uint64_t(r->num < 0 ? -r->num : r->num), den = uint64_t(r->den);
    std::string out = (r->num < 0 ? "-" : "") + std::to_string(n / den);
    uint64_t rest = n % den;
    if (rest) out += '.';
    while (rest) {
        // The next digit is 10 * rest / den, but 10 * rest may not fit in
        // 64 bits. So add rest ten times, taking den away whenever the sum
        // reaches it: the sum stays below 2 * den, which always fits.
        int digit = 0;
        uint64_t sum = 0;
        for (int k = 0; k < 10; ++k) {
            sum += rest;
            if (sum >= den) { sum -= den; ++digit; }
        }
        out += char('0' + digit);
        rest = sum;
    }
    return out;
}

enum class Mode { none, notes, plain };
struct Counts { size_t fired = 0, malformed = 0; };

inline bool is_blank(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

// In plain mode "x" means times only as in "16 x 7": a space on each side,
// and beyond the spaces a digit on each side. In "3x + 2" or "2 x3" it is
// a variable: algebra, not arithmetic.
inline bool times_sign(std::string_view text, size_t i) {
    if (i == 0 || i + 1 >= text.size() || text[i - 1] != ' ' || text[i + 1] != ' ') return false;
    size_t left = i - 1, right = i + 1;
    while (left > 0 && text[left - 1] == ' ') --left;
    while (right < text.size() && text[right] == ' ') ++right;
    return left > 0 && is_digit(text[left - 1]) && right < text.size() && is_digit(text[right]);
}

// Called only after a model token, never after our own result tokens.
// nullopt means no call, an empty string means a malformed call.
// `expression`, if given, gets what was worked out ("16 * 7" for "16 x 7"),
// for a call whose expression could be read.
inline std::optional<std::string> response(std::string_view text, Mode mode, std::string* expression = nullptr) {
    if (mode == Mode::none || text.empty()) return {};
    std::string expr;
    if (mode == Mode::notes) {
        if (text.back() != '=') return {};
        const size_t begin = text.rfind("<<");
        if (begin == std::string_view::npos || text.find(">>", begin) != std::string_view::npos) return {};
        expr = text.substr(begin + 2, text.size() - begin - 3);
        // A note is a call only at its first "=". A note left open
        // ("<<5/0=unknown ...") was already counted as malformed there;
        // a later "=" must not count it again.
        if (expr.find('=') != std::string::npos) return {};
        // "<<24*600,000=": the model copies the question's numbers with
        // their commas. Worked out (and reported) without them.
        expr = ungroup(expr);
    } else {
        // " = $" is a trigger too: in "$50 x 3 = $150" the number goes
        // after the "$".
        size_t end = 0;
        if (text.ends_with(" = ")) end = text.size() - 3;
        else if (text.ends_with(" = $")) end = text.size() - 4;
        else return {};
        size_t begin = end;
        // Only arithmetic bytes, spaces and multiplication aliases belong
        // to the suffix. Never rescue part of an unsupported expression.
        // An "x" that is not a times sign ends the suffix like any letter.
        while (begin > 0) {
            const unsigned char c = text[begin - 1];
            // ". " ends a sentence: in "He has 3 bags. 16 * 7 = " the
            // expression starts after it.
            if (c == '.' && is_blank(text[begin])) break;
            if (is_digit(char(c)) || std::string_view(".+-*/() \t").find(char(c)) != std::string_view::npos ||
                c == 0xc3 || c == 0x97 || (c == 'x' && times_sign(text, begin - 1))) --begin;
            else break;
        }
        if (begin == end) return {};
        // Where a clause starts, so an expression may start: the start of
        // the text, right after a newline, ':' or '$', or after one of
        // ". , ; ! ?" followed by whitespace ("So, 16 * 7 = ").
        const auto clause_start = [&](size_t i) {
            if (i == 0) return true;
            const char c = text[i - 1];
            if (c == '\n' || c == '\r' || c == ':' || c == '$') return true;
            return std::string_view(".,;!?").find(c) != std::string_view::npos && i < text.size() && is_blank(text[i]);
        };
        if (!clause_start(begin)) {
            // Otherwise only a prose word and a space may come before
            // ("She has 16 * 7"). But do not drop an unsupported word or
            // operator between operands, as in "2 X 3 + 4" or
            // "2 \\times 3 + 4": the word must not follow a number or an
            // operator, unless a clause starts there ("8, and 8 * 2").
            const auto letter = [](char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); };
            if (!letter(text[begin - 1]) || !is_blank(text[begin])) return std::string{};
            size_t word = begin;
            while (word && letter(text[word - 1])) --word;
            size_t left = word;
            while (left && (text[left - 1] == ' ' || text[left - 1] == '\t')) --left;
            if (left && !clause_start(left) &&
                (is_digit(text[left - 1]) ||
                 std::string_view(".+-*/()^%,\\=<>_@&").find(text[left - 1]) != std::string_view::npos))
                return std::string{};
        }
        expr = text.substr(begin, end - begin);
        const size_t first = expr.find_first_not_of(" \t");
        if (first == std::string::npos) return {};
        if (expr[first] == '*' || expr[first] == '/') return std::string{};
        if (expr[first] == '+' || expr[first] == '-') {
            if (first + 1 == expr.size() || expr[first + 1] == ' ' || expr[first + 1] == '\t') return std::string{};
            // "a -3*2" may be subtraction from a named value. Accept a
            // leading sign only where a clause starts, never after a word.
            if (!clause_start(begin)) return std::string{};
        }
        for (size_t i = 0; (i = expr.find("×", i)) != std::string::npos;) expr.replace(i, 2, "*");
        for (char& c : expr) if (c == 'x') c = '*';  // every "x" left is a times sign
        // A bare number or a named assignment is not a calculator call.
        if (expr.find_first_of("+-*/") == std::string::npos) return {};
        // No sign right after an operator: "3* + 2" (from algebra such as
        // "3x + 2") would read as 3 * (+2). In brackets, "3 * (-2)", it is fine.
        for (size_t i = 0; i < expr.size(); ++i) {
            if (expr[i] != '+' && expr[i] != '-') continue;
            size_t j = i;
            while (j > 0 && (expr[j - 1] == ' ' || expr[j - 1] == '\t')) --j;
            if (j > 0 && std::string_view("+-*/").find(expr[j - 1]) != std::string_view::npos) return std::string{};
        }
    }
    auto result = evaluate(expr);
    if (!result) return std::string{};
    if (expression) *expression = expr;
    if (mode == Mode::notes) *result += ">>";
    return result;
}
} // namespace calc
