// json.h — a small JSON reader, for dataset files (one object per line).
//
// Parses the whole grammar (RFC 8259): objects, arrays, strings with every
// escape including surrogate pairs (decoded to UTF-8), finite-range numbers
// (with their exact spelling retained), true, false,
// null. Objects keep their keys in file order. Nesting is limited to 256
// levels, so hostile input cannot exhaust the stack.
#pragma once

#include <charconv>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

struct Json {
    enum Type { Null, Bool, Number, String, Array, Object };
    Type type = Null;
    bool boolean = false;
    double number = 0.0;
    std::string str;
    std::vector<Json> items;        // Array elements, or Object values
    std::vector<std::string> keys;  // Object keys (keys[i] names items[i])

    // The value of key in an object, or nullptr.
    const Json* get(std::string_view key) const {
        for (size_t i = 0; i < keys.size(); ++i)
            if (keys[i] == key) return &items[i];
        return nullptr;
    }

    // Parses all of text (surrounding whitespace allowed). On failure
    // returns false and describes the problem in *error.
    static bool parse(std::string_view text, Json& out, std::string* error = nullptr) {
        Parser p{text, 0, {}};
        p.skip_ws();
        bool ok = p.value(out, 0);
        if (ok) {
            p.skip_ws();
            if (p.i != text.size()) ok = p.fail("trailing characters");
        }
        if (!ok && error) *error = p.err + " at offset " + std::to_string(p.i);
        return ok;
    }

private:
    struct Parser {
        std::string_view s;
        size_t i = 0;
        std::string err;

        bool fail(const char* what) {
            if (err.empty()) err = what;
            return false;
        }
        void skip_ws() {
            while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
        }
        bool literal(std::string_view word) {
            if (s.substr(i, word.size()) != word) return fail("invalid literal");
            i += word.size();
            return true;
        }

        bool value(Json& out, int depth) {
            if (depth > 256) return fail("nesting too deep");
            if (i >= s.size()) return fail("unexpected end");
            out = Json{};
            switch (s[i]) {
                case '{': return object(out, depth);
                case '[': return array(out, depth);
                case '"': out.type = String; return string(out.str);
                case 't': out.type = Bool; out.boolean = true; return literal("true");
                case 'f': out.type = Bool; return literal("false");
                case 'n': return literal("null");
                default: {
                    const size_t start = i;
                    out.type = Number;
                    if (!number(out.number)) return false;
                    out.str = std::string(s.substr(start,i-start));  // retain exact decimal spelling for numeric labels
                    return true;
                }
            }
        }

        bool object(Json& out, int depth) {
            out.type = Object;
            ++i;  // '{'
            skip_ws();
            if (i < s.size() && s[i] == '}') return ++i, true;
            while (true) {
                skip_ws();
                if (i >= s.size() || s[i] != '"') return fail("expected a key");
                std::string key;
                if (!string(key)) return false;
                skip_ws();
                if (i >= s.size() || s[i] != ':') return fail("expected ':'");
                ++i;
                skip_ws();
                Json v;
                if (!value(v, depth + 1)) return false;
                out.keys.push_back(std::move(key));
                out.items.push_back(std::move(v));
                skip_ws();
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == '}') return ++i, true;
                return fail("expected ',' or '}'");
            }
        }

        bool array(Json& out, int depth) {
            out.type = Array;
            ++i;  // '['
            skip_ws();
            if (i < s.size() && s[i] == ']') return ++i, true;
            while (true) {
                skip_ws();
                Json v;
                if (!value(v, depth + 1)) return false;
                out.items.push_back(std::move(v));
                skip_ws();
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == ']') return ++i, true;
                return fail("expected ',' or ']'");
            }
        }

        bool hex4(uint32_t& v) {
            if (i + 4 > s.size()) return fail("truncated \\u escape");
            v = 0;
            for (int k = 0; k < 4; ++k) {
                const char c = s[i++];
                v <<= 4;
                if (c >= '0' && c <= '9') v |= uint32_t(c - '0');
                else if (c >= 'a' && c <= 'f') v |= uint32_t(c - 'a' + 10);
                else if (c >= 'A' && c <= 'F') v |= uint32_t(c - 'A' + 10);
                else return fail("bad hex digit in \\u escape");
            }
            return true;
        }

        static void put_utf8(std::string& o, uint32_t cp) {
            if (cp < 0x80) {
                o += char(cp);
            } else if (cp < 0x800) {
                o += char(0xC0 | (cp >> 6));
                o += char(0x80 | (cp & 0x3F));
            } else if (cp < 0x10000) {
                o += char(0xE0 | (cp >> 12));
                o += char(0x80 | ((cp >> 6) & 0x3F));
                o += char(0x80 | (cp & 0x3F));
            } else {
                o += char(0xF0 | (cp >> 18));
                o += char(0x80 | ((cp >> 12) & 0x3F));
                o += char(0x80 | ((cp >> 6) & 0x3F));
                o += char(0x80 | (cp & 0x3F));
            }
        }

        bool string(std::string& o) {
            ++i;  // opening quote
            while (true) {
                if (i >= s.size()) return fail("unterminated string");
                const unsigned char c = s[i++];
                if (c == '"') return true;
                if (c < 0x20) return fail("control character in string");
                if (c != '\\') {
                    o += char(c);
                    continue;
                }
                if (i >= s.size()) return fail("unterminated escape");
                switch (s[i++]) {
                    case '"': o += '"'; break;
                    case '\\': o += '\\'; break;
                    case '/': o += '/'; break;
                    case 'b': o += '\b'; break;
                    case 'f': o += '\f'; break;
                    case 'n': o += '\n'; break;
                    case 'r': o += '\r'; break;
                    case 't': o += '\t'; break;
                    case 'u': {
                        uint32_t cp;
                        if (!hex4(cp)) return false;
                        if (cp >= 0xD800 && cp < 0xDC00) {  // high surrogate: a low one must follow
                            uint32_t lo;
                            if (s.substr(i, 2) != "\\u") return fail("lone high surrogate");
                            i += 2;
                            if (!hex4(lo)) return false;
                            if (lo < 0xDC00 || lo >= 0xE000) return fail("bad low surrogate");
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        } else if (cp >= 0xDC00 && cp < 0xE000) {
                            return fail("lone low surrogate");
                        }
                        put_utf8(o, cp);
                        break;
                    }
                    default: return fail("bad escape");
                }
            }
        }

        bool number(double& v) {
            const size_t start = i;
            auto digits = [&] {
                const size_t d = i;
                while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
                return i > d;
            };
            if (i < s.size() && s[i] == '-') ++i;
            if (i < s.size() && s[i] == '0') ++i;
            else if (!digits()) return fail("invalid value");
            if (i < s.size() && s[i] == '.') {
                ++i;
                if (!digits()) return fail("digits expected after '.'");
            }
            if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
                ++i;
                if (i < s.size() && (s[i] == '+' || s[i] == '-')) ++i;
                if (!digits()) return fail("digits expected in exponent");
            }
            const auto parsed = std::from_chars(s.data()+start,s.data()+i,v);  // locale-independent
            return parsed.ec == std::errc{} && parsed.ptr == s.data()+i ? true : fail("number out of range");
        }
    };
};
