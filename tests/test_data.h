// test_data.h — the JSON reader and the problems file format.
#pragma once

#include <algorithm>
#include <cstdio>
#include <filesystem>

#include "../src/json.h"
#include "../src/reasoning.h"
#include "../src/train.h"
#include "test_util.h"

inline void test_json() {
    Json j;
    std::string err;
    EXPECT(Json::parse(R"( {"a": [1, -2.5e3, true, false, null], "b": {"c": "x"}, "d": ""} )", j, &err),
           "valid JSON rejected: %s", err.c_str());
    EXPECT(j.type == Json::Object && j.keys.size() == 3 && j.items.size() == 3, "wrong object");
    const Json* arr = j.get("a");
    EXPECT(arr && arr->type == Json::Array && arr->items.size() == 5, "wrong array");
    EXPECT(arr->items[0].number == 1 && arr->items[1].number == -2500, "wrong numbers");
    EXPECT(arr->items[2].boolean && arr->items[2].type == Json::Bool && !arr->items[3].boolean &&
               arr->items[4].type == Json::Null,
           "wrong literals");
    EXPECT(j.get("b")->get("c")->str == "x" && j.get("d")->str.empty() && !j.get("zz"), "wrong lookup");

    // Escapes, including a surrogate pair, decode to UTF-8.
    EXPECT(Json::parse(R"("q\"\\\/\b\f\n\r\t \u00e9 \u20ac \ud83d\ude00")", j, &err), "escapes: %s", err.c_str());
    EXPECT(j.str == "q\"\\/\b\f\n\r\t \xC3\xA9 \xE2\x82\xAC \xF0\x9F\x98\x80", "escapes decoded wrongly");

    // Invalid documents are rejected, without crashing.
    const char* bad[] = {"", "{", "[1,]", "{\"a\" 1}", "{\"a\":1,}", "01", "1.", "-", "1e", "tru",
                         "\"abc", "\"\\x\"", "\"\\ud83d\"", "\"\\ude00\"", "\"\\u12\"", "\"a\nb\"",
                         "[1] 2", "{1: 2}", "nul"};
    for (const char* b : bad) EXPECT(!Json::parse(b, j, &err), "invalid JSON accepted: %s", b);
    std::string deep(1000, '[');
    EXPECT(!Json::parse(deep, j, &err), "deep nesting accepted");
    EXPECT(!Json::parse("1e999",j,&err), "overflowed JSON number was accepted");
    EXPECT(Json::parse("9007199254740993",j,&err) && j.str=="9007199254740993", "numeric spelling lost integer precision");
}

inline void test_problem_file() {
    std::vector<Problem> ps(2);
    ps[0] = {"gsm8k", 1, "a\tb \\ c", "line 1\nline 2\\n is literal", "42"};
    ps[1] = {"chain", 3, "x=1", "x is 1", "1"};
    // Into folders that do not exist yet: writing creates them.
    const std::string dir = test_path("new_dir");
    const std::string path = dir + "/a/b/problems.txt";
    std::filesystem::remove_all(dir);
    save_problems(path, ps);
    const std::vector<Problem> got = load_problems(path);
    std::filesystem::remove_all(dir);
    EXPECT(got.size() == 2, "read %zu problems", got.size());
    for (size_t i = 0; i < got.size() && i < 2; ++i)
        EXPECT(got[i].task == ps[i].task && got[i].difficulty == ps[i].difficulty &&
                   got[i].question == ps[i].question && got[i].steps == ps[i].steps && got[i].answer == ps[i].answer,
               "problem %zu changed in the round trip", i);
}

// A token file written in pieces reads back as one.
inline void test_token_writer() {
    const std::string path = test_path("tokens.tok");
    std::vector<uint16_t> all;
    {
        TokenWriter w(path, 1000);
        for (int piece = 0; piece < 3; ++piece) {
            std::vector<uint16_t> t;
            for (int i = 0; i < 100 + piece; ++i) t.push_back(uint16_t((piece * 37 + i * 11) % 1000));
            w.write(t);
            all.insert(all.end(), t.begin(), t.end());
        }
        w.write({});
    }  // the destructor finishes the file
    const TokenData d = TokenData::load(path);
    std::remove(path.c_str());
    EXPECT(d.vocab == 1000 && d.mask.empty() && d.tokens == all, "token file read back wrongly (%zu tokens)",
           d.tokens.size());
    // An empty file (gen-reasoning --val 0 writes one) saves and loads too.
    TokenData empty;
    empty.vocab = 50;
    empty.save(path);
    const TokenData e = TokenData::load(path);
    std::remove(path.c_str());
    EXPECT(e.vocab == 50 && e.tokens.empty() && e.mask.empty(), "empty token file read back wrongly");
}

// Files mixed with repeats: tokens in order, masks kept, files without a
// mask trained on everywhere, the largest vocabulary, and the header peek.
inline void test_token_list() {
    TokenData a, b;
    a.vocab = 300;
    b.vocab = 500;
    for (int i = 0; i < 7; ++i) a.tokens.push_back(uint16_t(i * 41 % 300));
    for (int i = 0; i < 5; ++i) b.tokens.push_back(uint16_t(400 + i)), b.mask.push_back(uint8_t(i % 2));
    const std::string pa = test_path("list_a.tok"), pb = test_path("list_b.tok");
    a.save(pa);
    b.save(pb);
    const TokenData::Header h = TokenData::peek(pb);
    EXPECT(h.vocab == 500 && h.has_mask == 1 && h.n == 5, "header of a masked file read wrongly");
    const TokenData d = TokenData::load_list({{pa, 1}, {pb, 2}, {pa, 1}});
    const TokenData plain = TokenData::load_list({{pa, 2}});
    std::remove(pa.c_str());
    std::remove(pb.c_str());
    std::vector<uint16_t> tokens;
    std::vector<uint8_t> mask;
    for (const TokenData* part : {&a, &b, &b, &a}) {
        tokens.insert(tokens.end(), part->tokens.begin(), part->tokens.end());
        if (part->mask.empty()) mask.insert(mask.end(), part->tokens.size(), uint8_t(1));
        else mask.insert(mask.end(), part->mask.begin(), part->mask.end());
    }
    EXPECT(d.vocab == 500 && d.tokens == tokens && d.mask == mask, "mixed token files combined wrongly");
    EXPECT(plain.vocab == 300 && plain.mask.empty() && plain.tokens.size() == 14 &&
               std::equal(a.tokens.begin(), a.tokens.end(), plain.tokens.begin() + 7),
           "a repeated unmasked file combined wrongly");
}

inline void test_data() {
    test_json();
    test_problem_file();
    test_token_writer();
    test_token_list();
}
