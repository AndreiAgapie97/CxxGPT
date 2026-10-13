// test_data.h — the JSON reader and the problems file format.
#pragma once

#include <cstdio>

#include "../src/json.h"
#include "../src/reasoning.h"
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
}

inline void test_problem_file() {
    std::vector<Problem> ps(2);
    ps[0] = {"gsm8k", 1, "a\tb \\ c", "line 1\nline 2\\n is literal", "42"};
    ps[1] = {"chain", 3, "x=1", "x is 1", "1"};
    const std::string path = "/tmp/cxxgpt_test_problems.txt";
    save_problems(path, ps);
    const std::vector<Problem> got = load_problems(path);
    std::remove(path.c_str());
    EXPECT(got.size() == 2, "read %zu problems", got.size());
    for (size_t i = 0; i < got.size() && i < 2; ++i)
        EXPECT(got[i].task == ps[i].task && got[i].difficulty == ps[i].difficulty &&
                   got[i].question == ps[i].question && got[i].steps == ps[i].steps && got[i].answer == ps[i].answer,
               "problem %zu changed in the round trip", i);
}

inline void test_data() {
    test_json();
    test_problem_file();
}
