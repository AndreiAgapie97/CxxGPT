#pragma once
#include "../src/reasoning_traces.h"
#include "../src/generate.h"
#include "test_util.h"

inline void test_reasoning_traces() {
    Problem large; large.task="add"; large.question="1000001+1234567";
    EXPECT(reasoning_traces::score(large,"1+7+0=8, 0+6+0=6, 0+5+0=5, 0+4+0=4, 0+3+0=3, 0+2+0=2, 1+1+0=2\n#### 2234568\n").all_grounded(),
           "native-format trace rejected a representable larger operand");

    Problem p; p.task = "add"; p.question = "12+19"; p.answer = "31";
    auto s = reasoning_traces::score(p, "2+9+0=11, 1+1+1=3\n#### 31\n");
    EXPECT(s.all_grounded() && s.expected == 2 && s.local_correct == 2, "correct carried addition rejected");
    s = reasoning_traces::score(p, "9+2+0=11, 1+1+1=3\n#### 31\n");
    EXPECT(s.all_grounded(), "commuted digit addition rejected");
    const std::string copied = "8+3+0=11, 1+1+1=3\n#### 31\n";
    s = reasoning_traces::score(p, copied);
    EXPECT(extract_answer(copied) == p.answer && s.local_correct == 2 && !s.all_grounded(),
           "correct answer with locally valid but ungrounded arithmetic passed");
    for (const std::string bad : {"2+9+0=10, 1+1+1=3", "2+9+0=11", "2+9+0=11, 1+1+0=2",
                                 "2+9+0=11, 1+1+1=3, 0+0+0=0", "", "2+9+0=11,"})
        EXPECT(!reasoning_traces::score(p, bad + "\n#### 31\n").all_grounded(), "invalid add trace passed");
    p.task = "cmp"; p.question = "max(123,129)"; p.answer = "129";
    s = reasoning_traces::score(p, "1=1; 2=2; 3<9\n#### 129\n");
    EXPECT(s.all_grounded() && s.expected == 3, "correct lexicographic comparison rejected");
    s = reasoning_traces::score(p, "1=1; 2=2; 8<9\n#### 129\n");
    EXPECT(s.local_correct == 3 && !s.all_grounded(), "wrong copied comparison digit passed");
    p.question = "max(99,100)";
    EXPECT(reasoning_traces::score(p, "2 digits vs 3\n#### 100\n").all_grounded(), "different lengths rejected");
    EXPECT(!reasoning_traces::score(p, "3 digits vs 2\n#### 100\n").all_grounded(), "wrong lengths passed");
    p.task = "add"; p.question = "12+19";
    EXPECT(!reasoning_traces::score(p, std::string(200, '9') + "+2+0=11").all_grounded(), "overflow text passed");
    Rng rng(81);
    for (const std::string task : {"add", "cmp"})
        for (int d = 1; d <= 4; ++d)
            for (int i = 0; i < 100; ++i) {
                const auto q = make_problem(task, d, rng);
                EXPECT(reasoning_traces::score(q, q.completion()).all_grounded(), "native gold trace rejected");
            }
}
