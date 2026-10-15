#pragma once

#include "../src/json.h"
#include "../src/reason_report.h"
#include "test_util.h"

inline void test_reason_report() {
    const std::string prefix = test_path("reason-report");
    const std::vector<Problem> problems{
        {"gsm8k", 1, "A \"quoted\" question\nwith 1 apple", "", "2"},
        {"gsm8k", 1, "Another question", "", "1200"},
        {"gsm8k", 1, "Third question", "", "3"},
    };
    const auto encode = [](const std::string& s) { return std::vector<int>(s.begin(), s.end()); };
    auto read = [&](long step) {
        const std::string path = prefix + ".step-" + std::to_string(step) + ".jsonl";
        std::ifstream in(path, std::ios::binary);
        std::vector<Json> rows;
        std::string line;
        while (std::getline(in, line)) {
            Json row;
            EXPECT(Json::parse(line, row), "invalid report JSON: %s", line.c_str());
            rows.push_back(std::move(row));
        }
        in.close();
        std::remove(path.c_str());
        EXPECT(rows.size() == 5, "expected settings, three problems and summary");
        return rows;
    };
    ReasonReport report;
    report.write(prefix, 500, "{\"step\":500}", problems,
                 {"#### 2\n", "unfinished", "#### 3"}, {0, 2, 5}, encode);
    const auto first = read(500);
    if (first.size() != 5) return;
    EXPECT(first[0].get("step")->number == 500 && first[2].get("problem")->number == 2 &&
           first[1].get("question")->str == problems[0].question, "settings, source ids or question lost");
    const auto& baseline = first.back();
    EXPECT(baseline.get("previous_step")->type == Json::Null &&
           baseline.get("became_correct")->items.empty() && baseline.get("regressed")->items.empty() &&
           baseline.get("labels")->get("correct")->number == 1 &&
           baseline.get("labels")->get("unfinished")->number == 2, "initial baseline or labels wrong");
    report.write(prefix, 1000, "{\"step\":1000}", problems,
                 {"#### 9\n", "#### 1,200\n", "#### 3\n"}, {0, 2, 5}, encode);
    const auto second = read(1000);
    if (second.size() != 5) return;
    const auto& changes = second.back();
    const auto& gained = changes.get("became_correct")->items;
    const auto& lost = changes.get("regressed")->items;
    EXPECT(changes.get("previous_step")->number == 500 && gained.size() == 2 && lost.size() == 1,
           "transition counts wrong");
    if (gained.size() == 2 && lost.size() == 1)
        EXPECT(gained[0].number == 2 && gained[1].number == 5 && lost[0].number == 0,
               "transitions must retain source ids; comma answers and truncated lines follow scoring");
    report.write(prefix, 1500, "{}", problems, {"#### 9\n", "#### 1200\n", "#### 3\n"}, {0, 2, 5}, encode);
    const auto unchanged = read(1500);
    if (unchanged.size() == 5)
        EXPECT(unchanged.back().get("became_correct")->items.empty() &&
               unchanged.back().get("regressed")->items.empty(), "unchanged results must have no transitions");
    ReasonReport resumed;
    resumed.write(prefix, 2000, "{}", problems, {"#### 9\n", "#### 1200\n", "#### 3\n"}, {0, 2, 5}, encode);
    const auto fresh = read(2000);
    if (fresh.size() == 5)
        EXPECT(fresh.back().get("previous_step")->type == Json::Null, "a resumed invocation needs a new baseline");
}
