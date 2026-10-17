// failure_labels.h (#16): each label on answers whose cause is known, two of
// them night 2's real answers to GSM8K's first test problem.
#pragma once

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "../src/failure_labels.h"
#include "../src/generate.h"
#include "test_util.h"

inline void test_failure_labels() {
    using labels::Label;
    using labels::label_answer;
    auto problem = [](const std::string& question, const std::string& answer) {
        Problem p;
        p.task = "gsm8k";
        p.question = question;
        p.answer = answer;
        return p;
    };
    const Problem ducks = problem(
        "Janet\xe2\x80\x99s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins "
        "for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh "
        "duck egg. How much in dollars does she make every day at the farmers' market?",
        "18");

    // Night 2, step 500: brings in "30 days in a week", then 480 * 480 is
    // not 112,400. The first wrong calculation decides the label.
    {
        const std::string got =
            "The ducks lay 16 eggs per day.\nThere are 30 days in a week.\nSo, the ducks lay 16 * 30 = 480 eggs per "
            "week.\nThere are 480 * 480 = 112,400 eggs per week.\nThere are 112,400 / 480 = 20 weeks in a week.\nSo, "
            "the ducks lay 112400 / 20 = 5600 eggs per week.\nThere are 5600 / 560 = 10 weeks in a week.\nSo, the "
            "ducks lay 10 * 5600 = 560000 eggs per week.\nThus, the ducks lay \\boxed{5600000} dollars in "
            "dollars.\n#### 5600000\n";
        const auto l = label_answer(ducks, got);
        EXPECT(l.label == Label::arithmetic, "step 500: %s", labels::label_name(l.label));
        // 112,400 / 480 does not end as a decimal: it cannot be checked.
        EXPECT(l.calcs.size() == 5 && l.wrong_calcs() == 3 && l.unreadable == 1,
               "step 500: %zu calculations, %zu wrong, %zu unreadable", l.calcs.size(), l.wrong_calcs(), l.unreadable);
        EXPECT(!l.calcs.empty() && l.calcs[0].ok && !l.calcs[1].ok && l.calcs[1].exact == "230400",
               "step 500: the first wrong calculation is 480 * 480");
        EXPECT(!l.outside.empty() && l.outside[0] == "30", "step 500: 30 days in a week is not from the question");
        EXPECT(!l.gold_in_work && !l.loop, "step 500: flags");
    }
    // Night 2, step 4500: earns the right 18 dollars a day, then multiplies
    // by the 7 days of a week.
    {
        const std::string got =
            "Janet's ducks lay 16 eggs per day for breakfast.\nShe eats three for breakfast every morning, so she eats "
            "3 * 3 = 9 eggs per day.\nShe sells each egg for $2, so she earns 9 * 2 = 18 dollars per day.\nIn a week, "
            "she earns 18 * 7 = 126 dollars.\nThus, Janet makes \\boxed{126} dollars every day at the farmers' "
            "market.\n#### 126\n";
        const auto l = label_answer(ducks, got);
        EXPECT(l.label == Label::passed_gold, "step 4500: %s", labels::label_name(l.label));
        EXPECT(l.gold_in_work && l.calcs.size() == 3 && l.wrong_calcs() == 0, "step 4500: %zu calcs, %zu wrong",
               l.calcs.size(), l.wrong_calcs());
    }

    const Problem apples =
        problem("Tom has 48 apples and gives half of them away. Then he buys 16 bags of 7 apples. How many apples?",
                "136");
    // Calculator notes, checked; GSM8K's "48/2 = <<48/2=24>>24" checked once.
    {
        const auto l = label_answer(apples, "He keeps 48/2 = <<48/2=24>>24.\nHe buys 16 x 7 = <<16*7=112>>112.\n"
                                            "24 + 112 = <<24+112=136>>136\n#### 136\n");
        EXPECT(l.label == Label::correct && l.calcs.size() == 3 && l.wrong_calcs() == 0 && l.unreadable == 0,
               "notes: %s, %zu calcs, %zu wrong, %zu unreadable", labels::label_name(l.label), l.calcs.size(),
               l.wrong_calcs(), l.unreadable);
    }
    {
        const auto l = label_answer(apples, "He keeps <<48/2=24>>24.\nHe buys <<16*7=102>>102.\n#### 126\n");
        EXPECT(l.label == Label::arithmetic && l.wrong_calcs() == 1 && l.calcs[1].exact == "112",
               "a wrong note: %s", labels::label_name(l.label));
    }
    // Plain products with "x" and "$", and digit-grouping commas.
    {
        const auto l = label_answer(problem("Each of 1,200 tickets costs $50 and 3 more cost $50.", "60150"),
                                    "1,200 x $50 = $60,000\n$50 x 3 = $150\n60,000 + 150 = 60,150\n#### 60,150\n");
        EXPECT(l.label == Label::correct && l.calcs.size() == 3 && l.wrong_calcs() == 0,
               "plain: %s, %zu calcs, %zu wrong", labels::label_name(l.label), l.calcs.size(), l.wrong_calcs());
    }
    // Not calculations: an "=" after a bare number, or after words.
    {
        const auto l = label_answer(ducks, "Half of 10 = 5 and the total = 9.\n#### 9\n");
        EXPECT(l.calcs.empty() && l.unreadable == 0, "not calculations: %zu checked, %zu unreadable", l.calcs.size(),
               l.unreadable);
    }
    // The right number written another way is right, as scoring has it
    // (canonical_answer); the right number among others, or with the wrong
    // sign, is a format error.
    for (const char* written : {"$18", "18 dollars", "18.00"}) {
        const auto l = label_answer(ducks, "16 - 3 - 4 = 9\n9 * 2 = 18\n#### " + std::string(written) + "\n");
        EXPECT(l.label == Label::correct, "#### %s: %s", written, labels::label_name(l.label));
    }
    for (const char* written : {"18 or 19", "-18", "about 18"}) {
        const auto l = label_answer(ducks, "16 - 3 - 4 = 9\n9 * 2 = 18\n#### " + std::string(written) + "\n");
        EXPECT(l.label == Label::format, "#### %s: %s", written, labels::label_name(l.label));
    }
    // The first event decides: the right number before a wrong calculation
    // (passed_gold), a wrong calculation before the right number (arithmetic).
    {
        const auto passed = label_answer(ducks, "16 - 3 - 4 = 9\n9 * 2 = 18\n18 * 7 = 120\n#### 120\n");
        EXPECT(passed.label == Label::passed_gold, "18, then 18 * 7 = 120: %s", labels::label_name(passed.label));
        const auto wrong = label_answer(ducks, "16 - 3 - 4 = 8\n8 * 2 = 18\n18 + 18 = 36\n#### 36\n");
        EXPECT(wrong.label == Label::arithmetic, "16 - 3 - 4 = 8, then 18: %s", labels::label_name(wrong.label));
    }
    // Right, with a wrong calculation on the way.
    {
        const auto l = label_answer(ducks, "16 - 3 - 4 = 8\n9 * 2 = 18\n#### 18\n");
        EXPECT(l.label == Label::correct && l.wrong_calcs() == 1, "lucky: %s", labels::label_name(l.label));
    }
    // Unfinished: a loop, and an answer line without its newline (as
    // completion_done() has it).
    {
        const std::string loop = "She sells the remainder for $2 per egg.\nShe sells the remainder for $2 per egg.\n"
                                 "She sells the remainder for $2 per egg.\n";
        const auto l = label_answer(ducks, loop);
        EXPECT(l.label == Label::unfinished && l.loop, "loop: %s", labels::label_name(l.label));
        const auto cut = label_answer(ducks, "9 * 2 = 18\n#### 18");
        EXPECT(cut.label == Label::unfinished && !completion_done("9 * 2 = 18\n#### 18"),
               "an answer line cut off: %s", labels::label_name(cut.label));
    }
    // Other: right arithmetic on the wrong plan.
    {
        const auto l = label_answer(problem("Tom has 3 bags of 4 apples. How many apples?", "12"), "3 + 4 = 7\n#### 7\n");
        EXPECT(l.label == Label::other, "wrong plan: %s", labels::label_name(l.label));
    }
    // The right number counts only as a whole number, and not when the
    // question already gives it.
    {
        const auto l = label_answer(ducks, "There are 118 eggs.\n#### 5\n");
        EXPECT(l.label == Label::other && !l.gold_in_work, "118 is not 18: %s", labels::label_name(l.label));
        const Problem dogs = problem("Ann has 6 cats and 4 dogs, and gives 4 cats away. How many dogs?", "4");
        const auto d = label_answer(dogs, "6 - 4 = 2\n#### 2\n");
        EXPECT(d.label == Label::other && !d.gold_in_work, "the question's own 4: %s", labels::label_name(d.label));
    }
    // Outside numbers: words of the question count, one step from what is
    // known counts, a new fact does not.
    {
        const auto l = label_answer(ducks, "She eats three and bakes four, so 16 - 3 - 4 = 9 are left; 9 + 2 is 11.\n"
                                           "A week has 7 days... and 365 days make a year.\n#### 3285\n");
        EXPECT(l.outside.size() == 1 && l.outside[0] == "365", "outside: %zu numbers%s%s", l.outside.size(),
               l.outside.empty() ? "" : ", first ", l.outside.empty() ? "" : l.outside[0].c_str());
    }
    // repeats_run uses NoRepeat on the answer's tokens.
    {
        const std::vector<int> looped{5, 6, 7, 5, 6, 7}, fresh{5, 6, 7, 8, 6, 9};
        EXPECT(label_answer(ducks, "x", &looped, 3).repeats_run && !label_answer(ducks, "x", &fresh, 3).repeats_run,
               "repeats_run with runs of 3 tokens");
    }
    // JSON: quotes, backslashes, newlines, control bytes, UTF-8 kept, and
    // bytes that are not UTF-8 replaced.
    {
        const std::string s = "a \"q\" \\boxed{1}\n\t\x01 Janet\xe2\x80\x99s \xff \xe2\x80";
        const std::string want = "a \\\"q\\\" \\\\boxed{1}\\n\\t\\u0001 Janet\xe2\x80\x99s \xef\xbf\xbd "
                                 "\xef\xbf\xbd\xef\xbf\xbd";
        EXPECT(labels::json_escape(s) == want, "json_escape: %s", labels::json_escape(s).c_str());
        const auto l = label_answer(ducks, "9 * 2 = 18\n#### 18\n");
        const std::string line = labels::json_line(7, ducks, l, "9 * 2 = 18\n#### 18\n");
        EXPECT(line.starts_with("{\"problem\":7,\"task\":\"gsm8k\",\"difficulty\":1,\"label\":\"correct\"") &&
                   line.find("\"calcs\":[{\"text\":\"9 * 2 = 18\",\"wrote\":\"18\",\"exact\":\"18\",\"ok\":true}]") !=
                       std::string::npos &&
                   line.ends_with("\"completion\":\"9 * 2 = 18\\n#### 18\\n\"}"),
               "json_line: %s", line.c_str());
    }
    // The file: the settings line, then one line per problem, numbered by ids.
    {
        const std::string path = test_path("labels.jsonl");
        const std::vector<Problem> ps{ducks, ducks};
        const std::vector<std::string> answers{"9 * 2 = 18\n#### 18\n", "loop"};
        const auto s = labels::write_labels(path, "{\"run\":1}", ps, answers, {4, 9},
                                            [](const std::string& t) { return std::vector<int>(t.begin(), t.end()); });
        std::ifstream in(path, std::ios::binary);
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        in.close();
        std::remove(path.c_str());
        EXPECT(s.total == 2 && text.starts_with("{\"run\":1}\n{\"problem\":4,") &&
                   text.find("\n{\"problem\":9,") != std::string::npos &&
                   std::count(text.begin(), text.end(), '\n') == 3,
               "write_labels: %s", text.c_str());
    }
    // The summary counts every label once.
    {
        labels::LabelSummary s;
        s.add(label_answer(ducks, "9 * 2 = 18\n#### 18\n"));
        s.add(label_answer(ducks, "no answer"));
        s.add(label_answer(ducks, "3 + 4 = 8\n#### 8\n"));
        EXPECT(s.total == 3 && s.count[size_t(Label::correct)] == 1 && s.count[size_t(Label::unfinished)] == 1 &&
                   s.count[size_t(Label::arithmetic)] == 1 && s.wrong_calcs == 1,
               "summary counts");
    }
}
