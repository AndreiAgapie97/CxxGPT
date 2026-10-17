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
    // A chain, link by link: "6 * 4 + 2 = 24 + 2" and "24 + 2 = 26" (rp1
    // wrote this one; it was read as "6 * 4 + 2 = 24", wrong, and the second
    // link was unreadable).
    {
        const Problem bags = problem("Tom has 6 bags of 4 apples and 2 more apples. How many apples?", "26");
        const auto l = label_answer(bags, "He has 6 * 4 + 2 = 24 + 2 = 26 apples.\n#### 26\n");
        EXPECT(l.label == Label::correct && l.calcs.size() == 2 && l.wrong_calcs() == 0 && l.unreadable == 0,
               "a right chain: %s, %zu calcs, %zu wrong, %zu unreadable", labels::label_name(l.label),
               l.calcs.size(), l.wrong_calcs(), l.unreadable);
        EXPECT(l.calcs.size() == 2 && l.calcs[0].wrote == "24 + 2" && l.calcs[0].exact == "26" &&
                   l.calcs[1].wrote == "26" && l.calcs[1].exact == "26" &&
                   l.calcs[1].text == "He has 6 * 4 + 2 = 24 + 2 = 26",
               "a right chain's links");
        // A wrong first link: 6 * 4 + 2 is not 25 + 2; 25 + 2 is 27.
        const auto first = label_answer(bags, "6 * 4 + 2 = 25 + 2 = 27\n#### 27\n");
        EXPECT(first.label == Label::arithmetic && first.calcs.size() == 2 && !first.calcs[0].ok &&
                   first.calcs[0].exact == "26" && first.calcs[1].ok,
               "a wrong first link: %s", labels::label_name(first.label));
        // A wrong second link: 24 + 2 is not 28.
        const auto second = label_answer(bags, "6 * 4 + 2 = 24 + 2 = 28\n#### 28\n");
        EXPECT(second.label == Label::arithmetic && second.calcs.size() == 2 && second.calcs[0].ok &&
                   !second.calcs[1].ok && second.calcs[1].exact == "26",
               "a wrong second link: %s", labels::label_name(second.label));
        // Three links, with "x" for times.
        const auto three = label_answer(bags, "2 x 3 x 4 + 2 = 6 x 4 + 2 = 24 + 2 = 26\n#### 26\n");
        EXPECT(three.label == Label::correct && three.calcs.size() == 3 && three.wrong_calcs() == 0 &&
                   three.unreadable == 0,
               "three links: %zu calcs, %zu wrong, %zu unreadable", three.calcs.size(), three.wrong_calcs(),
               three.unreadable);
        // After words, the first link is not a calculation, the second is.
        const auto words = label_answer(bags, "The total = 24 + 2 = 26\n#### 26\n");
        EXPECT(words.calcs.size() == 1 && words.calcs[0].ok && words.calcs[0].exact == "26" && words.unreadable == 0,
               "the total = 24 + 2 = 26: %zu calcs, %zu unreadable", words.calcs.size(), words.unreadable);
        // Text that is not all arithmetic is no link: the first number is read.
        const auto prose = label_answer(bags, "6 * 4 + 2 = 26 apples, so 26 = 26\n#### 26\n");
        EXPECT(prose.calcs.size() == 1 && prose.calcs[0].ok && prose.calcs[0].wrote == "26",
               "26 apples, so ...: %zu calcs", prose.calcs.size());
    }
    // A calculation after a word that comes right after a number (gen-math's
    // fractions without notes, and rp1, write "30% of 270 is 270*30/100 =
    // 81"): read and checked, as the calculator would read it at a clause's
    // start; not after a one-letter or operator word, where the word may
    // join two numbers.
    {
        const Problem marbles = problem("There are 270 marbles; 30% are given away. How many are left?", "189");
        const auto right = label_answer(marbles, "30% of 270 is 270*30/100 = 81, so 270-81 = 189 are left.\n#### 189\n");
        EXPECT(right.calcs.size() == 2 && right.wrong_calcs() == 0 && right.unreadable == 0 &&
                   right.calcs[0].expression == "270*30/100" && right.calcs[0].exact == "81",
               "after 'is': %zu calcs, %zu wrong, %zu unreadable", right.calcs.size(), right.wrong_calcs(),
               right.unreadable);
        const auto wrong = label_answer(marbles, "30% of 270 is 270*30/100 = 80, so 270-80 = 190.\n#### 190\n");
        EXPECT(wrong.calcs.size() == 2 && wrong.wrong_calcs() == 1 && wrong.label == Label::arithmetic,
               "a wrong one after 'is': %zu calcs, %zu wrong, %s", wrong.calcs.size(), wrong.wrong_calcs(),
               labels::label_name(wrong.label));
        for (const char* text : {"2 X 3 + 4 = 10\n#### 10\n", "2 times 3 + 4 = 10\n#### 10\n",
                                 "1/2 of 8 + 2 = 6\n#### 6\n"}) {
            const auto l = label_answer(marbles, text);
            EXPECT(l.calcs.empty() && l.unreadable == 1, "'%s': %zu calcs, %zu unreadable", text, l.calcs.size(),
                   l.unreadable);
        }
    }
    // Not calculations: an "=" after a bare number, or after words.
    {
        const auto l = label_answer(ducks, "Half of 10 = 5 and the total = 9.\n#### 9\n");
        EXPECT(l.calcs.empty() && l.unreadable == 0, "not calculations: %zu checked, %zu unreadable", l.calcs.size(),
               l.unreadable);
    }
    // The right number, written another way.
    for (const char* written : {"$18", "18 dollars", "18.00"}) {
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
