// test_redraw_qa.h — human word problems written again with other numbers
// (redraw_qa.h): which problems convert and why the others do not, the
// numbers drawn, and every variant checked against a formula written here
// by hand, apart from the tool's own.
#pragma once

#include "../src/redraw_qa.h"
#include "test_util.h"

namespace {

inline Problem qa(const std::string& question, const std::string& steps, const std::string& answer) {
    return {"gsm8k", 1, question, steps, answer};
}

// `formula` with "#k" for the variant's k-th question number (from 1),
// worked out exactly: the answer a person would compute from the variant's
// question alone.
inline std::optional<calc::Rational> by_hand(const std::string& formula, const std::string& question) {
    const auto numbers = labels::detail::numbers_in(question);
    std::string text;
    for (size_t i = 0; i < formula.size(); ++i) {
        if (formula[i] != '#') {
            text += formula[i];
            continue;
        }
        size_t k = 0;
        while (i + 1 < formula.size() && labels::detail::digit(formula[i + 1])) k = k * 10 + size_t(formula[++i] - '0');
        if (k == 0 || k > numbers.size()) return {};
        text += "(" + numbers[k - 1].written + ")";
    }
    return calc::Parser(text).parse();
}

struct Fixture {
    const char* name;
    Problem p;
    redraw_qa::Reason reason;
    const char* formula;  // by hand, over the variant's question numbers; "" for a rejection
};

inline std::vector<Fixture> redraw_qa_fixtures() {
    using R = redraw_qa::Reason;
    return {
        {"half as many",
         qa("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many "
            "clips did Natalia sell altogether in April and May?",
            "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in "
            "April and May.",
            "72"),
         R::converted, "#1+#1/2"},
        // A decimal result and 60 minutes an hour (a constant from nowhere,
        // allowed where the problem speaks of minutes).
        {"per minute",
         qa("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did "
            "she earn?",
            "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = "
            "$<<0.2*50=10>>10.",
            "10"),
         R::converted, "#1/60*#2"},
        // 100 stays (a usual constant), "half" and "twice" are words.
        {"wallet",
         qa("Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her "
            "parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How "
            "much more money does Betty need to buy the wallet?",
            "In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.\nBetty's grandparents gave her 15 * 2 = "
            "$<<15*2=30>>30.\nThis means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.",
            "5"),
         R::converted, "#1-#1/2-#2*2-#2"},
        // A percentage used as its hundredth (".1", read as 0.1).
        {"percent as a hundredth",
         qa("John weighs 220 pounds when he starts exercising.  He manages to lose 10% of his body weight.  He then "
            "gains back 2 pounds.  How much does he weigh at the end?",
            "He lost 220*.1=<<220*.1=22>>22 pounds\nSo after gaining some weight back he was 22-2=<<22-2=20>>20 pounds "
            "lighter than before\nSo his weight is 220-20=<<220-20=200>>200 pounds",
            "200"),
         R::converted, "#1-(#1*#2/100-#3)"},
        // The final result 2 and the question's 2 (fixed) share a value, but
        // the result is only worked out at the end: the first "2" is the
        // question's.
        {"a result worked out later",
         qa("James watched 2 episodes of Jeopardy and 2 episodes of Wheel of Fortune.  Jeopardy is 20 minutes and "
            "Wheel of Fortune is twice as long.  How many hours did he watch TV?",
            "He watched 2*20=<<2*20=40>>40 minutes of Jeopardy.\nWheel of Fortune is 2*20=<<2*20=40>>40 minutes each.\n"
            "So he watched it for 40*2=<<40*2=80>>80 minutes.\nSo he watched 40+80=<<40+80=120>>120 minutes of "
            "TV.\nThat "
            "means he watched 120/60=<<120/60=2>>2 hours of TV.",
            "2"),
         R::converted, "(#1*#3+#3*2*#2)/60"},
        // A year the solution never uses stays, so does the ordinal; a
        // plain calculation without a note is rewritten too; "an 8".
        {"year, ordinal, plain calculation",
         qa("In 2015, Ann worked an 8-hour shift on 5 days of the 3rd week. How many hours did she work?",
            "She worked 8 x 5 = 40 hours.", "40"),
         R::converted, "#2*#3"},
        // 16 slices, and 2 x 8 = 16: the later "16" could be either.
        {"a result equal to a question number",
         qa("Albert buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 "
            "slices. If he eats it all, how many pieces does he eat that day?",
            "He eats 32 from the largest pizzas because 2 x 16 = <<2*16=32>>32\nHe eats 16 from the small pizza "
            "because 2 x 8 = <<2*8=16>>16\nHe eats 48 pieces because 32 + 16 = <<32+16=48>>48",
            "48"),
         R::ambiguous, ""},
        // "3 sets of 20 minutes": 3 is 60 / 20 worked out in the head. It
        // would stay 3 when 20 changes.
        {"mental arithmetic",
         qa("Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
            "In one hour, there are 3 sets of 20 minutes.\nSo, Joy can read 8 x 3 = <<8*3=24>>24 pages in an hour.\n"
            "It will take her 120/24 = <<120/24=5>>5 hours to read 120 pages.",
            "5"),
         R::constant, ""},
        // 12 is $6 a month over two months, worked out in the head: a unit's
        // number (12 months a year) only where it is no step from the
        // question's numbers. It would stay 12 when the 6 changes.
        {"a unit's number from the head",
         qa("Sam saves $6 per month. How much does he have after two months and a $6 gift?",
            "He saves 12 dollars in two months, and 12+6=<<12+6=18>>18 dollars with the gift.", "18"),
         R::constant, ""},
        // "x 5 days" are the five weekdays, not Saturday's 5 hours: the
        // same value by chance (GSM8K train row 5204).
        {"another unit",
         qa("Mike practices for 3 hours every weekday, on Saturdays he practices for 5 hours, and he takes Sundays "
            "off. How many hours will he practice from now until the next game, if his team has a game in 3 weeks?",
            "Mike practices on weekdays 3 hours/day x 5 days = <<3*5=15>>15 hours in total.\nThen each week he will "
            "practice 15 hours + 5 hours = <<15+5=20>>20 hours in total.\nHe will practice before the game 20 "
            "hours/week x 3 weeks = <<20*3=60>>60 hours in total.",
            "60"),
         R::units, ""},
        {"an unused number",
         qa("Tom has 9 apples and 4 pears. He eats 2 apples. How many apples does he have left?",
            "He has 9-2=<<9-2=7>>7 apples left.", "7"),
         R::unused, ""},
        // "4 divided by 2 equals <<4/2=2>>2": the first 2 is the constant
        // (two people), not the result on its line.
        {"a constant or a later result",
         qa("Phil and Andre order a small pizza with 8 slices and a large one with 14 slices. They have both eaten 9 "
            "slices already. How many pieces are left per person?",
            "There are 22 pieces of pizza because 14 plus 8 equals <<14+8=22>>22.\nThey have eaten 18 pieces of pizza "
            "because 9 plus 9 equals <<9+9=18>>18.\nThere are 4 pieces of pizza left because 22 minus 18 equals "
            "<<22-18=4>>4.\nThere are 2 pieces per person because 4 divided by 2 equals <<4/2=2>>2.",
            "2"),
         R::text_ambiguous, ""},
        {"a number in words",
         qa("Sam has 5 friends. Each friend gets 3 cards. How many cards does Sam give away?",
            "Sam gives his five friends 5*3=<<5*3=15>>15 cards.", "15"),
         R::words, ""},
        {"a wrong gold calculation",
         qa("Sam has 5 friends. Each friend gets 3 cards. How many cards does Sam give away?",
            "Sam gives 5*3=<<5*3=16>>16 cards.", "16"),
         R::wrong, ""},
        {"nothing to change",
         qa("Ken has 2 pounds of jelly beans and adds enough to triple the weight. What is the weight now?",
            "The weight is 2*3=<<2*3=6>>6 pounds.", "6"),
         R::nothing, ""},
        {"no calculation", qa("Ken has 7 books. How many books does he have?", "He has 7 books.", "7"), R::no_calcs,
         ""},
    };
}

inline void test_redraw_qa_fixtures() {
    redraw_qa::Options o;
    o.versions = 3;
    size_t converted = 0, other_numbers = 0;
    for (const Fixture& f : redraw_qa_fixtures()) {
        const redraw_qa::Outcome r = redraw_qa::redraw_problem(f.p, 7, o);
        EXPECT(r.reason() == f.reason, "%s: %s (%s), expected %s", f.name, redraw_qa::reason_name(r.reason()),
               r.analysis.detail.c_str(), redraw_qa::reason_name(f.reason));
        if (f.reason != redraw_qa::Reason::converted) {
            EXPECT(r.variants.empty(), "%s: a rejected problem has variants", f.name);
            continue;
        }
        EXPECT(r.variants.size() == 3, "%s: %zu variants of 3", f.name, r.variants.size());
        std::set<std::string> questions = {f.p.question};
        const Problem original = redraw_qa::detail::normalized(f.p);
        for (const Problem& v : r.variants) {
            // The tool's own final check, and a formula written by hand.
            const auto answer = calc::Parser(v.answer).parse();
            EXPECT(answer && redraw_qa::check_variant(original, r.analysis, v, *answer).empty(),
                   "%s: a variant fails the final check:\n%s\n%s", f.name, v.question.c_str(), v.steps.c_str());
            const auto expected = by_hand(f.formula, v.question);
            EXPECT(expected && answer && labels::detail::same(*expected, *answer),
                   "%s: the answer %s is not %s by hand:\n%s\n%s", f.name, v.answer.c_str(),
                   expected ? redraw_qa::detail::decimal(*expected).value_or("?").c_str() : "?", v.question.c_str(),
                   v.steps.c_str());
            EXPECT(questions.insert(v.question).second, "%s: a variant repeats a question", f.name);
            EXPECT(v.task == "gsm8k-redraw", "%s: variants are marked as re-drawn", f.name);
        }
        // The same seed and row give the same variants; another row others.
        const redraw_qa::Outcome again = redraw_qa::redraw_problem(f.p, 7, o);
        EXPECT(again.variants.size() == r.variants.size() &&
                   (r.variants.empty() || again.variants.back().question == r.variants.back().question),
               "%s: the same seed must give the same variants", f.name);
        // Another row draws other numbers (a problem can have only a few
        // variants: "a result worked out later" has three).
        const redraw_qa::Outcome other = redraw_qa::redraw_problem(f.p, 8, o);
        ++converted;
        other_numbers +=
            !other.variants.empty() && !r.variants.empty() && other.variants[0].question != r.variants[0].question;
    }
    EXPECT(other_numbers * 2 > converted, "another row must draw other numbers (%zu of %zu)", other_numbers, converted);
    // What stays and what changes in "year, ordinal, plain calculation".
    const Problem shift = redraw_qa_fixtures()[5].p;
    const redraw_qa::Outcome r = redraw_qa::redraw_problem(shift, 0, o);
    for (const Problem& v : r.variants) {
        EXPECT(
            v.question.starts_with("In 2015, Ann worked a") && v.question.find("of the 3rd week") != std::string::npos,
            "the year and the ordinal stay: %s", v.question.c_str());
        const auto hours = labels::detail::numbers_in(v.question)[1].written;
        EXPECT(
            v.question.find((redraw_qa::detail::needs_an(hours) ? "an " : "a ") + hours + "-hour") != std::string::npos,
            "the article fits the number: %s", v.question.c_str());
    }
    EXPECT(r.analysis.question.size() == 4 && r.analysis.question[0].fixed_why == "year" &&
               r.analysis.question[3].fixed_why == "ordinal" && r.analysis.variables.size() == 2,
           "2015 is a year, 3rd an ordinal, 8 and 5 change");
    // Weng: the percentage-free decimal result, and the 60 that stays.
    const redraw_qa::Outcome weng = redraw_qa::redraw_problem(redraw_qa_fixtures()[1].p, 0, o);
    for (const Problem& v : weng.variants)
        EXPECT(v.steps.find("/60 = $<<") != std::string::npos, "60 minutes an hour stays: %s", v.steps.c_str());
    EXPECT(redraw_qa::redraw_problem(redraw_qa_fixtures()[3].p, 0, o).analysis.hundredths,
           "the weight problem uses 10%% as 0.1");
}

// The final check finds a variant with a wrong note, a changed word, the
// wrong answer or the original's question.
inline void test_redraw_qa_check() {
    redraw_qa::Options o;
    const Problem p = redraw_qa_fixtures()[0].p;
    const redraw_qa::Outcome r = redraw_qa::redraw_problem(p, 0, o);
    EXPECT(r.variants.size() == 1, "one variant");
    if (r.variants.empty()) return;
    const Problem& v = r.variants[0];
    const calc::Rational answer = *calc::Parser(v.answer).parse();
    EXPECT(redraw_qa::check_variant(p, r.analysis, v, answer).empty(), "the variant passes");
    Problem bad = v;
    const size_t close = bad.steps.find(">>");
    bad.steps.insert(close, "1");  // "<<56/2=28>>" -> "<<56/2=281>>"
    EXPECT(!redraw_qa::check_variant(p, r.analysis, bad, answer).empty(), "a wrong note is found: %s",
           bad.steps.c_str());
    bad = v;
    bad.question.replace(bad.question.find("friends"), 7, "cousins");
    EXPECT(!redraw_qa::check_variant(p, r.analysis, bad, answer).empty(), "a changed word is found");
    bad = v;
    bad.answer += "0";
    EXPECT(!redraw_qa::check_variant(p, r.analysis, bad, *calc::Parser(bad.answer).parse()).empty(),
           "a wrong answer is found");
    EXPECT(!redraw_qa::check_variant(p, r.analysis, p, *calc::Parser(p.answer).parse()).empty(),
           "the original itself is no variant");
}

// New values: as round as the original, the same decimal places, under 100
// for a percentage under 100, never 1, never the original, roughly half to
// twice it; and the kinds results must keep.
inline void test_redraw_qa_draws() {
    using calc::Rational;
    using redraw_qa::detail::draw_value;
    struct Case {
        Rational original;
        size_t decimals;
        bool percent;
        int64_t step_num, step_den;  // every value is a multiple of this
        Rational lo, hi;
        size_t different;  // draws that differ, at least
    };
    for (const Case& c :
         {Case{{120, 1}, 0, false, 10, 1, {60, 1}, {240, 1}, 10}, Case{{48, 1}, 0, false, 1, 1, {24, 1}, {96, 1}, 30},
          Case{{3, 1}, 0, false, 1, 1, {2, 1}, {8, 1}, 6}, Case{{25, 1}, 0, false, 5, 1, {15, 1}, {50, 1}, 6},
          Case{{5, 2}, 2, false, 1, 2, {3, 2}, {5, 1}, 6},     // $2.50: steps of 0.50
          Case{{3, 2}, 1, false, 1, 10, {7, 10}, {3, 1}, 10},  // 1.5: one place, never "2.0"
          Case{{15, 1}, 0, false, 1, 1, {7, 1}, {30, 1}, 15},
          Case{{90, 1}, 0, true, 10, 1, {50, 1}, {90, 1}, 4}}) {  // 90%: under 100
        std::set<std::string> seen;
        for (uint64_t seed = 1; seed <= 300; ++seed) {
            Rng rng(seed);
            const auto v = draw_value(c.original, c.decimals, c.percent, rng);
            if (!v) continue;
            const auto steps = calc::divide(*v, {c.step_num, c.step_den});
            const auto below = calc::plus(*v, {-c.lo.num, c.lo.den}), above = calc::plus(c.hi, {-v->num, v->den});
            EXPECT(steps && steps->den == 1 && below && below->num >= 0 && above && above->num >= 0 &&
                       !labels::detail::same(*v, c.original) && !(v->num == 1 && v->den == 1),
                   "a draw for %s: %s", redraw_qa::detail::rational_text(c.original).c_str(),
                   redraw_qa::detail::rational_text(*v).c_str());
            if (c.decimals == 1)
                EXPECT(redraw_qa::detail::places(redraw_qa::detail::decimal(*v).value_or("")) == 1,
                       "1.5 keeps one decimal place, not 0: %s", redraw_qa::detail::rational_text(*v).c_str());
            seen.insert(redraw_qa::detail::rational_text(*v));
        }
        EXPECT(seen.size() >= c.different, "%s: only %zu different draws",
               redraw_qa::detail::rational_text(c.original).c_str(), seen.size());
    }
    using redraw_qa::detail::same_kind;
    EXPECT(same_kind({24, 1}, {30, 1}) && same_kind({1, 5}, {1, 4}) && same_kind({1, 1}, {1, 1}),
           "whole stays whole, a decimal stays a short decimal, 1 stays 1");
    EXPECT(!same_kind({24, 1}, {49, 2}), "a whole number may not become 24.5");
    EXPECT(!same_kind({5, 1}, {0, 1}) && !same_kind({0, 1}, {3, 1}), "positive stays positive, zero stays zero");
    EXPECT(!same_kind({3, 1}, {1, 1}) && !same_kind({1, 1}, {2, 1}), "1 stays 1 and nothing else becomes it");
    EXPECT(!same_kind({1, 5}, {1, 3}), "no repeating decimal");
    EXPECT(!same_kind({1, 5}, {1, 8}), "0.2 may not become 0.125: more than two places");
    EXPECT(!same_kind({10, 1}, {260, 1}) && !same_kind({260, 1}, {10, 1}), "within 25 times either way");
}

inline void test_redraw_qa_text() {
    using namespace redraw_qa::detail;
    for (const char* an : {"8", "8.5", "11", "18", "80", "800", "11,000", "18000", "8,000,000"})
        EXPECT(needs_an(an), "an %s", an);
    for (const char* a : {"1", "7", "10", "110", "180", "1100", "100"}) EXPECT(!needs_an(a), "a %s", a);
    EXPECT(fix_articles("He works a 8-hour shift", {11}) == "He works an 8-hour shift", "a -> an");
    EXPECT(fix_articles("An 8 was there, and an $11 bill", {3, 24}) == "An 8 was there, and an $11 bill",
           "an stays an");
    EXPECT(fix_articles("An 7 and Ian 8", {3, 13}) == "A 7 and Ian 8", "only the article: '%s'",
           fix_articles("An 7 and Ian 8", {3, 13}).c_str());
    EXPECT(masked_words("an 8-hour shift") == masked_words("a 6-hour shift"), "articles aside, the same words");
    EXPECT(written_like({1234, 1}, 0, true) == "1,234" && written_like({5, 2}, 2, false) == "2.50" &&
               written_like({3, 1}, 2, false) == "3.00" && written_like({1, 4}, 1, false) == "0.25" &&
               written_like({1234567, 1}, 0, true) == "1,234,567",
           "numbers written as the original was");
    EXPECT(
        with_zero_points("It takes .5 hours at $.50 or 18*.5 (.25)") == "It takes 0.5 hours at $0.50 or 18*0.5 (0.25)",
        "%s", with_zero_points("It takes .5 hours at $.50 or 18*.5 (.25)").c_str());
    EXPECT(with_zero_points("No.5 is 3.5 hours.") == "No.5 is 3.5 hours.", "other points stay");
    EXPECT(digits_all_read("12 boxes of $2.50 and 1,200 cards") && !digits_all_read("a .5 hour"),
           "every digit is in a number, or the problem is not read");
    struct Fixed {
        const char* question;
        size_t number;
        const char* why;
    };
    for (const Fixed& f :
         {Fixed{"At 3:30 pm he left.", 0, "time or ratio"}, Fixed{"At 3:30 pm he left.", 1, "time or ratio"},
          Fixed{"The ratio is 5:7 today.", 1, "time or ratio"}, Fixed{"He wakes at 6 am daily.", 0, "time"},
          Fixed{"He wakes at 6am daily.", 0, "time"}, Fixed{"She ate 2/9 of 45 pies.", 1, "fraction"},
          Fixed{"The 4th of 30 days.", 0, "ordinal"}, Fixed{"He paid with a $20 bill and 35 coins.", 0, "bank note"},
          Fixed{"Five 10-dollar bills and 35 coins.", 0, "bank note"}, Fixed{"In 1998 he had 35.", 0, "year"},
          Fixed{"It runs 60 minutes and 35 more.", 0, "unit"}, Fixed{"A gallon is 8 pints; 35 pints.", 0, "unit"},
          Fixed{"He has 8 pints and 35 cups.", 0, ""}, Fixed{"Half of 2 and 35.", 0, "usual"},
          Fixed{"Three times 3 and 35.", 0, "in words"}, Fixed{"He had 35 cats and $1998.", 1, ""},
          Fixed{"He had 35 cats and 9 dogs.", 0, ""}}) {
        const auto numbers = labels::detail::numbers_in(f.question);
        const std::string why =
            why_fixed(f.question, numbers[f.number], labels::detail::number_words(f.question), lower(f.question));
        EXPECT(why == f.why, "'%s', number %zu: '%s', expected '%s'", f.question, f.number, why.c_str(), f.why);
    }
}

// Units: a number of the solution with another unit than the question's,
// and the most a number may become ("5 days a week": 7).
inline void test_redraw_qa_units() {
    using namespace redraw_qa::detail;
    auto differ = [](const std::string& solution, const std::string& question) {
        const auto end = [](const std::string& t) { return std::min(t.find_first_not_of("0123456789"), t.size()); };
        return units_differ(words_after(solution, end(solution)), words_after(question, end(question)));
    };
    EXPECT(differ("5 days a week", "5 hours on Saturday"), "5 days are not 5 hours");
    EXPECT(differ("3 people eat", "3 cartons of ice cream"), "3 people are not 3 cartons");
    EXPECT(differ("5 red cartons", "5 red cards") && differ("5 large boxes", "5 small cups"),
           "a shared word that describes is no shared unit");
    using redraw_qa::detail::constant_here;
    EXPECT(!constant_here({12, 1}, "$6 per month for two months", "12 dollars in two months"),
           "12 is 6 x 2, worked out in the head, not 12 months: no year is named");
    EXPECT(constant_here({12, 1}, "6 per month for two years", "2 years is 2*12 months"),
           "12 months a year where months and years are named");
    EXPECT(!constant_here({60, 1}, "$12 an hour, 50 minutes", "she earned 0.2 x 50"),
           "60 counts only where its own line names minutes or hours");
    EXPECT(constant_here({60, 1}, "$12 an hour, 50 minutes", "12/60 per minute") && constant_here({2, 1}, "", "") &&
               constant_here({1, 100}, "20%", "") && !constant_here({1, 100}, "20 apples", ""),
           "60 a minute where the line says so; 1, 2, 100 anywhere, 0.01 with a percentage");
    EXPECT(!differ("185 large rubber ducks", "185 large size ducks") &&
               !differ("12 medium ones", "12 medium ceiling lights") && !differ("5 chairs", "5 red chairs") &&
               !differ("6 ft", "6 feet") && !differ("180 guests", "180 people") &&
               !differ("120 pages", "120-page book") && !differ("5 more", "5 hours") && !differ("5", "5 hours") &&
               !differ("12 slices", "12"),
           "the same unit in other words, or nothing to compare");
    struct Cap {
        const char* text;
        int64_t most;
    };
    for (const Cap& c :
         {Cap{"5 days a week", 7}, Cap{"5 day work week", 7}, Cap{"5-day week", 7}, Cap{"8 hours a day", 24},
          Cap{"8-hour day", 24}, Cap{"8 hour days", 24}, Cap{"8 hours every weekday", 24},
          Cap{"3 months of the year", 12}, Cap{"20 minutes an hour", 60}, Cap{"5 days", 0}, Cap{"8 hours on Monday", 0},
          Cap{"5 dayweek", 0}}) {
        const std::string text = c.text;
        EXPECT(unit_cap(text, text.find_first_not_of("0123456789")) == c.most, "'%s': at most %lld, not %lld", c.text,
               (long long)c.most, (long long)unit_cap(text, text.find_first_not_of("0123456789")));
    }
    for (uint64_t seed = 1; seed <= 200; ++seed) {
        Rng rng(seed);
        const auto v = draw_value({5, 1}, 0, false, rng, 7);
        EXPECT(v && v->den == 1 && v->num >= 2 && v->num <= 7 && v->num != 5, "5 days a week: %s",
               v ? rational_text(*v).c_str() : "none");
    }
}

inline void test_redraw_qa() {
    test_redraw_qa_units();
    test_redraw_qa_text();
    test_redraw_qa_draws();
    test_redraw_qa_check();
    test_redraw_qa_fixtures();
}

}  // namespace
