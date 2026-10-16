// mathgen.h — math word problems built backward from a chosen answer, each
// one solved again by an independent checker before it is kept.
//
// This file holds what every topic shares (exclusions, the story builder,
// the note and answer checks) and the first topic; fractions, percentages,
// ratios and units are in mathgen_fractions.h.
//
// The first topic, equations (cxxgpt gen-math --topic equations), has seven
// kinds of story that GSM8K solves with one unknown, x:
//   puzzle       "Maya thinks of a number, multiplies it by 3, adds 7 and
//                 gets 31": undone step by step, from the end
//   ages_sum     "Maya is 5 years older than 3 times Leo's age; their ages
//                 add up to 41"
//   ages_future  "Maya is 4 times as old as Leo; in 6 years she will be twice
//                 as old"
//   mixture      two kinds of one thing: adult and child tickets (prices),
//                cows and chickens (legs), large and small packs (items)
//   consecutive  "three consecutive even numbers add up to 54"
//   rectangle    "the length is 3 meters more than twice the width; the
//                 perimeter is 54 meters"
//   shares       "Maya, Leo and Zoe share $120; Leo gets $5 more than Maya,
//                 Zoe gets 3 times as much as Leo"
//
// How a problem is made, and checked:
//   1. The unknown is drawn first (a whole number), then the story's other
//      numbers, then the total the story states. So the answer is right by
//      construction, and every number in the story is whole.
//   2. The worked solution writes the equation and solves it ("Taking 5
//      from both sides: 41-5 = <<41-5=36>>36, so 4x = 36."). Every number
//      step is a GSM8K calculator note, worked out again with calc.h's
//      exact evaluator; a wrong note stops the program.
//   3. equation_check, which shares no code with the generator, solves the
//      story again from its own numbers alone: it tries every whole number
//      the story allows (one ticket count after another, one age after
//      another) and must find exactly one that fits, and it must be the
//      answer written. (The puzzle is checked another way: see there.)
// A story that cannot be made with whole numbers is thrown away and drawn
// again; a number is never rounded or quietly changed.
//
// As in gen-word, each sentence family has six wordings: four for the train
// split and two held out for the test split, so the splits cannot share a
// question. Notes on or off and distractors on or off change no random
// draw: the same seed gives the same problems.
#pragma once

#include <algorithm>
#include <array>
#include <fstream>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "calc.h"
#include "leak_check.h"
#include "wordgen.h"  // Problem, Rng, word_detail::replace and its distractor sentences

// Questions a generated problem must not repeat (gen-math --exclude), such
// as GSM8K test and dev: the equations come from the same few classic
// stories, so a million of them will contain some test questions almost
// word for word ("Ruby is 6 times older than Sam. In 9 years, Ruby will be
// 3 times as old as Sam."). A generated problem is drawn again when
//   1. its question shares a 13-word run with an excluded question
//      (leak-check's rule), or
//   2. its question has the same numbers as an excluded question (in any
//      order) and the same answer: the same problem in other words, or by
//      chance the same numbers. Either way it is dropped; it costs a few
//      problems in a hundred thousand.
// The distractor's number does not count in rule 2: a distractor would
// otherwise hide a copy.
class Exclusions {
    std::vector<std::string> questions_;
    std::unique_ptr<leak_check::Index> index_;
    std::set<std::pair<std::vector<std::string>, std::string>> with_answer_;
    std::set<std::vector<std::string>> without_answer_;  // plain question lines have no answer
    static std::vector<std::string> pieces(const std::string& text, char separator) {
        std::vector<std::string> out;
        size_t start = 0;
        for (size_t at; (at = text.find(separator, start)) != std::string::npos; start = at + 1)
            out.push_back(text.substr(start, at - start));
        out.push_back(text.substr(start));
        return out;
    }
public:
    size_t by_run = 0, by_numbers = 0;  // problems dropped by rule 1 and rule 2

    // The numbers written in a text, as sorted strings ("1,000" is 1000).
    static std::vector<std::string> numbers(const std::string& text) {
        std::vector<std::string> found;
        for (size_t at = 0; at < text.size(); ++at) {
            if (text[at] < '0' || text[at] > '9') continue;
            std::string n;
            for (; at < text.size(); ++at) {
                const char c = text[at];
                const bool inside = at + 1 < text.size() && text[at + 1] >= '0' && text[at + 1] <= '9';
                if (c >= '0' && c <= '9') n += c;
                else if (c == '.' && inside) n += c;
                else if (c != ',' || !inside) break;
            }
            found.push_back(n);
        }
        std::sort(found.begin(), found.end());
        return found;
    }

    // Problems files (five tab-separated fields, as prep-qa writes them) or
    // one plain question per line; several files separated by commas.
    explicit Exclusions(const std::string& paths) {
        for (const std::string& path : pieces(paths, ',')) {
            std::ifstream in(path, std::ios::binary);
            CHECK(in, "cannot open %s", path.c_str());
            for (std::string line; std::getline(in, line);) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.empty()) continue;
                const std::vector<std::string> fields = pieces(line, '\t');
                const bool problem = fields.size() == 5;
                const std::string question = problem ? unescape_field(fields[2]) : line;
                questions_.push_back(question);
                auto key = numbers(question);
                if (key.size() < 2) continue;  // one number alone says too little
                if (problem) with_answer_.insert({key, strip_digit_commas(trim(unescape_field(fields[4])))});
                else without_answer_.insert(key);
            }
        }
        index_ = std::make_unique<leak_check::Index>(questions_);
    }

    size_t size() const { return questions_.size(); }

    // True if the problem repeats an excluded question; counts which rule.
    bool excludes(const Problem& p, int64_t distractor_number) {
        std::istringstream text(p.question);
        const auto hits = index_->scan(text);
        if (std::find(hits.begin(), hits.end(), true) != hits.end()) { ++by_run; return true; }
        auto key = numbers(p.question);
        if (distractor_number) {
            const auto at = std::find(key.begin(), key.end(), std::to_string(distractor_number));
            if (at != key.end()) key.erase(at);
        }
        if (with_answer_.count({key, p.answer}) || without_answer_.count(key)) { ++by_numbers; return true; }
        return false;
    }
};

struct MathOptions {
    bool distractors = true, notes = true, test = false;
    Exclusions* exclude = nullptr;  // questions not to repeat, or none
};

// What every topic's problem records: the text, and the story's own numbers,
// which the topic's checker solves again. Each topic adds its kinds.
struct MathRecord {
    Problem problem;
    // The story's numbers; what each one means is in the topic checker's table.
    std::vector<int64_t> inputs;
    std::string ops;  // the kinds of steps, in story order, where a kind has several
    int context = 0;  // which setting a kind's story uses (tickets, animals, ...)
    int asked = 0;    // which of the story's quantities the question asks for
    std::vector<int> patterns;  // sentence IDs (family * 6 + wording), for split tests
    int distractor = -1;
    int64_t distractor_number = 0;
    std::string name, other, third;
};

struct EquationProblem : MathRecord {
    enum Kind { puzzle, ages_sum, ages_future, mixture, consecutive, rectangle, shares, kind_count } kind = puzzle;
};

// What all topics share: building a story's sentences and solution, the
// distractor, and the final text.
namespace math_detail {

// A list of (placeholder, text) pairs to fill into a sentence pattern.
using Fills = std::vector<std::pair<std::string, std::string>>;

inline std::string str(int64_t v) { return std::to_string(v); }

// "1 year", "6 years": a number and its unit, singular for exactly one.
inline std::string count_of(const std::string& n, const std::string& one, const std::string& many) {
    return n + " " + (n == "1" ? one : many);
}
inline std::string years(int64_t n) { return count_of(str(n), "year", "years"); }

// Everything one story attempt shares: the random source, the people, the
// sentences and question being built, and the worked solution.
struct Story {
    Rng& rng;
    const MathOptions& options;
    MathRecord& out;
    const std::array<const char*, 6>* patterns;  // the topic's sentence families
    std::vector<std::string> sentences, lines;
    std::string question;
    std::vector<int64_t> results;  // one per calculator step

    Story(Rng& random, const MathOptions& opts, MathRecord& problem, const std::array<const char*, 6>* families)
        : rng(random), options(opts), out(problem), patterns(families) {
        static const std::array<const char*, 12> names = {
            "Alex", "Maya", "Noah", "Lina", "Sam", "Nora", "Leo", "Asha", "Ben", "Zoe", "Omar", "Iris"};
        out.name = names[rng.below(names.size())];
        do out.other = names[rng.below(names.size())]; while (out.other == out.name);
        do out.third = names[rng.below(names.size())]; while (out.third == out.name || out.third == out.other);
    }

    int64_t draw(int64_t lo, int64_t hi) { return lo + int64_t(rng.below(uint64_t(hi - lo + 1))); }

    // Fills one of the split's wordings of `family`.
    void sentence(int family, const Fills& fills) {
        const int variant = options.test ? 4 + int(rng.below(2)) : int(rng.below(4));
        sentences.push_back(fill(patterns[family][variant], fills));
        out.patterns.push_back(family * 6 + variant);
    }

    std::string fill(std::string s, const Fills& fills) const {
        for (const auto& [key, value] : fills) word_detail::replace(s, key, value);
        word_detail::replace(s, "{name}", out.name);
        word_detail::replace(s, "{other}", out.other);
        word_detail::replace(s, "{third}", out.third);
        return s;
    }

    // One number step of the solution: "41-5 = <<41-5=36>>36". Notes are
    // always written here; they are checked, then removed when --notes off.
    std::string calc(const std::string& expression, const std::string& value) {
        results.push_back(0);
        return expression + " = <<" + expression + "=" + value + ">>" + value;
    }
    std::string calc(const std::string& expression, int64_t value) { return calc(expression, str(value)); }

    void line(const std::string& text) { lines.push_back(text); }
};

// Where a distractor may go among the story's sentences.
enum class Place {
    anywhere,
    ends_only,  // first or last: the sentences say "it" for one another's numbers
    not_last,   // not just before the question, which says "that" about the story
};

// One irrelevant sentence, as in gen-word, from the split's own distractor
// wordings. Drawn even when disabled, so toggling distractors changes no
// other draw. Its number is never one the story or solution uses.
inline void add_distractor(Story& s, const std::string& answer, Place place) {
    MathRecord& out = s.out;
    const int extra = int(s.rng.below(12)) + (s.options.test ? 12 : 0);
    // Every whole number in the story, the solution and the answer, down to an "x + 4".
    std::set<int64_t> used(out.inputs.begin(), out.inputs.end());
    std::string texts = answer;
    for (const auto& text : s.lines) texts += " " + text;
    for (const auto& text : s.sentences) texts += " " + text;
    for (size_t at = 0; at < texts.size(); ++at) {
        if (texts[at] < '0' || texts[at] > '9') continue;
        int64_t v = 0;
        while (at < texts.size() && texts[at] >= '0' && texts[at] <= '9') v = v * 10 + (texts[at++] - '0');
        used.insert(v);
    }
    int64_t irrelevant = s.draw(2, 30);
    while (used.count(irrelevant)) ++irrelevant;
    size_t position = s.rng.below(s.sentences.size() + 1);
    if (place == Place::ends_only && position != 0) position = s.sentences.size();
    if (place == Place::not_last && position == s.sentences.size()) position = 0;
    if (!s.options.distractors) return;
    std::string text = word_detail::distractors[size_t(extra)];
    out.distractor = extra;
    if (text.find("{a}") != std::string::npos) out.distractor_number = irrelevant;
    word_detail::replace(text, "{a}", str(irrelevant));
    s.sentences.insert(s.sentences.begin() + std::ptrdiff_t(position), text);
}

// The finished problem's text: question, solution, answer and difficulty
// (the number of calculator steps).
inline void finish(Story& s, const std::string& answer, const std::string& task) {
    Problem& p = s.out.problem;
    for (const auto& sentence : s.sentences) p.question += sentence + ' ';
    p.question += s.question;
    for (const auto& text : s.lines) p.steps += (p.steps.empty() ? "" : "\n") + text;
    p.task = task;
    p.difficulty = int(s.results.size());
    p.answer = answer;
}

// Removes every calculator note: "41-5 = <<41-5=36>>36" -> "41-5 = 36".
inline std::string without_notes(std::string steps) {
    for (size_t at; (at = steps.find("<<")) != std::string::npos;) {
        const size_t end = steps.find(">>", at);
        CHECK(end != std::string::npos, "unclosed calculator note");
        steps.erase(at, end + 2 - at);
    }
    return steps;
}

}  // namespace math_detail

// The checks every topic's checker makes once it has its own answer.
namespace math_check {

// Every calculator note "<<e=v>>" must be written "e = <<e=v>>v", and calc.h
// must find e worth exactly v. Returns what is wrong (empty if nothing).
inline std::string check_notes(const std::string& steps) {
    for (size_t at = 0; (at = steps.find("<<", at)) != std::string::npos;) {
        const size_t equals = steps.find('=', at), end = steps.find(">>", at);
        if (equals == std::string::npos || end == std::string::npos || equals > end) return "a malformed note";
        const std::string expression = steps.substr(at + 2, equals - at - 2);
        const std::string value = steps.substr(equals + 1, end - equals - 1);
        const auto exact = calc::evaluate(expression);
        if (!exact || *exact != value) return "the note " + expression + "=" + value + " is wrong";
        if (at < expression.size() + 3 ||
            steps.compare(at - expression.size() - 3, expression.size() + 3, expression + " = ") != 0)
            return "the note " + expression + " is not written after its own expression";
        if (steps.compare(end + 2, value.size(), value) != 0)
            return "the note " + expression + " is not followed by its value";
        at = end + 2;
    }
    return "";
}

// The problem's answer must be `expected` (the checker's own), every note
// right, and the solution's last line must state the answer: as a number
// of its own ("is 12." or ">>12 dollars."), not part of another number.
inline std::string answer_and_notes(const MathRecord& p, const std::string& expected) {
    if (p.problem.answer != expected)
        return "the answer is " + p.problem.answer + ", but the story's numbers give " + expected;
    const std::string notes = check_notes(p.problem.steps);
    if (!notes.empty()) return notes;
    const size_t last = p.problem.steps.rfind('\n');
    const std::string line = p.problem.steps.substr(last == std::string::npos ? 0 : last + 1);
    const std::string& a = p.problem.answer;
    bool stated = false;
    for (size_t at = 0; (at = line.find(a, at)) != std::string::npos; ++at) {
        const size_t end = at + a.size();
        stated = stated || (at > 0 && (line[at - 1] == ' ' || line[at - 1] == '>') && end < line.size() &&
                            (line[end] == ' ' || (line[end] == '.' && end + 1 == line.size())));
    }
    if (!stated) return "the last solution line does not state the answer";
    return "";
}

}  // namespace math_check

// `count` different problems from `make_one(rng)`, which returns a record.
template <class MakeOne>
std::vector<Problem> make_unique_problems(uint64_t seed, int64_t count, MakeOne make_one) {
    CHECK(count >= 0, "problem count must be nonnegative");
    Rng rng(seed);
    std::vector<Problem> problems;
    std::set<std::string> questions;
    int duplicates = 0;
    while (problems.size() < uint64_t(count)) {
        Problem p = make_one(rng).problem;
        if (!questions.insert(p.question).second) {
            CHECK(++duplicates < 10000, "too many duplicate problems; request fewer problems");
            continue;
        }
        duplicates = 0;
        problems.push_back(std::move(p));
    }
    return problems;
}

namespace equation_detail {

using math_detail::Fills;
using math_detail::Story;
using math_detail::count_of;
using math_detail::str;
using math_detail::years;
using math_detail::without_notes;

enum Family { think, add, subtract, multiply, divide, result, ages_sum_text, ages_future_text, tickets,
              legs, packs, consecutive_text, rectangle_text, shares_text, family_count };

// Six wordings per family with the same meaning: 0-3 train, 4-5 test.
// The puzzle's sentences say "it" for the current number, so a distractor
// never goes between them (see add_distractor).
inline constexpr std::array<std::array<const char*, 6>, family_count> patterns = {{
    {{"{name} thinks of a number.",
      "{name} picks a secret number.",
      "{name} has a number in mind.",
      "{name} writes down a mystery number.",
      "{name} chooses a hidden number.",
      "A number is kept secret by {name}."}},
    {{"{name} adds {a} to it.",
      "Then {name} adds {a}.",
      "Next, {name} increases the number by {a}.",
      "{name} makes the number {a} bigger.",
      "After that, {a} is added to the number.",
      "{name} raises the current value by {a}."}},
    {{"{name} subtracts {a} from it.",
      "Then {name} takes away {a}.",
      "Next, {name} decreases the number by {a}.",
      "{name} removes {a} from the current value.",
      "After that, {a} is subtracted from the number.",
      "{name} lowers the current value by {a}."}},
    {{"{name} multiplies it by {a}.",
      "Then {name} multiplies the number by {a}.",
      "Next, {name} makes the number {a} times as large.",
      "{name} multiplies the current value by {a}.",
      "After that, the number is multiplied by {a}.",
      "{name} scales the current value up by a factor of {a}."}},
    {{"{name} divides it by {a}.",
      "Then {name} divides the number by {a}.",
      "Next, {name} divides that number by {a}.",
      "{name} divides the current value by {a}.",
      "After that, the number is divided by {a}.",
      "{name} takes the current value and divides it by {a}."}},
    {{"The result is {a}.",
      "{name} ends up with {a}.",
      "The final number is {a}.",
      "Now the number is {a}.",
      "In the end, {name} gets {a}.",
      "The value {name} finishes with is {a}."}},
    {{"{name} is {relation}. The sum of their ages is {s}.",
      "{name} is {relation}, and together they are {s} years old.",
      "{other} and {name} have ages that add up to {s} years, and {name} is {relation}.",
      "Right now {name} is {relation}. Their two ages total {s}.",
      "Added together, the ages of {name} and {other} make {s} years; {name} is {relation}.",
      "{name}, who is {relation}, has a combined age with {other} of {s} years."}},
    {{"{name} is {a} times as old as {other}. In {b} years, {name} will be {c} times as old as {other}.",
      "Today {name} is {a} times as old as {other}, and {b} years from now {name} will be {c} times as old.",
      "{name}'s age is {a} times {other}'s age. After {b} years, {name}'s age will be {c} times {other}'s age.",
      "At the moment, {name} is {a} times {other}'s age, and in {b} years' time {name} will be {c} times {other}'s age.",
      "{other} is younger: {name} is {a} times as old now, and once {b} years have passed {name} will be {c} times as old as {other}.",
      "The age of {name} is {a} times that of {other}; {b} years on, it will be {c} times that of {other}."}},
    {{"Adult tickets to a play cost ${a} and child tickets cost ${c}. {name} sells {n} tickets for a total of ${t}.",
      "At a school fair, {name} sells {n} tickets and collects ${t}. Each adult ticket costs ${a} and each child ticket costs ${c}.",
      "{name} sold {n} tickets to a concert, some for adults at ${a} each and the rest for children at ${c} each, and took in ${t}.",
      "A museum charges ${a} for an adult ticket and ${c} for a child ticket. {name} sells {n} tickets in one morning and earns ${t}.",
      "Selling {n} tickets brought {name} ${t}; adults paid ${a} per ticket and children paid ${c} per ticket.",
      "For a puppet show, {name} sells adult tickets at ${a} and child tickets at ${c}, {n} tickets in all, for ${t} altogether."}},
    {{"{name}'s farm has only {big} and {small}. There are {n} animals with {t} legs in all.",
      "On a farm there are {n} animals, all of them {big} or {small}. Together they have {t} legs.",
      "{name} counts {n} heads and {t} legs among the {big} and {small} in a field.",
      "A pen holds some {big} and some {small}: {n} animals and {t} legs altogether.",
      "Looking over the {big} and {small} in a barn, {name} sees {t} legs belonging to {n} animals.",
      "There are {t} legs and {n} heads in a yard where only {big} and {small} live."}},
    {{"{name} buys {n} packs of {object}. A large pack has {a} {object} and a small pack has {c}. In all there are {t} {object}.",
      "Large packs hold {a} {object} and small packs hold {c}. {name} has {n} packs with {t} {object} altogether.",
      "{name} sorts {t} {object} into {n} packs, with {a} in every large pack and {c} in every small pack.",
      "A shop sells {object} in large packs of {a} and small packs of {c}. {name} buys {n} packs and gets {t} {object}.",
      "There are {t} {object} in the {n} packs {name} bought; each large pack held {a} and each small pack held {c}.",
      "{name} ends up with {t} {object} from {n} packs, where large packs contain {a} {object} and small ones contain {c}."}},
    {{"{Count} consecutive {numbers} add up to {s}.",
      "The sum of {count} consecutive {numbers} is {s}.",
      "{name} writes down {count} consecutive {numbers}, and their total is {s}.",
      "Adding {count} consecutive {numbers} gives {s}.",
      "When {name} adds up {count} {numbers} that follow one after another, the total comes to {s}.",
      "A list of {count} consecutive {numbers} has a sum of {s}."}},
    {{"{name}'s rectangular {thing} has a length that is {relation}. Its perimeter is {p} {unit}.",
      "The length of a rectangular {thing} is {relation}, and the perimeter is {p} {unit}.",
      "A rectangular {thing} has a perimeter of {p} {unit}, and its length is {relation}.",
      "{name} measures a rectangular {thing}: the perimeter is {p} {unit} and the length is {relation}.",
      "Going once around a rectangular {thing} covers {p} {unit}; the {thing}'s length is {relation}.",
      "{name} puts a border of {p} {unit} all the way around a rectangular {thing} whose length is {relation}."}},
    {{"{name}, {other} and {third} share ${t}. {other} gets ${d} more than {name}, and {third} gets {k} times as much as {other}.",
      "{name}, {other} and {third} split ${t} between them. {other}'s share is ${d} more than {name}'s, and {third}'s share is {k} times {other}'s.",
      "A prize of ${t} goes to {name}, {other} and {third}. {other} receives ${d} more than {name}, while {third} receives {k} times what {other} receives.",
      "{name}, {other} and {third} earn ${t} together. {other} earns ${d} more than {name}, and {third} earns {k} times as much as {other}.",
      "Out of ${t} shared by three friends, {other} gets ${d} more than {name} and {third} gets {k} times {other}'s amount; {name} gets the rest.",
      "{third} ends up with {k} times as much money as {other}, who has ${d} more than {name}; the three of them have ${t} in total."}},
}};

// "3x + 5", "x - 4", "2x", "x", "7": a x + b as a person would write it.
inline std::string linear(int64_t a, int64_t b) {
    std::string s;
    if (a == 1) s = "x";
    else if (a != 0) s = std::to_string(a) + "x";
    if (s.empty()) return std::to_string(b);
    if (b > 0) s += " + " + std::to_string(b);
    if (b < 0) s += " - " + std::to_string(-b);
    return s;
}

// "3*9+5", "9+5", "3*9", "3*9-4": the value of a x + b at x, as a calculation.
inline std::string at(int64_t a, int64_t x, int64_t b) {
    std::string s = a == 1 ? str(x) : str(a) + "*" + str(x);
    if (b > 0) s += "+" + str(b);
    if (b < 0) s += "-" + str(-b);
    return s;
}

// Solves a x + b = c x + d for x (a > c >= 0), one line per step, and
// returns x. The caller has chosen the numbers so that x is whole.
inline int64_t solve_linear(Story& s, int64_t a, int64_t b, int64_t c, int64_t d) {
    CHECK(a > c && c >= 0, "solve_linear needs a > c >= 0");
    if (c > 0) {
        s.line("Taking " + linear(c, 0) + " from both sides: " + s.calc(str(a) + "-" + str(c), a - c) + ", so " +
               linear(a - c, b) + " = " + str(d) + ".");
        a -= c;
    }
    if (b > 0) {
        s.line("Taking " + str(b) + " from both sides: " + s.calc(str(d) + "-" + str(b), d - b) + ", so " +
               linear(a, 0) + " = " + str(d - b) + ".");
        d -= b;
    } else if (b < 0) {
        s.line("Adding " + str(-b) + " to both sides: " + s.calc(str(d) + "+" + str(-b), d - b) + ", so " +
               linear(a, 0) + " = " + str(d - b) + ".");
        d -= b;
    }
    CHECK(d % a == 0 && d / a > 0, "solve_linear was given numbers without a whole positive solution");
    if (a > 1) s.line("Dividing both sides by " + str(a) + ": " + s.calc(str(d) + "/" + str(a), d / a) + ", so x = " +
                      str(d / a) + ".");
    return d / a;
}

// "Maya thinks of a number, multiplies it by 3, adds 7 and gets 31."
// The solution undoes the operations from the last one back.
inline bool puzzle_story(Story& s, int64_t& answer) {
    const int count = 2 + int(s.rng.below(3));
    const int64_t x = s.draw(2, 50);
    int64_t v = x;
    std::vector<int64_t> values = {x};  // the number after each operation
    s.sentence(think, {});
    for (int i = 0; i < count; ++i) {
        const char op = "+-*/"[s.rng.below(4)];
        int64_t a = 0;
        if (op == '+') a = s.draw(1, 30);
        if (op == '-') a = s.draw(1, 30);
        if (op == '*') a = s.draw(2, 9);
        if (op == '/') {
            // Draw among the divisors 2..9 of the number, so the result stays whole.
            std::vector<int64_t> divisors;
            for (int64_t d = 2; d <= 9; ++d) if (v % d == 0) divisors.push_back(d);
            if (divisors.empty()) return false;
            a = divisors[s.rng.below(divisors.size())];
        }
        if (op == '+') v += a;
        if (op == '-') v -= a;
        if (op == '*') v *= a;
        if (op == '/') v /= a;
        if (v < 1 || v > 10000) return false;
        s.out.ops += op;
        s.out.inputs.push_back(a);
        values.push_back(v);
        s.sentence(op == '+' ? add : op == '-' ? subtract : op == '*' ? multiply : divide, {{"{a}", str(a)}});
    }
    s.out.inputs.push_back(v);
    s.sentence(result, {{"{a}", str(v)}});
    s.question = "What number did " + s.out.name + " start with?";
    s.line("Undo the steps one at a time, starting from the end, where the number is " + str(v) + ".");
    for (int i = count - 1; i >= 0; --i) {
        const int64_t a = s.out.inputs[size_t(i)], after = values[size_t(i) + 1], before = values[size_t(i)];
        const char op = s.out.ops[size_t(i)];
        std::string text;
        if (op == '+') text = "Before adding " + str(a) + ", the number was " + s.calc(str(after) + "-" + str(a), before);
        if (op == '-') text = "Before taking away " + str(a) + ", the number was " + s.calc(str(after) + "+" + str(a), before);
        if (op == '*') text = "Before multiplying by " + str(a) + ", the number was " + s.calc(str(after) + "/" + str(a), before);
        if (op == '/') text = "Before dividing by " + str(a) + ", the number was " + s.calc(str(after) + "*" + str(a), before);
        s.line(text + ".");
    }
    s.line("So " + s.out.name + " started with " + str(x) + ".");
    answer = x;
    return true;
}

// "Maya is 5 years older than 3 times Leo's age; their ages add up to 41."
// x is Leo's age ({other}); Maya ({name}) is k x + d.
inline bool ages_sum_story(Story& s, int64_t& answer) {
    const int64_t l = s.draw(2, 40), k = s.draw(1, 4), d = s.draw(-10, 15);
    if (k == 1 && d <= 0) return false;
    const int64_t m = k * l + d, total = m + l;
    if (m < 1 || m > 100) return false;
    const std::string& other = s.out.other;
    std::string relation;
    if (k == 1) relation = years(d) + " older than " + other;
    else if (d == 0) relation = str(k) + " times as old as " + other;
    else if (d > 0) relation = years(d) + " older than " + str(k) + " times " + other + "'s age";
    else relation = years(-d) + " younger than " + str(k) + " times " + other + "'s age";
    s.out.inputs = {k, d, total};
    s.out.asked = int(s.rng.below(2));
    s.sentence(ages_sum_text, {{"{relation}", relation}, {"{s}", str(total)}});
    s.question = "How old is " + (s.out.asked ? s.out.name : other) + "?";
    s.line("Let x be " + other + "'s age. Then " + s.out.name + " is " + linear(k, d) + " years old.");
    s.line("Their ages add up to x + (" + linear(k, d) + ") = " + str(total) + ".");
    s.line("Counting the x's: " + s.calc("1+" + str(k), k + 1) + ", so " + linear(k + 1, d) + " = " + str(total) + ".");
    const int64_t x = solve_linear(s, k + 1, d, 0, total);
    if (s.out.asked == 0) {
        s.line("So " + other + " is " + years(x) + " old.");
        answer = x;
    } else {
        s.line("So " + s.out.name + " is " + s.calc(at(k, x, d), m) + (m == 1 ? " year old." : " years old."));
        answer = m;
    }
    return true;
}

// "Maya is 4 times as old as Leo; in 6 years she will be twice as old."
// x is Leo's age now; Maya is k1 x now, and k1 x + n = k2 (x + n).
inline bool ages_future_story(Story& s, int64_t& answer) {
    const int64_t l = s.draw(1, 40), k1 = s.draw(2, 10), n = s.draw(2, 60);
    if ((k1 * l + n) % (l + n) != 0) return false;
    const int64_t k2 = (k1 * l + n) / (l + n), m = k1 * l;
    if (k2 < 2 || m > 100) return false;
    const std::string& other = s.out.other;
    s.out.inputs = {k1, n, k2};
    s.out.asked = int(s.rng.below(2));
    s.sentence(ages_future_text, {{"{a}", str(k1)}, {"{b}", str(n)}, {"{c}", str(k2)}});
    s.question = "How old is " + (s.out.asked ? s.out.name : other) + " now?";
    s.line("Let x be " + other + "'s age now. Then " + s.out.name + " is " + linear(k1, 0) + " years old.");
    s.line("In " + str(n) + " years, " + other + " will be " + linear(1, n) + " and " + s.out.name + " will be " +
           linear(k1, n) + ", so " + linear(k1, n) + " = " + str(k2) + "(" + linear(1, n) + ").");
    s.line("Multiplying out: " + s.calc(str(k2) + "*" + str(n), k2 * n) + ", so " + linear(k1, n) + " = " +
           linear(k2, k2 * n) + ".");
    const int64_t x = solve_linear(s, k1, n, k2, k2 * n);
    if (s.out.asked == 0) {
        s.line("So " + other + " is " + years(x) + " old now.");
        answer = x;
    } else {
        s.line("So " + s.out.name + " is " + s.calc(at(k1, x, 0), m) + " years old now.");
        answer = m;
    }
    return true;
}

// Two kinds of one thing: n of them, worth t together; x counts the first kind.
inline bool mixture_story(Story& s, int64_t& answer) {
    static const std::array<std::array<const char*, 2>, 4> animals = {
        {{"cows", "chickens"}, {"goats", "ducks"}, {"sheep", "geese"}, {"pigs", "hens"}}};
    static const std::array<const char*, 6> objects = {"pencils", "stickers", "markers", "crayons", "buttons",
                                                      "batteries"};
    const int context = s.out.context = int(s.rng.below(3));
    int64_t a = 4, c = 2;  // legs of the larger and the smaller animal
    if (context == 0) { a = s.draw(4, 25); c = s.draw(2, a - 1); }
    if (context == 2) { a = s.draw(5, 20); c = s.draw(2, a - 1); }
    const int64_t n = s.draw(5, 60), x = s.draw(1, n - 1), t = a * x + c * (n - x);
    const auto& pair = animals[s.rng.below(animals.size())];
    const std::string object = objects[s.rng.below(objects.size())];
    s.out.inputs = {a, c, n, t};
    s.out.asked = int(s.rng.below(2));
    std::string first = "adult tickets", second = "child tickets", counting = "In dollars, ";
    if (context == 0)
        s.sentence(tickets, {{"{a}", str(a)}, {"{c}", str(c)}, {"{n}", str(n)}, {"{t}", str(t)}});
    if (context == 1) {
        first = pair[0];
        second = pair[1];
        counting = "Counting legs, ";
        s.sentence(legs, {{"{big}", first}, {"{small}", second}, {"{n}", str(n)}, {"{t}", str(t)}});
    }
    if (context == 2) {
        first = "large packs";
        second = "small packs";
        counting = "Counting " + object + ", ";
        s.sentence(packs, {{"{object}", object}, {"{a}", str(a)}, {"{c}", str(c)}, {"{n}", str(n)}, {"{t}", str(t)}});
    }
    s.question = "How many " + (s.out.asked ? second : first) + (context == 0 ? " were sold?" : " are there?");
    s.line("Let x be the number of " + first + ", so there are " + str(n) + " - x " + second + ".");
    if (context == 1) s.line("Each of the " + first + " has 4 legs and each of the " + second + " has 2 legs.");
    s.line(counting + str(a) + "x + " + str(c) + "(" + str(n) + " - x) = " + str(t) + ".");
    s.line("Multiplying out: " + s.calc(str(c) + "*" + str(n), c * n) + ", so " + str(a) + "x + " + str(c * n) +
           " - " + linear(c, 0) + " = " + str(t) + ".");
    s.line("Counting the x's: " + s.calc(str(a) + "-" + str(c), a - c) + ", so " + linear(a - c, c * n) + " = " +
           str(t) + ".");
    const int64_t solved = solve_linear(s, a - c, c * n, 0, t);
    if (s.out.asked == 0) {
        s.line("So the number of " + first + " is " + str(solved) + ".");
        answer = solved;
    } else {
        s.line("So the number of " + second + " is " + s.calc(str(n) + "-" + str(solved), n - solved) + ".");
        answer = n - solved;
    }
    return true;
}

// "Three consecutive even numbers add up to 54." x is the smallest.
inline bool consecutive_story(Story& s, int64_t& answer) {
    static const std::array<const char*, 6> counts = {"two", "three", "four", "five", "six", "seven"};
    static const std::array<const char*, 3> numbers = {"whole numbers", "even numbers", "odd numbers"};
    const int64_t k = s.draw(2, 7);
    const int kind = int(s.rng.below(3));
    const int64_t step = kind == 0 ? 1 : 2;
    // A whole number from 1, an even number from 2, an odd number from 1.
    const int64_t x = kind == 0 ? s.draw(1, 300) : kind == 1 ? 2 * s.draw(1, 150) : 2 * s.draw(0, 149) + 1;
    const int64_t offsets = step * k * (k - 1) / 2, total = k * x + offsets;
    s.out.inputs = {k, step, kind == 2 ? 1 : 0, total};
    s.out.asked = int(s.rng.below(2));
    std::string count = counts[size_t(k - 2)], capital = count;
    capital[0] = char(capital[0] - 'a' + 'A');
    s.sentence(consecutive_text,
               {{"{Count}", capital}, {"{count}", count}, {"{numbers}", numbers[size_t(kind)]}, {"{s}", str(total)}});
    s.question = std::string("What is the ") + (s.out.asked ? "largest" : "smallest") + " of these numbers?";
    std::string list = "x", sum;
    for (int64_t i = 1; i < k; ++i) {
        list += (i + 1 == k ? ", and " : ", ") + linear(1, i * step);
        sum += (i > 1 ? "+" : "") + str(i * step);
    }
    if (k == 2) list = "x and " + linear(1, step);
    s.line("Let x be the smallest number. The numbers are " + list + ".");
    if (k == 2) s.line("Their sum is " + linear(k, offsets) + " = " + str(total) + ".");
    else s.line("Adding the extra amounts: " + s.calc(sum, offsets) + ", so the sum is " + linear(k, offsets) + " = " +
                str(total) + ".");
    const int64_t solved = solve_linear(s, k, offsets, 0, total);
    if (s.out.asked == 0) {
        s.line("So the smallest number is " + str(solved) + ".");
        answer = solved;
    } else {
        answer = solved + (k - 1) * step;
        s.line("So the largest number is " + s.calc(str(solved) + "+" + str((k - 1) * step), answer) + ".");
    }
    return true;
}

// "The length is 3 meters more than twice the width; the perimeter is 54
// meters." x is the width; the length is k x + d.
inline bool rectangle_story(Story& s, int64_t& answer) {
    // Sizes a real one could have, in meters: the width's range, the most
    // times the width the length may be, and the most meters added on top.
    // In feet, the ranges are three times as large.
    struct Thing { const char* name; int64_t width_min, width_max, times_max, add_max; };
    static const std::array<Thing, 5> things = {{{"garden", 3, 30, 3, 15},
                                                 {"room", 2, 8, 2, 4},
                                                 {"field", 20, 150, 3, 50},
                                                 {"rug", 1, 3, 2, 2},
                                                 {"pool", 3, 15, 3, 10}}};
    const Thing& thing_size = things[s.rng.below(things.size())];
    const bool feet = s.rng.below(2);
    const int64_t scale = feet ? 3 : 1;
    const std::string thing = thing_size.name, unit = feet ? "feet" : "meters";
    const int64_t w = s.draw(thing_size.width_min * scale, thing_size.width_max * scale);
    const int64_t k = s.draw(1, thing_size.times_max), d = s.draw(0, thing_size.add_max * scale);
    if (k == 1 && d == 0) return false;
    const int64_t length = k * w + d, perimeter = 2 * (w + length), half = perimeter / 2;
    const auto measure = [&](int64_t n) { return count_of(str(n), feet ? "foot" : "meter", unit); };
    std::string relation;
    if (k == 1) relation = measure(d) + " longer than its width";
    else if (d == 0) relation = str(k) + " times its width";
    else relation = measure(d) + " more than " + str(k) + " times its width";
    s.out.inputs = {k, d, perimeter};
    s.out.asked = int(s.rng.below(3));
    s.sentence(rectangle_text, {{"{thing}", thing}, {"{relation}", relation}, {"{p}", str(perimeter)},
                                 {"{unit}", unit}});
    const std::array<std::string, 3> asked = {"width of the " + thing + " in " + unit,
                                              "length of the " + thing + " in " + unit,
                                              "area of the " + thing + " in square " + unit};
    s.question = "What is the " + asked[size_t(s.out.asked)] + "?";
    s.line("Let x be the width in " + unit + ". Then the length is " + linear(k, d) + ".");
    s.line("Half the perimeter is one width plus one length: " + s.calc(str(perimeter) + "/2", half) + ", so x + (" +
           linear(k, d) + ") = " + str(half) + ".");
    s.line("Counting the x's: " + s.calc("1+" + str(k), k + 1) + ", so " + linear(k + 1, d) + " = " + str(half) + ".");
    const int64_t x = solve_linear(s, k + 1, d, 0, half);
    if (s.out.asked == 0) {
        s.line("So the width is " + measure(x) + ".");
        answer = x;
        return true;
    }
    s.line("The length is " + s.calc(at(k, x, d), length) + " " + unit + ".");
    answer = length;
    if (s.out.asked == 2) {
        answer = x * length;
        s.line("So the area is " + s.calc(str(x) + "*" + str(length), answer) + " square " + unit + ".");
    }
    return true;
}

// "Maya, Leo and Zoe share $120; Leo gets $5 more than Maya, Zoe gets 3
// times as much as Leo." x is Maya's share ({name}).
inline bool shares_story(Story& s, int64_t& answer) {
    const int64_t x = s.draw(1, 200), d = s.draw(1, 80), k = s.draw(2, 5);
    const int64_t second = x + d, third = k * second, total = x + second + third;
    const std::string &name = s.out.name, &other = s.out.other, &last = s.out.third;
    s.out.inputs = {d, k, total};
    s.out.asked = int(s.rng.below(3));
    s.sentence(shares_text, {{"{t}", str(total)}, {"{d}", str(d)}, {"{k}", str(k)}});
    s.question = "How many dollars does " + (s.out.asked == 0 ? name : s.out.asked == 1 ? other : last) + " get?";
    s.line("Let x be " + name + "'s share in dollars. Then " + other + " gets " + linear(1, d) + ".");
    s.line(last + " gets " + str(k) + "(" + linear(1, d) + "), and " + s.calc(str(k) + "*" + str(d), k * d) + ", so " +
           last + " gets " + linear(k, k * d) + ".");
    s.line("Together: x + (" + linear(1, d) + ") + (" + linear(k, k * d) + ") = " + str(total) + ".");
    s.line("Counting the x's: " + s.calc("1+1+" + str(k), k + 2) + ", and the rest: " +
           s.calc(str(d) + "+" + str(k * d), d + k * d) + ", so " + linear(k + 2, d + k * d) + " = " + str(total) + ".");
    const int64_t solved = solve_linear(s, k + 2, d + k * d, 0, total);
    if (s.out.asked == 0) {
        s.line("So " + name + " gets " + str(solved) + " dollars.");
        answer = solved;
        return true;
    }
    s.line(other + " gets " + s.calc(str(solved) + "+" + str(d), second) + " dollars.");
    answer = second;
    if (s.out.asked == 2) {
        s.line("So " + last + " gets " + s.calc(str(k) + "*" + str(second), third) + " dollars.");
        answer = third;
    }
    return true;
}

// One attempt at a story of the given kind; false means draw again.
inline bool candidate(Rng& rng, const MathOptions& options, EquationProblem& out, int64_t& answer) {
    Story s(rng, options, out, patterns.data());
    bool made = false;
    switch (out.kind) {
        case EquationProblem::puzzle: made = puzzle_story(s, answer); break;
        case EquationProblem::ages_sum: made = ages_sum_story(s, answer); break;
        case EquationProblem::ages_future: made = ages_future_story(s, answer); break;
        case EquationProblem::mixture: made = mixture_story(s, answer); break;
        case EquationProblem::consecutive: made = consecutive_story(s, answer); break;
        case EquationProblem::rectangle: made = rectangle_story(s, answer); break;
        case EquationProblem::shares: made = shares_story(s, answer); break;
        case EquationProblem::kind_count: break;
    }
    if (!made) return false;
    math_detail::add_distractor(s, str(answer), out.kind == EquationProblem::puzzle
                                                     ? math_detail::Place::ends_only
                                                     : math_detail::Place::anywhere);
    math_detail::finish(s, str(answer), "equations");
    return true;
}

}  // namespace equation_detail

// The second solver. It reads only the story's numbers (inputs, ops) and
// what is asked, never the generator's solution or answer, and uses none of
// equation_detail's code. What inputs mean, per kind:
//
//   kind         inputs                              asked
//   puzzle       operand of each op..., final number the starting number
//   ages_sum     k, d, total: {name} = k*{other}+d   0 {other}, 1 {name}
//   ages_future  k1, n, k2: {name} = k1*{other} now,  0 {other}, 1 {name}
//                and {name}+n = k2*({other}+n)
//   mixture      a, c, n, t: n things, a each of the 0 first kind, 1 second
//                first kind, c of the second, t in all
//   consecutive  k, step, parity, total               0 smallest, 1 largest
//   rectangle    k, d, perimeter: length = k*width+d 0 width, 1 length, 2 area
//   shares       d, k, total: {other} = {name}+d,     0 {name}, 1 {other}, 2 {third}
//                {third} = k*{other}
namespace equation_check {

// Exact fractions for the puzzle, kept small: this check's own arithmetic.
struct Fraction {
    int64_t num = 0, den = 1;
};
inline Fraction reduce(int64_t num, int64_t den) {
    CHECK(den != 0, "zero denominator in the equation check");
    if (den < 0) { num = -num; den = -den; }
    const int64_t g = std::gcd(num, den);
    return {num / g, den / g};
}

// Every whole-number solution the story allows, by trying each candidate
// in turn. A story that is right has exactly one.
struct Search {
    int found = 0;
    int64_t answer = 0;
    void hit(int64_t value) { ++found; answer = value; }
};

// The answer the story's numbers give. Returns what is wrong with the
// story instead (empty if nothing): no whole-number solution, or several.
inline std::string solve(const EquationProblem& p, int64_t& answer) {
    const auto& in = p.inputs;
    Search search;
    switch (p.kind) {
        case EquationProblem::puzzle: {
            // The story applies f(x) = q x + r to the start: compose q and r
            // forward through the operations, then solve q x + r = final.
            // Each operation can be undone (no multiplying or dividing by
            // zero), so there is one start, which must be a whole number.
            if (in.size() != p.ops.size() + 1 || p.ops.empty()) return "puzzle record has the wrong size";
            Fraction q{1, 1}, r{0, 1};
            for (size_t i = 0; i < p.ops.size(); ++i) {
                const int64_t a = in[i];
                const char op = p.ops[i];
                if (a <= 0) return "a puzzle operand is not positive";
                if (op == '+') r = reduce(r.num + a * r.den, r.den);
                else if (op == '-') r = reduce(r.num - a * r.den, r.den);
                else if (op == '*') { q = reduce(q.num * a, q.den); r = reduce(r.num * a, r.den); }
                else if (op == '/') { q = reduce(q.num, q.den * a); r = reduce(r.num, r.den * a); }
                else return std::string("unknown puzzle operation ") + op;
            }
            // x = (final - r) / q
            const Fraction top = reduce(in.back() * r.den - r.num, r.den);
            const Fraction x = reduce(top.num * q.den, top.den * q.num);
            if (x.den != 1 || x.num <= 0) return "the puzzle's start is not a whole positive number";
            search.hit(x.num);
            break;
        }
        case EquationProblem::ages_sum:
            if (in.size() != 3) return "ages record has the wrong size";
            for (int64_t young = 0; young <= 200; ++young) {
                const int64_t old = in[0] * young + in[1];
                if (old >= 0 && old + young == in[2]) search.hit(p.asked ? old : young);
            }
            break;
        case EquationProblem::ages_future:
            if (in.size() != 3) return "ages record has the wrong size";
            for (int64_t young = 0; young <= 200; ++young) {
                const int64_t old = in[0] * young;
                if (old + in[1] == in[2] * (young + in[1])) search.hit(p.asked ? old : young);
            }
            break;
        case EquationProblem::mixture:
            if (in.size() != 4) return "mixture record has the wrong size";
            for (int64_t first = 0; first <= in[2]; ++first)
                if (in[0] * first + in[1] * (in[2] - first) == in[3]) search.hit(p.asked ? in[2] - first : first);
            break;
        case EquationProblem::consecutive:
            if (in.size() != 4 || in[0] < 2 || (in[1] != 1 && in[1] != 2)) return "consecutive record is wrong";
            for (int64_t smallest = 0; smallest <= in[3]; ++smallest) {
                if (in[1] == 2 && smallest % 2 != in[2]) continue;
                int64_t sum = 0;
                for (int64_t i = 0; i < in[0]; ++i) sum += smallest + i * in[1];
                if (sum == in[3]) search.hit(p.asked ? smallest + (in[0] - 1) * in[1] : smallest);
            }
            break;
        case EquationProblem::rectangle:
            if (in.size() != 3) return "rectangle record has the wrong size";
            for (int64_t width = 1; width <= in[2]; ++width) {
                const int64_t length = in[0] * width + in[1];
                if (2 * width + 2 * length == in[2])
                    search.hit(p.asked == 0 ? width : p.asked == 1 ? length : width * length);
            }
            break;
        case EquationProblem::shares:
            if (in.size() != 3) return "shares record has the wrong size";
            for (int64_t first = 0; first <= in[2]; ++first) {
                const int64_t second = first + in[0], third = in[1] * second;
                if (first + second + third == in[2])
                    search.hit(p.asked == 0 ? first : p.asked == 1 ? second : third);
            }
            break;
        case EquationProblem::kind_count: return "no such kind";
    }
    if (search.found != 1) return "the story has " + std::to_string(search.found) + " whole-number solutions, not one";
    answer = search.answer;
    return "";
}

// Everything the second solver checks on a finished problem (with notes):
// returns what is wrong, or an empty string.
inline std::string problems_with(const EquationProblem& p) {
    int64_t answer = 0;
    const std::string unsolved = solve(p, answer);
    if (!unsolved.empty()) return unsolved;
    return math_check::answer_and_notes(p, std::to_string(answer));
}

}  // namespace equation_check

// How often each kind is drawn, out of 19 (in Kind's order). Kinds with
// few different problems are drawn less, so each repeats less often in a
// large file: with realistic ages there are only about 600 different
// ages_future problems, against more than 100,000 puzzles or mixtures.
inline constexpr std::array<int, EquationProblem::kind_count> kind_weights = {4, 2, 1, 4, 2, 3, 3};

// One problem; every one is checked by equation_check before it is returned.
inline EquationProblem make_equation_problem(Rng& rng, const MathOptions& options = {}) {
    int pick = int(rng.below(19)), k = 0;
    while (pick >= kind_weights[size_t(k)]) pick -= kind_weights[size_t(k++)];
    const auto kind = EquationProblem::Kind(k);
    for (int attempt = 0; attempt < 10000; ++attempt) {
        EquationProblem result;
        result.kind = kind;
        int64_t answer = 0;
        if (!equation_detail::candidate(rng, options, result, answer)) continue;
        const std::string wrong = equation_check::problems_with(result);
        CHECK(wrong.empty(), "the checker rejects a generated problem (%s): %s\n%s", wrong.c_str(),
              result.problem.question.c_str(), result.problem.steps.c_str());
        if (options.exclude && options.exclude->excludes(result.problem, result.distractor_number)) continue;
        if (!options.notes) result.problem.steps = math_detail::without_notes(result.problem.steps);
        return result;
    }
    CHECK(false, "could not generate an equation problem in 10000 attempts");
    return {};
}

inline std::vector<Problem> make_equation_problems(uint64_t seed, int64_t count, const MathOptions& options = {}) {
    return make_unique_problems(seed, count, [&](Rng& rng) { return make_equation_problem(rng, options); });
}
