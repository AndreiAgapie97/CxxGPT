// redraw.h — a generated problem written again with other numbers, for the
// re-draw check of self-train (docs/research-plan.md, item 3).
//
// A right method stays right when the question's numbers change; a lucky
// one ("4 / 2 = 2" where the story needs "4 - 2") almost never does. So
// self-train works a try's calculations out again on versions of its
// problem with new numbers (self_train.h). This file makes those versions.
//
// The generators (wordgen.h, mathgen.h, mathgen_fractions.h) make two kinds
// of random choices (Choices): what the story is, and the numbers in it.
// `gen-word`/`gen-math --recipes FILE` writes, for each problem, a Recipe:
// the generator, the options that change its question, and the random
// state its story was drawn from. To re-draw a problem:
//   1. its story is made again from the recipe, recording its story
//      choices; it must be the problem file's question and answer exactly,
//      so a recipe file cannot be paired with another problems file;
//   2. it is made again with the recorded story choices and numbers from a
//      new random stream: the same people, wordings and steps;
//   3. a version counts only if every story choice was played back, its
//      question is the original's word for word once every number is
//      masked ("Maya has # boxes of # cards."), and so is its worked
//      solution (the same plan, step for step), it is not the original or
//      an earlier version, and, for gen-math, the topic's second solver
//      accepts it. Story choices that depend on a number (an age gap that
//      turns "older" into "younger", "1 year" against "2 years") give
//      another masked question, and numbers that change the plan (a
//      division by 7 - 6 = 1 the solution leaves out, "1/2 of 344" written
//      as "344 - 172") another masked solution: those are drawn again.
// Masking also lines the numbers up: the third number of the original
// question is the third number of every version.
#pragma once

#include <cinttypes>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "failure_labels.h"
#include "mathgen_fractions.h"

namespace redraw {

// One recipe per line, in the problems file's order:
//   word distractors=1 test=0 steps=5 shape=2 state=<4 x 16 hex digits>
inline std::string recipe_line(const Recipe& r) {
    char state[4 * 17 + 1];
    std::snprintf(state, sizeof state, "%016" PRIx64 ",%016" PRIx64 ",%016" PRIx64 ",%016" PRIx64, r.state[0],
                  r.state[1], r.state[2], r.state[3]);
    return r.topic + " distractors=" + (r.distractors ? "1" : "0") + " test=" + (r.test ? "1" : "0") +
           " steps=" + std::to_string(r.steps) + " shape=" + std::to_string(r.shape) + " state=" + state;
}

inline std::optional<Recipe> parse_recipe(const std::string& line) {
    Recipe r;
    char topic[16] = {}, end = 0;
    int distractors = 0, test = 0;
    const int read = std::sscanf(line.c_str(),
                                 "%15s distractors=%d test=%d steps=%d shape=%d state=%16" SCNx64 ",%16" SCNx64
                                 ",%16" SCNx64 ",%16" SCNx64 "%c",
                                 topic, &distractors, &test, &r.steps, &r.shape, &r.state[0], &r.state[1],
                                 &r.state[2], &r.state[3], &end);
    r.topic = topic;
    if (read != 9 || (distractors != 0 && distractors != 1) || (test != 0 && test != 1)) return {};
    r.distractors = distractors == 1;
    r.test = test == 1;
    if (r.topic == "word") {
        if (r.steps < 2 || r.steps > 8 || r.shape < 0 || r.shape >= WordProblem::shape_count) return {};
    } else if (r.topic == "equations") {
        if (r.shape < 0 || r.shape >= EquationProblem::kind_count) return {};
    } else if (r.topic == "fractions") {
        if (r.shape < 0 || r.shape >= FractionProblem::kind_count) return {};
    } else {
        return {};
    }
    // xoshiro256** never reaches the all-zero state.
    if (!(r.state[0] | r.state[1] | r.state[2] | r.state[3])) return {};
    return r;
}

inline void save_recipes(const std::string& path, const std::vector<Recipe>& recipes) {
    FILE* f = create_file(path, "wb");
    CHECK(f, "cannot write %s", path.c_str());
    for (const Recipe& r : recipes) CHECK(std::fprintf(f, "%s\n", recipe_line(r).c_str()) >= 0, "error writing %s", path.c_str());
    CHECK(std::fclose(f) == 0, "error writing %s", path.c_str());
}

inline std::vector<Recipe> load_recipes(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "r");
    CHECK(f, "cannot open %s", path.c_str());
    std::vector<Recipe> out;
    std::string line;
    for (int c; (c = std::fgetc(f)) != EOF;) {
        if (c != '\n') {
            line += char(c);
            continue;
        }
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const auto r = parse_recipe(line);
        CHECK(r, "bad recipe on line %zu of %s: %s", out.size() + 1, path.c_str(), line.c_str());
        out.push_back(*r);
        line.clear();
    }
    std::fclose(f);
    CHECK(line.empty(), "%s does not end with a newline", path.c_str());
    return out;
}

// The problem `r` describes, with its random choices from `choices`; nothing
// when this draw makes no problem (the generator would draw again), or the
// topic's second solver rejects it.
inline std::optional<Problem> make(const Recipe& r, Choices& choices) {
    if (r.topic == "word") {
        WordOptions o;
        o.distractors = r.distractors;
        o.test = r.test;
        WordProblem w;
        w.shape = WordProblem::Shape(r.shape);
        if (!word_detail::candidate(choices, o, r.steps, w)) return {};
        return w.problem;
    }
    const MathOptions o{r.distractors, true, r.test, nullptr};
    if (r.topic == "equations") {
        EquationProblem e;
        e.kind = EquationProblem::Kind(r.shape);
        int64_t answer = 0;
        if (!equation_detail::candidate(choices, o, e, answer) || !equation_check::problems_with(e).empty()) return {};
        return e.problem;
    }
    CHECK(r.topic == "fractions", "no generator %s", r.topic.c_str());
    FractionProblem f;
    f.kind = FractionProblem::Kind(r.shape);
    if (!fraction_detail::candidate(choices, o, f) || !fraction_check::problems_with(f).empty()) return {};
    return f.problem;
}

// The question with every number masked: "Maya has # boxes of # cards."
inline std::string masked(const std::string& question) {
    std::string out;
    size_t at = 0;
    for (const labels::detail::Number& n : labels::detail::numbers_in(question)) {
        out += question.substr(at, n.at - at) + "#";
        at = n.end;
    }
    return out + question.substr(at);
}

// What a problem's re-draw found.
struct Versions {
    std::vector<Problem> versions;  // the same story with other numbers
    size_t draws = 0;               // stories drawn to find them
};

// Up to `count` versions of `original`, which `r` must make, drawing at
// most `max_draws` stories per version. Each draw takes its numbers from
// its own Rng, seeded from `seed` and the draw's number.
inline Versions versions_of(const Recipe& r, const Problem& original, int count, uint64_t seed,
                            int max_draws = 200) {
    std::vector<Choices::Choice> story;
    Rng rng(1);
    rng.set_state(r.state);
    Choices record(rng, &story);
    const auto again = make(r, record);
    CHECK(again && again->question == original.question && again->answer == original.answer,
          "the recipe does not make this problem (is the recipe file the problem file's?): %s",
          original.question.c_str());
    const std::string mask = masked(original.question), plan = masked(again->steps);
    std::set<std::string> seen = {original.question};
    Versions out;
    while (int(out.versions.size()) < count && out.draws < size_t(max_draws) * size_t(count)) {
        Rng numbers(seed + 0x9E3779B97F4A7C15ull * ++out.draws);
        Choices play(story, numbers);
        const auto p = make(r, play);
        if (p && play.played_back() && masked(p->question) == mask && masked(p->steps) == plan &&
            seen.insert(p->question).second)
            out.versions.push_back(*p);
    }
    return out;
}

}  // namespace redraw
