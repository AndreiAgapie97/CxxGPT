// wordgen.h — word problems made from sentence patterns and exact arithmetic.
// Values are integer hundredths: 125 means 1.25, never a rounded float.
// Counts must be whole. Money and measurements may have two decimal places;
// candidates needing finer precision are rejected, not rounded.
#pragma once

#include <algorithm>
#include <array>
#include <set>
#include <vector>

#include "reasoning.h"

// A generator's random choices, of two kinds (src/redraw.h): below() for
// what the story is (people, wordings, which steps, where the distractor
// goes) and number_below() for the numbers in it. To make a problem, both
// come from one Rng, in the order they are made, as before they were told
// apart: the same seed gives the same files. To write the same story with
// other numbers, its story choices are recorded once, then played back
// while the numbers come from another Rng.
class Choices {
public:
    // One choice: the range [0, n) and what was drawn.
    struct Choice {
        uint64_t n = 0, value = 0;
    };

    // Both kinds from `rng`; if `record` is given, it gets every story choice.
    explicit Choices(Rng& rng, std::vector<Choice>* record = nullptr) : rng_(rng), record_(record) {}
    // The story choices from `story`, in order; the numbers from `numbers`.
    Choices(const std::vector<Choice>& story, Rng& numbers) : rng_(numbers), story_(&story) {}

    uint64_t below(uint64_t n) {
        if (story_) {
            // A story made with other numbers asked for another choice: the
            // numbers changed the story. Go on with random choices (so loops
            // such as "draw a name until it differs" still end) and say so.
            if (next_ < story_->size() && (*story_)[next_].n == n) return (*story_)[next_++].value;
            diverged_ = true;
            return rng_.below(n);
        }
        const uint64_t v = rng_.below(n);
        if (record_) record_->push_back({n, v});
        return v;
    }
    uint64_t number_below(uint64_t n) { return rng_.below(n); }

    // Playing back: every recorded choice was asked for, and nothing else.
    bool played_back() const { return story_ && !diverged_ && next_ == story_->size(); }

private:
    Rng& rng_;
    std::vector<Choice>* record_ = nullptr;
    const std::vector<Choice>* story_ = nullptr;
    size_t next_ = 0;
    bool diverged_ = false;
};

// How to make one generated problem again (src/redraw.h): its generator,
// the options that change its question, what was chosen before its story
// (word: the steps and the shape; gen-math: the kind, in `shape`) and the
// random state its story was drawn from.
struct Recipe {
    std::string topic;  // "word", "equations", "fractions" or "number-theory"
    bool distractors = true, test = false;
    int steps = 0, shape = 0;
    uint64_t state[4] = {};
};

inline Recipe recipe_of(const std::string& topic, bool distractors, bool test, int steps, int shape, const Rng& rng) {
    Recipe r{topic, distractors, test, steps, shape, {}};
    for (int i = 0; i < 4; ++i) r.state[i] = rng.state()[i];
    return r;
}

struct WordOptions {
    int min_steps = 2, max_steps = 8;
    bool distractors = true, notes = true, test = false;
};

// The plan contains inputs and references to earlier nodes, never answers.
// Keeping it separate lets tests solve it without seeing computed values.
struct WordNode {
    char op = 0;  // 0 = an input number; otherwise +, -, *, /
    int64_t number = 0;
    size_t left = 0, right = 0;
    bool whole = false;
    std::string label;  // ends in a verb: "The bill now totals"
    // The GSM8K-style sentence around the step, naming the person and unit:
    // before + " 84+4 = <<84+4=88>>88 " + after + "." gives
    // "Maya now has 84+4 = <<84+4=88>>88 toy cars." Empty when the step has
    // no natural unit (multiplying by a percentage before dividing by 100).
    std::string before, after;
};

struct WordProblem {
    Problem problem;
    std::vector<WordNode> plan;
    std::vector<int64_t> values;  // one checked value per plan node
    std::vector<int> patterns;   // sentence IDs, for split/coverage tests
    // Story inputs let tests also solve each shape directly, independently
    // of the arithmetic plan and its computed results.
    enum Shape { chain, comparison, total, rate, part, change, shape_count } shape = chain;
    std::array<int64_t, 4> inputs{};
    std::vector<int64_t> additions;
    bool three_people = false, percentage = false;
    int distractor = -1, distractor_position = -1;
    int64_t distractor_number = 0;
    // The story's people and objects, so tests can check the solution names them.
    std::string name, other, third, object;
    Recipe recipe;  // how to make this problem again
};

namespace word_detail {

constexpr int64_t scale = 100;
constexpr int64_t cap = 1000000 * scale;

// With operands at most 100,000,000, the largest temporary is 10^16,
// safely below int64_t's limit. Check operands before doing any arithmetic.
inline bool calculate(char op, int64_t a, int64_t b, bool whole, int64_t& out) {
    if (a < 0 || b < 0 || a > cap || b > cap) return false;
    int64_t n = 0, d = 1;
    switch (op) {
        case '+': n = a + b; break;
        case '-': n = a - b; break;
        case '*': n = a * b; d = scale; break;
        case '/': n = a * scale; d = b; break;
        default: return false;
    }
    if (d == 0 || n < 0 || n % d != 0) return false;
    const int64_t v = n / d;
    if (v > cap || (whole && v % scale != 0)) return false;
    out = v;
    return true;
}

inline std::string number(int64_t value) {
    std::string s = std::to_string(value / scale);
    const int64_t cents = value % scale;
    if (cents) {
        s += '.';
        s += char('0' + cents / 10);
        if (cents % 10) s += char('0' + cents % 10);
    }
    return s;
}

inline std::string money(int64_t value) {
    std::string text = number(value);
    if (value % scale && value % 10 == 0) text += '0';
    return text;
}

enum Family { boxes, production, earnings, containers, receive, give, share,
              fraction, percent, sell, buy, convert, compare_people, total_two,
              total_three, time_rate, part_fraction, part_percent, shopping,
              extra_cost, part_extra, receive_b, receive_c, extra_cost_b, extra_cost_c, part_extra_b,
              part_extra_c, family_count };

// Six distinct question patterns per family: four for train, two held out
// for test. Every sentence of a question, its distractor included, comes from
// its split's own patterns, so even independently generated files with
// different seeds cannot share questions.
// All variants within a family have exactly the same arithmetic meaning.
inline constexpr std::array<std::array<const char*, 6>, family_count> patterns = {{
    {{"{name} has {a} boxes with {b} {unit} in each box.",
      "{name} unpacks {a} cartons, each containing {b} {unit}.",
      "{name} buys {a} packs of {b} {unit} each.",
      "{name} collects {a} bags that each hold {b} {unit}.",
      "On a shelf, {name} finds {a} trays of {b} {unit} apiece.",
      "A delivery for {name} contains {a} bundles, with {b} {unit} per bundle."}},
    {{"{name} makes {a} {unit} per hour for {b} hours.",
      "{name} spends {b} hours producing {a} {unit} each hour.",
      "{name} assembles {a} {unit} every hour during a {b}-hour shift.",
      "{name} works for {b} hours at a steady rate of {a} {unit} per hour.",
      "During a workshop lasting {b} hours, {name} completes {a} {unit} an hour.",
      "At a rate of {a} {unit} hourly, {name} finishes a {b}-hour session."}},
    {{"{name} earns ${a} per hour for {b} hours of work.",
      "{name} works {b} hours at an hourly wage of ${a}.",
      "{name} is paid ${a} for each of {b} hours spent helping a neighbor.",
      "{name} spends {b} hours tutoring for ${a} an hour.",
      "A {b}-hour job pays {name} ${a} for every hour.",
      "For working {b} hours, {name} receives an hourly payment of ${a}."}},
    {{"{name} has {a} containers holding {b} {unit} each.",
      "{name} fills {a} jars with {b} {unit} per jar.",
      "{name} brings {a} bottles, each containing {b} {unit}.",
      "{name} prepares {a} jugs of {b} {unit} each.",
      "A crate belonging to {name} holds {a} flasks with {b} {unit} in every flask.",
      "For a picnic, {name} packs {a} canisters containing {b} {unit} apiece."}},
    {{"Next, {name} receives {a} more {unit}.",
      "A friend gives {name} another {a} {unit}.",
      "Then {name} gets an additional {a} {unit}.",
      "A neighbor adds {a} {unit} to what {name} has.",
      "Afterward, a gift increases what {name} has by {a} {unit}.",
      "Later, {name} accepts a donation of {a} {unit}."}},
    {{"Next, {name} gives away {a} {unit}.",
      "Then {name} donates {a} {unit} to a neighbor.",
      "A friend receives {a} {unit} from {name}.",
      "After that, {name} hands over {a} {unit} to a cousin.",
      "Later, {name} contributes {a} {unit} to a community collection.",
      "Afterward, {name} sends {a} {unit} to a sibling."}},
    {{"{name} divides the current total into {a} equal shares and keeps one share.",
      "{name} splits everything equally among {a} people, including {name}.",
      "{name} makes {a} equal portions from the current amount and retains one.",
      "{name} shares the entire amount equally with a group of {a} people that includes {name}.",
      "Of {a} equal allocations made from the current total, exactly one belongs to {name}.",
      "The current amount is distributed evenly to {a} recipients; {name} is one of them."}},
    {{"{name} keeps 1/{a} of the current amount and gives away the rest.",
      "Then {name} donates all but 1/{a} of what remains.",
      "Only 1/{a} of the current total is retained by {name}.",
      "{name} passes on everything except 1/{a} of the amount now held.",
      "After a donation, {name} is left with 1/{a} of the amount held just before it.",
      "{name} saves a fraction of 1/{a} of the current amount and gives the rest to a friend."}},
    {{"{name} keeps {a}% of the current amount and donates the rest.",
      "Then {name} gives away everything except {a}% of the current total.",
      "Only {a}% of the amount now held stays with {name}.",
      "{name} retains {a}% of what remains and passes the rest to a friend.",
      "Following a donation, {name} has {a}% of the amount held immediately beforehand.",
      "{name} sets aside {a}% of the current total for personal use and gives all the rest away."}},
    {{"{name} sells all the remaining {unit} for ${a} each, starting with no money.",
      "With no money beforehand, {name} sells every remaining item at ${a} per item.",
      "{name} exchanges all the remaining {unit} for ${a} apiece and has no other money.",
      "Each remaining item sells for ${a}; {name} sells them all and has only the proceeds.",
      "A buyer pays {name} ${a} for each remaining item; this sale is now all of {name}'s money.",
      "All remaining {unit} are sold by {name} at ${a} each, leaving only the sale proceeds."}},
    {{"{name} spends all the money on {object} costing ${a} each.",
      "{name} uses the entire amount of money to buy {object} at ${a} apiece.",
      "With all the money, {name} purchases {object} for ${a} per item.",
      "{name} trades every dollar for {object} priced at ${a} each.",
      "At a price of ${a} per item, {name} buys {object} until the money is used up exactly.",
      "{name} converts the full amount of money into {object}, paying ${a} for each one."}},
    {{"{name} now measures the juice in milliliters, using {a} milliliters per liter.",
      "{name} converts the juice amount to milliliters; each liter is {a} milliliters.",
      "Using {a} milliliters for every liter, {name} changes the unit to milliliters.",
      "{name} expresses the juice volume in milliliters, with {a} milliliters in a liter.",
      "The juice is relabeled by {name} in milliliters at {a} milliliters to the liter.",
      "For the remaining juice, {name} switches to milliliters; one liter equals {a} milliliters."}},
    {{"{name} has {a} packs of {b} {unit}, while {other} has {c} {unit}.",
      "{name} owns {a} boxes with {b} {unit} in each; {other} owns {c} {unit}.",
      "{name} collects {a} bundles of {b} {unit} each, and {other} collects {c} {unit}.",
      "{name} counts {a} bags containing {b} {unit} apiece; {other} counts {c} {unit}.",
      "For a collection, {name} brings {a} trays of {b} {unit}; {other} brings {c} {unit}.",
      "On separate shelves, {name} stores {a} cartons of {b} {unit} and {other} stores {c} {unit}."}},
    {{"{name} has {a} boxes of {b} {unit}, and {other} has {c} {unit} to add to their joint collection.",
      "{name} brings {a} bags of {b} {unit} each; {other} brings {c} {unit} for them to pool.",
      "For their shared stall, {name} supplies {a} packs of {b} {unit} and {other} supplies {c} {unit}.",
      "{name} contributes {a} bundles of {b} {unit} apiece, with {c} {unit} contributed by {other}.",
      "A joint display uses {a} trays of {b} {unit} from {name} and {c} {unit} from {other}.",
      "To combine their collections, {name} sets out {a} cartons of {b} {unit} and {other} sets out {c} {unit}."}},
    {{"{name} has {a} boxes of {b} {unit}, {other} has {c} {unit}, and {third} has {d} {unit}.",
      "Three friends pool their collections: {name} brings {a} bags of {b} {unit}, {other} brings {c}, and {third} brings {d}.",
      "{name} contributes {a} packs with {b} {unit} each, {other} contributes {c} {unit}, and {third} contributes {d} {unit}.",
      "For a group project, {name} supplies {a} bundles of {b} {unit}; {other} supplies {c} {unit} and {third} supplies {d} {unit}.",
      "A shared exhibit has {a} trays of {b} {unit} from {name}, {c} {unit} from {other}, and {d} {unit} from {third}.",
      "The contributions are {a} cartons of {b} {unit} belonging to {name}, {c} {unit} belonging to {other}, and {d} {unit} belonging to {third}."}},
    {{"{name} makes {a} {unit} per {time} for {b} {times}, while {other} has already made {c} {unit}.",
      "{name} works for {b} {times}, producing {a} {unit} each {time}; {other} contributes {c} finished {unit}.",
      "For {b} {times}, {name} completes {a} {unit} every {time}, and {other} brings {c} ready-made {unit}.",
      "{name} produces {a} {unit} each {time} during a project lasting {b} {times}; {other} has {c} {unit} to add.",
      "During a workshop lasting {b} {times}, {name} makes {a} {unit} per {time}; {other} supplies {c} more {unit}.",
      "At a steady pace of {a} {unit} every {time}, {name} finishes {b} {times} of work before pooling the output with {other}'s {c} {unit}."}},
    {{"{name} has {a} red {unit} and {c} blue {unit}, and keeps 1/{b} of the red ones and all the blue ones.",
      "Of {name}'s {a} red {unit}, 1/{b} are kept; {c} blue {unit} are also kept.",
      "{name} saves 1/{b} of a set of {a} red {unit} together with a separate set of {c} blue {unit}.",
      "There are {a} red {unit} and {c} blue {unit}; {name} retains 1/{b} of the red ones and every blue one.",
      "From a collection of {a} red {unit} and {c} blue {unit}, {name} selects 1/{b} of the red ones plus all the blue ones.",
      "{name} sets aside every one of {c} blue {unit} along with 1/{b} of {a} red {unit}."}},
    {{"{name} has {a} red {unit} and {c} blue {unit}, and keeps {b}% of the red ones and all the blue ones.",
      "Of {name}'s {a} red {unit}, {b}% are kept; {c} blue {unit} are also kept.",
      "{name} saves {b}% of a set of {a} red {unit} together with a separate set of {c} blue {unit}.",
      "There are {a} red {unit} and {c} blue {unit}; {name} retains {b}% of the red ones and every blue one.",
      "From a collection of {a} red {unit} and {c} blue {unit}, {name} selects {b}% of the red ones plus all the blue ones.",
      "{name} sets aside every one of {c} blue {unit} along with {b}% of {a} red {unit}."}},
    {{"{name} buys {a} {object} at ${b} each and pays with ${c}.",
      "At a shop, {name} chooses {a} {object} costing ${b} apiece and hands over ${c}.",
      "{name} pays ${c} for a purchase of {a} {object} priced at ${b} per item.",
      "The cashier charges {name} ${b} each for {a} {object}; {name} gives the cashier ${c}.",
      "For {a} {object} with a unit price of ${b}, {name} presents a payment of ${c}.",
      "A bill covers {a} {object} at ${b} per item, and {name} settles it by handing over ${c}."}},
    {{"{name} also buys an accessory for ${a}.",
      "An extra item adds ${a} to the bill.",
      "{name} adds a gift costing ${a} to the purchase.",
      "Another small purchase costs {name} ${a}.",
      "A further item on the same bill is priced at ${a}.",
      "The cashier includes another purchase worth ${a}."}},
    // The part shape asks how many items are kept, so its extra items say
    // plainly that they are kept too (a gift alone would be ambiguous), and
    // where they come from (not "keeps 6 green cards" out of nowhere).
    {{"A friend gives {name} {a} more {unit}, and {name} keeps them.",
      "{name} also buys {a} {unit} and keeps them too.",
      "On top of that, {name} finds {a} green {unit} and keeps them.",
      "{name} also wins {a} yellow {unit} at a fair and keeps them.",
      "In addition, {name} retains {a} more {unit} received as a gift.",
      "A cousin hands {name} a further {a} {unit}, which {name} keeps."}},
    // Long stories add up to six amounts of one kind. Three families of
    // each kind take turns (in_turn), so a story never needs a wording
    // twice: twelve for train, six for test.
    {{"{name} is given {a} more {unit}.",
      "Soon after, {name} collects {a} more {unit}.",
      "{name} adds {a} {unit} that a relative brought.",
      "Later in the week, {name} obtains another {a} {unit}.",
      "A teacher hands {name} {a} more {unit}.",
      "{name} ends up with {a} extra {unit} from a classmate."}},
    {{"{name}'s aunt sends {a} more {unit}.",
      "{name} wins {a} more {unit} in a raffle.",
      "Then {name} is handed another {a} {unit}.",
      "{name} finds {a} more {unit} in a cupboard.",
      "Someone at school passes {a} more {unit} to {name}.",
      "{name} takes in {a} additional {unit} from an uncle."}},
    {{"{name} picks up a snack for ${a} as well.",
      "A greeting card for ${a} goes on the same bill.",
      "{name} also adds a bag costing ${a}.",
      "A small toy for ${a} is added to the purchase.",
      "{name} grabs a drink priced at ${a} too.",
      "One more thing, a notebook for ${a}, joins the bill."}},
    {{"{name} also pays ${a} for gift wrapping.",
      "The bill also includes a ${a} delivery fee.",
      "{name} adds batteries costing ${a}.",
      "A box of tissues for ${a} is also bought.",
      "{name} throws in a pen worth ${a} as well.",
      "There is also a ${a} charge for a sturdy bag."}},
    {{"{name}'s grandmother sends {a} {unit}, and {name} keeps all of them.",
      "{name} trades for {a} purple {unit} and keeps them.",
      "A classmate gives {name} {a} {unit} to keep.",
      "{name} keeps the {a} {unit} that came with a birthday card.",
      "{name} receives {a} orange {unit} at school and keeps every one.",
      "An uncle passes {a} {unit} to {name}, who keeps them."}},
    {{"{name} also gets {a} white {unit} from a swap and keeps them.",
      "A teacher rewards {name} with {a} {unit}, which {name} keeps.",
      "{name} makes {a} more {unit} and keeps them as well.",
      "Then {name} is given {a} pink {unit} and keeps them.",
      "{name} collects {a} more {unit} at a market and keeps them.",
      "A neighbor leaves {a} {unit} for {name}, who keeps them all."}}
}};

// The family for the index-th extra sentence of one kind: the three
// families of that kind take turns (receive, receive_b, receive_c, ...).
inline int in_turn(int family, size_t index) {
    const size_t k = index % 3;
    if (k == 0) return family;
    if (family == receive) return k == 1 ? receive_b : receive_c;
    if (family == extra_cost) return k == 1 ? extra_cost_b : extra_cost_c;
    if (family == part_extra) return k == 1 ? part_extra_b : part_extra_c;
    return family;
}

inline bool is_receive(int family) { return family == receive || family == receive_b || family == receive_c; }

// Separate wording in each split, just like the arithmetic sentences.
inline constexpr std::array<const char*, 24> distractors = {
    "The room has {a} windows.", "A nearby clock has {a} painted stars.",
    "There are {a} pictures on the wall.", "The front door has {a} stripes.",
    "The building's address is {a} Oak Street.", "The bus stop displays route {a}.",
    "A poster advertises an unrelated event in room {a}.",
    "The storage room is labeled {a}.", "A shelf has {a} decorative ribbons.",
    "A radio is tuned to station {a}.", "The notebook cover is green.",
    "Outside, a bird rests in a tree.", "The hallway has {a} lamps.",
    "A sign points toward platform {a}.", "The coat rack has {a} hooks.",
    "A calendar picture shows {a} boats.", "The office door is numbered {a}.",
    "A visitor wears a jersey numbered {a}.", "A notice lists extension {a} for the caretaker.",
    "The waiting room has {a} chairs.", "A mural depicts {a} clouds.",
    "A placard marks parking space {a}.", "The curtains are yellow.",
    "A breeze rustles the leaves outside."};

inline void replace(std::string& text, const std::string& key, const std::string& value) {
    for (size_t at = 0; (at = text.find(key, at)) != std::string::npos; at += value.size())
        text.replace(at, key.size(), value);
}

enum class Kind { objects, money, liters, milliliters };

// How the solution may describe one step: a label ending in a verb, and the
// sentence naming the person and unit around it (see WordNode).
struct StepText {
    std::string label, before, after;
};

// Everything one story attempt shares: the random source, the people and
// objects, the current unit, and the plan, sentences and question being built.
struct Story {
    Choices& rng;
    const WordOptions& options;
    WordProblem& out;
    std::string name, other, third, object, time;
    int start = boxes;  // the chain's opening family: boxes, production, earnings or containers
    Kind kind = Kind::objects;
    std::vector<std::string> sentences;
    std::string question;

    // These draws happen for every shape, in this order, even when a shape
    // does not use them, so the random sequence stays the same.
    Story(Choices& random, const WordOptions& opts, WordProblem& problem) : rng(random), options(opts), out(problem) {
        static const std::array<const char*, 12> names = {
            "Alex", "Maya", "Noah", "Lina", "Sam", "Nora", "Leo", "Asha", "Ben", "Zoe", "Omar", "Iris"};
        static const std::array<const char*, 8> objects = {
            "stickers", "cards", "beads", "pencils", "candles", "toy cars", "bookmarks", "paper flowers"};
        static const std::array<const char*, 3> times = {"hour", "day", "week"};
        // Each person is drawn separately, redrawn on a clash, so any two
        // names can appear together (132 ordered pairs, not 12 fixed ones).
        name = names[rng.below(names.size())];
        do other = names[rng.below(names.size())]; while (other == name);
        do third = names[rng.below(names.size())]; while (third == name || third == other);
        object = objects[rng.below(objects.size())];
        start = int(rng.below(4));
        kind = start == earnings ? Kind::money : start == containers ? Kind::liters : Kind::objects;
        time = times[rng.below(times.size())];
        out.name = name;
        out.other = other;
        out.third = third;
        out.object = object;
    }

    std::string unit() const {
        if (kind == Kind::money) return "dollars";
        if (kind == Kind::liters) return "liters of juice";
        if (kind == Kind::milliliters) return "milliliters of juice";
        return object;
    }

    int64_t draw(int lo, int hi) { return int64_t(lo + int(rng.number_below(uint64_t(hi - lo + 1)))); }

    // Fills one sentence pattern of `family` (a split-specific variant) with
    // the story's people, unit and the numbers a..d.
    // A family's wording is not used twice in one story until all of the
    // split's wordings have been (four for train, two for test).
    std::array<std::vector<int>, family_count> used_variants;

    void sentence(int family, int64_t a, int64_t b = 0, int64_t c = 0, int64_t d = 0) {
        std::vector<int>& used = used_variants[size_t(family)];
        std::vector<int> fresh;
        for (int v = options.test ? 4 : 0; v < (options.test ? 6 : 4); ++v)
            if (std::find(used.begin(), used.end(), v) == used.end()) fresh.push_back(v);
        if (fresh.empty()) {
            used.clear();
            for (int v = options.test ? 4 : 0; v < (options.test ? 6 : 4); ++v) fresh.push_back(v);
        }
        const int variant = fresh[rng.below(fresh.size())];
        used.push_back(variant);
        std::string s = patterns[family][variant];
        replace(s, "{name}", name);
        replace(s, "{other}", other);
        replace(s, "{third}", third);
        replace(s, "{time}", time);
        replace(s, "{times}", time + "s");
        replace(s, "{object}", object);
        replace(s, "{unit}", unit());
        replace(s, "${a}", "$" + money(a));
        replace(s, "${b}", "$" + money(b));
        replace(s, "${c}", "$" + money(c));
        replace(s, "{a}", kind == Kind::money && (is_receive(family) || family == give) ? money(a) : number(a));
        replace(s, "{b}", number(b));
        replace(s, "{c}", number(c));
        replace(s, "{d}", number(d));
        sentences.push_back(s);
        out.patterns.push_back(family * 6 + variant);
    }

    // Adds an input number to the plan and returns its index.
    size_t literal(int64_t value) {
        const size_t index = out.plan.size();
        out.plan.push_back({0, value, 0, 0, false, "", "", ""});
        out.values.push_back(value);
        return index;
    }

    // Adds `left op right` to the plan; false if the result is not allowed.
    bool operation(char op, size_t left, size_t right, bool whole, const StepText& text) {
        int64_t value = 0;
        if (!calculate(op, out.values[left], out.values[right], whole, value)) return false;
        // Nothing left, or exactly one ("1 cards", "1 dollars"), reads badly.
        if (value == 0 || value == scale) return false;
        out.plan.push_back({op, 0, left, right, whole, text.label, text.before, text.after});
        out.values.push_back(value);
        return true;
    }
};

// The chain: one running quantity, changed step by step.
inline bool chain_story(Story& s, int steps) {
    int64_t a = s.draw(2, 20) * scale, b = s.draw(2, 12) * scale;
    if (s.kind == Kind::money) a += s.draw(0, 3) * 25;
    if (s.kind == Kind::liters) b = s.draw(1, 16) * 25;
    if (b == scale) return false;  // "1 liters of juice per jar"
    s.sentence(s.start, a, b);
    const size_t left = s.literal(a), right = s.literal(b);
    if (!s.operation('*', left, right, s.kind == Kind::objects,
                     {"The starting amount is", s.name + " starts with", s.unit()})) return false;
    // A story a person would tell: at most two steps that cut the amount
    // down (an equal share, a fraction or a percentage kept), never two in
    // a row, and at most one sale and one purchase. Gifts take turns among
    // three families (at most six), giving away happens at most twice: no
    // wording is needed twice.
    int done = 1, cuts = 0, last = -1, gifts = 0, gives = 0;
    bool sold = false, bought = false;
    while (done < steps) {
        const bool can_cut = cuts < 2 && last != share && last != fraction && last != percent;
        std::vector<int> choices;
        if (gifts < 6) choices.push_back(receive);
        if (gives < 2) choices.push_back(give);
        if (can_cut) choices.insert(choices.end(), {share, fraction});
        if (can_cut && steps - done >= 2) choices.push_back(percent);
        if (s.kind == Kind::objects && !sold) choices.push_back(sell);
        if (s.kind == Kind::money && !bought) choices.push_back(buy);
        if (s.kind == Kind::liters) choices.push_back(convert);
        if (choices.empty()) return false;
        int family = choices[s.rng.below(choices.size())];
        if (family == receive) family = in_turn(receive, size_t(gifts++));
        if (family == give) ++gives;
        last = family;
        if (family == share || family == fraction || family == percent) ++cuts;
        if (family == sell) sold = true;
        if (family == buy) bought = true;
        a = s.draw(2, 24) * scale;
        if (family == share || family == fraction) a = s.draw(2, 5) * scale;
        if (family == percent) a = s.draw(1, 3) * 25 * scale;
        if (family == sell || family == buy) a = s.draw(1, 32) * 25;
        if (family == convert) a = 1000 * scale;
        if ((is_receive(family) || family == give) && s.kind != Kind::objects) {
            a += s.draw(0, 3) * 25;
            // Draw in liters first, then express the amount in the current unit.
            if (s.kind == Kind::milliliters) a *= 1000;
        }
        s.sentence(family, a);
        const size_t previous = s.out.values.size() - 1, operand = s.literal(a);
        char op = '+';
        StepText text{"After receiving more, the amount is", s.name + " now has", ""};
        // Selling, buying and converting change the unit from this step on.
        if (family == give) { op = '-'; text = {"After giving some away, the amount is", s.name + " has", ""}; }
        if (family == share) { op = '/'; text = {"One equal share is", s.name + "'s share is", ""}; }
        if (family == fraction) { op = '/'; text = {"The fraction kept is", s.name + " keeps", ""}; }
        if (family == percent) { op = '*'; text = {"Multiplying by the percentage gives", "", ""}; }
        if (family == sell) {
            op = '*';
            text = {"In dollars, the sale brings in", s.name + " gets", ""};
            s.kind = Kind::money;
        }
        if (family == buy) {
            op = '/';
            text = {"The number of items bought is", s.name + " buys", ""};
            s.kind = Kind::objects;
        }
        if (family == convert) {
            op = '*';
            text = {"The volume in milliliters is", s.name + " has", ""};
            s.kind = Kind::milliliters;
        }
        // The amount times a percentage (30*25 = 750) is not yet an amount
        // of anything, so that step gets no sentence with a unit.
        if (family != percent) text.after = s.unit();
        if (family == give) text.after += " left";
        if (family == sell) text.after += " from the sale";
        if (!s.operation(op, previous, operand, s.kind == Kind::objects, text)) return false;
        ++done;
        if (family == percent) {
            const size_t product = s.out.values.size() - 1, hundred = s.literal(100 * scale);
            if (!s.operation('/', product, hundred, s.kind == Kind::objects,
                             {"Dividing by 100, the amount kept is", s.name + " keeps", s.unit()}))
                return false;
            ++done;
        }
    }
    s.question = "How many " + s.unit() + " does " + s.name + " have now?";
    return true;
}

// The other five shapes keep their story's numbers in out.inputs, so tests
// can solve them a second way. What inputs[0..3] (a, b, c, d) mean:
//
//   shape       a                  b                      c                    d
//   comparison  packs {name} has   items per pack         {other}'s items      unused
//   total       boxes {name} has   items per box          {other}'s items      {third}'s items (three people only)
//   rate        items per {time}   number of {time}s      {other}'s items      unused
//   part        red items          keeps 1/b, or b%       blue items, all kept unused
//   change      items bought       price of one, dollars  payment, dollars     unused
//
// Chain stories leave inputs at zero: their numbers are only in the plan.
// Each of these shapes first draws all four numbers, then may redraw some.
inline std::array<int64_t, 4> draw_inputs(Story& s) {
    const int64_t a = s.draw(2, 20) * scale, b = s.draw(2, 12) * scale;
    const int64_t c = s.draw(2, 30) * scale, d = s.draw(2, 30) * scale;
    return {a, b, c, d};
}

// The opening sentence and its first operation, a op b.
inline bool opening(Story& s, int family, const std::array<int64_t, 4>& inputs, char op, const StepText& text) {
    s.out.inputs = inputs;
    s.sentence(family, inputs[0], inputs[1], inputs[2], inputs[3]);
    const size_t left = s.literal(inputs[0]), right = s.literal(inputs[1]);
    return s.operation(op, left, right, s.kind == Kind::objects, text);
}

// Longer stories add `count` extra amounts (items, or purchases on the
// bill), one sentence and one addition each.
inline bool add_amounts(Story& s, const std::vector<int64_t>& amounts, int family, const StepText& text) {
    for (const int64_t extra : amounts) {
        const int turn = in_turn(family, s.out.additions.size());
        s.out.additions.push_back(extra);
        s.sentence(turn, extra);
        const size_t previous = s.out.values.size() - 1, operand = s.literal(extra);
        if (!s.operation('+', previous, operand, s.kind == Kind::objects, text)) return false;
    }
    return true;
}

inline bool add_extras(Story& s, int count, int family, const StepText& text) {
    for (int i = 0; i < count; ++i)
        if (!add_amounts(s, {s.draw(2, 24) * scale}, family, text)) return false;
    return true;
}

// The last value so far plus (or minus) a new input number.
inline bool close_with(Story& s, char op, int64_t value, bool number_first, const StepText& text) {
    const size_t current = s.out.values.size() - 1, separate = s.literal(value);
    return number_first ? s.operation(op, separate, current, s.kind == Kind::objects, text)
                        : s.operation(op, current, separate, s.kind == Kind::objects, text);
}

// How many more items one person has than another.
inline bool comparison_story(Story& s, int steps) {
    s.kind = Kind::objects;
    const auto in = draw_inputs(s);
    if (!opening(s, compare_people, in, '*', {"The first collection contains", s.name + " has", s.object}))
        return false;
    if (!add_extras(s, steps - 2, receive, {"Including the extra items gives", s.name + " now has", s.object}))
        return false;
    if (!close_with(s, '-', in[2], false, {"The difference between the collections is", s.name + " has",
                                           "more " + s.object + " than " + s.other}))
        return false;
    s.question = "How many more " + s.object + " does " + s.name + " have than " + s.other + "?";
    return true;
}

// Two or three people's items added up.
inline bool total_story(Story& s, int steps) {
    s.kind = Kind::objects;
    const auto in = draw_inputs(s);
    const bool three = s.out.three_people = steps >= 3 && s.rng.below(2);
    const std::string pair = s.name + " and " + s.other + " have";
    if (!opening(s, three ? total_three : total_two, in, '*',
                 {"The first collection contains", s.name + " has", s.object}))
        return false;
    if (!add_extras(s, steps - 1 - (three ? 2 : 1), receive,
                    {"Including the extra items gives", s.name + " now has", s.object}))
        return false;
    if (!close_with(s, '+', in[2], false,
                    {"Combining the two amounts gives", three ? pair : "Together, " + pair, s.object}))
        return false;
    if (three && !close_with(s, '+', in[3], false,
                             {"Adding the third collection gives",
                              "Together, " + s.name + ", " + s.other + ", and " + s.third + " have", s.object}))
        return false;
    s.question = "How many " + s.object + " do " + s.name + (three ? ", " : " and ") + s.other +
                 (three ? ", and " + s.third : "") + " have altogether?";
    return true;
}

// Items made at a rate for some hours, days or weeks, plus another person's.
inline bool rate_story(Story& s, int steps) {
    s.kind = Kind::objects;
    const auto in = draw_inputs(s);
    if (!opening(s, time_rate, in, '*', {"The work produces", s.name + " makes", s.object})) return false;
    if (!add_extras(s, steps - 2, receive, {"Including the extra items gives", s.name + " now has", s.object}))
        return false;
    if (!close_with(s, '+', in[2], false, {"Combining the two amounts gives",
                                           "Together, " + s.name + " and " + s.other + " have", s.object}))
        return false;
    s.question = "How many " + s.object + " do " + s.name + " and " + s.other + " have altogether?";
    return true;
}

// A fraction or percentage of the red items, plus all the blue ones.
inline bool part_story(Story& s, int steps) {
    s.kind = Kind::objects;
    auto in = draw_inputs(s);
    const bool percentage = s.out.percentage = steps >= 3 && s.rng.below(2);
    in[0] = s.draw(3, 30) * 4 * scale;
    in[1] = percentage ? s.draw(1, 3) * 25 * scale : s.draw(2, 4) * scale;
    const StepText red_kept{"The number of red items kept is", s.name + " keeps", "red " + s.object};
    if (percentage) {
        // The red items times the percentage is not yet a count: no unit.
        if (!opening(s, part_percent, in, '*', {"Multiplying by the percentage gives", "", ""})) return false;
        const size_t product = s.out.values.size() - 1, hundred = s.literal(100 * scale);
        if (!s.operation('/', product, hundred, true, red_kept)) return false;
    } else if (!opening(s, part_fraction, in, '/', red_kept)) return false;
    if (!add_extras(s, steps - 1 - (percentage ? 2 : 1), part_extra,
                    {"Including the extra kept items gives", s.name + " now keeps", s.object}))
        return false;
    if (!close_with(s, '+', in[2], false,
                    {"Combining the two amounts gives", "In all, " + s.name + " keeps", s.object}))
        return false;
    s.question = "How many " + s.object + " does " + s.name + " keep in all?";
    return true;
}

// A purchase (and maybe extra items on the bill), then change from a payment.
// The payment is what a shopper would hand over: the next multiple of $5,
// $10, $20, $50 or $100 above the bill, never more than about twice it
// (no $180 for two stickers at $0.75).
inline bool change_story(Story& s, int steps) {
    s.kind = Kind::money;
    auto in = draw_inputs(s);
    in[1] = s.draw(1, 32) * 25;
    // The extra items are drawn first: the payment depends on the whole bill.
    std::vector<int64_t> extra_items;
    for (int i = 0; i < steps - 2; ++i) extra_items.push_back(s.draw(1, 32) * 25);
    int64_t bill = in[0] / scale * in[1];
    for (const int64_t extra : extra_items) bill += extra;
    std::vector<int64_t> payments;
    for (const int64_t note : {5, 10, 20, 50, 100}) {
        const int64_t pay = (bill / (note * scale) + 1) * note * scale;
        if (pay <= 2 * bill + 5 * scale && std::find(payments.begin(), payments.end(), pay) == payments.end())
            payments.push_back(pay);
    }
    in[2] = payments[s.rng.number_below(payments.size())];
    if (!opening(s, shopping, in, '*',
                 {"In dollars, the items cost", s.name + " spends", "dollars on the " + s.object}))
        return false;
    if (!add_amounts(s, extra_items, extra_cost, {"The bill now totals", s.name + "'s bill is now", "dollars"}))
        return false;
    if (!close_with(s, '-', in[2], true, {"The change in dollars is", s.name + " gets", "dollars in change"}))
        return false;
    const size_t extras = s.out.additions.size();
    const std::string after = extras == 0 ? "" : extras == 1 ? " after both purchases" : " after all these purchases";
    s.question = "How much change in dollars does " + s.name + " receive" + after + "?";
    return true;
}

// One irrelevant sentence, placed anywhere in the story. Drawn even when
// disabled, so toggling distractors does not change the math.
inline void add_distractor(Story& s) {
    WordProblem& out = s.out;
    const int extra = int(s.rng.below(12)) + (s.options.test ? 12 : 0);
    int64_t irrelevant = s.draw(2, 30) * scale;
    // Avoid even accidental reuse of a distractor number in the solution.
    while (std::find(out.values.begin(), out.values.end(), irrelevant) != out.values.end()) irrelevant += scale;
    const size_t position = s.rng.below(s.sentences.size() + 1);
    if (s.options.distractors) {
        std::string text = distractors[extra];
        out.distractor = extra;
        out.distractor_position = int(position);
        if (text.find("{a}") != std::string::npos) out.distractor_number = irrelevant;
        replace(text, "{a}", number(irrelevant));
        s.sentences.insert(s.sentences.begin() + position, text);
    }
}

// The worked solution: one line per operation, in one of five wordings:
//   0  The bill now totals: 75+2.5 = <<75+2.5=77.5>>77.5.
//   1  The bill now totals 75+2.5 = <<75+2.5=77.5>>77.5.
//   2  We calculate 75+2.5 = <<75+2.5=77.5>>77.5 for this step.
//   3  This gives 75+2.5 = <<75+2.5=77.5>>77.5.  (" as the result" on the last step only)
//   4  Maya's bill is now 75+2.5 = <<75+2.5=77.5>>77.5 dollars.  (the GSM8K way)
// Wording 4 is drawn twice as often as each other one. A step without a
// sentence of its own (see WordNode) uses wording 0 or 1 instead.
inline void write_solution(Story& s) {
    WordProblem& out = s.out;
    for (size_t i = 0; i < out.plan.size(); ++i) {
        const WordNode& node = out.plan[i];
        if (!node.op) continue;
        const bool last = i + 1 == out.plan.size();
        const std::string expression = number(out.values[node.left]) + node.op + number(out.values[node.right]);
        const std::string result = number(out.values[i]);
        std::string& steps = out.problem.steps;
        if (!steps.empty()) steps += '\n';
        int wording = int(s.rng.below(6));
        if (wording == 5) wording = 4;
        if (wording == 4 && node.before.empty()) wording = int(s.rng.below(2));
        if (wording == 0) steps += node.label + ": ";
        if (wording == 1) steps += node.label + ' ';
        if (wording == 2) steps += "We calculate ";
        if (wording == 3) steps += "This gives ";
        if (wording == 4) steps += node.before + ' ';
        steps += expression + " = ";
        if (s.options.notes) steps += "<<" + expression + "=" + result + ">>";
        steps += result;
        if (wording == 2) steps += " for this step";
        if (wording == 3 && last) steps += " as the result";
        if (wording == 4) steps += ' ' + node.after;
        steps += ".";
    }
}

// One attempt. A failed operation discards the whole story, including its
// partial solution. Randomly chosen inputs are never silently adjusted.
inline bool candidate(Choices& rng, const WordOptions& options, int steps, WordProblem& out) {
    Story s(rng, options, out);
    bool made = false;
    switch (out.shape) {
        case WordProblem::chain: made = chain_story(s, steps); break;
        case WordProblem::comparison: made = comparison_story(s, steps); break;
        case WordProblem::total: made = total_story(s, steps); break;
        case WordProblem::rate: made = rate_story(s, steps); break;
        case WordProblem::part: made = part_story(s, steps); break;
        case WordProblem::change: made = change_story(s, steps); break;
        case WordProblem::shape_count: break;
    }
    if (!made) return false;
    add_distractor(s);
    for (const auto& sentence : s.sentences) out.problem.question += sentence + ' ';
    out.problem.question += s.question;
    out.problem.task = "word";
    out.problem.difficulty = steps;
    write_solution(s);
    out.problem.answer = number(out.values.back());
    return true;
}

}  // namespace word_detail

inline WordProblem make_word_problem(Rng& rng, const WordOptions& options = {}) {
    CHECK(options.min_steps >= 2 && options.min_steps <= options.max_steps && options.max_steps <= 8,
          "word problem steps must satisfy 2 <= min-steps <= max-steps <= 8");
    // Keep the requested length fixed across rejections: long stories should
    // not become rarer just because they have more chances to be invalid.
    const int steps = options.min_steps + int(rng.below(uint64_t(options.max_steps - options.min_steps + 1)));
    const auto shape = WordProblem::Shape(rng.below(WordProblem::shape_count));
    for (int attempt = 0; attempt < 10000; ++attempt) {
        WordProblem result;
        result.shape = shape;
        result.recipe = recipe_of("word", options.distractors, options.test, steps, int(shape), rng);
        Choices choices(rng);
        if (word_detail::candidate(choices, options, steps, result)) return result;
    }
    CHECK(false, "could not generate a valid word problem in 10000 attempts");
    return {};
}

// `recipes`, if given, gets each problem's recipe.
inline std::vector<Problem> make_word_problems(uint64_t seed, int64_t count, const WordOptions& options = {},
                                               std::vector<Recipe>* recipes = nullptr) {
    CHECK(count >= 0, "word problem count must be nonnegative");
    Rng rng(seed);
    std::vector<Problem> problems;
    std::set<std::string> questions;
    int duplicates = 0;
    while (problems.size() < uint64_t(count)) {
        WordProblem w = make_word_problem(rng, options);
        if (!questions.insert(w.problem.question).second) {
            CHECK(++duplicates < 10000, "too many duplicate word problems; request fewer problems");
            continue;
        }
        duplicates = 0;
        problems.push_back(std::move(w.problem));
        if (recipes) recipes->push_back(w.recipe);
    }
    return problems;
}
