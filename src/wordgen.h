// wordgen.h — word problems made from sentence patterns and exact arithmetic.
// Values are integer hundredths: 125 means 1.25, never a rounded float.
// Counts must be whole. Money and measurements may have two decimal places;
// candidates needing finer precision are rejected, not rounded.
#pragma once

#include <array>
#include <set>

#include "reasoning.h"

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
    std::string label;
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
              extra_cost, family_count };

// Six distinct question patterns per family: four for train, two held out
// for test. Every question starts with a split-specific pattern, so even
// independently generated files with different seeds cannot share questions.
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
      "The cashier includes another purchase worth ${a}."}}
}};

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

// One attempt. A failed operation discards the whole story, including its
// partial solution. Randomly chosen inputs are never silently adjusted.
inline bool candidate(Rng& rng, const WordOptions& options, int steps, WordProblem& out) {
    static const std::array<const char*, 12> names = {
        "Alex", "Maya", "Noah", "Lina", "Sam", "Nora", "Leo", "Asha", "Ben", "Zoe", "Omar", "Iris"};
    static const std::array<const char*, 8> objects = {
        "stickers", "cards", "beads", "pencils", "candles", "toy cars", "bookmarks", "paper flowers"};
    const size_t name_index = rng.below(names.size());
    const std::string name = names[name_index];
    const std::string other = names[(name_index + 1) % names.size()];
    const std::string third = names[(name_index + 2) % names.size()];
    const std::string object = objects[rng.below(objects.size())];
    const int start = int(rng.below(4));
    enum class Kind { objects, money, liters, milliliters };
    Kind kind = start == earnings ? Kind::money : start == containers ? Kind::liters : Kind::objects;
    auto unit = [&]() -> std::string {
        if (kind == Kind::money) return "dollars";
        if (kind == Kind::liters) return "liters of juice";
        if (kind == Kind::milliliters) return "milliliters of juice";
        return object;
    };
    auto draw = [&](int lo, int hi) { return int64_t(lo + int(rng.below(uint64_t(hi - lo + 1)))); };
    static const std::array<const char*, 3> times = {"hour", "day", "week"};
    const std::string time = times[rng.below(times.size())];
    std::vector<std::string> sentences;
    auto sentence = [&](int family, int64_t a, int64_t b = 0, int64_t c = 0, int64_t d = 0) {
        const int variant = options.test ? 4 + int(rng.below(2)) : int(rng.below(4));
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
        replace(s, "{a}", kind == Kind::money && (family == receive || family == give) ? money(a) : number(a));
        replace(s, "{b}", number(b));
        replace(s, "{c}", number(c));
        replace(s, "{d}", number(d));
        sentences.push_back(s);
        out.patterns.push_back(family * 6 + variant);
    };
    auto literal = [&](int64_t value) {
        const size_t index = out.plan.size();
        out.plan.push_back({0, value, 0, 0, false, ""});
        out.values.push_back(value);
        return index;
    };
    auto operation = [&](char op, size_t left, size_t right, bool whole, const std::string& label) {
        int64_t value = 0;
        if (!calculate(op, out.values[left], out.values[right], whole, value)) return false;
        out.plan.push_back({op, 0, left, right, whole, label});
        out.values.push_back(value);
        return true;
    };
    std::string question;
    if (out.shape != WordProblem::chain) {
        kind = out.shape == WordProblem::change ? Kind::money : Kind::objects;
        int64_t a = draw(2, 20) * scale, b = draw(2, 12) * scale;
        int64_t c = draw(2, 30) * scale, d = draw(2, 30) * scale;
        int family = compare_people;
        if (out.shape == WordProblem::total) {
            out.three_people = steps >= 3 && rng.below(2);
            family = out.three_people ? total_three : total_two;
        }
        if (out.shape == WordProblem::rate) family = time_rate;
        if (out.shape == WordProblem::part) {
            out.percentage = steps >= 3 && rng.below(2);
            a = draw(3, 30) * 4 * scale;
            b = out.percentage ? draw(1, 3) * 25 * scale : draw(2, 4) * scale;
            family = out.percentage ? part_percent : part_fraction;
        }
        if (out.shape == WordProblem::change) {
            b = draw(1, 32) * 25;
            c = draw(5, 20) * 20 * scale;
            family = shopping;
        }
        out.inputs = {a, b, c, d};
        sentence(family, a, b, c, d);
        const size_t left = literal(a), right = literal(b);
        const char op = out.shape == WordProblem::part && !out.percentage ? '/' : '*';
        std::string label = "The first collection contains";
        if (out.shape == WordProblem::rate) label = "The work produces";
        if (out.shape == WordProblem::part) label = out.percentage ? "Multiplying by the percentage gives" : "The red items kept number";
        if (out.shape == WordProblem::change) label = "The items cost this many dollars";
        if (!operation(op, left, right, kind == Kind::objects, label)) return false;
        int done = 1;
        if (out.percentage) {
            const size_t product = out.values.size() - 1, hundred = literal(100 * scale);
            if (!operation('/', product, hundred, true, "The red items kept number")) return false;
            ++done;
        }
        const int closing_steps = out.three_people ? 2 : 1;
        while (done < steps - closing_steps) {
            const int64_t extra = out.shape == WordProblem::change ? draw(1, 32) * 25 : draw(1, 24) * scale;
            out.additions.push_back(extra);
            sentence(out.shape == WordProblem::change ? extra_cost : receive, extra);
            const size_t previous = out.values.size() - 1, operand = literal(extra);
            if (!operation('+', previous, operand, kind == Kind::objects,
                           out.shape == WordProblem::change ? "The bill now totals" : "Including the extra items gives")) return false;
            ++done;
        }
        const size_t current = out.values.size() - 1, separate = literal(c);
        if (out.shape == WordProblem::comparison) {
            if (!operation('-', current, separate, true, "The difference between the collections is")) return false;
            question = "How many more " + object + " does " + name + " have than " + other + "?";
        } else if (out.shape == WordProblem::change) {
            if (!operation('-', separate, current, false, "The change in dollars is")) return false;
            question = "How much change in dollars does " + name + " receive after all these purchases?";
        } else {
            if (!operation('+', current, separate, true, "Combining the two amounts gives")) return false;
            if (out.three_people) {
                const size_t subtotal = out.values.size() - 1, last = literal(d);
                if (!operation('+', subtotal, last, true, "Adding the third collection gives")) return false;
            }
            if (out.shape == WordProblem::part) question = "How many " + object + " does " + name + " keep in all?";
            else question = "How many " + object + " do " + name + (out.three_people ? ", " : " and ") + other +
                (out.three_people ? ", and " + third : "") + " have altogether?";
        }
    } else {
        int64_t a = draw(2, 20) * scale, b = draw(2, 12) * scale;
        if (kind == Kind::money) a += draw(0, 3) * 25;
        if (kind == Kind::liters) b = draw(1, 16) * 25;
        sentence(start, a, b);
        const size_t left = literal(a), right = literal(b);
        if (!operation('*', left, right, kind == Kind::objects, "The starting amount is")) return false;
        int done = 1;
        while (done < steps) {
            std::vector<int> choices = {receive, give, share, fraction};
            if (steps - done >= 2) choices.push_back(percent);
            if (kind == Kind::objects) choices.push_back(sell);
            if (kind == Kind::money) choices.push_back(buy);
            if (kind == Kind::liters) choices.push_back(convert);
            const int family = choices[rng.below(choices.size())];
            a = draw(1, 24) * scale;
            if (family == share || family == fraction) a = draw(2, 5) * scale;
            if (family == percent) a = draw(1, 3) * 25 * scale;
            if (family == sell || family == buy) a = draw(1, 32) * 25;
            if (family == convert) a = 1000 * scale;
            if ((family == receive || family == give) && kind != Kind::objects) {
                a += draw(0, 3) * 25;
                // Draw in liters first, then express the amount in the current unit.
                if (kind == Kind::milliliters) a *= 1000;
            }
            sentence(family, a);
            const size_t previous = out.values.size() - 1, operand = literal(a);
            char op = '+';
            std::string label = "After receiving more, the amount is";
            if (family == give) { op = '-'; label = "After giving some away, the amount is"; }
            if (family == share) { op = '/'; label = "One equal share is"; }
            if (family == fraction) { op = '/'; label = "The fraction kept is"; }
            if (family == percent) { op = '*'; label = "Multiplying by the percentage gives"; }
            if (family == sell) { op = '*'; label = "The sale brings in this many dollars"; kind = Kind::money; }
            if (family == buy) { op = '/'; label = "The number of items bought is"; kind = Kind::objects; }
            if (family == convert) { op = '*'; label = "The volume in milliliters is"; kind = Kind::milliliters; }
            if (!operation(op, previous, operand, kind == Kind::objects, label)) return false;
            ++done;
            if (family == percent) {
                const size_t product = out.values.size() - 1, hundred = literal(100 * scale);
                if (!operation('/', product, hundred, kind == Kind::objects, "Dividing by 100 gives the amount kept"))
                    return false;
                ++done;
            }
        }
        question = "How many " + unit() + " does " + name + " have now?";
    }
    // Draw even when disabled, so toggling distractors does not change the math.
    const int extra = int(rng.below(12)) + (options.test ? 12 : 0);
    int64_t irrelevant = draw(2, 30) * scale;
    // Avoid even accidental reuse of a distractor number in the solution.
    while (std::find(out.values.begin(), out.values.end(), irrelevant) != out.values.end()) irrelevant += scale;
    const size_t position = rng.below(sentences.size() + 1);
    if (options.distractors) {
        std::string s = distractors[extra];
        out.distractor = extra;
        out.distractor_position = int(position);
        if (s.find("{a}") != std::string::npos) out.distractor_number = irrelevant;
        replace(s, "{a}", number(irrelevant));
        sentences.insert(sentences.begin() + position, s);
    }
    for (const auto& s : sentences) out.problem.question += s + ' ';
    out.problem.question += question;
    out.problem.task = "word";
    out.problem.difficulty = steps;
    for (size_t i = 0; i < out.plan.size(); ++i) {
        const WordNode& node = out.plan[i];
        if (!node.op) continue;
        const std::string expression = number(out.values[node.left]) + node.op + number(out.values[node.right]);
        const std::string result = number(out.values[i]);
        if (!out.problem.steps.empty()) out.problem.steps += '\n';
        const int wording = int(rng.below(4));
        if (wording == 0) out.problem.steps += node.label + ": ";
        if (wording == 1) out.problem.steps += node.label + ' ';
        if (wording == 2) out.problem.steps += "We calculate ";
        if (wording == 3) out.problem.steps += "This gives ";
        out.problem.steps += expression + " = ";
        if (options.notes) out.problem.steps += "<<" + expression + "=" + result + ">>";
        out.problem.steps += result;
        if (wording == 2) out.problem.steps += " for this step";
        if (wording == 3) out.problem.steps += " as the result";
        out.problem.steps += ".";
    }
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
        if (word_detail::candidate(rng, options, steps, result)) return result;
    }
    CHECK(false, "could not generate a valid word problem in 10000 attempts");
    return {};
}

inline std::vector<Problem> make_word_problems(uint64_t seed, long count, const WordOptions& options = {}) {
    CHECK(count >= 0, "word problem count must be nonnegative");
    Rng rng(seed);
    std::vector<Problem> problems;
    std::set<std::string> questions;
    int duplicates = 0;
    while (problems.size() < uint64_t(count)) {
        Problem p = make_word_problem(rng, options).problem;
        if (!questions.insert(p.question).second) {
            CHECK(++duplicates < 10000, "too many duplicate word problems; request fewer problems");
            continue;
        }
        duplicates = 0;
        problems.push_back(std::move(p));
    }
    return problems;
}
