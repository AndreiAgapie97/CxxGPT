// mathgen_fractions.h — gen-math --topic fractions: parts of an amount,
// percentages, ratios, rates and units, built the way src/mathgen.h builds
// equations: numbers drawn so that every step comes out exact, a worked
// solution with calculator notes, and a second solver (fraction_check) that
// shares no code with the generator and works each problem out again before
// it is written.
//
// Seven kinds of story:
//   part_of       "Maya has 240 apples. She sells 25% of them, then gives
//                  1/3 of the rest to Leo. How many are left?" (1-3 steps)
//   price_change  a price cut or raised by a percentage, once or twice
//   what_percent  "18 of 24 right: what percent?", or a price that moved
//   ratio_split   an amount shared 3:5 or 2:3:4, or a class of boys and girls
//   ratio_known   the ratio and one count given: the other count, or the total
//   proportion    "5 notebooks cost $7.50; how much do 8 cost?", recipes
//   units         minutes to hours, grams to kilograms, kilometers to
//                 meters, cents to dollars
// Every answer is a whole number or a decimal with at most two places, as
// in GSM8K: a story whose answer would need more is drawn again.
// Money and measurements are kept in hundredths (1250 is 12.5), so nothing
// is ever rounded.
#pragma once

#include "mathgen.h"

struct FractionProblem : MathRecord {
    enum Kind { part_of, price_change, what_percent, ratio_split, ratio_known, proportion, units, kind_count } kind =
        part_of;
};

namespace fraction_detail {

using math_detail::Fills;
using math_detail::Place;
using math_detail::Story;
using math_detail::str;

enum Family { part_open, part_first, part_rest, change_open, change_down, change_up, change_then_down,
              change_then_up, percent_score, percent_moved, ratio_money2, ratio_money3, ratio_class,
              ratio_known_text, proportion_price, proportion_recipe, units_time, units_weight, units_distance,
              units_cents, family_count };

// Six wordings per family with the same meaning: 0-3 train, 4-5 test.
// No sentence says "it" or "them" for something another sentence named, so
// a distractor can go between them; questions that say "that" keep the
// distractor away from just before them (Place::not_last).
inline constexpr std::array<std::array<const char*, 6>, family_count> patterns = {{
    {{"{name} has {n} {objects}.",
      "{name} starts the day with {n} {objects}.",
      "There are {n} {objects} in {name}'s collection.",
      "{name} owns a box of {n} {objects}.",
      "A jar on {name}'s desk holds {n} {objects}.",
      "{name} has collected {n} {objects} so far."}},
    {{"{name} gives {f} of the {objects} to {other}.",
      "{name} sells {f} of the {objects}.",
      "{name} uses {f} of the {objects} for a school project.",
      "{other} receives {f} of the {objects} from {name}.",
      "{name} trades away {f} of the {objects}.",
      "{name} donates {f} of the {objects} to a library sale."}},
    {{"Then {name} gives {f} of the remaining {objects} to {third}.",
      "Next, {name} sells {f} of the {objects} that are left.",
      "After that, {name} uses {f} of the remaining {objects}.",
      "{name} then loses {f} of the {objects} still left.",
      "Of the {objects} left at that point, {name} trades away {f}.",
      "{name} next donates {f} of whatever {objects} remain."}},
    {{"A {item} costs ${p}.",
      "The price of a {item} is ${p}.",
      "{name} wants a {item} that is priced at ${p}.",
      "At a store, a {item} sells for ${p}.",
      "A shop has a {item} with a price tag of ${p}.",
      "{name} looks at a {item} listed for ${p}."}},
    {{"The store takes {q}% off the price of the {item}.",
      "The {item} goes on sale for {q}% off.",
      "{name} gets a discount of {q}% on the {item}.",
      "The price of the {item} is cut by {q}%.",
      "A coupon lowers the price of the {item} by {q}%.",
      "During a sale, the price of the {item} drops by {q}%."}},
    {{"The price of the {item} goes up by {q}%.",
      "The store raises the price of the {item} by {q}%.",
      "A sales tax of {q}% is added to the price of the {item}.",
      "Because of demand, the price of the {item} rises {q}%.",
      "A price increase of {q}% is announced for the {item}.",
      "The shop marks the price of the {item} up by {q}%."}},
    {{"Later, the new price is cut by another {q}%.",
      "Then the store takes a further {q}% off the new price.",
      "After that, the new price drops by {q}%.",
      "A week later, the new price is lowered by {q}%.",
      "On top of that, {q}% is taken off the new price.",
      "The new price is then reduced by a further {q}%."}},
    {{"Then a sales tax of {q}% is added to the new price.",
      "After that, the new price goes up by {q}%.",
      "Later, the store raises the new price by {q}%.",
      "A month later, the new price increases by {q}%.",
      "On top of that, the new price is marked up by {q}%.",
      "The new price then rises by a further {q}%."}},
    {{"Out of {n} seeds that {name} plants, {k} sprout.",
      "{name} asks {n} people a question for a survey, and {k} of them say yes.",
      "Of the {n} students at {name}'s school, {k} walk to school.",
      "{name} takes {n} practice shots at a basket and makes {k} of them.",
      "Of the {n} letters {name} sent, {k} got a reply.",
      "{name} checks {n} eggs from a farm and finds that {k} of them are brown."}},
    {{"The price of a {item} goes from ${a} to ${b}.",
      "Last year a {item} cost ${a}; this year it costs ${b}.",
      "A {item} that used to sell for ${a} now sells for ${b}.",
      "{name} sees the price of a {item} change from ${a} to ${b}.",
      "A {item} was ${a} last month and is ${b} today.",
      "The store relabels a {item} from ${a} to ${b}."}},
    {{"{name} and {other} share ${t} in the ratio {a}:{b}.",
      "${t} is split between {name} and {other} in the ratio {a}:{b}.",
      "{name} and {other} divide ${t} so that their shares are in the ratio {a}:{b}.",
      "A prize of ${t} goes to {name} and {other} in the ratio {a}:{b}.",
      "{name} and {other} earn ${t} together and divide it {a}:{b}.",
      "Splitting ${t} in the ratio {a}:{b}, {name} and {other} each get a share."}},
    {{"{name}, {other} and {third} share ${t} in the ratio {a}:{b}:{c}.",
      "${t} is split among {name}, {other} and {third} in the ratio {a}:{b}:{c}.",
      "{name}, {other} and {third} divide ${t} in the ratio {a}:{b}:{c}.",
      "A prize of ${t} goes to {name}, {other} and {third} in the ratio {a}:{b}:{c}.",
      "{name}, {other} and {third} earn ${t} together and split it {a}:{b}:{c}.",
      "In the ratio {a}:{b}:{c}, {name}, {other} and {third} share out ${t}."}},
    {{"A class of {t} students has boys and girls in the ratio {a}:{b}.",
      "There are {t} students in a club, and the ratio of boys to girls is {a}:{b}.",
      "In a school choir of {t} children, boys and girls are in the ratio {a}:{b}.",
      "The ratio of boys to girls on a team of {t} players is {a}:{b}.",
      "Boys and girls in a group of {t} campers are in the ratio {a}:{b}.",
      "Of the {t} children on a trip, the ratio of boys to girls is {a}:{b}."}},
    {{"In {place}, the ratio of {first} to {second} is {a}:{b}. There are {k} {known}.",
      "The ratio of {first} to {second} in {place} is {a}:{b}, and there are {k} {known}.",
      "There are {k} {known} in {place}, where {first} and {second} are in the ratio {a}:{b}.",
      "{name} counts {k} {known} in {place}. The ratio of {first} to {second} there is {a}:{b}.",
      "In {place}, {first} and {second} come in the ratio {a} to {b}, and there are {k} {known}.",
      "{name} finds {k} {known} in {place}, with {first} and {second} in the ratio {a}:{b}."}},
    {{"{n} {items} cost ${p}.",
      "{name} pays ${p} for {n} {items}.",
      "At a shop, {n} {items} sell for ${p}.",
      "A pack of {n} {items} costs ${p}.",
      "The price of {n} {items} is ${p} in all.",
      "{name} spends ${p} on {n} {items}."}},
    {{"A recipe uses {c} cups of flour for {n} cookies.",
      "{name} needs {c} cups of flour to bake {n} cookies.",
      "To make {n} cookies, {name} uses {c} cups of flour.",
      "With {c} cups of flour, {name} can bake {n} cookies.",
      "Every batch of {n} cookies takes {c} cups of flour.",
      "{name}'s cookie recipe calls for {c} cups of flour per {n} cookies."}},
    {{"{name} practices the piano for {m} minutes every day for {span}.",
      "Every day for {span}, {name} reads for {m} minutes.",
      "{name} spends {m} minutes a day on homework for {span}.",
      "For {span}, {name} jogs {m} minutes each day.",
      "Each day of {span}, {name} walks the dog for {m} minutes.",
      "{name} plays chess for {m} minutes daily over {span}."}},
    {{"Each bag of rice weighs {g} grams, and {name} buys {n} bags.",
      "{name} packs {n} jars of jam that weigh {g} grams each.",
      "A box holds {n} cans, each weighing {g} grams.",
      "{name} has {n} potatoes that weigh {g} grams apiece.",
      "There are {n} jars of honey, at {g} grams per jar.",
      "{name} carries {n} books weighing {g} grams each."}},
    {{"{name} walks {k} kilometers every day for {d} days.",
      "Each day for {d} days, {name} cycles {k} kilometers.",
      "{name} runs {k} kilometers a day, {d} days in a row.",
      "For {d} days, {name} drives {k} kilometers each day.",
      "On each of {d} days, {name} skates {k} kilometers.",
      "{name} hikes {k} kilometers daily for {d} days."}},
    {{"{name} buys {n} stickers that cost {c} cents each.",
      "Each pencil costs {c} cents, and {name} buys {n} pencils.",
      "{name} sells {n} cookies for {c} cents apiece.",
      "A stamp costs {c} cents, and {name} needs {n} stamps.",
      "{name} pays {c} cents for each of {n} erasers.",
      "There are {n} candies at {c} cents per candy in {name}'s basket."}},
}};

// Hundredths as written in a calculation or an answer ("12.5"), and as
// money in a question ("12.50").
inline std::string number(int64_t hundredths) { return word_detail::number(hundredths); }
inline std::string money(int64_t hundredths) { return word_detail::money(hundredths); }

// "Maya has 240 apples. She sells 25% of them, then gives 1/3 of the rest
// to Leo." Each step takes a fraction (num/den) of what is left. The start
// is a multiple of the smallest number that keeps every step whole.
inline bool part_of_story(Story& s, std::string& answer, Place& place) {
    // Things a person collects, gives away, trades or loses by the hundred.
    static const std::array<const char*, 6> objects = {"marbles", "stickers", "buttons", "cards", "beads", "stamps"};
    static const std::array<int64_t, 7> denominators = {2, 3, 4, 5, 6, 8, 10};
    const std::string object = objects[s.rng.below(objects.size())];
    const int steps = 1 + int(s.rng.below(3));
    std::vector<std::pair<int64_t, int64_t>> parts;
    for (int i = 0; i < steps; ++i) {
        if (s.rng.below(2)) {
            parts.push_back({5 * s.draw(1, 19), 100});
            s.out.ops += '%';
        } else {
            const int64_t den = denominators[s.rng.below(denominators.size())], num = s.draw(1, den - 1);
            if (std::gcd(num, den) != 1) return false;  // 2/4 is written 1/2
            parts.push_back({num, den});
            s.out.ops += '/';
        }
    }
    // left = what is left as a fraction of the start, in lowest terms. Each
    // step takes left * num/den of the start, so the start must be a
    // multiple of that fraction's denominator, and of the final left's.
    int64_t left_num = 1, left_den = 1, multiple = 1;
    for (const auto& [num, den] : parts) {
        const int64_t taken_num = left_num * num, taken_den = left_den * den;
        multiple = std::lcm(multiple, taken_den / std::gcd(taken_num, taken_den));
        left_num *= den - num;
        left_den *= den;
        const int64_t g = std::gcd(left_num, left_den);
        left_num /= g;
        left_den /= g;
        multiple = std::lcm(multiple, left_den);
    }
    if (multiple > 300) return false;
    const int64_t start = multiple * s.draw(1, std::max<int64_t>(1, 600 / multiple));
    s.out.inputs = {start};
    s.sentence(part_open, {{"{n}", str(start)}, {"{objects}", object}});
    int64_t current = start;
    s.line(s.out.name + " starts with " + str(start) + " " + object + ".");
    for (size_t i = 0; i < parts.size(); ++i) {
        const auto [num, den] = parts[i];
        s.out.inputs.push_back(num);
        s.out.inputs.push_back(den);
        const bool percent = s.out.ops[i] == '%';
        const std::string f = percent ? str(num) + "%" : str(num) + "/" + str(den);
        s.sentence(i == 0 ? part_first : part_rest, {{"{f}", f}, {"{objects}", object}});
        const int64_t taken = current * num / den;
        const std::string expression = percent ? str(current) + "*" + str(num) + "/100"
                                       : num == 1 ? str(current) + "/" + str(den)
                                                  : str(current) + "*" + str(num) + "/" + str(den);
        s.line(f + " of " + str(current) + " is " + s.calc(expression, taken) + ", so " +
               s.calc(str(current) + "-" + str(taken), current - taken) + " are left.");
        current -= taken;
        if (current < 2) return false;  // "1 are left" and "1 stamps" read badly
    }
    s.question = "How many " + object + " does " + s.out.name + " have left?";
    s.line("So " + s.out.name + " has " + str(current) + " " + object + " left.");
    answer = str(current);
    place = Place::anywhere;
    return true;
}

// Things with a price, and the range a real one costs in whole dollars.
struct Priced { const char* name; int64_t low, high; };
inline constexpr std::array<Priced, 8> priced = {{{"jacket", 40, 300}, {"bike", 100, 600}, {"lamp", 20, 150},
                                                  {"backpack", 20, 120}, {"guitar", 80, 900}, {"tent", 60, 400},
                                                  {"phone", 200, 900}, {"sofa", 300, 1500}}};

// "12 dollars", "1 dollar".
inline std::string dollars(const std::string& n) { return math_detail::count_of(n, "dollar", "dollars"); }

// The unit after a number already written: " cups" after 2.5, " cup" after 1.
inline std::string unit(const std::string& n, const std::string& one) { return n == "1" ? " " + one : " " + one + "s"; }

// A price cut or raised by a percentage, once or twice; each change is on
// the price just before it. In cents, so every step must be whole cents.
inline bool price_change_story(Story& s, std::string& answer, Place& place) {
    const Priced& thing = priced[s.rng.below(priced.size())];
    const std::string item = thing.name;
    const int changes = 1 + int(s.rng.below(2));
    const int64_t price = 50 * s.draw(2 * thing.low, 2 * thing.high);  // in 50-cent steps
    s.out.inputs = {price};
    s.out.asked = changes == 1 ? int(s.rng.below(2)) : 0;
    s.sentence(change_open, {{"{item}", item}, {"${p}", "$" + money(price)}});
    int64_t current = price, delta = 0;
    for (int i = 0; i < changes; ++i) {
        const bool down = s.rng.below(2);
        const int64_t q = down ? 5 * s.draw(1, 10) : s.draw(2, 25);
        if (current * q % 100 != 0) return false;  // the change would not be whole cents
        delta = current * q / 100;
        const int64_t next = down ? current - delta : current + delta;
        s.out.inputs.push_back(q);
        s.out.ops += down ? 'd' : 'u';
        const Family family = i == 0 ? (down ? change_down : change_up) : (down ? change_then_down : change_then_up);
        s.sentence(family, {{"{q}", str(q)}, {"{item}", item}});
        s.line(std::string("The price ") + (down ? "drops" : "goes up") + " by " +
               s.calc(number(current) + "*" + str(q) + "/100", number(delta)) + unit(number(delta), "dollar") +
               ", so it is now " +
               s.calc(number(current) + (down ? "-" : "+") + number(delta), number(next)) + " dollars.");
        current = next;
    }
    const bool down = s.out.ops.back() == 'd';
    if (s.out.asked == 0) {
        s.question = "How much does the " + item + " cost now, in dollars?";
        s.line("So the " + item + " costs " + dollars(number(current)) + " now.");
        answer = number(current);
    } else {
        s.question = down ? "How many dollars cheaper is the " + item + " now?"
                          : "How many dollars more does the " + item + " cost now?";
        s.line("So the " + item + (down ? " is " + dollars(number(delta)) + " cheaper." :
                                          " costs " + dollars(number(delta)) + " more."));
        answer = number(delta);
    }
    place = Place::anywhere;
    return true;
}

// "18 of 24 seeds sprout: what percent?" (context 0), or a price that went
// from one whole number of dollars to another (context 1).
inline bool what_percent_story(Story& s, std::string& answer, Place& place) {
    const Priced& thing = priced[s.rng.below(priced.size())];
    const std::string item = thing.name;
    const int context = s.out.context = int(s.rng.below(2));
    if (context == 0) {
        const int64_t p = s.draw(1, 99), step = 100 / std::gcd(p, int64_t(100));
        const int64_t n = step * s.draw(1, std::max<int64_t>(1, 500 / step)), k = n * p / 100;
        if (n < 4 || k < 1) return false;
        s.out.inputs = {k, n};
        s.sentence(percent_score, {{"{k}", str(k)}, {"{n}", str(n)}});
        s.question = "What percent of the total is that?";
        s.line(str(k) + " out of " + str(n) + " is " + s.calc(str(k) + "/" + str(n), number(p)) + " of the total, and " +
               s.calc(number(p) + "*100", p) + ".");
        s.line("So it is " + str(p) + " percent.");
        answer = str(p);
        place = Place::not_last;
        return true;
    }
    // A whole-dollar old price in the thing's range that p% of is whole.
    const int64_t p = s.draw(1, 60), step = 100 / std::gcd(p, int64_t(100));
    const int64_t lowest = (thing.low + step - 1) / step, highest = thing.high / step;
    if (lowest > highest) return false;
    const int64_t old = step * s.draw(lowest, highest), change = old * p / 100;
    const bool up = s.rng.below(2);
    const int64_t now = up ? old + change : old - change;
    s.out.inputs = {old, now};
    s.sentence(percent_moved, {{"{item}", item}, {"${a}", "$" + str(old)}, {"${b}", "$" + str(now)}});
    s.question = std::string("By what percent did the price go ") + (up ? "up" : "down") + "?";
    s.line("The price changed by " + s.calc(up ? str(now) + "-" + str(old) : str(old) + "-" + str(now), change) +
           (change == 1 ? " dollar." : " dollars."));
    s.line("As a share of the old price, that is " + s.calc(str(change) + "/" + str(old) + "*100", p) + " percent.");
    s.line("So the price went " + std::string(up ? "up" : "down") + " by " + str(p) + " percent.");
    answer = str(p);
    place = Place::anywhere;
    return true;
}

// The parts of a ratio, with no common factor (2:4 is written 1:2).
inline bool ratio_parts(Story& s, int count, std::vector<int64_t>& parts) {
    int64_t g = 0;
    for (int i = 0; i < count; ++i) {
        parts.push_back(s.draw(1, 9));
        g = std::gcd(g, parts.back());
    }
    return g == 1 && !(count == 2 && parts[0] == parts[1]);
}

// An amount shared 3:5 by two people (context 0) or 2:3:4 by three
// (context 1), or a class of boys and girls 3:5 (context 2).
inline bool ratio_split_story(Story& s, std::string& answer, Place& place) {
    const int context = s.out.context = int(s.rng.below(3));
    const int count = context == 1 ? 3 : 2;
    std::vector<int64_t> parts;
    if (!ratio_parts(s, count, parts)) return false;
    int64_t sum = 0;
    for (int64_t part : parts) sum += part;
    const int64_t unit = context == 2 ? s.draw(2, 8) : s.draw(2, 60), total = unit * sum;
    s.out.inputs = {total};
    s.out.inputs.insert(s.out.inputs.end(), parts.begin(), parts.end());
    s.out.asked = int(s.rng.below(3));
    const std::array<std::string, 3> people = {s.out.name, s.out.other, s.out.third};
    const std::array<std::string, 2> groups = {"boys", "girls"};
    const std::string thing = context == 2 ? "" : " dollars";  // a class part is just a number of children
    Fills fills = {{"{a}", str(parts[0])}, {"{b}", str(parts[1])}, {"{t}", str(total)}};
    if (context == 1) fills.push_back({"{c}", str(parts[2])});
    s.sentence(context == 0 ? ratio_money2 : context == 1 ? ratio_money3 : ratio_class, fills);
    std::string sum_text = str(parts[0]);
    for (int i = 1; i < count; ++i) sum_text += "+" + str(parts[size_t(i)]);
    s.line("The ratio has " + s.calc(sum_text, sum) + " parts in all.");
    s.line("One part is " + s.calc(str(total) + "/" + str(sum), unit) + thing + ".");
    const bool difference = count == 2 && s.out.asked == 2;
    if (difference) {
        const size_t big = parts[1] > parts[0] ? 1 : 0, small = 1 - big;
        const int64_t gap = parts[big] - parts[small];
        s.question = context == 2 ? "How many more " + groups[big] + " than " + groups[small] + " are there?"
                                  : "How many more dollars does " + people[big] + " get than " + people[small] + "?";
        s.line("The difference is " + s.calc(str(parts[big]) + "-" + str(parts[small]), gap) + " parts, which is " +
               s.calc(str(gap) + "*" + str(unit), gap * unit) + thing + ".");
        answer = str(gap * unit);
    } else {
        const size_t who = size_t(s.out.asked);
        s.question = context == 2 ? "How many " + groups[who] + " are there?"
                                  : "How many dollars does " + people[who] + " get?";
        const std::string share = s.calc(str(parts[who]) + "*" + str(unit), parts[who] * unit);
        s.line(context == 2 ? "So there are " + share + " " + groups[who] + "."
                            : "So " + people[who] + " gets " + share + " dollars.");
        answer = str(parts[who] * unit);
    }
    place = Place::anywhere;
    return true;
}

// "The ratio of cats to dogs is 2:3, and there are 18 dogs." Asked: the
// other kind (0) or both together (1).
inline bool ratio_known_story(Story& s, std::string& answer, Place& place) {
    // Each place, and the most of both kinds together that it could hold.
    struct Setting { const char* first; const char* second; const char* place; int64_t most; };
    static const std::array<Setting, 4> settings = {{{"cats", "dogs", "the shelter", 90},
                                                     {"red marbles", "blue marbles", "the bag", 300},
                                                     {"oak trees", "pine trees", "the park", 800},
                                                     {"novels", "comic books", "the library", 1200}}};
    const Setting& setting = settings[s.rng.below(settings.size())];
    std::vector<int64_t> parts;
    if (!ratio_parts(s, 2, parts)) return false;
    const int64_t unit = s.draw(2, 60);
    if ((parts[0] + parts[1]) * unit > setting.most) return false;
    const size_t known = s.rng.below(2), unknown = 1 - known;
    const int64_t count = parts[known] * unit, other = parts[unknown] * unit;
    s.out.inputs = {parts[0], parts[1], int64_t(known), count};
    s.out.asked = int(s.rng.below(2));
    const std::string first = setting.first, second = setting.second;
    const std::array<std::string, 2> kinds = {first, second};
    s.sentence(ratio_known_text, {{"{place}", setting.place}, {"{first}", first}, {"{second}", second},
                                  {"{a}", str(parts[0])}, {"{b}", str(parts[1])}, {"{k}", str(count)},
                                  {"{known}", kinds[known]}});
    s.question = s.out.asked == 0 ? "How many " + kinds[unknown] + " are there?"
                                  : "How many " + first + " and " + second + " are there in all?";
    s.line("The " + kinds[known] + " are " + str(parts[known]) + " parts, so one part is " +
           s.calc(str(count) + "/" + str(parts[known]), unit) + ".");
    s.line("There are " + s.calc(str(parts[unknown]) + "*" + str(unit), other) + " " + kinds[unknown] + ".");
    if (s.out.asked == 0) {
        answer = str(other);
    } else {
        s.line("In all there are " + s.calc(str(count) + "+" + str(other), count + other) + " " + first + " and " +
               second + ".");
        answer = str(count + other);
    }
    place = Place::anywhere;
    return true;
}

// The same rate for another amount: a price for more or fewer items
// (context 0), or a recipe for more or fewer cookies (context 1).
inline bool proportion_story(Story& s, std::string& answer, Place& place) {
    // Things sold by the piece, and what one really costs, in cents.
    struct Piece { const char* name; int64_t low, high; };
    static const std::array<Piece, 6> pieces = {{{"notebooks", 100, 800}, {"pens", 50, 400}, {"mugs", 300, 1500},
                                                 {"candles", 200, 1500}, {"towels", 300, 2500},
                                                 {"plants", 300, 3000}}};
    const Piece& piece = pieces[s.rng.below(pieces.size())];
    const std::string item = piece.name;
    const int context = s.out.context = int(s.rng.below(2));
    place = Place::anywhere;
    if (context == 0) {
        const int64_t each = 5 * s.draw(piece.low / 5, piece.high / 5), n1 = s.draw(2, 12), n2 = s.draw(2, 30);
        if (n1 == n2) return false;
        const int64_t cost = each * n1;
        s.out.inputs = {n1, cost, n2};
        s.sentence(proportion_price, {{"{n}", str(n1)}, {"{items}", item}, {"${p}", "$" + money(cost)}});
        s.question = "How much do " + str(n2) + " " + item + " cost, in dollars?";
        s.line("One costs " + s.calc(number(cost) + "/" + str(n1), number(each)) + unit(number(each), "dollar") + ".");
        s.line("So " + str(n2) + " " + item + " cost " + s.calc(number(each) + "*" + str(n2), number(each * n2)) +
               unit(number(each * n2), "dollar") + ".");
        answer = number(each * n2);
        return true;
    }
    // A real recipe: half a cup to 4 cups of flour, 8 to 24 cookies a cup.
    const int64_t cups = 50 * s.draw(1, 8), n1 = s.draw(6, 96), n2 = s.draw(6, 120);
    if (n1 * 100 < 8 * cups || n1 * 100 > 24 * cups) return false;
    if (n1 == n2 || cups * n2 % n1 != 0) return false;  // the answer needs more than two decimals
    const int64_t needed = cups * n2 / n1;
    s.out.inputs = {n1, cups, n2};
    s.sentence(proportion_recipe, {{"{c}", number(cups)}, {"{n}", str(n1)}});
    s.question = "How many cups of flour are needed for " + str(n2) + " cookies?";
    s.line("The flour grows with the number of cookies, so " + str(n2) + " cookies need " +
           s.calc(number(cups) + "*" + str(n2) + "/" + str(n1), number(needed)) + unit(number(needed), "cup") + ".");
    answer = number(needed);
    return true;
}

// Minutes to hours (context 0), grams to kilograms (1), kilometers to
// meters (2), cents to dollars (3). Each wording's things have the sizes the
// ranges allow: 100-400 g fits a bag of rice, a jar or a potato; up to 8 km a
// day fits a walk, a run or a drive.
inline bool units_story(Story& s, std::string& answer, Place& place) {
    const int context = s.out.context = int(s.rng.below(4));
    place = Place::anywhere;
    if (context == 0) {
        const int64_t minutes = 5 * s.draw(1, 36);
        const bool weeks = s.rng.below(2);
        const int64_t count = weeks ? s.draw(1, 4) : s.draw(2, 60), days = weeks ? 7 * count : count;
        const int64_t total = minutes * days;
        if (total * 100 % 60 != 0) return false;  // hours with more than two decimals
        const int64_t hours = total * 100 / 60;
        s.out.inputs = {minutes, count, weeks ? 1 : 0};
        const std::string span = str(count) + (weeks ? (count == 1 ? " week" : " weeks") : " days");
        s.sentence(units_time, {{"{m}", str(minutes)}, {"{span}", span}});
        s.question = "How many hours does " + s.out.name + " spend in all?";
        if (weeks) s.line(span + " is " + s.calc(str(count) + "*7", days) + " days.");
        s.line("In all that is " + s.calc(str(minutes) + "*" + str(days), total) + " minutes, which is " +
               s.calc(str(total) + "/60", number(hours)) + unit(number(hours), "hour") + ".");
        answer = number(hours);
        return true;
    }
    if (context == 1) {
        const int64_t grams = 10 * s.draw(10, 40), count = s.draw(2, 80), total = grams * count;
        s.out.inputs = {grams, count, 0};
        s.sentence(units_weight, {{"{g}", str(grams)}, {"{n}", str(count)}});
        s.question = "How many kilograms is that in all?";
        s.line("In all that is " + s.calc(str(grams) + "*" + str(count), total) + " grams, which is " +
               s.calc(str(total) + "/1000", number(total / 10)) + unit(number(total / 10), "kilogram") + ".");
        answer = number(total / 10);
        place = Place::not_last;
        return true;
    }
    if (context == 2) {
        const int64_t km = 25 * s.draw(2, 32), days = s.draw(2, 30), total = km * days;  // km in hundredths
        s.out.inputs = {km, days, 0};
        s.sentence(units_distance, {{"{k}", number(km)}, {"{d}", str(days)}});
        s.question = "How many meters does " + s.out.name + " cover in all?";
        s.line("In all that is " + s.calc(number(km) + "*" + str(days), number(total)) + " kilometers, which is " +
               s.calc(number(total) + "*1000", total * 10) + " meters.");
        answer = str(total * 10);
        return true;
    }
    const int64_t cents = s.draw(5, 99), count = s.draw(2, 120), total = cents * count;
    s.out.inputs = {cents, count, 0};
    s.sentence(units_cents, {{"{c}", str(cents)}, {"{n}", str(count)}});
    s.question = "How many dollars is that in all?";
    s.line("In all that is " + s.calc(str(count) + "*" + str(cents), total) + " cents, which is " +
           s.calc(str(total) + "/100", number(total)) + unit(number(total), "dollar") + ".");
    answer = number(total);
    place = Place::not_last;
    return true;
}

// One attempt at a story of the given kind; false means draw again.
inline bool candidate(Rng& rng, const MathOptions& options, FractionProblem& out) {
    Story s(rng, options, out, patterns.data());
    std::string answer;
    Place place = Place::anywhere;
    bool made = false;
    switch (out.kind) {
        case FractionProblem::part_of: made = part_of_story(s, answer, place); break;
        case FractionProblem::price_change: made = price_change_story(s, answer, place); break;
        case FractionProblem::what_percent: made = what_percent_story(s, answer, place); break;
        case FractionProblem::ratio_split: made = ratio_split_story(s, answer, place); break;
        case FractionProblem::ratio_known: made = ratio_known_story(s, answer, place); break;
        case FractionProblem::proportion: made = proportion_story(s, answer, place); break;
        case FractionProblem::units: made = units_story(s, answer, place); break;
        case FractionProblem::kind_count: break;
    }
    if (!made) return false;
    math_detail::add_distractor(s, answer, place);
    math_detail::finish(s, answer, "fractions");
    return true;
}

}  // namespace fraction_detail

// The second solver. It reads only the story's numbers (inputs, ops,
// context) and what is asked, never the generator's solution or answer,
// and uses none of fraction_detail's code. Amounts are in hundredths where
// the story allows decimals. What inputs mean, per kind:
//
//   kind          inputs                                   asked
//   part_of       start, then num, den per step; ops has   (the count left)
//                 % or / per step
//   price_change  price (cents), then the percent per      0 final price,
//                 change; ops has d (down) or u (up)        1 the one change
//   what_percent  context 0: k right of n                  (the percent)
//                 context 1: old and new price (dollars)
//   ratio_split   total, then the ratio's parts            a part's owner, or
//                                                           2 the difference of two
//   ratio_known   a, b (the ratio), which part is known,   0 the other, 1 both
//                 and its count
//   proportion    n1, amount for n1 (hundredths), n2       (the amount for n2)
//   units         per item, count, weeks (time only)       (the total, converted)
namespace fraction_check {

// Hundredths written as a number: 1250 is "12.5", 1200 is "12".
inline std::string decimal(int64_t hundredths) {
    std::string s = std::to_string(hundredths / 100);
    const int64_t rest = hundredths % 100;
    if (rest == 0) return s;
    s += "." + std::to_string(rest / 10);
    if (rest % 10) s += std::to_string(rest % 10);
    return s;
}

// The one unit size u that makes `part * u == count` true, trying each in
// turn; none or several is an error.
inline std::string find_unit(int64_t part, int64_t count, int64_t& unit) {
    int found = 0;
    for (int64_t u = 1; u <= count; ++u)
        if (part * u == count) { ++found; unit = u; }
    return found == 1 ? "" : "the ratio gives " + std::to_string(found) + " whole part sizes, not one";
}

// The answer the story's numbers give, in hundredths, or what is wrong.
inline std::string solve(const FractionProblem& p, int64_t& hundredths) {
    const auto& in = p.inputs;
    switch (p.kind) {
        case FractionProblem::part_of: {
            // Every step must take a whole number of things; the count left is
            // the start times the product of what each step keeps.
            if (in.size() != 1 + 2 * p.ops.size() || p.ops.empty()) return "part_of record has the wrong size";
            int64_t count = in[0], keep_num = 1, keep_den = 1;
            for (size_t i = 0; i < p.ops.size(); ++i) {
                const int64_t num = in[1 + 2 * i], den = in[2 + 2 * i];
                if (num <= 0 || num >= den || (p.ops[i] == '%') != (den == 100)) return "a step is not a proper part";
                if (count * num % den != 0) return "a step takes a part that is not whole";
                count -= count * num / den;
                keep_num *= den - num;
                keep_den *= den;
            }
            if (in[0] * keep_num % keep_den != 0 || in[0] * keep_num / keep_den != count)
                return "the steps and the product of the parts disagree";
            hundredths = count * 100;
            return "";
        }
        case FractionProblem::price_change: {
            // In cents; each change must be whole cents.
            if (in.size() != 1 + p.ops.size() || p.ops.empty()) return "price record has the wrong size";
            int64_t cents = in[0], change = 0;
            for (size_t i = 0; i < p.ops.size(); ++i) {
                if (cents * in[1 + i] % 100 != 0) return "a change is not whole cents";
                change = cents * in[1 + i] / 100;
                cents += p.ops[i] == 'd' ? -change : change;
            }
            if (cents <= 0) return "the price is not positive";
            if (p.asked == 1 && p.ops.size() != 1) return "only a single change can be asked for";
            hundredths = p.asked == 0 ? cents : change;
            return "";
        }
        case FractionProblem::what_percent: {
            // Try every whole percent; exactly one must fit.
            if (in.size() != 2) return "percent record has the wrong size";
            const int64_t base = p.context == 0 ? in[1] : in[0];
            const int64_t part = p.context == 0 ? in[0] : (in[1] > in[0] ? in[1] - in[0] : in[0] - in[1]);
            int found = 0;
            for (int64_t percent = 0; percent <= 1000; ++percent)
                if (percent * base == 100 * part) { ++found; hundredths = percent * 100; }
            return found == 1 ? "" : "no whole percent fits";
        }
        case FractionProblem::ratio_split: {
            if (in.size() != size_t(p.context == 1 ? 4 : 3)) return "ratio record has the wrong size";
            int64_t sum = 0;
            for (size_t i = 1; i < in.size(); ++i) sum += in[i];
            int64_t unit = 0;
            const std::string wrong = find_unit(sum, in[0], unit);
            if (!wrong.empty()) return wrong;
            const bool difference = in.size() == 3 && p.asked == 2;
            if (difference) hundredths = (in[1] > in[2] ? in[1] - in[2] : in[2] - in[1]) * unit * 100;
            else if (size_t(p.asked) + 1 < in.size()) hundredths = in[1 + size_t(p.asked)] * unit * 100;
            else return "the asked part is not in the ratio";
            return "";
        }
        case FractionProblem::ratio_known: {
            if (in.size() != 4 || (in[2] != 0 && in[2] != 1)) return "ratio record is wrong";
            int64_t unit = 0;
            const std::string wrong = find_unit(in[size_t(in[2])], in[3], unit);
            if (!wrong.empty()) return wrong;
            const int64_t other = in[size_t(1 - in[2])] * unit;
            hundredths = (p.asked == 0 ? other : other + in[3]) * 100;
            return "";
        }
        case FractionProblem::proportion: {
            // amount2 / n2 = amount1 / n1, crosswise.
            if (in.size() != 3 || in[0] <= 0) return "proportion record is wrong";
            if (in[1] * in[2] % in[0] != 0) return "the amount needs more than two decimals";
            hundredths = in[1] * in[2] / in[0];
            return "";
        }
        case FractionProblem::units: {
            if (in.size() != 3) return "units record has the wrong size";
            if (p.context == 0) {
                const int64_t minutes = in[0] * in[1] * (in[2] ? 7 : 1);
                if (minutes * 100 % 60 != 0) return "the hours need more than two decimals";
                hundredths = minutes * 100 / 60;
            } else if (p.context == 1) {
                if (in[0] * in[1] * 100 % 1000 != 0) return "the kilograms need more than two decimals";
                hundredths = in[0] * in[1] * 100 / 1000;
            } else if (p.context == 2) {
                // in[0] is in hundredths of a kilometer, which is 10 meters:
                // in[0] * in[1] * 10 meters, written in hundredths.
                hundredths = in[0] * in[1] * 10 * 100;
            } else {
                hundredths = in[0] * in[1];  // cents are hundredths of a dollar
            }
            return "";
        }
        case FractionProblem::kind_count: return "no such kind";
    }
    return "no such kind";
}

// Everything the second solver checks on a finished problem (with notes):
// returns what is wrong, or an empty string.
inline std::string problems_with(const FractionProblem& p) {
    int64_t hundredths = 0;
    const std::string unsolved = solve(p, hundredths);
    if (!unsolved.empty()) return unsolved;
    if (hundredths <= 0) return "the answer is not positive";
    return math_check::answer_and_notes(p, decimal(hundredths));
}

}  // namespace fraction_check

// How often each kind is drawn, out of 18 (in Kind's order). what_percent
// has the fewest different problems (about 6,000), so it is drawn least.
inline constexpr std::array<int, FractionProblem::kind_count> fraction_weights = {4, 3, 1, 3, 2, 2, 3};

// One problem; every one is checked by fraction_check before it is returned.
inline FractionProblem make_fraction_problem(Rng& rng, const MathOptions& options = {}) {
    int total = 0;
    for (int w : fraction_weights) total += w;
    int pick = int(rng.below(uint64_t(total))), k = 0;
    while (pick >= fraction_weights[size_t(k)]) pick -= fraction_weights[size_t(k++)];
    const auto kind = FractionProblem::Kind(k);
    for (int attempt = 0; attempt < 10000; ++attempt) {
        FractionProblem result;
        result.kind = kind;
        if (!fraction_detail::candidate(rng, options, result)) continue;
        const std::string wrong = fraction_check::problems_with(result);
        CHECK(wrong.empty(), "the checker rejects a generated problem (%s): %s\n%s", wrong.c_str(),
              result.problem.question.c_str(), result.problem.steps.c_str());
        if (options.exclude && options.exclude->excludes(result.problem, result.distractor_number)) continue;
        if (!options.notes) result.problem.steps = math_detail::without_notes(result.problem.steps);
        return result;
    }
    CHECK(false, "could not generate a fractions problem in 10000 attempts");
    return {};
}

inline std::vector<Problem> make_fraction_problems(uint64_t seed, int64_t count, const MathOptions& options = {}) {
    return make_unique_problems(seed, count, [&](Rng& rng) { return make_fraction_problem(rng, options); });
}
