// mathgen_number_theory.h — gen-math --topic number-theory: remainders,
// least common multiples, greatest common divisors, factors, multiples,
// digits and repeating patterns, built the way src/mathgen.h builds
// equations: whole numbers drawn so that the story works, a worked solution
// with calculator notes, and a second solver (number_theory_check) that
// shares no code with the generator and works each problem out again,
// by brute force, before it is written.
//
// Seven kinds of story:
//   remainder  "Maya packs 50 cookies into boxes of 6": full boxes, cookies
//              left over, boxes needed for all, cookies to fill one more;
//              or people in vans, tables or tents
//   lcm        two buses (lights, visits) that repeat every a and b
//              minutes: when do they next coincide, or how many times in
//              the next few hours? Or hot dogs in packs
//              of 8 and buns in packs of 6: the fewest packs for equal
//              numbers
//   gcd        identical gift bags from 24 pencils and 36 erasers, or two
//              ribbons cut into the longest equal pieces
//   factors    36 chairs in equal rows (or students in equal teams), at
//              least 2 (or 3) rows of at least 2 (3): how many numbers of
//              rows?
//   multiples  raffle tickets 1 to 200 that are multiples of 7; every 5th
//              customer; every 4th and every 6th person (both); the
//              multiples of 7 from 45 to 180; even-numbered lockers
//   digits     the digits needed to number a book's pages, or the pages
//              that a number of digits numbers; a two-digit number from its
//              digit sum and difference
//   pattern    beads in a repeating pattern (3 red, 2 blue); work days and
//              days off
//
// The calculator knows only + - * / and parentheses, so a whole-number
// quotient is written as two products around the number: "7*28 =
// <<7*28=196>>196 and 7*29 = <<7*29=203>>203 is more than 200, so 28
// tickets win." A least common multiple is found by counting up by the
// larger number, a greatest common divisor from the smaller number's
// factors, as a pupil would.
//
// Against a templated look, the questions and the main solution lines
// have three wordings each (one held out for the test split), and the
// topic has its own fillers, in half the problems.
#pragma once

#include "mathgen.h"

struct NumberTheoryProblem : MathRecord {
    enum Kind { remainder, lcm, gcd, factors, multiples, digits, pattern, kind_count } kind = remainder;
};

namespace number_detail {

using math_detail::Fills;
using math_detail::Place;
using math_detail::Story;
using math_detail::str;

enum Family { rem_pack, rem_seat, lcm_bus, lcm_light, lcm_visit, lcm_packs, gcd_kits, gcd_ribbon, factor_rows,
              factor_teams, mult_tickets, mult_every, mult_both, mult_range, parity_lockers, digit_pages,
              digit_total, digit_puzzle, pattern_beads, pattern_shifts, family_count };

// Six wordings per family with the same meaning: 0-3 train, 4-5 test.
// Each family is one sentence, or two that do not say "it" for something
// the other named, so a distractor can go before or after.
inline constexpr std::array<std::array<const char*, 6>, family_count> patterns = {{
    {{"{name} has {n} {things} and puts them into {holders} that hold {k} {things} each.",
      "{name} needs to pack {n} {things} into {holders}, with {k} {things} in each {holder}.",
      "There are {n} {things} for {name} to pack into {holders} of {k}.",
      "{name} fills {holders} with {k} {things} each from a pile of {n} {things}.",
      "A {holder} holds {k} {things}, and {name} has {n} {things} to pack.",
      "{name} is packing {n} {things}; every {holder} takes exactly {k} {things}."}},
    {{"There are {n} {things}, and each {holder} has room for {k} {things}.",
      "{name} has to find places for {n} {things}; a {holder} has room for {k} {things}.",
      "Each {holder} can take {k} {things}, and {n} {things} need a place.",
      "{name} counts {n} {things}, and every {holder} has {k} places.",
      "A {holder} has {k} places, and {name} must find a place for {n} {things}.",
      "With {k} places in every {holder}, {name} has {n} {things} to place."}},
    {{"Two buses leave the station at the same time. One bus leaves every {a} minutes and the other every {b} minutes.",
      "At a bus station, the red bus leaves every {a} minutes and the blue bus leaves every {b} minutes, and both have just left together.",
      "{name} sees two buses leave together; the first runs every {a} minutes and the second every {b} minutes.",
      "One bus route has a bus every {a} minutes and another has one every {b} minutes, and buses on both routes just left at the same moment.",
      "Buses on the north route leave every {a} minutes and buses on the south route every {b} minutes; one of each has just left together.",
      "Two buses pull out of the depot together, and after that one leaves every {a} minutes and the other every {b} minutes."}},
    {{"Two lights blink together. One blinks every {a} seconds and the other every {b} seconds.",
      "A red light flashes every {a} seconds and a green light flashes every {b} seconds, and they have just flashed at the same moment.",
      "{name} watches two lamps that have just blinked at the same time: one blinks every {a} seconds, the other every {b} seconds.",
      "On a tower, one light flashes every {a} seconds and another every {b} seconds, and both just flashed together.",
      "Two signal lights flash at the same instant; afterwards, one flashes every {a} seconds and the other every {b} seconds.",
      "A yellow lamp blinks every {a} seconds and a blue lamp every {b} seconds, and the two have just blinked together."}},
    {{"{name} goes to the library every {a} days and {other} goes every {b} days, and both went today.",
      "{name} visits the pool every {a} days, and {other} visits the pool every {b} days; they were both there today.",
      "Today {name} and {other} both went to the park, where {name} goes every {a} days and {other} every {b} days.",
      "{name} waters the plants every {a} days and {other} waters the plants every {b} days; both watered them today.",
      "{name} calls Grandma every {a} days and {other} calls her every {b} days, and both called today.",
      "{name} cleans the fish tank every {a} days and {other} cleans it every {b} days, and both cleaned it today."}},
    {{"{first} come in packs of {a} and {second} come in packs of {b}. {name} wants to buy exactly as many {first} as {second}.",
      "A store sells {first} in packs of {a} and {second} in packs of {b}, and {name} wants the same number of each.",
      "{name} needs the same number of {first} and {second}; {first} are sold {a} to a pack and {second} {b} to a pack.",
      "Packs of {first} hold {a} each and packs of {second} hold {b} each, and {name} wants equal numbers of the two.",
      "{name} buys {first} in packs of {a} and {second} in packs of {b}, and wants the same number of each.",
      "With {first} sold only in packs of {a} and {second} only in packs of {b}, {name} wants an equal number of both."}},
    {{"{name} has {a} {first} and {b} {second} and wants to put all of them into identical bags, with nothing left over.",
      "{name} is making identical bags from {a} {first} and {b} {second}, with the same number of each in every bag and nothing left over.",
      "To make identical gift bags, {name} shares out {a} {first} and {b} {second} so that every bag gets the same mix and none are left over.",
      "{name} splits {a} {first} and {b} {second} into identical bags, using all of them.",
      "With {a} {first} and {b} {second}, {name} fills identical bags so that every one is used and each bag holds the same mix.",
      "{name} wants to pack {a} {first} and {b} {second} into bags that are all exactly alike, leaving nothing out."}},
    {{"{name} has two ribbons, one {a} centimeters long and the other {b} centimeters long, and wants to cut both into pieces of the same length, as long as possible, with nothing wasted.",
      "Two ropes are {a} centimeters and {b} centimeters long, and {name} cuts both ropes into equal pieces, each as long as possible, without any rope left over.",
      "{name} must cut a {a}-centimeter board and a {b}-centimeter board into pieces of one length, as long as possible, using all of the wood.",
      "A {a}-centimeter strip of cloth and a {b}-centimeter strip of cloth are cut into equal pieces, each as long as possible, with nothing left over.",
      "{name} cuts a {a}-centimeter ribbon and a {b}-centimeter ribbon into equal lengths, the longest possible, wasting none.",
      "Two wires measure {a} centimeters and {b} centimeters, and all of both wires is cut into pieces of one length, as long as it can be."}},
    {{"{name} sets out {n} chairs in rows. Every row has the same number of chairs, there are at least {m} rows, and every row has at least {m} chairs.",
      "{n} chairs go into equal rows for a show, with at least {m} rows and at least {m} chairs in each row.",
      "{name} wants to arrange {n} chairs in rows of equal length, using at least {m} rows of at least {m} chairs.",
      "For a meeting, {name} must put all {n} chairs into rows with the same number of chairs; a row needs at least {m} chairs, and there must be at least {m} rows.",
      "All {n} chairs in a hall are set in equal rows: no fewer than {m} rows, and no fewer than {m} chairs in a row.",
      "{name} lines up {n} chairs in rows of the same size, with {m} or more rows and {m} or more chairs per row."}},
    {{"{n} students split into teams of equal size, with at least {m} teams and at least {m} students on each team.",
      "Coach {name} divides {n} players into equal teams; there must be at least {m} teams, each with at least {m} players.",
      "{name} must put all {n} campers into teams of the same size: at least {m} teams of at least {m} campers.",
      "At a sports day, {n} children are split into equal teams, at least {m} teams with {m} or more children per team.",
      "{name} sorts {n} players into teams that are all the same size, with no fewer than {m} teams and no fewer than {m} players per team.",
      "All {n} students in a class form equal teams: there are {m} or more teams, and each team has {m} or more students."}},
    {{"At a fair, raffle tickets are numbered from 1 to {n}. Every ticket whose number is a multiple of {k} wins a prize.",
      "{name} sells raffle tickets numbered 1 to {n}, and each ticket with a number that {k} divides exactly wins.",
      "Tickets numbered 1 through {n} go into a box; a ticket wins if its number is a multiple of {k}.",
      "In a school raffle, the tickets carry the numbers 1 to {n}, and the winning tickets are the multiples of {k}.",
      "A raffle uses tickets from 1 to {n}, and every multiple of {k} is a winning number.",
      "{name} prints tickets with the numbers 1 to {n}; the ones whose numbers are multiples of {k} win a small prize."}},
    {{"A bakery gives every {kth} customer a free {gift}, and {n} customers come in today.",
      "Every {kth} visitor to a museum gets a free {gift}, and {n} visitors come on Saturday.",
      "{name} hands a {gift} to every {kth} person who enters a shop, and {n} people enter in the morning.",
      "At a game, every {kth} fan through the gate receives a {gift}; {n} fans go through the gate.",
      "A cafe rewards every {kth} customer with a {gift}, and {n} customers order today.",
      "On opening day, a store gives a {gift} to every {kth} shopper, and {n} shoppers come."}},
    {{"At a concert, every {ath} person through the door gets a poster and every {bth} person gets a sticker, and {n} people come.",
      "{name} gives every {ath} customer a coupon and every {bth} customer a balloon, and {n} customers visit the shop today.",
      "In a race with {n} runners, every {ath} runner to finish gets a cap and every {bth} runner to finish gets a medal.",
      "A school lines up {n} students; every {ath} student gets a red card and every {bth} student gets a blue card.",
      "Every {ath} guest at a party wins a candle and every {bth} guest wins a mug, and {n} guests arrive.",
      "{n} visitors come to a zoo, and every {ath} visitor gets a map while every {bth} visitor gets a badge."}},
    {{"{name} reads pages {a} to {b} of a book, and every page whose number is a multiple of {k} has a picture.",
      "Seats {a} through {b} in a theater are sold, and every seat with a number that is a multiple of {k} gets a free drink.",
      "{name} writes the numbers from {a} to {b} and circles each number that is a multiple of {k}.",
      "The houses on a road are numbered {a} to {b}, and the ones whose numbers are multiples of {k} get a new mailbox.",
      "A teacher counts from {a} to {b} and claps on every multiple of {k}.",
      "Of the boxes numbered {a} to {b} in a warehouse, those numbered with a multiple of {k} are checked."}},
    {{"The lockers in a school hall are numbered from {a} to {b}, and the lockers with {parity} numbers are painted blue.",
      "A row of lockers is numbered {a} through {b}, and {name} has to clean every {parity}-numbered locker.",
      "{name} checks the {parity}-numbered lockers in a hall where the lockers go from {a} to {b}.",
      "In a gym, lockers {a} to {b} are in use, and the {parity}-numbered ones belong to the swim team.",
      "The lockers on one floor are numbered {a} up to {b}, and each {parity}-numbered locker gets a new lock.",
      "{name} paints the {parity}-numbered lockers among those numbered {a} to {b}."}},
    {{"A book has {n} pages, numbered from 1 to {n}.",
      "{name} numbers the pages of a {n}-page notebook by hand, starting at 1.",
      "The pages of a report are numbered 1 to {n}, one number on each page.",
      "{name} types a page number on each of the {n} pages of a story, from 1 up.",
      "Every page of a {n}-page book carries its number, starting from 1.",
      "{name} stamps the numbers 1 to {n} on the pages of a photo album."}},
    {{"{name} writes the page numbers of a book by hand, starting at 1, and writes {d} digits in all.",
      "Numbering the pages of a report, starting at 1, takes {d} digits in all.",
      "{name} counts {d} digits in all the page numbers of a notebook, which start at 1.",
      "The page numbers of a photo album, starting from 1, use {d} digits altogether.",
      "To number every page of a book from 1, a printer needs {d} digits.",
      "All the page numbers of {name}'s diary, from 1 up, take {d} digits."}},
    {{"{name} thinks of a two-digit number. Its digits add up to {s}, and {relation}.",
      "A two-digit number has digits that add up to {s}, and {relation}.",
      "The two digits of a two-digit number add up to {s}; {relation}.",
      "{name}'s locker code is a two-digit number whose digits have a sum of {s}, and {relation}.",
      "In a two-digit number, the sum of the digits is {s} and {relation}.",
      "{name} hides a two-digit number: {relation}, and the two digits add up to {s}."}},
    {{"{name} strings {n} beads in a repeating pattern: {r} {first} beads, then {b} {second} beads, then {r} {first} beads again, and so on.",
      "A necklace has {n} beads; starting from the clasp, the beads go {r} {first}, {b} {second}, and then the pattern repeats.",
      "{name} threads beads in the order {r} {first} then {b} {second}, over and over, until there are {n} beads.",
      "In a bracelet of {n} beads, a group of {r} {first} beads is always followed by {b} {second} beads, starting with {first}.",
      "The {n} beads on {name}'s string repeat the pattern of {r} {first} beads followed by {b} {second} beads.",
      "{name} makes a chain of {n} beads by repeating {r} {first} and {b} {second}, always in that order."}},
    {{"{name} works {w} days in a row and then has {r} days off, and keeps to this pattern from day 1.",
      "Starting on day 1, {name}'s schedule is {w} days of work followed by {r} days of rest, repeated.",
      "{name} works {w} days and then rests {r} days, again and again, beginning on day 1.",
      "{name}'s job runs in a cycle of {w} work days followed by {r} days off, starting on day 1.",
      "From day 1, {name} follows a routine: {w} days at work, then {r} days at home, then the same again.",
      "{name} keeps a pattern of working {w} days and resting {r} days, which starts on day 1."}},
}};

// This topic's own irrelevant sentences (fillers), 0-11 for train and
// 12-23 for test, in place of gen-word's shared ones: about the story's
// people, half of them with a number ({a}) that the question does not
// need. Half the problems get one (number_detail::candidate).
inline constexpr std::array<const char*, 24> fillers = {
    "{name} is {a} years old.",
    "{other} has {a} cousins who live in another town.",
    "{name}'s favorite color is blue.",
    "It is a sunny day outside.",
    "{other} read {a} pages of a novel before lunch.",
    "{name} has a dog named Biscuit.",
    "{name} lives {a} blocks from the school.",
    "{other} is wearing a red hat.",
    "{name}'s house has {a} steps up to the front door.",
    "Everyone is in a good mood.",
    "{other} practices the piano for {a} minutes every evening.",
    "{name} hums a song while working.",
    "{other} turned {a} last month.",
    "{name} owns {a} houseplants.",
    "The weather is cool and windy.",
    "{other} is drinking a glass of orange juice.",
    "{name} once ran {a} laps around the park.",
    "{other} keeps {a} postcards in a drawer.",
    "{name} is saving up for a new bicycle.",
    "{other} has a cat that sleeps most of the day.",
    "{name}'s street has {a} trees along it.",
    "{other} likes to knit on weekends.",
    "The radio is playing an old song.",
    "{name} has visited {a} different countries."};

// Questions (q_) and solution lines (s_), three wordings each with the
// same meaning: 0-1 train, 2 test. Like the story sentences, none says
// "it" or "they" for something a sentence before named.
enum Phrase { q_pack_full, q_pack_left, q_pack_need, q_pack_more, q_seat_full, q_seat_left, q_seat_need, q_seat_empty,
              q_bus_when, q_light_when, q_visit_when, q_bus_times, q_light_times, q_visit_times, q_packs, q_packs_count,
              q_bags, q_bag_each, q_piece_length, q_pieces, q_rows, q_teams, q_tickets, q_every, q_both, q_range,
              q_lockers, q_digits, q_pages, q_number, q_beads, q_days, s_fits, s_count_up, s_first_both, s_factors,
              s_largest, phrase_count };

inline constexpr std::array<std::array<const char*, 3>, phrase_count> phrases = {{
    {{"How many {holders} can {name} fill completely?",
      "How many full {holders} can {name} make?",
      "How many {holders} will {name} be able to fill all the way?"}},
    {{"How many {things} are left over after {name} fills as many {holders} as possible?",
      "After {name} fills as many {holders} as possible, how many {things} are left?",
      "How many {things} remain once {name} has filled every {holder} that can be filled?"}},
    {{"How many {holders} does {name} need for all the {things}?",
      "How many {holders} must {name} use to pack every one of the {things}?",
      "What is the smallest number of {holders} that can hold all the {things}?"}},
    {{"After filling as many {holders} as possible, how many more {things} would {name} need to fill one more {holder}?",
      "How many more {things} would {name} need so that one more {holder} is full too?",
      "To fill one more {holder} after the full ones, how many more {things} does {name} need?"}},
    {{"How many {holders} can be filled completely?",
      "How many {holders} will be full?",
      "How many {holders} can be completely filled?"}},
    {{"If as many {holders} as possible are filled, how many {things} are left over?",
      "How many {things} are left without a place once as many {holders} as possible are full?",
      "After every {holder} that can be filled is full, how many {things} are still waiting?"}},
    {{"How many {holders} are needed so that all the {things} have a place?",
      "How many {holders} are needed for all the {things}?",
      "What is the fewest {holders} that give every one of the {things} a place?"}},
    {{"If every {holder} but the last is full, how many empty places are in the last {holder}?",
      "With every {holder} full except the last one, how many places in the last {holder} are empty?",
      "How many places are empty in the last {holder} if all the other {holders} are full?"}},
    {{"How many minutes from now will both buses next leave at the same time?",
      "In how many minutes will the two buses next leave together?",
      "After how many minutes do both buses leave together again?"}},
    {{"After how many seconds will both lights next flash at the same time?",
      "In how many seconds will the two lights next flash together?",
      "How many seconds pass before both lights flash together again?"}},
    {{"In how many days will both of them next do this on the same day?",
      "How many days from today until both of them next do this on the same day?",
      "After how many days will {name} and {other} next do this on the same day?"}},
    {{"How many times in the next {span} hours will both buses leave at the same time?",
      "In the next {span} hours, how many times will the two buses leave together?",
      "Within the next {span} hours, how many more times do both buses leave together?"}},
    {{"How many times in the next {span} minutes will both lights flash at the same time?",
      "In the next {span} minutes, how many times will the two lights flash together?",
      "Within the next {span} minutes, how many more times do both lights flash together?"}},
    {{"How many times in the next {span} days will both of them do this on the same day?",
      "In the next {span} days, on how many days will both of them do this?",
      "Within the next {span} days, how many more times will {name} and {other} do this on the same day?"}},
    {{"What is the smallest number of packs of {item} that {name} can buy?",
      "What is the fewest packs of {item} {name} can buy?",
      "At least how many packs of {item} does {name} have to buy?"}},
    {{"Buying whole packs, what is the smallest number of {first} that {name} can end up with?",
      "If {name} buys whole packs, what is the fewest {first} {name} can end up with?",
      "What is the smallest number of {first} {name} can have when buying whole packs?"}},
    {{"What is the greatest number of bags {name} can make?",
      "How many bags can {name} make at most?",
      "What is the largest number of identical bags {name} can fill?"}},
    {{"If {name} makes as many bags as possible, how many {item} go in each bag?",
      "How many {item} are in each bag when {name} makes as many bags as possible?",
      "With the greatest number of bags, how many {item} does each bag get?"}},
    {{"How long is each piece, in centimeters?",
      "How many centimeters long is each piece?",
      "What is the length of each piece in centimeters?"}},
    {{"How many pieces are there in all?",
      "How many pieces are there altogether?",
      "What is the total number of pieces?"}},
    {{"How many different numbers of rows are possible?",
      "How many different row counts can be chosen?",
      "How many choices are there for the number of rows?"}},
    {{"How many different numbers of teams are possible?",
      "How many different team counts can be chosen?",
      "How many choices are there for the number of teams?"}},
    {{"How many winning tickets are there?",
      "How many tickets win?",
      "How many of the tickets are winners?"}},
    {{"How many people get a {gift}?",
      "How many {gift}s are given out?",
      "How many people receive a {gift}?"}},
    {{"How many people get both?",
      "How many people end up with both?",
      "How many people receive both?"}},
    {{"How many of the numbers from {a} to {b} are multiples of {k}?",
      "How many multiples of {k} are there from {a} to {b}?",
      "Counting both ends, how many numbers between {a} and {b} are multiples of {k}?"}},
    {{"How many {parity}-numbered lockers are there?",
      "How many lockers have {parity} numbers?",
      "What is the number of {parity}-numbered lockers?"}},
    {{"How many digits are written in all to number the pages?",
      "How many digits are needed in all for the page numbers?",
      "How many digits do all the page numbers use together?"}},
    {{"How many pages are there?",
      "How many pages are numbered?",
      "What is the number of pages?"}},
    {{"What is the two-digit number?",
      "What is the number?",
      "Which two-digit number is it?"}},
    {{"How many {asked} are there?",
      "How many {asked} are there in all?",
      "In total, how many {asked} are there?"}},
    {{"How many of the first {n} days are {asked}?",
      "Of the first {n} days, how many are {asked}?",
      "How many {asked} are there in the first {n} days?"}},
    {{"{p1} and {p2} is more than {n}",
      "{p1}, while {p2} is already more than {n}",
      "{p1}, but {p2} would be more than {n}"}},
    {{"Counting up by {big}: {list}.",
      "The multiples of {big} are {list}.",
      "Listing the multiples of {big}: {list}."}},
    {{"The first of these that {small} also goes into is {both}: {p1} and {p2}.",
      "Of these, {both} is the first that {small} also divides: {p1} and {p2}.",
      "The first one that {small} divides evenly is {both}: {p1} and {p2}."}},
    {{"The factors of {small} are {list}.",
      "The numbers that divide {small} evenly are {list}.",
      "{small} can be divided evenly by {list}."}},
    {{"The largest of these that also divides {big} is {g}: {p1} and {p2}.",
      "Of these, the largest that also divides {big} is {g}: {p1} and {p2}.",
      "The biggest one that goes into {big} as well is {g}: {p1} and {p2}."}},
}};

// One of a phrase's split's wordings, filled in. Its ID goes with the
// sentence IDs (after them, wording 2 counted as 4 so that test wordings
// are 4 and 5 everywhere).
inline std::string say(Story& s, Phrase phrase, const Fills& fills) {
    const int variant = s.options.test ? 2 : int(s.rng.below(2));
    s.out.patterns.push_back((int(family_count) + int(phrase)) * 6 + (variant == 2 ? 4 : variant));
    return s.fill(phrases[size_t(phrase)][size_t(variant)], fills);
}

// "3rd", "11th", "21st": every k-th, as a person writes it.
inline std::string ordinal(int64_t n) {
    const int64_t last = n % 10, tens = n % 100;
    const char* suffix = tens >= 11 && tens <= 13 ? "th" : last == 1 ? "st" : last == 2 ? "nd" : last == 3 ? "rd" : "th";
    return str(n) + suffix;
}

// "1, 2, 3 and 6".
inline std::string listing(const std::vector<int64_t>& values) {
    std::string out;
    for (size_t i = 0; i < values.size(); ++i)
        out += (i == 0 ? "" : i + 1 == values.size() ? " and " : ", ") + str(values[i]);
    return out;
}

// How many whole times k goes into n, written as two products around n:
// "7*28 = <<7*28=196>>196 and 7*29 = <<7*29=203>>203 is more than 200".
// The caller has made sure n is not a multiple of k.
inline std::string fits(Story& s, int64_t k, int64_t n) {
    const int64_t q = n / k;
    const std::string below = s.calc(str(k) + "*" + str(q), k * q), above = s.calc(str(k) + "*" + str(q + 1), k * (q + 1));
    return say(s, s_fits, {{"{p1}", below}, {"{p2}", above}, {"{n}", str(n)}});
}

// The least common multiple found by counting up by the larger number:
// "Counting up by 12: 12, 24. The first of these that 8 also goes into is
// 24: 12*2 = <<12*2=24>>24 and 8*3 = <<8*3=24>>24."
inline void common_multiple(Story& s, int64_t a, int64_t b) {
    const int64_t big = std::max(a, b), small = std::min(a, b), both = std::lcm(a, b);
    std::string list;
    for (int64_t m = big; m <= both; m += big) list += (list.empty() ? "" : ", ") + str(m);
    s.line(say(s, s_count_up, {{"{big}", str(big)}, {"{list}", list}}));
    const std::string by_big = s.calc(str(big) + "*" + str(both / big), both);
    const std::string by_small = s.calc(str(small) + "*" + str(both / small), both);
    s.line(say(s, s_first_both, {{"{small}", str(small)}, {"{both}", str(both)}, {"{p1}", by_big}, {"{p2}", by_small}}));
}

// Two cycle lengths for a least common multiple: different, neither a
// multiple of the other, the common multiple at most `most` and at most 8
// counts of the larger.
inline bool two_cycles(Story& s, int64_t low, int64_t high, int64_t most, int64_t& a, int64_t& b) {
    a = s.draw(low, high);
    b = s.draw(low, high);
    if (a == b || a % b == 0 || b % a == 0) return false;
    const int64_t both = std::lcm(a, b);
    return both <= most && both / std::max(a, b) <= 8;
}

// n things into holders of k: full holders (asked 0), things left over
// (1), holders needed for all (2), things or places to fill the last one
// (3). Packing things (context 0) or placing people (context 1).
inline bool remainder_story(Story& s, std::string& answer, Place& place) {
    // What is packed or placed, its holder, and the sizes a real one has.
    struct Holder { const char* things; const char* one; const char* many; int64_t k_low, k_high, n_high; };
    static const std::array<Holder, 4> packs = {{{"cookies", "box", "boxes", 6, 24, 400},
                                                 {"eggs", "carton", "cartons", 12, 12, 300},
                                                 {"muffins", "tray", "trays", 6, 24, 300},
                                                 {"apples", "bag", "bags", 4, 12, 200}}};
    static const std::array<Holder, 3> seats = {{{"students", "van", "vans", 7, 15, 150},
                                                 {"guests", "table", "tables", 4, 12, 200},
                                                 {"campers", "tent", "tents", 4, 8, 60}}};
    const int context = s.out.context = int(s.rng.below(2));
    const Holder& h = context == 0 ? packs[s.rng.below(packs.size())] : seats[s.rng.below(seats.size())];
    s.out.asked = int(s.rng.below(4));
    const int64_t k = s.draw(h.k_low, h.k_high), n = s.draw(2 * k, h.n_high);
    const int64_t q = n / k, r = n % k;
    // At least 2 of everything: "1 cookies" reads badly, and an exact fit
    // has nothing left over to ask about.
    if (q < 2 || r < 2 || k - r < 2) return false;
    s.out.inputs = {n, k};
    const std::string things = h.things, one = h.one, many = h.many, &name = s.out.name;
    s.sentence(context == 0 ? rem_pack : rem_seat, {{"{n}", str(n)}, {"{k}", str(k)}, {"{things}", things},
                                                    {"{holders}", many}, {"{holder}", one}});
    static constexpr std::array<Phrase, 4> pack_questions = {q_pack_full, q_pack_left, q_pack_need, q_pack_more};
    static constexpr std::array<Phrase, 4> seat_questions = {q_seat_full, q_seat_left, q_seat_need, q_seat_empty};
    s.question = say(s, (context == 0 ? pack_questions : seat_questions)[size_t(s.out.asked)],
                     {{"{things}", things}, {"{holders}", many}, {"{holder}", one}});
    s.line(fits(s, k, n) + ", so " + (context == 0 ? name + " can fill " + str(q) + " " + many + " completely."
                                                   : str(q) + " " + many + " can be filled completely."));
    place = Place::anywhere;
    if (s.out.asked == 0) {
        answer = str(q);
        return true;
    }
    const std::string left = s.calc(str(n) + "-" + str(k * q), r) + " " + things;
    if (s.out.asked == 1) {
        s.line("So " + left + " are left over.");
        answer = str(r);
    } else if (s.out.asked == 2) {
        s.line("That leaves " + left + ", which need one more " + one + ".");
        const std::string needed = s.calc(str(q) + "+1", q + 1) + " " + many;
        s.line(context == 0 ? "So " + name + " needs " + needed + "." : "So " + needed + " are needed.");
        answer = str(q + 1);
    } else {
        s.line("That leaves " + left + " for the next " + one + ".");
        const std::string more = s.calc(str(k) + "-" + str(r), k - r);
        s.line(context == 0 ? "So " + name + " would need " + more + " more " + things + "."
                            : "So the last " + one + " has " + more + " empty places.");
        answer = str(k - r);
    }
    return true;
}

// When do two cycles next meet: buses (context 0), lights (1), visits
// (2), asking when (0) or how many times in the next hours, minutes or
// days (1); or hot dogs and buns in packs (3), asking for the packs of the
// first (0) or the second (1), or the number of each (2).
inline bool lcm_story(Story& s, std::string& answer, Place& place) {
    // Things sold in packs of a few, in pairs that are bought one for one.
    static const std::array<std::array<const char*, 2>, 4> pairs = {
        {{"hot dogs", "buns"}, {"paper cups", "lids"}, {"plates", "napkins"}, {"juice boxes", "straws"}}};
    const int context = s.out.context = int(s.rng.below(4));
    const auto& pair = pairs[s.rng.below(pairs.size())];
    s.out.asked = int(s.rng.below(context == 3 ? 3 : 2));
    // Real periods (buses every 5-60 minutes, lights every 2-30 seconds,
    // visits every 2-20 days, packs of 4-20), the most their common multiple
    // may be, and the spans asked about (2-12 hours, 2-10 minutes, 30-365
    // days).
    struct Range { int64_t low, high, most, span_low, span_high, unit; };
    static const std::array<Range, 4> ranges = {
        {{5, 60, 240, 2, 12, 60}, {2, 30, 180, 2, 10, 60}, {2, 20, 120, 30, 365, 1}, {4, 20, 160, 0, 0, 1}}};
    const Range& range = ranges[size_t(context)];
    int64_t a = 0, b = 0;
    if (!two_cycles(s, range.low, range.high, range.most, a, b)) return false;
    const int64_t both = std::lcm(a, b);
    s.out.inputs = {a, b};
    const bool times = context < 3 && s.out.asked == 1;
    const int64_t span = times ? s.draw(range.span_low, range.span_high) : 0, whole = span * range.unit;
    // At least twice in the span, and not exactly at its end.
    if (times && (whole / both < 2 || whole % both == 0)) return false;
    if (times) s.out.inputs.push_back(span);
    const std::string first = pair[0], second = pair[1], &name = s.out.name;
    std::string capital = first;
    capital[0] = char(capital[0] - 'a' + 'A');
    const Family family = context == 0 ? lcm_bus : context == 1 ? lcm_light : context == 2 ? lcm_visit : lcm_packs;
    s.sentence(family, {{"{a}", str(a)}, {"{b}", str(b)}, {"{first} come", capital + " come"}, {"{first}", first},
                        {"{second}", second}});
    const std::string both_of = "a multiple of both " + str(a) + " and " + str(b);
    // Per cycle context: what meets, how it meets, and the units.
    static const std::array<std::array<const char*, 4>, 3> words = {{{"both buses", "leave together", "minutes", "hours"},
                                                                      {"both lights", "flash together", "seconds", "minutes"},
                                                                      {"both", "do it on the same day", "days", "days"}}};
    place = Place::anywhere;
    if (context < 3) {
        const auto& w = words[size_t(context)];
        static constexpr std::array<Phrase, 3> when = {q_bus_when, q_light_when, q_visit_when};
        static constexpr std::array<Phrase, 3> how_often = {q_bus_times, q_light_times, q_visit_times};
        s.question = say(s, (times ? how_often : when)[size_t(context)], {{"{span}", str(span)}});
        s.line(std::string(context == 2 ? "They" : context == 0 ? "The buses" : "The lights") + " next " + w[1] +
               " after a number of " + w[2] + " that is " + both_of + ".");
        common_multiple(s, a, b);
        if (!times) {
            s.line(std::string("So ") + w[0] + " next " + w[1] + " in " + str(both) + " " + w[2] + ".");
            answer = str(both);
            return true;
        }
        s.line(std::string("So ") + w[0] + " " + w[1] + " every " + str(both) + " " + w[2] + ".");
        if (range.unit != 1)
            s.line("The next " + str(span) + " " + w[3] + " are " + s.calc(str(span) + "*60", whole) + " " + w[2] + ".");
        s.line(fits(s, both, whole) + ", so " + w[0] + " " + w[1] + " " + str(whole / both) + " times.");
        answer = str(whole / both);
        return true;
    }
    s.question = say(s, s.out.asked == 2 ? q_packs_count : q_packs,
                     {{"{item}", s.out.asked == 0 ? first : second}, {"{first}", first}});
    s.line("The number of " + first + " is a multiple of " + str(a) + " and the number of " + second +
           " a multiple of " + str(b) + ", so the smallest equal number is the first multiple of both.");
    common_multiple(s, a, b);
    answer = str(s.out.asked == 0 ? both / a : s.out.asked == 1 ? both / b : both);
    s.line(s.out.asked == 2 ? "So " + name + " ends up with " + answer + " " + first + "."
                            : "So " + name + " buys " + answer + " packs of " + (s.out.asked == 0 ? first : second) + ".");
    return true;
}

// The greatest common divisor: identical bags from two kinds of things
// (context 0), asking for the number of bags (0) or how many of the first
// (1) or second (2) in each; or two ribbons cut into the longest equal
// pieces (context 1), asking for the length (0) or the number of pieces (1).
inline bool gcd_story(Story& s, std::string& answer, Place& place) {
    static const std::array<std::array<const char*, 2>, 5> kinds = {{{"pencils", "erasers"},
                                                                     {"apples", "oranges"},
                                                                     {"stickers", "bookmarks"},
                                                                     {"granola bars", "juice boxes"},
                                                                     {"red balloons", "blue balloons"}}};
    const int context = s.out.context = int(s.rng.below(2));
    const auto& kind = kinds[s.rng.below(kinds.size())];
    s.out.asked = int(s.rng.below(context == 0 ? 3 : 2));
    // a = g x, b = g y with x and y sharing no factor, both at least 2 (so
    // neither number divides the other).
    const int64_t g = s.draw(2, context == 0 ? 24 : 30), x = s.draw(2, 12), y = s.draw(2, 12);
    const int64_t a = g * x, b = g * y, most = context == 0 ? 200 : 300;
    if (x == y || std::gcd(x, y) != 1 || a > most || b > most) return false;
    s.out.inputs = {a, b};
    const std::string first = kind[0], second = kind[1], &name = s.out.name;
    s.sentence(context == 0 ? gcd_kits : gcd_ribbon,
               {{"{a}", str(a)}, {"{b}", str(b)}, {"{first}", first}, {"{second}", second}});
    if (context == 0) {
        s.question = say(s, s.out.asked == 0 ? q_bags : q_bag_each, {{"{item}", s.out.asked == 1 ? first : second}});
        s.line("Every bag gets the same number of " + first + " and the same number of " + second +
               ", so the number of bags must divide both " + str(a) + " and " + str(b) + ".");
    } else {
        s.question = say(s, s.out.asked == 0 ? q_piece_length : q_pieces, {});
        s.line("The length of a piece must divide both " + str(a) + " and " + str(b) + ".");
    }
    const int64_t small = std::min(a, b), big = std::max(a, b);
    std::vector<int64_t> factors;
    for (int64_t f = 1; f <= small; ++f)
        if (small % f == 0) factors.push_back(f);
    s.line(say(s, s_factors, {{"{small}", str(small)}, {"{list}", listing(factors)}}));
    const std::string first_parts = s.calc(str(a) + "/" + str(g), x), second_parts = s.calc(str(b) + "/" + str(g), y);
    s.line(say(s, s_largest, {{"{big}", str(big)}, {"{g}", str(g)}, {"{p1}", first_parts}, {"{p2}", second_parts}}));
    place = Place::anywhere;
    if (context == 0) {
        const std::array<int64_t, 3> values = {g, x, y};
        answer = str(values[size_t(s.out.asked)]);
        s.line(s.out.asked == 0 ? "So " + name + " can make at most " + answer + " bags."
                                : "So each bag has " + answer + " " + (s.out.asked == 1 ? first : second) + ".");
    } else if (s.out.asked == 0) {
        answer = str(g);
        s.line("So each piece is " + answer + " centimeters long.");
    } else {
        answer = str(x + y);
        s.line("In all there are " + s.calc(str(x) + "+" + str(y), x + y) + " pieces.");
    }
    return true;
}

// How many ways n things go into equal rows (context 0) or teams (1),
// with at least m rows and at least m in each (m is 2 or 3).
inline bool factors_story(Story& s, std::string& answer, Place& place) {
    const int context = s.out.context = int(s.rng.below(2));
    const int64_t n = s.draw(12, 200), m = s.draw(2, 3);
    std::vector<std::pair<int64_t, int64_t>> products;  // d * e = n with m <= d <= e
    for (int64_t d = m; d * d <= n; ++d)
        if (n % d == 0) products.push_back({d, n / d});
    if (products.size() < 2) return false;  // one way, or none, is not much of a question
    const bool square = products.back().first == products.back().second;
    const int64_t pairs = int64_t(products.size()), count = 2 * pairs - (square ? 1 : 0);
    s.out.inputs = {n, m};
    s.sentence(context == 0 ? factor_rows : factor_teams, {{"{n}", str(n)}, {"{m}", str(m)}});
    const std::string groups = context == 0 ? "rows" : "teams";
    s.question = say(s, context == 0 ? q_rows : q_teams, {});
    s.line((context == 0 ? "The number of rows times the number of chairs in each row is "
                         : "The number of teams times the number on each team is ") +
           str(n) + ", and both are at least " + str(m) + ".");
    std::vector<std::string> written;
    for (const auto& [d, e] : products) written.push_back(s.calc(str(d) + "*" + str(e), n));
    std::string list;
    for (size_t i = 0; i < written.size(); ++i)
        list += (i == 0 ? "" : i + 1 == written.size() ? " and " : ", ") + written[i];
    s.line("The ways to write " + str(n) + " as such a product are " + list + ".");
    const auto [d, e] = products.front();
    const std::string example = str(d) + " " + groups + " of " + str(e) + ", or " + str(e) + " " + groups + " of " + str(d);
    const int64_t last = products.back().first;
    s.line("Each product gives two choices (" + example + ")" +
           (square ? ", except " + str(last) + "*" + str(last) + ", which gives one: " +
                         s.calc(str(pairs) + "*2-1", count)
                   : ": " + s.calc(str(pairs) + "*2", count)) +
           ".");
    s.line("So there are " + str(count) + " possible numbers of " + groups + ".");
    answer = str(count);
    place = Place::anywhere;
    return true;
}

// Counting multiples: tickets 1 to n that are multiples of k (context 0);
// every k-th of n people (1); every a-th and every b-th of n people, both
// (2); multiples of k from a to b (3); even or odd lockers from a to b (4).
inline bool multiples_story(Story& s, std::string& answer, Place& place) {
    static const std::array<const char*, 5> gifts = {"muffin", "balloon", "pen", "badge", "cookie"};
    const int context = s.out.context = int(s.rng.below(5));
    const std::string gift = gifts[s.rng.below(gifts.size())];
    place = Place::anywhere;
    if (context <= 1) {
        const int64_t k = context == 0 ? s.draw(3, 19) : s.draw(3, 20), n = s.draw(30, context == 0 ? 500 : 400);
        if (n / k < 2 || n % k == 0) return false;
        const int64_t q = n / k;
        s.out.inputs = {n, k};
        s.sentence(context == 0 ? mult_tickets : mult_every,
                   {{"{n}", str(n)}, {"{k}", str(k)}, {"{kth}", ordinal(k)}, {"{gift}", gift}});
        if (context == 0) {
            s.question = say(s, q_tickets, {});
            s.line("The winning numbers are the multiples of " + str(k) + " up to " + str(n) + ".");
            s.line(fits(s, k, n) + ", so " + str(q) + " tickets win.");
        } else {
            s.question = say(s, q_every, {{"{gift}", gift}});
            s.line("The people who get a " + gift + " are those whose place in line is a multiple of " + str(k) +
                   ", up to " + str(n) + ".");
            s.line(fits(s, k, n) + ", so " + str(q) + " people get a " + gift + ".");
        }
        answer = str(q);
        return true;
    }
    if (context == 2) {
        int64_t a = 0, b = 0;
        if (!two_cycles(s, 2, 12, 60, a, b)) return false;
        const int64_t both = std::lcm(a, b), n = s.draw(30, 500);
        if (n / both < 2 || n % both == 0) return false;
        s.out.inputs = {n, a, b};
        s.sentence(mult_both, {{"{n}", str(n)}, {"{ath}", ordinal(a)}, {"{bth}", ordinal(b)}});
        s.question = say(s, q_both, {});
        s.line("A person gets both when their place in the count is a multiple of both " + str(a) + " and " + str(b) + ".");
        common_multiple(s, a, b);
        s.line("So every " + ordinal(both) + " person gets both.");
        s.line(fits(s, both, n) + ", so " + str(n / both) + " people get both.");
        answer = str(n / both);
        return true;
    }
    if (context == 3) {
        const int64_t k = s.draw(3, 15), a = s.draw(10, 200), b = a + s.draw(30, 300);
        // a and b are not multiples, so the first and last are found the same way.
        if (a <= k || a % k == 0 || b % k == 0) return false;
        const int64_t i = a / k + 1, j = b / k, count = j - i + 1;
        if (count < 2) return false;
        s.out.inputs = {a, b, k};
        s.sentence(mult_range, {{"{a}", str(a)}, {"{b}", str(b)}, {"{k}", str(k)}});
        s.question = say(s, q_range, {{"{a}", str(a)}, {"{b}", str(b)}, {"{k}", str(k)}});
        s.line("The first multiple of " + str(k) + " from " + str(a) + " on is " + s.calc(str(k) + "*" + str(i), k * i) +
               ", since " + s.calc(str(k) + "*" + str(i - 1), k * (i - 1)) + " is less than " + str(a) + ".");
        s.line("The last one up to " + str(b) + " is " + s.calc(str(k) + "*" + str(j), k * j) + ", since " +
               s.calc(str(k) + "*" + str(j + 1), k * (j + 1)) + " is more than " + str(b) + ".");
        s.line("So the multiples are " + str(k) + " times " + str(i) + " up to " + str(k) + " times " + str(j) + ": " +
               s.calc(str(j) + "-" + str(i) + "+1", count) + " numbers.");
        answer = str(count);
        return true;
    }
    const int odd = int(s.rng.below(2));
    const int64_t a = s.draw(1, 150), b = a + s.draw(20, 200);
    const int64_t first = a % 2 == odd ? a : a + 1, last = b % 2 == odd ? b : b - 1, count = (last - first) / 2 + 1;
    s.out.inputs = {a, b, odd};
    const std::string parity = odd ? "odd" : "even";
    s.sentence(parity_lockers, {{"{a}", str(a)}, {"{b}", str(b)}, {"{parity}", parity}});
    s.question = say(s, q_lockers, {{"{parity}", parity}});
    s.line("The first " + parity + " number from " + str(a) + " on is " + str(first) + ", and the last one up to " +
           str(b) + " is " + str(last) + ".");
    s.line("Going from " + str(first) + " to " + str(last) + " in steps of 2: " +
           s.calc("(" + str(last) + "-" + str(first) + ")/2+1", count) + " lockers.");
    s.line("So there are " + str(count) + " " + parity + "-numbered lockers.");
    answer = str(count);
    return true;
}

// The digits used to number n pages (context 0), the pages that d digits
// number (1), or a two-digit number from the sum and the difference of its
// digits (2).
inline bool digits_story(Story& s, std::string& answer, Place& place) {
    const int context = s.out.context = int(s.rng.below(3));
    place = Place::anywhere;
    if (context <= 1) {
        const bool long_book = s.rng.below(2);
        const int64_t n = long_book ? s.draw(101, 999) : s.draw(20, 99);  // 101 and up: not "1 pages"
        const int64_t short_pages = std::min<int64_t>(n, 99) - 9, long_pages = std::max<int64_t>(n, 99) - 99;
        const int64_t digits = 9 + 2 * short_pages + 3 * long_pages;
        s.out.inputs = {context == 0 ? n : digits};
        if (context == 0) {
            s.sentence(digit_pages, {{"{n}", str(n)}});
            s.question = say(s, q_digits, {});
            s.line("Pages 1 to 9 have one digit each: 9 digits.");
            if (!long_book) {
                s.line("Pages 10 to " + str(n) + " have two digits each: " + s.calc(str(n) + "-9", short_pages) +
                       " pages, so " + s.calc(str(short_pages) + "*2", 2 * short_pages) + " digits.");
                s.line("In all that is " + s.calc("9+" + str(2 * short_pages), digits) + " digits.");
            } else {
                s.line("Pages 10 to 99 have two digits each: " + s.calc("99-9", 90) + " pages, so " +
                       s.calc("90*2", 180) + " digits.");
                s.line("Pages 100 to " + str(n) + " have three digits each: " + s.calc(str(n) + "-99", long_pages) +
                       " pages, so " + s.calc(str(long_pages) + "*3", 3 * long_pages) + " digits.");
                s.line("In all that is " + s.calc("9+180+" + str(3 * long_pages), digits) + " digits.");
            }
            answer = str(digits);
            return true;
        }
        s.sentence(digit_total, {{"{d}", str(digits)}});
        s.question = say(s, q_pages, {});
        if (!long_book) {
            s.line("Pages 1 to 9 use 9 digits.");
            s.line("The other " + s.calc(str(digits) + "-9", 2 * short_pages) + " digits are on two-digit pages: " +
                   s.calc(str(2 * short_pages) + "/2", short_pages) + " pages.");
            s.line("So there are " + s.calc("9+" + str(short_pages), n) + " pages.");
        } else {
            s.line("Pages 1 to 9 use 9 digits, and pages 10 to 99 use " + s.calc("90*2", 180) + " digits: " +
                   s.calc("9+180", 189) + " digits.");
            s.line("The other " + s.calc(str(digits) + "-189", 3 * long_pages) + " digits are on three-digit pages: " +
                   s.calc(str(3 * long_pages) + "/3", long_pages) + " pages.");
            s.line("So there are " + s.calc("99+" + str(long_pages), n) + " pages.");
        }
        answer = str(n);
        return true;
    }
    const bool tens_larger = s.rng.below(2);
    const int64_t tens = tens_larger ? s.draw(2, 9) : s.draw(1, 8);
    const int64_t ones = tens_larger ? s.draw(0, tens - 1) : s.draw(tens + 1, 9);
    const int64_t sum = tens + ones, gap = tens_larger ? tens - ones : ones - tens;
    const int64_t larger = std::max(tens, ones), smaller = std::min(tens, ones);
    s.out.inputs = {sum, gap, tens_larger ? 0 : 1};
    const std::string relation = tens_larger ? "the tens digit is " + str(gap) + " more than the ones digit"
                                             : "the ones digit is " + str(gap) + " more than the tens digit";
    s.sentence(digit_puzzle, {{"{s}", str(sum)}, {"{relation}", relation}});
    s.question = say(s, q_number, {});
    s.line("Adding the sum and the difference gives twice the larger digit: " +
           s.calc(str(sum) + "+" + str(gap), sum + gap) + ", and " + s.calc(str(sum + gap) + "/2", larger) + ".");
    s.line("The smaller digit is " + s.calc(str(sum) + "-" + str(larger), smaller) + ".");
    answer = str(10 * tens + ones);
    s.line("The tens digit is " + str(tens) + " and the ones digit is " + str(ones) + ", so the number is " + answer + ".");
    return true;
}

// A repeating pattern: r of the first color then b of the second, n beads
// (context 0); or w work days then r days off, the first n days (1).
// Asked: the first (0) or the second (1).
inline bool pattern_story(Story& s, std::string& answer, Place& place) {
    static const std::array<std::array<const char*, 2>, 4> colors = {
        {{"red", "blue"}, {"white", "green"}, {"yellow", "black"}, {"pink", "purple"}}};
    const int context = s.out.context = int(s.rng.below(2));
    const auto& color = colors[s.rng.below(colors.size())];
    s.out.asked = int(s.rng.below(2));
    const int64_t r = s.draw(2, 6), b = context == 0 ? s.draw(2, 6) : s.draw(2, 4);
    const int64_t n = context == 0 ? s.draw(20, 150) : s.draw(20, 100);
    const int64_t cycle = r + b, rounds = n / cycle, extra = n % cycle;
    // At least 2 rounds and 2 more beads (days): "1 beads" reads badly.
    if (rounds < 2 || extra < 2) return false;
    s.out.inputs = {r, b, n};
    const std::string first = context == 0 ? std::string(color[0]) + " beads" : "work days";
    const std::string second = context == 0 ? std::string(color[1]) + " beads" : "days off";
    const std::string unit = context == 0 ? "beads" : "days";
    if (context == 0)
        s.sentence(pattern_beads, {{"{n}", str(n)}, {"{r}", str(r)}, {"{b}", str(b)}, {"{first}", color[0]},
                                   {"{second}", color[1]}});
    else
        s.sentence(pattern_shifts, {{"{w}", str(r)}, {"{r}", str(b)}});
    s.question = say(s, context == 0 ? q_beads : q_days,
                     {{"{asked}", s.out.asked == 0 ? first : second}, {"{n}", str(n)}});
    s.line("One round of the pattern is " + s.calc(str(r) + "+" + str(b), cycle) + " " + unit + ".");
    s.line(fits(s, cycle, n) + ", so there are " + str(rounds) + " full rounds and " +
           s.calc(str(n) + "-" + str(cycle * rounds), extra) + " " + unit + " more.");
    const int64_t own = s.out.asked == 0 ? r : b;
    const std::string asked = s.out.asked == 0 ? first : second;
    s.line("The full rounds have " + s.calc(str(rounds) + "*" + str(own), rounds * own) + " " + asked + ".");
    int64_t total = rounds * own;
    if (s.out.asked == 0) {
        const int64_t start = std::min(extra, r);
        total += start;
        s.line("The " + str(extra) + " " + unit + " more start the pattern again, so " +
               (extra <= r ? "all of them" : "the first " + str(r)) + " are " + first + ": " +
               s.calc(str(rounds * own) + "+" + str(start), total) + ".");
    } else if (extra <= r) {
        s.line("The " + str(extra) + " " + unit + " more start the pattern again, so none of them are " + second + ".");
    } else {
        const int64_t end = extra - r;
        total += end;
        s.line("The " + str(extra) + " " + unit + " more start the pattern again: the first " + str(r) + " are " + first +
               " and " + s.calc(str(extra) + "-" + str(r), end) + " are " + second + ", so " +
               s.calc(str(rounds * own) + "+" + str(end), total) + ".");
    }
    answer = str(total);
    s.line("So there are " + answer + " " + asked + ".");
    place = Place::anywhere;
    return true;
}

// One attempt at a story of the given kind; false means draw again.
inline bool candidate(Choices& rng, const MathOptions& options, NumberTheoryProblem& out) {
    Story s(rng, options, out, patterns.data());
    std::string answer;
    Place place = Place::anywhere;
    bool made = false;
    switch (out.kind) {
        case NumberTheoryProblem::remainder: made = remainder_story(s, answer, place); break;
        case NumberTheoryProblem::lcm: made = lcm_story(s, answer, place); break;
        case NumberTheoryProblem::gcd: made = gcd_story(s, answer, place); break;
        case NumberTheoryProblem::factors: made = factors_story(s, answer, place); break;
        case NumberTheoryProblem::multiples: made = multiples_story(s, answer, place); break;
        case NumberTheoryProblem::digits: made = digits_story(s, answer, place); break;
        case NumberTheoryProblem::pattern: made = pattern_story(s, answer, place); break;
        case NumberTheoryProblem::kind_count: break;
    }
    if (!made) return false;
    // This topic's own fillers, in one problem of two: gen-word's shared
    // ones ("The office door is numbered 14", "route 9") would read as part
    // of a story about numbered tickets, lockers or multiples.
    math_detail::add_distractor(s, answer, place, fillers, 2);
    math_detail::finish(s, answer, "number-theory");
    return true;
}

}  // namespace number_detail

// The second solver. It reads only the story's numbers (inputs, context)
// and what is asked, never the generator's solution or answer, and uses
// none of number_detail's code: every answer is found by trying numbers
// one at a time. What inputs mean, per kind:
//
//   kind       context              inputs              asked
//   remainder  0 pack, 1 seat       n, k                0 full holders, 1 left over,
//                                                       2 holders for all, 3 to fill one more
//   lcm        0-2 cycles, 3 packs  a, b                packs: 0 packs of a, 1 packs of b,
//                                                       2 the equal number; else 0 the time
//              0-2, asked 1         a, b, span          how many times in the span (hours,
//                                                       minutes, days)
//   gcd        0 bags               a, b                0 bags, 1 a per bag, 2 b per bag
//              1 ribbons            a, b                0 piece length, 1 pieces in all
//   factors    0 rows, 1 teams      n, m                (numbers of rows or teams, at
//                                                        least m rows of at least m)
//   multiples  0 tickets, 1 every   n, k                (multiples of k in 1..n)
//              2 both               n, a, b             (multiples of a and b in 1..n)
//              3 range              a, b, k             (multiples of k in a..b)
//              4 lockers            a, b, odd           (even or odd numbers in a..b)
//   digits     0 pages              n                   (digits to write 1..n)
//              1 digits             d                   (the pages d digits number)
//              2 puzzle             sum, gap, which     (the number; which: 0 tens digit
//                                                        larger, 1 ones digit larger)
//   pattern    0 beads, 1 days      first, second, n    0 first's count, 1 second's
namespace number_theory_check {

// The count of t in [from, to] for which keep(t) is true.
template <class Keep>
int64_t count_in(int64_t from, int64_t to, Keep keep) {
    int64_t found = 0;
    for (int64_t t = from; t <= to; ++t) found += keep(t) ? 1 : 0;
    return found;
}

// The answer the story's numbers give, or what is wrong with the story.
inline std::string solve(const NumberTheoryProblem& p, int64_t& answer) {
    const auto& in = p.inputs;
    for (int64_t v : in)
        if (v < 0 || v > 100000) return "a story number is out of range";
    switch (p.kind) {
        case NumberTheoryProblem::remainder: {
            if (in.size() != 2 || in[1] < 2) return "remainder record is wrong";
            // Fill holders one at a time while a whole one fits.
            int64_t full = 0, left = in[0];
            while (left >= in[1]) { left -= in[1]; ++full; }
            if (left == 0) return "nothing is left over";
            const std::array<int64_t, 4> answers = {full, left, full + 1, in[1] - left};
            if (p.asked < 0 || p.asked > 3) return "no such question";
            answer = answers[size_t(p.asked)];
            return "";
        }
        case NumberTheoryProblem::lcm: {
            const bool times = p.context < 3 && p.asked == 1;
            if (in.size() != (times ? 3u : 2u) || in[0] < 1 || in[1] < 1) return "lcm record is wrong";
            const auto meet = [&](int64_t t) { return t % in[0] == 0 && t % in[1] == 0; };
            if (times) {
                // Every minute (second, day) of the span: hours turned into
                // minutes, minutes into seconds.
                answer = count_in(1, in[2] * (p.context == 2 ? 1 : 60), meet);
                return answer >= 1 ? "" : "they never meet in the span";
            }
            // The first time after 0 at which both cycles come round.
            int64_t t = 1;
            while (!meet(t)) ++t;
            if (p.context != 3) answer = t;
            else if (p.asked == 0) answer = t / in[0];
            else if (p.asked == 1) answer = t / in[1];
            else answer = t;
            return "";
        }
        case NumberTheoryProblem::gcd: {
            if (in.size() != 2 || in[0] < 1 || in[1] < 1) return "gcd record is wrong";
            int64_t largest = 0;
            for (int64_t d = 1; d <= in[0] && d <= in[1]; ++d)
                if (in[0] % d == 0 && in[1] % d == 0) largest = d;
            if (largest == in[0] || largest == in[1]) return "one number divides the other";
            const int64_t first = in[0] / largest, second = in[1] / largest;
            if (p.context == 0) answer = p.asked == 0 ? largest : p.asked == 1 ? first : second;
            else answer = p.asked == 0 ? largest : first + second;
            return "";
        }
        case NumberTheoryProblem::factors:
            if (in.size() != 2 || in[1] < 1) return "factors record is wrong";
            answer = count_in(in[1], in[0], [&](int64_t rows) { return in[0] % rows == 0 && in[0] / rows >= in[1]; });
            return answer >= 2 ? "" : "fewer than two arrangements";
        case NumberTheoryProblem::multiples:
            if (p.context <= 1) {
                if (in.size() != 2 || in[1] < 1) return "multiples record is wrong";
                answer = count_in(1, in[0], [&](int64_t t) { return t % in[1] == 0; });
            } else if (p.context == 2) {
                if (in.size() != 3 || in[1] < 1 || in[2] < 1) return "multiples record is wrong";
                answer = count_in(1, in[0], [&](int64_t t) { return t % in[1] == 0 && t % in[2] == 0; });
            } else if (p.context == 3) {
                if (in.size() != 3 || in[2] < 1 || in[0] > in[1]) return "multiples record is wrong";
                answer = count_in(in[0], in[1], [&](int64_t t) { return t % in[2] == 0; });
            } else {
                if (in.size() != 3 || in[2] > 1 || in[0] > in[1]) return "lockers record is wrong";
                answer = count_in(in[0], in[1], [&](int64_t t) { return t % 2 == in[2]; });
            }
            return answer >= 1 ? "" : "nothing to count";
        case NumberTheoryProblem::digits:
            if (p.context <= 1) {
                if (in.size() != 1 || in[0] < 1) return "pages record is wrong";
                // Number the pages one at a time, counting the digits.
                int64_t written = 0, found = 0;
                for (int64_t page = 1; page <= 9999 && (p.context == 1 || page <= in[0]); ++page) {
                    written += int64_t(std::to_string(page).size());
                    if (p.context == 1 && written == in[0]) { ++found; answer = page; }
                }
                if (p.context == 0) answer = written;
                return p.context == 0 || found == 1 ? "" : "no number of pages uses exactly these digits";
            } else {
                if (in.size() != 3 || in[2] > 1) return "digits record is wrong";
                int found = 0;
                for (int64_t number = 10; number <= 99; ++number) {
                    const int64_t tens = number / 10, ones = number % 10;
                    const int64_t gap = in[2] == 0 ? tens - ones : ones - tens;
                    if (tens + ones == in[0] && gap == in[1]) { ++found; answer = number; }
                }
                return found == 1 ? "" : "the digits fit " + std::to_string(found) + " numbers, not one";
            }
        case NumberTheoryProblem::pattern: {
            if (in.size() != 3 || in[0] < 1 || in[1] < 1) return "pattern record is wrong";
            // Lay the beads (days) down one at a time.
            int64_t first = 0, second = 0, at = 0;
            for (int64_t i = 0; i < in[2]; ++i) {
                if (at < in[0]) ++first;
                else ++second;
                if (++at == in[0] + in[1]) at = 0;
            }
            answer = p.asked == 0 ? first : second;
            return "";
        }
        case NumberTheoryProblem::kind_count: return "no such kind";
    }
    return "no such kind";
}

// Everything the second solver checks on a finished problem (with notes):
// returns what is wrong, or an empty string.
inline std::string problems_with(const NumberTheoryProblem& p) {
    int64_t answer = 0;
    const std::string unsolved = solve(p, answer);
    if (!unsolved.empty()) return unsolved;
    if (answer <= 0) return "the answer is not positive";
    return math_check::answer_and_notes(p, std::to_string(answer));
}

}  // namespace number_theory_check

// How often each kind is drawn, out of 33 (in Kind's order). Kinds with
// few different problems are drawn less, so each repeats less often in a
// large file: in 1M problems there are about 97,000 different multiples
// stories, 25,000 remainders, 22,000 cycles, 7,900 common divisors, 6,500
// patterns, 2,000 digit stories and only about 300 factor stories.
inline constexpr std::array<int, NumberTheoryProblem::kind_count> number_theory_weights = {8, 6, 4, 1, 8, 2, 4};

// One problem; every one is checked by number_theory_check before it is returned.
inline NumberTheoryProblem make_number_theory_problem(Rng& rng, const MathOptions& options = {}) {
    int total = 0;
    for (int w : number_theory_weights) total += w;
    int pick = int(rng.below(uint64_t(total))), k = 0;
    while (pick >= number_theory_weights[size_t(k)]) pick -= number_theory_weights[size_t(k++)];
    const auto kind = NumberTheoryProblem::Kind(k);
    for (int attempt = 0; attempt < 10000; ++attempt) {
        NumberTheoryProblem result;
        result.kind = kind;
        result.recipe = recipe_of("number-theory", options.distractors, options.test, 0, int(kind), rng);
        Choices choices(rng);
        if (!number_detail::candidate(choices, options, result)) continue;
        const std::string wrong = number_theory_check::problems_with(result);
        CHECK(wrong.empty(), "the checker rejects a generated problem (%s): %s\n%s", wrong.c_str(),
              result.problem.question.c_str(), result.problem.steps.c_str());
        if (options.exclude && options.exclude->excludes(result.problem, result.distractor_number)) continue;
        if (!options.notes) result.problem.steps = math_detail::without_notes(result.problem.steps);
        return result;
    }
    CHECK(false, "could not generate a number-theory problem in 10000 attempts");
    return {};
}

inline std::vector<Problem> make_number_theory_problems(uint64_t seed, int64_t count, const MathOptions& options = {},
                                                        std::vector<Recipe>* recipes = nullptr) {
    return make_unique_problems(seed, count, [&](Rng& rng) { return make_number_theory_problem(rng, options); },
                                recipes);
}
