# Re-drawn GSM8K problems (`cxxgpt redraw-qa`)

2026-10-17. A prototype data tool, CPU only (the whole of GSM8K train takes
under a second). Nothing it writes is in a training mix: the owner decides.

## The idea, in plain words

The model has learned GSM8K train's gold solutions by heart, and the
generators' problems (`gen-word`, `gen-math`) by their templates: self-train
writes both back word for word. Human-written problems with answer keys and
an open licence are scarce (about 700 from textbooks). So: keep GSM8K's
human question, swap in new numbers, and work out the new answer exactly.
The wording stays varied and human, and the answer cannot be recited. (The
same spirit as GSM-Symbolic, which templates GSM8K test questions by hand;
here it is automatic, on train only.)

Example (GSM8K train row 1):

> Natalia sold clips to **56** of her friends in April, and then she sold
> half as many clips in May. How many clips did Natalia sell altogether in
> April and May?
>
> Natalia sold 56/2 = <<56/2=28>>28 clips in May.
> Natalia sold 56+28 = <<56+28=84>>84 clips altogether in April and May.
> #### 84

(The original has 48, 24 and 72.)

## How it works

`src/redraw_qa.h` explains each step; in short:

1. **A formula per problem.** The gold solution's calculations (its
   calculator notes `<<48/2=24>>` and plain `16 x 7 = 112`) are turned into
   formulas over the question's numbers by the code self-train's re-draw
   check already uses (`self_train::redraw_detail::read_calcs`): "q1/2",
   then "q1+(q1/2)". Every number in a calculation needs one clear source:
   a number of the question, a percentage as its hundredth ("0.3" for 30%),
   an earlier result, a number the question gives in words ("half",
   "dozen"), or a usual constant (1, 2, 100; 0.01 with a percentage; 7, 12,
   24, 52, 60, 365, 1000 only where the problem speaks of weeks, hours,
   minutes...). A number from nowhere else is rejected: it is usually
   arithmetic done in the head ("3 sets of 20 minutes" in an hour), which
   would stay 3 when the 20 changes.
2. **What changes and what stays.** Some question numbers never change: 1, 2
   and 100, numbers the question also gives in words, unit facts ("60
   minutes", "a gallon is 8 pints"), ordinals ("3rd"), times and ratios
   ("3:30", "5:7"), bank notes ("a $20 bill"), the parts of a fraction
   ("2/9") and years. Keeping a number is always right; it only gives less
   variety. Two question numbers with the same value change together (then
   whichever one the solution meant, it stays right). A question number
   that can change but that no calculation uses rejects the problem ("30
   students, 12 boys and 18 girls" would stop adding up).
3. **New numbers.** About half to twice the original, as round as the
   original (120 gives multiples of 10), the same decimal places ($2.50
   stays in cents), a percentage under 100 stays under 100, "5 days a week"
   stays at most 7 (24 hours a day, 12 months a year, 60 minutes an hour),
   never 1. Every calculation is worked out again exactly (calc.h) and
   must keep its kind: whole stays whole (no "2.5 people"), positive stays
   positive, 1 stays 1 ("1 hours"), a decimal gets at most two places (or
   its own), and nothing grows or shrinks more than 25 times. Up to 300
   draws per variant; a problem with none that passes is `no_draw`.
4. **Rewriting.** Every number of the question and the worked solution is
   written again, in its original style (commas, decimal places). The
   solution's numbers are found by value: each value must have one source
   (a result worked out before or on its line; a note's result and the
   number right after it are that note's). A number tied to a question
   number must name the same unit: "x 5 days" (the weekdays) is not the
   question's "5 hours" on Saturday (`units`, found in the hand check
   below). Each note's result is what the calculator writes for the new
   expression, and "a 8-hour shift" becomes "an 8-hour shift".
5. **Checks on each variant, from its text alone.** Every calculation right
   by calc.h, as many as the original's (the failure labels' reading); the
   question and solution are the original's word for word once every number
   is masked; the "####" answer is the formula's value; and self-train's
   re-draw check (`trace_try`, its own code path) accepts the original gold
   solution against the variants. That last check cannot read a percentage
   used as its hundredth, so it runs on 3,471 of the 3,658 variants. A
   variant failing any check stops the run with an error: none does.

Input: `data/gsm8k-notes/gsm8k-train.txt`, GSM8K train without the 500 dev
problems and 7 near-copies of dev/test questions, with its calculator notes
(`operations/build_gsm8k_notes.py`). The tool refuses a file whose name
says dev or test. (`data/gsm8k/train.jsonl` on the laptop is the older
download with all 7,473 problems, dev included: not used.)

```
build/cxxgpt.exe redraw-qa --problems data/gsm8k-notes/gsm8k-train.txt --out data/gsm8k-redraw/train [--versions 1] [--seed 1]
```

Writes `train.txt` (a problems file, task `gsm8k-redraw`), `train.jsonl`
(GSM8K's JSON Lines: `prep-qa --keep-calc` reads it back), `train.audit.jsonl`
(one line per input problem: its reason, formula, fixed numbers, draws) and
`train.json` (settings and counts).

## Results (seed 1)

**3,658 of 6,966 problems convert (52.5%)**, one variant each. With
`--versions 3`: 10,849 variants of 3,669 problems (a few problems have only
one or two possible variants). Of the 22,198 numbers in the problems read,
4,906 stay fixed. Converted problems change 1 to 8 numbers (most often 2
or 3).

Why the rest do not:

| reason | problems | what it means |
|---|---:|---|
| constant | 1,057 | a number in a calculation comes from nowhere: mostly arithmetic done in the head (4, 3, 5, 6 are the most common: "4 weeks", "3 sets of 20 minutes") |
| ambiguous | 935 | a number in a calculation could be two things (16 slices, and 2 x 8 = 16) |
| nothing | 220 | no number can change (all are 1, 2, 100, words, units...) |
| unused | 209 | a question number that could change is used by no calculation |
| unreadable | 199 | an "=" the calculator cannot check (197: algebra "X + 278 = 536", mixed numbers "8 x 1 1/2 = 12", fractions "= 2/3 pound"), a digit that is no number (2) |
| answer | 196 | no calculation gives the gold answer (142: a last step the labels cannot read, "3 + 1 + 1= 5", "* 100% = 25%", or a rounding up: 137.33 hours, so 138), or two different ones do (54) |
| units | 122 | a solution number tied to a question number by value names another unit ("5 days" for "5 hours") |
| words | 96 | the solution writes a changing number in words ("his five friends") |
| text_ambiguous | 73 | a number of the solution's text could stand for two things |
| no_calcs | 70 | no calculation at all |
| text_unsourced | 52 | a number of the solution's text comes from nowhere |
| wrong | 47 | a gold calculation is wrong (or below zero) |
| no_draw | 32 | no draw of new numbers kept every result whole and positive |

The strict rules are deliberate: "one clear source" in the calculations is
not enough once the prose must be rewritten too, so a value that means two
things anywhere in the solution rejects the problem. The units rule costs
the most for what it catches: of 25 sampled `units` rejections, 2 or 3
were real coincidences (rows 804 and 1621, perhaps 6924; besides 5204),
the rest synonyms ("pieces" and "slices"). A wrong answer costs more than a lost problem.

Where more problems could come from, roughly in order of size: constants
with their units (4 weeks a month, 3 feet a yard, 5 weekdays, 30 days a
month) where the question names the unit; ambiguous numbers whose readings
the variants could tell apart (draw, and keep the reading that the
solution's later text agrees with); last steps done in prose.

## Leak check

Against GSM8K dev (500) and test (1,319), both for one and three variants
per problem: `cxxgpt leak-check` finds **0** questions sharing a 13-word
run with the variants, and **0** with the same question once numbers are
masked. Leaving numbers out of the 13-word runs entirely, one dev question
shares a sentence frame with one train problem's variants ("bought 9 books
about whales ... Each book cost $11 and each magazine cost $1" against
"books about cats"): a different story, inherited from its train original,
which build_gsm8k_notes.py already kept.

## Hand checks

Three rounds of 30 random variants, each worked out again by hand from its
own question; each round's findings became a rule before the next.

1. **First round (early rules):** 30 of 30 answers right; 2 did not make
   sense: "$40 bills" and "$4 bills" (and "a $2000 bill"). Rule: bank notes
   stay.
2. **Second round:** 29 of 30 right, **1 wrong**: row 5204 ("3 hours every
   weekday, 5 hours on Saturdays"): the solution's "x 5 days" are the five
   weekdays, a number worked out in the head, that has the value of
   Saturday's 5 hours, so it changed with them (8 hours a weekday x 7
   days). Rule: `units`. Also "an 8 day work week" (row 802). Rule: days a
   week at most 7, hours a day at most 24, and so on.
3. **Third round (all rules but the last two):** 30 of 30 right for the variant's
   own question. One stated a false fact: row 4804 tied "8 doors", "$8.00 a
   pint" and "a gallon (the same as 8 pints)", so the gallon became 15
   pints. Rule: unit facts stay (8 pints a gallon, 4 quarts, 16 ounces, 3
   feet a yard, 12 inches a foot), and so do numbers equal to them. In the
   final output only that row of the 30 changed (now only its $55
   changes); the 30 with their verdicts are in
   [docs/results/redraw-qa-check.md](results/redraw-qa-check.md).

The last rule, 0.01 as a constant where the problem speaks of percent,
came after the third round: it added 60 variants (3,598 to 3,658) and
changed none of the others. No round saw those 60; 3 of them were read and
are right.

What remains, from the third round:
- **Odd but right.** New numbers can make a story lopsided: "290 campers in
  three weeks, 30 two weeks ago, 245 last week", "$5 an hour, 318 hours of
  tutoring in two months". Nothing checks plausibility beyond kinds and
  the 25x bound.
- **Inherited readings.** Row 4463: "bought a phone at 60% of the initial
  price" is worked as 60% off, as GSM8K's own gold solution reads it. The
  variants keep the gold's reading, right or not (GSM8K's labels are wrong
  a few percent of the time).
- **Ties change meaning together.** "a pair of 20s" ($20 bills) had the
  value of "20 households", so both became 50: real bills here by luck.
- **Coincidences the units rule cannot see.** A number worked out in the
  head with a question number's value *and* unit is still taken for that
  question number. The second round found one in 30 before the units rule;
  the third found none in 30. With about 3,600 variants, expect some.

## Tests

`tests/test_redraw_qa.h` (in `test_engine`): fifteen hand-written problems,
each with the reason it must get (converted, ambiguous, constant, units,
unused, words, wrong...), every variant checked against a formula written
by hand in the test, apart from the tool's own; the final check catches a
wrong note, a changed word, a wrong answer and the original itself; the
draws (roundness, places, percentages, caps, never 1); the kinds results
keep; articles, number spelling, ".5"; which numbers stay fixed and why;
the units comparison. `tests/test_cli.sh`: the command on two problems,
its files, `prep-qa --keep-calc` reading the variants back, the same seed
giving the same file, a dev file refused.
