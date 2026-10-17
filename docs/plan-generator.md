# A plan generator: word problems whose solution plan varies

Design, 2026-10-17. Not built: the owner decides whether it is. Nothing
here changes a mix, a data file or a default.

## The problem it solves

The model (rp1, the 267M MoE, about 3-5% on GSM8K dev) mostly gets the
arithmetic right once the calculator does it, and mostly picks the wrong
**plan**: which numbers to combine, in what order, with which operation.
It reads surface cues: "twice" means `* 2`, "total" means `+`, "left"
means `-`. An example from the GSM8K dev set, asked during an exam on
2026-10-17:

> Over three hundred years, it has consumed 847 people. Ships have been
> built larger over time, so each new ship has twice as many people as
> the last ship. How many people were on the ship the monster ate in the
> first hundred years?

The gold solution calls the first ship's people S and adds up the three
ships, `S + 2S + 4S = 7S = 847`, so `S = 847 / 7 = 121`. The model wrote
`847*2` and `1684+847`: right arithmetic, wrong plan. "Twice" became a
multiplication and "has consumed" a total to grow, where the story needed
the opposite: a total given, an earlier quantity asked, and a chain of
"twice the last one".

The evidence that this is a pattern, not one bad problem:
- the failure labels (`eval-reasoning --labels`): with the calculator,
  few failures left are arithmetic; most are `other`, "wrong with no
  wrong calculation in sight: a wrong plan or goal" (README). Their
  numbers are not yet written down in the repository: the Gate 1 exam
  below would record them;
- self-train (README, "The model's own checked solutions") finds rp1
  writing the generators' own solutions back, sentence for sentence, on
  new generated problems: it has learned the templates;
- the generators have few plans. `gen-word` has six story shapes, and at
  7-8 steps five of them are "one multiplication, then five or six
  additions" (README, "Generated word problems"). `gen-math` has seven
  kinds per topic (equations, fractions, number theory). About 27 plan
  skeletons in all, each written in many wordings (162 + 84 + 120 + 120
  sentence patterns). The model sees thousands of ways to say the same
  plan, and few plans.

So the next generator should vary the **plan**, and treat wording as
secondary. This is the idea of "Physics of Language Models, Part 2.1"
(below): problems built from a random map of which quantity depends on
which, so that no fixed template fits.

## The idea in one example

Every problem starts as a small **dependency graph**: a map of quantities
(the boxes) and how each one is worked out from others (the arrows).
The monster problem's graph:

```
  ship1 (asked) ──×2──> ship2 ──×2──> ship3
     │                    │             │
     └────────────────────┴──────+──────┴──> all ships = 847 (given)
```

Here the total is given and the first ship is asked: a **backward**
problem. The generator would make problems of this plan in its own
stories (never this one: `--exclude` keeps GSM8K dev and test questions
out, as for `gen-math`). For example:

> Nora planted tomato seedlings on three weekends, 455 in all. Each
> weekend she planted 3 times as many as the weekend before. Her
> neighbor Leo planted 60 seedlings on the second weekend. How many
> seedlings did Nora plant on the first weekend?

```
Let x be the number of seedlings Nora planted on the first weekend.
On the second weekend she planted 3 times as many: 3x.
On the third weekend she planted 3 times the second: 3*3 = <<3*3=9>>9, so 9x.
In all she planted x + 3x + 9x, and 1+3+9 = <<1+3+9=13>>13, so 13x = 455.
On the first weekend she planted 455/13 = <<455/13=35>>35 seedlings.
#### 35
```

Three things to notice:
- every number the solution works out is in a calculator note, checked by
  `src/calc.h`, which has only `+ - * /` and parentheses. "13x" is never
  inside a note: the unknown stays in the words around the notes, as the
  equations topic already does (`Let x be Leo's age...`). The coefficient
  13 is not worked out in the head, as GSM8K's gold does ("7S"); it gets
  its own note, so the re-draw check below can follow it;
- Leo's 60 is a real distractor: a quantity of the same kind, on the same
  weekends, that the question does not need. Not a stock sentence ("The
  weather was sunny") but a box in the graph that is not on the asked
  box's path;
- the same graph can ask another box: "How many did Nora plant on the
  third weekend?" with the first weekend given is a forward problem of
  one multiplication; with the total given it is the backward problem
  plus one step. One story, several plans.

A forward example with a shared input and a question about a middle
quantity:

> A school orders 14 boxes of markers for the art room. Each box holds 12
> markers and costs $8. The music room gets 5 fewer boxes than the art
> room. How much does the music room's order cost?

```
The music room gets 14-5 = <<14-5=9>>9 boxes.
Its order costs 9*8 = <<9*8=72>>72 dollars.
#### 72
```

"Each box holds 12 markers" is not needed: the markers are a branch of
the graph the question does not reach. Ask "How many markers do the two
rooms get in all?" and the same story needs `14+9 = 23`, `23*12 = 276`,
and the price becomes the distractor.

## The graph

### Quantities

A quantity (a box of the graph) has:
- an owner: a person, a place or a thing ("Nora", "the art room", "the
  second weekend");
- a kind, which fixes its unit and how its value may look: a **count**
  (whole, of one object: seedlings, boxes, markers), **money** (exact
  cents, as `gen-word`), and later **time**, **length** and **weight**;
- a value, exact: whole for counts, hundredths for money, inside the
  program as integers (as `src/wordgen.h`), never a float.

### Relations

Each arrow is one relation from a small table. Each has a forward step
(work out the result from the inputs) and an inverse (work out one input
from the result and the others), which backward problems use.

| relation | sentence | forward | inverse | first version |
|---|---|---|---|---|
| given | "Nora has 14 cards." | stated | - | yes |
| sum | "in all", "together", "altogether" | `a+b(+c)` | `total-others` | yes |
| more / fewer | "5 more than", "5 fewer than" | `a+d`, `a-d` | `b-d`, `b+d` | yes |
| how many more | "How many more ... than ...?" (asked only) | `a-b` | - | yes |
| multiple | "3 times as many", "twice as many", "half as many" | `a*k`, `a/2` | `b/k`, `b*2` | yes |
| each | "12 markers in each box", "$8 each" | `n*p` | `t/p`, `t/n` | yes |
| equal share | "shared equally among 4 friends" | `t/k` | `s*k` | yes |
| sequence | "each weekend 3 times as many as the weekend before" (2-4 terms) | unrolled multiples | collected: `(1+k+k*k)x` | yes |
| part, percent | "1/4 of", "25% of", "the rest" | `a*f` | `part/f` | later |
| rate | "6 km an hour for 3 hours", "$12 an hour" | `r*t` | `d/t`, `d/r` | later |
| unit change | dozens, hours to minutes, cents to dollars | `a*c` | `a/c` | later |
| payment and change | "pays with a $50 bill" | `p-cost` | - | later (gen-word has it) |

A sequence is written out as two to four quantities, one per term, so
every step stays one relation with one or two inputs (iGSM's rule too,
below). The "first version" column is the smallest useful version
(see the end).

### Building a graph

1. **Length.** Draw the number of operations the solution needs, 2-8
   (`--min-steps`, `--max-steps`, as `gen-word`).
2. **World.** Draw a setting (a bakery, a farm, a class trip, a library,
   a garden, a shop, a sports club, a school fair...), two or three
   people, and one or two kinds of object with their realistic ranges,
   containers and prices.
3. **Grow from the question down.** Start with the asked quantity as the
   result of a relation; then, until the solution has its length, replace
   a given quantity by a relation that produces it from new quantities
   (or from one already in the graph: a quantity used twice, like the
   price above, gives a diamond rather than a tree). Growing downward
   from the question means every operation is needed: the solution's
   length is exactly the drawn length.
4. **Add what is not needed.** Zero to two more relations whose results
   the question does not need (`--distractors`):
   - a **side branch**: from quantities already in the story ("Each box
     holds 12 markers" when the question is about cost);
   - the story **going on past the question**: a quantity worked out
     from the asked one ("Leo has twice as many as Nora" when the
     question asks for Nora's). The question then sits in the middle of
     the story, not at its end, which is iGSM's "reask".
   These are the distractors GSM-Symbolic's "NoOp" test shows models
   falling for (below): same people, same objects, numbers that look
   usable.
5. **Direction.** Forward: every leaf of the graph is stated, the asked
   quantity is worked out. Backward: one leaf is hidden and becomes the
   unknown, and a quantity further down its path is stated instead (the
   total, what is left, the last term). Backward problems are kept
   **linear in the unknown**: the stated quantity must equal `a*x + b`,
   with `a` and `b` worked out from the other stated numbers. That covers
   "the total is given, find the first", "after spending half and $5 she
   has $20 left: how much at the start?" and the sequences, and leaves out
   what GSM8K rarely asks at this level: two unknowns at once, an unknown
   squared, an unknown in a divisor.
6. **Numbers.** For a forward problem, draw the leaves; for a backward
   one, draw the unknown first (as the equations topic draws its answer),
   then the other leaves, and work out the stated quantity from them. The
   whole draw is thrown away and made again (never adjusted) if any value
   is 0 or 1 ("1 cards"), fractional where it must be whole, outside its
   kind's realistic range or above 1,000,000, or if two numbers of the
   question are equal, or one equals 1, 2 or 100. The last rule is new:
   repeated small numbers are what makes the re-draw check's tries
   ambiguous today (README, "The re-draw check": a third to two thirds of
   right tries with one reading).

### Two ways to solve a backward problem

The generator writes whichever a person would:
- the unknown passes through the steps once, one after another (`x`, then
  `* 3`, then `- 5`, then shared by 2, gives 20): undo the steps from the
  end, as the equations topic's number puzzle does. Every step is a
  note: `20*2 = <<20*2=40>>40`, `40+5 = <<40+5=45>>45`, `45/3 = <<45/3=15>>15`;
- the unknown appears more than once (a sequence, `x` and `x + 6` added
  up): "Let x be ...", the coefficients collected in notes, then
  `(stated - b) / a` in one or two notes, as in the seedling example.

## Holding plans out, not only wordings

The point of this generator is the plan, so its test split must hold
plans out. iGSM does this with a hash of the solution template (below).
Here:
- each problem's **plan signature** is its solution with names, wordings
  and numbers removed: the relations in order, which earlier result or
  stated number each one takes, and the direction (forward, undo, let-x);
- a signature whose hash (mod 23) is 17 or more is a **test plan**
  (about a quarter); the training split never makes one, the test split
  only makes them. iGSM uses the same rule;
- sentence wordings are split as in the other generators (some frames for
  train, the others for test).

So the test split has plans the training split never made, in wordings
it never used. With three test files (new plan in seen wording, seen plan
in new wording, both new), an exam tells which of the two transfers.
Few plans exist at 2-3 operations, so there the split is mostly by
wording; held-out plans are real evidence from 4 operations on. The
README already says that success on held-out wording alone is no
evidence of improvement on GSM8K; held-out plans are closer to what
GSM8K tests.

## From graph to story

**The honest trade-off.** iGSM got its enormous variety by giving up
natural text: invented hierarchies of items, one fixed sentence per
quantity, arithmetic mod 23. We want GSM8K-like stories, so the wording is
limited by sentence frames written by hand (a few hundred, as in the
other generators). The variety budget goes to the graph; wording only
needs to be varied enough not to give the plan away.

How a graph becomes a story:
- **One sentence per stated relation**, from frames per relation and
  per kind ("{owner} has {k} times as many {objects} as {other}", "{k}
  times as much money as" for money). Two relations sometimes share a
  sentence ("Nora buys 5 boxes of 12 markers"), and a relation can sit in
  the question ("If Leo has 3 fewer, how many ...?").
- **Order.** Stated sentences are shuffled (iGSM shuffles too: the story
  need not follow the solution's order), as long as every reference still
  reads: each sentence names its quantities in full, and "the weekend
  before" or "the last ship" keeps its sentence after the one it points
  to. The question comes last, as in GSM8K.
- **Words for numbers.** "Twice", "half" and "three weekends" are words;
  multipliers such as "3 times" are digits. Which form a multiplier takes
  is a story choice, and a digit multiplier is drawn from 3 up, so a
  re-drawn version never turns "3 times" into "twice" (that would change
  the masked question and lose the version). That matters for the
  re-draw check (below).
- **More people, objects and settings** than `gen-word`'s 12 names and 8
  objects, and each world's ranges realistic (a box of 6-48 markers, a
  marker $1-4).
- **The audit rules carried over** (README, "Generated word problems"):
  no 0 or 1 results, realistic payments, no wording used twice in one
  story, items that come from somewhere. Each rule a test on every
  generated problem, as now.

**Cues that do not give the plan away.** The model's habit is "twice
means times 2". The generator measures, for each cue word ("twice",
"times", "total", "in all", "left", "more", "each", "shared"), how often
the cue's usual operation is the step the solution takes at that
sentence. In a forward-only generator that is near 100%; here backward
problems and middle questions bring it down ("total" is given, so the
solution subtracts or divides; "twice" is undone, so it halves). The
target is a cue map no shortcut can learn: each cue meaning its usual
operation in about half of its steps, reported per 1M problems with the
variety counts.

## The worked solution

- The needed steps only, in an order that respects the arrows (when
  several orders work, one is drawn): what iGSM calls a level-1
  solution, never the distractors.
- One line per step, GSM8K style: a sentence naming the owner and unit
  with the note inline, `The music room gets 14-5 = <<14-5=9>>9 boxes.`,
  in several wordings per relation (as `gen-word`'s solution wordings).
- Every calculation in a note, with only `+ - * /` and parentheses
  (calc.h). The unknown and its coefficient are words ("13x"), never in a
  note. `--notes off` writes the same lines without the notes, as the
  other generators.
- Ends with `#### answer`, answers whole or with at most two decimals, as
  GSM8K.
- An option for later, for an A/B only: a first line saying what is
  needed ("We need the first weekend's count; we know the total."). GSM8K
  solutions do not do this, so it starts off.

## The second solver

As every generator here has: `plan_check`, in its own file
(`src/plan_check.h`), sharing no code with the generator. If it disagrees,
the program stops instead of writing the problem.

- **Input.** The generator gives it the stated facts: for each stated
  relation its kind, the quantities it links and its number, and which
  quantity is asked. A test checks that these numbers are exactly the
  question's numbers, in order (read from the text with the labels'
  number reader, `labels::detail::numbers_in`), so the facts cannot
  drift from the text.
- **Forward problems.** It works the facts out by itself: again and again,
  any fact whose inputs are known gives its result, until nothing more
  can be worked out. The asked quantity must be known by then, and equal
  the written answer.
- **Backward problems.** It does not do algebra: it tries every whole
  value (for money, every cent) the unknown's range allows, works every
  fact out forward for each, and exactly one value must satisfy every
  stated number: the written answer. This is `equation_check`'s method.
- **The solution.** Every note must check by calc.h, and every note's
  result must be a value the checker worked out for some quantity (or a
  coefficient of the unknown). That also gives exact labels for each step
  (research plan, idea E) for free.
- **Every stated number used.** Each stated number must belong to some
  fact: no number in the story without a quantity behind it.
- **Tests show it catches mistakes**: a changed number, a wrong answer, a
  wrong note, a backward problem with two solutions, a distractor used in
  the solution.
- **`--exclude`**, as `gen-math`: a problem sharing a 13-word run, or its
  numbers and answer, with a GSM8K test or dev question is drawn again.
  Both `gen-math` topics needed it; this one, built to look like GSM8K,
  will too.

## The re-draw check

Self-train's re-draw check (`src/redraw.h`, README "The re-draw check")
asks whether a try's plan is right by working its calculations out again
on the same story with other numbers. This generator fits it as it is:
- **Story choices** (`Choices::below`): the world, people, the graph's
  shape and relations, the asked quantity, the direction, the hidden
  leaf, the wordings, the sentence and solution order, the number words
  ("twice", "three weekends").
- **Number choices** (`Choices::number_below`): the leaves, or the
  unknown and the other leaves of a backward problem. Worked-out values
  follow from them; a version with a 0, 1, a fraction or a repeated
  number is drawn again (up to 200 draws, as now).
- `redraw::versions_of`'s check stays as it is: a version counts only if
  its question and its solution, numbers masked, are the original's word
  for word.
- New in code: the topic name in `redraw::parse_recipe` and
  `redraw::make`, which list the topics they know.

What the check then sees:
- `847*2` (or `455*3`) gives the wrong answer on every version: dropped
  as `redraw`. That is the wrong plan we want to catch, and the one the
  other checks pass whenever the number happens to be right;
- a right plan passes, through every reading, as the generators' own
  solutions do today (895 of 896);
- a number worked out in the head: the seedling problem's 13, written
  without its note, is a constant to the check, and the multiplier 3 is
  a digit, so it changes between versions and 13 stops being right: the
  try fails. A "twice" is a word and stays in every version, so a
  mental `1+2+4 = 7` would pass. Writing multipliers as digits more
  often makes the check stricter; that is a knob, not a fixed rule.

With varied plans the re-draw check becomes a real **plan check** for
self-train: a right try on a held-out plan shape, which passes every
version, is the model's own plan, not a copy of a template.

## What to measure before it enters any mix

Following the funnel (docs/research-plan.md, "The funnel"), each gate
before the next. Decisions on GSM8K dev, never on GSM8K test.

**Gate 0: the generator itself, CPU only.**
- Tests: every rule above, on every problem of a large test draw; the
  second solver agrees on 1M problems; the checker's catch tests.
- Variety per 1M problems, as counted for the other generators:
  different plan signatures (and per step count), different questions;
  and the cue table above.
- Leaks: with `--exclude`, no GSM8K test or dev question shares a 13-word
  run or its numbers and answer; `leak-check` agrees.
- The re-draw check on the generator's own solutions: at least 99% pass
  with every reading, as the other topics.
- A hand check of 50 problems (30 train, 20 test) read as a person would,
  until a round finds nothing (the other generators needed two or three).

**Gate 1: where rp1 stands, minutes of GPU, in the day only with the
GPU free.** `eval-reasoning` on 500 test-split problems (greedy and
maj@8, with and without `--tools calc-notes`, `--labels`), per step
count, direction and the three test files. If rp1 already solves the
held-out plans, the generator teaches it little; low scores, especially
on backward problems and middle questions, are the room to grow.

**Gate 2: small models from scratch, 3 seeds.** The smallest `probe.sh`
size on the math mix, with a share of `gen-word`'s tokens replaced by
this generator's (the same tokens in all), against the mix as it is, 3
seeds each, 1-2 hours a run. Models this small rarely finish a GSM8K
answer (the unlikelihood runs: 9 right of 750), so judge on generated
test sets: this generator's held-out plans, and `gen-word`'s and
`gen-math`'s test splits, which no arm trains more of. The question:
does training on varied plans transfer to plans and stories it never
saw, without losing the old generators' skills?

**Gate 3: a continuation of rp1, 3 seeds.** Copies of rp1's checkpoint,
1-2 hours each, the same mix against the mix with this generator in place
of part of `gen-word`, at the same tokens, three data-order seeds per arm.
rp1's mix holds the 500 dev problems out (CLAUDE.md, "The model and its
data"), so these continuations can be judged on GSM8K dev (the research
plan's warning that continuations have seen dev was about night 1's
model, trained on all of GSM8K train). Judge:
- dev accuracy, greedy and maj@8, with and without the calculator. Near
  4% on 500 problems one standard error is about 0.9 points, so a gap
  under about 2.5 points in one pair is noise; compare the same problems
  pair by pair, and ask for a gain in all three pairs;
- the failures: `--labels`, and 50 failures per run sorted by hand into
  wrong plan, arithmetic slip, unfinished and other. The bet wins only if
  "wrong plan" falls and accuracy rises;
- what must not get worse: validation loss on the general-text sources,
  `gen-word` and `gen-math` test accuracy;
- the re-draw check on new problems of this generator: the share of
  right tries that pass every version (right by plan).

Only then a night, and only on the owner's decision.

## Related work, checked at the source

**Physics of Language Models, Part 2.1: Grade-School Math and the Hidden
Reasoning Process** (Tian Ye, Zicheng Xu, Yuanzhi Li, Zeyuan
Allen-Zhu; arXiv 2407.20311, July 2024; ICML 2024 tutorial). Read on
arXiv for this design. Its synthetic set iGSM:
- a random "structure graph" of items in four-layer hierarchies (100
  items a layer), whose edges are the stated quantities ("the number of
  backpacks in each dance studio"); a "dependency graph" over them, each
  quantity depending on up to four others; quantities that are sums over
  the hierarchy must be worked out;
- every stated quantity gets one sentence, the sentences are shuffled,
  the question comes before or after; arithmetic is mod 23, so large
  numbers never get in the way of the plan;
- the solution works out only the quantities the question needs, in an
  order that respects the dependencies, one operation at a time;
- training at most 15 operations (iGSM-med) or 21 (iGSM-hard); test
  splits with more operations (20-23, 28-32) and "reask": the same
  problem with another quantity asked. Train and test differ by a hash of
  the solution template mod 23 (below 17 train, 17 and over test);
- by a birthday-paradox count, at least 77 billion solution templates
  at 15 operations: far more than a GPT-2-small (12 layers, about 100M
  parameters) can memorize. That model reaches about 99% in distribution
  and solves longer problems than it trained on;
- probing finds that before writing, the model already knows which
  quantities are needed ("level-1", the shortest solution), and even
  which depend on which ("level-2"); its mistakes come from that plan
  (a quantity taken as ready when it is not); a 4-layer model, even 1,920
  wide, does worse than deeper ones.
What we take: plans from random graphs; the solution covering only
what is needed; distractors that are real quantities; questions about
middle quantities; sentences in shuffled order; the held-out split by a
hash of the plan; counting the different plans. What we do not take: mod
23 (we keep real numbers, and the calculator does the arithmetic), and
its invented hierarchies (we keep stories a person would write). Its
results are on iGSM itself: it shows a small model *can* learn to plan
from such data, not that this transfers to GSM8K. Gates 2 and 3 test
that.

**Part 2.2: How to Learn From Mistakes on Grade-School Math Problems**
(the same authors; arXiv 2408.16293, August 2024): pretraining data with
a wrong step followed by its correction did better than the same amount
of data without mistakes, on iGSM. It is behind our planted mistakes
(`gen-reasoning --mistakes`, research plan idea B); this generator could
plant a wrong *plan* step the same way later.

**GSM-Symbolic** (Iman Mirzadeh, Keivan Alizadeh, Hooman Shahrokhi,
Oncel Tuzel, Samy Bengio, Mehrdad Farajtabar; arXiv 2410.05229, October
2024; ICLR 2025): 100 GSM8K test questions made into templates, 50 versions
each. Changing only the numbers lowers every model's score, and so do
more clauses. Its GSM-NoOp adds one clause that seems to matter and does
not ("five of them were a bit smaller than average"); scores fall by up
to 65%, models subtracting the five. Why our distractors are quantities
of the story, not stock sentences. Its method of changing only the
numbers is our re-draw check's.

**TinyGSM** (Bingbin Liu and others; arXiv 2312.09241, December 2023):
12.3 million problems with Python solutions, all written by GPT-3.5; a
1.3B model with a 1.3B verifier reaches 81.5% on GSM8K. The kind of
result other small-model work gets from another company's model. Our
data rule excludes it (README, "Data provenance"), so we need our own
programs to supply the variety. This generator's output is our own
program's text, so it is clean under that rule; only the ideas above are
taken from the papers, no text.

## Size and the smallest useful version

For scale: the equations and fractions topics were both built in one day
by one session (2026-10-16, about 700-1,000 lines of generator, 200-300 of tests, CLI
tests and a README section), and the number theory topic (PR #87) in one
session. This generator is bigger: a graph, a backward solver, story
rendering from the graph, and plans held out.

**First version, about 3 days of work.** Counts and money; the relations
marked "yes" in the table (given, sum, more/fewer, how many more,
multiple, each, equal share, a sequence of 2-4 terms); 2-6 operations;
forward and backward (undo and let-x); questions about middle quantities;
one or two real distractor relations; four to six worlds; the plan split
and the wording split; `plan_check`; recipes and the re-draw check;
`--exclude`; tests, CLI tests, a README section; Gate 0. As a `gen-math`
topic (`gen-math --topic plans`, `src/mathgen_plans.h`), it reuses the
command's options, files and `--exclude`.

**Later, about 4-6 more days.** Parts and percentages, rates, unit
changes, payments and change; time, length and weight; up to 8
operations; more worlds; two relations in one sentence and relations
inside the question; the plan line option; a planted wrong plan step;
per-knob accuracy for "a generator at the edge of ability" (research plan
idea D: steps, direction, middle questions, distractors are its knobs).

**The experiments.** Gate 1 is minutes of GPU; Gate 2 is about six runs
of 1-2 hours (a night); Gate 3 about six more. Each needs the GPU free
and the owner's go.
