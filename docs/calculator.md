# Calculator during generation

`eval-reasoning --tools calc-notes` fills in GSM8K notes: after `<<48/2=`
it writes `24>>`. `--tools calc-plain` fills in an arithmetic expression
ending in ` = `: `16 x 7 = ` becomes `16 x 7 = 112`, and `$50 x 3 = $`
becomes `$50 x 3 = $150`. `sample` accepts the same options. Both default
to `--tools none`, preserving existing scores. The code is `src/calc.h`
(the arithmetic and the two detectors) and the hook in `generate()`
(`src/generate.h`).

Evaluation prints the ordinary score first, then the assisted score on the
same problems with the same batch and token budget, plus the numbers of
successful and malformed calls. Sampling prints the counts to stderr so
stdout remains the generated text. Sampling has no gold answers to score.

Result tokens are fed through the model's normal cache before it continues.
They count towards `--max-new` (or sample's `--n`) and the context limit.
If a whole result will not fit, the call is left alone and is not counted
as fired or malformed. Every batch slot has its own pending result. Tool
tokens do not use random draws.

The tokenizer joins a space to the punctuation after it, so a BPE token
can hold the trigger and more: `= $150` arrives as ` =`, ` $`, and a
negative answer as ` =`, ` -`; notes can merge `=-` or `=.`. The result
then replaces the rest of that token. The rest was the model's own start
of the answer (a sign, a point), and keeping it would garble the number
(`= 150$150`). The prefixes of a token are tried longest first, so ` $`
keeps its `$` (` = $` is a trigger) while ` -` falls back to its space.

The calculator watches the prompt as well as the generated text, so a
prompt ending in `16 x 7 =` fires when the model writes ` `. Calls are
still checked only after a model token, never after the prompt alone or
after the calculator's own result. A BOS token starts the watched text
afresh, as it starts a new document.

## Not on the reasoning curriculum

Neither mode fires on the synthetic curriculum (`gen-reasoning`), so
`eval-reasoning --tools ...` on a curriculum test file prints two equal
scores. Its solutions write steps as `47*6=240+42=282` and `c=a+b=5+8=13`: no
spaces around `=` for plain mode, and no `<<` for notes mode. This is on
purpose, for now: a third mode for that format is not worth adding before
the calculator has shown its value on GSM8K-style data, where the model
learns the notes from the training text. Use the tool on `prep-qa` or
`gen-word` test files instead.

## Arithmetic and limits

The evaluator accepts integers, decimal literals, unary signs, `+ - * /`,
parentheses and whitespace, with ordinary precedence. It keeps exact
fractions throughout: `7/2*2` is `7`, and even `1/3*3` is `1`. Answers have
no trailing decimal zeros. A final repeating decimal such as `1/3` is
rejected because there is no exact finite decimal to insert.

Digit-grouping commas are read as part of the number (since 2026-10-17):
`<<24*600,000=` is finished as `14400000>>`. The model copies numbers
from the question with their commas ('24*600,000', '100,000*30'); on
2026-10-17, 13 of the 75 refused notes on the 383 textbook problems were
this. Only a strict group counts: a number that starts with 1-3 digits,
the first not 0 and with no digit or `.` right before it, then one or more
`,ddd`, and no digit right after (`1,000`, `24,600,000`, `1,234.5`). Any
other comma stays and the call is malformed: `1,2345`, `1,2`, `2,3+4` (a
list), `1234,567`, `1,234,56`, `1.5,000`, `0,125` (a decimal comma in some
countries), `1, 000`. The result is written without commas, as always.
`calc::ungroup` removes the commas before the parser sees the expression;
`operations/build_gsm8k_notes.py`'s `ungroup` is the same rule as a regular
expression, and `tests/test_calc_notes.py` checks the two agree. No note
in the training data has a comma (GSM8K's 28,000 train and test notes, the
generated note files), so the note builders and checker give the same
files and counts as before. The failure labels (`--labels`) already read
the work without grouping commas (`strip_digit_commas`, a looser rule: it
also drops the comma of `1234,567`), so they read these notes as before.

Plain mode keeps its rule: a comma ends the expression there, so
`1,000 * 2 = ` and `24 * 600,000 = ` stay malformed (see below). Plain
mode has no `<<` to say where the expression starts, and reading through
a comma would have to tell `8,000 * 2` from a list such as `8,000 and
8 * 2`, `3, 8 * 2` or `2,3 * 2`; notes mode does not have that problem.

There are no variables, exponentiation, percentages,
units inside expressions, LaTeX commands or implicit multiplication. Invalid
calls remain unchanged, letting the model continue. Numeric overflow and
division by zero also count as malformed. Reduced numerators and denominators
must fit in signed 64 bits (excluding its asymmetric minimum); literals must
fit before reduction. Expressions are limited to 4096 bytes and 128 nested
parentheses/signs. All arithmetic is in 64-bit integers with overflow
checks (`__builtin_mul_overflow`, `__builtin_add_overflow`), cancelling
common factors before multiplying, so a product overflows only when the
exact answer does not fit. An addition also checks its scaled numerators,
which can reject an answer that would fit only for values close to 2^63.
There is no floating-point rounding and no 128-bit arithmetic (128-bit
division needs runtime helpers that the Windows MSVC runtime lacks).

The plain detector uses the arithmetic suffix before the exact delimiter
` = ` (or ` = $`). It requires an operator; a bare `24 = ` or prose
assignment does not invoke the calculator. Unsupported adjacent symbols are
rejected instead of extracting a smaller, misleading expression. Unknown
symbol boundaries (such as `@` or `&`) and a word between numeric operands
(such as `2 X 3` or `2 \times 3`) are rejected; ordinary preceding prose is
still allowed.

The expression may start where a clause starts: at the start of the text,
right after a newline, `:` or `$`, or after one of `. , ; ! ?` followed by
whitespace. So `He has 3 bags. 16 * 7 = `, `So, 16 * 7 = ` and
`8, and 8 * 2 = ` compute `16 * 7` and `8 * 2`. Without the whitespace the
punctuation is no boundary (`1,000 * 2 = ` is rejected: plain mode reads
no grouping commas; `3.5 * 2 = ` is `7`). A side effect: in `16 * 7. = ` the `. ` reads as a sentence end, so
it is no call.

A leading unary sign must be attached to its number where a clause starts,
as in `-3 * 2 = `; this avoids treating `a -3*2` as independent arithmetic.
A sign right after an operator is rejected (`3 * -2 = `): it is how algebra
such as `3x + 2` would slip through as `3 * (+2)`. Inside parentheses a
sign is fine: `3 * (-2) = ` is `-6`.

Notes mode only accepts the evaluator's `*` spelling. A note is a call only
at its first `=`: a note left open (`<<5/0=unknown`, no `>>`) is counted as
malformed there once, and a later `=` in the same open note is no call.

## Multiplication spellings

Plain mode normalizes `x`, `*` and the typographic equivalent `×` to `*`.
Lowercase `x` is a times sign only with a space on each side and a digit
beyond each space, as in `16 x 7`. Any other `x` is a letter: `3x + 2 = `
and `2x - 4 = ` are algebra, counted as malformed like `a + 3 = `, and
`3x = ` is no call. Uppercase `X` and LaTeX `\\times` are not recognized.

This choice follows GSM8K's own documentation and the forms worked
solutions commonly write:

- [GSM8K README](https://github.com/openai/grade-school-math/blob/master/README.md)
  uses `$50 x 3 = $<<50*3=150>>150`: `x` in prose and `*` in notes.
- Written solutions also use `5 * 4 = 20` and `5 x 3 = 15`, and sometimes
  `3 hours x 2 chapters = 6 hours` or `25*2= 50`. Units inside an
  expression and delimiters without the surrounding spaces remain outside
  this first plain-mode grammar. `×` is additionally accepted as an
  unambiguous typographic alias; no dataset-frequency claim is made.

These deliberately narrow rules will miss equations containing currency
symbols on both operands, worded units or mathematical markup. They avoid
silently inventing the meaning of unsupported text.

## Notes in the training data

The notes mode needs a model that writes `<<48/2=`. Until 2026-10-17 no
training file showed it one: `prep-qa` removes GSM8K's notes unless
`--keep-calc`, and the generators ran with `--notes off`. That is why
`calc-notes` never fired on GSM8K with the RunPod model, and why
`calc-plain` fired on whatever the model happened to write (409 calls on
the 383 textbook problems, 217 of them malformed).

Night 2 (2026-10-17) trains on the same math problems as planned, with
their notes kept:

| File | Problems | Tokens without, with notes |
|---|---|---|
| `data/gsm8k-notes/gsm8k-train.tok` (`operations/build_gsm8k_notes.py`) | 6,966 | 1.24M, 1.48M |
| `data/thin-notes-2026-10-17/word-notes-5.tok` (a fifth of `data/word-notes-1`) | 486,603 | 99.8M, 127.2M |
| `data/thin-notes-2026-10-17/eq-train-3.tok` (a third of `data/gen-math-notes`) | 165,830 | 37.3M, 44.1M |
| `data/thin-notes-2026-10-17/frac-train-3.tok` | 166,358 | 20.5M, 26.0M |

The checks:
- **The same problems.** For each file, `prep-qa` without `--keep-calc` on
  the new JSON Lines gives the file it replaces byte for byte. The
  generators write the same problems with notes on and off, and the
  builders' overlap filters read each solution without its notes.
- **Every note's result is exact.** A separate checker
  (`operations/check_calc_notes.py FILE.jsonl ...`) evaluates each note
  with exact fractions and calc.h's 64-bit limits, without calc.h's code
  (`tests/test_calc_notes.py` checks the two agree), and compares the
  result with what the calculator writes. In the generated files it found 15.6
  million notes, none wrong.
  GSM8K's own 22,088 train notes have no wrong result either. 638 of them
  write the result another way (`144.00`, `.4`), and the builder rewrites
  those as the calculator does (`144`, `0.4`), inside the note only. The
  2 notes with `//` are left as they are.
- **All of it is trained.** Every added token is an answer token.

## In training's exams

`train --reason-tools calc-notes` (or `calc-plain`; default `none`) takes
each `--reason-test` exam a second time, with the calculator, as
`eval-reasoning --tools` does. The exam without it prints as before, and
the dashboard and `scripts/report.sh` read that one. The second exam is one
line of its own, which `report.sh` shows in a column "with calculator":

    reasoning with calculator (calc-notes) at step 58866: 2.8% (7 / 250), calls 417 fired, 12 malformed (123.9 s)

The CSV log gets `# reasoning step N with calculator: c/t`. On the night
model (rp1 after 82 steps of night 2's mix with notes, 250 GSM8K dev
problems on the GPU), the exam took 23 s and the one with the calculator
124 s (why it is five times slower is not yet measured). With an exam
every 500 steps that is about 1 minute in 18.
