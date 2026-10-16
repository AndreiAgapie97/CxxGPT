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

There are no variables, exponentiation, thousands separators, percentages,
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
punctuation is no boundary (`1,000 * 2 = ` is rejected, `3.5 * 2 = ` is
`7`). A side effect: in `16 * 7. = ` the `. ` reads as a sentence end, so
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
