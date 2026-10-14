# Calculator during generation

`eval-reasoning --tools calc-notes` fills in GSM8K notes: after `<<48/2=`
it writes `24>>`. `--tools calc-plain` fills in an arithmetic expression
ending in ` = `: `16 x 7 = ` becomes `16 x 7 = 112`. `sample` accepts the
same options. Both default to `--tools none`, preserving existing scores.

Evaluation prints the ordinary score first, then the assisted score on the
same problems with the same batch and token budget, plus the numbers of
successful and malformed calls. Sampling prints the counts to stderr so
stdout remains the generated text. Sampling has no gold answers to score.

Result tokens are fed through the model's normal cache before it continues.
They count towards `--max-new` (or sample's `--n`) and the context limit.
If a whole result will not fit, the call is left alone and is not counted
as fired or malformed. Every batch slot has its own pending result. A BPE
token containing the trigger and more punctuation is split, preserving its
trailing text after the result. Tool tokens do not use random draws.

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
parentheses/signs. Wider intermediates detect overflow before narrowing;
there is no floating-point rounding in the evaluator.

The plain detector uses the arithmetic suffix before the exact delimiter
` = `. It requires an operator; a bare `24 = ` or prose assignment does not
invoke the calculator. Unsupported adjacent symbols are rejected instead
of extracting a smaller, misleading expression. Unknown symbol boundaries
(such as `@` or `&`) and a word between numeric operands (such as `2 X 3`
or `2 \times 3`) are rejected; ordinary preceding prose is still allowed. A leading unary sign must
be attached to its number at a clear clause boundary (start, newline, colon
or currency marker), as in `-3 * 2 = `; this avoids treating `a -3*2` as
independent arithmetic. Parentheses permit spaced unary signs as well. Notes mode only accepts the evaluator's `*` spelling.

## Multiplication spellings

Plain mode normalizes `x`, `*` and the typographic equivalent `×` to `*`.
Lowercase `x` is accepted only within the numeric expression, not as a named
value; uppercase `X` and LaTeX `\\times` are not recognized.

This choice follows the public upstream examples below, read as documentation
and prompt source; no dataset was downloaded for this change:

- [GSM8K README](https://github.com/openai/grade-school-math/blob/master/README.md)
  uses `$50 x 3 = $<<50*3=150>>150`: `x` in prose and `*` in notes.
- [MetaMath GSM8K answer-generation prompt](https://github.com/meta-math/MetaMath/blob/main/code_for_generating_data/configs/ansaug_cot_gsm8k.txt)
  includes `3 hours x 2 chapters = 6 hours` and `25*2= 50`. Units inside an
  expression and delimiters without the surrounding spaces remain outside
  this first plain-mode grammar.
- [NeMo-Skills GSM8K solution examples](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/prompt/few_shot_examples/gsm8k.py),
  from the OpenMathInstruct-2 project's repository, include `5 * 4 = 20`
  and `5 x 3 = 15`. This is evidence from public solution prompts, not a
  frequency survey of the OpenMath2 dataset. `×` is additionally accepted
  as an unambiguous typographic alias; no dataset-frequency claim is made.

These deliberately narrow rules will miss equations containing currency
symbols on both operands, worded units or mathematical markup. They avoid
silently inventing the meaning of unsupported text.
