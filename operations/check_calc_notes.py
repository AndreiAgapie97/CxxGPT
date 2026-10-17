"""Checks every calculator note (<<expr=result>>) in JSON Lines problem files.

    python -I operations/check_calc_notes.py FILE.jsonl ...

Each note's expression is evaluated with exact fractions (build_gsm8k_notes's
evaluator, not calc.h) and its result compared with what the calculator would
write. Prints, per file, how many notes are right as the calculator writes
them, right but spelled another way (144.00), wrong, or unreadable to the
calculator, with examples; exits 1 if any note is wrong.
"""
import collections
import fractions
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder out
from build_gsm8k_notes import NOTE, evaluate, written  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    wrong_anywhere = False
    for path in sys.argv[1:]:
        counts, examples = collections.Counter(), collections.defaultdict(list)
        with open(path, encoding='utf-8') as f:
            for line in f:
                answer = json.loads(line)['answer']
                counts['problems'] += 1
                for m in NOTE.finditer(answer):
                    counts['notes'] += 1
                    result = m.group(2).strip()
                    v = evaluate(m.group(1))
                    w = None if v is None else written(v)
                    if w is None:
                        kind = 'unreadable or repeating (the calculator leaves it)'
                    elif w == result:
                        kind = 'right'
                    elif re.fullmatch(r'-?(\d+\.?\d*|\.\d+)', result) and fractions.Fraction(result) == v:
                        kind = 'right, spelled another way'
                    else:
                        kind = 'WRONG'
                        wrong_anywhere = True
                    counts[kind] += 1
                    if kind != 'right' and len(examples[kind]) < 3:
                        examples[kind].append(answer[max(0, m.start() - 40):m.end() + 10].replace('\n', ' | '))
        print(path, dict(counts))
        for kind, found in examples.items():
            for e in found:
                print(f'   {kind}: ...{e}...')
    sys.exit(1 if wrong_anywhere else 0)


if __name__ == '__main__':
    main()
