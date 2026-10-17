"""GSM8K train and dev with their calculator notes kept, for eval-reasoning --tools calc-notes.

GSM8K's solutions carry notes ("48/2 = <<48/2=24>>24"), and prep-qa removes
them unless --keep-calc. A model that never sees them never writes one, so
the calculator's notes mode never fires. This script makes the same files as
the RunPod data, with the notes:

  gsm8k-train  GSM8K train without the 500 dev problems (those in
               heldout.jsonl) and the 7 whose question shares a 13-word run
               with a GSM8K dev or test question: 6,966 problems
  gsm8k-dev    the 500 dev problems (heldout.jsonl without GSM8K test), for
               validation

Inside each note, the result is written as the calculator writes it ("144.00"
is 144, ".4" is 0.4), so training shows what eval-reasoning's calculator will
put there; the text after ">>" stays as GSM8K wrote it. The results are
checked with exact fractions (here, not with calc.h); a note whose result is
wrong, or that the calculator cannot read ("560//10"), is left as it is and
counted.

The check that nothing else changed: prep-qa without --keep-calc on the new
JSON Lines gives --check-train and --check-dev byte for byte (the RunPod
model's files). Usage, from the repository root:

  python -I operations/build_gsm8k_notes.py --cxxgpt build/cxxgpt.exe --tok data/licensed-10b/tok.bpe \\
      --train data/gsm8k/train.jsonl --test data/gsm8k/test.jsonl --heldout data/targeted-math/heldout.jsonl \\
      --check-train data/targeted-math/gsm8k-train.tok --check-dev data/licensed-10b/gsm8k-dev.tok \\
      --out data/gsm8k-notes
"""
import argparse
import collections
import datetime
import fractions
import json
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # -I leaves the script's folder out
from build_targeted_math import grams, sha256  # noqa: E402

NOTE = re.compile(r'<<([^<>]*?)=([^<>]*)>>')
TOKEN = re.compile(r'\s*(\d+\.\d+|\d+\.|\d+|\.\d+|[-+*/()])')


def evaluate(expression):
    """The exact value of + - * / ( ) over decimal numbers, or None if it is anything else."""
    tokens, pos = [], 0
    expression = expression.strip()
    while pos < len(expression):
        m = TOKEN.match(expression, pos)
        if not m:
            return None
        tokens.append(m.group(1))
        pos = m.end()
    i = 0

    def peek():
        return tokens[i] if i < len(tokens) else None

    def sum_():
        nonlocal i
        v = product()
        while peek() in ('+', '-'):
            op = tokens[i]
            i += 1
            w = product()
            v = v + w if op == '+' else v - w
        return v

    def product():
        nonlocal i
        v = signed()
        while peek() in ('*', '/'):
            op = tokens[i]
            i += 1
            w = signed()
            v = v * w if op == '*' else v / w  # a division by zero raises
        return v

    def signed():
        nonlocal i
        if peek() in ('+', '-'):
            op = tokens[i]
            i += 1
            v = signed()
            return v if op == '+' else -v
        if peek() == '(':
            i += 1
            v = sum_()
            if peek() != ')':
                raise ValueError
            i += 1
            return v
        if peek() is None or not re.fullmatch(r'[\d.]+', peek()):
            raise ValueError
        i += 1
        return fractions.Fraction(tokens[i - 1])

    try:
        v = sum_()
    except (ValueError, ZeroDivisionError):
        return None
    return v if i == len(tokens) else None


def written(v):
    """v as the calculator writes it (src/calc.h, evaluate): no trailing zeros; None if it repeats."""
    d = v.denominator
    for p in (2, 5):
        while d % p == 0:
            d //= p
    if d != 1:
        return None
    n, den = abs(v.numerator), v.denominator
    out = ('-' if v < 0 else '') + str(n // den)
    rest = n % den
    if rest:
        out += '.'
    while rest:
        out += str(rest * 10 // den)
        rest = rest * 10 % den
    return out


def with_calculator_results(answer, stats):
    """The answer with each note's result as the calculator writes it."""
    def note(m):
        expression, result = m.group(1), m.group(2)
        v = evaluate(expression)
        w = None if v is None else written(v)
        stats['notes'] += 1
        if w is None:
            stats['notes the calculator cannot read or write (left as they are)'] += 1
            return m.group(0)
        if w == result.strip():
            return m.group(0)
        if re.fullmatch(r'-?(\d+\.?\d*|\.\d+)', result.strip()) and fractions.Fraction(result.strip()) == v:
            stats['results respelled (144.00 is 144, .4 is 0.4)'] += 1
            return '<<' + expression + '=' + w + '>>'
        stats['wrong results (left as they are)'] += 1
        return m.group(0)
    return NOTE.sub(note, answer)


def main():
    ap = argparse.ArgumentParser()
    for name in ['cxxgpt', 'tok', 'train', 'test', 'heldout', 'check-train', 'check-dev', 'out']:
        ap.add_argument('--' + name, required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    args.cxxgpt = str(pathlib.Path(args.cxxgpt).resolve())  # Windows does not search relative paths
    assert not out.exists(), f'{out} exists; choose a new folder'

    def read(path):
        return [json.loads(line) for line in pathlib.Path(path).read_text(encoding='utf-8').splitlines()]
    heldout, test = read(args.heldout), read(args.test)
    test_q = {p['question'] for p in test}
    held_q, held = {p['question'] for p in heldout}, set()
    for p in heldout:
        held |= grams(p['question'])
    train = [p for p in read(args.train) if p['question'] not in held_q]
    near = [p for p in train if grams(p['question']) & held]
    train = [p for p in train if not grams(p['question']) & held]
    dev = [p for p in heldout if p['question'] not in test_q]
    assert (len(train), len(near), len(dev)) == (6966, 7, 500), (len(train), len(near), len(dev))

    out.mkdir(parents=True)
    records = {}
    for name, problems, check in [('gsm8k-train', train, args.check_train), ('gsm8k-dev', dev, args.check_dev)]:
        stats = collections.Counter()
        prefix = out / name
        with open(prefix.with_suffix('.jsonl'), 'w', encoding='utf-8', newline='\n') as dest:
            for p in problems:
                dest.write(json.dumps({'question': p['question'],
                                       'answer': with_calculator_results(p['answer'], stats)},
                                      ensure_ascii=False) + '\n')
        subprocess.run([args.cxxgpt, 'prep-qa', '--in', str(prefix.with_suffix('.jsonl')), '--out', str(prefix),
                        '--bpe', args.tok, '--task', 'gsm8k', '--keep-calc'], check=True)
        bare = out / (name + '-without-notes')
        subprocess.run([args.cxxgpt, 'prep-qa', '--in', str(prefix.with_suffix('.jsonl')), '--out', str(bare),
                        '--bpe', args.tok, '--task', 'gsm8k'], check=True)
        same = sha256(bare.with_suffix('.tok')) == sha256(check)
        assert same, f'{name} without its notes differs from {check}'
        for suffix in ['.tok', '.txt']:
            bare.with_suffix(suffix).unlink()
        records[name] = {'questions': len(problems), 'sha256': sha256(prefix.with_suffix('.tok')),
                         'without_notes_equals': f'{check} ({sha256(check)})', **stats}
        print('ready', name, records[name], flush=True)

    manifest = {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source': 'GSM8K (MIT licence), data/gsm8k from scripts/get_data.py',
        'builder': 'operations/build_gsm8k_notes.py',
        'builder_sha256': sha256(__file__),
        'builder_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                                         check=True).stdout.strip(),
        'cxxgpt_sha256': sha256(args.cxxgpt),
        'tokenizer_sha256': sha256(args.tok),
        'held_out': '500 dev and 1,319 test problems (heldout.jsonl) and 7 train near-copies of them',
        'splits': records,
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('done', out, flush=True)


if __name__ == '__main__':
    main()
