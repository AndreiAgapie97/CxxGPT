"""Builds data/gen-math: gen-math's equations and fractions, tokenized for night.sh's mix.

For each topic:

  TOPIC-dev    1,000 problems from the held-out (test) sentence patterns
  TOPIC-train  500,000 problems from the training patterns

gen-math --exclude already draws a problem again when its question shares a
13-word run with GSM8K test or dev (or the topic's dev set), or has one of
their questions' numbers and answer. This script checks that again in Python
on the question and the solution (build_targeted_math.py's rule) and drops
any problem that fails. No 13-word run is ever found; the numbers rule here
is cruder than gen-math's (it reads only digits, so "4/5 of 360 stickers,
6 pictures" has a GSM8K question's 4, 5 and 6), and drops about 0.2-0.5% of
the problems, harmless ones. Solutions have no calculator notes, like
data/targeted-math's word problems, unless --notes on: then they have
GSM8K's (48/2 = <<48/2=24>>24), kept in the tokens for eval-reasoning
--tools calc-notes. The problems are the same either way (the generator
draws the same ones, and the checks here read solutions without notes).

It writes into a new folder (--out) and never touches another one. Usage
(from the repository root):

  python -I operations/build_gen_math.py --cxxgpt build/cxxgpt.exe \\
      --heldout data/targeted-math/heldout.jsonl --gsm8k data/licensed-10b \\
      --out data/gen-math --tok data/licensed-10b/tok.bpe
"""
import argparse
import datetime
import hashlib
import json
import pathlib
import re
import struct
import subprocess

COUNTS = {'dev': 1000, 'train': 500000}
SEEDS = {('equations', 'dev'): 202610171, ('equations', 'train'): 202610170,
         ('fractions', 'dev'): 202610173, ('fractions', 'train'): 202610172}
SHORT = {'equations': 'eq', 'fractions': 'frac'}


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def unescape(s):
    esc = {'n': '\n', 't': '\t', '\\': '\\'}
    out, i = [], 0
    while i < len(s):
        if s[i] == '\\' and i + 1 < len(s):
            out.append(esc[s[i + 1]])
            i += 2
        else:
            out.append(s[i])
            i += 1
    return ''.join(out)


def numbers(text):
    """The numbers written in a text, as sorted strings ("1,000" is 1000)."""
    return sorted(n.replace(',', '') for n in re.findall(r'\d(?:[\d,]*\d)?(?:\.\d+)?', text))


def grams(s):
    words = re.findall(r'[a-z0-9]+', s.lower())
    return {' '.join(words[i:i + 13]) for i in range(len(words) - 12)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cxxgpt', required=True)
    ap.add_argument('--heldout', required=True, help='GSM8K test and dev as JSON Lines')
    ap.add_argument('--gsm8k', required=True, help="the folder with gsm8k-test.txt and gsm8k-dev.txt")
    ap.add_argument('--out', required=True, help='a new folder')
    ap.add_argument('--tok', required=True, help="the RunPod model's tokenizer")
    ap.add_argument('--train-count', type=int, default=COUNTS['train'])
    ap.add_argument('--notes', choices=['off', 'on'], default='off', help='calculator notes, kept in the tokens')
    args = ap.parse_args()
    COUNTS['train'] = args.train_count
    out = pathlib.Path(args.out)
    args.cxxgpt = str(pathlib.Path(args.cxxgpt).resolve())  # Windows does not search relative paths
    assert not out.exists(), f'{out} exists; choose a new folder'
    out.mkdir(parents=True)
    gsm8k = pathlib.Path(args.gsm8k)
    exclude = [str(gsm8k / 'gsm8k-test.txt'), str(gsm8k / 'gsm8k-dev.txt')]

    held, held_numbers = set(), set()
    for line in pathlib.Path(args.heldout).read_text(encoding='utf-8').splitlines():
        problem = json.loads(line)
        held.update(grams(problem['question']))
        key = numbers(problem['question'])
        if len(key) >= 2:  # one number alone says too little
            answer = problem['answer'].split('####')[-1].strip().replace(',', '')
            held_numbers.add((tuple(key), answer))

    def same_numbers(question, answer):
        key = numbers(question)
        keys = [key] + [key[:i] + key[i + 1:] for i in range(len(key))]
        return any((tuple(k), answer) in held_numbers for k in keys)

    records = {}
    for topic in ['equations', 'fractions']:
        dev_grams = set()
        for split, style in [('dev', 'test'), ('train', 'train')]:
            name = SHORT[topic] + '-' + split
            prefix, raw = out / name, out / ('generated-' + name)
            seed = SEEDS[(topic, split)]
            files = exclude + ([str(out / ('generated-' + SHORT[topic] + '-dev.txt'))] if split == 'train' else [])
            run = subprocess.run([args.cxxgpt, 'gen-math', '--topic', topic, '--out', str(raw),
                                  '--count', str(COUNTS[split]), '--seed', str(seed), '--split', style,
                                  '--notes', args.notes, '--exclude', ','.join(files)],
                                 check=True, capture_output=True, text=True)
            print(run.stdout.strip(), flush=True)
            kept = dropped = dropped_numbers = 0
            with open(raw.with_suffix('.txt'), encoding='utf-8', newline='') as f, \
                    open(prefix.with_suffix('.jsonl'), 'w', encoding='utf-8', newline='\n') as dest:
                for line in f:
                    _, _, q, steps, answer = line.rstrip('\n').split('\t')
                    q, steps, answer = unescape(q), unescape(steps), unescape(answer)
                    g = grams(q + ' ' + re.sub(r'<<[^<>]*>>', '', steps))  # as prep-qa strips notes
                    if g & held or (split == 'train' and g & dev_grams):
                        dropped += 1
                        continue
                    if same_numbers(q, answer):
                        dropped_numbers += 1
                        continue
                    if split == 'dev':
                        dev_grams.update(grams(q))
                    dest.write(json.dumps({'question': q, 'answer': steps + '\n#### ' + answer},
                                          ensure_ascii=False) + '\n')
                    kept += 1
            raw.with_suffix('.text').unlink(missing_ok=True)
            if split == 'train':
                raw.with_suffix('.txt').unlink(missing_ok=True)
            subprocess.run([args.cxxgpt, 'prep-qa', '--in', str(prefix.with_suffix('.jsonl')), '--out', str(prefix),
                            '--bpe', args.tok, '--task', topic] + (['--keep-calc'] if args.notes == 'on' else []),
                           check=True)
            with open(prefix.with_suffix('.tok'), 'rb') as f:
                magic, _, masked, tokens = struct.unpack('<8sIIQ', f.read(24))
            assert magic == b'CXTOKEN1' and masked == 1
            records[name] = {'topic': topic, 'seed': seed, 'split_patterns': style, 'questions': kept,
                             'dropped_near_heldout': dropped, 'dropped_same_numbers_and_answer': dropped_numbers,
                             'tokens': tokens, 'sha256': sha256(prefix.with_suffix('.tok'))}
            print('ready', name, records[name], flush=True)
        (out / ('generated-' + SHORT[topic] + '-dev.txt')).unlink(missing_ok=True)

    # The generators' hashes as git stores them (LF line ends).
    sources = {f: hashlib.sha256(pathlib.Path(f).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
               for f in ['src/mathgen.h', 'src/mathgen_fractions.h']}
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    manifest = {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'generator': 'CxxGPT gen-math --topic equations and fractions; exact arithmetic, every problem solved '
                     'again by a second solver; ' +
                     ('calculator notes kept' if args.notes == 'on' else 'no calculator notes'),
        'cxxgpt_sha256': sha256(args.cxxgpt),  # the program that ran, which can be older than the checkout
        'source': 'locally generated from repository templates, not an externally downloaded dataset',
        'generator_source_sha256': sources,
        'generator_commit': commit,
        'tokenizer_sha256': sha256(args.tok),
        'split': 'generator train/test sentence patterns; distinct seeds; gen-math --exclude against GSM8K test '
                 'and dev (and the topic dev set for train), then checked again: no 13-word run of question or '
                 "solution shared with GSM8K dev/test (or the dev set), no GSM8K dev/test question's numbers "
                 '(with or without any one of them) and answer',
        'splits': records,
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('done', out, flush=True)


if __name__ == '__main__':
    main()
