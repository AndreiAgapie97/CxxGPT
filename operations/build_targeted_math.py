"""Builds data/targeted-math's generated word problems with the current gen-word.

The RunPod scripts (prepare_targeted_math.py, rebuild_targeted_math.py) made
the first set and can only recreate it byte for byte. This one makes a new
set with whatever src/wordgen.h the cxxgpt binary was built from (after the
2026-10-16 audit), with the same seeds, counts and filters:

  word-dev    1,000 problems from the held-out (test) sentence patterns
  word-train  1,000,000 problems from the training patterns, minus any whose
              question or solution shares a 13-word run with a held-out
              question (heldout.jsonl: GSM8K test and dev) or with word-dev,
              and (new) any with a held-out question's numbers and answer,
              counting the question's numbers with or without any one of
              them (a distractor sentence adds one): gen-math --exclude's
              rule. "2 pencils at $5, pays $20" is GSM8K test's "5 crayons
              at $2, pays $20" in other words.

It writes into a new folder (--out) and never touches the old one; gsm8k-train.tok
and heldout.jsonl are copied over unchanged, so the folder can replace
data/targeted-math as a whole. Usage (from the repository root):

  python -I operations/build_targeted_math.py --cxxgpt build/cxxgpt.exe \\
      --old data/targeted-math --out data/targeted-math-new --tok data/licensed-10b/tok.bpe

Fresh problems for another night, so the word problems need not repeat:
--seed and --count choose the training problems, and --dev-from FOLDER copies
that folder's word-dev (the one the runs validate on) instead of generating
it, so the new training set is filtered against the same dev set:

  python -I operations/build_targeted_math.py --cxxgpt build/cxxgpt.exe \\
      --old data/targeted-math --out data/word-fresh-1 --tok data/licensed-10b/tok.bpe \\
      --dev-from data/targeted-math --seed 202610170 --count 2500000

The cxxgpt given must be built from the checkout: the manifest records the
source's hash, and a program built before the last change to src/wordgen.h
writes other problems. One check: with --seed 202610160, the first problems
gen-word writes are data/targeted-math's (those its filter kept).
"""
import argparse
import datetime
import hashlib
import json
import pathlib
import re
import shutil
import struct
import subprocess


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
    ap.add_argument('--old', required=True, help='the current data/targeted-math (read only)')
    ap.add_argument('--out', required=True, help='a new folder for the rebuilt set')
    ap.add_argument('--tok', required=True, help="the RunPod model's tokenizer")
    ap.add_argument('--seed', type=int, default=202610160, help="the training problems' seed")
    ap.add_argument('--count', type=int, default=1000000, help='training problems to generate')
    ap.add_argument('--dev-from', help='copy word-dev from this folder instead of generating it')
    args = ap.parse_args()
    old, out = pathlib.Path(args.old), pathlib.Path(args.out)
    args.cxxgpt = str(pathlib.Path(args.cxxgpt).resolve())  # Windows does not search relative paths
    assert not out.exists(), f'{out} exists; choose a new folder'
    out.mkdir(parents=True)
    for name in ['gsm8k-train.tok', 'heldout.jsonl']:
        shutil.copy2(old / name, out / name)

    held, held_numbers = set(), set()
    for line in (out / 'heldout.jsonl').read_text(encoding='utf-8').splitlines():
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
    if args.dev_from:
        dev_from = pathlib.Path(args.dev_from)
        for suffix in ['.jsonl', '.tok', '.txt']:
            shutil.copy2(dev_from / ('word-dev' + suffix), out / ('word-dev' + suffix))
        for line in (out / 'word-dev.jsonl').read_text(encoding='utf-8').splitlines():
            held.update(grams(json.loads(line)['question']))
        records['dev'] = json.loads((dev_from / 'manifest.json').read_text(encoding='utf-8'))['splits']['dev']
        assert records['dev']['sha256'] == sha256(out / 'word-dev.tok'), 'word-dev.tok differs from its manifest'
        records['dev']['copied_from'] = str(dev_from)
    splits = [] if args.dev_from else [('dev', 202610161, 1000, 'test')]
    for split, seed, count, style in splits + [('train', args.seed, args.count, 'train')]:
        prefix, raw = out / ('word-' + split), out / ('generated-' + split)
        subprocess.run([args.cxxgpt, 'gen-word', '--out', str(raw), '--count', str(count), '--seed', str(seed),
                        '--split', style, '--notes', 'off'], check=True)
        kept = dropped = dropped_numbers = 0
        with open(raw.with_suffix('.txt'), encoding='utf-8', newline='') as f, \
                open(prefix.with_suffix('.jsonl'), 'w', encoding='utf-8', newline='\n') as dest:
            for line in f:
                _, _, q, steps, answer = line.rstrip('\n').split('\t')
                q, steps, answer = unescape(q), unescape(steps), unescape(answer)
                if split == 'train' and grams(q + ' ' + steps) & held:
                    dropped += 1
                    continue
                if split == 'train' and same_numbers(q, answer):
                    dropped_numbers += 1
                    continue
                dest.write(json.dumps({'question': q, 'answer': steps + '\n#### ' + answer},
                                      ensure_ascii=False) + '\n')
                kept += 1
        raw.with_suffix('.text').unlink(missing_ok=True)
        raw.with_suffix('.txt').unlink(missing_ok=True)
        subprocess.run([args.cxxgpt, 'prep-qa', '--in', str(prefix.with_suffix('.jsonl')), '--out', str(prefix),
                        '--bpe', args.tok, '--task', 'word'], check=True)
        if split == 'dev':  # keep the generated dev set out of training too
            for line in prefix.with_suffix('.jsonl').read_text(encoding='utf-8').splitlines():
                held.update(grams(json.loads(line)['question']))
        with open(prefix.with_suffix('.tok'), 'rb') as f:
            magic, _, masked, tokens = struct.unpack('<8sIIQ', f.read(24))
        assert magic == b'CXTOKEN1' and masked == 1
        records[split] = {'seed': seed, 'questions': kept, 'dropped_near_heldout': dropped,
                          'dropped_same_numbers_and_answer': dropped_numbers, 'tokens': tokens,
                          'sha256': sha256(prefix.with_suffix('.tok'))}
        print('ready', split, records[split], flush=True)

    # The generator's hash as git stores it (LF line ends), as the RunPod manifest has it.
    source = pathlib.Path('src/wordgen.h').read_bytes().replace(b'\r\n', b'\n')
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    manifest = {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'generator': 'CxxGPT gen-word after the 2026-10-16 audit; exact arithmetic; no calculator notes',
        'source': 'locally generated from repository templates, not an externally downloaded dataset',
        'generator_source_sha256': hashlib.sha256(source).hexdigest(),
        'generator_commit': commit,
        # The source hash above is the checkout's; this is the program that ran, which
        # can be older than the checkout if it was not rebuilt.
        'cxxgpt_sha256': sha256(args.cxxgpt),
        'tokenizer_sha256': sha256(args.tok),
        'split': 'generator train/test sentence templates; distinct seeds; train rejects 13-word overlaps with '
                 "GSM8K dev/test and generated word dev, and problems with a GSM8K dev/test question's numbers "
                 '(with or without any one of them) and answer',
        'replaces': json.loads((old / 'manifest.json').read_text(encoding='utf-8')),
        'splits': records,
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    recipe = json.loads((old / 'balanced-recipe.json').read_text(encoding='utf-8'))
    recipe['weighted_word_tokens'] = records['train']['tokens'] * recipe['word_weight']
    (out / 'balanced-recipe.json').write_text(json.dumps(recipe, indent=2), encoding='utf-8')
    print('done', out, flush=True)


if __name__ == '__main__':
    main()
