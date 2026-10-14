#!/usr/bin/env python3
"""Independent file/regex audit of native curriculum and generated evidence."""
import argparse
import collections
import csv
import hashlib
import io
import pathlib
import re
import struct


def problems(path):
    rows = []
    for line in path.read_text().splitlines():
        task, difficulty, question, steps, answer = line.split('\t')
        rows.append(dict(task=task, difficulty=int(difficulty), question=question, steps=steps, answer=answer))
    return rows


def corpus_questions(path):
    b = path.read_bytes()
    assert b[:8] == b'CXTOKEN1'
    vocab, mask, count = struct.unpack_from('<IIQ', b, 8)
    assert vocab == 256 and len(b) == 24 + count * (2 + bool(mask))
    stream = bytes(x[0] for x in struct.iter_unpack('<H', b[24:24+count*2]))
    assert stream.startswith(b'\x01')
    out = set()
    for doc in stream.split(b'\x01')[1:]:
        assert doc.startswith(b'Q: ') and b'\nA: ' in doc
        out.add(doc[3:].split(b'\nA: ', 1)[0].decode('latin-1'))
    return out


def verify_hashes(path):
    for line in path.read_text().splitlines():
        digest, file = line.split(maxsplit=1)
        assert hashlib.sha256(pathlib.Path(file).read_bytes()).hexdigest() == digest, file


def trace(p, text):
    body = text.split('\n#### ', 1)[0].strip(' \r\n')
    steps = [s.strip(' \r\n') for s in body.split(',' if p['task'] == 'add' else ';')] if body else []
    local = grounded = 0
    if p['task'] == 'add':
        a, b = map(int, p['question'].split('+'))
        carry = 0
        expected = []
        while a or b:
            x, y = a % 10, b % 10
            expected.append((x, y, carry, x+y+carry))
            carry = (x+y+carry) // 10
            a, b = a // 10, b // 10
        numeric = r'[ \r\n]*([0-9]+)[ \r\n]*'
        for s, (x, y, c, total) in zip(steps, expected):
            m = re.fullmatch(numeric + r'\+' + numeric + r'\+' + numeric + '=' + numeric, s)
            if not m:
                continue
            dx, dy, dc, z = map(int, m.groups())
            ok = dx <= 9 and dy <= 9 and dc <= 1 and z == dx+dy+dc
            local += ok
            grounded += ok and sorted((dx, dy)) == sorted((x, y)) and dc == c and z == total
    else:
        m = re.fullmatch(r'max\(([0-9]+),([0-9]+)\)', p['question'])
        a, b = map(int, m.groups())
        sa, sb = str(a), str(b)
        expected = []
        if len(sa) != len(sb):
            expected = [f'{len(sa)} digits vs {len(sb)}']
            if steps:
                m = re.fullmatch(r'([0-9]+)[ \r\n]* digits vs [ \r\n]*([0-9]+)', steps[0])
                if m:
                    x, y = map(int, m.groups())
                    local = int(0 < x <= 1000000 and 0 < y <= 1000000 and x != y)
                    grounded = int(local and x == len(sa) and y == len(sb))
        else:
            for x, y in zip(sa, sb):
                expected.append(x + ('=' if x == y else '<' if x < y else '>') + y)
                if x != y:
                    break
            for s, want in zip(steps, expected):
                m = re.fullmatch(r'([0-9])([=<>])([0-9])', s)
                if m:
                    x, relation, y = m.groups()
                    ok = relation == ('=' if x == y else '<' if x < y else '>')
                    local += ok
                    grounded += ok and s == want
    exact = text.endswith('\n') and '#### ' in text and text.split('#### ', 1)[1].split('\n', 1)[0] == p['answer']
    full = bool(expected) and len(steps) == len(expected) == grounded
    return (1, int(exact), int(full), int(exact and full), len(expected), len(steps), local, grounded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=pathlib.Path)
    parser.add_argument('--data-only', action='store_true')
    args = parser.parse_args()
    root, data = args.directory, args.directory/'data'
    prior = set()
    for file in (root/'exclusion-inputs.txt').read_text().splitlines():
        path = pathlib.Path(file)
        prior |= corpus_questions(path) if path.suffix == '.tok' else {p['question'] for p in problems(path)}
    assert prior == set((data/'excluded-questions.txt').read_text().splitlines())
    splits = {s: problems(data/(s+'.txt')) for s in ('control', 'expanded', 'validation', 'probe')}
    sets = {s: {p['question'] for p in rows} for s, rows in splits.items()}
    for name, rows in splits.items():
        assert len(sets[name]) == len(rows) and not sets[name] & prior, name
        for p in rows:
            if p['task'] == 'add':
                answer = sum(map(int, p['question'].split('+')))
            else:
                answer = max(map(int, re.fullmatch(r'max\(([0-9]+),([0-9]+)\)', p['question']).groups()))
            assert str(answer) == p['answer']
            assert trace(p, p['steps']+'\n#### '+p['answer']+'\n')[2] == 1
    assert len(sets['control']) == len(sets['expanded']) == 2400
    assert len(sets['control'] & sets['expanded']) == 1600
    assert len(sets['probe']) == 800 and len(sets['validation']) == 360
    for name in ('probe', 'validation'):
        assert not sets[name] & (sets['control'] | sets['expanded'])
    assert not sets['probe'] & sets['validation']
    for name, max_diff, per_cell in [('control', 2, 600), ('expanded', 3, 400), ('validation', 3, 60), ('probe', 4, 100)]:
        count = collections.Counter((p['task'], p['difficulty']) for p in splits[name])
        assert count == {(t, d): per_cell for t in ('add', 'cmp') for d in range(1, max_diff+1)}, name
        if name != 'probe':
            assert corpus_questions(data/(name+'.tok')) == sets[name]
    print(f'Independent exclusions: {len(prior)} prior questions, 800 fresh final, 360 validation; 1600 paired examples shared')
    print('Every new gold answer and complete grounded trace agrees with independent arithmetic')
    if args.data_only:
        return
    for file in ('protocol-sha256.txt', 'final-checkpoints-sha256.txt', 'data-sha256.txt', 'initial-sha256.txt'):
        verify_hashes(root/file)
    fields = ['problems', 'exact', 'full_grounded_trace', 'exact_and_grounded', 'expected_statements', 'generated_statements', 'local_correct', 'grounded_correct']
    all_rows = []
    for variant in ('dense', 'moe-ternary'):
        for seed in range(1, 4):
            rng_states = []
            for arm in ('initial', 'control', 'expanded'):
                prefix = root/f'{variant}-seed{seed}-{arm}'
                b = prefix.with_suffix('.ckpt').read_bytes()
                count = struct.unpack_from('<i', b, 8)[0]
                offset = 12+4*count
                actual_steps = struct.unpack_from('<q', b, offset)[0]
                expected_steps = 2000 if arm == 'initial' else int((root/'settings.txt').read_text().split()[0].split('=')[1])
                assert actual_steps == expected_steps
                if arm != 'initial':
                    rng_states.append(b[offset+8:offset+40])
                recorded = list(csv.DictReader(prefix.with_suffix('.transcripts.tsv').open(), delimiter='\t'))
                assert len(recorded) == 800
                counts = collections.defaultdict(lambda: [0]*8)
                for p, row in zip(splits['probe'], recorded):
                    assert row['question'] == p['question'] and row['task'] == p['task'] and int(row['difficulty']) == p['difficulty']
                    values = trace(p, bytes.fromhex(row['completion_hex']).decode('latin-1'))
                    for key in ((p['task'], p['difficulty']), ('overall', 0)):
                        counts[key] = [x+y for x, y in zip(counts[key], values)]
                native = list(csv.DictReader(prefix.with_suffix('.trace.csv').open()))
                assert len(native) == 9
                for row in native:
                    assert [int(row[f]) for f in fields] == counts[(row['task'], int(row['difficulty']))]
                    all_rows.append(dict(variant=variant, seed=seed, arm=arm, **row))
                audit = {row['metric']: row for row in csv.DictReader(prefix.with_suffix('.audit.csv').open())}
                assert int(audit['teacher_coverage']['correct']) == 800
                assert int(audit['exact_answer']['correct']) == counts[('overall', 0)][1]
            assert rng_states[0] == rng_states[1], 'paired sampler draw counts differ'
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(all_rows[0]))
    writer.writeheader(); writer.writerows(all_rows)
    destination = root/'summary.csv'
    encoded = output.getvalue().encode()
    if destination.exists():
        assert destination.read_bytes() == encoded, 'existing summary differs'
    else:
        destination.write_bytes(encoded)
    print('All 18 native answer/trace/cell summaries match 14400 independently parsed generated completions')
    print('All gold coverage, model/update hashes and paired sampler states pass')


if __name__ == '__main__':
    main()
