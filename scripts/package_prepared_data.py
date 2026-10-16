#!/usr/bin/env python3
"""Losslessly package the completed corpus within the existing disk budget.

Raw token shards are removed only after their archived bytes match the original
build manifest's SHA-256. The plan and per-part proofs support safe continuation.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path('/workspace/CxxGPT')
DATA = ROOT / 'data/licensed-10b'
OUT = Path('/workspace/library-files')
STATE = Path('/workspace/cxxgpt-ops/data-packaging')
STATE.mkdir(exist_ok=True)


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def verify(archive, entries):
    expected = {e['archive_path']: e for e in entries}
    found = set()
    p = subprocess.Popen(['zstd', '-q', '-dc', str(archive)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=p.stdout, mode='r|') as tar:
            for member in tar:
                if not member.isfile():
                    continue
                assert member.name in expected, member.name
                item = expected[member.name]
                assert member.size == item['bytes'], member.name
                actual = hashlib.file_digest(tar.extractfile(member), 'sha256').hexdigest()
                assert actual == item['sha256'], member.name
                found.add(member.name)
        # Drain tar padding and require a successful compression integrity check.
        while p.stdout.read(1024 * 1024):
            pass
        assert p.wait() == 0
    finally:
        if p.poll() is None:
            p.terminate()
            p.wait()
    assert found == set(expected)


def package(name, entries, remove_raw=False):
    destination = OUT / name
    proof_path = STATE / (name + '.json')
    if proof_path.exists():
        proof = json.loads(proof_path.read_text())
        assert destination.exists() and digest(destination) == proof['sha256']
        assert proof['files'] == entries
    else:
        temporary = destination.with_name(destination.name + '.part')
        with temporary.open('wb') as output:
            p = subprocess.Popen(['zstd', '-q', '-6', '-T2', '-c'], stdin=subprocess.PIPE, stdout=output)
            try:
                with tarfile.open(fileobj=p.stdin, mode='w|') as tar:
                    for item in entries:
                        source = Path(item['source'])
                        assert source.stat().st_size == item['bytes']
                        tar.add(source, arcname=item['archive_path'], recursive=False)
                p.stdin.close()
                assert p.wait() == 0
            finally:
                if p.poll() is None:
                    p.terminate()
                    p.wait()
        verify(temporary, entries)
        temporary.replace(destination)
        proof = {'archive': name, 'bytes': destination.stat().st_size,
                 'sha256': digest(destination), 'verified_every_file_sha256': True,
                 'files': entries}
        atomic_json(proof_path, proof)
    if remove_raw:
        for item in entries:
            source = Path(item['source'])
            if source.exists():
                assert source.stat().st_size == item['bytes']
                source.unlink()
    destination.with_name(destination.name + '.sha256').write_text(proof['sha256'] + '  ' + name + '\n')
    print('VERIFIED DATA PART', name, proof['bytes'], 'files', len(entries), flush=True)
    return proof


def main():
    plan_path = STATE / 'plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
    else:
        manifest = json.loads((DATA / 'manifest.json').read_text())
        assert manifest['status'] == 'complete'
        tokens = []
        for block, record in sorted(manifest['completed'].items()):
            attribution = DATA / 'blocks' / block / 'attribution.jsonl.gz'
            assert digest(attribution) == record['attribution_sha256']
            for f in record['files']:
                source = DATA / 'blocks' / block / f['file']
                tokens.append({'source': str(source), 'archive_path': 'CxxGPT/' + str(source.relative_to(ROOT)),
                               'bytes': f['bytes'], 'sha256': f['sha256'], 'split': f['split'], 'tokens': f['tokens']})
        assert {Path(x['source']) for x in tokens} == set((DATA / 'blocks').glob('*/*.tok'))
        assert sum(x['tokens'] for x in tokens if x['split'] == 'train') == manifest['tokens']['train']
        groups = []
        current = []
        size = 0
        for item in tokens:
            if current and size + item['bytes'] > 4_000_000_000:
                groups.append(current)
                current = []
                size = 0
            current.append(item)
            size += item['bytes']
        if current:
            groups.append(current)
        plan = {'tokens': manifest['tokens'], 'parts': groups}
        atomic_json(plan_path, plan)
    results = []
    for i, group in enumerate(plan['parts'], 1):
        results.append(package(f'CxxGPT-data-{i:02d}.tar.zst', group, remove_raw=True))
    # Keep provenance, licenses, the paired tokenizer, and human GSM8K data.
    metadata = []
    for p in sorted(DATA.rglob('*')):
        if not p.is_file() or p.name.startswith('dedup.sqlite') or 'raw' in p.relative_to(DATA).parts:
            continue
        if p.suffix == '.tok' and 'blocks' in p.relative_to(DATA).parts:
            continue
        metadata.append({'source': str(p), 'archive_path': 'CxxGPT/' + str(p.relative_to(ROOT)),
                         'bytes': p.stat().st_size, 'sha256': digest(p)})
    source = Path(__file__).resolve()
    metadata.append({'source': str(source), 'archive_path': 'CxxGPT/scripts/' + source.name,
                     'bytes': source.stat().st_size, 'sha256': digest(source)})
    results.insert(0, package('CxxGPT-data-00-metadata.tar.zst', metadata))
    atomic_json(STATE / 'corpus-ready.json', {'tokens': plan['tokens'], 'parts': results})
    print('ALL LICENSED CORPUS PARTS READY', len(results), sum(x['bytes'] for x in results), flush=True)


if __name__ == '__main__':
    main()
