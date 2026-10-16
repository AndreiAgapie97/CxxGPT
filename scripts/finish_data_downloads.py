#!/usr/bin/env python3
"""Verify the recreated worked math, then publish complete and split downloads."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from package_prepared_data import ROOT, DATA, OUT, STATE, package, verify, digest, atomic_json


def main():
    corpus = json.loads((STATE / 'corpus-ready.json').read_text())
    math = ROOT / 'data/targeted-math'
    original = json.loads((math / 'original-training-manifest.json').read_text())
    regenerated = json.loads((math / 'manifest.json').read_text())
    for split in ('train', 'dev'):
        assert digest(math / f'word-{split}.tok') == original['splits'][split]['sha256']
        for k in ('tokens', 'questions', 'dropped_near_heldout', 'seed'):
            assert regenerated['splits'][split][k] == original['splits'][split][k]
    source = DATA / 'gsm8k-train.tok'
    destination = math / 'gsm8k-train.tok'
    if not destination.exists():
        os.link(source, destination)
    recipe = json.loads(Path('/workspace/cxxgpt-ops/recovery/logs/math-focus-recipe.json').read_text())
    recipe['data'] = recipe['data'].replace('data/corpus/', 'data/licensed-10b/blocks/')
    recipe['restore_note'] = 'Paths adapted to the supplied corpus folder; same sampling weights and files as the final training mix.'
    atomic_json(math / 'balanced-recipe.json', recipe)
    guide = ROOT / 'docs/PREPARED_DATA.md'
    guide.write_text('''# Prepared CxxGPT data

The complete bundle and split parts contain the same data. Choose either the
complete archive or all numbered parts; do not download both sets unnecessarily.
Extract them into the same parent folder as the supplied CxxGPT repository.
Use 7-Zip (Zstandard support) or GNU tar with zstd installed:

    tar --zstd -xf CxxGPT-prepared-data.tar.zst

For numbered parts, extract every CxxGPT-data-*.tar.zst into the same folder.
Their members have distinct paths. Check the provided SHA256SUMS first.
Uncompressed prepared data needs roughly 23GB; allow extra space for models.

data/licensed-10b/blocks contains native .tok shards for 10,000,016,185 training,
103,058,222 validation and 101,416,161 test tokens. Keep these splits separate.
Use the paired data/licensed-10b/tok.bpe tokenizer (16,384 vocabulary).
The prepared general corpus is tokenized; original downloaded JSON sources are
not duplicated here. Pinned source revisions, original checksums, licenses,
document attribution and token offsets are supplied. Internal dedup SQLite
build caches and superseded pilot datasets are excluded.

data/targeted-math contains the worked-math continuation: 963,089 generated
training problems and 1,000 generated development problems, in readable JSONL,
evaluation TXT and masked token formats. These were reconstructed on the free
cloud CPU from the original generator, seeds and held-out exclusion records;
BOTH token files match the original Pod training SHA-256 byte-for-byte.
original-training-manifest.json preserves the original provenance; manifest.json
records reconstruction. Human GSM8K train/dev/test files and MIT licensing are
also retained. Never mix held-out development/test files into training.

balanced-recipe.json preserves the approximately 53% worked-math /47% general
input sampling mixture, with corpus paths adapted to this download layout.
Sampling weights repeat data; they do not increase unique token counts. The
generated worked-math corpus is additional to the 10B licensed general corpus.

The lossless packaging process verified every archived file against its original
SHA-256 before replacing uncompressed copies in the space-limited workspace.
The complete bundle is independently reverified against every part's file list.
No Runpod resource or GPU spending was needed to prepare these downloads.
''')
    sources = [(p, 'CxxGPT/' + str(p.relative_to(ROOT))) for p in sorted(math.rglob('*')) if p.is_file()]
    for p in [guide, Path(__file__).resolve()]:
        sources.append((p, 'CxxGPT/' + str(p.relative_to(ROOT))))
    for name in ('prepare_targeted_math.py', 'rebuild_targeted_math.py'):
        p = Path('/workspace/cxxgpt-ops') / name
        sources.append((p, 'CxxGPT/operations/' + name))
    entries = [{'source': str(p), 'archive_path': arc, 'bytes': p.stat().st_size,
                'sha256': digest(p)} for p, arc in sources]
    math_part = package(f"CxxGPT-data-{len(corpus['parts']):02d}-worked-math.tar.zst", entries)
    parts = corpus['parts'] + [math_part]
    all_entries = [item for part in parts for item in part['files']]
    assert len({e['archive_path'] for e in all_entries}) == len(all_entries)
    target = OUT / 'CxxGPT-prepared-data.tar.zst'
    proof_path = STATE / 'complete-ready.json'
    if proof_path.exists():
        complete = json.loads(proof_path.read_text())
        assert digest(target) == complete['sha256']
    else:
        # Keep all verified parts until the complete archive passes every SHA.
        assert shutil.disk_usage(OUT).free > sum(p['bytes'] for p in parts) * 1.12
        temporary = target.with_name(target.name + '.part')
        with temporary.open('wb') as output:
            compressor = subprocess.Popen(['zstd', '-q', '-6', '-T2', '-c'], stdin=subprocess.PIPE, stdout=output)
            try:
                with tarfile.open(fileobj=compressor.stdin, mode='w|', bufsize=1024 * 1024) as destination_tar:
                    for part in parts:
                        print('COMBINING VERIFIED PART', part['archive'], flush=True)
                        reader = subprocess.Popen(['zstd', '-q', '-dc', str(OUT / part['archive'])], stdout=subprocess.PIPE)
                        try:
                            with tarfile.open(fileobj=reader.stdout, mode='r|', bufsize=1024 * 1024) as source_tar:
                                for member in source_tar:
                                    assert member.isfile()
                                    destination_tar.addfile(member, source_tar.extractfile(member))
                            while reader.stdout.read(1024 * 1024):
                                pass
                            assert reader.wait() == 0
                        finally:
                            if reader.poll() is None:
                                reader.terminate()
                                reader.wait()
                compressor.stdin.close()
                assert compressor.wait() == 0
            finally:
                if compressor.poll() is None:
                    compressor.terminate()
                    compressor.wait()
        verify(temporary, all_entries)
        temporary.replace(target)
        complete = {'archive': target.name, 'bytes': target.stat().st_size,
                    'sha256': digest(target), 'verified_every_file_sha256': True}
        atomic_json(proof_path, complete)
    manifest = {'licensed_corpus_tokens': corpus['tokens'],
                'worked_math_original_training_manifest': original,
                'worked_math_token_files_match_original_training': True,
                'complete': complete, 'parts': parts}
    atomic_json(OUT / 'CxxGPT-prepared-data-manifest.json', manifest)
    (OUT / 'CxxGPT-prepared-data-SHA256SUMS.txt').write_text(''.join(
        p['sha256'] + '  ' + p['archive'] + '\n' for p in [complete] + parts))
    rows = '\n'.join(f"| [{p['archive']}]({p['archive']}) | {p['bytes']/1e9:.2f}GB |" for p in parts)
    page = f'''# Prepared CxxGPT data downloads

**All prepared data is ready and verified.** The complete bundle includes the
10B-token general corpus, held-out splits, worked-math continuation data,
paired tokenizer, licenses, document attribution and source provenance.

[Download everything — {complete['bytes']/1e9:.2f}GB]({complete['archive']})

Or download every numbered part below instead. Each is independently extractable
into the same folder. The complete bundle already includes all these parts' data.

| Part | Download size |
| --- | --- |
{rows}

[SHA-256 checksums](CxxGPT-prepared-data-SHA256SUMS.txt) ·
[Detailed file manifest](CxxGPT-prepared-data-manifest.json) ·
[Restore and data-use instructions](CxxGPT-data-README.md)

General corpus: 10,000,016,185 train / 103,058,222 validation / 101,416,161 test tokens.
Worked math: 963,089 generated train problems, 1,000 generated development problems,
plus 6,966 human GSM8K training questions. Generated token files were reconstructed
from saved seeds and match the original training SHA-256 exactly. All 142 general
token shards were verified against their original manifest checksums too.

The general corpus is in native tokenized format; worked math also includes
readable JSONL/TXT. Allow roughly 23GB for extraction. These data downloads are
separate from the 3.18GB model/source recovery archive. No new GPU spending.
'''
    (OUT / 'CxxGPT-data-downloads.md').write_text(page)
    (OUT / 'CxxGPT-data-README.md').write_text(guide.read_text())
    print('COMPLETE DATA DOWNLOAD VERIFIED', json.dumps(complete), flush=True)


if __name__ == '__main__':
    main()
