#!/usr/bin/env python3
"""Build ~10B measured CxxGPT tokens without holding full plaintext on disk.

Sources and LFS SHA-256 checksums are pinned in configs/licensed-10b-sources.json.
Raw shards are verified before processing and deleted after committed output.
The final corpus keeps exact source locations, complete metadata, license texts,
document hashes, per-document token offsets, and measured token counts.
Resume by rerunning the same command. No GPU or cloud charges are needed.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import struct
import subprocess
import time
import urllib.request
from urllib.parse import unquote, urlsplit

from get_data import download
from prepare_licensed import cleaned, sha256

ROOT = Path(__file__).resolve().parents[1]
LICENSES = {"CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/"}
LICENSES["CC0-1.0"] = "https://creativecommons.org/publicdomain/zero/1.0/"
HEADER = struct.Struct("<8sIIQ")


def allowed_license(value):
    if not isinstance(value, str) or re.search(r"non.?commercial|no.?derivatives|share.?alike|by-(?:nc|nd|sa)|gfdl|gnu free", value, re.I):
        return None
    urls = re.findall(r"https?://(?:www\.)?creativecommons\.org/[^\s<>\"']+", value.lower())
    urls = {u.replace("http://", "https://").rstrip("/.,;") for u in urls}
    return next((name for name, url in LICENSES.items() if urls == {url.rstrip("/")}), None)


def eligible(record, source):
    meta = record.get("metadata") or {}
    license = allowed_license(meta.get("license"))
    if not license:
        return None, "license"
    author = meta.get("author") or meta.get("authors")
    if not author or not meta.get("url") or not record.get("id"):
        return None, "provenance"
    if source == "textbooks" and not (str(author).strip().casefold() == "openstax"
            and "/Bookshelves/" in meta["url"] and "(openstax)" in unquote(meta["url"]).casefold()):
        return None, "editorial_selection"
    text = cleaned(record.get("text") or "").replace(chr(1), "")
    if len(text) < 1000 or len(text.encode()) > 8 << 20:
        return None, "length"
    if text.count("\ufffd") > len(text) * .001:
        return None, "decoding"
    sample = text[:16000]
    if len(re.findall(r"[A-Za-z]", sample)) < len(sample) * .25:
        return None, "quality"
    group = meta.get("book_id") or meta.get("book_url") or str(record["id"])
    if source == "arxiv":
        group = re.sub(r"v\d+$", "", group)
    group = str(group).rstrip("/")
    bucket = int(hashlib.sha256((source + ":" + group).encode()).hexdigest()[:16], 16) % 100
    split = "val" if bucket == 0 else "test" if bucket == 1 else "train"
    return (text, license, group, split), None


def acquire(source, shard, out):
    raw = out / "raw" / shard["path"]
    raw.parent.mkdir(parents=True, exist_ok=True)
    if source['name'] == 'textbooks' and not raw.exists():
        existing = ROOT / 'data/licensed-textbooks-v1/raw/libretexts' / shard['path']
        if existing.exists():
            os.link(existing, raw)
    if not raw.exists() and shutil.disk_usage(out).free < shard["bytes"] + (2 << 30):
        raise RuntimeError("Insufficient disk headroom for the next raw shard")
    url = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{shard['path']}"
    download(url, str(raw))
    if raw.stat().st_size != shard["bytes"] or sha256(raw) != shard["sha256"]:
        raise RuntimeError(f"Source checksum verification failed: {raw}")
    print(f"SOURCE VERIFIED {source['name']} {shard['path']}", flush=True)
    return raw


def save_manifest(out, manifest):
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    p = out / "manifest.json.part"
    p.write_text(json.dumps(manifest, indent=2) + "\n")
    p.replace(out / "manifest.json")


def make_tokenizer(out, sources, binary, vocab):
    tok = out / "tok.bpe"
    if tok.exists():
        return tok
    print("Preparing a training-only tokenizer sample", flush=True)
    sample = out / "bpe-sample.txt"
    with open(sample, "w", encoding="utf-8") as dest:
        # The first corpus's OpenStax sample is already separated by book.
        textbooks = ROOT / "data/licensed-textbooks-v1/libretexts/train.txt"
        if textbooks.exists():
            with open(textbooks, encoding="utf-8") as f:
                dest.write(f.read(16 << 20).replace("<|endoftext|>", "") + "\n")
        for source in (s for s in sources if s['name'] in ('doab','arxiv')):
            raw = acquire(source, source["files"][0], out)
            written = 0
            with gzip.open(raw, "rt", encoding="utf-8") as f:
                for line in f:
                    item, _ = eligible(json.loads(line), source["name"])
                    if item and item[3] == "train":
                        text = item[0]
                        dest.write(text + "\n")
                        written += len(text.encode())
                    if written >= 32 << 20:
                        break
            print(f"Tokenizer sample: {source['name']} {written:,} bytes", flush=True)
    subprocess.run([str(binary), "bpe-train", "--in", str(sample), "--out", str(tok), "--vocab", str(vocab),
                    "--max-bytes", str(sample.stat().st_size)], check=True)
    return tok


def math_label(record, source, text):
    meta = record.get('metadata') or {}
    if source == 'textbooks' and urlsplit(meta.get('url','')).hostname == 'math.libretexts.org':
        return 'math-textbook'
    title = meta.get('title') or text[:500]
    if source in ('doab','arxiv') and re.search(r'\b(mathematics|algebra|geometry|calculus|combinatorics|topology|number theory|probability theory|mathematical)\b',title,re.I):
        return 'math-title'
    if source == 'arxiv' and len(re.findall(r'\b(theorem|lemma|corollary|proposition|proof)\b',text,re.I)) >= 6:
        return 'proof-oriented-heuristic'
    return 'general'


def read_exact(stream, count):
    data = bytearray()
    while len(data) < count:
        block = stream.read(count - len(data))
        if not block:
            raise RuntimeError("Native tokenizer stopped before completing a document")
        data.extend(block)
    return data


class Encoder:
    def __init__(self, executable, tokenizer):
        self.process = subprocess.Popen([str(executable), str(tokenizer)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def encode(self, text):
        data = text.encode()
        self.process.stdin.write(struct.pack("<Q", len(data)))
        self.process.stdin.write(data)
        self.process.stdin.flush()
        count = struct.unpack("<Q", read_exact(self.process.stdout, 8))[0]
        return count, read_exact(self.process.stdout, count * 2)

    def close(self):
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError("Native tokenizer failed")


class Shards:
    def __init__(self, directory, vocab, token_limit=100_000_000):
        self.directory, self.vocab, self.limit = directory, vocab, token_limit
        self.active, self.results, self.numbers = {}, [], Counter()

    def finish(self, split):
        if split not in self.active:
            return
        f, path, count = self.active.pop(split)
        f.seek(0)
        f.write(HEADER.pack(b"CXTOKEN1", self.vocab, 0, count))
        f.flush()
        os.fsync(f.fileno())
        f.close()
        self.results.append({"file":path.name, "split":split, "tokens":count,
                             "bytes":path.stat().st_size, "sha256":sha256(path)})

    def write(self, split, count, payload):
        if split in self.active and self.active[split][2] >= self.limit:
            self.finish(split)
        if split not in self.active:
            path = self.directory / f"{split}-{self.numbers[split]:04d}.tok"
            self.numbers[split] += 1
            f = open(path, "w+b")
            f.write(HEADER.pack(b"CXTOKEN1", self.vocab, 0, 0))
            self.active[split] = [f, path, 0]
        entry = self.active[split]
        offset = entry[2]
        entry[0].write(payload)
        entry[2] += count
        return entry[1].name, offset

    def close(self):
        for split in list(self.active):
            self.finish(split)


def process(source, shard, args, manifest, db, encoder):
    job = source["name"] + "-" + shard["path"].removesuffix(".json.gz")
    if job in manifest["completed"]:
        return
    raw = acquire(source, shard, args.out)
    final, work = args.out / "blocks" / job, args.out / "blocks" / (job + ".work")
    for path in (final, work):
        if path.exists():
            shutil.rmtree(path)
    work.mkdir(parents=True)
    db.execute("DELETE FROM seen WHERE job=?", (job,))
    db.commit()
    counts, token_counts, license_counts, math_counts = Counter(), Counter(), Counter(), Counter()
    writer = Shards(work, args.vocab)
    t0, last = time.time(), time.time()
    with gzip.open(work / "attribution.jsonl.gz", "wt", encoding="utf-8") as audit, gzip.open(raw, "rt", encoding="utf-8") as input:
        for line_no, line in enumerate(input, 1):
            record = json.loads(line)
            counts["read"] += 1
            item, reason = eligible(record, source["name"])
            if not item:
                counts["drop_" + reason] += 1
                continue
            text, license, group, split = item
            digest = hashlib.sha256(" ".join(text.casefold().split()).encode()).digest()
            if not db.execute("INSERT OR IGNORE INTO seen VALUES (?,?)", (digest, job)).rowcount:
                counts["drop_duplicate"] += 1
                continue
            count, payload = encoder.encode(text)
            math = math_label(record, source['name'], text)
            filename, offset = writer.write(split, count, payload)
            provenance = {"source":source["repo"], "revision":source["revision"], "shard":shard["path"],
                          "line":line_no, "id":record["id"], "metadata":record["metadata"], "split":split,
                          "group":group, "license":license, "license_url":LICENSES[license],
                          "math_label":math,
                          "text_sha256":hashlib.sha256(text.encode()).hexdigest(),
                          "normalized_sha256":digest.hex(), "token_file":filename,
                          "token_offset":offset, "tokens":count,
                          "changes":"CRLF, navigation, quiz markup and reserved BOS/document-separator controls removed; surrounding whitespace stripped"}
            audit.write(json.dumps(provenance, ensure_ascii=False) + "\n")
            counts["kept_" + split] += 1
            token_counts[split] += count
            license_counts[license] += 1
            if split == 'train':
                math_counts[math] += count
            if time.time() - last > 30:
                total = manifest["tokens"].get("train", 0) + token_counts["train"]
                print(f"PROGRESS {job}: {line_no:,} records; {total:,} training tokens; {(time.time()-t0)/60:.1f} min", flush=True)
                last = time.time()
                if shutil.disk_usage(args.out).free < 1 << 30:
                    raise RuntimeError("Stopped before exhausting the workspace disk")
            if manifest["tokens"].get("train", 0) + token_counts["train"] >= args.target_tokens:
                break
    writer.close()
    db.commit()
    work.rename(final)
    for split, count in token_counts.items():
        manifest["tokens"][split] = manifest["tokens"].get(split, 0) + count
    manifest["completed"][job] = {"source":source["repo"], "revision":source["revision"], "raw":shard,
                                "counts":dict(counts), "tokens":dict(token_counts),
                                "licenses":dict(license_counts), "files":writer.results,
                                "math_training_tokens":dict(math_counts),
                                "attribution_sha256":sha256(final / "attribution.jsonl.gz")}
    manifest["status"] = "complete" if manifest["tokens"].get("train", 0) >= args.target_tokens else "building"
    save_manifest(args.out, manifest)
    raw.unlink()
    print(f"COMMITTED {job}: {json.dumps(dict(token_counts))}; corpus {json.dumps(manifest['tokens'])}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/licensed-10b")
    parser.add_argument("--inventory", type=Path, default=ROOT / "configs/licensed-10b-sources.json")
    parser.add_argument("--target-tokens", type=int, default=10_000_000_000)
    parser.add_argument("--vocab", type=int, default=16384)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    sources = json.loads(args.inventory.read_text())
    binary = ROOT / "cxxgpt"
    if not binary.exists():
        subprocess.run(["make", "cxxgpt"], cwd=ROOT, check=True)
    tok = make_tokenizer(args.out, sources, binary, args.vocab)
    executable = args.out / "tokenize-stream"
    subprocess.run(["g++", "-O3", "-std=c++20", str(ROOT / "tools/tokenize_stream.cpp"), "-o", str(executable)], check=True)
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["tokenizer_sha256"] != sha256(tok) or manifest["target_tokens"] != args.target_tokens:
            raise RuntimeError("Resume configuration differs from the saved manifest")
    else:
        manifest = {"created_utc":datetime.now(timezone.utc).isoformat(), "target_tokens":args.target_tokens,
                    "status":"building", "tokens":{}, "completed":{}, "sources":sources,
                    "tokenizer_sha256":sha256(tok), "vocab":args.vocab,
                    "licenses":LICENSES, "license_evidence":"Publisher-reported metadata; not independent certification",
                    "split_policy":"SHA256(source:book-or-paper-id) modulo 100: val=0, test=1, train=2..99",
                    "deduplication":"Global normalized exact-document SHA-256, including across splits and sources; near duplicates are not removed",
                    "raw_retention":"Pinned remote revision and raw SHA-256 retained; local raw shards deleted after committed tokenization"}
        save_manifest(args.out, manifest)
    for name, url in LICENSES.items():
        path = args.out / "licenses" / f"{name}.txt"
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            request = urllib.request.Request(url + 'legalcode.txt', headers={'User-Agent':'CxxGPT-licensed-data/1'})
            with urllib.request.urlopen(request, timeout=45) as r:
                path.write_bytes(r.read())
    db = sqlite3.connect(args.out / "dedup.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS seen (hash BLOB PRIMARY KEY, job TEXT NOT NULL) WITHOUT ROWID")
    db.execute("PRAGMA cache_size=-32768")
    encoder = Encoder(executable, tok)
    try:
        for source in sources:
            for shard in source["files"]:
                if manifest["tokens"].get("train", 0) >= args.target_tokens:
                    break
                process(source, shard, args, manifest, db, encoder)
        if manifest["status"] != "complete":
            raise RuntimeError("Source inventory exhausted before reaching the measured token target")
    finally:
        encoder.close()
        db.close()
    data_files = []
    for job, record in manifest["completed"].items():
        data_files.extend(str(args.out / "blocks" / job / f["file"]) for f in record["files"] if f["split"] == "train")
    (args.out / "train-files.txt").write_text("\n".join(data_files) + "\n")
    print("COMPLETE", json.dumps(manifest["tokens"]), flush=True)


if __name__ == "__main__":
    main()
