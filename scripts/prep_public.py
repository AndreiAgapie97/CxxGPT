#!/usr/bin/env python3
"""prep_public.py: turn the Common Pile downloads into text files for cxxgpt.

    python scripts/prep_public.py --out data/clean [--src data/public-20261015/common-pile]
        [--only wiki,textbooks] [--shards doab=2,arxiv=1] [--workers 8]

Reads the gzip JSON Lines shards that get downloaded into
data/public-*/common-pile/ (they are never changed) and writes one text file
per source into --out, documents separated by "<|endoftext|>" lines (for
`cxxgpt text2tok --doc-sep "<|endoftext|>"`):

  textbooks.txt  LibreTexts and Pressbooks (open textbooks)
  wiki.txt       Wikimedia articles (namespace 0): Wikipedia, Wikibooks,
                 Wikisource and Wikinews; Wikiquote is left out (lists of
                 quotations)
  doab.txt       academic books from the Directory of Open Access Books
  arxiv.txt      openly licensed arXiv papers

A record is dropped when its text is empty or shorter than --min-chars, when
it is not an article (Wikimedia), when more than 0.1% of its characters are
U+FFFD (text that was decoded wrongly), or when the same text (ignoring
case and whitespace) was already kept by this run, in any source (in --only's
order). "\\r\\n" becomes "\\n",
and a "<|endoftext|>" inside a text is removed, so documents stay apart. In
the textbooks, the lines of page buttons ("- Save as PDF", "- Last updated")
and quiz markup ([reveal-answer ...], [hidden-answer ...], [/hidden-answer])
are removed.

--shards limits how many shards (in name order) a source reads: one night
reads about 2.5 GB of text, far less than DOAB and arXiv hold. The shards
are read in parallel (--workers), one process each; the results are then
merged in shard order, so the output does not depend on --workers.

--out/manifest.json records, per source: the shards read (with their size),
the records read, kept and dropped (by reason), the bytes written and the
output's SHA-256. Needs only the standard library.
"""
import argparse
import gzip
import hashlib
import json
import os
import re
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

DOC_SEP = "<|endoftext|>"

# Output name -> the Common Pile folders it reads.
SOURCES = {
    "textbooks": ["libretexts_filtered", "pressbooks_filtered"],
    "wiki": ["wikimedia_filtered"],
    "doab": ["doab_filtered"],
    "arxiv": ["arxiv_papers_filtered"],
}
SKIP_WIKIS = {"wikiquote.com"}


def reject_reason(record, min_chars):
    """Why a record is dropped, or None to keep it."""
    text = record.get("text") or ""
    if len(text.strip()) < min_chars:
        return "empty" if not text.strip() else "short"
    meta = record.get("metadata") or {}
    if str(record.get("source", "")).startswith("wiki/"):
        if str(meta.get("namespace")) != "0":
            return "not_article"
        if meta.get("wiki") in SKIP_WIKIS:
            return "wikiquote"
    if text.count("�") > 0.001 * len(text):
        return "bad_decoding"
    return None


# Lines of the textbook sites' page furniture, not text: LibreTexts' page
# buttons and Pressbooks' quiz markup. Dropped from those two sources only.
TEXTBOOK_SOURCES = {"libretexts", "pressbooks"}
FURNITURE = re.compile(r"- Save as PDF|- Last updated|\[(reveal|hidden)-answer[^\]]*\].*|\[/hidden-answer\]")


def clean(text, source=""):
    text = text.replace("\r\n", "\n").replace(DOC_SEP, "")
    if source in TEXTBOOK_SOURCES:
        text = "\n".join(line for line in text.split("\n") if not FURNITURE.fullmatch(line.strip()))
    return text.strip()


def text_key(text):
    """The same text, ignoring case and whitespace, gives the same key."""
    return hashlib.sha1(" ".join(text.lower().split()).encode("utf-8")).digest()


def read_shard(job):
    """One shard into a temporary file of (key, length, text) records."""
    path, min_chars, tmp_dir = job
    counts = {"read": 0}
    fd, part = tempfile.mkstemp(dir=tmp_dir, suffix=".part")
    with os.fdopen(fd, "wb") as out, gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            counts["read"] += 1
            record = json.loads(line)
            reason = reject_reason(record, min_chars)
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
                continue
            data = clean(record["text"], str(record.get("source", ""))).encode("utf-8")
            out.write(text_key(record["text"]))
            out.write(len(data).to_bytes(8, "little"))
            out.write(data)
    return part, counts


def records(part):
    with open(part, "rb") as f:
        while key := f.read(20):
            n = int.from_bytes(f.read(8), "little")
            yield key, f.read(n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/public-20261015/common-pile")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=",".join(SOURCES), help="sources to write, comma-separated")
    ap.add_argument("--shards", default="", help="NAME=N: read only the first N shards of a source")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    src, out_dir = Path(args.src), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    only = [s for s in args.only.split(",") if s]
    for s in only:
        if s not in SOURCES:
            sys.exit(f"unknown source {s}; known: {', '.join(SOURCES)}")
    limits = {}
    for item in filter(None, args.shards.split(",")):
        name, n = item.split("=")
        limits[name] = int(n)

    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    seen = set()  # text keys kept so far, across sources (in --only's order)
    for name in only:
        shards = []
        for folder in SOURCES[name]:
            shards += sorted((src / folder).glob("*.json.gz"))
        if name in limits:
            shards = shards[: limits[name]]
        if not shards:
            sys.exit(f"{name}: no shards under {src}")
        print(f"== {name}: {len(shards)} shards", flush=True)
        with tempfile.TemporaryDirectory(dir=out_dir) as tmp, Pool(min(args.workers, len(shards))) as pool:
            results = pool.map(read_shard, [(str(p), args.min_chars, tmp) for p in shards])
            counts = {"read": 0, "kept": 0, "duplicate": 0}
            digest = hashlib.sha256()
            written = 0
            dest = out_dir / f"{name}.txt"
            with open(str(dest) + ".part", "wb") as out:
                for part, c in results:
                    for k, v in c.items():
                        counts[k] = counts.get(k, 0) + v
                    for key, data in records(part):
                        if key in seen:
                            counts["duplicate"] += 1
                            continue
                        seen.add(key)
                        chunk = (f"\n{DOC_SEP}\n".encode() if counts["kept"] else b"") + data
                        out.write(chunk)
                        digest.update(chunk)
                        written += len(chunk)
                        counts["kept"] += 1
                    os.remove(part)
            os.replace(str(dest) + ".part", dest)
        manifest[name] = {
            "shards": [{"file": str(p), "bytes": p.stat().st_size} for p in shards],
            "min_chars": args.min_chars,
            "counts": counts,
            "bytes": written,
            "sha256": digest.hexdigest(),
        }
        print(f"   {counts['kept']} of {counts['read']} records kept, {written / 1e9:.2f} GB; dropped: "
              + ", ".join(f"{k} {v}" for k, v in counts.items() if k not in ("read", "kept")), flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
