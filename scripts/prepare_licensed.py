#!/usr/bin/env python3
"""Fetch pinned Common Pile textbooks and retain a record-level license audit.

Publisher-reported licenses are evidence, not independent rights certification.
Only explicit CC BY 4.0 and CC0 1.0 URLs are accepted. Raw shards, dataset cards,
attribution, document hashes, and book-level split decisions remain on disk.
Uses the standard library. Run: python3 scripts/prepare_licensed.py
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote
import urllib.request

SOURCES = (
    {"name": "libretexts", "repo": "common-pile/libretexts_filtered",
     "revision": "70388bca52b4a93515e14b1d56618fd7944988fd",
     "file": "libretexts-0000.json.gz", "size": 115037937,
     "sha256": "71851a2b5acfccf80659729a0eccd07ab6ed737f0edc659a662aa9c1cd2194db"},
    {"name": "pressbooks", "repo": "common-pile/pressbooks_filtered",
     "revision": "1a1d3b50d77f834370f8eb4c0d174668dd1676bb",
     "file": "pressbooks-0000.json.gz", "size": 191725499,
     "sha256": "dd251acfa9b535a0aadeb7f61743a61934a1783bd1bdac947d4d136781b22d22"},
)
SEP = "\n<|endoftext|>\n"
LICENSES = {
    "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
    "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, path, digest=None, size=None):
    if path.exists() and (not digest or sha256(path) == digest):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "CxxGPT-licensed-data/1"})
    with urllib.request.urlopen(request, timeout=60) as response, open(part, "wb") as out:
        for chunk in iter(lambda: response.read(4 << 20), b""):
            out.write(chunk)
    if size is not None and part.stat().st_size != size:
        raise ValueError(f"Unexpected file size: {path}")
    if digest is not None and sha256(part) != digest:
        raise ValueError(f"SHA-256 mismatch: {path}")
    part.replace(path)
    print(f"Downloaded {path}: {path.stat().st_size:,} bytes", flush=True)


def acquire(source, root):
    base = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}"
    dest = root / "raw" / source["name"]
    fetch(f"{base}/README.md", dest / "DATASET_CARD.md")
    fetch(f"{base}/{source['file']}", dest / source["file"], source["sha256"], source["size"])


def license_id(value):
    # Fail closed on missing, multiple, or unrecognized license statements.
    if not isinstance(value, str):
        return None
    if re.search(r"non.?commercial|no.?derivatives|share.?alike|by-(?:nc|nd|sa)|gfdl|gnu free", value, re.I):
        return None
    urls = re.findall(r"https?://(?:www\.)?creativecommons\.org/[^\s<>\"']+", value.lower())
    normalized = {u.replace("http://", "https://").rstrip("/.,;") for u in urls}
    for name, url in LICENSES.items():
        if normalized == {url.rstrip("/")}:
            return name
    return None


def cleaned(text):
    text = text.replace("\r\n", "\n").replace("<|endoftext|>", "")
    return "\n".join(line for line in text.split("\n")
                     if not re.fullmatch(r"- Save as PDF|- Last updated|\[(?:reveal|hidden)-answer[^\]]*\].*|\[/hidden-answer\]", line.strip())).strip()


def prepare(source, root, seen, group_splits, min_chars, quality="openstax"):
    name = source["name"]
    dest = root / name
    dest.mkdir(exist_ok=True)
    counts, licenses = Counter(), Counter()
    handles = {split: open(dest / f"{split}.txt.part", "wb") for split in ("train", "val", "test")}
    sizes = Counter()
    attribution = open(dest / "attribution.jsonl.part", "w", encoding="utf-8")
    try:
        with gzip.open(root / "raw" / name / source["file"], "rt", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                record = json.loads(line)
                counts["read"] += 1
                meta = record.get("metadata") or {}
                license = license_id(meta.get("license"))
                if not license:
                    counts["drop_license"] += 1
                    continue
                url, author, title, book = (meta.get(k) for k in ("url", "author", "title", "book_url"))
                if not all(isinstance(v, str) and v.strip() for v in (url, author, title, book)):
                    counts["drop_provenance"] += 1
                    continue
                is_openstax = (source["name"] == "libretexts" and author.strip().casefold() == "openstax"
                               and "/Bookshelves/" in url and "(openstax)" in unquote(url).casefold())
                if quality == "openstax" and not is_openstax:
                    counts["drop_editorial_selection"] += 1
                    continue
                text = cleaned(record.get("text") or "")
                if len(text) < min_chars:
                    counts["drop_short"] += 1
                    continue
                if text.count("\ufffd") > len(text) * .001 or sum(c.isalpha() for c in text) < len(text) * .35:
                    counts["drop_quality"] += 1
                    continue
                key = hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()
                if key in seen:
                    counts["drop_duplicate"] += 1
                    continue
                seen.add(key)
                # Chapters from the same book stay together. Identical text is
                # deduplicated globally before splitting across both sources.
                group = book.rstrip("/")
                bucket = int(hashlib.sha256(group.encode()).hexdigest()[:16], 16) % 100
                split = group_splits.setdefault(group, "test" if bucket < 5 else "val" if bucket < 10 else "train")
                out = handles[split]
                if sizes[split]:
                    out.write(SEP.encode())
                    sizes[split] += len(SEP)
                data = text.encode("utf-8")
                offset = sizes[split]
                out.write(data)
                sizes[split] += len(data)
                provenance = {
                    "source": source["repo"], "revision": source["revision"],
                    "shard": source["file"], "line": line_no,
                    "id": record.get("id"), "url": url, "book_url": book,
                    "author": author, "title": title, "license": license,
                    "editorial_selection": "OpenStax attributed; canonical LibreTexts Bookshelves entry" if is_openstax else "other openly licensed educational source",
                    "license_url": LICENSES[license], "license_evidence": meta["license"],
                    "publisher_provenance": meta.get("provenance"), "split": split,
                    "byte_offset": offset, "bytes": len(data),
                    "text_sha256": hashlib.sha256(data).hexdigest(),
                    "changes": "CRLF normalized; document separator and navigation/quiz markup removed",
                }
                attribution.write(json.dumps(provenance, ensure_ascii=False) + "\n")
                counts[f"kept_{split}"] += 1
                licenses[license] += 1
    finally:
        attribution.close()
        for out in handles.values():
            out.close()
    for split in handles:
        (dest / f"{split}.txt.part").replace(dest / f"{split}.txt")
    (dest / "attribution.jsonl.part").replace(dest / "attribution.jsonl")
    result = {"source": source,
              "dataset_card_sha256": sha256(root / "raw" / name / "DATASET_CARD.md"),
              "counts": dict(counts), "licenses": dict(licenses), "files": {}}
    for p in sorted(dest.iterdir()):
        if p.is_file() and not p.name.endswith(".part"):
            result["files"][str(p.relative_to(root))] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    print(name, json.dumps({"counts": counts, "licenses": licenses, "text_bytes": sizes}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/licensed-textbooks-v1"))
    parser.add_argument("--min-chars", type=int, default=1000)
    parser.add_argument("--quality", choices=("openstax", "licensed"), default="openstax",
                        help="openstax: select attributed OpenStax Bookshelves entries; licensed: retain all eligible educational documents")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda s: acquire(s, args.out), SOURCES))
    for name, url in LICENSES.items():
        fetch(url + "legalcode.txt", args.out / "licenses" / f"{name}.txt")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": sha256(Path(__file__)), "minimum_characters": args.min_chars,
        "allowed_licenses": LICENSES,
        "editorial_selection": args.quality,
        "license_text_sha256": {name: sha256(args.out / "licenses" / f"{name}.txt") for name in LICENSES},
        "license_evidence_status": "Publisher-reported per-document metadata; not third-party certification or a rights warranty",
        "split_policy": "SHA-256(book_url.rstrip('/')) first 16 hex digits modulo 100: test=0..4, val=5..9, train=10..99; global normalized exact-text deduplication",
        "sources": [],
    }
    seen, groups = set(), {}
    for source in SOURCES:
        manifest["sources"].append(prepare(source, args.out, seen, groups, args.min_chars, args.quality))
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Manifest: {args.out / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
