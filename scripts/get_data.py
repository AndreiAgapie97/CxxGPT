#!/usr/bin/env python3
"""get_data.py: download public datasets and convert them for cxxgpt.

    python3 scripts/get_data.py gsm8k                  # grade-school math, train + test
    python3 scripts/get_data.py tinystories            # simple English stories
    python3 scripts/get_data.py fineweb-edu --files 2  # educational web text
    python3 scripts/get_data.py openmath2 --split train_1M
    python3 scripts/get_data.py metamath

Everything lands in data/<name>/ in one of two formats:
  - text (*.txt): documents separated by "<|endoftext|>" lines, for
    `cxxgpt text2tok --doc-sep "<|endoftext|>"`;
  - questions and answers (*.jsonl): one {"question", "answer"} object per
    line, the answer being the worked solution, then "\\n#### " and the
    final answer (GSM8K's format), for `cxxgpt prep-qa`.

Downloads resume where they stopped. Needs only the standard library, plus
pyarrow for Parquet files (pip install pyarrow). For gated datasets, set
HF_TOKEN to a Hugging Face access token.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")  # or a mirror
DOC_SEP = "<|endoftext|>"


def log(msg):
    print(msg, flush=True)


def headers(extra=None):
    h = {"User-Agent": "cxxgpt-get-data"}
    if os.environ.get("HF_TOKEN"):
        h["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
    h.update(extra or {})
    return h


def download(url, dest, retries=6):
    """Downloads url to dest (skipped if dest exists), resuming a .part file."""
    if os.path.exists(dest):
        log(f"  have {dest}")
        return dest
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    part = dest + ".part"
    for attempt in range(retries):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        req = urllib.request.Request(url, headers=headers({"Range": f"bytes={have}-"} if have else None))
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:  # the server ignored the range: start over
                    have = 0
                total = have + int(r.headers.get("Content-Length") or 0)
                t0, done, last = time.time(), have, 0.0
                with open(part, "ab" if have else "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if time.time() - last > 5:
                            last = time.time()
                            rate = (done - have) / 1e6 / max(last - t0, 1e-9)
                            pct = f" of {total / 1e6:.0f} MB" if total else ""
                            log(f"  {os.path.basename(dest)}: {done / 1e6:.0f} MB{pct} ({rate:.1f} MB/s)")
                if total and done < total:  # the connection dropped early
                    raise ConnectionError(f"got {done} of {total} bytes")
            os.replace(part, dest)
            log(f"  saved {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")
            return dest
        except urllib.error.HTTPError as e:
            if e.code == 416:  # range past the end: the .part file is complete
                os.replace(part, dest)
                return dest
            if e.code in (401, 403):
                sys.exit(f"{url}: HTTP {e.code}. A gated dataset? Accept its terms on huggingface.co "
                         "and set HF_TOKEN.")
            if e.code == 404:
                sys.exit(f"{url}: not found")
            err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            err = e
        wait = 2 ** attempt
        log(f"  {err}; retrying in {wait} s")
        time.sleep(wait)
    sys.exit(f"giving up on {url}")


def hf_files(repo, path=""):
    """Paths of the files in a Hugging Face dataset repo under path (recursive)."""
    url = f"{HF}/api/datasets/{repo}/tree/main/{path}?recursive=true"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers()), timeout=60) as r:
        return sorted(e["path"] for e in json.load(r) if e.get("type") == "file")


def hf_download(repo, path, dest):
    return download(f"{HF}/datasets/{repo}/resolve/main/{path}", dest)


def parquet_rows(path, columns, batch=8192):
    """Yields each row of a Parquet file as a dict of the given columns."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("reading Parquet needs pyarrow: pip install pyarrow")
    f = pq.ParquetFile(path)
    for b in f.iter_batches(batch_size=batch, columns=columns):
        cols = [b.column(c).to_pylist() for c in columns]
        for values in zip(*cols):
            yield dict(zip(columns, values))


def write_qa(rows, dest):
    """Writes (question, solution, final) triples as GSM8K-style JSON Lines."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    n = skipped = 0
    with open(dest + ".part", "w", encoding="utf-8") as f:
        for q, sol, final in rows:
            q, sol, final = (q or "").strip(), (sol or "").strip(), str(final if final is not None else "").strip()
            if not q or not sol or not final or "\n" in final:
                skipped += 1
                continue
            f.write(json.dumps({"question": q, "answer": f"{sol}\n#### {final}"}, ensure_ascii=False) + "\n")
            n += 1
    os.replace(dest + ".part", dest)
    log(f"  {dest}: {n} problems ({skipped} skipped)")


def write_docs(docs, dest):
    """Writes text documents separated by DOC_SEP lines."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    n = 0
    with open(dest + ".part", "w", encoding="utf-8") as f:
        for d in docs:
            d = (d or "").strip()
            if not d:
                continue
            if n:
                f.write(f"\n{DOC_SEP}\n")
            f.write(d)
            n += 1
        f.write("\n")
    os.replace(dest + ".part", dest)
    log(f"  {dest}: {n} documents ({os.path.getsize(dest) / 1e6:.0f} MB)")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def get_gsm8k(out, args):
    base = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data"
    for split in ("train", "test"):
        download(f"{base}/{split}.jsonl", os.path.join(out, f"{split}.jsonl"))


def get_tinystories(out, args):
    # The V2 files (stories by GPT-4 only) already separate stories with DOC_SEP.
    for split in ("train", "valid"):
        hf_download("roneneldan/TinyStories", f"TinyStoriesV2-GPT4-{split}.txt", os.path.join(out, f"{split}.txt"))


def get_fineweb_edu(out, args):
    repo = "HuggingFaceFW/fineweb-edu"
    files = [p for p in hf_files(repo, f"sample/{args.sample}") if p.endswith(".parquet")]
    if not files:
        sys.exit(f"no Parquet files under sample/{args.sample} in {repo}")
    log(f"  {len(files)} files in sample/{args.sample}; taking {min(args.files, len(files))}")
    for i, p in enumerate(files[: args.files]):
        txt = os.path.join(out, f"{i:03d}.txt")
        if os.path.exists(txt):
            log(f"  have {txt}")
            continue
        pq_path = hf_download(repo, p, os.path.join(out, "parquet", os.path.basename(p)))
        write_docs((r["text"] for r in parquet_rows(pq_path, ["text"])), txt)
        if not args.keep_parquet:
            os.remove(pq_path)


def get_openmath2(out, args):
    # NVIDIA's OpenMathInstruct-2: problems with worked solutions written by a
    # large model; problem_source says where each problem came from.
    repo = "nvidia/OpenMathInstruct-2"
    files = [p for p in hf_files(repo, "data") if os.path.basename(p).startswith(args.split + "-")]
    if not files:
        sys.exit(f"no files for split {args.split}; see {HF}/datasets/{repo}/tree/main/data")
    sources = set(args.sources.split(",")) if args.sources else None
    paths = [hf_download(repo, p, os.path.join(out, "parquet", os.path.basename(p))) for p in files]
    cols = ["problem", "generated_solution", "expected_answer", "problem_source"]

    def rows():
        for path in paths:
            for r in parquet_rows(path, cols):
                if sources is None or r["problem_source"] in sources:
                    yield r["problem"], r["generated_solution"], r["expected_answer"]

    write_qa(rows(), os.path.join(out, f"{args.split}.jsonl"))


def get_metamath(out, args):
    # MetaMathQA: GSM8K and MATH training problems, rephrased and augmented;
    # each response ends with "The answer is: X".
    path = hf_download("meta-math/MetaMathQA", "MetaMathQA-395K.json", os.path.join(out, "MetaMathQA-395K.json"))
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    marker = "The answer is: "

    def rows():
        for r in data:
            resp = r.get("response", "")
            k = resp.rfind(marker)
            if k < 0:
                continue
            yield r.get("query", ""), resp[:k], resp[k + len(marker):]

    write_qa(rows(), os.path.join(out, "train.jsonl"))


DATASETS = {
    "gsm8k": get_gsm8k,
    "tinystories": get_tinystories,
    "fineweb-edu": get_fineweb_edu,
    "openmath2": get_openmath2,
    "metamath": get_metamath,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", choices=sorted(DATASETS))
    ap.add_argument("--out", default="data", help="parent directory (default: data)")
    ap.add_argument("--files", type=int, default=1, help="fineweb-edu: how many Parquet files (about 2 GB each)")
    ap.add_argument("--sample", default="10BT", help="fineweb-edu: which sample (10BT, 100BT, 350BT)")
    ap.add_argument("--keep-parquet", action="store_true", help="fineweb-edu: keep the Parquet files")
    ap.add_argument("--split", default="train_1M", help="openmath2: train_1M, train_2M, train_5M or train")
    ap.add_argument("--sources", default="", help="openmath2: only these problem_source values (comma list)")
    args = ap.parse_args()
    for name in args.names:
        log(f"== {name}")
        DATASETS[name](os.path.join(args.out, name), args)


if __name__ == "__main__":
    main()
