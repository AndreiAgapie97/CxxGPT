#!/usr/bin/env python3
"""get_data.py: download GSM8K and hold out its dev set.

    python3 scripts/get_data.py gsm8k           # grade-school math, train + test
    python3 scripts/get_data.py --check gsm8k   # only check the source

Writes data/gsm8k/ as questions and answers (*.jsonl): one {"question",
"answer"} object per line, the answer being the worked solution, then
"\\n#### " and the final answer, for `cxxgpt prep-qa`:
  - raw/train.jsonl: the upstream train file, unchanged;
  - train.jsonl and dev.jsonl: train without and with the 500 dev problems
    (the 500 questions with the smallest sha256), never trained on, for
    choosing settings;
  - test.jsonl: the test set, for headline numbers only.

--check fetches only a few bytes of each file: it confirms that the files
are where this script expects them, with the fields it reads, without
downloading them.

Downloads resume where they stopped. Needs only the standard library.
"""
import argparse
import filecmp
import hashlib
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.request

DEV_SIZE = 500


def log(msg):
    print(msg, flush=True)


def headers(extra=None):
    h = {"User-Agent": "cxxgpt-get-data"}
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
            if e.code in (401, 403, 404):
                sys.exit(f"{url}: HTTP {e.code}")
            err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            err = e
        wait = 2 ** attempt
        log(f"  {err}; retrying in {wait} s")
        time.sleep(wait)
    sys.exit(f"giving up on {url}")


def retrying(what, fetch, retries=5):
    """fetch(), tried again after 1, 2, 4... s when the network fails or the
    server is busy (HTTP 429, 5xx); other HTTP errors stop the script."""
    for attempt in range(retries):
        try:
            return fetch()
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500:
                sys.exit(f"{what}: HTTP {e.code}")
            err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as e:
            err = e
        if attempt + 1 < retries:
            log(f"  {err}; retrying in {2 ** attempt} s")
            time.sleep(2 ** attempt)
    sys.exit(f"{what}: {err}; giving up")


def fetch_range(url, start, end):
    """Bytes start..end (inclusive) of url, and the size of the whole file."""
    def fetch():
        req = urllib.request.Request(url, headers=headers({"Range": f"bytes={start}-{end}"}))
        with urllib.request.urlopen(req, timeout=60) as r:
            if r.status != 206 and start > 0:
                sys.exit(f"{url}: the server ignored the byte range")
            # "bytes 0-0/12345" on a 206; a 200 sends the whole file.
            size = r.headers.get("Content-Range", "").rpartition("/")[2]
            size = int(size) if size.isdigit() else int(r.headers.get("Content-Length") or 0)
            data = r.read(end - start + 1)
            if len(data) < min(end + 1, size) - start:  # the connection dropped early
                raise ConnectionError(f"got {len(data)} of {min(end + 1, size) - start} bytes")
            return data, size
    return retrying(url, fetch)


def check_url(url, n=1):
    """Checks that url can be downloaded; returns its first n bytes."""
    data, size = fetch_range(url, 0, n - 1)
    log(f"  ok {url} ({size / 1e6:.1f} MB)")
    return data


def check_text(url, n, *needles):
    """Checks that the first n bytes of url contain each needle."""
    head = check_url(url, n).decode("utf-8", "replace")
    for s in needles:
        if s not in head:
            sys.exit(f"{url}: no {s!r} in its first {n} bytes")


def replace_if_changed(part, dest):
    """Moves part over dest, unless dest already holds the same bytes.

    An identical rerun then leaves dest and its timestamp alone, so
    prepare_data.sh does not re-encode token files that are still valid.
    Returns True when dest was already up to date."""
    # filecmp remembers results by size and timestamp; a new .part file can
    # match an earlier one's on a coarse clock, so compare afresh each time.
    filecmp.clear_cache()
    if os.path.exists(dest) and filecmp.cmp(part, dest, shallow=False):
        os.remove(part)
        return True
    os.replace(part, dest)
    return False


def normalize_question(question):
    """Match original problem text, ignoring only whitespace differences."""
    if not isinstance(question, str):
        sys.exit("a GSM8K question must be text, not null or another type")
    return " ".join(question.split())


def dev_rows(questions):
    """The row numbers of the dev problems: the DEV_SIZE questions whose
    sha256 digest is smallest.

    Each question's digest depends only on its own (whitespace-normalized)
    text, so the choice does not depend on Python's random module or on the
    order of the rows in the upstream file. Equal digests would need equal
    questions, which get_gsm8k rejects; the row number only makes the sort
    key complete."""
    def key(i):
        return hashlib.sha256(questions[i].encode("utf-8")).digest(), i
    return set(sorted(range(len(questions)), key=key)[:DEV_SIZE])


def get_gsm8k(out, args):
    base = os.environ.get("GSM8K_ENDPOINT",
                          "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data")
    if args.check:
        for split in ("train", "test"):
            check_text(f"{base}/{split}.jsonl", 4096, '"question"', '"answer"', "####")
        return
    # Keep the unmodified upstream file separately: reruns must never split an
    # already reduced train.jsonl, including when upgrading old preparations.
    path = download(f"{base}/train.jsonl", os.path.join(out, "raw", "train.jsonl"))
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    questions = [normalize_question(row["question"]) for row in rows]
    if len(set(questions)) != len(questions) or len(rows) <= DEV_SIZE:
        sys.exit("GSM8K train must contain more than 500 distinct questions")
    held_out = dev_rows(questions)
    for split in ("train", "dev"):
        dest = os.path.join(out, f"{split}.jsonl")
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for i, row in enumerate(rows)
                       if (i in held_out) == (split == "dev"))
        with open(dest + ".part", "w", encoding="utf-8") as f:
            f.write(text)
        replace_if_changed(dest + ".part", dest)
    log(f"  GSM8K: {len(rows) - DEV_SIZE} train, {DEV_SIZE} dev (smallest sha256 of the question)")
    download(f"{base}/test.jsonl", os.path.join(out, "test.jsonl"))


DATASETS = {
    "gsm8k": get_gsm8k,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", choices=sorted(DATASETS))
    ap.add_argument("--out", default="data", help="parent directory (default: data)")
    ap.add_argument("--check", action="store_true",
                    help="only check that the files exist, with the fields read here (downloads a few KB)")
    args = ap.parse_args()
    for name in args.names:
        log(f"== {name}")
        DATASETS[name](os.path.join(args.out, name), args)


if __name__ == "__main__":
    main()
