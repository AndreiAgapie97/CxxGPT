#!/usr/bin/env python3
"""Exercise generated loss masks and streamed annotations through the real CLI."""
import hashlib
import os
from pathlib import Path
import struct
import subprocess
import tempfile

engine = os.environ.get("CXXGPT", "./cxxgpt")


def run(*args, succeeds=True):
    result = subprocess.run([engine, *map(str, args)], capture_output=True, text=True)
    assert (result.returncode == 0) == succeeds, result.stdout + result.stderr
    return result


def load(path):
    raw = Path(path).read_bytes()
    magic, vocab, masked, count = struct.unpack_from("<8sIIQ", raw)
    assert magic == b"CXTOKEN1"
    ids = struct.unpack_from(f"<{count}H", raw, 24)
    mask = raw[24 + count * 2:] if masked else bytes([1]) * count
    assert len(mask) == count
    return vocab, ids, mask, masked


def byte_text(path):
    vocab, ids, mask, _ = load(path)
    assert vocab == 256
    return bytes(ids), mask


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    common = ["--seed", 41, "--train", 120, "--val", 12, "--test-per-cell", 2,
              "--max-diff", 3, "--chain-plan", "--expr-rewrite"]
    for name, options in [("default", []), ("zero", ["--mistakes", 0]),
                          ("all", ["--mistakes", 1]), ("repeat", ["--mistakes", 1]),
                          ("partial", ["--mistakes", .3])]:
        run("gen-reasoning", "--out", root / name, *common, *options)
    for suffix in ["train.tok", "val.tok", "test.txt"]:
        assert (root / f"default.{suffix}").read_bytes() == (root / f"zero.{suffix}").read_bytes()
        assert (root / f"all.{suffix}").read_bytes() == (root / f"repeat.{suffix}").read_bytes()
    # Captured from unmodified main (d287a66), using the exact common flags.
    baseline = {
        "train.tok": "1d0612d12505ef56a690f23f5dc2e723819744ad5149b55aef0282b8ac25561a",
        "val.tok": "c35a29a476f94e8a167984f1893ed9a47edea983e5665944b15371884be3a5f7",
        "test.txt": "5c4bf5213374d70de66a3955c4e73e7111723fc420674c8b001aa3f277268b0f",
    }
    for suffix, digest in baseline.items():
        assert hashlib.sha256((root / f"zero.{suffix}").read_bytes()).hexdigest() == digest
    assert not (root / "zero.train.text").exists()
    for name in ["all", "partial"]:
        for suffix in ["val.tok", "test.txt"]:
            assert (root / f"default.{suffix}").read_bytes() == (root / f"{name}.{suffix}").read_bytes()
        text = (root / f"{name}.train.text").read_text()
        documents = text.split("<|endoftext|>\n")[:-1]
        assert len(documents) == 120
        planted = sum("[[no-loss]]" in doc for doc in documents)
        assert planted == 120 if name == "all" else 15 < planted < 60
        tokens, masks = byte_text(root / f"{name}.train.tok")
        expected_tokens, expected_masks = bytearray(), bytearray()
        for doc in documents:
            prompt_end = doc.index("\nA: ") + 4
            pos, loss = 0, True
            expected_tokens.append(1)
            expected_masks.append(0)
            while pos < len(doc):
                if doc.startswith("[[no-loss]]", pos):
                    loss = False
                    pos += len("[[no-loss]]")
                elif doc.startswith("[[/no-loss]]", pos):
                    loss = True
                    pos += len("[[/no-loss]]")
                else:
                    expected_tokens.extend(doc[pos].encode())
                    expected_masks.append(int(loss and pos >= prompt_end))
                    pos += 1
        assert tokens == expected_tokens and masks == expected_masks
        assert b"[[no-loss]]" not in tokens
    for value in ["-0.1", "1.1", "nan"]:
        run("gen-reasoning", "--out", root / "bad", *common, "--mistakes", value, succeeds=False)

    for rate in [.1, .3]:
        prefix = root / f"rate-{rate}"
        run("gen-reasoning", "--out", prefix, "--train", 2500, "--val", 0,
            "--test-per-cell", 1, "--seed", 17, "--mistakes", rate)
        planted = Path(str(prefix) + ".train.text").read_text().count("[[no-loss]]")
        assert abs(planted / 2500 - rate) < .025

    # The marker starts before the 8 MiB reader/block boundary and its close
    # lies in the next read; the wrong word can itself contain learned merges.
    wrong = "alpha beta gamma " * 30
    prefix = "a " * ((8 << 20) // 2 - 3)
    annotated = prefix + "[[no-loss]]" + wrong + "[[/no-loss]] correction\n"
    source = root / "stream.txt"
    source.write_text(annotated)
    bpe_source = root / "bpe.txt"
    bpe_source.write_text("alpha beta gamma correction " * 50)
    run("bpe-train", "--in", bpe_source, "--vocab", 285, "--out", root / "mask.bpe")
    raw_vocab = (root / "mask.bpe").read_text().splitlines()
    vocab = [bytes([i]) for i in range(256)]
    for pair in raw_vocab[2:]:
        left, right = map(int, pair.split())
        vocab.append(vocab[left] + vocab[right])
    for bpe in [False, True]:
        outputs = []
        for threads in [1, 2]:
            path = root / f"stream-{bpe}-{threads}.tok"
            options = ["--bpe", root / "mask.bpe"] if bpe else []
            run("text2tok", "--in", source, "--out", path, "--threads", threads, *options)
            _, ids, mask, masked = load(path)
            assert masked == 1
            decoded = b"".join(vocab[i] for i in ids)
            ignored = b"".join(vocab[i] for i, keep in zip(ids, mask) if not keep)
            assert decoded == (prefix + wrong + " correction\n").encode()
            assert ignored == wrong.encode()
            outputs.append(path.read_bytes())
        assert outputs[0] == outputs[1]
    # A later wrong span also covers TokenWriter's prefix-of-ones backfill.
    source.write_text("z " * ((8 << 20) // 2 + 20) + "[[no-loss]]wrong[[/no-loss]] right\n")
    run("text2tok", "--in", source, "--out", root / "late.tok", "--threads", 1)
    text, mask = byte_text(root / "late.tok")
    assert bytes(value for value, keep in zip(text, mask) if not keep) == b"wrong"
    for malformed in ["[[no-loss]]unfinished", "[[/no-loss]]", 
                                        "[[no-loss]]a[[no-loss]]b[[/no-loss]]",
                                        "[[no-loss]]a<|endoftext|>b[[/no-loss]]"]:
        source.write_text(malformed)
        run("text2tok", "--in", source, "--out", root / "bad.tok",
            "--doc-sep", "<|endoftext|>", succeeds=False)
print("CLI: planted mistakes, clean holdouts, repeatability and streamed byte/BPE loss masks passed")
