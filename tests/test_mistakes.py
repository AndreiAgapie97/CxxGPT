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
    common = ["--seed", 41, "--train", 120, "--val", 12, "--test-per-cell", 2, "--max-diff", 3]
    # With --mistakes 0 the output must stay byte-identical to main's, so
    # these SHA-256 digests were taken from main with the same flags (renewed
    # 2026-10-16 when the curriculum lost its degenerate zeros, PR #60; the
    # same Windows build reproduced the old digests first). A mismatch means the generator changed. If that change is
    # intended (it alters every problem, not only planted ones), regenerate
    # them from the new main: for each format set below,
    #   ./cxxgpt gen-reasoning --out base --seed 41 --train 120 --val 12 \
    #       --test-per-cell 2 --max-diff 3 [--chain-plan --expr-rewrite]
    #   sha256sum base.train.tok base.val.tok base.test.txt
    baselines = [
        ([], {
            "train.tok": "9a38333ce0bbf411176e43e97492aeee84e6ebd099089140d6eff5973aa95184",
            "val.tok": "7f2d7d3ecedd2d58df7346b491b871c363ab2f3769443a123537c12ed6e364c1",
            "test.txt": "cb8263c52df03ffcc5ce0daef367b9f55499c7d79eecddeb45c833875a93fe0b",
        }),
        (["--chain-plan", "--expr-rewrite"], {
            "train.tok": "1fdb1f06a952aacb556779a8776d0e8e3e6435d4e0f712f0415eff9b5bbe7cbb",
            "val.tok": "80371fabbbdc6f3d222732996828fba2d18047624fd0233ab56cfd667d209850",
            "test.txt": "ec8e74e5ad84a898c9e63a42039182e522734c2b7d9019c793f1a06be99ad74b",
        }),
    ]
    for formats, baseline in baselines:
        tag = "formats" if formats else "plain"
        for name, options in [("default", []), ("zero", ["--mistakes", 0]),
                              ("all", ["--mistakes", 1]), ("repeat", ["--mistakes", 1]),
                              ("partial", ["--mistakes", .3])]:
            run("gen-reasoning", "--out", root / f"{tag}-{name}", *common, *formats, *options)
        for suffix in ["train.tok", "val.tok", "test.txt"]:
            assert (root / f"{tag}-default.{suffix}").read_bytes() == (root / f"{tag}-zero.{suffix}").read_bytes()
            assert (root / f"{tag}-all.{suffix}").read_bytes() == (root / f"{tag}-repeat.{suffix}").read_bytes()
        for suffix, digest in baseline.items():
            assert hashlib.sha256((root / f"{tag}-zero.{suffix}").read_bytes()).hexdigest() == digest, \
                f"{tag}-zero.{suffix} differs from main's output; see the comment above the digests"
        assert not (root / f"{tag}-zero.train.text").exists()
        for name in ["all", "partial"]:
            for suffix in ["val.tok", "test.txt"]:
                assert (root / f"{tag}-default.{suffix}").read_bytes() == \
                    (root / f"{tag}-{name}.{suffix}").read_bytes()
            text = (root / f"{tag}-{name}.train.text").read_text()
            documents = text.split("<|endoftext|>\n")[:-1]
            assert len(documents) == 120
            planted = sum("[[no-loss]]" in doc for doc in documents)
            assert planted == 120 if name == "all" else 15 < planted < 60
            tokens, masks = byte_text(root / f"{tag}-{name}.train.tok")
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
            run("text2tok", "--in", source, "--out", path, "--threads", threads, "--loss-markers", *options)
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
    run("text2tok", "--in", source, "--out", root / "late.tok", "--threads", 1, "--loss-markers")
    text, mask = byte_text(root / "late.tok")
    assert bytes(value for value, keep in zip(text, mask) if not keep) == b"wrong"
    for malformed in ["[[no-loss]]unfinished", "[[/no-loss]]",
                      "[[no-loss]]a[[no-loss]]b[[/no-loss]]",
                      "[[no-loss]]a<|endoftext|>b[[/no-loss]]"]:
        source.write_text(malformed)
        run("text2tok", "--in", source, "--out", root / "bad.tok",
            "--doc-sep", "<|endoftext|>", "--loss-markers", succeeds=False)
        # Without --loss-markers the markers are plain text, as on main: web
        # text that happens to contain them must neither fail nor be masked.
        run("text2tok", "--in", source, "--out", root / "plain.tok", "--doc-sep", "<|endoftext|>")
        _, ids, mask, masked = load(root / "plain.tok")
        expected = b"".join(b"\x01" + doc.encode() for doc in malformed.split("<|endoftext|>"))
        assert masked == 0 and bytes(ids) == expected
    source.write_text(annotated)
    run("text2tok", "--in", source, "--out", root / "plain.tok", "--threads", 2)
    _, ids, mask, masked = load(root / "plain.tok")
    assert masked == 0 and bytes(ids) == annotated.encode()
print("CLI: planted mistakes, clean holdouts, repeatability and streamed byte/BPE loss masks passed")
