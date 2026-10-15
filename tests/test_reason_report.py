#!/usr/bin/env python3
"""Exercise optional training reports through the real CLI, on a tiny CPU model."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    binary = str(Path(sys.argv[1]).resolve())
    with tempfile.TemporaryDirectory(prefix="cxxgpt-reason-report-") as tmp:
        root = Path(tmp)

        def run(*args, success=True):
            result = subprocess.run([binary, *map(str, args)], cwd=root, capture_output=True,
                                    text=True, encoding="utf-8", errors="replace")
            assert (result.returncode == 0) == success, result.stdout + result.stderr
            return result.stdout + result.stderr

        (root / "text.txt").write_text("abcdefghi\n" * 30, encoding="utf-8")
        run("text2tok", "--in", "text.txt", "--out", "data.tok", "--threads", 1)
        (root / "problems.txt").write_text(
            "".join(f"gsm8k\t1\tQuestion {i}?\t1 + 1 = 2\t2\n" for i in range(7)), encoding="utf-8")
        common = ["train", "--data", "data.tok", "--val", "data.tok", "--seq", 64,
                  "--dim", 8, "--heads", 2, "--ffn", 12, "--layers", 1, "--batch", 1,
                  "--steps", 3, "--warmup", 0, "--eval-every", 2, "--eval-batches", 1,
                  "--save-every", 0, "--log-every", 1, "--threads", 1,
                  "--reason-test", "problems.txt", "--reason-every", 2, "--reason-limit", 3, "--max-new", 8]
        plain = run(*common, "--out", "plain.ckpt")
        assert not list(root.glob("*.jsonl")), "reports must be opt-in"
        reported = run(*common, "--out", "reported.ckpt", "--reason-report", "reports/exam")
        assert (root / "plain.ckpt").read_bytes() == (root / "reported.ckpt").read_bytes(), \
            "reporting changed model, optimizer or sampler state"

        def read(step):
            rows = [json.loads(line) for line in (root / f"reports/exam.step-{step}.jsonl").read_text().splitlines()]
            assert len(rows) == 5
            settings, *answers, summary = rows
            assert settings["step"] == step and settings["test"] == "problems.txt"
            assert settings["max_new"] == 8 and not settings["gpu_requested"]
            assert [row["problem"] for row in answers] == [0, 2, 4]
            assert [row["question"] for row in answers] == [f"Question {i}?" for i in (0, 2, 4)]
            assert all(isinstance(row["completion"], str) for row in answers)
            assert sum(summary["labels"].values()) == 3
            assert summary["type"] == "summary"
            return summary

        assert read(2)["previous_step"] is None
        assert read(3)["previous_step"] == 2
        assert "since reasoning step 2:" in reported and "reasoning report:" not in plain
        run(*common, "--out", "reported.ckpt", "--resume", "--steps", 4, "--reason-report", "reports/exam")
        assert read(4)["previous_step"] is None, "resume must start a new baseline"
        error = run("train", "--data", "data.tok", "--reason-report", "invalid", success=False)
        assert "--reason-report requires --reason-test" in error
    print("CLI: reasoning reports, source IDs, resume baseline and unchanged checkpoints passed")


if __name__ == "__main__":
    main()
