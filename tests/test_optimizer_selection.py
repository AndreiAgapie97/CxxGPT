#!/usr/bin/env python3
"""Independent examples for the optional native-experiment selection protocol."""
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

selector = Path(__file__).resolve().parents[1] / "scripts/select-reasoning-optimizer.py"
with tempfile.TemporaryDirectory() as tmp:
    run = Path(tmp)
    (run / "targets").mkdir()
    (run / "dev-probe.txt").write_text("development fixture\n")
    (run / "targets/probe.txt").write_text("final fixture\n")
    for variant in ["dense", "moe-ternary"]:
        for arm in ["adamw-lr0015", "adamw-lr003", "muon-lr0015", "muon-lr003"]:
            for seed in [1, 2, 3]:
                # Median prefers [11,11,11] over [0,10,600], although its
                # mean is lower. The other arms tie, so validation then ID
                # determine the choice without any final-score input.
                count = [0, 10, 600][seed-1] if arm == "adamw-lr003" else 11
                loss = 0.5 if arm == "muon-lr0015" else 0.4
                prefix = run / f"{variant}-seed{seed}-{arm}"
                prefix.with_suffix(".ckpt").write_bytes(b"fixture")
                with prefix.with_suffix(".dev.audit.csv").open("w") as f:
                    w = csv.writer(f)
                    w.writerow(["metric", "correct", "total", "mean_loss", "tokens"])
                    w.writerow(["exact_answer", count, 600, 0, 0])
                    w.writerow(["sampled_val_token_mean", 0, 0, loss, 694])
    def execute():
        return subprocess.run([sys.executable, str(selector), str(run)], capture_output=True, text=True)
    assert execute().returncode == 0
    selection = run / "selection.json"
    manifest = json.loads(selection.read_text())
    assert all(v["selected"] == "adamw-lr0015" for v in manifest["selections"].values())
    original = selection.read_bytes()
    assert execute().returncode != 0 and selection.read_bytes() == original
    selection.unlink(); (run / "development-summary.csv").unlink()
    final = run / "dense-seed1-arm.final.audit.csv"
    final.write_text("any final-score access blocks selection\n")
    assert execute().returncode != 0 and not selection.exists()
    final.unlink()
    bad = run / "dense-seed1-adamw-lr0015.dev.audit.csv"
    valid = bad.read_text()
    bad.write_text(valid.replace("0.4", "nan"))
    assert execute().returncode != 0 and not selection.exists()
    bad.write_text(valid)
    # Equal medians but a larger seed sum exercises the second criterion.
    for seed in [1, 2, 3]:
        p = run / f"moe-ternary-seed{seed}-adamw-lr003.dev.audit.csv"
        p.write_text(p.read_text().replace(f"exact_answer,{[0,10,600][seed-1]},", f"exact_answer,{[0,11,600][seed-1]},"))
    assert execute().returncode == 0
    manifest = json.loads(selection.read_text())
    assert manifest["selections"]["moe-ternary"]["selected"] == "adamw-lr003"
    assert manifest["final_probe_sha256"] == hashlib.sha256((run / "targets/probe.txt").read_bytes()).hexdigest()
print("Optimizer selection: median/sum/loss/ID rules, immutable manifest, final-score gate and non-finite rejection passed")
