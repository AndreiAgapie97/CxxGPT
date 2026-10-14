#!/usr/bin/env python3
"""Summarize native development audits and freeze a selection before testing."""
import csv
import datetime
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

run = Path(sys.argv[1])
arms = ["adamw-lr0015", "adamw-lr003", "muon-lr0015", "muon-lr003"]
selections = {}
summary = []
for variant in ["dense", "moe-ternary"]:
    candidates = []
    for arm in arms:
        scores = []
        for seed in [1, 2, 3]:
            prefix = run / f"{variant}-seed{seed}-{arm}"
            with prefix.with_suffix(".dev.audit.csv").open() as f:
                rows = {r["metric"]: r for r in csv.DictReader(f)}
            exact = rows["exact_answer"]
            val = rows["sampled_val_token_mean"]
            count, total = int(exact["correct"]), int(exact["total"])
            loss = float(val["mean_loss"])
            if total != 600 or not 0 <= count <= total or not math.isfinite(loss) or int(val["tokens"]) <= 0:
                raise ValueError(f"invalid development audit: {prefix}")
            scores.append({"seed": seed, "dev_correct": count, "validation_loss": loss,
                           "checkpoint_sha256": hashlib.sha256(prefix.with_suffix(".ckpt").read_bytes()).hexdigest()})
            summary.append([variant, arm, seed, count, total, loss])
        candidates.append({"arm": arm, "median_dev_correct": statistics.median(s["dev_correct"] for s in scores),
                           "sum_dev_correct": sum(s["dev_correct"] for s in scores),
                           "median_validation_loss": statistics.median(s["validation_loss"] for s in scores),
                           "scores": scores})
    candidates.sort(key=lambda x: (-x["median_dev_correct"], -x["sum_dev_correct"],
                                    x["median_validation_loss"], x["arm"]))
    selections[variant] = {"selected": candidates[0]["arm"], "candidates": candidates}

manifest = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "rule": "maximum median development exact count, then sum exact, then minimum median validation loss, then arm ID",
            "dev_probe_sha256": hashlib.sha256((run / "dev-probe.txt").read_bytes()).hexdigest(),
            "final_probe_sha256": hashlib.sha256((run / "targets/probe.txt").read_bytes()).hexdigest(),
            "selections": selections}
# Refuse to replace an existing selection or score after final-test access.
if any(run.glob("*.final.*")):
    raise ValueError("final scores already exist; selection must precede them")
with (run / "selection.json").open("x") as f:
    json.dump(manifest, f, indent=2); f.write("\n")
with (run / "development-summary.csv").open("x") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["variant", "arm", "seed", "dev_correct", "dev_total", "validation_loss"])
    w.writerows(summary)
for variant, selection in selections.items():
    print(variant, selection["selected"])
