# Training runs

One row per training run on the laptop, so the history survives without the
`runs/` folder (git-ignored, on the laptop only). `scripts/ledger.sh RUN
"note"` writes a run's row from its files after the run; running it again
updates the row (for example once the full test set has been scored).

What the columns mean:

- **From**: `new` (random weights), or the run whose weights it started
  from (`INIT=`), with a fresh optimizer and learning-rate schedule.
- **Settings**: what the log records: the peak learning rate, the GPU
  products (`tf32`, `bf16`) and attention kernel, and the opt-in training
  options (`sampler qa`, `unlikelihood`). The optimizer is not in the log:
  every night so far used `night.sh`'s default, Muon for the matrices
  inside the blocks.
- **Steps**: done, of planned (`--time-limit` can stop a run early, with a
  checkpoint).
- **Tokens (B)**: billions of tokens trained in this run; **Total** adds
  the runs it started from.
- **Train h**: hours spent in training steps. **GPU h** and **kWh**: the
  whole run by the GPU's own 10-second log (exams and checkpoints
  included), when the run kept one.
- **Val loss**: the last validation loss, on the GSM8K test set's worked
  solutions (lower is better).
- **Exam /250**: the last exam during training (250 GSM8K test questions,
  greedy, no calculator), and the run's best. 250 questions is a small
  exam: 20 / 250 means a true score between about 5% and 12%.
- **Test /1319**: the whole GSM8K test set, from `eval-reasoning --labels`
  (greedy, no calculator, 400 new tokens at most).

| Run | Ended | From | Model | Settings | Steps | Tokens (B) | Total (B) | Train h | GPU h | kWh | Val loss | Exam /250 | Test /1319 | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| night1 | 2026-10-14 | new | 267M (98M active), MoE 8 x top-2 | lr 1.50e-03, tf32, cuBLAS attention | 5352 of 5700 | 0.351 | 0.351 | 7.25 | - | - | 1.1065 | 20 (best 20) | 92 / 1319 (7.0%) | Under WSL2, before the native Windows port. The first night: from random weights. |
| night2 | 2026-10-15 | night1 | 267M (98M active), MoE 8 x top-2 | lr 1.50e-03, bf16, fused tensor-core attention | 9640 of 9640 | 0.632 | 0.983 | 6.83 | 6.75 | 0.69 | 0.9757 | 37 (best 44) | 202 / 1319 (15.3%) | Native Windows, from night 1's weights. The re-warmed learning rate first shook them (exam 4-9 by step 2500, val 1.25); they recovered by step ~4500 and passed night 1 from step 5500. |
| rp1 | 2026-10-17 | RunPod fresh-eu (step 46,946) | 267M (98M active), MoE 8 x top-2 | lr 1.20e-04 flat (wsd), bf16, fused tensor-core attention, grouped experts, sampler qa | 46946 -> 58784 of 90995 | 0.776 | 3.852 | 7.26 | 7.30 | 0.76 | 1.8198 | 7 (best 13) | - | Night 1 of rp1: the RunPod model (step 46,946) continued at a flat lr 1.2e-4 on the 2026-10-16 mix (word problems x3, GSM8K train x64, equations, fractions, reasoning, 12 licensed shards), --sampler qa. Exams flat (mean 8.2 / 250, 200 new tokens). GSM8K answer loss: dev 1.79 -> 1.89, train 0.255 -> 0.269 (memorized since RunPod). Dev 500 at 400 tokens: greedy 3.6%, maj@8 3.8%, pass@8 17.2%; with calc-plain maj@8 4.8%, pass@8 18.8%. |
