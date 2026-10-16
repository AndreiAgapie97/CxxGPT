# CxxGPT: overnight cloud-training handoff for Claude

## Progress update — 16 October 2026, 06:08 UTC (08:08 CEST)

Training remains balanced at approximately53% worked math /47% general books and papers, without resetting weights, optimizer or RNG. Stage32 is running on the same Romanian RTX4090, with100%GPU/10,680MiBVRAM. Directly checked step41,800 means **2,739,404,800 sampled tokens including repeats**; the separately assembled10B corpus remains complete.

**Best reserved math development score is now19/500 (3.8%)**, stage30, with book loss2.5306. **Latest evaluated score is17/500 (3.4%)**, stage31 at06:03:53UTC, book loss**2.5226**. Baseline was6/500 (1.2%), book loss2.5745. General book loss improved throughout these checks; this is limited evidence on a fixed held-out sample. Math accuracy remains low. The original1319test still has not been evaluated in this cloud run; both frozen models will be scored once after selection closes.

The latest verified **interim** recovery archive is2,188,182,814bytes (2.19GB), SHA`50da122ba69a0f05f0a7d0b5071bae83191cebaf5acceadea9c0ba6071c7ea02`. Its full checkpoint was captured05:56:19UTC and verified06:01:13UTC, SHA`91c0d12de6b791cab5145698ee9c5b79f3d5694428f2b28d3fc3924abb725c5e`. The final best inference model is not yet included. The final evaluator and both budget guards are alive; final recovery still requires both models and all answers, with checksums, before deleting resources.

Estimated cost is approximately **$7.76 including storage**, not a final invoice. Training still stops09:10CEST/07:10UTC; hardGPUoff09:45CEST/07:45UTC; $10total, Europeonly. No new resources or recipe changes. Source remains uncommitted/unpushed; recoveryarchive is the transfer method. Older dated snapshots follow.

## Progress update — 16 October 2026, 05:14 UTC (07:14 CEST)

This update supersedes changing counts/results in the original snapshot below; its model architecture, licensing limits, source state and budget instructions still apply.

- **The corpus is complete:** 10,000,016,185 train tokens, 103,058,222 validation and 101,416,161 test tokens. Train sources: OpenStax/LibreTexts 15,071,804; DOAB 434,942,604; arXiv 5,870,495,210; PubMed 3,679,506,567. Math/proof heuristic labels total 2,653,164,709.
- **Training is still running**, with the same weights/optimizer/RNG and balanced approximately 53% worked math /47% books and papers. Stage29, measured step37500 =2,457,600,000 sampled tokens, including repeats; this is not a full10B-token pass. RTX4090 in Romania, 100% GPU, 10,680MiB VRAM, approximately94k active tokens/s.
- **Math dev baseline:** 6/500 (1.2%); fixed held-out DOAB book loss2.5745. **Best preserved so far:** stage27, 17/500 (3.4%), book loss2.568. **Latest evaluated:** stage28, 14/500 (2.8%), book loss2.5565. Same500 reserved dev questions; greedy, no calculator, max400 generated tokens. Math remains weak. The book-loss check shows slight improvement on that fixed sample, not proof of general capability.
- Original1319-question GSM8K test remains reserved until development selection freezes. Both frozen best and latest will then be scored once, with no test-driven selection. Laptop202/1319=15.3% is a different split/recipe from current development results.
- Newer **interim** off-Pod backup: snapshot04:50:58UTC, verified04:55:48UTC; full checkpoint SHA`8210e4592095b2a847744068a654ad02a48c640352760cf0955df19f783e6028`. Recovery archive2,187,344,037bytes (2.19GB decimal), independently rehashed. It contains the latest copied full checkpoint, source/history/tokenizer/provenance. The final best inference model is still pending; final archive expected around3GB.
- Current spend estimate about **$6.98 including storage**, not an invoice. Training deadline remains **09:10CEST/07:10UTC**, absolute GPUoff **09:45CEST/07:45UTC**, total budget$10. No extra resources. Final cleanup requires both models safely copied and hashed, final evaluation completion, and a packaged archive. GPU must stop by cutoff even if recovery fails; retain volume if model copies are unsafe.
- Live operational check confirmed controller3188, final watcher3702, Podguard2995, localguard4366. The final watcher is the latest version scoring both models. Local recovery was tightened to require the final completion marker, then reloaded through the existing foreground supervisor. Source changes remain uncommitted and unpushed; use the archive to recover them.

Measurements checked on **16 October 2026, approximately 06:29 CEST (04:29 UTC)**. Training and corpus assembly are **still running**. This is an interim handoff, not a final-results report.

## Owner's instructions and current objective

Repository: `AndreiAgapie97/CxxGPT`. The owner authorized a fresh run, approximately 10B tokens of high-quality traceably licensed data, math included, with **$10 total Runpod spending including storage, European GPUs only**, and downloadable source/model recovery when training ends or funds run out. The owner has a phone and no available laptop terminal.

The latest preference is **improve math while preserving general ability**. After a disappointing math baseline, we briefly increased worked-math exposure to about 82% of sampling weight. The owner objected to making the model too specialized. The current continuation is approximately **53% worked math and 47% general books/papers** by input sampling weight. Answer masks change the fraction of actual loss-bearing targets. Please preserve this balance rather than reverting to math-only or broad-only training.

## Hardware, cost and timing

- Runpod Secure Cloud, **one RTX 4090, 24GB VRAM**, in **EU-RO-1, Romania**. No additional GPUs were rented.
- Actual `nvidia-smi`: 24,564 MiB total, about 10,680 MiB used, 99–100% GPU utilization, approximately 420W during training.
- Host allocation reported approximately 16 vCPUs and 61GB RAM.
- Pod: `sg78jrp2eapkyj`; 50GB STANDARD network volume: `llweco3tg6`, mounted at `/workspace`.
- GPU rate: **$0.89/hour**. Network volume: $3.50/month prorated hourly. Container disk: 30GB, $0.10/GB/month prorated. Budget estimate through the hard cutoff: approximately **$9.25**, not a verified final invoice.
- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`. The image provides the CUDA environment; the model itself runs in the repository's **native C++/CUDA implementation**, without a Python model framework. Python is used for data and orchestration.
- Pod creation: **15 October, 23:28 CEST (21:28 UTC)**.
- Planned end of training: **16 October, 09:10 CEST (07:10 UTC)**, allowing final evaluation and recovery.
- Absolute GPU cutoff: **16 October, 09:45 CEST (07:45 UTC)**. Do not extend this or rent replacement resources without new owner instructions.
- Independent cloud-side and on-Pod stop guards are armed. Actual shutdown and resource deletion still need final readback verification.

## Code and model

Cloud checkout: `/workspace/CxxGPT`, base commit:

`e1ae9f95641e5c05f8a16d5e9d5df1b8ef9e46a7`

Cloud additions are **uncommitted and have not been pushed to GitHub**. A plain `git pull` on the laptop will not retrieve them. The downloadable recovery archive includes the working tree and a Git history bundle. Compare against the base commit before integrating with any newer laptop work.

The run began from **random weights**, not the laptop's trained checkpoint:

- **267,479,808 total parameters**, approximately **97.610M active per token**.
- Width 768, 12 layers, 12 heads, head size 64.
- MoE: 8 experts, top-2 routing, expert hidden size 1024, no shared expert.
- Context 512; microbatch 16; accumulation 8: **65,536 input tokens per optimizer update**.
- BF16 matrix inputs and fused tensor-core attention; fp32 master weights and optimizer state. This run is not ternary.
- Muon, clipping 1.0, seed 1. Original broad run used LR 0.0015 and a 43,608-step cosine schedule. The continuation uses peak schedule parameter 0.0011 with goal 50,000 steps and cosine minimum fraction 0.1, approximately continuing the current LR rather than resetting to a large fresh LR. Optimizer and sampler RNG state are preserved.
- Around the snapshot: **step 33,900**, approximately **2.22B sampled input tokens** including repetitions. Latest balanced training loss was around 1.50–1.55; sampled GSM8K answer-token validation loss around 1.39. These are different objectives/data mixes from the earlier book losses.

The tokenizer is a fresh **16,384-token byte BPE**, trained on a training-only sample. Canonical SHA-256:

`6e811ced25afd7cf85f9d6254e421165d17490f61a78407f47eb714eee8213fa`

**Keep this tokenizer paired with these checkpoints.** The laptop tokenizer/token files may differ; equal vocabulary sizes do not imply compatible token IDs.

## Throughput and memory measurements

An isolated short same-checkpoint/data benchmark kept 65,536 tokens/update fixed:

| Microbatch / accumulation | Median training-step throughput | Peak VRAM |
| --- | ---: | ---: |
| 16 / 8 | 94,674 tokens/s | 10,680 MiB = 10.43 GiB |
| 32 / 4 | 94,145 tokens/s | 17,196 MiB = 16.79 GiB |

Each arm ran about 15 seconds / 22 updates. The larger batch did not improve speed, so 16/8 was retained. These are training-step timings excluding loading, evaluation, checkpoint I/O and transfers. They do not establish an efficiency advantage over another implementation at matched quality.

The current configuration should fit a 16GB GPU, but that hardware has not been benchmarked. The 4090's 24GB capacity is not required by this particular batch. Training 10B tokens at 94k/s would take roughly 29.5 hours of steps; this $10 run does **not** cover a full 10B-token pass.

## Broad corpus: collected data is not trained-token exposure

Target: **10,000,000,000 measured training tokens**, excluding validation/test and sampling repetitions. At the snapshot, committed corpus totals were:

- Train: **8,674,058,365 tokens**.
- Validation: 88,933,541.
- Test: 88,065,826.
- Status: building; the 10B target was not yet complete.

| Source | Committed training tokens | Source shards committed |
| --- | ---: | ---: |
| Common Pile LibreTexts, selected OpenStax material | 15,071,804 | 1 |
| Common Pile DOAB open books | 434,942,604 | 5 |
| Common Pile openly licensed arXiv papers | 5,870,495,210 | 8 |
| Common Pile licensed PubMed papers | 2,353,548,747 | 1 |

Pinned upstream revisions:

- `common-pile/libretexts_filtered`: `70388bca52b4a93515e14b1d56618fd7944988fd`
- `common-pile/doab_filtered`: `defb24ca72ef6aba6ce0228b669eec06dcfbffbc`
- `common-pile/arxiv_papers_filtered`: `033cf7f53f9b348deec868c1a5a48484f3ee9e52`
- `common-pile/pubmed_filtered`: `c156f0569a92d8f2edc33cebe1f72f7d3e1cae84`

The pipeline accepts explicit **CC BY 4.0 or CC0 1.0**, requires source URL/author metadata, and applies text length/decoding/quality checks. Missing, ambiguous, NC, ND and SA licensing is excluded. Licensing is traced to publisher metadata and retained evidence; there is no independent certification or universal factual-quality guarantee. arXiv papers are not all peer reviewed.

Raw source shards are checksum-verified before processing. Document attribution retains upstream metadata, source revision, raw line, license, normalized text hash, token offsets/counts and math labels. Raw files are deleted after committed tokenization to fit the 32GiB cloud disk and remain reproducible from pinned sources. Global normalized exact-document deduplication is applied; near duplicates are not comprehensively removed.

Book sections and paper versions are grouped when splitting: hash of source/group ID modulo 100, validation=0, test=1, training=2–99. The tokenizer sample uses only training groups.

Approximately 2.653B corpus tokens had math/proof labels at the snapshot. These include title-based and proof-text heuristics; they are not a certified subject classification. Mathematical papers are also not equivalent to worked question/answer training.

The corpus lives in cloud `/workspace/CxxGPT/data/licensed-10b`. Assembly runs separately from GPU training, so neither 8.67B collected tokens nor the weighted virtual training-stream size means the model has read all those unique tokens.

## Worked-math data and current balanced mixture

### Human-written GSM8K

- OpenAI GSM8K, actual **MIT license** downloaded and retained.
- Reserve 500 original training questions as development data; retain all 1,319 original test questions separately.
- From 6,973 remaining training questions, remove seven records with held-out 13-word overlap: **6,966 training questions, 1,238,589 input tokens**.
- Preserve answer-only loss masks. Dev/test questions are excluded from this supplement.

### Repository-generated exact word problems

- Generated 1M unique problems with native `gen-word`, no calculator notes, 2–8 operations.
- Training seed `202610160`; separate generated dev seed `202610161`, with distinct train/test sentence patterns.
- Reject 36,911 training records with 13-word overlaps against GSM8K dev/test questions or the separate generated dev questions.
- Retain **963,089 training records, 193,460,542 input tokens**.
- Separate generated dev: 1,000 questions, 222,566 tokens.
- Source is locally generated mathematical data from repository templates, not an external dataset asserted to have a new license. Generator source hash, seeds, filtering and checksums are retained.
- Training token-file SHA-256: `fe251bda53379cdafbe39a22e4f01b2e354a1601e0a92993cb37c3bb3b2787a1`.

### Current mixture

- Generated word problems x4: 773,842,168 weighted input tokens.
- GSM8K training x64: 79,269,696.
- General replay: 750,046,644 book/paper tokens, a deterministic subset of uploaded corpus shards.
- Total virtual weighted stream: **1,603,158,508 tokens**, approximately **53.2% worked math / 46.8% replay**. This is a sampling distribution, not additional unique corpus content or a completed pass.
- `--sampler qa` keeps the full question visible when the QA document fits the context. Masks train answer tokens; general text remains fully supervised.
- Current unique input material in this continuation is approximately 944,745,775 tokens.
- Broader academic literature was not exhaustively screened for benchmark quotations; do not claim complete benchmark decontamination.

## Evaluations: completed results versus pending results

The broad-pretraining model scored **6/500 = 1.2%** on the full reserved GSM8K development set: greedy decoding, BF16 GPU, batch 16, no calculator, maximum 400 new tokens, context 512. All gold completions fit the 400-token budget. There were 61 unfinished answers. One of six correct final answers contained an incorrect intermediate calculation.

A preceding 32-question check scored 0/32, on an earlier checkpoint with a 256-token generation cap. It should not replace the full baseline or be treated as the same evaluation protocol.

Full baseline checkpoint SHA-256:

`5bbde6eeb6a55203e313b4ccb12f9996a7405af67758fe62f67fb35782f772c0`

The repository's laptop ledger records night 2 as **202/1319 = 15.3% on the original test**, after 0.983B cumulative training tokens. That is meaningful prior project evidence, but a different split, checkpoint history and data recipe. The current cloud result is not a controlled comparison of hardware or architectures. Falling prediction loss alone does not establish reasoning quality.

A fixed held-out DOAB book check on the broad checkpoint gave **2.5745 loss**. The check performs **zero training updates**: it uses a hardlinked copy with `--resume --steps` equal to the existing checkpoint step, and measures eight fixed B16/T512 validation batches with the native evaluator. The source model remains intact.

Every approximately 20-minute continuation stage now:

1. Saves the latest full checkpoint.
2. Generates and saves all 500 GSM8K dev answers and exact-answer scores.
3. Measures fixed general-text book loss on the same checkpoint.
4. Keeps the strictly best math-dev checkpoint separately, without discarding the latest balanced checkpoint.

At this snapshot, **the first complete evaluation after rebalancing was still pending**. The latest training loss/answer-token validation is not a new exact-answer accuracy result.

At the end, selection freezes using dev only. A final watcher exports the best math model and evaluates both that frozen model and the latest balanced model on the original 1,319-question test, skipping duplicate work if their weights match. These final test results were not yet available. Test scores must not be used to choose a new recipe or cherry-pick a model.

The project also has older controlled synthetic-task studies in `docs/research.md` and `docs/results/2026-10-14/`; these show narrow learned arithmetic/comparison behavior and failures on unseen digit lengths. Do not confuse those studies with GSM8K results from this fresh 267M run or with proof of general reasoning.

## Validation and issues fixed during the night

- CUDA compilation succeeded. The original full GPU suite reported **three tolerance failures**: fp32 fused head64/head128 normalization-gradient checks and a ternary MoE gradient check. BF16 MoE/Muon checks passed, plus a targeted BF16 MoE check for the selected head64 path under existing BF16 tolerances. Do not describe the entire GPU suite as passing.
- Data/license tests passed, the native streaming tokenizer matched stock `text2tok` payloads, and math token masks/payloads were checked.
- Some cloud-side detached workers stopped while GPU training continued. Helpers now run under a foreground supervising process, plus scheduled follow-ups. On-Pod training/stop guards remain independent of the chat workspace.
- Re-uploading the math file directly caused a stage-restart `truncated tokens` failure. Step 30,590 had saved safely. The file was SHA-verified and atomically replaced, and training resumed without losing saved progress. Immutable tokenizer/math files are no longer rewritten by corpus sync startup.
- A Jupyter protocol bug was found: using `chunk=-1` for the first/only upload chunk appends if the target exists. `Client.put` now uses normal overwrite PUT for small files. A repeated-overwrite byte-exact runtime check passed. Existing controller/guard files were replaced cleanly. The tokenizer had five identical serialized copies; native loading consumed the first declared merge table, so token semantics remained unchanged. The canonical single-copy tokenizer was restored and hashed before the full 500-question baseline.
- Empty/mutable log downloads caused HTTP416 with ranged transfer. Logs now use the contents API. Immutable checkpoints use four resumable 8MiB range workers, a completed-chunk ledger, stable hardlinked snapshots and full-file SHA verification. A preallocated `.part` file's apparent size does not indicate completed transfer.
- **Never restart the original `train_controller.py` startup**: it creates a fresh probe and can overwrite a trained output. The active balanced continuation is `math_focus_controller.py`, which preserves weights/optimizer/RNG. A previous resume-only controller is retained for reference but would use the older data recipe.

## Recovery, model formats and current artifacts

The cloud-side recovery archive is `/workspace/library-files/CxxGPT-recovery.tar.gz`, with a SHA-256 sidecar. It includes working source, a Git history bundle, tokenizer, provenance/metadata, operational scripts and logs. The bulk approximately 20GB token corpus at 10B tokens remains separate.

The existing archive at this snapshot was **2,098,545,675 bytes**, backed up around 05:51 CEST. It contains an **earlier approximately 2B-sampled-token checkpoint**, not the final balanced model. Its checkpoint SHA is:

`c712c9fe28c61952af98837c22cb26d38fb6c147210bf60a1f5680d6eb5b0cea`

Final recovery is configured to preserve both:

- `runs/fresh-eu.ckpt`: latest full model plus optimizer/RNG, approximately 3.21GB uncompressed; supports `--resume`.
- `runs/best-math-inference.ckpt`: best math-dev model without optimizer moments, approximately 1.07GB uncompressed; use for inference/evaluation or `--init` for further training.

New native utility `tools/export_weights.cpp` compiled and passed a full round-trip check: every parameter, routing bias, training step and RNG state is identical after export/reload, and optimizer moments are absent. No weight quantization is involved. The final compressed archive is expected to be around 3GB, subject to actual compression and logs.

The final supervisor must verify **both** models off-Pod, package the archive, then stop/delete only this Pod and volume and confirm readback. If recovery fails, preserve the volume and stop the GPU by the hard cutoff. An older verified archive is not proof that final training progress is safe.

## Where Claude should look

Cloud-local checkout additions:

- `configs/licensed-10b-sources.json`: pinned source inventory and raw-shard checksums.
- `scripts/prepare_licensed.py`, `scripts/build_licensed_corpus.py`, `scripts/prepare_math_supplement.py`.
- `tools/tokenize_stream.cpp`, `tools/export_weights.cpp`.
- `scripts/package_recovery.py`, `tests/test_licensed_data.py`.
- `docs/RUNPOD_TRAINING.md` and this handoff.

On the Pod, under `/workspace/CxxGPT/runs`:

- `training-status.json`, `stage-NNN.json`: actual active recipe/commands.
- `fresh-eu.csv/.log`: broad-pretraining history; `math-focus.csv/.log/.load.csv`: continuation history.
- `math-focus-recipe.json`, `math-best.json`, `math-latest-eval.json`.
- `math-stage-NNN.json/.log/.answers.jsonl`: stage scores, general loss and all dev answers.
- `math-general-baseline.json`, `math-dev500-metadata.json`, `math-dev500-answers.jsonl`.
- Final outputs: `math-final-test.*`, `math-final-latest-test.*`, `best-inference.json`, `final-eval-ready.json`.

Cloud operations are in `/workspace/cxxgpt-ops`; sanitized copies are included under `CxxGPT/operations` in the archive. `SESSION_NOTES.md` is historical: read the newest entry first. `keep_workers.py` supervises local helpers; `math_focus_controller.py`, `prepare_targeted_math.py`, `final_math_evaluation.py` run on the Pod. Treat PIDs as transient and inspect current state before restarting anything.

For authorized access, the existing official Runpod OAuth connection is used via `mcp_bridge.py`; remote commands/file transfer use `jupyter_client.py`. **Never print, copy into this handoff, or package credential files.** A raw `get-pod` response includes the Jupyter password in `env`, so sanitize it before showing it. Do not provision new resources, extend the budget, or overwrite newer laptop changes.

User-facing files in `/workspace/library-files`:

- `CxxGPT-evaluation.md`: readable full-baseline assessment with real example outputs.
- `CxxGPT-math-answers.jsonl`: all 500 baseline questions, expected answers, generated responses and diagnostic labels.
- `CxxGPT-status.md`: periodically refreshed training/corpus status.
- `CxxGPT-math-progress.md` and `CxxGPT-math-latest-answers.jsonl`: published after a continuation-stage evaluation becomes available.
- `CxxGPT-recovery.tar.gz` and its checksum: inspect `RECOVERY.json` for the exact contents/version.

Please judge progress using actual answer accuracy and general-text retention together, preserve the original test as a final check, and distinguish measured results from planned evaluations or future recovery actions.
