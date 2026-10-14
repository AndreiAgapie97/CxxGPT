#!/usr/bin/env bash
# Exercise the real CLI after config parsing and backend selection.
set -euo pipefail
CXXGPT=${CXXGPT:-./cxxgpt}
CXXGPT_TRAINING_CLI=${CXXGPT_TRAINING_CLI:-./test_training_cli}
CXXGPT_AUDIT=${CXXGPT_AUDIT:-./reasoning_audit}
CXXGPT_TARGETS=${CXXGPT_TARGETS:-./reasoning_targets}
CXXGPT_TRACES=${CXXGPT_TRACES:-./reasoning_traces}
CXXGPT_CURRICULUM=${CXXGPT_CURRICULUM:-./reasoning_curriculum}
cd "$(dirname "$0")/.."
TEST_DIR=$(mktemp -d)
trap 'rm -rf -- "$TEST_DIR"' EXIT
printf 'abcdefghi' > "$TEST_DIR/text.txt"
"$CXXGPT" text2tok --in "$TEST_DIR/text.txt" --out "$TEST_DIR/data.tok" > "$TEST_DIR/data.log"
COMMON=(--data "$TEST_DIR/data.tok" --val "$TEST_DIR/data.tok" --seq 8 --dim 8 --heads 2
        --ffn 12 --layers 1 --batch 1 --steps 1 --warmup 0 --eval-every 1 --eval-batches 1
        --save-every 0 --threads 2 --out "$TEST_DIR/model.ckpt")
"$CXXGPT" train "${COMMON[@]}" > "$TEST_DIR/train.log"
grep -q '9 train tokens, 9 val tokens' "$TEST_DIR/train.log"
grep -q '| val ' "$TEST_DIR/train.log"
for option in dim heads experts shared ffn ehidden topk batch accum log-every eval-batches nope-every; do
    value=-1
    if [[ "$option" == dim || "$option" == heads || "$option" == topk || "$option" == batch || "$option" == accum || "$option" == log-every || "$option" == eval-batches ]]; then value=0; fi
    status=0
    "$CXXGPT" train "${COMMON[@]}" "--$option" "$value" > "$TEST_DIR/invalid.log" 2>&1 || status=$?
    if [[ "$status" != 1 ]] || ! grep -q 'check failed' "$TEST_DIR/invalid.log"; then
        cat "$TEST_DIR/invalid.log"
        printf 'invalid --%s %s returned %s instead of a checked error\n' "$option" "$value" "$status"
        exit 1
    fi
done
printf 'CLI: single-window train/validation and twelve invalid configurations passed\n'
for threads in 0 -1; do
    status=0
    "$CXXGPT" text2tok --in "$TEST_DIR/text.txt" --out "$TEST_DIR/thread-bad.tok" --threads "$threads" \
        > "$TEST_DIR/thread-bad.log" 2>&1 || status=$?
    [[ "$status" == 1 ]] && grep -q 'threads must be positive' "$TEST_DIR/thread-bad.log"
done
for options in '--dim 4294967304' '--steps 9999999999999999999999999' '--lr nan' '--lr inf' '--lr 1e999' '--lr 0.1junk'; do
    status=0
    # Fixed cases; all must fail in parsing before construction/allocation.
    "$CXXGPT" train "${COMMON[@]}" $options > "$TEST_DIR/strict-args.log" 2>&1 || status=$?
    [[ "$status" == 1 ]] && grep -q 'check failed' "$TEST_DIR/strict-args.log"
done
status=0
"$CXXGPT" train "${COMMON[@]}" --experts 4 --topk 1 > "$TEST_DIR/top1.log" 2>&1 || status=$?
[[ "$status" == 1 ]] && grep -q 'one-logit softmax has zero task gradient' "$TEST_DIR/top1.log"

"$CXXGPT_TRAINING_CLI" generate "$TEST_DIR"
for batch in 0 -1; do
    status=0
    "$CXXGPT" eval-reasoning --ckpt "$TEST_DIR/audit.ckpt" --test "$TEST_DIR/audit.txt" --batch "$batch" \
        --threads 2 > "$TEST_DIR/eval-batch.log" 2>&1 || status=$?
    [[ "$status" == 1 ]] && grep -q 'evaluation batch and max_new must be positive' "$TEST_DIR/eval-batch.log"
done
for count in '' 1junk 99999999999999999999999 0 -1; do
    status=0
    "$CXXGPT" train "${COMMON[@]}" --data "$TEST_DIR/data.tok*$count" > "$TEST_DIR/repeat-count.log" 2>&1 || status=$?
    [[ "$status" == 1 ]] && grep -q 'bad repeat count' "$TEST_DIR/repeat-count.log"
done
status=0
"$CXXGPT" eval-reasoning --ckpt "$TEST_DIR/audit.ckpt" --test "$TEST_DIR/audit.txt" --as-engine \
    --threads 2 > "$TEST_DIR/engine-config.log" 2>&1 || status=$?
[[ "$status" == 1 ]] && grep -q 'requires a ternary model' "$TEST_DIR/engine-config.log"
for target in 1 -1; do
    status=0
    "$CXXGPT_TRAINING_CLI" repeat-backward "$target" > "$TEST_DIR/repeat-backward.log" 2>&1 || status=$?
    [[ "$status" == 1 ]] && grep -q 'needs a new forward' "$TEST_DIR/repeat-backward.log"
done
for command in bad-target bad-target-backward; do
  for target in -2 3; do
    status=0
    "$CXXGPT_TRAINING_CLI" "$command" "$target" > "$TEST_DIR/target-range.log" 2>&1 || status=$?
    [[ "$status" == 1 ]] && grep -q 'target .* out of range' "$TEST_DIR/target-range.log"
  done
done
printf 'qa\t1\tq\twork\t14\r\nqa\t2\ttail\twork\t1080' > "$TEST_DIR/line-endings.txt"
"$CXXGPT_TRAINING_CLI" problems "$TEST_DIR/line-endings.txt"
"$CXXGPT" eval-reasoning --ckpt "$TEST_DIR/audit.ckpt" --test "$TEST_DIR/line-endings.txt" --max-new 1 --threads 2 > "$TEST_DIR/line-endings.log"
grep -q '^overall .* / 2)' "$TEST_DIR/line-endings.log"
for difficulty in '' 1junk 999999999999999999999999 -1; do
    printf 'qa\t%s\tq\twork\t14' "$difficulty" > "$TEST_DIR/bad-difficulty.txt"
    status=0
    "$CXXGPT" eval-reasoning --ckpt "$TEST_DIR/audit.ckpt" --test "$TEST_DIR/bad-difficulty.txt" --threads 2 \
        > "$TEST_DIR/bad-difficulty.log" 2>&1 || status=$?
    [[ "$status" == 1 ]] && grep -q 'bad difficulty' "$TEST_DIR/bad-difficulty.log"
done
"$CXXGPT" eval-reasoning --ckpt "$TEST_DIR/audit.ckpt" --test "$TEST_DIR/audit.txt" --max-new 1 --threads 2 > "$TEST_DIR/cap.log"
grep -q 'gold completions exceeding --max-new 1: 1 / 1' "$TEST_DIR/cap.log"
"$CXXGPT" eval-reasoning --ckpt "$TEST_DIR/audit.ckpt" --test "$TEST_DIR/audit.txt" --threads 2 > "$TEST_DIR/cap-default.log"
grep -q 'gold completions exceeding --max-new 200: 0 / 1' "$TEST_DIR/cap-default.log"
"$CXXGPT" train "${COMMON[@]}" --init "$TEST_DIR/audit.ckpt" --out "$TEST_DIR/cap.ckpt" \
    --reason-test "$TEST_DIR/audit.txt" --max-new 1 > "$TEST_DIR/train-cap.log"
grep -q 'gold completions exceeding --max-new 1: 1 / 1' "$TEST_DIR/train-cap.log"
status=0
"$CXXGPT" eval-reasoning --ckpt "$TEST_DIR/audit.ckpt" --test "$TEST_DIR/audit.txt" --max-new 0 > "$TEST_DIR/bad-cap.log" 2>&1 || status=$?
[[ "$status" == 1 ]] && grep -q 'max-new must be positive' "$TEST_DIR/bad-cap.log"
# Vocab compatibility must fail before constructing/training the model.
for kind in val reason; do
    EXTRA=(--data "$TEST_DIR/masked.tok")
    message='validation vocab'
    if [[ "$kind" == reason ]]; then
        EXTRA+=(--val "$TEST_DIR/masked.tok" --reason-test "$TEST_DIR/audit.txt")
        message='reasoning tokenizer vocab'
    fi
    status=0
    "$CXXGPT" train "${COMMON[@]}" "${EXTRA[@]}" --out "$TEST_DIR/vocab-$kind.ckpt" \
        > "$TEST_DIR/vocab-$kind.log" 2>&1 || status=$?
    [[ "$status" == 1 ]] && grep -q "$message" "$TEST_DIR/vocab-$kind.log"
    ! grep -q '^step ' "$TEST_DIR/vocab-$kind.log"
    [[ ! -e "$TEST_DIR/vocab-$kind.ckpt" ]]
done
# A2/C10: the final save precedes reasoning; SIGINT saves without another eval.
"$CXXGPT" train "${COMMON[@]}" --init "$TEST_DIR/audit.ckpt" --out "$TEST_DIR/save-first.ckpt" \
    --reason-test "$TEST_DIR/audit.txt" > "$TEST_DIR/save-first.log"
awk '/^saved / {saved=NR} /^reasoning eval/ {if (!saved || saved>=NR) exit 1; seen=1} END {if (!seen) exit 1}' "$TEST_DIR/save-first.log"
"$CXXGPT" train "${COMMON[@]}" --init "$TEST_DIR/audit.ckpt" --out "$TEST_DIR/signal.ckpt" \
    --steps 1000000 --save-every 1 --log-every 1 --eval-every 0 --reason-test "$TEST_DIR/audit.txt" \
    > "$TEST_DIR/signal.log" 2>&1 &
trainer=$!
for ((poll=0; poll<1000; ++poll)); do
    [[ -f "$TEST_DIR/signal.ckpt" ]] && break
    if ! kill -0 "$trainer" 2>/dev/null; then wait "$trainer"; exit 1; fi
    sleep 0.01
done
if [[ ! -f "$TEST_DIR/signal.ckpt" ]]; then kill -TERM "$trainer"; wait "$trainer"; exit 1; fi
kill -INT "$trainer"
wait "$trainer"
grep -q 'saved .* (interrupted)' "$TEST_DIR/signal.log"
! grep -q '^reasoning eval' "$TEST_DIR/signal.log"
# Loading/resuming the interrupted checkpoint proves it is complete and usable.
"$CXXGPT" train "${COMMON[@]}" --out "$TEST_DIR/signal.ckpt" --steps 0 --resume > "$TEST_DIR/signal-resume.log"
printf 'CLI: save-before-reasoning and actual SIGINT save/resume passed\n'
# A3/C10: normal time limit and one bad attempt both stop with a usable save.
"$CXXGPT" train "${COMMON[@]}" --steps 20 --time-limit 0.000000000001 \
    --out "$TEST_DIR/timed.ckpt" > "$TEST_DIR/timed.log"
grep -q 'time limit reached' "$TEST_DIR/timed.log"
grep -q 'saved .* at step 1' "$TEST_DIR/timed.log"
"$CXXGPT" train "${COMMON[@]}" --init "$TEST_DIR/unstable.ckpt" --steps 20 \
    --time-limit 0.000000000001 --out "$TEST_DIR/bad-timed.ckpt" > "$TEST_DIR/bad-timed.log"
[[ $(grep -c 'update skipped' "$TEST_DIR/bad-timed.log") == 1 ]]
grep -q 'time limit reached' "$TEST_DIR/bad-timed.log"
grep -q 'saved .* at step 0' "$TEST_DIR/bad-timed.log"
! grep -q 'diverged\|lower --lr' "$TEST_DIR/bad-timed.log"
"$CXXGPT_TRAINING_CLI" unchanged-zero-step "$TEST_DIR/unstable.ckpt" "$TEST_DIR/bad-timed.ckpt"
printf 'CLI: good/bad-attempt time limits preserve progress; divergence protection remains separate\n'
"$CXXGPT" train "${COMMON[@]}" --steps 20 --time-limit 0.000000000001 --eval-every 999 \
    --out "$TEST_DIR/final-val.ckpt" --log "$TEST_DIR/final-val.csv" > "$TEST_DIR/final-val.log"
grep -q '^step *1 | val .*final validation' "$TEST_DIR/final-val.log"
grep -q '^# final validation step 1:' "$TEST_DIR/final-val.csv"
"$CXXGPT" train "${COMMON[@]}" --steps 1 --resume --eval-every 999 \
    --out "$TEST_DIR/final-val.ckpt" --log "$TEST_DIR/final-val.csv" > "$TEST_DIR/final-val-resume.log"
grep -q '^# resumed at step 1$' "$TEST_DIR/final-val.csv"
[[ $(grep -c '^step *1 | val .*final validation' "$TEST_DIR/final-val-resume.log") == 1 ]]
awk -F, '!/^#/ && NF!=7 {exit 1}' "$TEST_DIR/final-val.csv"
for variant in dense ternary moe ternary-moe hybrid hybrid-moe nope nope-ternary-moe; do
    ARCH=()
    if [[ "$variant" == *ternary* ]]; then ARCH+=(--ternary); fi
    if [[ "$variant" == *moe* ]]; then ARCH+=(--experts 4 --topk 2 --ehidden 6 --shared 1); fi
    if [[ "$variant" == hybrid* ]]; then ARCH+=(--ternary --fp-attn); fi
    if [[ "$variant" == nope* ]]; then ARCH+=(--nope-every 1); fi
    for optimizer in adamw muon; do
      OPT=()
      if [[ "$optimizer" == muon ]]; then OPT+=(--muon); fi
      for dataset in masked sparse; do
        TRAIN=(--data "$TEST_DIR/$dataset.tok" --val "$TEST_DIR/$dataset.tok"
               --seq 8 --dim 8 --heads 2 --ffn 12 --layers 1 --steps 4 --warmup 0
               --seed 3 --lr 0.001 --clip 0.1 --eval-every 1 --log-every 1 --save-every 0 --threads 2)
        "$CXXGPT" train "${TRAIN[@]}" "${ARCH[@]}" "${OPT[@]}" --batch 4 --accum 1 --eval-batches 1 \
            --out "$TEST_DIR/macro.ckpt" --log "$TEST_DIR/macro.csv" > "$TEST_DIR/macro.log"
        "$CXXGPT" train "${TRAIN[@]}" "${ARCH[@]}" "${OPT[@]}" --batch 1 --accum 4 --eval-batches 4 \
            --out "$TEST_DIR/micro.ckpt" --log "$TEST_DIR/micro.csv" > "$TEST_DIR/micro.log"
        "$CXXGPT_TRAINING_CLI" compare "$TEST_DIR/macro.ckpt" "$TEST_DIR/micro.ckpt" "${OPT[@]}"
        # Identical eval samples must give the same token-weighted metric,
        # including all-ignored eval batches. Ignore timing columns.
        awk -F, 'NR>1 {print $1 "," $2 "," $3}' "$TEST_DIR/macro.csv" > "$TEST_DIR/macro.loss"
        awk -F, 'NR>1 {print $1 "," $2 "," $3}' "$TEST_DIR/micro.csv" > "$TEST_DIR/micro.loss"
        diff -u "$TEST_DIR/macro.loss" "$TEST_DIR/micro.loss"
        if [[ "$dataset" == sparse ]]; then grep -q 'skipped .* empty sampled groups' "$TEST_DIR/micro.log"; fi
        rm "$TEST_DIR/macro.csv" "$TEST_DIR/micro.csv"
      done
    done
done
status=0
"$CXXGPT" train "${COMMON[@]}" --data "$TEST_DIR/ignored.tok" --val "$TEST_DIR/ignored.tok" > "$TEST_DIR/ignored.log" 2>&1 || status=$?
[[ "$status" == 1 ]] && grep -q 'no predictable targets' "$TEST_DIR/ignored.log"
printf 'CLI: AdamW/Muon masked/sparse accumulation, weighted validation, empty-group skips, and fully masked rejection passed\n'

"$CXXGPT" bench-infer --vocab 13 --seq 8 --dim 32 --heads 2 --layers 1 --ffn 64 \
    --tokens 8 --reps 4 --warmup 1 --threads 2 --log "$TEST_DIR/decode.csv" > "$TEST_DIR/decode.log"
[[ $(wc -l < "$TEST_DIR/decode.csv") == 5 ]]
awk -F, 'NR>1 {if ($1 != NR-2 || $2 != (NR%2==0) || $5 <= 0 || $6 <= 0 || $7 < 0 || $7 > 1 || $10 != 1) exit 1}' "$TEST_DIR/decode.csv"
status=0
"$CXXGPT" bench-infer --vocab 13 --seq 8 --dim 32 --heads 2 --layers 1 --ffn 64 \
    --tokens 8 --reps 0 --threads 2 > "$TEST_DIR/bad-reps.log" 2>&1 || status=$?
[[ "$status" == 1 ]] && grep -q 'reps must be positive' "$TEST_DIR/bad-reps.log"
printf 'CLI: benchmark emits alternating raw trials and rejects zero repeats\n'

"$CXXGPT" train "${COMMON[@]}" --steps 0 --init "$TEST_DIR/model.ckpt" --out "$TEST_DIR/fresh.ckpt" > "$TEST_DIR/init.log"
"$CXXGPT_TRAINING_CLI" fresh "$TEST_DIR/model.ckpt" "$TEST_DIR/fresh.ckpt"
"$CXXGPT" train "${COMMON[@]}" --steps 0 --seed 2 --init "$TEST_DIR/model.ckpt" --out "$TEST_DIR/night2.ckpt" > "$TEST_DIR/night2.log"
"$CXXGPT_TRAINING_CLI" fresh "$TEST_DIR/model.ckpt" "$TEST_DIR/night2.ckpt" 2
"$CXXGPT_TRAINING_CLI" different-order "$TEST_DIR/fresh.ckpt" "$TEST_DIR/night2.ckpt"
# Capture night.sh arguments without reading real data or touching night1 files.
printf '#!/usr/bin/env bash\nprintf "%%s\\n" "$@" > "$MOCK_ARGS"\n' > "$TEST_DIR/night-mock"
chmod +x "$TEST_DIR/night-mock"
for night in 1 2; do
    run="review${night}-${BASHPID}"
    MOCK_ARGS="$TEST_DIR/night.args" RUN="$run" TOKS=100 MIX=unused INIT="$TEST_DIR/model.ckpt" \
        CX="$TEST_DIR/night-mock" bash scripts/night.sh > "$TEST_DIR/night-script.log"
    [[ $(awk '$0=="--seed" {getline; print}' "$TEST_DIR/night.args") == "$night" ]]
    rm "runs/$run.log"
done
"$CXXGPT" train "${COMMON[@]}" --steps 0 --ternary --init "$TEST_DIR/model.ckpt" --out "$TEST_DIR/fresh-ternary.ckpt" > "$TEST_DIR/init-ternary.log"
"$CXXGPT_TRAINING_CLI" fresh-ternary "$TEST_DIR/model.ckpt" "$TEST_DIR/fresh-ternary.ckpt"
"$CXXGPT" train "${COMMON[@]}" --steps 0 --ternary --fp-attn --init "$TEST_DIR/model.ckpt" --out "$TEST_DIR/fresh-hybrid.ckpt" > "$TEST_DIR/init-hybrid.log"
"$CXXGPT_TRAINING_CLI" fresh-hybrid "$TEST_DIR/model.ckpt" "$TEST_DIR/fresh-hybrid.ckpt"
"$CXXGPT" train "${COMMON[@]}" --steps 0 --ternary --fp-attn 0 --init "$TEST_DIR/fresh-hybrid.ckpt" --out "$TEST_DIR/fresh-all.ckpt" > "$TEST_DIR/init-all.log"
"$CXXGPT_TRAINING_CLI" fresh-all-ternary "$TEST_DIR/fresh-hybrid.ckpt" "$TEST_DIR/fresh-all.ckpt"
"$CXXGPT" train "${COMMON[@]}" --steps 0 --nope-every 1 --init "$TEST_DIR/model.ckpt" --out "$TEST_DIR/fresh-nope.ckpt" > "$TEST_DIR/init-nope.log"
"$CXXGPT_TRAINING_CLI" fresh-nope "$TEST_DIR/model.ckpt" "$TEST_DIR/fresh-nope.ckpt"
for options in '--matmul invalid' '--no-tf32 --matmul bf16' '--fused-attn --no-fused-attn'; do
    status=0
    # Fixed literal cases, no user-supplied option splitting.
    # shellcheck disable=SC2086
    "$CXXGPT" train "${COMMON[@]}" $options > "$TEST_DIR/precision-invalid.log" 2>&1 || status=$?
    [[ $status == 1 ]] && grep -q 'check failed' "$TEST_DIR/precision-invalid.log"
done
"$CXXGPT" train "${COMMON[@]}" --steps 0 --no-tf32 --out "$TEST_DIR/legacy-precision.ckpt" > "$TEST_DIR/legacy-precision.log"
status=0
"$CXXGPT_TRAINING_CLI" same-sampler "$TEST_DIR/model.ckpt" "$TEST_DIR/fresh.ckpt" > "$TEST_DIR/rng-different.log" 2>&1 || status=$?
[[ "$status" == 1 ]] && grep -q 'different samples' "$TEST_DIR/rng-different.log"
cp "$TEST_DIR/model.ckpt" "$TEST_DIR/protected.ckpt"
"$CXXGPT_TRAINING_CLI" recover-save "$TEST_DIR/protected.ckpt" > "$TEST_DIR/recover-save.log" 2>&1
cmp "$TEST_DIR/model.ckpt" "$TEST_DIR/protected.ckpt"
# Rename failures during periodic saves must not abort the three updates.
mkdir "$TEST_DIR/failing.ckpt"
printf 'keep' > "$TEST_DIR/failing.ckpt/marker"
status=0
"$CXXGPT" train "${COMMON[@]}" --steps 3 --save-every 1 --log-every 1 --eval-every 0 \
    --out "$TEST_DIR/failing.ckpt" > "$TEST_DIR/failing-save.log" 2>&1 || status=$?
[[ "$status" == 1 ]]  # final failure is still a checked error
[[ $(grep -c 'periodic checkpoint failed' "$TEST_DIR/failing-save.log") == 3 ]]
grep -q '^step *3 |' "$TEST_DIR/failing-save.log"
[[ $(cat "$TEST_DIR/failing.ckpt/marker") == keep ]]
[[ ! -e "$TEST_DIR/failing.ckpt.tmp" ]]
"$CXXGPT" train "${COMMON[@]}" --resume --init "$TEST_DIR/does-not-exist.ckpt" > "$TEST_DIR/resume.log"
cmp "$TEST_DIR/model.ckpt" "$TEST_DIR/protected.ckpt"
# A skipped seek must still detect a truncated moment array in weight-only loads.
cp "$TEST_DIR/model.ckpt" "$TEST_DIR/truncated.ckpt"
"$CXXGPT_TRAINING_CLI" truncate "$TEST_DIR/truncated.ckpt"
status=0
"$CXXGPT" train "${COMMON[@]}" --steps 0 --init "$TEST_DIR/truncated.ckpt" > "$TEST_DIR/truncated.log" 2>&1 || status=$?
[[ "$status" == 1 ]] && grep -q 'truncated optimizer state' "$TEST_DIR/truncated.log"
for kind in params m v short-v; do
    status=0
    "$CXXGPT_TRAINING_CLI" bad-save "$TEST_DIR/protected.ckpt" "$kind" > "$TEST_DIR/bad-save.log" 2>&1 || status=$?
    [[ "$status" == 1 ]] && grep -q 'check failed' "$TEST_DIR/bad-save.log"
    cmp "$TEST_DIR/model.ckpt" "$TEST_DIR/protected.ckpt"
done
status=0
"$CXXGPT" train "${COMMON[@]}" --init "$TEST_DIR/unstable.ckpt" --out "$TEST_DIR/protected.ckpt" \
    --steps 20 --save-every 1 > "$TEST_DIR/diverged.log" 2>&1 || status=$?
[[ "$status" == 1 ]] && grep -q '10 diverged steps' "$TEST_DIR/diverged.log"
[[ $(grep -c 'step 1: .*update skipped' "$TEST_DIR/diverged.log") == 10 ]]
cmp "$TEST_DIR/model.ckpt" "$TEST_DIR/protected.ckpt"
printf 'CLI: warm start/reset, resume precedence, truncated moments and checkpoint protection passed\n'

"$CXXGPT_AUDIT" --ckpt "$TEST_DIR/audit.ckpt" --problems "$TEST_DIR/audit.txt" --threads 2 > "$TEST_DIR/audit.csv"
awk -F, '$2=="all_completion" {if ($3!=13 || $4<=0) exit 1; all=$4*13}
    $2=="step_digit" || $2=="step_other" || $2=="delimiter" || $2=="answer_digit" || $2=="answer_other" {sum+=$3*$4; n+=$3}
    $2=="teacher_coverage" {if ($5!=1 || $6!=1) exit 1}
    END {if (n!=13 || (sum-all)>0.000001 || (all-sum)>0.000001) exit 1}' "$TEST_DIR/audit.csv"
printf 'CLI: reasoning audit covers the gold completion and preserves weighted segment loss\n'

"$CXXGPT_TARGETS" --train "$TEST_DIR/targets.tok" --val "$TEST_DIR/targets.tok" \
    --exclude-test "$TEST_DIR/audit.txt" --out "$TEST_DIR/targets" --per-cell 3 > "$TEST_DIR/targets.log"
"$CXXGPT_TRAINING_CLI" targets "$TEST_DIR/targets.tok" "$TEST_DIR/targets/train-final.tok" "$TEST_DIR/targets/probe.txt"
printf 'CLI: final-digit masks preserve tokens and fresh probes exclude known questions\n'
"$CXXGPT_CURRICULUM" --out "$TEST_DIR/curriculum" --exclude-corpora "$TEST_DIR/targets.tok" \
    --exclude-problems "$TEST_DIR/audit.txt" --train 24 --per-cell 2 --val-per-cell 2 > "$TEST_DIR/curriculum.log"
"$CXXGPT_TRAINING_CLI" curriculum "$TEST_DIR/curriculum"
"$CXXGPT_CURRICULUM" --mode budget --data "$TEST_DIR/curriculum/control.tok" --seq 8 --steps 4 > "$TEST_DIR/budget.csv"
grep -q '^4,64,' "$TEST_DIR/budget.csv"
"$CXXGPT_TRACES" --ckpt "$TEST_DIR/audit.ckpt" --problems "$TEST_DIR/audit.txt" \
    --transcripts "$TEST_DIR/trace.tsv" > "$TEST_DIR/trace.csv"
grep -q 'overall,0,1,' "$TEST_DIR/trace.csv"
[[ $(wc -l < "$TEST_DIR/trace.tsv") == 2 ]]
printf 'CLI: paired curriculum excludes known questions; sampled budget and generated trace audit passed\n'
