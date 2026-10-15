#!/usr/bin/env bash
# Exercise the real CLI after config parsing and backend selection.
set -euo pipefail
CXXGPT=${CXXGPT:-./cxxgpt}
CXXGPT_TRAINING_CLI=${CXXGPT_TRAINING_CLI:-./test_training_cli}
CXXGPT_AUDIT=${CXXGPT_AUDIT:-./reasoning_audit}
CXXGPT_TARGETS=${CXXGPT_TARGETS:-./reasoning_targets}
CXXGPT_TRACES=${CXXGPT_TRACES:-./reasoning_traces}
CXXGPT_CURRICULUM=${CXXGPT_CURRICULUM:-./reasoning_curriculum}
# A negated command is exempt from set -e. Require grep's specific no-match
# status instead, so a match or a read error fails the surrounding test.
require_absent() {
    local result=0
    grep "$@" || result=$?
    [[ "$result" == 1 ]]
}
cd "$(dirname "$0")/.."
TEST_DIR=$(mktemp -d)
trap 'rm -rf -- "$TEST_DIR"' EXIT
"$CXXGPT_TRAINING_CLI" clone-fixture "$TEST_DIR"
for kind in adam muon weights; do
    for mode in explicit auto; do
        CLONE_OPTIONS=(--layer 0 --from 0 --to 2)
        if [[ "$mode" == auto ]]; then CLONE_OPTIONS=(--auto); fi
        "$CXXGPT" clone-experts --ckpt "$TEST_DIR/clone-$kind.ckpt" --out "$TEST_DIR/repaired.ckpt" \
            "${CLONE_OPTIONS[@]}" --noise 0.01 --seed 31 > "$TEST_DIR/clone.log"
        "$CXXGPT_TRAINING_CLI" clone-reference "$TEST_DIR/clone-$kind.ckpt" "$TEST_DIR/repaired.ckpt" "$mode" 0.01
    done
done
"$CXXGPT" clone-experts --ckpt "$TEST_DIR/clone-adam.ckpt" --out "$TEST_DIR/exact.ckpt" \
    --layer 0 --from 0 --to 2 --noise 0 --seed 31 > "$TEST_DIR/clone.log"
"$CXXGPT_TRAINING_CLI" clone-reference "$TEST_DIR/clone-adam.ckpt" "$TEST_DIR/exact.ckpt" explicit 0
for options in '--auto --layer 0' '--layer 0 --from 0' '--layer -1 --from 0 --to 2' \
               '--layer 2 --from 0 --to 2' '--layer 0 --from 0 --to 0' '--layer 0 --from 4 --to 2' \
               '--auto --noise -1' '--auto --noise nan' '--auto --seed -1'; do
    status=0
    # Fixed literal cases, no user-supplied option splitting.
    # shellcheck disable=SC2086
    "$CXXGPT" clone-experts --ckpt "$TEST_DIR/clone-adam.ckpt" --out "$TEST_DIR/exact.ckpt" \
        $options > "$TEST_DIR/clone-invalid.log" 2>&1 || status=$?
    [[ "$status" == 1 ]]
    grep -q 'check failed' "$TEST_DIR/clone-invalid.log"
    "$CXXGPT_TRAINING_CLI" clone-reference "$TEST_DIR/clone-adam.ckpt" "$TEST_DIR/exact.ckpt" explicit 0
done
printf 'CLI: explicit/automatic cloning preserves unrelated checkpoint state, optimizer kind and RNG\n'
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
    [[ "$status" == 1 ]]
    grep -q 'threads must be positive' "$TEST_DIR/thread-bad.log"
done
for options in '--dim 4294967304' '--steps 9999999999999999999999999' '--lr nan' '--lr inf' '--lr 1e999' '--lr 0.1junk'; do
    status=0
    # Fixed cases; all must fail in parsing before construction/allocation.
    "$CXXGPT" train "${COMMON[@]}" $options > "$TEST_DIR/strict-args.log" 2>&1 || status=$?
    [[ "$status" == 1 ]]
    grep -q 'check failed' "$TEST_DIR/strict-args.log"
done
status=0
"$CXXGPT" train "${COMMON[@]}" --experts 4 --topk 1 > "$TEST_DIR/top1.log" 2>&1 || status=$?
[[ "$status" == 1 ]]
grep -q 'one-logit softmax has zero task gradient' "$TEST_DIR/top1.log"

"$CXXGPT_TRAINING_CLI" generate "$TEST_DIR"
# Read the real CPU backend's interval log against independently sorted router
# scores. Evaluation every step must not split the two-step logging interval.
"$CXXGPT" train "${COMMON[@]}" --init "$TEST_DIR/load.ckpt" --steps 3 --accum 2 --lr 0 --bias-rate 0 \
    --log-every 2 --load-log "$TEST_DIR/load.csv" --out "$TEST_DIR/load-trained.ckpt" > "$TEST_DIR/load.log"
"$CXXGPT_TRAINING_CLI" load-log "$TEST_DIR/load.ckpt" "$TEST_DIR/data.tok" "$TEST_DIR/load.csv"
# One warning line per logging interval (steps 2 and 3), naming the dead expert.
[[ $(grep -c '^under 1%: .*layer 1 expert 3 (0\.0%)' "$TEST_DIR/load.log") == 2 ]]
# Without --load-log the same model prints no warnings.
"$CXXGPT" train "${COMMON[@]}" --init "$TEST_DIR/load.ckpt" --steps 2 --lr 0 --bias-rate 0 \
    --log-every 2 --out "$TEST_DIR/load-quiet.ckpt" > "$TEST_DIR/load-quiet.log"
grep -q '| load ' "$TEST_DIR/load-quiet.log"
require_absent -q 'under 1%' "$TEST_DIR/load-quiet.log"
"$CXXGPT" train "${COMMON[@]}" --resume --steps 4 --accum 2 --lr 0 --bias-rate 0 --log-every 2 \
    --load-log "$TEST_DIR/load.csv" --out "$TEST_DIR/load-trained.ckpt" > "$TEST_DIR/load-resume.log"
[[ $(grep -c '^step,layer,' "$TEST_DIR/load.csv") == 1 ]]
grep -q '^# resumed at step 3$' "$TEST_DIR/load.csv"
[[ $(grep -c '^4,' "$TEST_DIR/load.csv") == 2 ]]
printf 'wrong,header\n' > "$TEST_DIR/load-invalid.csv"
status=0
"$CXXGPT" train "${COMMON[@]}" --init "$TEST_DIR/load.ckpt" --load-log "$TEST_DIR/load-invalid.csv" \
    --out "$TEST_DIR/load-invalid.ckpt" > "$TEST_DIR/load-invalid.log" 2>&1 || status=$?
[[ "$status" == 1 ]]
grep -q 'expert load log header differs' "$TEST_DIR/load-invalid.log"
# A dense model has no experts, so --load-log writes no file.
"$CXXGPT" train "${COMMON[@]}" --load-log "$TEST_DIR/dense.load.csv" > "$TEST_DIR/dense-load.log"
[[ ! -e "$TEST_DIR/dense.load.csv" ]]

# Run the unchanged report script in a disposable runs directory. The CSV's
# %.6g writes tiny shares in exponent form (5e-05); awk must still read them.
mkdir -p "$TEST_DIR/report/scripts" "$TEST_DIR/report/runs"
cp scripts/report.sh "$TEST_DIR/report/scripts/report.sh"
printf 'model: tiny fixture\nstep 40 | loss 1.0 | 100 tok/s\n' > "$TEST_DIR/report/runs/fixture.log"
cat > "$TEST_DIR/report/runs/fixture.load.csv" <<'CSV'
step,layer,expert_0,expert_1,max_load_ratio
10,0,0.5,0.5,1
10,1,0,1,2
10,2,0.01,0.99,1.98
20,0,5e-05,0.99995,1.9999
20,1,0.2,0.8,1.6
20,2,0.5,0.5,1
# resumed at step 20
40,0,0.4,0.6,1.2
40,1,0.3,0.7,1.4
40,2,0.6,0.4,1.2
CSV
bash "$TEST_DIR/report/scripts/report.sh" fixture > "$TEST_DIR/report.txt"
grep -Fq 'layer 0: final shares [0.4000 0.6000]; first under 1%: step 20' "$TEST_DIR/report.txt"
grep -Fq 'layer 1: final shares [0.3000 0.7000]; first under 1%: step 10' "$TEST_DIR/report.txt"
grep -Fq 'layer 2: final shares [0.6000 0.4000]; first under 1%: never' "$TEST_DIR/report.txt"
printf 'CLI: expert load intervals, dead-expert warnings, resume and report passed\n'
"$CXXGPT" train "${COMMON[@]}" --data "$TEST_DIR/data.tok*3" --val '' --val-frac 0.5 --steps 0 \
    --out "$TEST_DIR/repeated-val.ckpt" > "$TEST_DIR/repeated-val.log" 2>&1
grep -q 'validation slices repeated data' "$TEST_DIR/repeated-val.log"
"$CXXGPT" train "${COMMON[@]}" --data "$TEST_DIR/data.tok*3" --steps 0 \
    --out "$TEST_DIR/explicit-val.ckpt" > "$TEST_DIR/explicit-val.log" 2>&1
require_absent -q 'validation slices repeated data' "$TEST_DIR/explicit-val.log"
"$CXXGPT" train "${COMMON[@]}" --steps 2 --seed 7 --eval-every 0 --out "$TEST_DIR/budget-proof.ckpt" \
    > "$TEST_DIR/budget-proof.log"
"$CXXGPT_CURRICULUM" --mode budget --data "$TEST_DIR/data.tok" --ckpt "$TEST_DIR/budget-proof.ckpt" \
    --log "$TEST_DIR/budget-proof.log" --seq 8 --batch 1 --steps 2 --seed 7 > "$TEST_DIR/budget-proof.csv"
# A masked tail split has one predictable target per full training window;
# using the whole corpus would sample a different range with identical draw counts.
"$CXXGPT" train "${COMMON[@]}" --data "$TEST_DIR/masked.tok" --val '' --val-frac 0.3 \
    --steps 2 --seed 7 --eval-every 0 --out "$TEST_DIR/range-proof.ckpt" > "$TEST_DIR/range-proof.log"
"$CXXGPT_CURRICULUM" --mode budget --data "$TEST_DIR/masked.tok" --ckpt "$TEST_DIR/range-proof.ckpt" \
    --log "$TEST_DIR/range-proof.log" --seq 8 --batch 1 --steps 2 --seed 7 > "$TEST_DIR/range-proof.csv"
grep -q '^2,16,2,1$' "$TEST_DIR/range-proof.csv"
for options in '--seq 4' '--batch 2' '--steps 1'; do
    status=0
    "$CXXGPT_CURRICULUM" --mode budget --data "$TEST_DIR/data.tok" --ckpt "$TEST_DIR/budget-proof.ckpt" \
        --log "$TEST_DIR/budget-proof.log" --seq 8 --batch 1 --steps 2 --seed 7 $options \
        > "$TEST_DIR/budget-mismatch.log" 2>&1 || status=$?
    [[ "$status" == 1 ]]
    grep -q 'sampling shape\|budget mismatch' "$TEST_DIR/budget-mismatch.log"
done
"$CXXGPT" train "${COMMON[@]}" --steps 10000000 --log-every 10000000 --eval-every 0 --save-every 1 \
    --out "$TEST_DIR/buffer.ckpt" > "$TEST_DIR/buffer.log" 2>&1 &
trainer=$!
for ((poll=0; poll<1000; ++poll)); do
    [[ -f "$TEST_DIR/buffer.ckpt" ]] && break
    if ! kill -0 "$trainer" 2>/dev/null; then wait "$trainer"; exit 1; fi
    sleep 0.01
done
seen_header=0
if grep -q '^params:' "$TEST_DIR/buffer.log"; then seen_header=1; fi
kill -INT "$trainer"
wait "$trainer"
[[ "$seen_header" == 1 ]]
grep -q 'saved .* (interrupted)' "$TEST_DIR/buffer.log"
status=0
"$CXXGPT" text2tok --in "$TEST_DIR" --out "$TEST_DIR/read-error.tok" --threads 2 > "$TEST_DIR/read-error.log" 2>&1 || status=$?
[[ "$status" == 1 ]]
grep -q 'error reading' "$TEST_DIR/read-error.log"
status=0
"$CXXGPT_TRAINING_CLI" bad-token-save > "$TEST_DIR/token-save.log" 2>&1 || status=$?
[[ "$status" == 1 ]]
grep -q 'error writing /dev/full' "$TEST_DIR/token-save.log"
printf 'a\001b' > "$TEST_DIR/bos.txt"
"$CXXGPT" text2tok --in "$TEST_DIR/bos.txt" --out "$TEST_DIR/bos.tok" --threads 2 > "$TEST_DIR/bos.log"
printf '%s\n' '{"question":"one\u0001two","answer":"work\u0001\n#### 2\u0001"}' > "$TEST_DIR/bos.jsonl"
"$CXXGPT" prep-qa --in "$TEST_DIR/bos.jsonl" --out "$TEST_DIR/bos-qa" > "$TEST_DIR/bos-qa.log"
"$CXXGPT_TRAINING_CLI" clean-bos "$TEST_DIR/bos.tok" "$TEST_DIR/bos-qa.tok"
printf '%s\n' '{"question":"q","answer":"work","final":9007199254740993}' 'broken' \
    '{"question":"q2","answer":"work","final":1.20e3}' \
    '{"question":"q3","answer":"work","final":1e999}' \
    '{"question":"q4","answer":"work","final":-0.00120}' \
    '{"question":"missing answer","final":1}' \
    '{"question":"missing final","answer":"work"}' > "$TEST_DIR/numeric.jsonl"
"$CXXGPT" prep-qa --in "$TEST_DIR/numeric.jsonl" --out "$TEST_DIR/numeric" --final-key final > "$TEST_DIR/numeric.log" 2>&1
grep -q '3 problems (4 without a usable answer skipped)' "$TEST_DIR/numeric.log"
"$CXXGPT_TRAINING_CLI" numeric-labels "$TEST_DIR/numeric.txt"
awk -F '\t' 'NR==1 && $5!="9007199254740993" {exit 1} NR==2 && $5!="1200" {exit 1} NR==3 && $5!="-0.0012" {exit 1} END {if(NR!=3)exit 1}' "$TEST_DIR/numeric.txt"
# Cross the streaming block size without a newline; threaded cuts preserve bytes.
python3 - "$TEST_DIR/one-line.txt" <<'PYLINE'
import pathlib,sys
pathlib.Path(sys.argv[1]).write_text(' alpha beta gamma '*500000)
PYLINE
"$CXXGPT" text2tok --in "$TEST_DIR/one-line.txt" --out "$TEST_DIR/one-line-one.tok" --threads 1 > "$TEST_DIR/one-line-one.log"
"$CXXGPT" text2tok --in "$TEST_DIR/one-line.txt" --out "$TEST_DIR/one-line-two.tok" --threads 2 > "$TEST_DIR/one-line-two.log"
cmp "$TEST_DIR/one-line-one.tok" "$TEST_DIR/one-line-two.tok"
# A byte tokenizer alone cannot expose a cut through a learned merge.
"$CXXGPT" bpe-train --in "$TEST_DIR/one-line.txt" --max-bytes 4096 --vocab 300 \
    --out "$TEST_DIR/stream.bpe" > "$TEST_DIR/stream-bpe.log"
for threads in 1 2; do
    "$CXXGPT" text2tok --in "$TEST_DIR/one-line.txt" --bpe "$TEST_DIR/stream.bpe" \
        --out "$TEST_DIR/stream-$threads.tok" --threads "$threads" > "$TEST_DIR/stream-$threads.log"
    "$CXXGPT_TRAINING_CLI" bpe-stream-reference "$TEST_DIR/stream.bpe" "$TEST_DIR/one-line.txt" "$TEST_DIR/stream-$threads.tok"
done
cmp "$TEST_DIR/stream-1.tok" "$TEST_DIR/stream-2.tok"

python3 - "$TEST_DIR/long.txt" <<'PYLONG'
import pathlib,sys
pathlib.Path(sys.argv[1]).write_text('a'*65537)
PYLONG
# An over-long chunk is reported, not fatal: a long data run keeps going.
"$CXXGPT" text2tok --in "$TEST_DIR/long.txt" --out "$TEST_DIR/chunk-out.tok" --threads 2 > "$TEST_DIR/chunk.log"
grep -q 'warning: 1 chunks longer than 65536 bytes were written as plain bytes' "$TEST_DIR/chunk.log"
"$CXXGPT" bpe-train --in "$TEST_DIR/long.txt" --out "$TEST_DIR/chunk-out.bpe" --vocab 300 > "$TEST_DIR/chunk.log"
grep -q 'warning: 1 chunks longer than 65536 bytes were left out of the merges' "$TEST_DIR/chunk.log"
status=0
"$CXXGPT_TRAINING_CLI" bad-tokenizer-vocab > "$TEST_DIR/core-vocab.log" 2>&1 || status=$?
[[ "$status" == 1 ]]
grep -q 'uint16 token ids' "$TEST_DIR/core-vocab.log"
printf 'cxxgpt-bpe 1\n65281\n' > "$TEST_DIR/oversized.bpe"
status=0
"$CXXGPT" text2tok --in "$TEST_DIR/text.txt" --out "$TEST_DIR/oversized.tok" --bpe "$TEST_DIR/oversized.bpe" \
    > "$TEST_DIR/oversized.log" 2>&1 || status=$?
[[ "$status" == 1 ]]
grep -q 'too many merges' "$TEST_DIR/oversized.log"
status=0
"$CXXGPT" bpe-train --in "$TEST_DIR/text.txt" --out "$TEST_DIR/too-large.bpe" --vocab 65537 \
    > "$TEST_DIR/too-large.log" 2>&1 || status=$?
[[ "$status" == 1 ]]
grep -q 'uint16' "$TEST_DIR/too-large.log"
python3 tests/test_mistakes.py

for options in '--min-diff 4 --max-diff 2' '--max-diff 10' '--test-max-diff 17' '--test-per-cell 0'; do
    status=0
    "$CXXGPT" gen-reasoning --out "$TEST_DIR/invalid-reason" $options --train 0 --val 0 \
        > "$TEST_DIR/invalid-reason.log" 2>&1 || status=$?
    [[ "$status" == 1 ]]
    grep -q 'check failed' "$TEST_DIR/invalid-reason.log"
done
status=0
"$CXXGPT" gen-reasoning --out "$TEST_DIR/exhausted" --tasks mul --min-diff 1 --max-diff 1 \
    --test-max-diff 1 --test-per-cell 721 --train 0 --val 0 > "$TEST_DIR/exhausted.log" 2>&1 || status=$?
[[ "$status" == 1 ]]
grep -q 'question domain exhausted' "$TEST_DIR/exhausted.log"
# Calculator evaluation always retains the unassisted baseline. A one-token
# budget cannot complete a tool call, so both passes must have the same score.
for tool in calc-notes calc-plain; do
    "$CXXGPT" eval-reasoning --ckpt "$TEST_DIR/audit.ckpt" --test "$TEST_DIR/audit.txt" \
        --tools "$tool" --max-new 1 --threads 2 > "$TEST_DIR/calc-eval.log"
    grep -q '^without tool:$' "$TEST_DIR/calc-eval.log"
    grep -q '^with calculator:$' "$TEST_DIR/calc-eval.log"
    [[ $(grep -c '^overall ' "$TEST_DIR/calc-eval.log") == 2 ]]
    [[ $(grep '^overall ' "$TEST_DIR/calc-eval.log" | sort -u | wc -l) == 1 ]]
    grep -q '^calculator calls: 0 fired, 0 malformed$' "$TEST_DIR/calc-eval.log"
    "$CXXGPT" sample --ckpt "$TEST_DIR/audit.ckpt" --tools "$tool" --n 1 --temp 0 --threads 2 \
        > "$TEST_DIR/calc-sample.txt" 2> "$TEST_DIR/calc-sample.log"
    grep -q '^calculator calls: 0 fired, 0 malformed$' "$TEST_DIR/calc-sample.log"
done
for command in sample eval-reasoning; do
    "$CXXGPT" "$command" --help > "$TEST_DIR/calc-help.log"
    grep -q -- '--tools' "$TEST_DIR/calc-help.log"
    status=0
    "$CXXGPT" "$command" --tools typo > "$TEST_DIR/calc-invalid.log" 2>&1 || status=$?
    [[ "$status" == 1 ]]
    grep -q -- '--tools must be none, calc-notes or calc-plain' "$TEST_DIR/calc-invalid.log"
done
printf 'CLI: calculator modes, paired scores, counts and strict options passed\n'
for batch in 0 -1; do
    status=0
    "$CXXGPT" eval-reasoning --ckpt "$TEST_DIR/audit.ckpt" --test "$TEST_DIR/audit.txt" --batch "$batch" \
        --threads 2 > "$TEST_DIR/eval-batch.log" 2>&1 || status=$?
    [[ "$status" == 1 ]]
    grep -q 'evaluation batch and max_new must be positive' "$TEST_DIR/eval-batch.log"
done
for count in '' 1junk 99999999999999999999999 0 -1; do
    status=0
    "$CXXGPT" train "${COMMON[@]}" --data "$TEST_DIR/data.tok*$count" > "$TEST_DIR/repeat-count.log" 2>&1 || status=$?
    [[ "$status" == 1 ]]
    grep -q 'bad repeat count' "$TEST_DIR/repeat-count.log"
done
status=0
"$CXXGPT" eval-reasoning --ckpt "$TEST_DIR/audit.ckpt" --test "$TEST_DIR/audit.txt" --as-engine \
    --threads 2 > "$TEST_DIR/engine-config.log" 2>&1 || status=$?
[[ "$status" == 1 ]]
grep -q 'requires a ternary model' "$TEST_DIR/engine-config.log"
for target in 1 -1; do
    status=0
    "$CXXGPT_TRAINING_CLI" repeat-backward "$target" > "$TEST_DIR/repeat-backward.log" 2>&1 || status=$?
    [[ "$status" == 1 ]]
    grep -q 'needs a new forward' "$TEST_DIR/repeat-backward.log"
done
for command in bad-target bad-target-backward; do
  for target in -2 3; do
    status=0
    "$CXXGPT_TRAINING_CLI" "$command" "$target" > "$TEST_DIR/target-range.log" 2>&1 || status=$?
    [[ "$status" == 1 ]]
    grep -q 'target .* out of range' "$TEST_DIR/target-range.log"
  done
done
# gen-word uses the existing problem reader and text2tok unchanged.
for split in train test; do
    prefix="$TEST_DIR/word-$split"
    "$CXXGPT" gen-word --out "$prefix" --seed 73 --count 28 --split "$split" > "$prefix.log"
    "$CXXGPT" gen-word --out "$prefix-repeat" --seed 73 --count 28 --split "$split" > /dev/null
    cmp "$prefix.txt" "$prefix-repeat.txt"
    cmp "$prefix.text" "$prefix-repeat.text"
    "$CXXGPT" text2tok --in "$prefix.text" --out "$prefix.tok" --doc-sep '<|endoftext|>' --threads 2 > /dev/null
    "$CXXGPT_TRAINING_CLI" word "$prefix"
    "$CXXGPT" eval-reasoning --ckpt "$TEST_DIR/audit.ckpt" --test "$prefix.txt" --max-new 1 --threads 2 > "$prefix-eval.log"
    grep -q '^overall .* / 28)' "$prefix-eval.log"
done
"$CXXGPT" gen-word --out "$TEST_DIR/plain-word" --count 4 --min-steps 8 --max-steps 8 \
    --notes off --distractors off > /dev/null
require_absent -q '<<' "$TEST_DIR/plain-word.text"
awk -F '\t' 'NF != 5 || $2 != 8 {exit 1} END {if (NR != 4) exit 1}' "$TEST_DIR/plain-word.txt"
"$CXXGPT" gen-word --out "$TEST_DIR/empty-word" --count 0 > /dev/null
[[ ! -s "$TEST_DIR/empty-word.txt" && ! -s "$TEST_DIR/empty-word.text" ]]
for pair in 'min-steps 1' 'max-steps 9' 'min-steps 9' 'max-steps 2147483649' 'count -1' 'seed -1' \
            'split dev' 'notes maybe' 'distractors maybe'; do
    read -r option value <<< "$pair"
    status=0
    "$CXXGPT" gen-word --out "$TEST_DIR/invalid-word" "--$option" "$value" > "$TEST_DIR/invalid-word.log" 2>&1 || status=$?
    [[ "$status" == 1 && ! -e "$TEST_DIR/invalid-word.txt" ]]
    grep -q 'check failed' "$TEST_DIR/invalid-word.log"
done
printf 'CLI: word generation, repeatability, text2tok, eval-reasoning and invalid options passed\n'

printf 'qa\t1\tq\twork\t14\r\nqa\t2\ttail\twork\t1080' > "$TEST_DIR/line-endings.txt"
"$CXXGPT_TRAINING_CLI" problems "$TEST_DIR/line-endings.txt"
"$CXXGPT" eval-reasoning --ckpt "$TEST_DIR/audit.ckpt" --test "$TEST_DIR/line-endings.txt" --max-new 1 --threads 2 > "$TEST_DIR/line-endings.log"
grep -q '^overall .* / 2)' "$TEST_DIR/line-endings.log"
for difficulty in '' 1junk 999999999999999999999999 -1; do
    printf 'qa\t%s\tq\twork\t14' "$difficulty" > "$TEST_DIR/bad-difficulty.txt"
    status=0
    "$CXXGPT" eval-reasoning --ckpt "$TEST_DIR/audit.ckpt" --test "$TEST_DIR/bad-difficulty.txt" --threads 2 \
        > "$TEST_DIR/bad-difficulty.log" 2>&1 || status=$?
    [[ "$status" == 1 ]]
    grep -q 'bad difficulty' "$TEST_DIR/bad-difficulty.log"
done
"$CXXGPT" eval-reasoning --ckpt "$TEST_DIR/audit.ckpt" --test "$TEST_DIR/audit.txt" --max-new 1 --threads 2 > "$TEST_DIR/cap.log"
grep -q 'gold completions exceeding --max-new 1: 1 / 1' "$TEST_DIR/cap.log"
"$CXXGPT" eval-reasoning --ckpt "$TEST_DIR/audit.ckpt" --test "$TEST_DIR/audit.txt" --threads 2 > "$TEST_DIR/cap-default.log"
grep -q 'gold completions exceeding --max-new 200: 0 / 1' "$TEST_DIR/cap-default.log"
"$CXXGPT" eval-reasoning --ckpt "$TEST_DIR/audit.ckpt" --test "$TEST_DIR/audit.txt" \
    --max-new 2147483647 --threads 2 > "$TEST_DIR/cap-largest.log"
grep -q 'gold completions exceeding --max-new 2147483647: 0 / 1' "$TEST_DIR/cap-largest.log"

"$CXXGPT" train "${COMMON[@]}" --init "$TEST_DIR/audit.ckpt" --out "$TEST_DIR/cap.ckpt" \
    --reason-test "$TEST_DIR/audit.txt" --max-new 1 > "$TEST_DIR/train-cap.log"
grep -q 'gold completions exceeding --max-new 1: 1 / 1' "$TEST_DIR/train-cap.log"
status=0
"$CXXGPT" eval-reasoning --ckpt "$TEST_DIR/audit.ckpt" --test "$TEST_DIR/audit.txt" --max-new 0 > "$TEST_DIR/bad-cap.log" 2>&1 || status=$?
[[ "$status" == 1 ]]
grep -q 'max-new must be positive' "$TEST_DIR/bad-cap.log"
# --gpu needs the CUDA build (cxxgpt-gpu); this one says so. --matmul
# without --gpu is refused too.
status=0
"$CXXGPT" eval-reasoning --ckpt "$TEST_DIR/audit.ckpt" --test "$TEST_DIR/audit.txt" --gpu > "$TEST_DIR/no-gpu.log" 2>&1 || status=$?
[[ "$status" == 1 ]]
grep -q 'built without CUDA; use cxxgpt-gpu' "$TEST_DIR/no-gpu.log"
status=0
"$CXXGPT" eval-reasoning --ckpt "$TEST_DIR/audit.ckpt" --test "$TEST_DIR/audit.txt" --matmul tf32 > "$TEST_DIR/no-gpu.log" 2>&1 || status=$?
[[ "$status" == 1 ]]
grep -q 'matmul needs --gpu' "$TEST_DIR/no-gpu.log"

# The vote: --tries N with --temp samples each problem N times and reports
# first-sampled-try, vote and any-try accuracies; --effort is a shorthand for N.
# '--tries 2' without --temp, non-positive tries and an unknown effort all
# fail with a checked error.
for options in '--tries 2' '--tries 0 --temp 1.0' '--tries 1junk --temp 1.0' '--effort huge'; do
    status=0
    "$CXXGPT" eval-reasoning --ckpt "$TEST_DIR/audit.ckpt" --test "$TEST_DIR/audit.txt" \
        $options --threads 2 > "$TEST_DIR/invalid-tries.log" 2>&1 || status=$?
    [[ "$status" == 1 ]]
    grep -q 'check failed' "$TEST_DIR/invalid-tries.log"
done
# One problem's tries are generated together, so they must fit in --batch.
status=0
"$CXXGPT" eval-reasoning --ckpt "$TEST_DIR/audit.ckpt" --test "$TEST_DIR/audit.txt" \
    --tries 4 --temp 1.0 --batch 2 --threads 2 > "$TEST_DIR/tries-batch.log" 2>&1 || status=$?
[[ "$status" == 1 ]]
grep -q -- '--tries 4 is more than --batch 2' "$TEST_DIR/tries-batch.log"
# The random model never finishes an answer, but --show-failures prints the
# sampled text, so the same seed must repeat it and another seed must not
# (the timing line left out).
vote_run() {  # vote_run SEED OUT: the report at that seed, without its timing line
    "$CXXGPT" eval-reasoning --ckpt "$TEST_DIR/audit.ckpt" --test "$TEST_DIR/audit.txt" --show-failures \
        --tries 4 --temp 0.9 --seed "$1" --max-new 6 --threads 2 > "$TEST_DIR/vote-full.log"
    grep -a -v '^([0-9.]* s)$' "$TEST_DIR/vote-full.log" > "$2"  # -a: the sampled bytes may look binary
}
vote_run 3 "$TEST_DIR/vote-a.log"
vote_run 3 "$TEST_DIR/vote-b.log"
vote_run 4 "$TEST_DIR/vote-c.log"
cmp "$TEST_DIR/vote-a.log" "$TEST_DIR/vote-b.log"
grep -a -q 'got:' "$TEST_DIR/vote-a.log"
status=0
cmp -s "$TEST_DIR/vote-a.log" "$TEST_DIR/vote-c.log" || status=$?
[[ "$status" == 1 ]]
grep -q '^effort: tries 4, temperature 0.9, seed 3$' "$TEST_DIR/vote-a.log"
grep -q 'first sampled try / vote maj@4 / any try pass@4 / mean distinct answers' "$TEST_DIR/vote-a.log"
# Every headline names its effort.
grep -q '^overall .* (maj@4, temp 0.9, seed 3)$' "$TEST_DIR/vote-a.log"
"$CXXGPT" eval-reasoning --ckpt "$TEST_DIR/audit.ckpt" --test "$TEST_DIR/audit.txt" \
    --temp 0.8 --max-new 6 --threads 2 > "$TEST_DIR/vote-one-sampled.log"
grep -q '^overall .* (1 sampled try, temp 0.8, seed 1)$' "$TEST_DIR/vote-one-sampled.log"
"$CXXGPT" eval-reasoning --ckpt "$TEST_DIR/audit.ckpt" --test "$TEST_DIR/audit.txt" \
    --effort medium --temp 0.5 --max-new 6 --threads 2 > "$TEST_DIR/vote-effort.log"
grep -q '^effort: tries 8, temperature 0.5, seed 1$' "$TEST_DIR/vote-effort.log"
"$CXXGPT" eval-reasoning --ckpt "$TEST_DIR/audit.ckpt" --test "$TEST_DIR/audit.txt" \
    --effort high --temp 0.5 --max-new 6 --threads 2 > "$TEST_DIR/vote-effort-high.log"
grep -q '^effort: tries 32, temperature 0.5, seed 1$' "$TEST_DIR/vote-effort-high.log"
"$CXXGPT" eval-reasoning --ckpt "$TEST_DIR/audit.ckpt" --test "$TEST_DIR/audit.txt" \
    --effort low --threads 2 > "$TEST_DIR/vote-effort-low.log"
require_absent -q 'effort:' "$TEST_DIR/vote-effort-low.log"
grep -q '^overall .* (greedy)$' "$TEST_DIR/vote-effort-low.log"
# A model that has memorized the one problem finishes its tries, so the
# vote has answers to count. Each document ends in "\n." because text2tok
# trims documents, and the answer line must end with its newline.
for i in $(seq 64); do printf 'Q: 1+1\nA: 1+1=2\n#### 2\n.<|endoftext|>'; done > "$TEST_DIR/memo.txt"
"$CXXGPT" text2tok --in "$TEST_DIR/memo.txt" --out "$TEST_DIR/memo.tok" --doc-sep '<|endoftext|>' \
    > "$TEST_DIR/memo-tok.log"
"$CXXGPT" train --data "$TEST_DIR/memo.tok" --val "$TEST_DIR/memo.tok" --seq 32 --dim 32 --heads 2 --ffn 64 \
    --layers 2 --batch 8 --steps 400 --lr 3e-3 --warmup 0 --eval-every 400 --eval-batches 1 --save-every 0 \
    --threads 2 --out "$TEST_DIR/memo.ckpt" > "$TEST_DIR/memo-train.log"
"$CXXGPT" eval-reasoning --ckpt "$TEST_DIR/memo.ckpt" --test "$TEST_DIR/audit.txt" \
    --tries 4 --temp 0.5 --seed 3 --max-new 20 --threads 2 > "$TEST_DIR/vote-memo.log"
grep -q '^overall 100.0% (1 / 1) (maj@4, temp 0.5, seed 3)$' "$TEST_DIR/vote-memo.log"
printf 'CLI: eval-reasoning tries/vote options passed\n'

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
    [[ "$status" == 1 ]]
    grep -q "$message" "$TEST_DIR/vocab-$kind.log"
    require_absent -q '^step ' "$TEST_DIR/vocab-$kind.log"
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
require_absent -q '^reasoning eval' "$TEST_DIR/signal.log"
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
require_absent -q 'diverged\|lower --lr' "$TEST_DIR/bad-timed.log"
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
[[ "$status" == 1 ]]
grep -q 'no predictable targets' "$TEST_DIR/ignored.log"
printf 'CLI: AdamW/Muon masked/sparse accumulation, weighted validation, empty-group skips, and fully masked rejection passed\n'

"$CXXGPT" bench-infer --vocab 13 --seq 8 --dim 32 --heads 2 --layers 1 --ffn 64 \
    --tokens 8 --reps 4 --warmup 1 --threads 2 --log "$TEST_DIR/decode.csv" > "$TEST_DIR/decode.log"
[[ $(wc -l < "$TEST_DIR/decode.csv") == 5 ]]
awk -F, 'NR>1 {if ($1 != NR-2 || $2 != (NR%2==0) || $5 <= 0 || $6 <= 0 || $7 < 0 || $7 > 1 || $10 != 1) exit 1}' "$TEST_DIR/decode.csv"
status=0
"$CXXGPT" bench-infer --vocab 13 --seq 8 --dim 32 --heads 2 --layers 1 --ffn 64 \
    --tokens 8 --reps 0 --threads 2 > "$TEST_DIR/bad-reps.log" 2>&1 || status=$?
[[ "$status" == 1 ]]
grep -q 'reps must be positive' "$TEST_DIR/bad-reps.log"
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
    [[ $status == 1 ]]
    grep -q 'check failed' "$TEST_DIR/precision-invalid.log"
done
"$CXXGPT" train "${COMMON[@]}" --steps 0 --no-tf32 --out "$TEST_DIR/legacy-precision.ckpt" > "$TEST_DIR/legacy-precision.log"
status=0
"$CXXGPT_TRAINING_CLI" same-sampler "$TEST_DIR/model.ckpt" "$TEST_DIR/fresh.ckpt" > "$TEST_DIR/rng-different.log" 2>&1 || status=$?
[[ "$status" == 1 ]]
grep -q 'different samples' "$TEST_DIR/rng-different.log"
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
[[ "$status" == 1 ]]
grep -q 'truncated optimizer state' "$TEST_DIR/truncated.log"
for kind in params m v short-v; do
    status=0
    "$CXXGPT_TRAINING_CLI" bad-save "$TEST_DIR/protected.ckpt" "$kind" > "$TEST_DIR/bad-save.log" 2>&1 || status=$?
    [[ "$status" == 1 ]]
    grep -q 'check failed' "$TEST_DIR/bad-save.log"
    cmp "$TEST_DIR/model.ckpt" "$TEST_DIR/protected.ckpt"
done
status=0
"$CXXGPT" train "${COMMON[@]}" --init "$TEST_DIR/unstable.ckpt" --out "$TEST_DIR/protected.ckpt" \
    --steps 20 --save-every 1 > "$TEST_DIR/diverged.log" 2>&1 || status=$?
[[ "$status" == 1 ]]
grep -q '10 diverged steps' "$TEST_DIR/diverged.log"
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

# Leak checker accepts native problem records and only indexes their questions.
printf 'gsm8k\t1\tOne two three four five six seven eight nine ten eleven twelve 13\tSolution words never count as question words in a training overlap search ever\t13\n' > "$TEST_DIR/leak-questions.txt"
printf 'ONE two three four five six\nseven eight nine ten eleven twelve 13\n' > "$TEST_DIR/leak-hit.txt"
printf 'One two three four five six seven eight nine ten eleven twelve\n' > "$TEST_DIR/leak-miss.txt"
printf 'Solution words never count as question words in a training overlap search ever\n' > "$TEST_DIR/leak-answer.txt"
# A stored line break (the two characters \n, as in prep-qa .txt files) separates words too.
printf 'ONE two three four five six\\nseven eight nine ten eleven twelve 13\n' > "$TEST_DIR/leak-escaped.txt"
"$CXXGPT" leak-check --questions "$TEST_DIR/leak-questions.txt" \
    --train "$TEST_DIR/leak-hit.txt,$TEST_DIR/leak-miss.txt,$TEST_DIR/leak-answer.txt,$TEST_DIR/leak-escaped.txt" > "$TEST_DIR/leak.log"
grep -q 'leak-hit.txt: 1 / 1' "$TEST_DIR/leak.log"
grep -q 'leak-miss.txt: 0 / 1' "$TEST_DIR/leak.log"
grep -q 'leak-answer.txt: 0 / 1' "$TEST_DIR/leak.log"
grep -q 'leak-escaped.txt: 1 / 1' "$TEST_DIR/leak.log"
grep -q 'question 1: One two' "$TEST_DIR/leak.log"
# The example is cut at 240 bytes without splitting a UTF-8 character: here
# bytes 239-240 are one two-byte character, so the cut moves back to 239.
printf 'One two three four five six seven eight nine ten eleven twelve 13 %s\xc3\xa9 end\n' \
    "$(head -c 173 /dev/zero | tr '\0' a)" > "$TEST_DIR/leak-utf8.txt"
"$CXXGPT" leak-check --questions "$TEST_DIR/leak-utf8.txt" --train "$TEST_DIR/leak-hit.txt" > "$TEST_DIR/leak-utf8.log"
grep -q 'aaaa\.\.\.$' "$TEST_DIR/leak-utf8.log"
if "$CXXGPT" leak-check --questions "$TEST_DIR/leak-utf8.txt" > /dev/null 2>&1; then
    echo "leak-check without --train must fail"; exit 1
fi
for dev in 0 1; do
    run="dev-check-${BASHPID}"
    MOCK_ARGS="$TEST_DIR/night.args" DEV="$dev" RUN="$run" TOKS=100 MIX=unused \
        CX="$TEST_DIR/night-mock" bash scripts/night.sh > "$TEST_DIR/night-script.log"
    split=test
    [[ "$dev" == 0 ]] || split=dev
    [[ $(awk '$0=="--val" {getline; print}' "$TEST_DIR/night.args") == "data/gsm8k/$split.tok" ]]
    [[ $(awk '$0=="--reason-test" {getline; print}' "$TEST_DIR/night.args") == "data/gsm8k/$split.txt" ]]
    rm "runs/$run.log"
done
printf 'CLI: leak-check scans questions across lines; DEV selects both validation files\n'
