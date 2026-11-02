#!/usr/bin/env bash
# Score all five mode-2 adapters (specialists + joint) on the 20% eval holdout,
# the same way mode 1 was scored, so the pooled nDCG@10/recall@10 line up.
# Sequential on GPU 2; stops on first failure.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/mnt/Data/unsloth/studio/unsloth_studio/bin/python
NAMES=("$@")
[ ${#NAMES[@]} -eq 0 ] && NAMES=(fiqa arguana scifact gaming joint)
MASTER=results/eval_all.log
: > "$MASTER"
for name in "${NAMES[@]}"; do
  echo "=== $(date '+%Y-%m-%dT%H:%M:%S')  START $name ===" | tee -a "$MASTER"
  set +e
  CUDA_VISIBLE_DEVICES=2 "$PY" scripts/eval_mode2.py --name "$name" \
    > "results/eval_${name}.log" 2>&1
  rc=$?
  set -e
  echo "=== $(date '+%Y-%m-%dT%H:%M:%S')  DONE  $name  exit=$rc ===" | tee -a "$MASTER"
  [ "$rc" -ne 0 ] && { echo "FAILED $name"; tail -30 "results/eval_${name}.log" | tee -a "$MASTER"; exit "$rc"; }
done
echo "=== $(date '+%Y-%m-%dT%H:%M:%S')  ALL DONE ===" | tee -a "$MASTER"