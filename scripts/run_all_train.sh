#!/usr/bin/env bash
# Train all five mode-2 adapters sequentially on GPU 2.
#
# Order is specialists first, joint control last (if a specialist fails we
# stop before spending the long joint run). Each adapter is a fresh process:
# it reloads the base checkpoint and recompiles Triton kernels, so the first
# step of each is slower than the rest. Stops on the first non-zero exit so
# a broken config is not repeated five times.
#
# Usage:  bash scripts/run_all_train.sh        (all five)
#         bash scripts/run_all_train.sh joint  (just one, for re-runs)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/mnt/Data/unsloth/studio/unsloth_studio/bin/python
STEPS=200

if [ $# -ge 1 ]; then
  NAMES=("$@")
else
  NAMES=(fiqa arguana scifact gaming joint)
fi

MASTER=results/train_all.log
: > "$MASTER"
for name in "${NAMES[@]}"; do
  echo "=== $(date '+%Y-%m-%dT%H:%M:%S')  START $name (max_steps=$STEPS) ===" | tee -a "$MASTER"
  set +e
  CUDA_VISIBLE_DEVICES=2 "$PY" scripts/train_lora.py --name "$name" --max-steps "$STEPS" \
    > "results/train_${name}.log" 2>&1
  rc=$?
  set -e
  echo "=== $(date '+%Y-%m-%dT%H:%M:%S')  DONE  $name  exit=$rc ===" | tee -a "$MASTER"
  if [ "$rc" -ne 0 ]; then
    echo "FAILED on $name — stopping. Last 30 lines of results/train_${name}.log:" | tee -a "$MASTER"
    tail -30 "results/train_${name}.log" | tee -a "$MASTER"
    exit "$rc"
  fi
done
echo "=== $(date '+%Y-%m-%dT%H:%M:%S')  ALL DONE ===" | tee -a "$MASTER"