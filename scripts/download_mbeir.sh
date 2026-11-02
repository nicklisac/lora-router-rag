#!/usr/bin/env bash
# SUPERSEDED 2026-10-30: M-BEIR dropped (its text-viable tasks are mostly
# image-query tasks). Dataset is now the four BEIR corpora; see
# scripts/download_beir.sh. Kept for reference only.
#
# Downloads M-BEIR text files (queries, candidate pools, qrels, instructions)
# for the domains used by the experiment. Text only: the 170 GB image tarball
# is intentionally not fetched.
#
# Domains:
#   fashion:  fashion200k task0 (text->image), fashion200k task3, fashioniq task7
#   news:     visualnews task0 (text->image), edis task2, nights task4 (img->img)
#   wiki:     webqa task1 (text->text), oven task8 (img,text->text)
#
# Splits: test for evaluation, train for LoRA training, local cand pools.
#
# Usage: bash scripts/download_mbeir.sh
set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/mbeir"
BASE="https://huggingface.co/datasets/TIGER-Lab/M-BEIR/resolve/main"

FILES=(
  # instructions (all domains, one small file)
  instructions/query_instructions.tsv
  # fashion
  query/test/mbeir_fashion200k_task0_test.jsonl
  query/test/mbeir_fashion200k_task3_test.jsonl
  query/test/mbeir_fashioniq_task7_test.jsonl
  query/train/mbeir_fashion200k_train.jsonl
  query/train/mbeir_fashioniq_train.jsonl
  cand_pool/local/mbeir_fashion200k_task0_cand_pool.jsonl
  cand_pool/local/mbeir_fashion200k_task3_cand_pool.jsonl
  cand_pool/local/mbeir_fashioniq_task7_cand_pool.jsonl
  qrels/test/mbeir_fashion200k_task0_test_qrels.txt
  qrels/test/mbeir_fashion200k_task3_test_qrels.txt
  qrels/test/mbeir_fashioniq_task7_test_qrels.txt
  qrels/train/mbeir_fashion200k_train_qrels.txt
  qrels/train/mbeir_fashioniq_train_qrels.txt
  # news
  query/test/mbeir_visualnews_task0_test.jsonl
  query/test/mbeir_edis_task2_test.jsonl
  query/test/mbeir_nights_task4_test.jsonl
  query/train/mbeir_visualnews_train.jsonl
  query/train/mbeir_edis_train.jsonl
  query/train/mbeir_nights_train.jsonl
  cand_pool/local/mbeir_visualnews_task0_cand_pool.jsonl
  cand_pool/local/mbeir_edis_task2_cand_pool.jsonl
  cand_pool/local/mbeir_nights_task4_cand_pool.jsonl
  qrels/test/mbeir_visualnews_task0_test_qrels.txt
  qrels/test/mbeir_edis_task2_test_qrels.txt
  qrels/test/mbeir_nights_task4_test_qrels.txt
  qrels/train/mbeir_visualnews_train_qrels.txt
  qrels/train/mbeir_edis_train_qrels.txt
  qrels/train/mbeir_nights_train_qrels.txt
  # wiki
  query/test/mbeir_webqa_task1_test.jsonl
  query/test/mbeir_oven_task8_test.jsonl
  query/train/mbeir_webqa_train.jsonl
  query/train/mbeir_oven_train.jsonl
  cand_pool/local/mbeir_webqa_task1_cand_pool.jsonl
  cand_pool/local/mbeir_oven_task8_cand_pool.jsonl
  qrels/test/mbeir_webqa_task1_test_qrels.txt
  qrels/test/mbeir_oven_task8_test_qrels.txt
  qrels/train/mbeir_webqa_train_qrels.txt
  qrels/train/mbeir_oven_train_qrels.txt
)

for f in "${FILES[@]}"; do
  dest="$DATA_DIR/$f"
  mkdir -p "$(dirname "$dest")"
  if [ -f "$dest" ]; then
    echo "skip $f"
    continue
  fi
  echo "get  $f"
  curl -fL --retry 3 -C - -o "$dest.partial" "$BASE/$f"
  mv "$dest.partial" "$dest"
done

echo "done."
du -sh "$DATA_DIR"
find "$DATA_DIR" -name "*.partial" -size +0 | grep . && echo "INCOMPLETE FILES PRESENT" || true