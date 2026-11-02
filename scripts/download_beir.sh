#!/usr/bin/env bash
# Downloads the four BEIR text-to-text datasets that form the enterprise-KB
# testbed:
#   finance:    BeIR/fiqa            (BeIR/fiqa-qrels)   train+dev+test
#   legal:      BeIR/arguana         (BeIR/arguana-qrels) test only
#   science:    BeIR/scifact         (BeIR/scifact-qrels) train+test
#   community:  mteb/cqadupstack-gaming                   test only
#
# Each file is verified with a byte-range request before download. Failures
# (401/404) are logged, not fatal, because HF rate-limits this box's API
# access intermittently; rerun the script to pick up anything that failed.
#
# Usage: bash scripts/download_beir.sh
set -uo pipefail

DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/beir"
mkdir -p "$DATA_DIR"
SUMMARY="$DATA_DIR/_SUMMARY.txt"
: > "$SUMMARY"

# repo path dest
FILES=(
  # finance: fiqa (BeIR/fiqa, not the 401'ing fiqa2018 repo)
  "BeIR/fiqa"                    "corpus/corpus-00000-of-00001.parquet" "fiqa/corpus.parquet"
  "BeIR/fiqa"                    "queries/queries-00000-of-00001.parquet" "fiqa/queries.parquet"
  "BeIR/fiqa-qrels"              "test.tsv" "fiqa/qrels_test.tsv"
  "BeIR/fiqa-qrels"              "train.tsv" "fiqa/qrels_train.tsv"
  "BeIR/fiqa-qrels"              "dev.tsv" "fiqa/qrels_dev.tsv"
  # legal: arguana
  "BeIR/arguana"                 "corpus/corpus-00000-of-00001.parquet" "arguana/corpus.parquet"
  "BeIR/arguana"                 "queries/queries-00000-of-00001.parquet" "arguana/queries.parquet"
  "BeIR/arguana-qrels"           "test.tsv" "arguana/qrels_test.tsv"
  # science: scifact
  "BeIR/scifact"                 "corpus/corpus-00000-of-00001.parquet" "scifact/corpus.parquet"
  "BeIR/scifact"                 "queries/queries-00000-of-00001.parquet" "scifact/queries.parquet"
  "BeIR/scifact-qrels"           "test.tsv" "scifact/qrels_test.tsv"
  "BeIR/scifact-qrels"           "train.tsv" "scifact/qrels_train.tsv"
  # community: cqadupstack-gaming (mteb mirror; the BeIR/cqadupstack* repos 401)
  "mteb/cqadupstack-gaming"      "corpus.jsonl" "gaming/corpus.jsonl"
  "mteb/cqadupstack-gaming"      "queries.jsonl" "gaming/queries.jsonl"
  "mteb/cqadupstack-gaming"      "qrels/test.tsv" "gaming/qrels_test.tsv"
)

ok=0; failed=0
i=0
while [ $i -lt ${#FILES[@]} ]; do
  repo="${FILES[$i]}"; rpath="${FILES[$((i+1))]}"; destrel="${FILES[$((i+2))]}"
  i=$((i+3))
  dest="$DATA_DIR/$destrel"
  url="https://huggingface.co/datasets/$repo/resolve/main/$rpath"
  mkdir -p "$(dirname "$dest")"

  if [ -f "$dest" ] && [ -s "$dest" ]; then
    echo "SKIP   $destrel (exists)"
    echo "skip $destrel" >> "$SUMMARY"
    continue
  fi

  code=$(curl -sL -o /dev/null -w "%{http_code}" -r 0-0 "$url")
  if [ "$code" != "200" ] && [ "$code" != "206" ]; then
    echo "FAIL($code) $destrel"
    echo "fail $code $destrel" >> "$SUMMARY"
    failed=$((failed+1))
    sleep 3
    continue
  fi

  echo "GET    $destrel"
  if curl -fL --retry 2 -C - -o "$dest.partial" "$url"; then
    mv "$dest.partial" "$dest"
    echo "ok $destrel $(stat -c%s "$dest")" >> "$SUMMARY"
    ok=$((ok+1))
  else
    rm -f "$dest.partial"
    echo "FAIL(curl) $destrel"
    echo "fail curl $destrel" >> "$SUMMARY"
    failed=$((failed+1))
  fi
done

echo
echo "done: $ok ok, $failed failed (see $SUMMARY)"
du -sh "$DATA_DIR"