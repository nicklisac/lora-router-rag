#!/usr/bin/env bash
# SUPERSEDED for the dataset by scripts/download_beir.sh (the SQID/ESCI parquets
# turned out to be CLIP vectors only - see PLAN.md "Ruled out"). Still valid for
# the EG2 GGUF + mmproj files in data/eg2/.
#
# Downloads model weights and dataset parquets into the repo's data/ dir.
# Idempotent: skips files that already exist at full size (uses HF resume via
# huggingface_hub, which handles partial files).
#
# Usage:
#   bash scripts/download.sh            # everything
#   bash scripts/download.sh eg2        # just the embedding model
#   bash scripts/download.sh sqid       # just the dataset parquets
set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
mkdir -p "$DATA_DIR"

EG2_FILES=(
  "embeddinggemma-2-Q8_0.gguf"
  "mmproj-Q8_0.gguf"
)
SQID_FILES=(
  "product_features.parquet"
  "query_features.parquet"
)
SQID_PREFIX="https://huggingface.co/datasets/crossingminds/shopping-queries-image-dataset/resolve/main/data"

fetch() {
  # fetch <url> <dest>
  local url="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  if [ -f "$dest" ]; then
    echo "exists, skip: $dest"
    return
  fi
  echo "downloading: $url -> $dest"
  curl -fL --retry 3 -C - -o "$dest.partial" "$url"
  mv "$dest.partial" "$dest"
}

what="${1:-all}"

if [ "$what" = "eg2" ] || [ "$what" = "all" ]; then
  for f in "${EG2_FILES[@]}"; do
    fetch "https://huggingface.co/unsloth/embeddinggemma-2-GGUF/resolve/main/$f" \
      "$DATA_DIR/eg2/$f"
  done
fi

if [ "$what" = "sqid" ] || [ "$what" = "all" ]; then
  for f in "${SQID_FILES[@]}"; do
    fetch "$SQID_PREFIX/$f" "$DATA_DIR/sqid/$f"
  done
fi

echo "done."
ls -la "$DATA_DIR"/eg2 "$DATA_DIR"/sqid 2>/dev/null