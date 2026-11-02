#!/usr/bin/env bash
# Start the llama.cpp server that hosts both embedders for the project:
#   - EG2 (Q8_0) for /v1/embeddings, on GPU 2
#   - Clef-Flash (Q8_0) for /v1/systemone routing, on GPU 2
# Each model is loaded separately (they cannot share one llama-server), so this
# runs two servers on different ports.
#
# Usage:
#   bash scripts/start_servers.sh          # both
#   bash scripts/start_servers.sh eg2      # just the embedder
#   bash scripts/start_servers.sh clef     # just the router
#   bash scripts/stop_servers.sh           # kill both
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
LCC="$HOME/llama.cpp-p2p/b11985/llama-server"
EG2="$HERE/data/eg2/embeddinggemma-2-Q8_0.gguf"
CLEF="$HOME/models/Clef-Flash-Q8_0.gguf"
EG2_PORT=18124
CLEF_PORT=18125

# The Unsloth cuda13 bundle links libggml-cuda.so against libcudart.so.13 and
# libcublas.so.13, which are NOT on the default library path. Without them the
# CUDA backend silently fails to load and everything runs on CPU. Find a
# directory that has both (the Unsloth Studio venv ships nvidia cu13 wheels).
CU13_DIR="$(find /mnt/Data/unsloth /mnt/Models/unsloth /home/nick/.local -name libcudart.so.13 -path "*nvidia/cu13*" 2>/dev/null | head -1 | xargs -r dirname)"
if [ -n "$CU13_DIR" ] && [ -f "$CU13_DIR/libcublas.so.13" ]; then
  export LD_LIBRARY_PATH="$CU13_DIR:${LD_LIBRARY_PATH:-}"
  echo "CUDA 13 runtime: $CU13_DIR"
else
  echo "WARNING: nvidia cu13 runtime libs not found; servers will run on CPU"
fi

check_port() { # port label
  if curl -s -o /dev/null "http://127.0.0.1:$1/health"; then
    echo "$2 already up on :$1"
    return 0
  fi
  return 1
}

if [ "${1:-both}" = "eg2" ] || [ "${1:-both}" = "both" ]; then
  if ! check_port "$EG2_PORT" "EG2"; then
    echo "starting EG2 on :$EG2_PORT"
    CUDA_VISIBLE_DEVICES=2 nohup "$LCC" -m "$EG2" \
      --embeddings --pooling mean --flash-attn off \
      -ngl 99 --port "$EG2_PORT" --ctx-size 2048 \
      > "$HERE/results/eg2-server.log" 2>&1 &
    echo $! > "$HERE/results/eg2-server.pid"
  fi
fi

if [ "${1:-both}" = "clef" ] || [ "${1:-both}" = "both" ]; then
  if ! check_port "$CLEF_PORT" "Clef"; then
    echo "starting Clef on :$CLEF_PORT"
    CUDA_VISIBLE_DEVICES=2 nohup "$LCC" -m "$CLEF" \
      -ngl 99 --port "$CLEF_PORT" --ctx-size 4096 \
      -b 4096 -ub 4096 \
      > "$HERE/results/clef-server.log" 2>&1 &
    echo $! > "$HERE/results/clef-server.pid"
  fi
fi

echo "waiting for readiness..."
for port in $EG2_PORT $CLEF_PORT; do
  for i in $(seq 1 120); do
    curl -s -o /dev/null "http://127.0.0.1:$port/health" && { echo "ready: :$port"; break; }
    sleep 2
  done
done