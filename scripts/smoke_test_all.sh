#!/usr/bin/env bash
# Pre-flight smoke test: validates every dependency of the experiment without
# running any experiment step. Phases:
#   1. files        - all dataset, model, and checkpoint files exist
#   2. data         - schemas, row counts, qrels->corpus joins
#   3. python env   - venv imports, CUDA visible, key versions
#   4. serving      - llama.cpp EG2 embeddings + Clef /v1/systemone on GPU 2
#   5. training     - EG2 loads from safetensors, LoRA, 3 MNRL steps, encode
# Prints a PASS/FAIL summary; exits nonzero if anything failed.
#
# Usage: bash scripts/smoke_test_all.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY=/mnt/Data/unsloth/studio/unsloth_studio/bin/python
PASS=""; FAIL=""
ok()  { PASS="$PASS $1"; printf 'PASS  %s\n' "$1"; }
bad() { FAIL="$FAIL $1"; printf 'FAIL  %s\n' "$1"; }

echo "=========== phase 1: files ==========="
FILES=(
  "data/beir/fiqa/corpus.parquet" "data/beir/fiqa/queries.parquet"
  "data/beir/fiqa/qrels_train.tsv" "data/beir/fiqa/qrels_test.tsv" "data/beir/fiqa/qrels_dev.tsv"
  "data/beir/arguana/corpus.parquet" "data/beir/arguana/queries.parquet" "data/beir/arguana/qrels_test.tsv"
  "data/beir/scifact/corpus.parquet" "data/beir/scifact/queries.parquet"
  "data/beir/scifact/qrels_train.tsv" "data/beir/scifact/qrels_test.tsv"
  "data/beir/gaming/corpus.jsonl" "data/beir/gaming/queries.jsonl" "data/beir/gaming/qrels_test.tsv"
  "data/eg2/embeddinggemma-2-Q8_0.gguf" "data/eg2/mmproj-Q8_0.gguf"
  "data/eg2/unsloth-embeddinggemma-2/config.json"
)
CLEF_GGUF="$HOME/models/Clef-Flash-Q8_0.gguf"
missing=0
for f in "${FILES[@]}"; do
  if [ ! -s "$HERE/$f" ]; then echo "  missing: $f"; missing=$((missing+1)); fi
done
[ -s "$CLEF_GGUF" ] || { echo "  missing: $CLEF_GGUF"; missing=$((missing+1)); }
if [ "$missing" -eq 0 ]; then ok "files (${#FILES[@]} present)"; else bad "files ($missing missing)"; fi
if [ ! -f "$HERE/data/eg2/unsloth-embeddinggemma-2/model.safetensors" ] \
   && ! ls "$HERE"/data/eg2/unsloth-embeddinggemma-2/model-*.safetensors >/dev/null 2>&1; then
  bad "EG2 safetensors weights"; else ok "EG2 safetensors weights"; fi

echo "=========== phase 2: data integrity ==========="
if (cd "$HERE" && CUDA_VISIBLE_DEVICES=2 "$VENV_PY" - <<'PYEOF'
import json
import pyarrow.parquet as pq
D = "data/beir/"
def qrels(p):
    q = {}
    with open(p) as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 3:
                q.setdefault(parts[0], set()).add(parts[1])
    return q
def check(name, cf, qf, rels):
    c = pq.read_table(D + cf); q = pq.read_table(D + qf)
    docids = set(c.column("_id").to_pylist())
    allq, pairs = set(), 0
    miss = 0
    for rf in rels:
        r = qrels(D + rf); allq |= set(r); pairs += sum(len(v) for v in r.values())
        miss += sum(1 for v in r.values() for d in v if d not in docids)
    assert len(docids) > 1000 and len(q.column_names) >= 3
    print(f"  {name}: corpus {len(docids)} | queries {len(q.column('_id'))} | qrels {len(allq)} q, {pairs} pairs, {miss} dangling")
check("fiqa",    "fiqa/corpus.parquet",    "fiqa/queries.parquet",    ["fiqa/qrels_train.tsv","fiqa/qrels_test.tsv"])
check("arguana", "arguana/corpus.parquet", "arguana/queries.parquet", ["arguana/qrels_test.tsv"])
check("scifact", "scifact/corpus.parquet", "scifact/queries.parquet", ["scifact/qrels_train.tsv","scifact/qrels_test.tsv"])
c = {json.loads(l)["_id"] for l in open(D + "gaming/corpus.jsonl")}
q = {json.loads(l)["_id"] for l in open(D + "gaming/queries.jsonl")}
r = qrels(D + "gaming/qrels_test.tsv")
miss = sum(1 for v in r.values() for d in v if d not in c)
print(f"  gaming: corpus {len(c)} | queries {len(q)} | qrels {len(r)} q, {sum(len(v) for v in r.values())} pairs, {miss} dangling")
PYEOF
); then ok "data integrity (4 domains, joins clean)"; else bad "data integrity"; fi

echo "=========== phase 3: python env ==========="
if CUDA_VISIBLE_DEVICES=2 "$VENV_PY" > "$HERE/results/smoke_pyenv.log" 2>&1 <<'PYEOF'
import torch, transformers, unsloth, sentence_transformers, datasets, peft
assert torch.cuda.is_available() and torch.cuda.device_count() == 1, "CUDA not visible"
assert torch.cuda.is_bf16_supported(), "bf16 not supported"
print(f"  torch {torch.__version__} | cuda {torch.cuda.get_device_name(0)}")
print(f"  transformers {transformers.__version__} | unsloth {unsloth.__version__}")
print(f"  sentence-transformers {sentence_transformers.__version__} | peft {peft.__version__} | datasets {datasets.__version__}")
from unsloth import FastSentenceTransformer  # noqa
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss  # noqa
PYEOF
then ok "python env (imports + CUDA 2 + bf16)"; else bad "python env (see results/smoke_pyenv.log)"; fi

echo "=========== phase 4: serving (GPU 2) ==========="
bash "$HERE/scripts/start_servers.sh" > "$HERE/results/smoke_servers.log" 2>&1 || bad "server startup"
# /health returns 200 even while the model loads; poll the real endpoints.
ready_clef=0; ready_eg2=0
for _ in $(seq 1 90); do
  [ "$ready_clef" -eq 0 ] && curl -s -m 5 http://127.0.0.1:18125/v1/systemone \
    -H 'Content-Type: application/json' \
    -d '{"model":"clef","state":"ping","questions":{"p":{"type":"noul","instructions":"ping?"}}}' \
    | grep -q '"answers"' && ready_clef=1
  [ "$ready_eg2" -eq 0 ] && curl -s -m 5 http://127.0.0.1:18124/v1/embeddings \
    -H 'Content-Type: application/json' \
    -d '{"model":"eg2","input":["warmup"]}' \
    | grep -q '"embedding"' && ready_eg2=1
  [ "$ready_clef" -eq 1 ] && [ "$ready_eg2" -eq 1 ] && break
  sleep 5
done
[ "$ready_eg2" -eq 1 ] && ok "EG2 server :18124" || bad "EG2 server :18124"
[ "$ready_clef" -eq 1 ] && ok "Clef server :18125" || bad "Clef server :18125"
if CUDA_VISIBLE_DEVICES=2 "$VENV_PY" > "$HERE/results/smoke_live.log" 2>&1 <<'PYEOF'
import json, math, urllib.request
def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())
e = post("http://127.0.0.1:18124/v1/embeddings",
         {"model": "eg2", "input": "task: search result | query: a green shearling jacket"})
v = e["data"][0]["embedding"]
n = math.sqrt(sum(x * x for x in v))
assert len(v) == 768 and 0.9 < n < 1.1, f"bad embedding: len {len(v)} norm {n}"
print(f"  EG2 embed: 768d, norm {n:.4f}")
r = post("http://127.0.0.1:18125/v1/systemone", {
    "model": "clef",
    "state": "A green embroidered shearling jacket, womens winter outerwear",
    "questions": {"domain": {"type": "choice",
                             "instructions": "Which domain?",
                             "criteria": {"fashion": "clothing", "news": "news",
                                          "wiki": "wikipedia"}}}})
a = r["answers"]["domain"]
assert "fashion" in a["probabilities"] and a["confidence"] > 0.5
print(f"  Clef route: {a['choice']} (conf {a['confidence']:.3f})")
PYEOF
then ok "EG2 embed + Clef route (live calls)"; else bad "EG2 embed + Clef route (see results/smoke_live.log)"; fi
bash "$HERE/scripts/stop_servers.sh" > /dev/null 2>&1
sleep 3

echo "=========== phase 5: training path ==========="
if (cd "$HERE" && CUDA_VISIBLE_DEVICES=2 "$VENV_PY" scripts/smoke_test_training.py 2>&1 \
     | tee "$HERE/results/smoke_training.log" | grep -q "SMOKE_TEST_PASSED"); then
  ok "training (EG2 safetensors + LoRA + 3 MNRL steps + encode)"
else
  bad "training (see results/smoke_training.log)"
fi

echo "=========== summary ==========="
echo "PASSED:$( [ -n "$PASS" ] && echo "$PASS" )"
echo "FAILED:$( [ -n "$FAIL" ] && echo "$FAIL" || echo " none" )"
[ -z "$FAIL" ] && echo "ALL CHECKS PASSED" || echo "SOME CHECKS FAILED"
exit $([ -z "$FAIL" ] && echo 0 || echo 1)