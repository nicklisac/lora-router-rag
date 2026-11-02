#!/usr/bin/env python3
"""Mode 2: score one LoRA-finetuned EG2 adapter on the same eval holdout as
mode 1, so the numbers line up directly.

Each adapter is loaded as a MERGED standalone bf16 checkpoint
(results/lora/<name>_merged, produced by train_lora.py). We deliberately do NOT
reload the raw adapter through FastSentenceTransformer: the research notes flag
that doing so without for_inference=True returns untransformed vectors and
throws nothing. Merging sidesteps that whole path.

Same union index, same eval queries, same prefixes (src/prefixes.py), same
nDCG@10/recall@10, same 20% global holdout as mode 1. The ONLY thing that
changes from mode 1 is the model doing the embedding.

Embeddings are checkpointed per adapter (results/embeddings/mode2_<name>_
corpus.npy and ..._queries.npy) with an id-gate copied from mode 1: a .npy is
only trusted if its saved .ids.json matches the current data exactly, so a
corpus.jsonl regenerated in a different row order triggers a re-embed instead
of a silent misalignment.

Usage:
  CUDA_VISIBLE_DEVICES=2 python scripts/eval_mode2.py --name fiqa
  CUDA_VISIBLE_DEVICES=2 python scripts/eval_mode2.py --name joint
"""
import argparse
import json
import math
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys_path = os.path.join(HERE, "src")
import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)
from prefixes import format_query, format_doc  # noqa: E402

DERIVED = os.path.join(HERE, "data", "derived")
EMB_DIR = os.path.join(HERE, "results", "embeddings")
LORA_DIR = os.path.join(HERE, "results", "lora")
RUN_LOG = os.path.join(HERE, "results", "run_log.jsonl")
K = 10
BATCH = 64


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def embed_checkpointed(model, texts, ids, out_path, label, batch=BATCH):
    # Same id-gated resume as mode 1: the .npy is only trusted if the saved id
    # list matches the current data exactly. This is what catches a corpus.jsonl
    # regenerated in a different row order — otherwise a stale .npy would
    # silently misalign against the new order and score garbage with no error.
    ids_path = out_path + ".ids.json"
    if os.path.exists(out_path) and os.path.exists(ids_path):
        m = np.load(out_path)
        kept = json.load(open(ids_path))
        if len(kept) == len(ids) and kept == ids:
            print(f"[{label}] loaded checkpoint {m.shape}")
            return m
        print(f"[{label}] checkpoint ids mismatch; re-embedding")
    rows = []
    t0 = time.time()
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        out = model.encode(
            chunk, batch_size=len(chunk), show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        rows.append(np.asarray(out, dtype=np.float32))
        done = min(i + batch, len(texts))
        if done % (batch * 10) == 0 or done == len(texts):
            rate = done / (time.time() - t0)
            print(f"[{label}] {done}/{len(texts)}  {rate:.0f} docs/s  "
                  f"eta {(len(texts) - done) / max(rate, 1e-6):.0f}s")
    m = np.vstack(rows)
    np.save(out_path, m)
    json.dump(ids, open(ids_path, "w"))
    return m


def ndcg_at_k(ranked, rel, k=K):
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in rel)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg if idcg else 0.0


def recall_at_k(ranked, rel, k=K):
    return len(rel & set(ranked[:k])) / len(rel) if rel else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True,
                    choices=["fiqa", "arguana", "scifact", "gaming", "joint"])
    args = ap.parse_args()
    os.makedirs(EMB_DIR, exist_ok=True)

    from unsloth import FastSentenceTransformer
    merged_dir = os.path.join(LORA_DIR, args.name + "_merged")
    print(f"[model] loading merged adapter {merged_dir} ...", flush=True)
    model = FastSentenceTransformer.from_pretrained(
        merged_dir, max_seq_length=512, full_finetuning=False,
        config_kwargs={"vision_config": None, "audio_config": None},
    )
    model.eval()

    corpus = load_jsonl(os.path.join(DERIVED, "corpus.jsonl"))
    queries = [q for q in load_jsonl(os.path.join(DERIVED, "queries.jsonl"))
               if q["split"] == "eval"]

    corpus_ids = [f'{c["domain"]}:{c["did"]}' for c in corpus]
    query_ids = [q["qid"] for q in queries]
    C = embed_checkpointed(
        model,
        [format_doc(c["text"], c["title"]) for c in corpus],
        corpus_ids,
        os.path.join(EMB_DIR, f"mode2_{args.name}_corpus.npy"),
        "corpus",
    )
    Q = embed_checkpointed(
        model,
        [format_query(q["query"]) for q in queries],
        query_ids,
        os.path.join(EMB_DIR, f"mode2_{args.name}_queries.npy"),
        "queries",
    )

    # doc "domain:did" -> row, matching mode 1's keying
    did_of_row = {i: f'{c["domain"]}:{c["did"]}' for i, c in enumerate(corpus)}

    nd = {d: [] for d in ("fiqa", "arguana", "scifact", "gaming")}
    rc = {d: [] for d in nd}
    t0 = time.time()
    for i, q in enumerate(queries):
        rel = set(q["pos_dids"])
        sims = C @ Q[i]
        top = np.argpartition(-sims, min(K, len(sims) - 1))[:K]
        top = top[np.argsort(-sims[top])]
        ranked = [did_of_row[j].split(":", 1)[1] for j in top]
        nd[q["domain"]].append(ndcg_at_k(ranked, rel))
        rc[q["domain"]].append(recall_at_k(ranked, rel))
        if (i + 1) % 500 == 0:
            print(f"[score] {i + 1}/{len(queries)}  {time.time() - t0:.0f}s")

    per_domain = {}
    for d in nd:
        if nd[d]:
            per_domain[d] = {
                "n_queries": len(nd[d]),
                "ndcg@10": round(float(np.mean(nd[d])), 4),
                "recall@10": round(float(np.mean(rc[d])), 4),
            }
    nq = sum(len(v) for v in nd.values())
    pooled = {
        "ndcg@10": round(sum(v for vs in nd.values() for v in vs) / nq, 4),
        "recall@10": round(sum(v for vs in rc.values() for v in vs) / nq, 4),
    }

    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run": f"mode2_{args.name}",
        "embedder": f"embeddinggemma-2 bf16 + {args.name} LoRA (merged)",
        "index": "union of 4 domains",
        "n_corpus": len(corpus),
        "n_eval_queries": nq,
        "per_domain": per_domain,
        "pooled": pooled,
        "seconds": round(time.time() - t0, 1),
    }
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"=== MODE 2 [{args.name}] SUMMARY (paste-ready) ===")
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()