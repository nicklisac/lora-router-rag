#!/usr/bin/env python3
"""Mode 1 baseline: base EG2 (bf16 safetensors), one union index over all
four domains, no router. Scores recall@10 and nDCG@10 on the 20% eval
holdout, per domain and pooled.

Embedding runs through the python bf16 path (not the llama.cpp Q8_0 GGUF)
because mode 2's LoRAs are deltas on the bf16 base; the baseline must share
that base and precision or a quantization gap could masquerade as the effect.

Checkpointed: corpus embeddings save to results/embeddings/corpus_union.npy
so a killed run resumes.

Usage:
  CUDA_VISIBLE_DEVICES=2 python scripts/eval_mode1.py
  CUDA_VISIBLE_DEVICES=2 python scripts/eval_mode1.py --domain fiqa
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from prefixes import format_query, format_doc  # noqa: E402

DERIVED = os.path.join(HERE, "data", "derived")
EMB_DIR = os.path.join(HERE, "results", "embeddings")
RUN_LOG = os.path.join(HERE, "results", "run_log.jsonl")
CKPT = os.path.join(HERE, "data", "eg2", "unsloth-embeddinggemma-2")
K = 10
BATCH = 64


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def embed_checkpointed(model, texts, ids, out_path, is_query, label, batch=BATCH):
    if os.path.exists(out_path) and os.path.exists(out_path + ".ids.json"):
        m = np.load(out_path)
        ids_kept = json.load(open(out_path + ".ids.json"))
        if len(ids_kept) == len(ids):
            print(f"[{label}] loaded checkpoint {m.shape}")
            return m, ids_kept
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
    json.dump(ids, open(out_path + ".ids.json", "w"))
    return m, ids


def ndcg_at_k(ranked_dids, rel, k=K):
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked_dids[:k]) if d in rel)
    ideal = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
    return dcg / idcg if idcg else 0.0


def recall_at_k(ranked_dids, rel, k=K):
    return len(rel & set(ranked_dids[:k])) / len(rel) if rel else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="all",
                    help="score eval queries from one domain (index is still the union)")
    args = ap.parse_args()
    os.makedirs(EMB_DIR, exist_ok=True)

    from unsloth import FastSentenceTransformer
    model = FastSentenceTransformer.from_pretrained(
        CKPT, max_seq_length=512, full_finetuning=False,
        config_kwargs={"vision_config": None, "audio_config": None},
    )
    model.eval()

    corpus = load_jsonl(os.path.join(DERIVED, "corpus.jsonl"))
    queries = load_jsonl(os.path.join(DERIVED, "queries.jsonl"))
    queries = [q for q in queries if q["split"] == "eval"]
    if args.domain != "all":
        queries = [q for q in queries if q["domain"] == args.domain]

    print(f"embedding {len(corpus)} corpus docs (union index)...")
    C, cids = embed_checkpointed(
        model,
        [format_doc(c["text"], c["title"]) for c in corpus],
        [f'{c["domain"]}:{c["did"]}' for c in corpus],
        os.path.join(EMB_DIR, "corpus_union.npy"), False, "corpus",
    )
    id2row = {d: i for i, d in enumerate(cids)}
    did_of_row = {i: d for d, i in id2row.items()}

    print(f"embedding {len(queries)} eval queries...")
    Q, qids = embed_checkpointed(
        model,
        [format_query(q["query"]) for q in queries],
        [q["qid"] for q in queries],
        os.path.join(EMB_DIR, f"queries_eval{'' if args.domain == 'all' else '_' + args.domain}.npy"),
        True, "queries",
    )

    nd = {d: [] for d in ("fiqa", "arguana", "scifact", "gaming")}
    rc = {d: [] for d in nd}
    t0 = time.time()
    for i, q in enumerate(queries):
        rel = set(q["pos_dids"])
        sims = C @ Q[i]
        top = np.argpartition(-sims, min(K, len(sims) - 1))[:K]
        top = top[np.argsort(-sims[top])]
        ranked = [did_of_row[j] for j in top]
        # relevant docs are stored as global "domain:did" keys
        ranked_local = [d.split(":", 1)[1] for d in ranked]
        nd[q["domain"]].append(ndcg_at_k(ranked_local, rel))
        rc[q["domain"]].append(recall_at_k(ranked_local, rel))
        if (i + 1) % 200 == 0:
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
        "run": "mode1_baseline",
        "embedder": "embeddinggemma-2 bf16 safetensors",
        "index": "union of 4 domains",
        "n_corpus": len(cids),
        "n_eval_queries": nq,
        "per_domain": per_domain,
        "pooled": pooled,
        "seconds": round(time.time() - t0, 1),
    }
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print("=== MODE 1 SUMMARY (paste-ready) ===")
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()