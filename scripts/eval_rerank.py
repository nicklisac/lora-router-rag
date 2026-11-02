#!/usr/bin/env python3
"""Clef as a post-retrieval relevance pruner (mode 4).

Different from mode 3 (routing): the retriever runs first, then Clef judges
each retrieved doc and we drop the ones it marks irrelevant. The SystemOne
API answers N questions about ONE state in a single forward pass, so one
request per query carries all K doc judgments:

  state     = the query + the K retrieved docs as one text block
  questions = K noul questions: "Is doc i relevant to the query?"

Variants (--source base|joint, --topk N, keep top N/2 of them):
  base   retrieve with base EG2 (the 0.4503 reference)
  joint  retrieve with the joint LoRA adapter (the 0.4947 best-so-far)

Each doc keeps BOTH its embedding rank and its Clef score; we measure:
  embedding-only   baseline for this (source, topk)
  rerank           docs re-ordered by Clef score
  prune            low-Clef docs removed, remainder in original order
Metrics are nDCG@10 / recall@10 on the same 2,161-query holdout.

Clef calls checkpoint to results/rerank_scores_<tag>.jsonl (one line per
query) so a killed run resumes.

Usage:
  CUDA_VISIBLE_DEVICES=2 python scripts/eval_rerank.py --source base  --topk 10
  CUDA_VISIBLE_DEVICES=2 python scripts/eval_rerank.py --source joint --topk 20
"""
import argparse
import json
import math
import os
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DERIVED = os.path.join(HERE, "data", "derived")
EMB_DIR = os.path.join(HERE, "results", "embeddings")
RUN_LOG = os.path.join(HERE, "results", "run_log.jsonl")
CLEF_URL = "http://127.0.0.1:18125/v1/systemone"
K_FINAL = 10
DOC_CHARS = 300  # per-doc cap in the state; 20 docs stay under the 4096-token ubatch


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def ndcg_at_k(ranked, rel, k=K_FINAL):
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in rel)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg if idcg else 0.0


def recall_at_k(ranked, rel, k=K_FINAL):
    return len(rel & set(ranked[:k])) / len(rel) if rel else 0.0


def judge(query, docs, attempt=0):
    """One Clef call, all docs judged at once. Returns list of noul probs."""
    state = "Query: " + query + "\n\n" + "\n".join(
        f"Doc {i + 1}: {d}" for i, d in enumerate(docs))
    questions = {f"doc_{i + 1}": {"type": "noul",
                                  "instructions": f"Is Doc {i + 1} relevant to the query?"}
                 for i in range(len(docs))}
    body = json.dumps({"model": "clef", "state": state,
                       "questions": questions}).encode()
    for i in range(3):
        try:
            req = urllib.request.Request(CLEF_URL, data=body,
                                         headers={"Content-Type": "application/json"})
            d = json.loads(urllib.request.urlopen(req, timeout=300).read())
            return [float(d["answers"][f"doc_{j + 1}"]["noul"])
                    for j in range(len(docs))]
        except Exception:
            if i == 2:
                raise
            time.sleep(3 * (i + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["base", "joint"], default="base")
    ap.add_argument("--topk", type=int, default=20,
                    help="retrieve this many; keep the top half of Clef-scored docs")
    args = ap.parse_args()
    topk = args.topk
    keep = topk // 2
    tag = f"{args.source}_k{topk}"
    ckpt = os.path.join(HERE, "results", f"rerank_scores_{tag}.jsonl")

    corpus = load_jsonl(os.path.join(DERIVED, "corpus.jsonl"))
    queries = [q for q in load_jsonl(os.path.join(DERIVED, "queries.jsonl"))
               if q["split"] == "eval"]
    nq = len(queries)
    cids = [f'{c["domain"]}:{c["did"]}' for c in corpus]
    doc_of = {(c["domain"], c["did"]): (c["title"], c["text"]) for c in corpus}

    if args.source == "base":
        C = np.load(os.path.join(EMB_DIR, "corpus_union.npy"))
        Q = np.load(os.path.join(EMB_DIR, "queries_eval.npy"))
    else:
        C = np.load(os.path.join(EMB_DIR, f"mode2_{args.source}_corpus.npy"))
        Q = np.load(os.path.join(EMB_DIR, f"mode2_{args.source}_queries.npy"))
    assert len(C) == len(cids) and len(Q) == nq

    # retrieve topk for every query
    print(f"[retrieve] {nq} queries x top{topk} via {args.source} ...", flush=True)
    tops = []
    for i, q in enumerate(queries):
        sims = C @ Q[i]
        idx = np.argpartition(-sims, min(topk, len(sims) - 1))[:topk]
        idx = idx[np.argsort(-sims[idx])]
        tops.append([cids[j] for j in idx])

    # Clef relevance judgments, checkpointed
    scored = {}
    if os.path.exists(ckpt):
        for line in open(ckpt):
            r = json.loads(line)
            scored[r["qid"]] = r["scores"]
    with open(ckpt, "a") as cf:
        t0 = time.time()
        for i, q in enumerate(queries):
            if q["qid"] in scored:
                continue
            docs = []
            for cid in tops[i]:
                dom, did = cid.split(":", 1)
                title, text = doc_of[(dom, did)]
                body = (f"[{title}] " if title else "") + text
                docs.append(body[:DOC_CHARS])
            scores = judge(q["query"], docs)
            scored[q["qid"]] = scores
            cf.write(json.dumps({"qid": q["qid"], "scores": scores}) + "\n")
            cf.flush()
            if (i + 1) % 100 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"[clef] {i + 1}/{nq}  {rate:.1f} q/s  "
                      f"eta {(nq - i - 1) / max(rate, 1e-6) / 60:.1f}m", flush=True)
    print(f"[clef] done in {time.time() - t0:.0f}s", flush=True)

    # score the three strategies per query
    emb, rerank, prune = [], [], []
    for i, q in enumerate(queries):
        rel = set(q["pos_dids"])
        cands = [cid.split(":", 1)[1] for cid in tops[i]]
        scores = scored[q["qid"]]
        emb.append((ndcg_at_k(cands, rel), recall_at_k(cands, rel)))
        order = sorted(range(topk), key=lambda j: -scores[j])
        rerank.append((ndcg_at_k([cands[j] for j in order], rel),
                       recall_at_k([cands[j] for j in order], rel)))
        keep_idx = order[:keep]  # highest-Clef docs, original relative order
        keep_set = set(keep_idx)
        pruned = [cands[j] for j in range(topk) if j in keep_set]
        prune.append((ndcg_at_k(pruned, rel), recall_at_k(pruned, rel)))

    def agg(col):
        return {"nDCG@10": round(float(np.mean([c[0] for c in col])), 4),
                "recall@10": round(float(np.mean([c[1] for c in col])), 4)}

    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run": f"mode4_rerank_{tag}",
        "retriever": args.source, "topk": topk, "keep": keep,
        "n_eval_queries": nq,
        "embedding_only": agg(emb),
        "clef_rerank": agg(rerank),
        "clef_prune": agg(prune),
    }
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print("=== MODE 4 RERANK SUMMARY ===")
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()