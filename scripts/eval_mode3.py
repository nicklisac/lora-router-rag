#!/usr/bin/env python3
"""Mode 3: base EG2 embeddings (reused from mode 1), Clef routes each query
to a domain, search restricted to that domain's slice of the union corpus.

Also computes the oracle variant (search restricted to the TRUE domain) as
the ceiling of "narrowing the search set" with no finetuning.

Routing decisions checkpoint to results/routing_eval.jsonl so a rerun after
a kill does not re-ask Clef.

Usage:
  bash scripts/start_servers.sh clef   # first, if not running
  CUDA_VISIBLE_DEVICES=2 python scripts/eval_mode3.py
"""
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
ROUTING = os.path.join(HERE, "results", "routing_eval.jsonl")
CLEF_URL = "http://127.0.0.1:18125/v1/systemone"
K = 10
DOMAINS = ("fiqa", "arguana", "scifact", "gaming")
CRITERIA = {
    "fiqa": "Finance and investment questions",
    "arguana": "Legal argument and case questions",
    "scifact": "Scientific research and fact-checking questions",
    "gaming": "Gaming and computer questions from a help forum",
}


def route_one(query, attempt=0):
    body = json.dumps({
        "model": "clef",
        "state": query,
        "questions": {"domain": {
            "type": "choice",
            "instructions": "Which knowledge area should this query be routed to?",
            "criteria": CRITERIA,
        }},
    }).encode()
    req = urllib.request.Request(CLEF_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    for i in range(3):
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=120).read())
            a = r["answers"]["domain"]
            return a["choice"], a["confidence"], a["probabilities"]
        except Exception:
            if i == 2:
                raise
            time.sleep(3 * (i + 1))


def ndcg_at_k(ranked, rel, k=K):
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in rel)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg if idcg else 0.0


def recall_at_k(ranked, rel, k=K):
    return len(rel & set(ranked[:k])) / len(rel) if rel else 0.0


def main():
    corpus = [json.loads(l) for l in
              open(os.path.join(DERIVED, "corpus.jsonl"))]
    C = np.load(os.path.join(EMB_DIR, "corpus_union.npy"))
    cids = json.load(open(os.path.join(EMB_DIR, "corpus_union.npy.ids.json")))
    queries = [json.loads(l) for l in
               open(os.path.join(DERIVED, "queries.jsonl"))]
    queries = [q for q in queries if q["split"] == "eval"]
    Q = np.load(os.path.join(EMB_DIR, "queries_eval.npy"))
    assert len(Q) == len(queries) and len(cids) == len(C), \
        "embeddings out of sync with data/derived (rerun mode 1 or prep_data)"

    # row index per domain
    rows_by_domain = {d: [] for d in DOMAINS}
    for i, c in enumerate(corpus):
        rows_by_domain[c["domain"]].append(i)

    # route, with checkpointing
    routed = {}
    if os.path.exists(ROUTING):
        for line in open(ROUTING):
            r = json.loads(line)
            routed[r["qid"]] = r
    rf = open(ROUTING, "a")
    t0 = time.time()
    for i, q in enumerate(queries):
        if q["qid"] in routed:
            continue
        choice, conf, probs = route_one(q["query"])
        rec = {"qid": q["qid"], "gold": q["domain"], "routed": choice,
               "confidence": conf, "probs": probs}
        routed[q["qid"]] = rec
        rf.write(json.dumps(rec) + "\n")
        rf.flush()
        if (i + 1) % 100 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"[route] {i + 1}/{len(queries)}  {rate:.1f} q/s")
    rf.close()

    # score: mode3 (routed slice) and oracle (true slice)
    per_domain = {d: {"n": 0, "m3_ndcg": 0.0, "m3_recall": 0.0,
                      "oracle_ndcg": 0.0, "oracle_recall": 0.0}
                  for d in DOMAINS}
    misrouted = 0
    for i, q in enumerate(queries):
        rel = set(q["pos_dids"])
        sims = C @ Q[i]
        r = routed[q["qid"]]
        if r["routed"] != q["domain"]:
            misrouted += 1
        per_domain[q["domain"]]["n"] += 1
        for tag, dom in (("m3", r["routed"]), ("oracle", q["domain"])):
            rows = rows_by_domain[dom]
            sub = sims[rows]
            top = np.argpartition(-sub, min(K, len(sub) - 1))[:K]
            top = top[np.argsort(-sub[top])]
            ranked = [cids[rows[j]] for j in top]
            ranked = [d.split(":", 1)[1] for d in ranked]
            p = per_domain[q["domain"]]
            p[f"{tag}_ndcg"] += ndcg_at_k(ranked, rel)
            p[f"{tag}_recall"] += recall_at_k(ranked, rel)

    out = {}
    nq = len(queries)
    for d in DOMAINS:
        p = per_domain[d]
        n = p["n"]
        out[d] = {"n_queries": n,
                  "mode3_ndcg@10": round(p["m3_ndcg"] / n, 4),
                  "mode3_recall@10": round(p["m3_recall"] / n, 4),
                  "oracle_ndcg@10": round(p["oracle_ndcg"] / n, 4),
                  "oracle_recall@10": round(p["oracle_recall"] / n, 4)}
    pooled = {
        "mode3_ndcg@10": round(sum(p["m3_ndcg"] for p in per_domain.values()) / nq, 4),
        "mode3_recall@10": round(sum(p["m3_recall"] for p in per_domain.values()) / nq, 4),
        "oracle_ndcg@10": round(sum(p["oracle_ndcg"] for p in per_domain.values()) / nq, 4),
        "oracle_recall@10": round(sum(p["oracle_recall"] for p in per_domain.values()) / nq, 4),
    }
    acc = (nq - misrouted) / nq

    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run": "mode3_router_filter",
        "embedder": "embeddinggemma-2 bf16 (mode 1 embeddings, reused)",
        "router": "clef-flash Q8 zero-shot",
        "n_eval_queries": nq,
        "routing_accuracy": round(acc, 4),
        "n_misrouted": misrouted,
        "per_domain": out,
        "pooled": pooled,
        "compare": "mode1_baseline pooled ndcg@10 0.4503 recall@10 0.577",
    }
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print("=== MODE 3 SUMMARY (paste-ready) ===")
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()