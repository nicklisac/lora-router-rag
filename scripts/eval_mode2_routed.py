#!/usr/bin/env python3
"""Mode 2 (routed): the number that actually answers the experiment.

Each mode-2 adapter re-embedded the SAME union corpus and the SAME eval
queries (results/embeddings/mode2_<name>_{corpus,queries}.npy), so per query
we can choose WHICH adapter's query vector to dot against THAT adapter's
corpus. This lets us test routing at the adapter level, against the full
union index, without re-embedding anything:

  base      mode-1 base EG2 (the 0.4503 reference)
  joint     always the joint control adapter
  router    Clef's routing decision picks the specialist adapter (mode 3 data)
  oracle    perfect routing picks the specialist adapter

All four use the union corpus for the candidate set (no slice restriction),
so the comparison isolates "which adapter embeds best for this query" from
"which slice to search". That's the real specialization question.

Usage:
  python scripts/eval_mode2_routed.py
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys_path = os.path.join(HERE, "src")
import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

DERIVED = os.path.join(HERE, "data", "derived")
EMB_DIR = os.path.join(HERE, "results", "embeddings")
RUN_LOG = os.path.join(HERE, "results", "run_log.jsonl")
ROUTING = os.path.join(HERE, "results", "routing_eval.jsonl")
K = 10
DOMAINS = ("fiqa", "arguana", "scifact", "gaming")
ADAPTERS = DOMAINS + ("joint",)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def ndcg_at_k(ranked, rel, k=K):
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in rel)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg if idcg else 0.0


def recall_at_k(ranked, rel, k=K):
    return len(rel & set(ranked[:k])) / len(rel) if rel else 0.0


def main():
    corpus = load_jsonl(os.path.join(DERIVED, "corpus.jsonl"))
    queries = [q for q in load_jsonl(os.path.join(DERIVED, "queries.jsonl"))
               if q["split"] == "eval"]
    nq = len(queries)

    # routing decisions keyed by qid (mode 3 checkpoint)
    routed = {json.loads(l)["qid"]: json.loads(l)
              for l in open(ROUTING)}
    missing = [q["qid"] for q in queries if q["qid"] not in routed]
    assert not missing, f"no routing decision for {len(missing)} eval queries"

    # base (mode 1) embeddings
    C_base = np.load(os.path.join(EMB_DIR, "corpus_union.npy"))
    Q_base = np.load(os.path.join(EMB_DIR, "queries_eval.npy"))

    # per-adapter embeddings
    C = {}
    Q = {}
    for a in ADAPTERS:
        C[a] = np.load(os.path.join(EMB_DIR, f"mode2_{a}_corpus.npy"))
        Q[a] = np.load(os.path.join(EMB_DIR, f"mode2_{a}_queries.npy"))

    def which_adapter(strategy, q):
        if strategy == "base":
            return None  # use base model
        if strategy == "joint":
            return "joint"
        dom = routed[q["qid"]]["routed"] if strategy == "router" else q["domain"]
        return dom

    per = {s: {d: [] for d in DOMAINS} for s in
           ("base", "joint", "router", "oracle")}
    per_rcl = {s: {d: [] for d in DOMAINS} for s in per}

    for i, q in enumerate(queries):
        rel = set(q["pos_dids"])
        for s in ("base", "joint", "router", "oracle"):
            a = which_adapter(s, q)
            Cv = C_base if a is None else C[a]
            Qv = (Q_base if a is None else Q[a])[i]
            sims = Cv @ Qv
            top = np.argpartition(-sims, min(K, len(sims) - 1))[:K]
            top = top[np.argsort(-sims[top])]
            dids = [f'{corpus[j]["domain"]}:{corpus[j]["did"]}' for j in top]
            ranked = [d.split(":", 1)[1] for d in dids]
            per[s][q["domain"]].append(ndcg_at_k(ranked, rel))
            per_rcl[s][q["domain"]].append(recall_at_k(ranked, rel))
        if (i + 1) % 500 == 0:
            print(f"[score] {i + 1}/{nq}")

    def pooled(p):
        return round(sum(v for d in DOMAINS for v in p[d]) / nq, 4)

    entry = {"ts": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
             "run": "mode2_routed", "n_eval_queries": nq, "strategies": {}}
    for s in ("base", "joint", "router", "oracle"):
        entry["strategies"][s] = {
            "pooled_ndcg@10": pooled(per[s]),
            "pooled_recall@10": pooled(per_rcl[s]),
            "per_domain_ndcg@10": {d: round(float(np.mean(per[s][d])), 4)
                                   for d in DOMAINS},
        }
    # routing accuracy for reference
    acc = sum(1 for q in queries
              if routed[q["qid"]]["routed"] == q["domain"]) / nq
    entry["routing_accuracy"] = round(acc, 4)

    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print("=== MODE 2 (ROUTED) SUMMARY ===")
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()