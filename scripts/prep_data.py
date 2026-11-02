#!/usr/bin/env python3
"""Build the derived training/eval set from the four BEIR corpora.

Outputs (data/derived/):
  corpus.jsonl   one row per doc: {domain, did, title, text}
  queries.jsonl  one row per query: {domain, qid, query, pos_dids, split}
  stats.json     counts, dangling refs, split sizes

Rules:
  - pos_dids come from ALL qrels splits of each domain (train+test+dev where
    present). A query's router label is its domain.
  - Global 80/20 holdout PER QUERY, keyed by a stable hash of (domain, qid),
    so the split is reproducible and identical for router and adapter work.
  - Qrels referencing a doc missing from the corpus are dropped and counted.
  - Queries left with zero valid positives after that are dropped and counted.

Usage: python scripts/prep_data.py
"""
import hashlib
import json
import os

import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(HERE, "data", "beir")
OUT = os.path.join(HERE, "data", "derived")
EVAL_FRAC = 0.2

DOMAINS = {
    "fiqa":    {"kind": "parquet", "corpus": "fiqa/corpus.parquet",
                "queries": "fiqa/queries.parquet",
                "qrels": ["fiqa/qrels_train.tsv", "fiqa/qrels_dev.tsv", "fiqa/qrels_test.tsv"]},
    "arguana": {"kind": "parquet", "corpus": "arguana/corpus.parquet",
                "queries": "arguana/queries.parquet",
                "qrels": ["arguana/qrels_test.tsv"]},
    "scifact": {"kind": "parquet", "corpus": "scifact/corpus.parquet",
                "queries": "scifact/queries.parquet",
                "qrels": ["scifact/qrels_train.tsv", "scifact/qrels_test.tsv"]},
    "gaming":  {"kind": "jsonl", "corpus": "gaming/corpus.jsonl",
                "queries": "gaming/queries.jsonl",
                "qrels": ["gaming/qrels_test.tsv"]},
}


def split_of(domain, qid):
    h = int(hashlib.sha256(f"{domain}:{qid}".encode()).hexdigest(), 16)
    return "eval" if (h % 100) < int(EVAL_FRAC * 100) else "train"


def load_corpus(spec, domain):
    if spec["kind"] == "parquet":
        t = pq.read_table(os.path.join(D, spec["corpus"]),
                          columns=["_id", "title", "text"])
        rows = t.to_pylist()
    else:
        rows = [json.loads(l) for l in open(os.path.join(D, spec["corpus"]))]
    out, seen = [], set()
    for r in rows:
        did = str(r["_id"])
        if did in seen:
            continue
        seen.add(did)
        out.append({"domain": domain, "did": did,
                    "title": r.get("title") or "", "text": r.get("text") or ""})
    return out


def load_qrels(spec):
    qrels = {}
    for rf in spec["qrels"]:
        with open(os.path.join(D, rf)) as f:
            next(f)  # header
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 3:
                    continue
                qid, did, score = parts[0], parts[1], int(parts[2])
                if score <= 0:
                    continue
                qrels.setdefault(qid, set()).add(did)
    return qrels


def main():
    os.makedirs(OUT, exist_ok=True)
    corpus, queries, stats = [], [], {}
    for domain, spec in DOMAINS.items():
        docs = load_corpus(spec, domain)
        docids = {d["did"] for d in docs}
        corpus.extend(docs)
        if spec["kind"] == "parquet":
            t = pq.read_table(os.path.join(D, spec["queries"]), columns=["_id", "text"])
            qrows = t.to_pylist()
        else:
            qrows = [json.loads(l) for l in open(os.path.join(D, spec["queries"]))]
        qrels = load_qrels(spec)
        kept = dangling = empty = 0
        for r in qrows:
            qid = str(r["_id"])  # raw id, used for qrels lookup
            qtext = (r.get("text") or "").strip()
            if not qtext:
                empty += 1
                continue
            pos = sorted(d for d in qrels.get(qid, set()) if d in docids)
            dangling += len(qrels.get(qid, set())) - len(pos)
            if not pos:
                empty += 1
                continue
            # store a globally-unique, domain-prefixed qid; split key unchanged
            queries.append({"domain": domain, "qid": f"{domain}:{qid}", "query": qtext,
                            "pos_dids": pos, "split": split_of(domain, qid)})
            kept += 1
        stats[domain] = {"docs": len(docs), "queries_kept": kept,
                         "queries_dropped": empty, "qrels_dangling": dangling,
                         "qrel_pairs": sum(len(v) for v in qrels.values())}

    with open(os.path.join(OUT, "corpus.jsonl"), "w") as f:
        for c in corpus:
            f.write(json.dumps(c) + "\n")
    with open(os.path.join(OUT, "queries.jsonl"), "w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")

    train = [q for q in queries if q["split"] == "train"]
    ev = [q for q in queries if q["split"] == "eval"]
    summary = {
        "corpus_total": len(corpus),
        "queries_total": len(queries),
        "train": {"n": len(train),
                  "by_domain": {d: sum(1 for q in train if q["domain"] == d)
                                for d in DOMAINS}},
        "eval": {"n": len(ev),
                 "by_domain": {d: sum(1 for q in ev if q["domain"] == d)
                               for d in DOMAINS}},
        "domains": stats,
    }
    with open(os.path.join(OUT, "stats.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()