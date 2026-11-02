# Plan

Goal: produce the three numbers from the README (mode 1, 2, 3) on one dataset, with
one eval metric, and a verdict on the hypothesis. Everything else is a follow-up.

## Steps

0. Done: dataset downloaded and verified (see Dataset section below).
1. Build the pooled pair set with the global 80/20 holdout, then embed all four
   corpora with base EmbeddingGemma 2. Score nDCG@10 and recall@k on the held-out
   queries, per domain and pooled. That is mode 1, the baseline. No training.
2. Router labels are the domain of the query's relevant docs (already in the pair
   set). Train the router on the 80% train side only. Measure routing accuracy two
   ways: a logistic regression on base embeddings, and clef-flash zero-shot with a
   `choice` question. Record both; use whichever wins.
3. Mode 3: same base embeddings, router restricts the search to its chosen subset.
   If this captures most of mode 2's expected gain, stop and report that a filter
   explains the result.
4. Mode 2: train one LoRA per domain on that domain's pairs (from the 80% train
   side). Embed each corpus once per adapter, same eval. Report mode 2 against mode 1
   and mode 3, and against the oracle router (routing by the gold label) so routing
   error and embedding error stay separate.
5. Control run: one LoRA trained on the pooled pairs from all four domains, same
   total training budget as a specialist. If the pooled adapter matches the
   specialists, specialization was never the effect being measured.

Do not add multimodal queries, task prefixes, dimension truncation, score fusion, or
transfer tests until the five steps above have numbers.

## How we run

Since 2026-10-30 the assistant runs as a 2-card EXL3 27B model on GPUs 0 and 1, and
GPU 2 is free for the experiment, so most steps now run directly without the
prepare-and-unload dance. If a step still needs the assistant gone, the scripts are
written for it anyway. Scripts check their environment first, checkpoint so a stopped
run resumes, append every number to `results/run_log.jsonl`, and print a paste-ready
summary at the end.

Tooling:
- Embedding: llama.cpp (Unsloth prebuilt b11985 at `~/llama.cpp-p2p/b11985/`) serves
  `data/eg2/embeddinggemma-2-Q8_0.gguf` (310 MB) with `--embeddings`, GPU 2, ~600
  docs/s. **Gotcha: the cuda13 bundle needs `libcudart.so.13`/`libcublas.so.13` on
  `LD_LIBRARY_PATH` or it silently runs on CPU.** `scripts/start_servers.sh` finds
  them in the Studio venv's nvidia wheels.
- Routing: `~/models/Clef-Flash-Q8_0.gguf` (9.7 GB) through the same server's
  `/v1/systemone` endpoint. Verified on GPU 2 2026-10-30 (picked a fashion query at
  0.98).
- Servers: `scripts/start_servers.sh` / `stop_servers.sh` manage both on ports
  18124 (EG2) and 18125 (Clef), with pidfiles in `results/`.
- Training: Unsloth Studio venv at `/mnt/Data/unsloth/studio/unsloth_studio/`.
  Upgraded 2026-10-31 and verified with `scripts/smoke_test_training.py`
  (LoRA attaches, 3 MNRL steps run, encode is unit-norm 768d). Working versions:
  unsloth 2026.10.11, **transformers at the commit the official EG2 notebook pins
  (`92cd495`, reports 5.19.0.dev0), installed with `--no-deps`** - EG2's
  `embedding_gemma2` type needs >=5.18 but unsloth's released pin is <=5.17, so a
  plain `pip install -U` later will downgrade it and break EG2 loading. The
  notebook's install cell is the canonical recipe; follow it if the venv ever needs
  a rebuild. sentence-transformers 6.2.0, torch 2.11.0+cu130, huggingface-hub 1.33.
  Version snapshots in
  `/mnt/Data/unsloth/studio/VENV_PRE_UPGRADE.txt` and `VENV_POST_UPGRADE.txt`.
  ST 6.x API changes that bite: `MultipleNegativesRankingLoss(model)` now takes the
  model, the trainer wants a `datasets.Dataset` not a list, and use
  `model.encode()` not `model([...])`.

## Dataset (decided)

Framing: one shared search box over an enterprise knowledge base with four topic
areas maintained by different teams. The router's job is "which team's knowledge
should this query hit", the same problem as support-desk ticket triage.

Four BEIR text-to-text corpora, all open, all downloaded and verified 2026-10-30:

| Route     | Source repo                         | Corpus | Queries | Splits        |
|-----------|-------------------------------------|--------|---------|---------------|
| finance   | `BeIR/fiqa` + `BeIR/fiqa-qrels`     | 57,638 | 6,648   | train dev test|
| legal     | `BeIR/arguana` + `arguana-qrels`    | 8,674  | 1,406   | test only     |
| science   | `BeIR/scifact` + `scifact-qrels`    | 5,183  | 1,109   | train test    |
| community | `mteb/cqadupstack-gaming`           | 45,301 | 1,595   | test only     |

Verified: qrels join to corpus in every case (5 arguana qrel docs missing from the
corpus; drop those 5 rows). Qrels TSVs have a header row (M-BEIR's did not). Schema is
`_id, title, text` everywhere; gaming is jsonl with the same fields.

**Split policy (replaces the per-domain train/test assumption).** Two domains have no
train split, so a per-domain train/test split is not available. Instead: pool all
(query, relevant-doc) pairs across domains, build one global 80/20 holdout per query,
and use it for both router training and adapter training. Evaluation is always on the
held-out 20%. Weaker than the M-BEIR plan's split-disjoint splits, but the only policy
that works with all four corpora, and it keeps router and adapters honest (neither
ever sees eval queries).

Ruled out:
- **ESCI/SQID** - CLIP embeddings only, no text, no qrels (verified by inspection).
- **MMDocIR** - eval set gated on Hugging Face.
- **M-BEIR** - looked text-friendly, but the text-viable tasks are mostly
  image-query tasks; only webqa_task1 is clean text-to-text, so it can't provide
  three distinct routes.
- `BeIR/fiqa2018` and `BeIR/cqadupstack-*` return 401 even on their metadata endpoint
  while sibling repos work - effectively unusable without a token. Use `BeIR/fiqa`
  and `mteb/cqadupstack-gaming` instead.

## Open decisions

- Router fallback. If clef-flash confidence is low, route to the base index instead.
  This requires keeping a base index around, so decide it before building indexes.
- Four routes (current) vs three (drop gaming, the most casual voice). Four is the
  default; a "community help forum" category is a real one.

## What would kill the hypothesis

- Mode 3 matches mode 2. Routing helped, finetuning didn't.
- The all-data LoRA matches the specialists. More finetuning helped, specialization
  didn't.
- Routing accuracy is poor and oracle routing is required to see any gain. The
  decision model can't place queries, so mode 2 is unreachable in practice.
- Specialists win in-subset but degrade elsewhere (check on a small general
  retrieval set). Narrow adapters can damage the general space.

## Cost notes

Mode 2 costs N adapters and N indexes, kept in sync with corpus updates. Mode 1 costs
one of each. Mode 3 costs one embedder, N slices of one index, and a router call per
query. Mode 2 only makes sense if it beats mode 3 by more than that overhead is
worth, so report the gap in quality and in infrastructure on the same page.