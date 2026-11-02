# LoRA Embedding Router

An experiment in multi-domain retrieval: does finetuning the embedder help a
knowledge base of mixed domains, and does a decision-model router on top help
further?

**Short answer:** finetuning the embedder is the real lever. A joint LoRA over all
domains improved pooled nDCG@10 by +0.044 over the base model. A decision-model
router added nothing: the base embeddings already separate the domains, so a
top-N union search routes on its own.

Full methods, results, and analysis are in [REPORT.md](REPORT.md). Every reported
number traces to an entry in [`results/run_log.jsonl`](results/run_log.jsonl).

## What was tested

Six arms, all evaluated on the same 1,081 held-out queries:

| Arm | Embedder | Router | Result |
|---|---|---|---|
| Base | EmbeddingGemma 2 | none | nDCG@10 0.4503 |
| Specialists + DM | one LoRA per domain | Clef-Flash | 0.4923 |
| Base + DM | EmbeddingGemma 2 | Clef-Flash | 0.4084 |
| Specialists only | one LoRA per domain | self-routing (top-N) | 0.4991 (oracle ceiling) |
| **Joint LoRA** | **one LoRA, all domains** | self-routing | **0.4947** (best deployable) |
| DM rerank/prune | base or joint | Clef-Flash, post-hoc | worse than base |

## Setup

- **Embedder:** [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2)
  (740M, 768-d, Apache 2.0). Trained via Unsloth on the
  [`unsloth/embeddinggemma-2`](https://huggingface.co/unsloth/embeddinggemma-2) mirror.
  Served through llama.cpp.
- **Router:** [`Cloudflare/clef-flash`](https://huggingface.co/Cloudflare/clef-flash)
  (9B decision model, Apache 2.0). Given a query and a `choice` question with the
  domains as options, it returns a probability per option in one forward pass. No
  router training was needed.
- **Data:** four BEIR text-to-text corpora standing in for an enterprise knowledge
  base — finance (FiQA), legal (ArguAna), science (SciFact), and community Q&A
  (CQADupstack-gaming). Open queries and qrels. See [PLAN.md](PLAN.md) for sizes
  and the split policy.

One RTX 3090 (24 GB) covers training and serving. The base model fits in ~1.5 GB
bf16; each LoRA trains in a few minutes.

## Reproduction

```bash
bash scripts/download.sh        # corpora (4 BEIR datasets) + models
bash scripts/start_servers.sh   # EG2 on :18124, Clef-Flash on :18125
python scripts/prep_data.py     # train/val/test splits -> data/prepared/
bash scripts/run_all_train.sh   # 5 LoRA adapters (finance, legal, science, community, joint)
bash scripts/run_all_eval.sh    # per-domain and pooled eval for every arm
```

Individual eval scripts (`scripts/eval_*.py`) can be run on their own. Routing
decisions for the four domains live in [`results/routing_eval.jsonl`](results/routing_eval.jsonl).

## Layout

```
REPORT.md                  the report: abstract, methods, results, findings, conclusion
PLAN.md                    the plan and setup constraints
NOTES.md                   running log of what was verified and when
research/                  source-checked notes on datasets and the models
scripts/                   download, server, prep, training, and eval scripts
src/                       shared prompt prefixes (single source of truth)
data/                      downloads (gitignored)
results/                   run outputs (gitignored except run_log.jsonl, routing_eval.jsonl)
```

## Constraints

1. Local and public Hugging Face only — no gated datasets.
2. Every reported number traces to an entry in `results/run_log.jsonl`.
3. EmbeddingGemma 2 is multimodal (text, vision, audio) into one space; this
   experiment uses the text path only.