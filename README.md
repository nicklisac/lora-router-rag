# lora-embedding-router

Test whether routing a query to a domain-specialized LoRA embedder beats using one base
embedder for everything.

The premise: one base embedding model has to serve every kind of query from one vector
space. If queries fall into recognizable types, a specialist LoRA per type should produce
a better-shaped space for that type, at the cost of needing something upstream to pick the
right one. That upstream thing is the router. This is the "many narrow encoders" bet
measured against the "one wide encoder" default.

## Pipeline under test

1. Question arrives.
2. Router classifies it and picks a route.
3. Query is embedded with that route's LoRA.
4. Fetch against the corpus pre-embedded with the *same* LoRA.

Index side and query side must use the same adapter or the vectors aren't in the same
space. Each route needs its own pre-embedded index: N routes = N adapters to train, store,
and keep in sync with corpus updates. That cost is part of what's being tested.

## Arms

```
Arm                     Embedder                        Router job
──────────────────────  ──────────────────────────────  ─────────────────────────
A. Baseline             Base model only (EG2)           none - one index
B. Specialist router    Per-route LoRA adapters         pick which adapter + index
C. Pre-filter ablation  Base model only                 pick which index subset
D. Prefix routing       Base model only                 pick which task prefix
```

Arm C makes the experiment honest: embedder held constant, router kept, so any gap between
C and B belongs to the LoRA specialization and not the routing. If C matches B the win was
just a narrower search set - a free pre-filter, no finetuning required.

Arm D is the cheap kill-shot and runs first. The checkpoint ships seven task prefixes, so
a router that picks a prefix needs no training and no extra indexes. If D captures most of
B's gain, the LoRA suite is answering a question nobody needed answered.

Base model: `google/embeddinggemma-2` (training mirror `unsloth/embeddinggemma-2`),
Apache 2.0, 740M, natively multimodal into one 768d space.

## Layout

```
PLAN.md                              arms, controls, order of operations, cost side
NOTES.md                             running log - what was verified, when, from where
research/datasets.md                 candidate testbeds + the training set, measured
research/unsloth-embeddinggemma2.md  model facts, verified LoRA configs, gotchas, serving
data/                                downloads (gitignored)
results/                             run outputs (gitignored)
```

Related KB page: `knowledge/wiki/Idea_LoRA_Embedding_Router.md` (idea record, linked to
the embeddings notes). This repo is the working copy; the KB page is the index entry.

## Constraints (carry over from embed-predict)

1. GPU 2 only. GPUs 0 and 1 run Rooster.
2. Local + public HF only.
3. GPU 2 is a PCIe eGPU: big packets slow, many small packets wedge it. Low concurrency,
   large batches, sequential transfers.
4. Keep every intermediate artifact. Log every run to `results/run_log.jsonl`. Every
   reported number traces to a run entry.