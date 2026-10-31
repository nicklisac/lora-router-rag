# NOTES — lora-embedding-router

## 2026-10-30 — idea logged, datasets and training path researched

No code written. Everything below is verification of external sources, done before any
compute is spent.

**Idea as stated by Nick:** pair a custom decision model with a suite of LoRA-finetuned
embedding models. Router reads the incoming question, picks which embedding model fits,
the query gets embedded with that route's LoRA and fetched against data pre-embedded with
the same LoRA. Compare against base-only, and against router + base-only (a pre-filter).

**What was checked and what came back:**

1. *Candidate testbeds.* Compared MMDocIR, SQID/ESCI, M-BEIR, Amazon Reviews 2023, Kaggle
   Fashion Products, LOC Flickr, Flickr8k on four requirements (partitionable corpus, real
   qrels, derivable router label, multimodal content). MMDocIR wins because its `domain`
   field is simultaneously the router label and the LoRA partition key.
2. *MMDocIR train set, measured not assumed.* Downloaded the seven
   `annotations_top1_negative/*_train.jsonl` files (19 MB) and counted. 65,319 triplets
   against 173,843 advertised on the org page. Per-domain doc counts pulled at the same
   time - that's how the TAT-DQA memorization risk (14k triplets / 163 docs) surfaced.
3. *Unsloth EmbeddingGemma 2 docs*, then the raw notebook cells, because the prose page
   omits that the vision tower is trainable. `finetune_vision_layers` in
   `get_peft_model` selects which towers receive adapters. Configs transcribed from cells.
4. *Model card.* EG2 is 740M and multimodal (270M text + 170M vision + 300M audio,
   selectively loadable, one 768d space). This **contradicted my first pass**, which
   assumed EG2 was text-only and proposed OCR/VLM-caption workarounds. Corrected in
   `research/` and the ArxivQA route is back in - it was 74% visual queries and dead under
   a text-only embedder.
5. *GGUF availability.* `unsloth/embeddinggemma-2-GGUF` exists with Q4-Q8 + mmproj files,
   so trained routes serve from llama.cpp on GPU 2 with the existing local stack.
6. *Eval set access.* `MMDocIR-Evaluation-Dataset` returns auth errors without a token and
   accepted terms. Train set is open. First setup friction, not yet resolved.

**Design changes that came out of the research, not out of the original sketch:**

- Arm D added (task-prefix routing). Seven prefixes ship with the checkpoint, so it costs
  zero training and zero indexes. It's the cheapest way to kill the whole idea and runs
  before any adapter is trained.
- Cross-route scoring is impossible, not merely degraded. Two LoRAs on one base give
  incomparable spaces. Score-level fallback is off the table; only rank fusion works.
- The joint-LoRA control (one adapter, all domains, same steps) was missing from the
  original three arms. Without it, "specialization helps" and "more finetuning helps" are
  the same measurement.
- MRL at 256d reframes the N-indexes cost objection.

**Known unknowns, both cheap to close:** multimodal LoRA VRAM (nothing published; read peak
reserved off the Flickr8k smoke test), and whether a shared projection layer keeps per-route
spaces loosely aligned.

**Files in the KB, not here:** the idea record and its graph edges live in
`knowledge/wiki/Idea_LoRA_Embedding_Router.md`.