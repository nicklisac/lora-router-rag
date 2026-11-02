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

- Cross-route scoring is impossible, not merely degraded. Two LoRAs on one base give
  incomparable spaces. Score-level fallback is off the table; only rank fusion works.
- The joint-LoRA control (one adapter, all domains, same steps) was missing from the
  original arms. Without it, "specialization helps" and "more finetuning helps" are
  the same measurement.
- MRL at 256d reframes the N-indexes cost objection. (Follow-up, not first run.)

**Known unknowns:** multimodal LoRA VRAM (nothing published; read peak reserved off the
first smoke test), and whether a shared projection layer keeps per-route spaces loosely
aligned.

## 2026-10-30 — replanned around three modes

Nick cut the plan back: the hypothesis is specialists-plus-router versus base-only
versus base-plus-filter, nothing more until those numbers exist. PLAN.md and README.md
rewritten to match. `research/` kept as source-checked reference.

Also verified today, against the machine and the model card:

- Environment is empty: no torch or sentence_transformers, `src/` and `scripts/` have
  no files. Only the docs are committed.
- GPU 2 is not free. Rooster's process holds memory on all three cards; GPU 2 has
  ~1.5 GB of 24 GB free right now. Any real run needs a window.
- Router decision: decision models are a model class (Jev defined the API; the model
  answers typed questions with per-option probabilities in one forward pass, no text
  generation). Cloudflare released open-weight ones on 2026-10-01: `Cloudflare/clef`
  (27B) and `Cloudflare/clef-flash` (9B, Qwen3.5-9B backbone, multimodal, Apache 2.0,
  Jev-compatible). The repo ships `joint_schema_model.py` with `load_release_model`, so
  it runs in-process from transformers; tested with torch 2.11 + transformers 5.10.2.
  BF16 weights are ~18 GB, 4-bit ~5-9 GB. Picking clef-flash: it can route zero-shot,
  so the router costs nothing at first and router quality is measured, not assumed.
  Its backbone being Qwen (not Gemma) means no shared weights with EG2, which is fine
  and keeps router and embedder errors independent.
- Cheap floor for the router anyway: logistic regression on base embeddings. If a
  linear probe beats a 9B decision model, that is worth knowing and worth using.

## 2026-10-30 — llama.cpp updated to b11985, Clef-Flash GGUF verified locally

- Old build: Unsloth prebuilt b10909 (cuda13-older, installed 2026-09-13) at
  `~/llama.cpp-p2p`, P2P env wrapper at `~/llama.cpp-p2p/llama-server`. Decision-model
  support (`/v1/systemone`) merged upstream ~2026-10-02, after b10909 was built, so the
  old build could not serve Clef.
- New build: Unsloth prebuilt b11985-mix-c116c32 (released 2026-10-30, same
  cuda13-older bundle line), extracted to `~/llama.cpp-p2p/b11985/`. Wrapper repointed
  at it; old wrapper saved as `llama-server.bak-b10909`, old binaries still in
  `~/llama.cpp-p2p/build/`. Verified: version banner b11985, server logs "decision
  model type: clef", `/v1/systemone` route present.
- Model: `ggml-org/Clef-Flash-GGUF` Q8_0 (9.7 GB) at `~/models/`, official ggml-org
  conversion, carries `clef.decision.type=clef` metadata. ggml-org also ships Q4_K_M
  and BF16 plus mmproj files for vision. No conversion needed.
- End-to-end smoke test on CPU passed: served Q8_0, posted a choice question to
  /v1/systemone, got `billing` at p=0.93 with confidence 0.89. So the whole router
  path works without torch or transformers; the Q8 output-matrix caveat is moot for
  this GGUF (ggml-org conversion handles it).
- GPU test not done: GPU 2 was occupied. `llama-quantize` symlink still points at the
  old b10909 build.
- Disk: home partition at 88% (22 GB free) after adding the GGUF. EG2 GGUFs and
  dataset downloads will need space; watch this.

## 2026-10-30 — training env found (Unsloth Studio venv), llama.cpp perf scan for Qwen 3.8 27B

- Training env: Unsloth Studio venv at `/mnt/Data/unsloth/studio/unsloth_studio/`
  (python 3.13). Contains unsloth 2026.9.7, torch 2.11.0+cu130, transformers 5.5.0,
  sentence-transformers 5.2.0, trl, peft, datasets, accelerate, bitsandbytes. Verified
  with CUDA_VISIBLE_DEVICES=2: torch sees the GPU, bf16 supported, and
  `from unsloth import FastSentenceTransformer` imports clean. So no new training env
  needed, with one caveat: unsloth 2026.9.7 predates EmbeddingGemma 2 (released
  10-28; newest pypi is 2026.10.11, also predates it). EG2's `vision_config` /
  `finetune_vision_layers` path likely needs unsloth from git main. Plan: clone the
  Studio venv or make a project venv on top of its interpreter with `pip install
  unsloth@git+main`, don't touch Studio's copy (Rooster may depend on it).
- llama.cpp b10909 -> b11985 perf scan, relevant to Qwen 3.8 27B (dense hybrid: GDN
  linear attention on 48 of 64 layers). CUDA commits in the window that apply to
  Ampere (SM86):
  - GDN chunked kernel (#29353, 10-14) and four-GDN-columns-per-warp (#30087, 10-08):
    the linear-attention layers. The PR's own table benchmarks Qwen3.8-27B.
  - FFN gate/up + GLU fusion in MMQ for prompt processing (#29948, 10-25), Turing+.
  - qwen MRoPE + set-rows fusion (#31025, 10-25): vision/multimodal paths.
  - MoE fusions (#31188) do NOT apply; 27B is dense.
  - Biggest single lever may be outside the window: MTP speculative decoding for
    Qwen's qwen4exp arch exists mainly in Unsloth's fork, and the b11985 prebuilt we
    just installed carries qwen4exp work (MTP heads sharing token_embd). Community
    reports 1.3-2.2x decode speedups with `--spec-type draft-mtp` on Qwen3.8 GGUFs.
  - MoE top-k fusion (#31288, 10-28) landed same day as our b11985 cut; not in it,
    and dense-ineligible anyway.

## 2026-10-30 — SQID ruled out, dataset switched to M-BEIR (text side)

Nick swapped to a 2-card EXL3 27B for the assistant; GPU 2 is now free for the
experiment, so steps mostly run directly.

- **ESCI/SQID is a dead end.** Downloaded `crossingminds/shopping-queries-image-dataset`
  (parquet only, 8956 queries / 164900 products) and inspected the columns.
  `query_features.parquet` is `query_id` + `clip_text_features` (a 512-d CLIP vector).
  `product_features.parquet` is `product_id` + `clip_text_features` + `clip_image_features`.
  No query text, no product title/description, no relevance labels. The research doc's
  "product title, description, bullets" and "up to 40 judged products" came from the
  original ESCI paper, not this repo, and were transcribed without checking the files.
  The original `amazon-science/esci-data` is also gated (auth error), like MMDocIR's
  eval. A dataset with no qrels cannot score the experiment.
- **M-BEIR (`TIGER-Lab/M-BEIR`) is the dataset.** Open, 1.5M queries, human task
  instructions, graded qrels, per-dataset local candidate pools, text + image per item.
  Text files verified by direct listing. Three domains, each a route:
  - fashion: fashion200k task0 (text->image) + fashioniq task7
  - news: visualnews task0 + edis task2 (+ nights task4 img->img, dropped for text)
  - wiki: webqa task1 (text->text)
  Eval = each task's `test` split; training = each dataset's `train` split. Router label
  = domain = domain of the relevant candidate. Row formats confirmed by byte-range peek:
  query jsonl `{qid, query_txt, query_img_path, query_modality, pos_cand_list, ...}`,
  cand jsonl `{did, txt, img_path, modality}`, qrels TSV `qid 0 did score 0`.
- **Downloaded to `data/eg2/`:** `embeddinggemma-2-Q8_0.gguf` (310 MB) +
  `mmproj-Q8_0.gguf` (555 MB) from `unsloth/embeddinggemma-2-GGUF`. Note: Q8_0 of a 740M
  model is only 310 MB, so it's a heavy quant; EG2 has no published Q8, treat a quality
  check vs BF16 as a possible follow-up if numbers look off.
- M-BEIR text download running in background (`data/mbeir_download.log`). The 170 GB
  image tarball is intentionally NOT fetched.

## 2026-10-30 — llama.cpp b11985 CUDA silently falls back to CPU; fix + mode-1 harness

- **Bug: b11985 runs on CPU unless the CUDA 13 runtime is on the library path.** The
  Unsloth `cuda13-older` bundle's `libggml-cuda.so` links `libcudart.so.13` and
  `libcublas.so.13`, which are not on the default LD path (system has CUDA 12.0 toolkit,
  driver 610). When they're missing, llama.cpp 0.8.0 does NOT error — it prints
  `llama threadpool init, n_threads = 6` and silently serves everything on CPU. Symptom:
  GPU 2 at 6 MiB / 0% while the server "works". Fix: put a dir containing both cu13
  libs on `LD_LIBRARY_PATH`. The Unsloth Studio venv ships them at
  `.../unsloth_studio/lib/python3.13/site-packages/nvidia/cu13/lib/`. `scripts/start_servers.sh`
  now finds and exports that dir automatically. Verified after fix: GPU 2 at 9.1 GB,
  ~600 docs/s embedding (was ~15-20 docs/s on CPU).
- **M-BEIR candidate pools are modality-mixed; text-only works for 4 tasks.** Counted
  embeddable records per task. Text-viable: fashion200k_task3 (61k), webqa_task1 (544k),
  edis_task2 (1.05m, text+image; embed the text), oven_task8 (335k, text+image). Image-only
  (need the 170 GB tarball, dropped for text run): fashion200k_task0, fashioniq_task7,
  visualnews_task0, nights_task4. So the three domains for a text-only run are:
  fashion = fashion200k_task3, wiki = webqa_task1, news = edis_task2.
- **Embedding path proven end to end on GPU 2.** llama.cpp serves EG2 Q8_0 with
  `--embeddings --pooling mean`; returns unit-norm 768-d vectors; a relevant doc scores
  cos 0.79 vs 0.48 for an unrelated one, with the `task:`/`title:`/`text:` prefixes.
  Clef routing also proven on a real project query: chose fashion at 0.98.
- `scripts/eval_mode1.py` is the mode-1 baseline: embeds each domain's candidate pool +
  test queries, scores recall@10 and nDCG@10 vs the test qrels, checkpoints embeddings to
  `results/embeddings/`, appends to `results/run_log.jsonl`. Running in background
  (results/mode1_all.log), ~45 min for all three domains.

## 2026-10-30 — M-BEIR dropped, dataset = 4 BEIR corpora (enterprise-KB framing)

Nick: M-BEIR's text side is too lopsided (only webqa_task1 is clean text-to-text; the
fashion tasks are image-QUERY tasks). Wanted something that feels like a real product
catalog or enterprise KB. Settled on four BEIR text-to-text corpora = four topic areas
of a shared enterprise search box, which is also the natural home for a decision-model
router (ticket-triage shape).

Downloaded to `data/beir/`, all verified:
- `BeIR/fiqa` (finance): 57,638 docs, 6,648 queries; qrels train+dev+test.
- `BeIR/arguana` (legal): 8,674 docs, 1,406 queries; qrels test only.
- `BeIR/scifact` (science): 5,183 docs, 1,109 queries; qrels train+test.
- `mteb/cqadupstack-gaming` (community): 45,301 docs, 1,595 queries; qrels test only.

Schema is `_id, title, text` everywhere. Qrels TSVs have a header row (M-BEIR's did
not, which broke a parser I'd written against the M-BEIR format). 5 arguana qrel docs
are missing from its corpus; drop those rows. Qrels join to corpus in all four cases.

Repo gotchas: `BeIR/fiqa2018` and `BeIR/cqadupstack-*` 401 even on metadata while
sibling repos work - effectively unusable without a token. Worked around with `BeIR/fiqa`
and `mteb/cqadupstack-gaming`. HF API rate-limits this box intermittently; the download
script verifies each file with a byte-range request before fetching and logs failures
for a rerun.

**Split policy change:** two of the four domains have no train split, so per-domain
train/test isn't available. Pool all (query, relevant-doc) pairs, build one global
80/20 holdout per query, use it for both router and adapter training. Eval is always
the held-out 20%. Documented in PLAN.md.

## 2026-10-30 — inspiration video checked

Prompt Engineering channel, "Training Your Own Embedding Model Is Not As Hard As You
Think" (Oct 8). Same EG2 + Unsloth LoRA path we're on. Confirms three things we already
had (MNRL + no-duplicates sampler, train/search prompt matching, the for_inference
reload trap). One new useful number: a few minutes of LoRA clearly helps the target
task "at the cost of a few points of general text search" - that's the over-
specialization tax with a ballpark, so the per-adapter NanoBEIR canary is not optional.
Doesn't cover the router, N-indexes, or the joint-LoRA control, which are what make
this experiment different.

**Files in the KB, not here:** the idea record and its graph edges live in
`knowledge/wiki/Idea_LoRA_Embedding_Router.md`.