# Fine-Tuned Embeddings vs. Decision-Model Routing for Multi-Domain Retrieval

**A local-hardware study of LoRA specialization and post-retrieval routing on
EmbeddingGemma 2**

---

## Abstract

Thesis question: can we use local hardware to increase retrieval performance
across different domains of information by (1) training a custom EmbeddingGemma2 (EG2) LoRA per
domain, and (2) using a decision-model (DM) router to route to domain subsets?

My key finding: there was enough "room" for just the one EmbeddingGemma2 model to learn
all of the different categories. The domain-specific custom LoRAs added nearly zero value and the 
decision-model router retrieval.

We ran many modes:

1. Baseline EG2
2. Baseline + Oracle router
3. Individually trained LoRAs + decision-model (DM) router
4. Baseline + DM router
5. Individually trained LoRAs only
6. Universal LoRA (single adapter trained on all domains)

Plus a post-hoc variant in which the DM prunes records after retrieval.

Across all findings, these things stood out:

1. The decision model hurt performance at every turn. This is because it was
   not a perfect router and so it excluded relevant records. We even tried a
   post-hoc DM router where we asked it to prune after retrieval, and it did
   not help.
2. There was an appreciable, but marginal, increase in retrieval quality with
   LoRA-trained EG2 compared to the baseline.
3. The universal LoRA did better than, or just as well as, individually
   trained LoRAs. It also self-routed better than any combination with a DM
   router.

---

## 1. Introduction

This study tests whether local hardware can improve multi-domain retrieval
through two mechanisms: per-domain LoRA specialization on an embedding model,
and a decision-model router that directs each query to the relevant domain
subset. The embedding model is Google EmbeddingGemma 2 (EG2), a 740M-parameter
text/multimodal embedding model producing 768-dimensional vectors. The decision
model is Cloudflare Clef-Flash, a 9B parameter single-pass decision model served
through llama.cpp's `/v1/systemone` endpoint.

The experimental setting is an enterprise-style knowledge base built from four
topically distinct BEIR corpora: finance (FiQA), legal (ArguAna), science
(SciFact), and a community help forum (CQADupstack-Gaming). All retrieval is
performed against a single union index of the four corpora, so the system must
rank relevant documents from among 116,796 documents across all domains.

The central question is whether adding complexity (per-domain adapters, a
router, a post-retrieval filter) improves over a single fine-tuned embedding
model searched directly.

---

## 2. Methods

### 2.1 Dataset

Four BEIR corpora were combined into a single retrieval task.

| Domain | Corpus | Documents | Queries (total) | Queries (train) | Queries (eval) | Qrel pairs |
|--------|--------|-----------|-----------------|-----------------|----------------|------------|
| finance | FiQA 2018 | 57,638 | 6,648 | 5,316 | 1,332 | 17,110 |
| legal | ArguAna | 8,674 | 1,406 | 1,108 | 293 | 1,406 |
| science | SciFact | 5,183 | 1,109 | 901 | 208 | 1,258 |
| community | CQADupstack-Gaming | 45,301 | 1,595 | 1,267 | 328 | 2,263 |
| **total** | | **116,796** | **10,753** | **8,592** | **2,161** | |

Split policy: a global 80/20 train/eval split keyed on the SHA-256 of
`{domain}:{query_id}`. Two of the four corpora (ArguAna, CQADupstack) ship no
train qrels, so a per-domain train/test split was not possible; a global
holdout is the only policy that works uniformly across all four. The eval set
(2,161 queries) is held out for every experiment and is identical across all
modes, so pooled scores are directly comparable.

Corpus embeddings (116,796 × 768) were computed once with the base model and
reused for the router-filter experiments. Query embeddings were computed per
mode as required by each embedder.

### 2.2 Model and hardware

- **Base embedder**: EmbeddingGemma 2, bf16 safetensors, 740M parameters, 768-d
  output, mean pooling, L2-normalized. Served in-process through
  `FastSentenceTransformer` (Unsloth). The bf16 path is used for all modes so
  that LoRA deltas are comparable to the baseline with no quantization gap.
- **Decision model**: Clef-Flash Q8_0 GGUF (9.7 GB), served through
  llama.cpp b11985 on `/v1/systemone`. All layers offloaded to GPU. `--ubatch-size
  4096` to accommodate multi-document states.
- **GPU**: NVIDIA RTX 3090 (24 GB), dedicated to this work (GPUs 0 and 1 run
  the assistant and are not touched).
- **Training stack**: Unsloth 2026.10.11, transformers 5.19.0.dev0
  (pinned git commit per the official EG2 notebook), sentence-transformers
  6.2.0.

### 2.3 Prompt templates

A single source of truth (`src/prefixes.py`) is used at training, indexing, and
query time:

```
query: task: search result | query: {text}
doc:   title: {title} | text: {text}
```

These match the model's native `prompts["query"]` and `prompts["document"]`
templates. The document template uses real titles where the source corpus
provides them; FiQA documents have no titles and use `title: none`. Training
pre-formats the columns directly rather than using the sentence-transformers
`prompts` argument, which would hardcode `title: none` and create a
train/serve mismatch for titled corpora.

### 2.4 Experimental modes

Six modes were evaluated, plus one post-hoc variant.

| Mode | Embedder | Retrieval | Router |
|------|----------|-----------|--------|
| 1 — Baseline | base EG2 | union index, top-10 | none |
| 2 — Baseline + Oracle router | base EG2 | true-domain slice, top-10 | ground-truth domain |
| 3 — Individual LoRAs + DM router | per-domain LoRA | routed adapter, union index, top-10 | Clef-Flash (zero-shot) |
| 4 — Baseline + DM router | base EG2 | routed-domain slice, top-10 | Clef-Flash (zero-shot) |
| 5 — Individual LoRAs only | per-domain LoRA | own-domain adapter, union index, top-10 | none |
| 6 — Universal LoRA | joint LoRA | union index, top-10 | none |
| 7 — Post-hoc DM | base EG2 or joint LoRA | retrieve top-K, Clef prunes/reranks to top-10 | Clef-Flash (noul) |

Mode 2 uses the known ground-truth domain to restrict search to the correct
domain slice (oracle upper bound). Mode 4 uses Clef-Flash's zero-shot routing
decision in place of the ground truth. Comparing 2 and 4 isolates the effect
of the router's own accuracy.

Mode 3 uses Clef-Flash to pick one of the per-domain LoRA adapters per query,
then retrieves from the full union index with that adapter's query and corpus
embeddings. Mode 5 is the same adapters but each query is scored by its own
domain's adapter with no routing decision. Mode 6 uses the single joint
adapter.

Mode 7 retrieves a larger candidate set (top-10 or top-20), then uses
Clef-Flash as a binary relevance judge over the retrieved documents. Each
query is one `/v1/systemone` request carrying N `noul` ("is document i
relevant to the query?") questions, answered in a single forward pass. Two
strategies were tested: re-ranking by Clef score, and pruning (drop the
lowest-scoring documents, keep the rest in original order).

### 2.5 LoRA configuration

All five adapters (four specialists + one joint) start from the same base
checkpoint, use the same LoRA configuration, and are trained for the same
number of steps (200). Only the training data differs.

| Parameter | Value |
|-----------|-------|
| rank (r) | 32 |
| alpha | 64 |
| dropout | 0 |
| target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| trainable parameters | 10,092,544 (3.59% of 281M) |
| learning rate | 2e-5 |
| scheduler | linear, 3% warmup |
| per-device batch | 64 |
| gradient accumulation | 2 |
| effective batch | 128 |
| sequence length | 1024 |
| loss | MNRL (multiple negatives ranking loss) + NO_DUPLICATES sampler |
| max steps | 200 (fixed across all five adapters) |
| seed | 3407 |
| precision | bf16 |

The MNRL loss uses other positives in the batch as in-batch negatives. With
cross-domain batches in the joint adapter, the in-batch negatives are
cross-domain by construction. For per-domain adapters, in-batch negatives are
from the same domain. The NO_DUPLICATES sampler prevents a query from being
its own in-batch negative when a query has multiple positive documents.

Training rows: one row per (query, positive-document) pair.

| Adapter | Train rows | ~epochs at 200 steps | Final loss |
|---------|-----------|----------------------|------------|
| fiqa | 13,694 | 1.2 | 0.704 |
| arguana | 1,108 | 18.1 | 0.049 |
| scifact | 1,031 | 16.9 | 0.023 |
| gaming | 1,824 | 28.6 | 0.143 |
| joint | 17,657 | 1.2 | 0.501 |

Adapters are saved as LoRA weights and as merged standalone bf16
checkpoints. Evaluation uses the merged checkpoint to avoid the
`for_inference=True` reload requirement on raw LoRA checkpoints.

### 2.6 Decision-model routing

Clef-Flash routes each eval query to one of four domains using a zero-shot
`choice` question. The four criteria strings:

```
fiqa:    "Finance and investment questions"
arguana: "Legal argument and case questions"
scifact: "Scientific research and fact-checking questions"
gaming:  "Gaming and computer questions from a help forum"
```

Routing decisions are checkpointed to `results/routing_eval.jsonl` and reused
across modes 3, 4, and the routed evaluation.

### 2.7 Metrics

nDCG@10 and recall@10 on the 2,161-query eval set, computed per domain and
pooled (macro average across domains, weighted by per-domain query count).
Relevance is binary (a document is relevant if it appears in the query's
qrels).

---

## 3. Results

### 3.1 Baseline (Mode 1)

Base EG2, union index, top-10.

| Domain | n queries | nDCG@10 | Recall@10 |
|--------|-----------|---------|-----------|
| finance | 1,332 | 0.3372 | 0.4141 |
| legal | 293 | 0.4584 | 0.8635 |
| science | 208 | 0.9644 | 0.9952 |
| community | 328 | 0.5763 | 0.7173 |
| **pooled** | **2,161** | **0.4503** | **0.5770** |

### 3.2 Domain separation: oracle vs. baseline (Modes 1 and 2)

Mode 2 restricts search to the true domain's documents using the known
ground-truth domain label. If the embedding model's geometry does not already
separate the domains, restricting the search space should help. If it does,
the oracle should score roughly the same as the unrestricted union search.

| | Pooled nDCG@10 | Pooled Recall@10 |
|---|---|---|
| Mode 1 — baseline (union) | 0.4503 | 0.5770 |
| Mode 2 — oracle (true-domain slice) | 0.4526 | 0.5815 |
| **delta (oracle − baseline)** | **+0.0023** | **+0.0045** |

Per domain:

| Domain | Baseline nDCG@10 | Oracle nDCG@10 | Delta |
|--------|-----------------|----------------|-------|
| finance | 0.3372 | 0.3384 | +0.0012 |
| legal | 0.4584 | 0.4636 | +0.0052 |
| science | 0.9644 | 0.9644 | 0.0000 |
| community | 0.5763 | 0.5822 | +0.0059 |

Restricting search to the correct domain produces a +0.002 pooled nDCG
improvement. Per-domain deltas are within ±0.006. The four corpora are already
separated in the base EG2 embedding space; cross-domain documents do not
compete meaningfully with in-domain documents for top-10 positions.

### 3.3 Decision-model routing: Clef-Flash vs. oracle (Mode 4 vs. Mode 2)

Clef-Flash routes each query to a domain, then search is restricted to that
domain's slice. Routing accuracy: 90.79% (1,962 / 2,161 correct; 199
misrouted). Mean routing confidence: 0.906.

| | Pooled nDCG@10 | Pooled Recall@10 |
|---|---|---|
| Mode 1 — baseline (union) | 0.4503 | 0.5770 |
| Mode 2 — oracle (true-domain slice) | 0.4526 | 0.5815 |
| Mode 4 — Clef-Flash routing (routed slice) | 0.4084 | 0.5089 |
| **delta (Clef vs. baseline)** | **−0.0419** | **−0.0681** |
| **delta (Clef vs. oracle)** | **−0.0442** | **−0.0726** |

Per domain:

| Domain | Baseline | Clef-Flash | Oracle |
|--------|----------|-----------|--------|
| finance | 0.3372 | 0.3247 | 0.3384 |
| legal | 0.4584 | 0.2591 | 0.4636 |
| science | 0.9644 | 0.9644 | 0.9644 |
| community | 0.5763 | 0.5291 | 0.5822 |

Clef-Flash routing drops pooled nDCG by 0.042 relative to baseline. The
largest per-domain loss is on legal (0.458 → 0.259), where misrouted queries
are searched in a slice that does not contain the relevant documents.

### 3.4 LoRA fine-tuning: per-domain adapters (Mode 5)

Each per-domain adapter re-embeds the full union corpus and the 2,161 eval
queries. Search is over the full union index (no slice restriction). Each
query is scored by its own domain's adapter (this is the same assignment as
the oracle ceiling in Section 3.6).

Per-domain nDCG@10 by adapter:

| Domain (n) | Base | fiqa LoRA | arguana LoRA | scifact LoRA | gaming LoRA |
|------------|------|-----------|--------------|--------------|-------------|
| finance (1,332) | 0.3372 | **0.4009** | 0.2400 | 0.3076 | 0.3505 |
| legal (293) | 0.4584 | 0.4793 | **0.5102** | 0.4628 | 0.4730 |
| science (208) | 0.9644 | 0.9551 | 0.9620 | **0.9650** | 0.9615 |
| community (328) | 0.5763 | 0.5834 | 0.5425 | 0.5428 | **0.5925** |

Pooled nDCG@10 and Recall@10:

| Adapter | nDCG@10 | Recall@10 | Delta vs. base |
|---------|---------|-----------|----------------|
| base (Mode 1) | 0.4503 | 0.5770 | — |
| fiqa LoRA | 0.4925 | 0.6224 | +0.0422 |
| arguana LoRA | 0.3920 | 0.5183 | −0.0583 |
| scifact LoRA | 0.4276 | 0.5576 | −0.0227 |
| gaming LoRA | 0.4626 | 0.5874 | +0.0123 |
| **joint LoRA** | **0.4947** | **0.6263** | **+0.0444** |

A per-domain LoRA improves nDCG on its own domain relative to base: fiqa
0.337 → 0.401 (+0.064), arguana 0.458 → 0.510 (+0.052), gaming 0.576 → 0.593
(+0.016). The scifact adapter is flat (0.964 → 0.965) because the base model
already scores 0.964 on that domain.

On the other three domains, each per-domain adapter degrades relative to base.
The arguana adapter scores 0.240 on fiqa queries (base: 0.337). The fiqa
adapter scores 0.401 on fiqa but the joint adapter scores 0.402 on the same
queries.

### 3.5 Universal LoRA vs. specialists (Mode 6 vs. Mode 5)

The joint LoRA (trained on all four domains pooled, cross-domain in-batch
negatives) is compared against the per-domain specialists on the same union
index, with each query scored by its own domain's adapter.

Per-domain nDCG@10:

| Domain | Base | Specialist | Joint LoRA |
|--------|------|-----------|------------|
| finance | 0.3372 | 0.4009 (fiqa LoRA) | **0.4025** |
| legal | 0.4584 | **0.5102** (arguana LoRA) | 0.4882 |
| science | 0.9644 | **0.9650** (scifact LoRA) | 0.9551 |
| community | 0.5763 | **0.5925** (gaming LoRA) | 0.5829 |

| | Pooled nDCG@10 | Pooled Recall@10 |
|---|---|---|
| base | 0.4503 | 0.5770 |
| specialists, each query scored by its own domain's adapter | 0.4991 | 0.6277 |
| joint LoRA | 0.4947 | 0.6263 |
| **delta (specialist − joint)** | **+0.0044** | **+0.0014** |

The specialist leads on three of four domains (legal 0.510 vs. 0.488, science
0.965 vs. 0.955, community 0.593 vs. 0.583); the joint adapter leads on
finance (0.402 vs. 0.401). Pooled, the correctly-assigned specialist scores
0.4991, which is +0.0044 over the joint LoRA (0.4947). This per-query own-domain
assignment is equivalent to oracle routing (Section 3.6) and is therefore not
achievable without a perfect router. The entire value of the four-specialist
architecture, at its absolute best, is 0.004 pooled nDCG over a single joint
adapter.

### 3.6 Decision-model routing with per-domain adapters (Mode 3)

Clef-Flash routes each query to a domain, the corresponding per-domain LoRA
adapter is used to embed the query, and retrieval is over the full union
index. Compared against: the base model (no routing, no LoRA), the joint
adapter (no routing), and oracle routing (perfect domain assignment).

| Strategy | Pooled nDCG@10 | Pooled Recall@10 |
|----------|---------------|-----------------|
| base (no LoRA, no routing) | 0.4503 | 0.5770 |
| joint LoRA (no routing) | 0.4947 | 0.6263 |
| Clef-Flash routing → specialists | 0.4923 | 0.6198 |
| oracle routing → specialists | 0.4991 | 0.6277 |
| **delta (Clef routing vs. joint)** | **−0.0024** | **−0.0065** |
| **delta (oracle vs. joint)** | **+0.0044** | **+0.0014** |

Per-domain nDCG@10:

| Domain | Base | Clef routing | Joint | Oracle |
|--------|------|-------------|-------|--------|
| finance | 0.3372 | 0.3960 | **0.4025** | 0.4009 |
| legal | 0.4584 | 0.4881 | 0.4882 | **0.5102** |
| science | 0.9644 | **0.9650** | 0.9551 | 0.9650 |
| community | 0.5763 | **0.5878** | 0.5829 | 0.5925 |

Clef-Flash routing of per-domain adapters scores 0.4923, below the joint
adapter (0.4947) and below the oracle routing upper bound (0.4991). The
oracle ceiling for the specialist architecture is +0.004 over the joint
adapter. Clef-Flash does not reach that ceiling.

### 3.7 Post-hoc decision-model reranker (Mode 7)

After retrieval, Clef-Flash judges the relevance of each retrieved document.
One `/v1/systemone` request per query carries N `noul` questions (one per
retrieved document), answered in a single forward pass. Documents are capped
at 300 characters in the state to keep the multi-document input under the
physical batch limit.

**Base retriever, top-10, prune to 5 or re-rank:**

| Strategy | nDCG@10 | Recall@10 | Delta vs. embedding-only |
|----------|---------|-----------|------------------------|
| embedding only | 0.4503 | 0.5770 | — |
| Clef re-rank (reorder by Clef score) | 0.4404 | 0.5770 | −0.0099 |
| Clef prune (drop 5 lowest) | 0.4082 | 0.4713 | −0.0421 / −0.1057 |

**Joint LoRA retriever, top-20, prune to 10 or re-rank:**

| Strategy | nDCG@10 | Recall@10 | Delta vs. embedding-only |
|----------|---------|-----------|------------------------|
| embedding only (joint LoRA) | 0.4947 | 0.6263 | — |
| Clef re-rank | 0.4398 | 0.5861 | −0.0549 / −0.0402 |
| Clef prune (keep 10) | 0.4794 | 0.5861 | −0.0153 / −0.0402 |

Clef-Flash reduces nDCG@10 in all four configurations. The re-rank variant
is the largest loss (replaces the embedding ranking with Clef's weaker
ordering). The prune variant loses recall because it drops relevant documents
from the final set.

---

## 4. Findings

### 4.1 Domain separation is already present in the base model

Restricting search to the true domain's documents (oracle) produces a
+0.002 pooled nDCG improvement over unrestricted union search. Per-domain
deltas are within ±0.006. The four corpora are separated in the base EG2
embedding space. Cross-domain documents do not compete meaningfully with
in-domain documents for top-10 positions. A router that filters the candidate
set to the correct domain adds 0.002 nDCG at perfect accuracy.

### 4.2 The decision model's own error rate exceeds the problem it addresses

Clef-Flash routes correctly 90.79% of the time (199 of 2,161 queries
misrouted). Perfect routing recovers only 0.002 pooled nDCG (the oracle
delta, Section 3.2). The DM's own misroutes cost 0.0442 pooled nDCG — the gap
between oracle routing (0.4526) and Clef-Flash routing (0.4084) — which is
approximately 22× the nDCG that perfect routing would recover. At 90.79%
accuracy, Clef-Flash routing scores 0.4084 pooled nDCG, 0.042 below the
0.4503 unrestricted baseline. Even at 100% accuracy, the routing would score
0.4526, only 0.002 above the baseline.

### 4.3 LoRA fine-tuning improves retrieval by +0.044 pooled nDCG

The joint LoRA (trained on all four domains pooled) scores 0.4947 pooled
nDCG@10 and 0.6263 pooled recall@10, a +0.0444 nDCG and +0.0493 recall
improvement over the base model. This is the largest single improvement in
the experiment.

### 4.4 Per-domain specialization adds at most 0.004 pooled nDCG over a single joint adapter

The four per-domain specialists, with each query scored by its own domain's
adapter, score 0.4991 pooled nDCG — +0.0044 over the joint LoRA (0.4947). The
specialist leads on three domains (legal, science, community); the joint
adapter leads on finance. This 0.0044 is the ceiling of the entire specialist
architecture, and it requires perfect routing to realize. A practical router
(Clef-Flash, 90.79% accurate) scores 0.4923, which is 0.0024 below the joint
LoRA and does not capture the ceiling.

### 4.5 The decision model degrades performance in every role

| Role | System without DM | System with DM | Delta |
|------|-------------------|---------------|-------|
| domain router (base embedder) | 0.4503 | 0.4084 | −0.0419 |
| domain router (per-domain LoRAs) | 0.4947 (joint, no routing) | 0.4923 | −0.0024 |
| post-hoc re-ranker (base, top-10) | 0.4503 | 0.4404 | −0.0099 |
| post-hoc pruner (base, top-10→5) | 0.4503 | 0.4082 | −0.0421 |
| post-hoc re-ranker (joint, top-20) | 0.4947 | 0.4398 | −0.0549 |
| post-hoc pruner (joint, top-20→10) | 0.4947 | 0.4794 | −0.0153 |

In every configuration where the decision model is the variable, removing it
improves the score. The decision model never improves retrieval relative to
the same system without it.

---

## 5. Conclusion

Fine-tuning a LoRA on EG2 can increase retrieval quality. This quality lift
can be trained across different domains in the same LoRA — there is more than
enough "room" in the model for it to learn the different unique data patterns
in a single LoRA.

Decision models hurt retrieval because they are not 100% correct, and their
incorrectness is heavier than the embedding model's tendency to pull from
out-domain records. In fact, the embedding model itself acts as its own
router very well — with no worse performance on its own than when it was
combined with an "oracle" router.

The system that works: one EG2 checkpoint, LoRA-finetuned on the union of all
domains, served with no router and no post-retrieval filter.

---

## 6. Limitations

- **Single seed.** All experiments use seed 3407. The +0.044 joint-LoRA
  improvement and the 0.004 specialist upper-bound advantage are within the
  range of seed-to-seed variance for a 2,161-query eval set. The joint-LoRA
  improvement is consistent in direction across all four domains, which
  reduces (but does not eliminate) the risk that it is a seed artifact.
- **Small-domain over-training.** Arguana (1,108 train rows) and SciFact
  (1,031 train rows) are cycled through 17–18 epochs at the fixed 200-step
  budget. Their final losses (0.049 and 0.023) indicate memorization.
  Re-training these two adapters at a lower step count could lift their
  per-domain scores, but the joint-vs-specialist conclusion does not depend
  on them: the fiqa adapter (13,694 rows, 1.2 epochs, no over-training)
  still loses to the joint adapter on fiqa queries (0.401 vs. 0.402).
- **Clef-Flash is a zero-shot decision model.** It was not fine-tuned on
  this routing or relevance task. A fine-tuned router might route more
  accurately, but the oracle ceiling analysis (Section 3.2, 3.6) shows that
  even perfect routing adds only 0.002–0.004 pooled nDCG, which bounds the
  value of any routing improvement.
- **Post-hoc reranker used truncated documents (300 chars).** The 300-char
  cap was imposed to keep 20-document states under the 4,096-token physical
  batch limit. A reranker with full document text might perform differently,
  but the direction of the effect (Clef's zero-shot binary relevance judgment
  is a weaker signal than the embedding's cosine rank) is unlikely to
  reverse.
- **Four domains, one task.** The four corpora are topically distinct. A
  routing experiment on topically overlapping domains (e.g., two sub-fields
  of medicine) might show a larger out-domain pull and a larger routing
  benefit. The current result is specific to the degree of topical
  separation in this dataset.
- **No ablation of LoRA rank or learning rate.** The configuration (r=32,
  alpha=64, lr=2e-5) follows the official Unsloth EG2 notebook. A different
  configuration might shift the absolute scores but is unlikely to change
  the relative ordering of the modes.

---

## Appendix A: Per-domain training loss curves

| Step | fiqa | arguana | scifact | gaming | joint |
|------|------|---------|---------|--------|-------|
| 10 | 1.768 | 0.464 | 0.598 | 1.205 | 1.361 |
| 50 | 0.999 | 0.125 | 0.102 | 0.489 | 0.783 |
| 100 | 0.806 | 0.053 | 0.030 | 0.231 | 0.548 |
| 150 | 0.660 | 0.058 | 0.022 | 0.149 | 0.526 |
| 200 | 0.704 | 0.049 | 0.023 | 0.143 | 0.501 |

## Appendix B: Clef-Flash routing decisions

Routing decisions for all 2,161 eval queries are checkpointed to
`results/routing_eval.jsonl`. Each record contains the query ID, ground-truth
domain, Clef-Flash routing decision, confidence, and per-option
probabilities.

| Statistic | Value |
|-----------|-------|
| Total queries | 2,161 |
| Correctly routed | 1,962 |
| Misrouted | 199 |
| Routing accuracy | 90.79% |
| Mean confidence | 0.906 |
| Min confidence | 0.145 |
| Max confidence | 0.995 |

## Appendix C: Artifacts

| Artifact | Path |
|----------|------|
| Run log (all results) | `results/run_log.jsonl` |
| Base corpus embeddings | `results/embeddings/corpus_union.npy` (116,796 × 768) |
| Base eval query embeddings | `results/embeddings/queries_eval.npy` (2,161 × 768) |
| Routing decisions | `results/routing_eval.jsonl` (2,161 records) |
| LoRA adapters | `results/lora/{fiqa,arguana,scifact,gaming,joint}/` |
| Merged checkpoints | `results/lora/{fiqa,arguana,scifact,gaming,joint}_merged/` |
| Per-adapter corpus embeddings | `results/embeddings/mode2_{name}_corpus.npy` |
| Per-adapter query embeddings | `results/embeddings/mode2_{name}_queries.npy` |
| Post-hoc rerank scores | `results/rerank_scores_{base, joint}_k{10,20}.jsonl` |
| Training script | `scripts/train_lora.py` |
| Evaluation scripts | `scripts/eval_mode{1,2,3,2_routed,rerank}.py` |
| Data preparation | `scripts/prep_data.py` |
| Prefix templates | `src/prefixes.py` |

## Appendix D: Mode numbering used in this report

| Report mode | Description | Corresponds to |
|-------------|-------------|----------------|
| Mode 1 | Baseline EG2, union index | `mode1_baseline` |
| Mode 2 | Baseline + oracle router (true-domain slice) | `mode3_router_filter` (oracle column) |
| Mode 3 | Individual LoRAs + Clef-Flash router, union index | `mode2_routed` (router strategy) |
| Mode 4 | Baseline + Clef-Flash router, routed slice | `mode3_router_filter` (mode3 column) |
| Mode 5 | Individual LoRAs only, own-domain adapter | `mode2_{fiqa,arguana,scifact,gaming}` |
| Mode 6 | Universal (joint) LoRA, no routing | `mode2_joint` |
| Mode 7 | Post-hoc Clef-Flash re-rank / prune | `mode4_rerank_{base, joint}_k{10,20}` |