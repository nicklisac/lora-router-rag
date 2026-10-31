# Plan

Status: research complete, nothing trained yet. Order of operations below is deliberate -
the cheap tests come first and each one can end the project early.

## Decisions still needed before running

- **Route taxonomy.** How many routes, what defines each. This is the load-bearing
  decision: the router can only be as good as its label set. Too coarse and specialists
  don't specialize; too fine and the router gets queries it can't place. Proposed default
  is the four-route MMDocIR partition in `research/datasets.md`.
- **Corpus partitioning.** Does each route hold a full corpus copy (routing changes only
  the space) or its own slice (routing changes the space *and* the search set)? The second
  is arm C's confound. Different experiments; probably run both.
- **Modality partitioning.** Each route can also differ in tower (text-only vs text+vision)
  and dimension (768/512/256 via MRL). Both are free knobs here, so the design space is
  routes x towers x dims. Pin one config per route or the ablation gets muddy.
- **Router implementation.** Small classifier over query text is the cheap version. Needs
  labeled queries by route - if that labeling is expensive, the whole approach gets
  expensive. Alternative: a reranker that scores the query against each route's centroid.
- **Eval.** recall@k and nDCG@10 on labeled qrels, per arm and per route. Report router
  accuracy separately from retrieval quality.

## Order of operations

1. **Flickr8k smoke test.** Prove the multimodal LoRA path trains and scores on this
   checkpoint. Fastest loop available, and it's the exact setup Unsloth's own notebook
   uses. Also gives the first real VRAM number for GPU 2.
2. **Arm D (prefix routing).** No training, no extra indexes. Router picks among the seven
   shipped task prefixes. If D captures most of B's expected gain, stop here and report
   that the LoRA suite was unnecessary.
3. **Arm C (pre-filter) on M-BEIR.** Its global vs local candidate pools make this nearly
   free. Establishes how much of any win comes from narrowing the search set.
4. **Arm A baseline** on MMDocIR pooled retrieval.
5. **Train specialists** (arm B), four routes, document-level held-out splits.
6. **Train the joint LoRA control.** One adapter, all domains, same total steps.
7. **Router** trained on a held-out slice of the same queries, never the eval set.
8. **Transfer check:** MMDocIR-trained finance adapter evaluated on SQID product queries.

## Controls the run must include

1. **Joint LoRA** - one adapter on all domains, same total steps. This is the control that
   matters most: it separates "specialization helps" from "more fine-tuning helps." If
   joint matches or beats the specialists, the router was never needed.
2. **Untouched base model** (arm A) as the floor.
3. **Document-level held-out split** per route, because TAT-DQA has 14k triplets over 163
   documents and memorization will otherwise pass as domain knowledge.
4. **Router accuracy reported separately.** A 70% router caps what B can score, and it's
   worth knowing how much of B's loss is misrouting vs bad embeddings.
5. **NanoBEIR per adapter** as the out-of-domain canary.

## Failure modes

**Misrouting is unrecoverable.** A wrong route means the query never reaches the right
index, so no downstream math fixes it. That argues for a confidence threshold with fallback
to the base index, which quietly makes B a superset of A.

**The worse version of that:** two LoRAs on one base produce *incomparable* spaces. A query
embedded under route A can't be scored against route B's index at all, and cosine values
can't be compared across routes either. So there is no score-level "I'm unsure, fall back."
Any cross-route combination must be rank-based (RRF), never score-based.

**Over-specialization.** LoRA adapters improve task fit but can degrade the general
geometry of the embedding space. Benchmark clustering/retrieval quality before and after
each adapter; SLERP-merging is the known repair.

## Cost side

Latency added by the router hop, N indexes to build and refresh, N adapters in memory. If
B beats A by 2 points and costs 4x the infra, that's a finding, not a win. MRL at 256d is
the main lever against the storage objection - price it there before arguing at 768d.

## Open questions

- Does a *query-dependent* specialist beat a *corpus-dependent* one? Routes defined by
  question type vs by document domain are different partitions with different winners.
- Router as classifier, or as a reranker scoring the query against each route's centroid?
- Does this collapse into base embedder + per-domain metadata filters with the router
  choosing filters? That's arm C with extra steps, and within noise of B it's the better
  engineering answer.
- Would fusion beat routing entirely: no router, embed with every LoRA, RRF the rankings.
  Fusion has no unrecoverable-error mode, routing does. Related to the multi-scale chunking
  + RRF pattern, pointed at this problem.