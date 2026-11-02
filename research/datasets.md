# Datasets

## What a testbed has to provide

Without all four, the experiment can't be scored:

1. A corpus that partitions into distinct domains/categories - these become the routes.
2. Real queries with relevance ground truth (qrels) - needed to measure retrieval quality
   per arm.
3. A derivable router label per query. Cheapest version: the domain of the query's gold
   document.
4. Multimodal content, so the real-world texture is there and the vision tower gets
   exercised.

## Candidates, ranked by how much labeling comes free

### MMDocIR (Amazon, EMNLP 2025) - primary

313 long documents (~65 pages each) across 10 domains: research reports, tutorials, legal,
financial statements, government, news. 20,395 page screenshots with OCR text *and* VLM
captions per page. 1,658 expert-annotated questions with page-level and layout-level
qrels. Plus a bootstrapped training set.

The `domain` field on every doc and every question is exactly the router's label set, and
the domains have genuinely different multimodal mixes (legal is text-heavy, financial
statements are tables). Nothing to synthesize.

HF: `MMDocIR/MMDocIR-Evaluation-Dataset`, `MMDocIR/MMDocIR-Challenge`,
`MMDocIR/MMDocIR_Train_Dataset`.

Caveat: the official task retrieves pages *within* a document. A corpus-level router test
pools pages across documents, which changes difficulty but keeps the qrels valid.

### SQID / Shopping Queries ESCI (Amazon) - secondary, most production-feeling

~130k real Amazon search queries, up to 40 judged products each, labeled
Exact/Substitute/Complement/Irrelevant. Product title, description, bullets, brand. SQID
adds image URLs and precomputed CLIP embeddings for 190k products.

Product category gives the partition. The ESCI labels give graded relevance and free hard
negatives: Substitute and Complement are adversarial in a way random negatives never are
("fleece" for a sweater query). This is where a base embedder gets caught confusing
categories, which is the failure mode the specialists are supposed to fix.

HF: `crossingminds/shopping-queries-image-dataset`; base: `amazon-science/esci-data`.

**CORRECTION (2026-10-30, verified by downloading and inspecting the files):**
`crossingminds/shopping-queries-image-dataset` does NOT contain the text or labels
described above. Its full contents are:
- `product_features.parquet`: 164,900 rows of `product_id` + `clip_text_features`
  (512-dim) + `clip_image_features` (512-dim). No titles, no descriptions.
- `query_features.parquet`: 8,956 rows of `query_id` (integer) + `clip_text_features`.
  No query strings.
- `product_image_urls.parquet`, `supp_product_image_urls.parquet`: URL lookups only.
- No relevance labels, no qrels, no category field.
The title/description/judgments come from the original ESCI paper and its separate
release (`amazon-science/esci-data`), which is itself auth-gated on Hugging Face
(returns `Invalid username or password` without a token). So SQID-as-published is
CLIP vectors only and **cannot be used for this experiment**: there is nothing to
embed and nothing to score against.

### M-BEIR (TIGER-Lab) - best structural fit for the arm C ablation

1.5M queries, 5.6M heterogeneous candidates across 10 datasets and 4 domains (fashion,
news, Wikipedia, everyday imagery), human-written task instructions per query, with qrels.

It ships both a *global* candidate pool and *local* per-dataset pools. That is "one index"
vs "routed indexes" already laid out in the file layout, so arm C can be prototyped here
for free. The instruction field is nearly a router label already. More interesting now
that EmbeddingGemma 2 embeds the image side directly instead of needing a caption pass.

### Corpus-side only (queries must be synthesized)

**Amazon Reviews 2023 (McAuley-Lab)** - 33 product categories, review text plus
user-uploaded images plus item metadata (description, hierarchical categories, details).
Enormous; subsample 5-8 categories x 10-20k docs. Research-use license. Use if the route
taxonomy should be our choice rather than inherited.

**Fashion Product Images (Kaggle, Param Aggarwal)** - 44k products, high-res images plus
descriptions plus hierarchical catalog labels (masterCategory -> articleType -> usage).
Small enough to run all arms overnight on one GPU. No queries.

**Library of Congress Flickr / LOC-flickr-KB** - ~44k historic images with catalog titles,
descriptions, notes; the KB extension adds 100 categories and a 7.2k-entry knowledge base
linking text to visual entities. Public domain, real archival use case, diverse text
styles. The 100-class partition is finer than most routers place cleanly: stress test, not
a starting point.

**Flickr8k / Flickr-1k-test** (`jxie/flickr8k`,
`nlphuji/flickr_1k_test_image_text_retrieval`) - not a router testbed, only 8k images and
no domain partition. But it is the exact image-text setup Unsloth's own notebook trains and
scores, so it's the fastest way to prove the multimodal training path works before
pointing it at MMDocIR.

**Pick (superseded 2026-10-30):** was MMDocIR primary / SQID secondary / M-BEIR for arm C.
Final decision is four BEIR text-to-text corpora (FiQA, ArguAna, SciFact,
CQADupstack-gaming) as an enterprise-KB testbed: MMDocIR's eval set is gated, SQID
turned out to be CLIP vectors only, and M-BEIR's text-viable tasks are mostly
image-query tasks. See PLAN.md for the full rationale.

## Which set to train the LoRAs on

**Superseded 2026-10-30.** The experiment no longer uses MMDocIR (eval set gated). The
LoRAs now train on (query, relevant-doc) pairs pooled from the four BEIR corpora, held
out 20% globally for eval. Everything below is retained as source-checked MMDocIR
reference in case it's ever revisited.

_(Original recommendation)_ `MMDocIR/MMDocIR_Train_Dataset`, split by its own `domain`
field, one LoRA per domain group. Three reasons:

1. It ships finished triplets. Every row is
   `{query, domain, positive_passages:[{doc_name, page_id}], negative_passages:[{doc_name, page_id}]}` -
   a query, a gold page, one mined hard negative. Contrastive embedding training needs
   exactly that shape, so there's no pair-generation or negative-mining stage to build.
2. The `domain` field does double duty: router label *and* partition key for the
   specialists. One field drives both halves of the experiment and they can't drift apart.
3. It shares the eval set's source distribution, so a per-route gain can't be waved away
   as train/test mismatch.

Measured contents - pulled the seven `annotations_top1_negative/*_train.jsonl` files
directly, 19 MB of text:

```
Domain       Triplets   UniqQ    Docs   Neg=same-doc   Visual-query
Wiki-ss        15,293  14,968  12,752        0%             2%
TAT-DQA        14,249  13,254     163      100%             4%
SlideVQA       11,606  10,471   1,990      100%            11%
ArxivQA        10,838   9,028   1,571      100%            74%
MP-DocVQA       6,122   5,462     512      100%             0%
SciQAG          4,496   4,496   1,157      100%             1%
DUDE            2,715   2,557     767      100%             7%
─────────────  ───────  ──────  ──────     ──────────      ──────────
total          65,319  60,236
```

Four things in that table change the design:

- **The published count is wrong.** The org page advertises 173,843 training questions;
  the released files contain 65,319 triplets, 92% unique queries. Plan against the files.
- **TAT-DQA is 14k triplets over only 163 documents** with same-document negatives. An
  adapter there can score well by learning *which page of this filing* rather than what
  financial language means - document memorization dressed up as domain knowledge. Fix:
  document-level held-out split, train on 80% of a domain's docs, report on the rest.
- **ArxivQA is 74% visual queries** ("what does the red hatched area represent"). Under a
  text-only embedder that route has a ceiling unrelated to routing. EmbeddingGemma 2
  removes it, so keep the route but make it multimodal, and treat its score as
  non-comparable to a text-only route's.
- **Wiki-ss negatives are cross-document** while every other domain uses same-document page
  negatives. Loss character differs per route, so log loss curves per route rather than
  expecting one comparable curve.

Recommended four-route partition (roughly balanced, multimodal-tractable):

```
Route                 Sources                 Triplets   Docs
finance               TAT-DQA + MP-DocVQA       20,371    675
slides                SlideVQA                  11,606  1,990
technical reports     ArxivQA + DUDE            13,553  2,338
web infographics      Wiki-ss                   15,293 12,752
(optional 5th)        SciQAG                     4,496  1,157
```

Second run: **SQID / ESCI**, not a second MMDocIR partition. (query, Exact) positives with
Substitute and Complement products as free hard negatives. More usefully, it makes the
transfer question testable: train on MMDocIR finance, evaluate on SQID product queries. If
a domain specialist only helps inside the corpus it saw, then every deployment needs its
own retrain, which changes the cost story for the whole idea.

## Practical access notes

- The evaluation set (`MMDocIR/MMDocIR-Evaluation-Dataset`, 1,685 expert questions) is
  **gated on Hugging Face**. The API returns `Invalid username or password` until terms are
  accepted with a token. The train set downloads openly, so eval auth is the first setup
  friction.
- The annotation JSONLs carry only `doc_name` + `page_id`. Passage *text* lives in
  `parquet/{domain}_filter.parquet` (4-10 GB per domain, mostly page images), joined on
  `(file_name, page)`, with the `layouts` column holding text blocks. Read with pyarrow
  column pruning over HF's HTTP filesystem so the image column isn't downloaded, and pull
  only the parquets for routes in use.