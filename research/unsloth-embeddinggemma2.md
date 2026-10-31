# EmbeddingGemma 2 + Unsloth: what's actually verified

Sources: the Unsloth docs page for EmbeddingGemma 2, the model card, and the raw notebook
cells of `unslothai/notebooks/nb/EmbeddingGemma2_(300M).ipynb` and
`...-Image_Text.ipynb`. The notebooks' filenames say 300M but they load
`unsloth/embeddinggemma-2`, so ignore the number in the filename. Read the cells, not the
docs prose - the prose hides the flag that matters.

## Model facts

`google/embeddinggemma-2` (training mirror `unsloth/embeddinggemma-2`), Apache 2.0, task
`feature-extraction`, last modified 2026-10-28.

740M total: 270M text (130M transformer + 140M embedder) + 170M vision + 300M audio. The
vision and audio encoders are selectively loadable, so one checkpoint serves a text-only
route or a multimodal route. Native 768d, mean pooling, 512->768 projection, 8K context.
Text, code, image, video and audio all land in the same 768d space.

Both uploads ship `modules.json` plus `1_Pooling` and `2_Normalize`, which is Unsloth's
best-supported path.

**Correction to the first pass at this idea:** EG2 is not text-only. Earlier notes routed
page screenshots and product photos through OCR text or a VLM caption pass. That's
unnecessary - images embed directly. The OCR/caption fields stay useful as the *text* arm
of the same route, which gives a free tower ablation: same route, same queries, index
built from page text vs page image. Prefixes apply to text only; images, video and audio go
in unprefixed.

## Verified configs

| Route | r | alpha | lr | batch | seq | loss |
|---|---|---|---|---|---|---|
| text-only | 32 | 64 | 2e-5 | 64 + GA2 | 1024 | MNRL + NO_DUPLICATES |
| image+text | 16 | 32 | 1e-4 | 16 | 512 | MNRL |

Text-only route, from the notebook:

```python
from unsloth import FastSentenceTransformer
model = FastSentenceTransformer.from_pretrained(
    "unsloth/embeddinggemma-2", max_seq_length = 1024, full_finetuning = False,
    config_kwargs = {"vision_config": None, "audio_config": None})

model = FastSentenceTransformer.get_peft_model(
    model, r = 32, lora_alpha = 64, lora_dropout = 0, bias = "none",
    target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing = "unsloth", task_type = "FEATURE_EXTRACTION", random_state = 3407)
```

Then `SentenceTransformerTrainer` (not TRL), `losses.MultipleNegativesRankingLoss`,
`per_device_train_batch_size = 64`, `gradient_accumulation_steps = 2`,
`learning_rate = 2e-5`, `lr_scheduler_type = "linear"`,
`batch_sampler = BatchSamplers.NO_DUPLICATES`, `bf16 = is_bf16_supported()`, and
`prompts = {"question": model.prompts["query"], "passage_text": model.prompts["document"]}`
so training uses the same instruction templates as inference.

Image-text route: `max_seq_length = 512` (their note: an image costs ~280 tokens),
`config_kwargs = {"audio_config": None}` to drop the audio tower, dataset rows
`{"anchor": caption, "positive": image_path}`, then
`get_peft_model(..., finetune_vision_layers = True, r = 16, lora_alpha = 32)`,
`learning_rate = 1e-4`, batch 16, 60 steps, and `remove_unused_columns = False`.

`finetune_vision_layers` / `finetune_language_layers` choose **which towers get adapters**,
not which modalities the model can ingest. So text-only / vision-only / both is a real
ablation axis per route, and it's one flag rather than a forked codebase.

## Gotchas that cost time

- **No FP16, ever.** EG2 emits NaN or silently degraded vectors in FP16. BF16 or FP32 only,
  including compute dtype under quantized weights. The generic Unsloth template writes
  `fp16 = not is_bf16_supported()`, which is safe on the 3090s (Ampere has BF16) but not on
  older cards. Check rather than assume.
- **`for_inference=True` on reload.** Loading a saved adapter back through
  `FastSentenceTransformer` without it returns untransformed vectors and throws nothing.
  The notebooks dodge this by scoring in-process; a harness that reloads per arm hits it
  every time.
- **Prefixes are asymmetric and mandatory.** `task: search result | query: {q}` on the query
  side, `title: {t} | text: {c}` on the corpus side, applied identically at train, index and
  query time. `prompt_name="Document"` hardcodes `title: none`, so docs with real titles
  must be formatted by hand or titles get dropped on the index side - a real loss on
  MMDocIR page text.
- **Re-normalize after MRL truncation.** Slicing a unit vector doesn't preserve unit length.
  Skipping the renormalize quietly degrades ranking instead of erroring. Pass `truncate_dim`
  with `normalize_embeddings=True` and let the library handle it.
- **`target_modules` are the Gemma set**, not the BERT/RoBERTa set
  `["key","query","value","dense"]`. Wrong names attach adapters nowhere and the run
  completes looking healthy.
- **`task_type = "FEATURE_EXTRACTION"`,** not `SEQ_CLS`.
- **Learning rate differs by modality** - 2e-5 text-only vs 1e-4 image-text. No published
  guidance for a route that trains both towers together, so that interaction is ours to
  sweep.

## Consequences for the experiment

1. **Task prefixes are a free router (arm D).** The checkpoint ships seven steering
   prefixes: `SearchQuery`, `QuestionAnswering`, `FactChecking`, `CodeRetrieval`,
   `Classification`, `Clustering`, `SentenceSimilarity`. A router picking a *prefix* needs
   no extra indexes and no training. Run D before training anything.
2. **Vector dimension is a per-route cost knob.** Native MRL: reported near-lossless from
   768 to 256 (128d degrades multimodal substantially). Price the N-indexes-x-full-corpus
   design at 256d before defending it at 768d.
3. **Their eval scaffolding already answers the transfer question.** The text notebook wires
   `InformationRetrievalEvaluator` in-domain plus
   `NanoBEIREvaluator(["nfcorpus","scifact","fiqa2018","arguana"])` as an out-of-domain
   canary, printing NDCG@10 before and after training. Keep that per adapter.

## Serving

`save_pretrained_merged` gives a standalone model; `save_pretrained_gguf` exports
quantized GGUFs. `unsloth/embeddinggemma-2-GGUF` already ships BF16/F16/Q8_0 and UD-Q4..Q6
plus `mmproj-*.gguf` files for the vision side, so a trained route serves from llama.cpp
with `--embeddings --pooling mean` alongside the rest of the local stack.

## Still unknown

- VRAM for multimodal LoRA is not published anywhere. Text-only EG-300M anchors at 3GB
  QLoRA / 6GB LoRA. The notebook prints peak reserved memory - read it off the first run on
  GPU 2 instead of planning around a number nobody published.
- Whether a shared projection layer across per-route LoRAs keeps the vector spaces loosely
  comparable, or whether that's hand-waving. Nobody has tested it as far as I can tell.