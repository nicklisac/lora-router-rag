#!/usr/bin/env python3
"""Train one LoRA adapter on EG2 for a single domain, or a single joint
adapter over all four domains (the control arm).

Every adapter starts from the SAME base checkpoint that mode 1 and the smoke
test use (data/eg2/unsloth-embeddinggemma-2, bf16), the SAME Gemma target
modules, the SAME rank/alpha/lr as the Unsloth EG2 text-only notebook, and the
SAME prompt templates as serving (via src/prefixes.py). Only the data differs:

  --name fiqa      -> train rows from fiqa queries only
  --name arguana   -> ...
  --name scifact   -> ...
  --name gaming    -> ...
  --name joint     -> all four domains pooled (the control)

One row per (query, positive-doc) pair. MNRL with in-batch negatives supplies
the contrastive signal, so no negative mining is needed. For `joint`, the
in-batch negatives are cross-domain by construction, which is exactly the
control the plan wants: if the joint adapter matches the specialists, domain
specialization was not the effect.

Why we pre-format the columns ourselves and do NOT pass SentenceTransformer's
`prompts=` arg (which the notebook uses): `model.prompts["document"]` is
`title: none | text: ` — it hardcodes the title. Mode 1 indexed the corpus with
REAL titles via format_doc. Training through the ST prompt mechanism would see
`title: none` at train time and real titles at serving time, a silent train/serve
skew. Pre-formatting with src/prefixes.py keeps train == index == query.

Artifacts per adapter (all kept):
  results/lora/<name>/            raw LoRA adapter (adapter_model.safetensors + config)
  results/lora/<name>/trainer_state.json  (copied from the HF run dir)
  results/lora/<name>_merged/     base + adapter merged to a standalone bf16
                                  checkpoint, used by eval so there is no
                                  for_inference reload path to get wrong
Plus a "mode2_train_<name>" entry appended to results/run_log.jsonl.

Usage:
  CUDA_VISIBLE_DEVICES=2 python scripts/train_lora.py --name fiqa
  CUDA_VISIBLE_DEVICES=2 python scripts/train_lora.py --name joint
  CUDA_VISIBLE_DEVICES=2 python scripts/train_lora.py --name fiqa --max-steps 50
"""
import argparse
import json
import os
import time


HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DERIVED = os.path.join(HERE, "data", "derived")
RUN_LOG = os.path.join(HERE, "results", "run_log.jsonl")
LORA_DIR = os.path.join(HERE, "results", "lora")
CKPT = os.path.join(HERE, "data", "eg2", "unsloth-embeddinggemma-2")

# Same as the notebook's text-only route, and as scripts/smoke_test_training.py.
GEMMA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def build_dataset(domain):
    """Return a list of {anchor, positive} rows, pre-formatted with the
    serving prefixes. domain == 'joint' pools all four."""
    sys_path = os.path.join(HERE, "src")
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from prefixes import format_query, format_doc

    queries = load_jsonl(os.path.join(DERIVED, "queries.jsonl"))
    queries = [q for q in queries if q["split"] == "train"]
    if domain != "joint":
        queries = [q for q in queries if q["domain"] == domain]

    # (domain, did) -> (title, text)
    corpus = load_jsonl(os.path.join(DERIVED, "corpus.jsonl"))
    doc = {(c["domain"], c["did"]): (c["title"], c["text"]) for c in corpus}

    rows = []
    missing = 0
    for q in queries:
        for did in q["pos_dids"]:
            key = (q["domain"], did)
            if key not in doc:
                missing += 1
                continue
            title, text = doc[key]
            rows.append({
                "anchor": format_query(q["query"]),
                "positive": format_doc(text, title),
            })
    return rows, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True,
                    choices=["fiqa", "arguana", "scifact", "gaming", "joint"])
    ap.add_argument("--max-steps", type=int, default=200,
                    help="fixed step budget so all 5 adapters get equal compute")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--batch", type=int, default=64,
                    help="per-device batch; GA doubles the effective batch to 128")
    ap.add_argument("--ga", type=int, default=2)
    ap.add_argument("--seq", type=int, default=1024,
                    help="training seq len (notebook default); eval encodes at 512")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--epochs-instead-of-steps", action="store_true",
                    help="ignore max_steps, train 1 epoch (diagnostic only)")
    args = ap.parse_args()

    from unsloth import FastSentenceTransformer
    from sentence_transformers import (
        SentenceTransformerTrainer, SentenceTransformerTrainingArguments)
    from sentence_transformers.sentence_transformer import losses
    from sentence_transformers.sentence_transformer.training_args import BatchSamplers
    from datasets import Dataset
    from unsloth import is_bf16_supported

    os.makedirs(LORA_DIR, exist_ok=True)
    out_dir = os.path.join(LORA_DIR, args.name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[data] building {args.name} train set ...", flush=True)
    rows, missing = build_dataset(args.name)
    print(f"[data] {len(rows)} (query, positive-doc) rows; "
          f"{missing} pos docs not in corpus (dropped)", flush=True)
    ds = Dataset.from_list(rows)

    print("[model] loading base EG2 (bf16) ...", flush=True)
    model = FastSentenceTransformer.from_pretrained(
        CKPT, max_seq_length=args.seq, full_finetuning=False,
        config_kwargs={"vision_config": None, "audio_config": None},
    )
    model = FastSentenceTransformer.get_peft_model(
        model, r=args.r, lora_alpha=args.alpha, lora_dropout=0, bias="none",
        target_modules=GEMMA_TARGETS,
        use_gradient_checkpointing="unsloth", task_type="FEATURE_EXTRACTION",
        random_state=args.seed,
    )
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] LoRA attached, trainable: {n_train/1e6:.2f}M", flush=True)

    loss = losses.MultipleNegativesRankingLoss(model)
    train_args = SentenceTransformerTrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.ga,
        learning_rate=args.lr,
        logging_steps=10,
        warmup_steps=0.03,            # float == fraction of total steps
        lr_scheduler_type="linear",
        report_to="none",
        bf16=is_bf16_supported(),
        remove_unused_columns=False,
        save_strategy="no",           # we save the adapter ourselves below
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        num_train_epochs=1 if args.epochs_instead_of_steps else 3,
        max_steps=None if args.epochs_instead_of_steps else args.max_steps,
        seed=args.seed,
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=train_args, train_dataset=ds, loss=loss,
    )

    t0 = time.time()
    trainer.train()
    n_steps = int(trainer.state.global_step)

    # loss curve for the run log
    hist = [(int(h["step"]), round(float(h["loss"]), 5))
            for h in trainer.state.log_history
            if "loss" in h]

    # canonical adapter artifact (small, the real thing)
    model.save_pretrained(out_dir)
    model.tokenizer.save_pretrained(out_dir)
    # trainer_state.json isn't written under save_strategy="no"; persist HF's own
    # TrainerState (full schema: global_step, epoch, log_history, ...) so the
    # loss curve and final step are a durable artifact, not just a run-log line.
    trainer.state.save_to_json(os.path.join(out_dir, "trainer_state.json"))

    # merged bf16 standalone checkpoint for eval (no for_inference reload path)
    merged_dir = os.path.join(LORA_DIR, args.name + "_merged")
    print(f"[save] merging to standalone bf16 -> {merged_dir} ...", flush=True)
    model.save_pretrained_merged(merged_dir, tokenizer=model.tokenizer,
                                 save_method="merged_16bit")

    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run": f"mode2_train_{args.name}",
        "base": "embeddinggemma-2 bf16 safetensors",
        "adapter": f"results/lora/{args.name}",
        "merged": f"results/lora/{args.name}_merged",
        "lora": {"r": args.r, "alpha": args.alpha, "dropout": 0,
                 "targets": GEMMA_TARGETS,
                 "trainable_params": n_train},
        "optimizer": {"lr": args.lr, "scheduler": "linear",
                      "warmup_frac": 0.03, "batch": args.batch,
                      "grad_accum": args.ga, "effective_batch": args.batch * args.ga,
                      "seq": args.seq, "seed": args.seed},
        "loss": "MNRL + NO_DUPLICATES",
        "n_train_rows": len(rows),
        "n_steps": n_steps,
        "seconds": round(time.time() - t0, 1),
        "loss_curve": hist,
    }
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print("=== TRAIN SUMMARY (paste-ready) ===")
    print(json.dumps(entry, indent=2))
    print(f"DONE {args.name}: adapter at {out_dir}, merged at {merged_dir}")


if __name__ == "__main__":
    main()