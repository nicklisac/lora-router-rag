#!/usr/bin/env python3
"""Smoke test: EG2 loads, LoRA attaches, one MNRL training step runs,
and .encode returns unit-norm 768-d vectors. Run before any real training
to catch version drift in the Studio venv.

Usage: CUDA_VISIBLE_DEVICES=2 python scripts/smoke_test_training.py
"""
from unsloth import FastSentenceTransformer
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss

import os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(HERE, "data", "eg2", "unsloth-embeddinggemma-2")

model = FastSentenceTransformer.from_pretrained(
    CKPT, max_seq_length=1024, full_finetuning=False,
    config_kwargs={"vision_config": None, "audio_config": None},
)
model = FastSentenceTransformer.get_peft_model(
    model, r=32, lora_alpha=64, lora_dropout=0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", task_type="FEATURE_EXTRACTION",
    random_state=3407,
)
n = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[1/3] LoRA attached, trainable: {n/1e6:.2f}M")

rows = [
    ("how do I file taxes for a small business",
     "A small business files Schedule C with its 1040 to report profit or loss."),
    ("what is the statute of limitations on a contract",
     "The statute of limitations for a written contract is typically six years."),
    ("how does photosynthesis convert light to energy",
     "Photosynthesis uses chlorophyll to convert light energy into glucose."),
    ("best way to level a patio",
     "To level a patio, use a spirit level and a straightedge."),
    ("what causes a high fever in children",
     "High fever in children is usually caused by a viral infection."),
    ("how to cite a book in APA",
     "In APA, cite a book as author, year, title, publisher, and DOI."),
    ("boiling point of water at altitude",
     "Water boils lower at higher altitude because pressure is reduced."),
    ("how do I reset a router",
     "To reset a router, hold the reset button for 10 seconds."),
]
from datasets import Dataset
ds = Dataset.from_list([
    {"anchor": f"task: search result | query: {q}",
     "positive": f"title: none | text: {p}"}
    for q, p in rows
])

args = SentenceTransformerTrainingArguments(
    output_dir="/tmp/eg2_smoke", per_device_train_batch_size=8, max_steps=3,
    learning_rate=2e-5, logging_steps=1, report_to="none", bf16=True,
    remove_unused_columns=False, save_strategy="no",
)
trainer = SentenceTransformerTrainer(
    model=model, args=args, train_dataset=ds,
    loss=MultipleNegativesRankingLoss(model),
)
trainer.train()
print("[2/3] 3 MNRL steps trained")

emb = model.encode(["task: search result | query: a green shearling jacket"])
norm = float((emb[0] ** 2).sum() ** 0.5)
print(f"[3/3] encode OK, shape {tuple(emb.shape)}, norm {norm:.4f}")
assert emb.shape == (1, 768) and 0.9 < norm < 1.1
print("SMOKE_TEST_PASSED")