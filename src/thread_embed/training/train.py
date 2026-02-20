"""Main training loop for ThreadEmbed.

Uses sentence-transformers v5+ API with HuggingFace Dataset format.
Supports MPS (Apple Silicon) and CUDA training.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

from .config import TrainingConfig


def load_pairs_dataset(pairs_dir: Path, max_samples: int | None = None) -> Dataset:
    """Load training pairs from JSONL files into a HuggingFace Dataset.

    sentence-transformers v5+ expects Dataset with 'anchor' and 'positive' columns
    for MultipleNegativesRankingLoss.
    """
    anchors = []
    positives = []

    for jsonl in sorted(pairs_dir.rglob("pairs.jsonl")):
        with open(jsonl) as f:
            for line in f:
                pair = json.loads(line)
                anchors.append(pair["anchor"])
                positives.append(pair["positive"])

    if max_samples and len(anchors) > max_samples:
        anchors = anchors[:max_samples]
        positives = positives[:max_samples]

    return Dataset.from_dict({"anchor": anchors, "positive": positives})


def train(config: TrainingConfig | None = None):
    """Run the training loop."""
    if config is None:
        config = TrainingConfig()

    # Detect device capabilities
    use_fp16 = config.fp16 and torch.cuda.is_available()
    use_bf16 = not use_fp16 and torch.backends.mps.is_available()

    # 1. Load base model
    model = SentenceTransformer(config.base_model)
    model.max_seq_length = config.max_seq_length

    # 2. Load training pairs
    train_dataset = load_pairs_dataset(
        Path("data/pairs"),
        max_samples=config.max_train_samples,
    )
    print(f"Loaded {len(train_dataset):,} training pairs")

    # 3. Configure loss
    base_loss = losses.MultipleNegativesRankingLoss(model)
    if config.matryoshka_dims:
        loss = losses.MatryoshkaLoss(
            model, base_loss, matryoshka_dims=config.matryoshka_dims
        )
    else:
        loss = base_loss

    # 4. Training arguments (adapt for available hardware)
    # Calculate warmup steps from ratio
    total_steps = (len(train_dataset) // config.batch_size) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    args = SentenceTransformerTrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=config.weight_decay,
        fp16=use_fp16,
        bf16=use_bf16,
        eval_strategy="no",
        save_steps=config.save_steps,
        logging_steps=100,
        save_total_limit=3,
        dataloader_num_workers=0,  # MPS doesn't support multiprocess data loading
    )

    # 5. Train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()

    # 6. Save final model
    model.save(config.output_dir)
    print(f"Model saved to {config.output_dir}")
