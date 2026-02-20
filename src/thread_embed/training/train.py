"""Main training loop for ThreadEmbed."""

from __future__ import annotations

import json
from pathlib import Path

from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from torch.utils.data import Dataset

from .config import TrainingConfig


class PairsDataset(Dataset):
    """Load training pairs from JSONL files."""

    def __init__(self, pairs_dir: Path, max_samples: int | None = None):
        self.examples = []
        for jsonl in sorted(pairs_dir.rglob("pairs.jsonl")):
            with open(jsonl) as f:
                for line in f:
                    pair = json.loads(line)
                    self.examples.append(
                        InputExample(texts=[pair["anchor"], pair["positive"]])
                    )
        if max_samples:
            self.examples = self.examples[:max_samples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def train(config: TrainingConfig | None = None):
    """Run the training loop."""
    if config is None:
        config = TrainingConfig()

    # 1. Load base model
    model = SentenceTransformer(config.base_model)
    model.max_seq_length = config.max_seq_length

    # 2. Load training pairs
    train_dataset = PairsDataset(
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

    # 4. Training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        fp16=config.fp16,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        logging_steps=100,
        save_total_limit=3,
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
