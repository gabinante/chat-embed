"""Training configuration for ThreadEmbed."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    # Model
    base_model: str = "BAAI/bge-base-en-v1.5"
    max_seq_length: int = 512
    pooling: str = "mean"

    # Training
    epochs: int = 3
    batch_size: int = 64
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    fp16: bool = True  # auto-disabled on MPS, falls back to bf16

    # Loss
    loss: str = "MultipleNegativesRankingLoss"

    # Matryoshka (flexible dimensionality)
    matryoshka_dims: list[int] = field(default_factory=lambda: [64, 128, 256, 512, 768])

    # Data
    max_train_samples: int | None = None
    eval_steps: int = 1000
    save_steps: int = 5000

    # Output
    output_dir: str = "models/thread-embed-v1"
