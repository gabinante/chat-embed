#!/usr/bin/env python3
"""Train the ThreadEmbed model.

Usage:
    python scripts/train.py
    python scripts/train.py --base-model BAAI/bge-base-en-v1.5 --epochs 3
"""

import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thread_embed.training.config import TrainingConfig
from thread_embed.training.train import train


@click.command()
@click.option("--base-model", default="BAAI/bge-base-en-v1.5")
@click.option("--epochs", default=3, type=int)
@click.option("--batch-size", default=256, type=int)
@click.option("--lr", default=2e-5, type=float)
@click.option("--output-dir", default="models/thread-embed-v1")
@click.option("--max-samples", default=None, type=int, help="Limit training samples (for testing)")
def main(base_model, epochs, batch_size, lr, output_dir, max_samples):
    """Train ThreadEmbed."""
    config = TrainingConfig(
        base_model=base_model,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        output_dir=output_dir,
        max_train_samples=max_samples,
    )
    train(config)


if __name__ == "__main__":
    main()
