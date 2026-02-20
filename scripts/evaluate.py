#!/usr/bin/env python3
"""Evaluate models on the chat retrieval benchmark.

Usage:
    python scripts/evaluate.py --model models/thread-embed-v1
    python scripts/evaluate.py --model BAAI/bge-base-en-v1.5  # baseline
"""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

console = Console()


@click.command()
@click.option("--model", required=True, help="Model path or HuggingFace model ID")
@click.option("--eval-data", default="data/eval", help="Path to evaluation data")
@click.option("--batch-size", default=128, type=int)
def main(model: str, eval_data: str, batch_size: int):
    """Run benchmark evaluation."""
    from sentence_transformers import SentenceTransformer

    from thread_embed.eval.benchmark import BenchmarkResult, run_retrieval_eval

    console.print(f"[bold]Evaluating model: {model}[/]")

    st_model = SentenceTransformer(model)

    # TODO: Load eval datasets from data/eval/
    # For now, print a placeholder
    console.print("[yellow]Evaluation data not yet built. Run preprocessing + eval data creation first.[/]")


if __name__ == "__main__":
    main()
