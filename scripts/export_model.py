#!/usr/bin/env python3
"""Export trained model to HuggingFace Hub format.

Usage:
    python scripts/export_model.py --model-dir models/thread-embed-v1 --repo-id your-username/thread-embed
"""

import sys
from pathlib import Path

import click
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

console = Console()


@click.command()
@click.option("--model-dir", required=True, help="Local model directory")
@click.option("--repo-id", required=True, help="HuggingFace repo ID (e.g. username/thread-embed)")
@click.option("--private", is_flag=True, help="Make the repo private")
def main(model_dir: str, repo_id: str, private: bool):
    """Export model to HuggingFace Hub."""
    from sentence_transformers import SentenceTransformer

    console.print(f"[bold]Loading model from {model_dir}...[/]")
    model = SentenceTransformer(model_dir)

    console.print(f"[bold]Pushing to {repo_id}...[/]")
    model.push_to_hub(repo_id, private=private)

    console.print(f"[green]✓[/] Model pushed to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
