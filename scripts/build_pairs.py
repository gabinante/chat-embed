#!/usr/bin/env python3
"""Build training pairs from preprocessed conversations.

Usage:
    python scripts/build_pairs.py --strategy thread_based
    python scripts/build_pairs.py --strategy query_response
    python scripts/build_pairs.py --strategy temporal
    python scripts/build_pairs.py --all
"""

import sys
from pathlib import Path

import click
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thread_embed.data.pair_builders import (
    build_query_response_pairs,
    build_temporal_pairs,
    build_thread_pairs,
    load_conversations,
    save_pairs,
)

console = Console()

PROCESSED_DIR = Path("data/processed")


@click.command()
@click.option(
    "--strategy",
    type=click.Choice(["thread_based", "query_response", "temporal", "all"]),
    default="all",
    help="Which pair generation strategy to run",
)
@click.option("--source", type=str, default=None, help="Only use a specific source dataset")
@click.option("--max-per-source", type=int, default=50_000, help="Max conversations to load per source")
@click.option("--max-pairs", type=int, default=500_000, help="Max pairs per strategy")
def main(strategy: str, source: str | None, max_per_source: int, max_pairs: int):
    """Build training pairs from preprocessed conversations."""
    # Load all processed conversations (sampled per source)
    if source:
        source_dirs = [PROCESSED_DIR / source]
    else:
        source_dirs = [d for d in PROCESSED_DIR.iterdir() if d.is_dir()]

    all_conversations = []
    for d in source_dirs:
        all_conversations.extend(load_conversations(d, max_conversations=max_per_source))

    console.print(f"[bold]Total conversations: {len(all_conversations):,}[/]")

    if strategy in ("thread_based", "all"):
        pairs = build_thread_pairs(all_conversations, max_pairs=max_pairs)
        save_pairs(pairs, "thread_based")

    if strategy in ("query_response", "all"):
        pairs = build_query_response_pairs(all_conversations, max_pairs=max_pairs)
        save_pairs(pairs, "query_response")

    if strategy in ("temporal", "all"):
        pairs = build_temporal_pairs(all_conversations, max_pairs=max_pairs)
        save_pairs(pairs, "temporal")


if __name__ == "__main__":
    main()
