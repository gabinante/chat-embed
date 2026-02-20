#!/usr/bin/env python3
"""Preprocess raw datasets into normalized conversation format.

Usage:
    python scripts/preprocess_data.py --source discord_dialogues
    python scripts/preprocess_data.py --source irc_disentangle
    python scripts/preprocess_data.py --all
"""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thread_embed.data.preprocessors import PREPROCESSORS, preprocess_dataset

console = Console()

RAW_DIR = Path("data/raw")


@click.command()
@click.option("--source", type=str, default=None, help="Process a specific source")
@click.option("--all", "process_all", is_flag=True, help="Process all downloaded sources")
@click.option("--list", "list_sources", is_flag=True, help="List available sources")
def main(source: str | None, process_all: bool, list_sources: bool):
    """Preprocess raw chat datasets into normalized conversations."""
    if list_sources:
        table = Table(title="Available Sources")
        table.add_column("Source", style="cyan")
        table.add_column("Downloaded", style="green")
        table.add_column("Preprocessor")
        for name in PREPROCESSORS:
            downloaded = "✓" if (RAW_DIR / name).exists() else "✗"
            table.add_row(name, downloaded, "yes")
        # Also check for any raw dirs without explicit preprocessors
        for d in sorted(RAW_DIR.iterdir()):
            if d.is_dir() and d.name not in PREPROCESSORS:
                table.add_row(d.name, "✓", "auto (slack)" if d.name.startswith("slack_") else "none")
        console.print(table)
        return

    if source:
        stats = preprocess_dataset(source)
        console.print(f"\n[green]Done![/] {stats.num_conversations:,} conversations, {stats.num_messages:,} messages")
    elif process_all:
        all_stats = []
        for name in PREPROCESSORS:
            if (RAW_DIR / name).exists():
                try:
                    stats = preprocess_dataset(name)
                    all_stats.append(stats)
                except Exception as e:
                    console.print(f"[red]Error processing {name}:[/] {e}")
            else:
                console.print(f"[yellow]Skipping {name} — not downloaded[/]")

        # Summary
        table = Table(title="Preprocessing Summary")
        table.add_column("Source", style="cyan")
        table.add_column("Conversations", justify="right")
        table.add_column("Messages", justify="right")
        for s in all_stats:
            table.add_row(s.source, f"{s.num_conversations:,}", f"{s.num_messages:,}")
        console.print(table)
    else:
        console.print("[yellow]No action specified.[/] Use --help for usage.")


if __name__ == "__main__":
    main()
