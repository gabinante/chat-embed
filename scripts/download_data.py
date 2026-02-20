#!/usr/bin/env python3
"""Download datasets for ThreadEmbed training.

Usage:
    # Download all P0 (highest priority) datasets
    python scripts/download_data.py --priority P0

    # Download a specific dataset
    python scripts/download_data.py --dataset discord_dialogues

    # Download everything
    python scripts/download_data.py --all

    # List available datasets
    python scripts/download_data.py --list
"""

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

# Add src to path so we can import thread_embed
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thread_embed.data.downloaders import DATASETS, download_all, download_dataset

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@click.command()
@click.option("--priority", type=click.Choice(["P0", "P1", "P2"]), help="Download by priority level")
@click.option("--dataset", type=str, help="Download a specific dataset by name")
@click.option("--all", "download_everything", is_flag=True, help="Download all datasets")
@click.option("--list", "list_datasets", is_flag=True, help="List available datasets")
def main(
    priority: str | None,
    dataset: str | None,
    download_everything: bool,
    list_datasets: bool,
):
    """Download chat datasets for ThreadEmbed training."""
    if list_datasets:
        table = Table(title="Available Datasets")
        table.add_column("Name", style="cyan")
        table.add_column("Priority", style="yellow")
        table.add_column("Description")
        for name, info in DATASETS.items():
            table.add_row(name, info.get("priority", "?"), info.get("description", ""))
        console.print(table)
        return

    if dataset:
        download_dataset(dataset)
    elif priority:
        download_all(priority=priority)
    elif download_everything:
        download_all()
    else:
        console.print("[yellow]No action specified.[/] Use --help for usage.")
        console.print("Recommended: [bold]python scripts/download_data.py --priority P0[/]")


if __name__ == "__main__":
    main()
