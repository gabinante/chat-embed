"""Dataset downloaders for ThreadEmbed training data."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from datasets import load_dataset
from rich.console import Console
from rich.table import Table

console = Console()
log = logging.getLogger(__name__)

RAW_DATA_DIR = Path("data/raw")


# ---------------------------------------------------------------------------
# Registry of known datasets
# ---------------------------------------------------------------------------

DATASETS = {
    # P0 — highest value/effort
    "discord_dialogues": {
        "hf_path": "mookiezi/Discord-Dialogues",
        "priority": "P0",
        "description": "7.3M exchanges, 16M turns of Discord conversations in ChatML format",
    },
    "irc_disentangle": {
        "hf_path": "jkkummerfeld/irc-disentangle",
        "priority": "P0",
        "description": "~70k annotated IRC messages with conversation disentanglement labels",
    },
    # P1 — good value, moderate effort
    "discord_unveiled": {
        "hf_path": "SaisExperiments/Discord-Unveiled-Compressed",
        "priority": "P1",
        "description": "2B+ Discord messages (will sample ~1%)",
        "sample_fraction": 0.01,
    },
    "slack_dev_chats": {
        "source": "github",
        "url": "https://github.com/preethac/Software-related-Slack-Chats",
        "priority": "P1",
        "description": "~2 years of developer Slack chats across 5 channels (XML)",
    },
    # P1 — open Slack exports from OSS communities
    "slack_dev_chats_disentangled": {
        "source": "github",
        "url": "https://github.com/preethac/Software-related-Slack-Chats-with-Disentangled-Conversations",
        "priority": "P1",
        "description": "~2 years, 5 channels (pythondev, clojurians, elmlang, racket) with conversation_id annotations (XML)",
    },
    "flyte_slack": {
        "hf_path": "unionai/flyte-slack-data",
        "priority": "P1",
        "description": "28k Flyte OSS community Slack Q&A pairs (input/output format)",
    },
    "flyte_slack_long": {
        "hf_path": "Samhita/slack-data-long-responses",
        "priority": "P1",
        "description": "24.5k long-form Slack responses from Flyte community",
    },
    "clojurians_log": {
        "source": "github",
        "url": "https://github.com/GaiwanTeam/clojurians-log-v2",
        "priority": "P1",
        "description": "Clojurians Slack archive — ~2M messages across hundreds of channels",
    },
    # P2 — supplementary
    "topical_chat": {
        "hf_path": "Conversational-Reasoning/Topical-Chat",
        "priority": "P2",
        "description": "~11k structured conversations with topic annotations",
    },
    "ultrachat_200k": {
        "hf_path": "HuggingFaceH4/ultrachat_200k",
        "priority": "P2",
        "description": "200k multi-turn dialogues (synthetic but useful for volume)",
    },
}


# ---------------------------------------------------------------------------
# Download functions
# ---------------------------------------------------------------------------


def download_hf_dataset(
    name: str,
    hf_path: str,
    output_dir: Path,
    split: str | None = None,
    sample_fraction: float | None = None,
) -> dict:
    """Download a dataset from HuggingFace and save to disk.

    Returns dict with download stats.
    """
    console.print(f"[bold blue]Downloading {name}[/] from {hf_path}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        ds = load_dataset(hf_path, split=split, trust_remote_code=True)
    except Exception:
        # Some datasets need specific config names — try without split
        ds = load_dataset(hf_path, trust_remote_code=True)

    # If it's a DatasetDict, save all splits
    from datasets import DatasetDict

    if isinstance(ds, DatasetDict):
        total_rows = sum(len(ds[s]) for s in ds)
        if sample_fraction and sample_fraction < 1.0:
            for s in ds:
                n = max(1, int(len(ds[s]) * sample_fraction))
                ds[s] = ds[s].shuffle(seed=42).select(range(n))
            total_rows = sum(len(ds[s]) for s in ds)
            console.print(f"  Sampled {sample_fraction:.1%} → {total_rows:,} rows")
        ds.save_to_disk(str(output_dir))
        splits = list(ds.keys())
    else:
        total_rows = len(ds)
        if sample_fraction and sample_fraction < 1.0:
            n = max(1, int(total_rows * sample_fraction))
            ds = ds.shuffle(seed=42).select(range(n))
            total_rows = len(ds)
            console.print(f"  Sampled {sample_fraction:.1%} → {total_rows:,} rows")
        ds.save_to_disk(str(output_dir))
        splits = [split or "all"]

    stats = {
        "name": name,
        "hf_path": hf_path,
        "total_rows": total_rows,
        "splits": splits,
        "output_dir": str(output_dir),
    }

    # Save stats alongside
    with open(output_dir / "download_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    console.print(f"  [green]✓[/] Saved {total_rows:,} rows to {output_dir}")
    return stats


def download_github_repo(name: str, url: str, output_dir: Path) -> dict:
    """Clone a GitHub repo for non-HF data sources."""
    import subprocess

    console.print(f"[bold blue]Cloning {name}[/] from {url}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["git", "clone", "--depth", "1", url, str(output_dir)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        if "already exists" in result.stderr:
            console.print(f"  [yellow]Already cloned[/] at {output_dir}")
        else:
            console.print(f"  [red]Error:[/] {result.stderr}")
            raise RuntimeError(f"Failed to clone {url}: {result.stderr}")

    stats = {
        "name": name,
        "url": url,
        "output_dir": str(output_dir),
    }

    with open(output_dir / "download_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    console.print(f"  [green]✓[/] Cloned to {output_dir}")
    return stats


def download_dataset(name: str) -> dict:
    """Download a single dataset by name."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Known: {list(DATASETS.keys())}")

    info = DATASETS[name]
    output_dir = RAW_DATA_DIR / name

    if "hf_path" in info:
        return download_hf_dataset(
            name=name,
            hf_path=info["hf_path"],
            output_dir=output_dir,
            sample_fraction=info.get("sample_fraction"),
        )
    elif info.get("source") == "github":
        return download_github_repo(
            name=name,
            url=info["url"],
            output_dir=output_dir,
        )
    else:
        raise ValueError(f"Don't know how to download {name}")


def download_all(priority: str | None = None) -> list[dict]:
    """Download all datasets, optionally filtered by priority."""
    all_stats = []

    for name, info in DATASETS.items():
        if priority and info.get("priority") != priority:
            continue
        try:
            stats = download_dataset(name)
            all_stats.append(stats)
        except Exception as e:
            console.print(f"  [red]Failed to download {name}:[/] {e}")
            log.exception(f"Failed to download {name}")

    # Print summary table
    print_download_summary(all_stats)
    return all_stats


def print_download_summary(all_stats: list[dict]) -> None:
    """Print a rich table summarizing all downloads."""
    table = Table(title="Download Summary")
    table.add_column("Dataset", style="cyan")
    table.add_column("Rows", justify="right")
    table.add_column("Output Dir")

    for s in all_stats:
        rows = f"{s.get('total_rows', 'N/A'):,}" if isinstance(s.get("total_rows"), int) else "N/A"
        table.add_row(s["name"], rows, s["output_dir"])

    console.print(table)
