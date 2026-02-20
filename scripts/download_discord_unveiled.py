#!/usr/bin/env python3
"""Download a sample of Discord Unveiled via streaming.

This dataset has 2B+ messages stored as per-server JSON files.
We stream and sample to get a manageable amount of multi-party Discord data
with real channel context, multiple participants, and reply threading.

Usage:
    python scripts/download_discord_unveiled.py --max-messages 5000000
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

console = Console()

OUTPUT_DIR = Path("data/raw/discord_unveiled_sampled")


@click.command()
@click.option("--max-messages", default=5_000_000, type=int, help="Max messages to download")
@click.option("--min-content-len", default=10, type=int, help="Skip messages shorter than this")
def main(max_messages: int, min_content_len: int):
    """Stream and sample Discord Unveiled for multi-party data."""
    from datasets import load_dataset

    console.print(f"[bold blue]Streaming Discord Unveiled[/] (target: {max_messages:,} messages)")
    console.print("  Filtering for: English, non-bot, with content, multi-party channels")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("fvdfs41/Discord-Unveiled", streaming=True, split="train")

    messages_by_channel: dict[str, list[dict]] = defaultdict(list)
    total_kept = 0
    total_seen = 0
    channels_seen = set()

    for row in ds:
        total_seen += 1

        # Skip bots
        if row.get("is_bot"):
            continue

        content = row.get("content", "")
        if not content or len(content) < min_content_len:
            continue

        # Skip non-text message types (joins, pins, etc.) — keep normal (0) and replies (19)
        msg_type = row.get("type", 0)
        if msg_type not in (0, 19):
            continue

        channel_id = row.get("channel_id", "")
        channel_name = row.get("channel_name", "")
        author = row.get("author", {})
        timestamp = row.get("timestamp", "")
        msg_ref = row.get("message_reference")

        # Store compact message
        msg = {
            "id": row.get("id", ""),
            "content": content,
            "author_id": author.get("id", ""),
            "author_name": author.get("username", ""),
            "timestamp": timestamp,
            "channel_id": channel_id,
            "channel_name": channel_name,
            "type": msg_type,
        }
        if msg_ref and isinstance(msg_ref, dict):
            msg["reply_to"] = msg_ref.get("message_id", "")

        messages_by_channel[channel_id].append(msg)
        channels_seen.add(channel_id)
        total_kept += 1

        if total_kept >= max_messages:
            break

        # Status update every 10k messages kept (use print+flush for background compatibility)
        if total_kept % 10_000 == 0:
            print(
                f"  Seen {total_seen:,} | Kept {total_kept:,}/{max_messages:,} | "
                f"Channels: {len(channels_seen):,}",
                flush=True,
            )

    # Save grouped by channel
    output_path = OUTPUT_DIR / "messages_by_channel.jsonl"
    channel_count = 0
    with open(output_path, "w") as f:
        for channel_id, msgs in messages_by_channel.items():
            if len(msgs) >= 5:  # Only keep channels with enough messages
                f.write(json.dumps({
                    "channel_id": channel_id,
                    "channel_name": msgs[0]["channel_name"],
                    "messages": msgs,
                }) + "\n")
                channel_count += 1

    stats = {
        "total_seen": total_seen,
        "total_kept": total_kept,
        "channels": channel_count,
        "output_file": str(output_path),
    }
    with open(OUTPUT_DIR / "download_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    console.print(f"\n[green]Done![/]")
    console.print(f"  Messages: {total_kept:,} (from {total_seen:,} seen)")
    console.print(f"  Channels: {channel_count:,}")
    console.print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
