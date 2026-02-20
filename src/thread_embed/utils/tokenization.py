"""Tokenization utilities for chat message formatting."""

from __future__ import annotations


def format_messages_as_text(
    messages: list[dict],
    include_timestamps: bool = False,
) -> str:
    """Format a list of messages into a single text block for embedding.

    Args:
        messages: List of dicts with 'author_id', 'content', and optionally 'timestamp'
        include_timestamps: Whether to prepend relative timestamps
    """
    lines = []
    base_ts = None

    for msg in messages:
        prefix = ""
        if include_timestamps and msg.get("timestamp"):
            if base_ts is None:
                base_ts = msg["timestamp"]
                prefix = "[+0s] "
            else:
                delta = (msg["timestamp"] - base_ts).total_seconds()
                prefix = f"[+{int(delta)}s] "

        lines.append(f"{prefix}{msg['author_id']}: {msg['content']}")

    return "\n".join(lines)
