"""Training pair construction from normalized conversations.

Implements the four pair generation strategies:
- Strategy A: Thread-based positives
- Strategy B: Query-response positives
- Strategy C: Temporal adjacency positives
- Strategy D: Summary-to-conversation positives (synthetic)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from rich.console import Console
from rich.progress import track

from .schemas import Conversation, TrainingPair

console = Console()

PAIRS_DIR = Path("data/pairs")


# ---------------------------------------------------------------------------
# Conversation windowing
# ---------------------------------------------------------------------------


def format_window(conversation: Conversation, start: int, end: int) -> str:
    """Format a window of messages as a text block.

    Output format:
        speaker_1: message content here
        speaker_2: response content here
    """
    lines = []
    for msg in conversation.messages[start:end]:
        lines.append(f"{msg.author_id}: {msg.content}")
    return "\n".join(lines)


def random_window(
    conversation: Conversation,
    min_size: int = 3,
    max_size: int = 8,
) -> tuple[int, int]:
    """Select a random window within a conversation.

    Returns (start_idx, end_idx) tuple.
    """
    n = conversation.num_messages
    size = random.randint(min_size, min(max_size, n))
    start = random.randint(0, n - size)
    return start, start + size


# ---------------------------------------------------------------------------
# Strategy A: Thread-based positives
# ---------------------------------------------------------------------------


def build_thread_pairs(
    conversations: list[Conversation],
    pairs_per_conversation: int = 3,
    min_window: int = 3,
    max_window: int = 8,
) -> list[TrainingPair]:
    """Generate thread-based positive pairs.

    Anchor: a window of messages from a conversation
    Positive: a different, non-overlapping window from the SAME conversation
    """
    pairs = []

    for conv in track(conversations, description="Strategy A: Thread pairs"):
        if conv.num_messages < min_window * 2:
            continue  # need room for two non-overlapping windows

        for _ in range(pairs_per_conversation):
            # Pick two non-overlapping windows
            mid = conv.num_messages // 2
            w1_start, w1_end = random_window(
                conv, min_size=min_window, max_size=min(max_window, mid)
            )
            # Second window from the other half
            remaining_start = w1_end
            remaining_len = conv.num_messages - remaining_start
            if remaining_len < min_window:
                continue

            w2_size = random.randint(min_window, min(max_window, remaining_len))
            w2_start = random.randint(remaining_start, conv.num_messages - w2_size)

            pairs.append(
                TrainingPair(
                    anchor=format_window(conv, w1_start, w1_end),
                    positive=format_window(conv, w2_start, w2_start + w2_size),
                    strategy="thread_based",
                    source=conv.source,
                    metadata={
                        "conversation_id": conv.id,
                        "anchor_window": [w1_start, w1_end],
                        "positive_window": [w2_start, w2_start + w2_size],
                    },
                )
            )

    return pairs


# ---------------------------------------------------------------------------
# Strategy B: Query-response positives
# ---------------------------------------------------------------------------


def is_question(text: str) -> bool:
    """Simple heuristic to detect questions."""
    return text.rstrip().endswith("?") or text.lower().startswith(
        ("how", "what", "why", "when", "where", "who", "can", "could", "is", "are", "do", "does")
    )


def build_query_response_pairs(
    conversations: list[Conversation],
    min_response_messages: int = 1,
    max_response_messages: int = 5,
) -> list[TrainingPair]:
    """Generate query-response positive pairs.

    Anchor: a question/request message
    Positive: the response message(s)
    """
    pairs = []

    for conv in track(conversations, description="Strategy B: Query-response pairs"):
        for i, msg in enumerate(conv.messages[:-1]):
            if not is_question(msg.content):
                continue

            # Response is the next 1-5 messages
            resp_end = min(i + 1 + max_response_messages, conv.num_messages)
            if resp_end - (i + 1) < min_response_messages:
                continue

            response_window = format_window(conv, i + 1, resp_end)
            anchor = f"{msg.author_id}: {msg.content}"

            pairs.append(
                TrainingPair(
                    anchor=anchor,
                    positive=response_window,
                    strategy="query_response",
                    source=conv.source,
                    metadata={
                        "conversation_id": conv.id,
                        "query_idx": i,
                        "response_window": [i + 1, resp_end],
                    },
                )
            )

    return pairs


# ---------------------------------------------------------------------------
# Strategy C: Temporal adjacency positives
# ---------------------------------------------------------------------------


def build_temporal_pairs(
    conversations: list[Conversation],
    window_size: int = 5,
) -> list[TrainingPair]:
    """Generate temporal adjacency positive pairs.

    Anchor: message window at position T
    Positive: message window immediately following at T+1
    """
    pairs = []

    for conv in track(conversations, description="Strategy C: Temporal pairs"):
        if conv.num_messages < window_size * 2:
            continue

        # Slide through the conversation
        for start in range(0, conv.num_messages - window_size * 2 + 1, window_size):
            anchor_end = start + window_size
            positive_end = anchor_end + window_size

            if positive_end > conv.num_messages:
                break

            pairs.append(
                TrainingPair(
                    anchor=format_window(conv, start, anchor_end),
                    positive=format_window(conv, anchor_end, positive_end),
                    strategy="temporal_adjacency",
                    source=conv.source,
                    metadata={
                        "conversation_id": conv.id,
                        "anchor_window": [start, anchor_end],
                        "positive_window": [anchor_end, positive_end],
                    },
                )
            )

    return pairs


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def save_pairs(pairs: list[TrainingPair], strategy_name: str) -> Path:
    """Save training pairs to JSONL."""
    output_dir = PAIRS_DIR / strategy_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "pairs.jsonl"

    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(pair.model_dump_json() + "\n")

    console.print(
        f"  [green]✓[/] Saved {len(pairs):,} {strategy_name} pairs to {output_path}"
    )
    return output_path


def load_conversations(processed_dir: Path) -> list[Conversation]:
    """Load normalized conversations from a processed directory."""
    convs = []
    for jsonl_file in sorted(processed_dir.rglob("conversations.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                convs.append(Conversation.model_validate_json(line))
    console.print(f"  Loaded {len(convs):,} conversations from {processed_dir}")
    return convs
