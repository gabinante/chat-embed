"""Preprocessors to normalize raw chat data into common Conversation schema.

Each source gets its own preprocessor function that reads from data/raw/{source}/
and writes normalized Conversations to data/processed/{source}/conversations.jsonl
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path

from rich.console import Console
from rich.progress import track

from .schemas import Conversation, DatasetStats, Message

console = Console()
log = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")


# ---------------------------------------------------------------------------
# Common cleaning utilities
# ---------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """Apply common text cleaning to message content."""
    # Replace URLs with token
    text = re.sub(r"https?://\S+", "[URL]", text)
    # Strip Discord markdown (bold, italic, strikethrough, spoiler)
    text = re.sub(r"[*_~|]{1,3}(.+?)[*_~|]{1,3}", r"\1", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Normalize unicode quotes, dashes, etc.
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2014", "—").replace("\u2013", "-")
    return text


def is_bot_message(content: str, author: str) -> bool:
    """Heuristic check for bot/automated messages."""
    bot_patterns = [
        r"^\[bot\]",
        r"^![\w]+",  # command prefix
        r"has joined the",
        r"has left the",
        r"pinned a message",
        r"started a call",
    ]
    if any(re.search(p, content, re.IGNORECASE) for p in bot_patterns):
        return True
    if "bot" in author.lower():
        return True
    return False


def passes_quality_filter(conversation: Conversation, min_messages: int = 3) -> bool:
    """Check if a conversation meets minimum quality standards."""
    if conversation.num_messages < min_messages:
        return False
    # Skip if >50% of messages are very short (under 3 words)
    short_count = sum(1 for m in conversation.messages if len(m.content.split()) < 3)
    if short_count / conversation.num_messages > 0.5:
        return False
    return True


def anonymize_speakers(messages: list[Message]) -> list[Message]:
    """Replace author IDs with speaker_1, speaker_2, etc."""
    author_map: dict[str, str] = {}
    counter = 1
    result = []
    for m in messages:
        if m.author_id not in author_map:
            author_map[m.author_id] = f"speaker_{counter}"
            counter += 1
        result.append(m.model_copy(update={"author_id": author_map[m.author_id]}))
    return result


def save_conversations(conversations: list[Conversation], output_dir: Path) -> DatasetStats:
    """Save conversations to JSONL and return stats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "conversations.jsonl"

    num_messages = 0
    all_participants = set()

    with open(output_path, "w") as f:
        for conv in conversations:
            f.write(conv.model_dump_json() + "\n")
            num_messages += conv.num_messages
            all_participants.update(conv.participants)

    stats = DatasetStats(
        source=conversations[0].source if conversations else "unknown",
        num_conversations=len(conversations),
        num_messages=num_messages,
        num_participants=len(all_participants),
    )

    console.print(
        f"  [green]✓[/] Saved {stats.num_conversations:,} conversations "
        f"({stats.num_messages:,} messages) to {output_path}"
    )
    return stats


# ---------------------------------------------------------------------------
# Source-specific preprocessors
# ---------------------------------------------------------------------------


def preprocess_discord_dialogues(raw_dir: Path) -> list[Conversation]:
    """Process Discord-Dialogues dataset (ChatML format).

    Already structured as exchanges — map to Conversation objects.
    Strip ChatML formatting tokens.
    """
    from datasets import load_from_disk

    console.print("[bold]Processing Discord-Dialogues...[/]")
    ds = load_from_disk(str(raw_dir))

    conversations = []
    # Dataset has 'train' split with ChatML-formatted exchanges
    split = ds["train"] if "train" in ds else ds

    for idx, row in enumerate(track(split, description="Processing exchanges")):
        text = row.get("text", row.get("content", ""))
        if not text:
            continue

        # Parse ChatML format: <|im_start|>role\ncontent<|im_end|>
        turns = re.split(r"<\|im_start\|>", text)
        messages = []
        speaker_counter = 0

        for turn in turns:
            turn = turn.strip()
            if not turn:
                continue
            # Remove end token
            turn = turn.replace("<|im_end|>", "").strip()
            # Split role from content
            parts = turn.split("\n", 1)
            if len(parts) < 2:
                continue
            role = parts[0].strip()
            content = clean_text(parts[1].strip())

            if not content or is_bot_message(content, role):
                continue

            speaker_counter += 1
            messages.append(
                Message(
                    id=f"dd_{idx}_{speaker_counter}",
                    author_id=role,
                    content=content,
                )
            )

        if len(messages) >= 2:
            messages = anonymize_speakers(messages)
            conv = Conversation(
                id=f"discord_dialogues_{idx}",
                source="discord",
                messages=messages,
            )
            if passes_quality_filter(conv):
                conversations.append(conv)

    return conversations


def preprocess_irc_disentangle(raw_dir: Path) -> list[Conversation]:
    """Process IRC Disentanglement dataset.

    Uses disentanglement annotations to group messages into conversations.
    """
    from datasets import load_from_disk

    console.print("[bold]Processing IRC Disentanglement...[/]")
    ds = load_from_disk(str(raw_dir))

    # The dataset has annotations mapping each message to a conversation cluster
    conversations = []
    split = ds["train"] if "train" in ds else ds

    # Group messages by their conversation annotation
    conv_groups: dict[str, list[dict]] = {}

    for row in track(split, description="Grouping messages"):
        # Dataset structure varies — adapt to actual schema
        msg_id = str(row.get("id", row.get("message_id", uuid.uuid4().hex[:8])))
        content = row.get("text", row.get("message", ""))
        author = row.get("author", row.get("sender", row.get("user", "unknown")))
        annotation = str(row.get("annotation", row.get("conversation_id", row.get("cluster", msg_id))))

        if not content or is_bot_message(str(content), str(author)):
            continue

        if annotation not in conv_groups:
            conv_groups[annotation] = []

        conv_groups[annotation].append({
            "id": msg_id,
            "author": str(author),
            "content": clean_text(str(content)),
        })

    # Convert groups to Conversation objects
    for conv_id, msgs in track(conv_groups.items(), description="Building conversations"):
        messages = [
            Message(
                id=m["id"],
                author_id=m["author"],
                content=m["content"],
            )
            for m in msgs
        ]
        messages = anonymize_speakers(messages)
        conv = Conversation(
            id=f"irc_{conv_id}",
            source="irc",
            messages=messages,
            metadata={"channel": "ubuntu"},  # primary channel in this dataset
        )
        if passes_quality_filter(conv):
            conversations.append(conv)

    return conversations


def preprocess_slack_export(raw_dir: Path, source_name: str = "slack") -> list[Conversation]:
    """Process a standard Slack JSON export directory.

    Slack exports have structure: channel_name/YYYY-MM-DD.json
    Each file contains messages with optional thread_ts for threading.
    """
    console.print(f"[bold]Processing Slack export from {raw_dir}...[/]")
    conversations = []

    # Find all JSON message files
    json_files = sorted(raw_dir.rglob("*.json"))
    if not json_files:
        console.print(f"  [yellow]No JSON files found in {raw_dir}[/]")
        return []

    # Group messages by thread
    threads: dict[str, list[dict]] = {}

    for jf in track(json_files, description="Reading Slack messages"):
        # Skip non-message files (channels.json, users.json, etc.)
        if jf.name in ("channels.json", "users.json", "integration_logs.json"):
            continue

        channel = jf.parent.name

        try:
            with open(jf) as f:
                messages = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        if not isinstance(messages, list):
            continue

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            # Skip subtypes (join, leave, bot, etc.)
            if msg.get("subtype") in ("channel_join", "channel_leave", "bot_message", "channel_topic"):
                continue

            content = msg.get("text", "")
            if not content:
                continue

            user = msg.get("user", msg.get("username", "unknown"))
            ts = msg.get("ts", "")
            thread_ts = msg.get("thread_ts", ts)  # thread parent timestamp

            thread_key = f"{channel}:{thread_ts}"
            if thread_key not in threads:
                threads[thread_key] = []

            threads[thread_key].append({
                "id": f"{channel}_{ts}",
                "author": user,
                "content": clean_text(content),
                "ts": ts,
                "channel": channel,
            })

    # Convert threads to Conversations
    for thread_key, msgs in track(threads.items(), description="Building conversations"):
        # Sort by timestamp within thread
        msgs.sort(key=lambda m: m["ts"])
        channel = msgs[0]["channel"]

        messages = [
            Message(
                id=m["id"],
                author_id=m["author"],
                content=m["content"],
            )
            for m in msgs
        ]
        messages = anonymize_speakers(messages)

        conv = Conversation(
            id=f"{source_name}_{thread_key}",
            source="slack",
            messages=messages,
            metadata={"channel": channel},
        )
        if passes_quality_filter(conv):
            conversations.append(conv)

    return conversations


# ---------------------------------------------------------------------------
# Main preprocessing dispatch
# ---------------------------------------------------------------------------


PREPROCESSORS = {
    "discord_dialogues": preprocess_discord_dialogues,
    "irc_disentangle": preprocess_irc_disentangle,
}


def preprocess_dataset(name: str, raw_dir: Path | None = None) -> DatasetStats:
    """Preprocess a raw dataset into normalized conversations."""
    if raw_dir is None:
        raw_dir = Path("data/raw") / name

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data not found at {raw_dir}. Run download first.")

    if name in PREPROCESSORS:
        conversations = PREPROCESSORS[name](raw_dir)
    elif name.startswith("slack_"):
        conversations = preprocess_slack_export(raw_dir, source_name=name)
    else:
        raise ValueError(f"No preprocessor for {name}")

    output_dir = PROCESSED_DIR / name
    return save_conversations(conversations, output_dir)
