"""Preprocessors to normalize raw chat data into common Conversation schema.

Each source gets its own preprocessor function that reads from data/raw/{source}/
and writes normalized Conversations to data/processed/{source}/conversations.jsonl
"""

from __future__ import annotations

import json
import logging
import re
import uuid
import xml.etree.ElementTree as ET
from collections import defaultdict
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
    """Process IRC Disentanglement dataset from GitHub repo.

    Data format:
    - data/{split}/{date}.{split_id}.ascii.txt — IRC messages: [HH:MM] <user> text
    - data/{split}/{date}.{split_id}.annotation.txt — parent_id msg_id annotator
      The first column is the parent message ID that this message responds to.
      Messages that start a new conversation have parent_id == msg_id.

    We use the annotation graph to group messages into conversation trees.
    """
    console.print("[bold]Processing IRC Disentanglement...[/]")
    data_dir = raw_dir / "data"
    conversations = []
    file_count = 0

    # Process all splits (train, dev, test)
    for split_dir in sorted(data_dir.iterdir()):
        if not split_dir.is_dir():
            continue

        # Find all annotation files in this split
        annotation_files = sorted(split_dir.glob("*.annotation.txt"))

        for ann_file in track(annotation_files, description=f"Processing IRC {split_dir.name}"):
            # Derive the corresponding ascii file
            ascii_file = Path(str(ann_file).replace(".annotation.txt", ".ascii.txt"))
            if not ascii_file.exists():
                continue

            file_count += 1

            # Parse messages from ascii file
            # Lines: [HH:MM] <username> message text
            # Or: === username [~info] has joined/left
            msg_lines: dict[int, dict] = {}
            line_num = 1000  # IRC disentangle uses 1000-based line numbers

            with open(ascii_file, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.rstrip("\n")
                    # Parse IRC message: [HH:MM] <user> content
                    m = re.match(r"\[(\d{2}:\d{2})\]\s+<([^>]+)>\s+(.*)", line)
                    if m:
                        timestamp, author, content = m.groups()
                        content = clean_text(content)
                        if content and not is_bot_message(content, author):
                            msg_lines[line_num] = {
                                "id": str(line_num),
                                "author": author.strip(),
                                "content": content,
                                "timestamp": timestamp,
                            }
                    line_num += 1

            # Parse annotation file to build conversation graph
            # Format: parent_id msg_id annotator
            # parent_id == msg_id means this starts a new conversation
            conv_roots: dict[int, list[int]] = {}  # root_id -> [msg_ids in order]
            parent_map: dict[int, int] = {}

            with open(ann_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    parent_id = int(parts[0])
                    msg_id = int(parts[1])
                    parent_map[msg_id] = parent_id

            # Trace each message to its conversation root
            def find_root(mid: int) -> int:
                visited = set()
                while parent_map.get(mid, mid) != mid:
                    if mid in visited:
                        break
                    visited.add(mid)
                    mid = parent_map[mid]
                return mid

            root_groups: dict[int, list[int]] = defaultdict(list)
            for mid in sorted(parent_map.keys()):
                root = find_root(mid)
                root_groups[root].append(mid)

            # Convert to Conversation objects
            date_prefix = ann_file.stem.split(".")[0]
            for root_id, msg_ids in root_groups.items():
                messages = []
                for mid in msg_ids:
                    if mid in msg_lines:
                        ml = msg_lines[mid]
                        messages.append(
                            Message(
                                id=ml["id"],
                                author_id=ml["author"],
                                content=ml["content"],
                            )
                        )

                if len(messages) >= 2:
                    messages = anonymize_speakers(messages)
                    conv = Conversation(
                        id=f"irc_{date_prefix}_{root_id}",
                        source="irc",
                        messages=messages,
                        metadata={"channel": "ubuntu", "date": date_prefix},
                    )
                    if passes_quality_filter(conv):
                        conversations.append(conv)

    console.print(f"  Processed {file_count} IRC log files")
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


def preprocess_slack_disentangled(raw_dir: Path) -> list[Conversation]:
    """Process Software-related Slack Chats with disentangled conversations.

    Data format: XML files with <message conversation_id="N"> elements.
    Structure: data/{community}/{year}/merged-{channel}.xml
    """
    console.print("[bold]Processing Slack Disentangled Chats...[/]")
    data_dir = raw_dir / "data"
    conversations = []

    xml_files = sorted(data_dir.rglob("*.xml"))
    console.print(f"  Found {len(xml_files)} XML files")

    for xml_file in track(xml_files, description="Processing Slack XML"):
        community = xml_file.parts[-3]  # e.g. "pythondev"

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
        except ET.ParseError as e:
            console.print(f"  [yellow]Parse error in {xml_file}: {e}[/]")
            continue

        # Group messages by conversation_id
        conv_groups: dict[str, list[dict]] = defaultdict(list)
        channel = root.findtext("channel_name", "unknown")

        for msg_elem in root.findall("message"):
            conv_id = msg_elem.get("conversation_id", "0")
            user = msg_elem.findtext("user", "unknown")
            text = msg_elem.findtext("text", "")
            ts = msg_elem.findtext("ts", "")

            if not text or is_bot_message(text, user):
                continue

            conv_groups[conv_id].append({
                "user": user,
                "text": clean_text(text),
                "ts": ts,
            })

        # Convert groups to Conversations
        for conv_id, msgs in conv_groups.items():
            messages = [
                Message(
                    id=f"slack_{community}_{conv_id}_{i}",
                    author_id=m["user"],
                    content=m["text"],
                )
                for i, m in enumerate(msgs)
            ]
            messages = anonymize_speakers(messages)

            conv = Conversation(
                id=f"slack_disentangled_{community}_{conv_id}",
                source="slack",
                messages=messages,
                metadata={"channel": channel, "community": community},
            )
            if passes_quality_filter(conv):
                conversations.append(conv)

    return conversations


def preprocess_flyte_slack(raw_dir: Path) -> list[Conversation]:
    """Process Flyte Slack Q&A pairs (input/output format from HuggingFace).

    Each row has an 'input' and 'output' field representing a conversation turn pair.
    We group consecutive pairs into conversations.
    """
    from datasets import load_from_disk

    console.print("[bold]Processing Flyte Slack Data...[/]")
    ds = load_from_disk(str(raw_dir))
    split = ds["train"] if "train" in ds else ds

    conversations = []
    for idx, row in enumerate(track(split, description="Processing Flyte Slack")):
        input_text = clean_text(str(row.get("input", "")))
        output_text = clean_text(str(row.get("output", "")))

        if not input_text or not output_text:
            continue

        messages = [
            Message(id=f"flyte_{idx}_0", author_id="speaker_1", content=input_text),
            Message(id=f"flyte_{idx}_1", author_id="speaker_2", content=output_text),
        ]

        conv = Conversation(
            id=f"flyte_slack_{idx}",
            source="slack",
            messages=messages,
            metadata={"community": "flyte"},
        )
        # Relax quality filter for 2-message Q&A pairs
        if len(input_text.split()) >= 3 and len(output_text.split()) >= 3:
            conversations.append(conv)

    return conversations


# ---------------------------------------------------------------------------
# Main preprocessing dispatch
# ---------------------------------------------------------------------------


PREPROCESSORS = {
    "discord_dialogues": preprocess_discord_dialogues,
    "irc_disentangle": preprocess_irc_disentangle,
    "slack_dev_chats_disentangled": preprocess_slack_disentangled,
    "flyte_slack": preprocess_flyte_slack,
    "flyte_slack_long": preprocess_flyte_slack,  # same format
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
