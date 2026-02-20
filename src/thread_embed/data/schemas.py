"""Pydantic models for ThreadEmbed data structures."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message within a conversation."""

    id: str
    author_id: str  # anonymized speaker identifier
    content: str  # cleaned text content
    timestamp: datetime | None = None
    reply_to: str | None = None  # parent message id if threaded


class Conversation(BaseModel):
    """A sequence of related messages forming a conversation or thread."""

    id: str
    source: str  # "discord", "irc", "slack", etc.
    messages: list[Message]
    metadata: dict = Field(default_factory=dict)  # channel, topic, etc.

    @property
    def num_messages(self) -> int:
        return len(self.messages)

    @property
    def participants(self) -> set[str]:
        return {m.author_id for m in self.messages}


class TrainingPair(BaseModel):
    """A contrastive training pair for embedding model training."""

    anchor: str  # concatenated message window
    positive: str  # related message window
    negative: str | None = None  # explicit hard negative (optional, can use in-batch)
    strategy: str  # which strategy generated this pair
    source: str  # dataset source
    metadata: dict = Field(default_factory=dict)  # conversation_id, window positions, etc.


class DatasetStats(BaseModel):
    """Statistics for a downloaded/processed dataset."""

    source: str
    num_conversations: int = 0
    num_messages: int = 0
    num_participants: int = 0
    date_range: tuple[str, str] | None = None
    raw_size_bytes: int = 0
