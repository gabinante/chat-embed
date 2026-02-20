"""Tests for training pair construction."""

from thread_embed.data.pair_builders import (
    build_query_response_pairs,
    build_temporal_pairs,
    build_thread_pairs,
    format_window,
    is_question,
)
from thread_embed.data.schemas import Conversation, Message


def _make_conversation(n_messages: int = 10) -> Conversation:
    """Create a test conversation with n messages."""
    return Conversation(
        id="test_conv",
        source="test",
        messages=[
            Message(
                id=f"msg_{i}",
                author_id=f"speaker_{i % 3 + 1}",
                content=f"This is message number {i} with enough words to pass filters",
            )
            for i in range(n_messages)
        ],
    )


def test_format_window():
    conv = _make_conversation(5)
    text = format_window(conv, 0, 3)
    assert "speaker_1:" in text
    assert "message number 0" in text
    assert text.count("\n") == 2


def test_is_question():
    assert is_question("How do I install this?")
    assert is_question("What is the best approach?")
    assert not is_question("I think we should do it this way.")


def test_thread_pairs_generated():
    conv = _make_conversation(20)
    pairs = build_thread_pairs([conv], pairs_per_conversation=2)
    assert len(pairs) > 0
    assert all(p.strategy == "thread_based" for p in pairs)


def test_query_response_pairs():
    conv = Conversation(
        id="qa_conv",
        source="test",
        messages=[
            Message(id="1", author_id="a", content="How do I fix this error with the build?"),
            Message(id="2", author_id="b", content="Try clearing the cache and rebuilding"),
            Message(id="3", author_id="b", content="That usually fixes it for me"),
            Message(id="4", author_id="a", content="Thanks that worked great"),
        ],
    )
    pairs = build_query_response_pairs([conv])
    assert len(pairs) >= 1
    assert pairs[0].strategy == "query_response"


def test_temporal_pairs():
    conv = _make_conversation(20)
    pairs = build_temporal_pairs([conv], window_size=5)
    assert len(pairs) > 0
    assert all(p.strategy == "temporal_adjacency" for p in pairs)
