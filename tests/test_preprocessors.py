"""Tests for data preprocessing utilities."""

from thread_embed.data.preprocessors import anonymize_speakers, clean_text, is_bot_message, passes_quality_filter
from thread_embed.data.schemas import Conversation, Message


def test_clean_text_urls():
    assert "[URL]" in clean_text("check out https://example.com for details")


def test_clean_text_discord_markdown():
    assert clean_text("**bold** and *italic*") == "bold and italic"


def test_clean_text_whitespace():
    assert clean_text("  too   many   spaces  ") == "too many spaces"


def test_is_bot_message():
    assert is_bot_message("!help command", "user123")
    assert is_bot_message("hello", "ModeratorBot")
    assert not is_bot_message("hello world", "alice")


def test_anonymize_speakers():
    messages = [
        Message(id="1", author_id="alice", content="hi"),
        Message(id="2", author_id="bob", content="hello"),
        Message(id="3", author_id="alice", content="how are you"),
    ]
    result = anonymize_speakers(messages)
    assert result[0].author_id == "speaker_1"
    assert result[1].author_id == "speaker_2"
    assert result[2].author_id == "speaker_1"


def test_passes_quality_filter():
    good = Conversation(
        id="1",
        source="test",
        messages=[
            Message(id="1", author_id="a", content="hello how are you doing"),
            Message(id="2", author_id="b", content="I am doing well thanks"),
            Message(id="3", author_id="a", content="great to hear that"),
        ],
    )
    assert passes_quality_filter(good)

    # Too few messages
    short = Conversation(
        id="2",
        source="test",
        messages=[
            Message(id="1", author_id="a", content="hi"),
            Message(id="2", author_id="b", content="bye"),
        ],
    )
    assert not passes_quality_filter(short)
