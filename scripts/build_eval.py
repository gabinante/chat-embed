#!/usr/bin/env python3
"""Build evaluation datasets for the chat retrieval benchmark.

Creates held-out test sets from processed conversations for three tasks:
1. Thread Retrieval — given a message, find its parent thread
2. Response Retrieval — given a prefix, find the continuation
3. Cross-Platform Transfer — evaluate on held-out platforms

Usage:
    python scripts/build_eval.py
    python scripts/build_eval.py --num-queries 500 --pool-size 100
"""

import json
import random
import sys
from pathlib import Path

import click
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thread_embed.data.pair_builders import format_window, load_conversations
from thread_embed.data.schemas import Conversation

console = Console()

PROCESSED_DIR = Path("data/processed")
EVAL_DIR = Path("data/eval")


def build_thread_retrieval(
    conversations: list[Conversation],
    num_queries: int = 500,
    pool_size: int = 100,
    min_messages: int = 8,
    seed: int = 42,
) -> list[dict]:
    """Build thread retrieval eval set.

    Task: given a single query message extracted from a conversation,
    retrieve the correct full thread from a pool of distractors.

    Returns list of eval examples, each with:
        - query: str (a single message)
        - corpus: list[str] (pool_size thread texts, one correct)
        - relevant_idx: int (index of correct thread in corpus)
        - source: str
    """
    rng = random.Random(seed)

    # Filter conversations with enough messages
    eligible = [c for c in conversations if c.num_messages >= min_messages]
    if len(eligible) < pool_size:
        console.print(f"[yellow]Only {len(eligible)} eligible conversations (need {pool_size})[/]")
        return []

    rng.shuffle(eligible)
    examples = []

    for i in range(min(num_queries, len(eligible) - pool_size)):
        target_conv = eligible[i]

        # Extract a query message from the middle of the conversation
        mid = target_conv.num_messages // 2
        query_idx = rng.randint(max(1, mid - 2), min(mid + 2, target_conv.num_messages - 2))
        query = f"{target_conv.messages[query_idx].author_id}: {target_conv.messages[query_idx].content}"

        # Build the target thread text (excluding the query message)
        thread_msgs = [
            f"{m.author_id}: {m.content}"
            for j, m in enumerate(target_conv.messages)
            if j != query_idx
        ]
        target_thread = "\n".join(thread_msgs)

        # Sample distractors
        distractor_pool = [c for c in eligible if c.id != target_conv.id]
        distractors = rng.sample(distractor_pool, min(pool_size - 1, len(distractor_pool)))

        corpus = []
        for d in distractors:
            thread_text = "\n".join(f"{m.author_id}: {m.content}" for m in d.messages)
            corpus.append(thread_text)

        # Insert target at random position
        insert_pos = rng.randint(0, len(corpus))
        corpus.insert(insert_pos, target_thread)

        examples.append({
            "query": query,
            "corpus": corpus,
            "relevant_idx": insert_pos,
            "source": target_conv.source,
            "conversation_id": target_conv.id,
        })

    return examples


def build_response_retrieval(
    conversations: list[Conversation],
    num_queries: int = 500,
    pool_size: int = 100,
    prefix_size: int = 5,
    response_size: int = 5,
    min_messages: int = 12,
    seed: int = 42,
) -> list[dict]:
    """Build response retrieval eval set.

    Task: given a conversation prefix, retrieve the correct continuation
    from a pool of distractors.
    """
    rng = random.Random(seed)

    eligible = [c for c in conversations if c.num_messages >= min_messages]
    if len(eligible) < pool_size:
        console.print(f"[yellow]Only {len(eligible)} eligible convos for response retrieval[/]")
        return []

    rng.shuffle(eligible)
    examples = []

    for i in range(min(num_queries, len(eligible) - pool_size)):
        conv = eligible[i]

        # Split: prefix | response | rest
        split_point = rng.randint(prefix_size, conv.num_messages - response_size)
        prefix = format_window(conv, split_point - prefix_size, split_point)
        response = format_window(conv, split_point, split_point + response_size)

        # Sample distractor responses from other conversations
        distractor_pool = [c for c in eligible if c.id != conv.id and c.num_messages >= response_size * 2]
        distractors = rng.sample(distractor_pool, min(pool_size - 1, len(distractor_pool)))

        corpus = []
        for d in distractors:
            d_start = rng.randint(0, d.num_messages - response_size)
            corpus.append(format_window(d, d_start, d_start + response_size))

        insert_pos = rng.randint(0, len(corpus))
        corpus.insert(insert_pos, response)

        examples.append({
            "query": prefix,
            "corpus": corpus,
            "relevant_idx": insert_pos,
            "source": conv.source,
            "conversation_id": conv.id,
        })

    return examples


@click.command()
@click.option("--num-queries", default=500, type=int, help="Number of eval queries per task")
@click.option("--pool-size", default=100, type=int, help="Retrieval pool size (distractors + 1)")
@click.option("--seed", default=42, type=int)
@click.option("--max-convos-per-source", default=10000, type=int, help="Max conversations per source for eval")
def main(num_queries: int, pool_size: int, seed: int, max_convos_per_source: int):
    """Build evaluation datasets."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Load conversations from all sources
    source_dirs = sorted(d for d in PROCESSED_DIR.iterdir() if d.is_dir())

    all_conversations = []
    platform_conversations: dict[str, list[Conversation]] = {}

    for d in source_dirs:
        convos = load_conversations(d, max_conversations=max_convos_per_source, seed=seed)
        all_conversations.extend(convos)
        source_name = d.name
        platform_conversations[source_name] = convos

    console.print(f"\n[bold]Total eval conversations: {len(all_conversations):,}[/]")

    # ---------------------------------------------------------------
    # Task 1: Thread Retrieval (all platforms mixed)
    # ---------------------------------------------------------------
    console.print("\n[bold cyan]Task 1: Thread Retrieval[/]")
    thread_examples = build_thread_retrieval(
        all_conversations, num_queries=num_queries, pool_size=pool_size, seed=seed
    )
    if thread_examples:
        out = EVAL_DIR / "thread_retrieval.jsonl"
        with open(out, "w") as f:
            for ex in thread_examples:
                f.write(json.dumps(ex) + "\n")
        console.print(f"  [green]✓[/] {len(thread_examples):,} examples → {out}")

    # ---------------------------------------------------------------
    # Task 2: Response Retrieval (all platforms mixed)
    # ---------------------------------------------------------------
    console.print("\n[bold cyan]Task 2: Response Retrieval[/]")
    response_examples = build_response_retrieval(
        all_conversations, num_queries=num_queries, pool_size=pool_size, seed=seed + 1
    )
    if response_examples:
        out = EVAL_DIR / "response_retrieval.jsonl"
        with open(out, "w") as f:
            for ex in response_examples:
                f.write(json.dumps(ex) + "\n")
        console.print(f"  [green]✓[/] {len(response_examples):,} examples → {out}")

    # ---------------------------------------------------------------
    # Task 3: Cross-Platform Transfer (per-platform eval)
    # ---------------------------------------------------------------
    console.print("\n[bold cyan]Task 3: Cross-Platform Transfer[/]")
    xplatform_dir = EVAL_DIR / "cross_platform"
    xplatform_dir.mkdir(parents=True, exist_ok=True)

    for platform, convos in platform_conversations.items():
        if len(convos) < pool_size + 10:
            console.print(f"  [yellow]Skipping {platform}: too few conversations ({len(convos)})[/]")
            continue

        per_platform_queries = min(num_queries // len(platform_conversations), len(convos) - pool_size)
        examples = build_thread_retrieval(
            convos, num_queries=per_platform_queries, pool_size=pool_size, seed=seed + 2
        )
        if examples:
            out = xplatform_dir / f"{platform}.jsonl"
            with open(out, "w") as f:
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")
            console.print(f"  [green]✓[/] {platform}: {len(examples):,} examples → {out}")

    console.print("\n[bold green]Eval data built![/]")


if __name__ == "__main__":
    main()
