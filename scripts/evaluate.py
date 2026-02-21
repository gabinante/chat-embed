#!/usr/bin/env python3
"""Evaluate models on the chat retrieval benchmark.

Usage:
    python scripts/evaluate.py --model models/thread-embed-v1
    python scripts/evaluate.py --model BAAI/bge-base-en-v1.5  # baseline
    python scripts/evaluate.py --model openai:text-embedding-3-small  # OpenAI API
    python scripts/evaluate.py --compare models/thread-embed-v1 BAAI/bge-base-en-v1.5
"""

import json
import os
import sys
import time
from pathlib import Path

import click
import numpy as np
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thread_embed.eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k

load_dotenv()

console = Console()

EVAL_DIR = Path("data/eval")

OPENAI_PREFIX = "openai:"


def _truncate_text(text: str, max_chars: int = 20000) -> str:
    """Truncate text to stay within API token limits (~3 chars/token for chat data)."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


class OpenAIEmbeddingModel:
    """Wrapper around OpenAI embeddings API to match SentenceTransformer interface."""

    def __init__(self, model_name: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model_name = model_name
        self._request_count = 0

    def encode(self, texts: list[str], batch_size: int = 64, normalize_embeddings: bool = True) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = [_truncate_text(t) for t in texts[i:i + batch_size]]
            self._request_count += 1
            if self._request_count % 50 == 0:
                time.sleep(0.5)
            response = self.client.embeddings.create(input=batch, model=self.model_name)
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)
        result = np.array(all_embeddings, dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            result = result / np.maximum(norms, 1e-12)
        return result


VOYAGE_PREFIX = "voyage:"


class VoyageEmbeddingModel:
    """Wrapper around Voyage AI embeddings API."""

    def __init__(self, model_name: str):
        import voyageai
        self.client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
        self.model_name = model_name
        self._request_count = 0

    def encode(self, texts: list[str], batch_size: int = 64, normalize_embeddings: bool = True) -> np.ndarray:
        all_embeddings = []
        # Voyage batch limit is 128 texts, use smaller batches to be safe
        chunk_size = min(batch_size, 64)
        for i in range(0, len(texts), chunk_size):
            batch = [_truncate_text(t) for t in texts[i:i + chunk_size]]
            self._request_count += 1
            if self._request_count % 50 == 0:
                time.sleep(0.5)
            result = self.client.embed(batch, model=self.model_name)
            all_embeddings.extend(result.embeddings)
        result = np.array(all_embeddings, dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            result = result / np.maximum(norms, 1e-12)
        return result


def load_eval_task(filepath: Path) -> list[dict]:
    """Load eval examples from JSONL."""
    examples = []
    with open(filepath) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def evaluate_model_on_task(
    model,
    examples: list[dict],
    task_name: str,
    model_name: str,
    batch_size: int = 64,
) -> dict:
    """Evaluate a model on a retrieval task.

    Each example has: query, corpus (list of texts), relevant_idx.
    """
    all_results = []

    # Process in chunks to avoid memory issues
    for ex in examples:
        query = ex["query"]
        corpus = ex["corpus"]
        relevant_idx = ex["relevant_idx"]

        # Encode
        q_emb = model.encode([query], batch_size=1, normalize_embeddings=True)
        c_embs = model.encode(corpus, batch_size=batch_size, normalize_embeddings=True)

        # Compute similarities
        similarities = (q_emb @ c_embs.T)[0]
        ranked_indices = np.argsort(-similarities)

        all_results.append({
            "relevant_id": str(relevant_idx),
            "retrieved_ids": [str(idx) for idx in ranked_indices],
        })

    metrics = {
        "task": task_name,
        "model": model_name,
        "MRR@10": round(mrr_at_k(all_results, k=10), 4),
        "R@1": round(recall_at_k(all_results, k=1), 4),
        "R@5": round(recall_at_k(all_results, k=5), 4),
        "R@10": round(recall_at_k(all_results, k=10), 4),
        "NDCG@10": round(ndcg_at_k(all_results, k=10), 4),
        "n_queries": len(examples),
    }
    return metrics


def print_results_table(all_results: list[dict]):
    """Print a rich table of benchmark results."""
    table = Table(title="Chat Retrieval Benchmark Results")
    table.add_column("Task", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("MRR@10", justify="right")
    table.add_column("R@1", justify="right")
    table.add_column("R@5", justify="right")
    table.add_column("R@10", justify="right")
    table.add_column("NDCG@10", justify="right")
    table.add_column("N", justify="right", style="dim")

    for r in sorted(all_results, key=lambda x: (x["task"], x["model"])):
        table.add_row(
            r["task"],
            r["model"],
            f"{r['MRR@10']:.4f}",
            f"{r['R@1']:.4f}",
            f"{r['R@5']:.4f}",
            f"{r['R@10']:.4f}",
            f"{r['NDCG@10']:.4f}",
            str(r["n_queries"]),
        )

    console.print(table)


@click.command()
@click.option("--model", multiple=True, help="Model path(s) or HuggingFace ID(s)")
@click.option("--compare", multiple=True, help="Additional models to compare")
@click.option("--eval-data", default="data/eval", help="Path to evaluation data")
@click.option("--batch-size", default=64, type=int)
@click.option("--output", default=None, help="Save results to JSON file")
def main(model: tuple, compare: tuple, eval_data: str, batch_size: int, output: str | None):
    """Run benchmark evaluation."""
    from sentence_transformers import SentenceTransformer

    eval_path = Path(eval_data)

    # Collect all models to evaluate
    model_ids = list(model) + list(compare)
    if not model_ids:
        console.print("[red]Specify at least one model with --model[/]")
        return

    # Discover eval tasks
    tasks = {}
    if (eval_path / "thread_retrieval.jsonl").exists():
        tasks["Thread Retrieval"] = load_eval_task(eval_path / "thread_retrieval.jsonl")
    if (eval_path / "response_retrieval.jsonl").exists():
        tasks["Response Retrieval"] = load_eval_task(eval_path / "response_retrieval.jsonl")

    # Cross-platform tasks
    xplatform = eval_path / "cross_platform"
    if xplatform.exists():
        for f in sorted(xplatform.glob("*.jsonl")):
            platform = f.stem
            tasks[f"XPlatform/{platform}"] = load_eval_task(f)

    if not tasks:
        console.print("[red]No eval data found. Run: python scripts/build_eval.py[/]")
        return

    console.print(f"[bold]Found {len(tasks)} eval tasks:[/]")
    for name, examples in tasks.items():
        console.print(f"  {name}: {len(examples):,} queries")

    all_results = []

    for model_id in model_ids:
        console.print(f"\n[bold]Loading model: {model_id}[/]")
        if model_id.startswith(OPENAI_PREFIX):
            openai_model = model_id[len(OPENAI_PREFIX):]
            st_model = OpenAIEmbeddingModel(openai_model)
        elif model_id.startswith(VOYAGE_PREFIX):
            voyage_model = model_id[len(VOYAGE_PREFIX):]
            st_model = VoyageEmbeddingModel(voyage_model)
        else:
            st_model = SentenceTransformer(model_id)

        for task_name, examples in tasks.items():
            console.print(f"  Evaluating on {task_name}...")
            result = evaluate_model_on_task(
                st_model, examples, task_name, model_id, batch_size=batch_size
            )
            all_results.append(result)

        # Free memory
        del st_model

    # Print results
    console.print()
    print_results_table(all_results)

    # Save results
    if output:
        with open(output, "w") as f:
            json.dump(all_results, f, indent=2)
        console.print(f"\n[green]Results saved to {output}[/]")


if __name__ == "__main__":
    main()
