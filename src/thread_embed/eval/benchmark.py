"""Chat retrieval benchmark runner.

Implements four benchmark tasks:
1. Thread Retrieval — given a query message, find the correct thread
2. Response Retrieval — given a conversation prefix, find the next window
3. Summary-to-Thread Matching — given a description, find the matching conversation
4. Cross-Platform Transfer — evaluate on held-out platform data
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from .metrics import mrr_at_k, ndcg_at_k, recall_at_k


@dataclass
class BenchmarkResult:
    task_name: str
    model_name: str
    mrr_at_10: float
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    ndcg_at_10: float
    num_queries: int

    def to_dict(self) -> dict:
        return {
            "task": self.task_name,
            "model": self.model_name,
            "MRR@10": round(self.mrr_at_10, 4),
            "R@1": round(self.recall_at_1, 4),
            "R@5": round(self.recall_at_5, 4),
            "R@10": round(self.recall_at_10, 4),
            "NDCG@10": round(self.ndcg_at_10, 4),
            "n_queries": self.num_queries,
        }


def run_retrieval_eval(
    model: SentenceTransformer,
    queries: list[str],
    corpus: list[str],
    relevant_ids: list[int],
    model_name: str = "",
    task_name: str = "",
    batch_size: int = 128,
) -> BenchmarkResult:
    """Run a retrieval evaluation.

    Args:
        model: The embedding model to evaluate
        queries: List of query texts
        corpus: List of corpus document texts
        relevant_ids: For each query, the index in corpus of the relevant doc
        model_name: Name for reporting
        task_name: Benchmark task name
        batch_size: Encoding batch size
    """
    # Encode
    query_embeddings = model.encode(queries, batch_size=batch_size, show_progress_bar=True)
    corpus_embeddings = model.encode(corpus, batch_size=batch_size, show_progress_bar=True)

    # Normalize for cosine similarity
    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

    # Compute similarities and rank
    similarities = query_embeddings @ corpus_embeddings.T
    results = []

    for i in range(len(queries)):
        ranked_indices = np.argsort(-similarities[i])
        results.append({
            "relevant_id": str(relevant_ids[i]),
            "retrieved_ids": [str(idx) for idx in ranked_indices],
        })

    return BenchmarkResult(
        task_name=task_name,
        model_name=model_name,
        mrr_at_10=mrr_at_k(results, k=10),
        recall_at_1=recall_at_k(results, k=1),
        recall_at_5=recall_at_k(results, k=5),
        recall_at_10=recall_at_k(results, k=10),
        ndcg_at_10=ndcg_at_k(results, k=10),
        num_queries=len(queries),
    )
