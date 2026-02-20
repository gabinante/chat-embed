"""Retrieval evaluation metrics: MRR, NDCG, Recall@k."""

from __future__ import annotations

import numpy as np


def reciprocal_rank(relevant_idx: int, retrieved: list[str], relevant_id: str) -> float:
    """Reciprocal rank of the first relevant result."""
    for i, doc_id in enumerate(retrieved):
        if doc_id == relevant_id:
            return 1.0 / (i + 1)
    return 0.0


def mrr_at_k(results: list[dict], k: int = 10) -> float:
    """Mean Reciprocal Rank @ k.

    Each result dict has:
        - "relevant_id": str — the correct document ID
        - "retrieved_ids": list[str] — ranked list of retrieved document IDs
    """
    rrs = []
    for r in results:
        retrieved = r["retrieved_ids"][:k]
        rr = reciprocal_rank(0, retrieved, r["relevant_id"])
        rrs.append(rr)
    return float(np.mean(rrs))


def recall_at_k(results: list[dict], k: int = 10) -> float:
    """Recall @ k (binary: is the relevant doc in top-k?)."""
    hits = sum(1 for r in results if r["relevant_id"] in r["retrieved_ids"][:k])
    return hits / len(results) if results else 0.0


def ndcg_at_k(results: list[dict], k: int = 10) -> float:
    """NDCG @ k (binary relevance)."""
    ndcgs = []
    for r in results:
        retrieved = r["retrieved_ids"][:k]
        dcg = 0.0
        for i, doc_id in enumerate(retrieved):
            if doc_id == r["relevant_id"]:
                dcg = 1.0 / np.log2(i + 2)  # log2(rank+1), 1-indexed
                break
        idcg = 1.0  # best case: relevant doc at position 1
        ndcgs.append(dcg / idcg)
    return float(np.mean(ndcgs))
