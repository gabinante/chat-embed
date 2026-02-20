"""Tests for evaluation metrics."""

from thread_embed.eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k


def test_mrr_perfect():
    results = [{"relevant_id": "0", "retrieved_ids": ["0", "1", "2"]}]
    assert mrr_at_k(results, k=10) == 1.0


def test_mrr_second_position():
    results = [{"relevant_id": "1", "retrieved_ids": ["0", "1", "2"]}]
    assert mrr_at_k(results, k=10) == 0.5


def test_mrr_not_found():
    results = [{"relevant_id": "99", "retrieved_ids": ["0", "1", "2"]}]
    assert mrr_at_k(results, k=3) == 0.0


def test_recall_at_1():
    results = [
        {"relevant_id": "0", "retrieved_ids": ["0", "1", "2"]},
        {"relevant_id": "2", "retrieved_ids": ["0", "1", "2"]},
    ]
    assert recall_at_k(results, k=1) == 0.5


def test_recall_at_5():
    results = [
        {"relevant_id": "3", "retrieved_ids": ["0", "1", "2", "3", "4"]},
    ]
    assert recall_at_k(results, k=5) == 1.0


def test_ndcg_first_position():
    results = [{"relevant_id": "0", "retrieved_ids": ["0", "1", "2"]}]
    assert ndcg_at_k(results, k=10) == 1.0


def test_ndcg_second_position():
    results = [{"relevant_id": "1", "retrieved_ids": ["0", "1", "2"]}]
    ndcg = ndcg_at_k(results, k=10)
    assert 0 < ndcg < 1.0
