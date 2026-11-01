"""Retrieval metrics: P@K, R@K, AP, mAP, graded nDCG."""

from __future__ import annotations

import math


def precision_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = ranked_ids[:k]
    return sum(1 for i in top if i in relevant) / k


def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(ranked_ids[:k])
    return len(top & relevant) / len(relevant)


def average_precision(ranked_ids: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    hit = 0
    sum_prec = 0.0
    for i, doc_id in enumerate(ranked_ids, 1):
        if doc_id in relevant:
            hit += 1
            sum_prec += hit / i
    return sum_prec / len(relevant)


def mean_average_precision(aps: list[float]) -> float:
    return sum(aps) / len(aps) if aps else 0.0


def dcg_at_k(gains: list[float], k: int) -> float:
    total = 0.0
    for i, g in enumerate(gains[:k], 1):
        total += (2**g - 1) / math.log2(i + 1)
    return total


def ndcg_at_k(ranked_ids: list[str], graded: dict[str, float], k: int = 10) -> float:
    """graded: id -> relevance in [0,1] (or any non-negative gain)."""
    gains = [float(graded.get(i, 0.0)) for i in ranked_ids[:k]]
    ideal = sorted(graded.values(), reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg <= 0:
        return 0.0
    return dcg_at_k(gains, k) / idcg
