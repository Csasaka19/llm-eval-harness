"""Ranked-retrieval metrics with binary relevance.

Conventions (stated here because papers and libraries disagree):

* ``retrieved`` is de-duplicated, keeping the first occurrence, before scoring. A system
  cannot inflate precision by returning the same relevant document several times.
* ``precision@k`` divides by ``k``, not by the number of documents actually returned.
  Returning fewer than ``k`` documents is not rewarded.
* ``recall@k`` divides by the number of relevant documents. It is undefined (raises) when
  there are none; callers should skip such cases rather than score them as 0 or 1.
* ``nDCG@k`` uses gain 1 for relevant documents, discount ``1 / log2(rank + 1)``, and an
  ideal ranking that places ``min(k, |relevant|)`` relevant documents first.
* Reciprocal rank is ``1 / rank`` of the first relevant document over the full ranking,
  or 0 if none is retrieved. MRR is its mean across cases.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence


def dedupe(retrieved: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for doc_id in retrieved:
        if doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
    return out


def _check(relevant: Collection[str], k: int | None = None) -> None:
    if not relevant:
        raise ValueError("metric is undefined when there are no relevant documents")
    if k is not None and k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


def _hits_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> int:
    rel = set(relevant)
    return sum(1 for d in dedupe(retrieved)[:k] if d in rel)


def recall_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    _check(relevant, k)
    return _hits_at_k(retrieved, relevant, k) / len(set(relevant))


def precision_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    _check(relevant, k)
    return _hits_at_k(retrieved, relevant, k) / k


def hit_rate_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    _check(relevant, k)
    return 1.0 if _hits_at_k(retrieved, relevant, k) > 0 else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: Collection[str]) -> float:
    _check(relevant)
    rel = set(relevant)
    for rank, doc_id in enumerate(dedupe(retrieved), start=1):
        if doc_id in rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    _check(relevant, k)
    rel = set(relevant)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(dedupe(retrieved)[:k], start=1)
        if doc_id in rel
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(rel)) + 1))
    return dcg / ideal


def retrieval_scores(
    retrieved: Sequence[str], relevant: Collection[str], ks: Sequence[int]
) -> dict[str, float]:
    """All retrieval metrics for one case, keyed like ``recall@5`` and ``mrr``."""
    scores: dict[str, float] = {"mrr": reciprocal_rank(retrieved, relevant)}
    for k in ks:
        scores[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        scores[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        scores[f"hit_rate@{k}"] = hit_rate_at_k(retrieved, relevant, k)
        scores[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)
    return scores
