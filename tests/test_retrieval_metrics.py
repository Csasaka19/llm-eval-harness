"""Expected values are computed by hand in the comments, not by re-running the code."""

import math

import pytest

from llm_eval.metrics.retrieval import (
    dedupe,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    retrieval_scores,
)

# Ranking used throughout: relevant docs are a and c, at ranks 2 and 4.
RANKED = ["x", "a", "y", "c", "z"]
RELEVANT = {"a", "c"}


def test_recall() -> None:
    assert recall_at_k(RANKED, RELEVANT, 1) == 0.0  # top1 = x
    assert recall_at_k(RANKED, RELEVANT, 3) == 0.5  # a found, c not yet
    assert recall_at_k(RANKED, RELEVANT, 5) == 1.0
    assert recall_at_k(RANKED, RELEVANT, 100) == 1.0  # k beyond list length


def test_precision_divides_by_k_not_by_returned() -> None:
    assert precision_at_k(RANKED, RELEVANT, 2) == 0.5  # 1 of 2
    assert precision_at_k(RANKED, RELEVANT, 5) == 0.4  # 2 of 5
    # Only 1 doc returned, but k=5: 1/5, not 1/1.
    assert precision_at_k(["a"], RELEVANT, 5) == 0.2


def test_hit_rate() -> None:
    assert hit_rate_at_k(RANKED, RELEVANT, 1) == 0.0
    assert hit_rate_at_k(RANKED, RELEVANT, 2) == 1.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(RANKED, RELEVANT) == 0.5  # first relevant at rank 2
    assert reciprocal_rank(["a"], RELEVANT) == 1.0
    assert reciprocal_rank(["x", "y"], RELEVANT) == 0.0
    assert reciprocal_rank([], RELEVANT) == 0.0


def test_ndcg() -> None:
    # DCG@5  = 1/log2(3) + 1/log2(5)       (hits at ranks 2 and 4)
    # IDCG@5 = 1/log2(2) + 1/log2(3)       (2 relevant docs at ranks 1 and 2)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3)
    assert ndcg_at_k(RANKED, RELEVANT, 5) == pytest.approx(dcg / idcg)  # ~0.651
    assert ndcg_at_k(RANKED, RELEVANT, 5) == pytest.approx(0.6509209298)
    # k=1: no hit at rank 1.
    assert ndcg_at_k(RANKED, RELEVANT, 1) == 0.0
    # Ideal ordering scores 1.
    assert ndcg_at_k(["a", "c", "x"], RELEVANT, 3) == pytest.approx(1.0)
    # IDCG is capped at k: one relevant at rank 1 with k=1 is perfect.
    assert ndcg_at_k(["c"], RELEVANT, 1) == pytest.approx(1.0)


def test_duplicates_cannot_inflate_scores() -> None:
    assert dedupe(["a", "a", "b", "a"]) == ["a", "b"]
    assert precision_at_k(["a", "a", "a"], RELEVANT, 3) == pytest.approx(1 / 3)
    assert ndcg_at_k(["a", "a"], RELEVANT, 2) == pytest.approx(1 / (1 + 1 / math.log2(3)))


def test_duplicate_relevant_labels_counted_once() -> None:
    assert recall_at_k(["a"], ["a", "a"], 1) == 1.0


@pytest.mark.parametrize("fn", [recall_at_k, precision_at_k, hit_rate_at_k, ndcg_at_k])
def test_undefined_without_relevant_docs(fn: object) -> None:
    assert callable(fn)
    with pytest.raises(ValueError, match="no relevant documents"):
        fn(["a"], [], 1)


def test_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        recall_at_k(["a"], ["a"], 0)


def test_retrieval_scores_keys() -> None:
    scores = retrieval_scores(RANKED, RELEVANT, [1, 3])
    assert set(scores) == {
        "mrr",
        "recall@1", "precision@1", "hit_rate@1", "ndcg@1",
        "recall@3", "precision@3", "hit_rate@3", "ndcg@3",
    }  # fmt: skip
