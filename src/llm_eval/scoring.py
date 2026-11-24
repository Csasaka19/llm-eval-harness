"""Join raw results with gold labels and compute per-case and aggregate scores.

Honesty rules applied here:

* A case whose system call errored is scored as if it retrieved nothing (all zeros).
  Errors are never silently dropped from the denominator.
* A case without gold labels for a metric family is excluded from that family, and the
  number of included cases is always reported next to the mean.
* Results and dataset must cover exactly the same case ids.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from llm_eval.metrics.retrieval import retrieval_scores
from llm_eval.types import CaseResult, EvalCase

DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)


class ScoringError(ValueError):
    """Raised when results cannot be matched to the dataset."""


@dataclass(frozen=True)
class CaseScores:
    case_id: str
    errored: bool
    retrieval: dict[str, float] | None = None


@dataclass(frozen=True)
class Aggregate:
    """Mean of a metric over ``n`` cases (``n_errored`` of which scored 0 due to errors)."""

    mean: float
    n: int
    n_errored: int


@dataclass(frozen=True)
class Scorecard:
    cases: tuple[CaseScores, ...]
    retrieval: dict[str, Aggregate] = field(default_factory=dict)


def score_run(
    cases: Sequence[EvalCase], results: Sequence[CaseResult], ks: Sequence[int] = DEFAULT_KS
) -> Scorecard:
    by_id = {r.case_id: r for r in results}
    if len(by_id) != len(results):
        raise ScoringError("results contain duplicate case ids")
    case_ids = {c.id for c in cases}
    missing = sorted(case_ids - by_id.keys())
    extra = sorted(by_id.keys() - case_ids)
    if missing or extra:
        raise ScoringError(
            f"results do not match dataset: missing={missing[:5]} extra={extra[:5]}"
            + (" (truncated)" if len(missing) > 5 or len(extra) > 5 else "")
        )

    per_case: list[CaseScores] = []
    for case in cases:
        result = by_id[case.id]
        retrieval: dict[str, float] | None = None
        if case.expected_doc_ids:
            retrieved = result.response.retrieved_doc_ids if result.response else ()
            retrieval = retrieval_scores(retrieved, case.expected_doc_ids, ks)
        per_case.append(CaseScores(case_id=case.id, errored=not result.ok, retrieval=retrieval))

    return Scorecard(cases=tuple(per_case), retrieval=_aggregate(per_case))


def _aggregate(per_case: Sequence[CaseScores]) -> dict[str, Aggregate]:
    scored = [c for c in per_case if c.retrieval is not None]
    if not scored:
        return {}
    n_errored = sum(1 for c in scored if c.errored)
    keys = scored[0].retrieval.keys() if scored[0].retrieval else []
    return {
        key: Aggregate(
            mean=sum(c.retrieval[key] for c in scored if c.retrieval) / len(scored),
            n=len(scored),
            n_errored=n_errored,
        )
        for key in keys
    }
