"""Join raw results with gold labels and compute per-case and aggregate scores.

Honesty rules applied here:

* A case whose system call errored is scored as if it retrieved nothing (all zeros).
  Errors are never silently dropped from the denominator.
* A case without gold labels for a metric family is excluded from that family, and the
  number of included cases is always reported next to the mean.
* Results and dataset must cover exactly the same case ids.
* Latency covers every request, errored or not: a timeout is part of what users see.
* Cost is ``None`` (unknown) unless the model and token usage are known and priced.
* Tool-call scores for an errored case are all 0, even when no calls were expected:
  a crash is not a correct "no tool needed" decision.
* Faithfulness (only when a judge is given) covers RAG cases: those with
  ``expected_doc_ids`` or whose response includes retrieved contexts. An errored RAG case
  scores 0. A RAG case that retrieved nothing is judged against empty context, so any
  claim it makes is unsupported.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from llm_eval.judges import ClaimVerdict, Judge
from llm_eval.metrics.cost import CostSummary, summarize_cost
from llm_eval.metrics.faithfulness import judge_answer, zero_faithfulness_scores
from llm_eval.metrics.latency import LatencySummary, summarize_latency
from llm_eval.metrics.retrieval import retrieval_scores
from llm_eval.metrics.tool_calls import tool_call_scores, zero_tool_call_scores
from llm_eval.pricing import PriceTable
from llm_eval.types import CaseResult, EvalCase

DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)


class ScoringError(ValueError):
    """Raised when results cannot be matched to the dataset."""


@dataclass(frozen=True)
class CaseScores:
    case_id: str
    errored: bool
    latency_ms: float
    cost: float | None = None
    retrieval: dict[str, float] | None = None
    tool_calls: dict[str, float] | None = None
    faithfulness: dict[str, float] | None = None
    claims: tuple[ClaimVerdict, ...] | None = None


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
    tool_calls: dict[str, Aggregate] = field(default_factory=dict)
    faithfulness: dict[str, Aggregate] = field(default_factory=dict)
    judge: str | None = None
    latency: LatencySummary | None = None
    cost: CostSummary | None = None


def score_run(
    cases: Sequence[EvalCase],
    results: Sequence[CaseResult],
    ks: Sequence[int] = DEFAULT_KS,
    *,
    pricing: PriceTable | None = None,
    judge: Judge | None = None,
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
    unpriced_models: list[str] = []
    for case in cases:
        result = by_id[case.id]
        retrieval: dict[str, float] | None = None
        if case.expected_doc_ids:
            retrieved = result.response.retrieved_doc_ids if result.response else ()
            retrieval = retrieval_scores(retrieved, case.expected_doc_ids, ks)
        tools: dict[str, float] | None = None
        if case.expected_tool_calls is not None:
            if result.response is None:
                tools = zero_tool_call_scores()
            else:
                tools = tool_call_scores(case.expected_tool_calls, result.response.tool_calls)
        faith: dict[str, float] | None = None
        claims: tuple[ClaimVerdict, ...] | None = None
        if judge is not None:
            contexts = result.response.retrieved_contexts if result.response else ()
            if result.response is None:
                if case.expected_doc_ids:
                    faith = zero_faithfulness_scores()
            elif case.expected_doc_ids or contexts:
                faith, claims = judge_answer(result.response.answer, contexts, judge)
        cost: float | None = None
        if pricing is not None and result.response is not None:
            cost = pricing.cost(result.response.model, result.response.usage)
            model = result.response.model
            if cost is None and model is not None and model not in pricing.models:
                unpriced_models.append(model)
        per_case.append(
            CaseScores(
                case_id=case.id,
                errored=not result.ok,
                latency_ms=result.latency_ms,
                cost=cost,
                retrieval=retrieval,
                tool_calls=tools,
                faithfulness=faith,
                claims=claims,
            )
        )

    cost_summary = None
    if pricing is not None:
        cost_summary = summarize_cost(
            [c.cost for c in per_case],
            currency=pricing.currency,
            pricing_as_of=pricing.as_of,
            pricing_source=pricing.source,
            unpriced_models=unpriced_models,
        )
    return Scorecard(
        cases=tuple(per_case),
        retrieval=_aggregate(per_case, lambda c: c.retrieval),
        tool_calls=_aggregate(per_case, lambda c: c.tool_calls),
        faithfulness=_aggregate(per_case, lambda c: c.faithfulness),
        judge=judge.name if judge else None,
        latency=summarize_latency([c.latency_ms for c in per_case]),
        cost=cost_summary,
    )


def _aggregate(
    per_case: Sequence[CaseScores], get: Callable[[CaseScores], dict[str, float] | None]
) -> dict[str, Aggregate]:
    """Mean of each metric over the cases that have scores for this metric family."""
    scored = [(c, scores) for c in per_case if (scores := get(c)) is not None]
    if not scored:
        return {}
    n_errored = sum(1 for c, _ in scored if c.errored)
    keys = scored[0][1].keys()
    return {
        key: Aggregate(
            mean=sum(scores[key] for _, scores in scored) / len(scored),
            n=len(scored),
            n_errored=n_errored,
        )
        for key in keys
    }
