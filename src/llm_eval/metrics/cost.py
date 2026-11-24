"""Cost-per-request aggregation.

Requests whose cost is unknown (no usage reported, or model missing from the price table)
are counted separately and never treated as free. Errored requests that still report usage
are included: failed calls cost money too.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CostSummary:
    currency: str
    pricing_as_of: str
    pricing_source: str
    n_priced: int
    n_unpriced: int
    total: float
    mean_per_request: float | None
    unpriced_models: tuple[str, ...]


def summarize_cost(
    costs: Sequence[float | None],
    *,
    currency: str,
    pricing_as_of: str,
    pricing_source: str,
    unpriced_models: Sequence[str] = (),
) -> CostSummary:
    priced = [c for c in costs if c is not None]
    return CostSummary(
        currency=currency,
        pricing_as_of=pricing_as_of,
        pricing_source=pricing_source,
        n_priced=len(priced),
        n_unpriced=len(costs) - len(priced),
        total=sum(priced),
        mean_per_request=sum(priced) / len(priced) if priced else None,
        unpriced_models=tuple(sorted(set(unpriced_models))),
    )
