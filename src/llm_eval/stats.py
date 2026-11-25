"""Bootstrap confidence intervals.

Percentile bootstrap: resample the per-case values with replacement, recompute the
statistic, and take the (alpha/2, 1 - alpha/2) quantiles. A fixed seed makes reports
reproducible. Caveats: it treats cases as independent draws from the population of
interest, and it under-covers for small n and for statistics near 0 or 1, so intervals
on small datasets should be read as optimistic.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from llm_eval.metrics.latency import percentile

DEFAULT_RESAMPLES = 2000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 0


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    n: int


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = mean,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> Interval:
    if not values:
        raise ValueError("bootstrap of empty sequence")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    rng = random.Random(seed)
    n = len(values)
    stats = [statistic(rng.choices(values, k=n)) for _ in range(n_resamples)]
    alpha = (1 - confidence) / 2 * 100
    return Interval(
        estimate=statistic(values),
        low=percentile(stats, alpha),
        high=percentile(stats, 100 - alpha),
        n=n,
    )
