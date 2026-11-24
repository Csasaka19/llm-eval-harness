"""Latency summaries with explicit sample-size guards.

Tail percentiles from small samples are mostly noise: with 20 requests, "p99" is just
an interpolation near the maximum. Rather than print a misleading number, a percentile is
reported as ``None`` (with a warning) unless ``n >= 1 / (1 - p)``, the minimum needed for
at least one observation to lie beyond it (p95: 20, p99: 100). That is a floor, not a
guarantee of precision; reports add bootstrap confidence intervals on top.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

PERCENTILES: tuple[int, ...] = (50, 95, 99)
MIN_SAMPLES: dict[int, int] = {50: 1, 95: 20, 99: 100}


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile (same as numpy's default ``method="linear"``)."""
    if not values:
        raise ValueError("percentile of empty sequence")
    if not 0 <= p <= 100:
        raise ValueError(f"p must be in [0, 100], got {p}")
    xs = sorted(values)
    pos = (len(xs) - 1) * p / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


@dataclass(frozen=True)
class LatencySummary:
    n: int
    mean_ms: float
    min_ms: float
    max_ms: float
    percentiles_ms: dict[str, float | None]
    warnings: tuple[str, ...] = field(default_factory=tuple)


def summarize_latency(values_ms: Sequence[float]) -> LatencySummary | None:
    if not values_ms:
        return None
    n = len(values_ms)
    pcts: dict[str, float | None] = {}
    warnings: list[str] = []
    for p in PERCENTILES:
        needed = MIN_SAMPLES[p]
        if n >= needed:
            pcts[f"p{p}"] = percentile(values_ms, p)
        else:
            pcts[f"p{p}"] = None
            warnings.append(f"p{p} withheld: needs n >= {needed}, have {n}")
    return LatencySummary(
        n=n,
        mean_ms=sum(values_ms) / n,
        min_ms=min(values_ms),
        max_ms=max(values_ms),
        percentiles_ms=pcts,
        warnings=tuple(warnings),
    )
