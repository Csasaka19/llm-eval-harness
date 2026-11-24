import pytest

from llm_eval.metrics.latency import percentile, summarize_latency


def test_percentile_linear_interpolation() -> None:
    xs = [10.0, 20.0, 30.0, 40.0]
    # position = (n-1) * p/100 = 3 * 0.5 = 1.5 -> halfway between 20 and 30
    assert percentile(xs, 50) == 25.0
    # 3 * 0.95 = 2.85 -> 30 + 0.85 * 10
    assert percentile(xs, 95) == pytest.approx(38.5)
    assert percentile(xs, 0) == 10.0
    assert percentile(xs, 100) == 40.0
    assert percentile([7.0], 99) == 7.0


def test_percentile_is_order_independent() -> None:
    assert percentile([40.0, 10.0, 30.0, 20.0], 50) == 25.0


@pytest.mark.parametrize("p", [-1, 101])
def test_percentile_rejects_out_of_range(p: float) -> None:
    with pytest.raises(ValueError, match="p must be in"):
        percentile([1.0], p)


def test_percentile_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        percentile([], 50)


def test_summary_withholds_tail_percentiles_for_small_samples() -> None:
    summary = summarize_latency([1.0, 2.0, 3.0])
    assert summary is not None
    assert summary.n == 3
    assert summary.mean_ms == 2.0
    assert (summary.min_ms, summary.max_ms) == (1.0, 3.0)
    assert summary.percentiles_ms == {"p50": 2.0, "p95": None, "p99": None}
    assert summary.warnings == (
        "p95 withheld: needs n >= 20, have 3",
        "p99 withheld: needs n >= 100, have 3",
    )


def test_summary_reports_tail_percentiles_with_enough_samples() -> None:
    values = [float(i) for i in range(1, 101)]  # 1..100
    summary = summarize_latency(values)
    assert summary is not None
    assert summary.warnings == ()
    # (n-1)*p: 99*0.95 = 94.05 -> 95 + 0.05; 99*0.99 = 98.01 -> 99 + 0.01
    assert summary.percentiles_ms["p95"] == pytest.approx(95.05)
    assert summary.percentiles_ms["p99"] == pytest.approx(99.01)


def test_summary_of_nothing_is_none() -> None:
    assert summarize_latency([]) is None
