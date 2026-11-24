from pathlib import Path

import pytest

from llm_eval.metrics.cost import summarize_cost
from llm_eval.pricing import PricingError, load_pricing
from llm_eval.scoring import score_run
from llm_eval.types import CaseResult, EvalCase, SystemResponse, Usage

EXAMPLES = Path(__file__).parent.parent / "examples"


def _table(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "prices.toml"
    path.write_text(body)
    return path


VALID = """
as_of = 2026-11-01
source = "test"
[models.m]
input_per_mtok = 2.0
output_per_mtok = 10
"""


def test_loads_example_table() -> None:
    table = load_pricing(EXAMPLES / "pricing.example.toml")
    assert table.currency == "USD"
    assert "mock-small" in table.models


def test_cost_arithmetic(tmp_path: Path) -> None:
    table = load_pricing(_table(tmp_path, VALID))
    assert table.as_of == "2026-11-01"  # TOML date normalised to string
    # 1000 * 2.0 / 1M + 500 * 10 / 1M = 0.002 + 0.005
    assert table.cost("m", Usage(1000, 500)) == pytest.approx(0.007)


def test_unknown_cost_is_none_not_zero(tmp_path: Path) -> None:
    table = load_pricing(_table(tmp_path, VALID))
    assert table.cost("other", Usage(1, 1)) is None
    assert table.cost(None, Usage(1, 1)) is None
    assert table.cost("m", None) is None


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("source = 'x'\n[models.m]\ninput_per_mtok = 1\noutput_per_mtok = 1", "'as_of'"),
        ("as_of = 'x'\n[models.m]\ninput_per_mtok = 1\noutput_per_mtok = 1", "'source'"),
        ("as_of = 'x'\nsource = 'y'", "at least one"),
        ("as_of = 'x'\nsource = 'y'\n[models.m]\ninput_per_mtok = 1", "'output_per_mtok'"),
        (
            "as_of = 'x'\nsource = 'y'\n[models.m]\ninput_per_mtok = -1\noutput_per_mtok = 1",
            ">= 0",
        ),
        ("not = = toml", "invalid TOML"),
    ],
)
def test_rejects_bad_tables(tmp_path: Path, body: str, message: str) -> None:
    with pytest.raises(PricingError, match=message):
        load_pricing(_table(tmp_path, body))


def test_summary_counts_unpriced_separately() -> None:
    summary = summarize_cost(
        [0.01, None, 0.03], currency="USD", pricing_as_of="d", pricing_source="s"
    )
    assert (summary.n_priced, summary.n_unpriced) == (2, 1)
    assert summary.total == pytest.approx(0.04)
    assert summary.mean_per_request == pytest.approx(0.02)  # not 0.04 / 3


def test_summary_with_nothing_priced() -> None:
    summary = summarize_cost([None], currency="USD", pricing_as_of="d", pricing_source="s")
    assert summary.mean_per_request is None
    assert summary.total == 0.0


def test_score_run_includes_latency_and_cost(tmp_path: Path) -> None:
    table = load_pricing(_table(tmp_path, VALID))
    cases = [EvalCase("a", "q"), EvalCase("b", "q"), EvalCase("c", "q"), EvalCase("d", "q")]
    results = [
        CaseResult("a", 100.0, SystemResponse("x", usage=Usage(1000, 0), model="m")),
        CaseResult("b", 300.0, SystemResponse("x", usage=None, model="m")),
        CaseResult("c", 200.0, SystemResponse("x", usage=Usage(1, 1), model="unknown-model")),
        CaseResult("d", 5000.0, error="Timeout"),
    ]
    card = score_run(cases, results, pricing=table)

    assert card.latency is not None
    assert card.latency.n == 4  # errored request's latency included
    assert card.latency.max_ms == 5000.0

    assert card.cost is not None
    assert (card.cost.n_priced, card.cost.n_unpriced) == (1, 3)
    assert card.cost.total == pytest.approx(0.002)
    assert card.cost.unpriced_models == ("unknown-model",)
    assert [c.cost for c in card.cases] == [pytest.approx(0.002), None, None, None]


def test_score_run_without_pricing_has_no_cost() -> None:
    card = score_run([EvalCase("a", "q")], [CaseResult("a", 1.0, SystemResponse("x"))])
    assert card.cost is None
    assert card.latency is not None
