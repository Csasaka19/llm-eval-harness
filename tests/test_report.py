import dataclasses
from pathlib import Path

import pytest

from llm_eval.calibration import LabelledClaim, calibrate
from llm_eval.cli import main
from llm_eval.judges.overlap import TokenOverlapJudge
from llm_eval.report import ReportInputs, render_report
from llm_eval.scoring import score_run
from llm_eval.stats import bootstrap_ci, mean
from llm_eval.types import CaseResult, EvalCase, JSONObject, SystemResponse, ToolCall

ROOT = Path(__file__).parent.parent


# --- bootstrap -------------------------------------------------------------------


def test_bootstrap_is_reproducible_and_brackets_the_mean() -> None:
    values = [0.0, 1.0] * 20
    a, b = bootstrap_ci(values), bootstrap_ci(values)
    assert a == b
    assert a.estimate == 0.5
    assert a.low < 0.5 < a.high
    assert a.n == 40


def test_bootstrap_constant_values_has_zero_width() -> None:
    ci = bootstrap_ci([1.0] * 10)
    assert (ci.low, ci.high) == (1.0, 1.0)


def test_bootstrap_width_shrinks_with_n() -> None:
    small = bootstrap_ci([0.0, 1.0] * 5)
    large = bootstrap_ci([0.0, 1.0] * 500)
    assert (large.high - large.low) < (small.high - small.low) / 3


def test_bootstrap_custom_statistic() -> None:
    ci = bootstrap_ci([1.0, 2.0, 3.0], statistic=max)
    assert ci.estimate == 3.0
    assert ci.high == 3.0


def test_bootstrap_validation() -> None:
    with pytest.raises(ValueError, match="empty"):
        bootstrap_ci([])
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci([1.0], confidence=1.0)
    assert mean([1.0, 3.0]) == 2.0


# --- report rendering ------------------------------------------------------------

CASES = [
    EvalCase("r1", "q1", expected_doc_ids=("d1",), expected_answer="42", metadata={"type": "easy"}),
    EvalCase(
        "r2", "q2 | pipe", expected_doc_ids=("d2",), expected_answer="7", metadata={"type": "hard"}
    ),
    EvalCase(
        "t1", "call it", expected_tool_calls=(ToolCall("f", {"x": 1}),), metadata={"type": "easy"}
    ),
    EvalCase("t2", "crash", expected_tool_calls=(), metadata={"type": "hard"}),
]
RESULTS = [
    CaseResult("r1", 10.0, SystemResponse("It is 42.", ("d1",), ("It is 42.",))),
    CaseResult("r2", 20.0, SystemResponse("No idea.", ("d9",), ("unrelated",))),
    CaseResult("t1", 30.0, SystemResponse("ok", tool_calls=(ToolCall("f", {"x": 2}),))),
    CaseResult("t2", 40.0, error="TimeoutError: slow"),
]
MANIFEST: JSONObject = {
    "system": "demo",
    "system_spec": "demo:Demo",
    "dataset": "d.jsonl",
    "dataset_sha256": "ab" * 32,
    "started_at": "2026-11-25T00:00:00+00:00",
    "harness_version": "0.1.0",
}


def _render(**overrides: object) -> str:
    card = score_run(CASES, RESULTS, [1], judge=TokenOverlapJudge())
    inputs = ReportInputs(card=card, cases=CASES, results=RESULTS, manifest=MANIFEST, ks=[1])
    return render_report(dataclasses.replace(inputs, **overrides))  # type: ignore[arg-type]


def test_report_sections_present() -> None:
    text = _render()
    for heading in [
        "# Evaluation report: demo",
        "## Summary",
        "## Results by case type",
        "## Latency and cost",
        "## Judge calibration",
        "## Worst cases",
        "## Methodology",
        "## Limitations",
    ]:
        assert heading in text


def test_report_summary_rows_have_ci_and_errors() -> None:
    text = _render()
    assert "| Retrieval recall@1 | 0.500 | [" in text
    assert "| Tool-call F1 (exact) | 0.000 | [0.000, 0.000] | 2 | 1 |" in text
    assert "| Answer token recall | 0.500 |" in text


def test_report_breakdown_by_type() -> None:
    text = _render()
    assert "| easy | 2 | 1.000 | 1.000 | 0.000 |" in text


def test_report_flags_uncalibrated_judge_and_small_n() -> None:
    text = _render()
    assert "**Not provided.**" in text
    assert "The faithfulness judge was not calibrated for this report." in text
    assert "**Small dataset (n=4).**" in text


def test_report_includes_calibration_with_source() -> None:
    labelled = [
        LabelledClaim("a", "cats purr", ("cats purr loudly",), supported=True),
        LabelledClaim("b", "dogs fly", ("cats purr loudly",), supported=False),
    ]
    report = calibrate(TokenOverlapJudge(), labelled)
    text = _render(calibration=report, calibration_source="labels.jsonl")
    assert "against labelled claims from `labels.jsonl`" in text
    assert "| Cohen's kappa | 1.000 |" in text
    assert "*almost perfect*" in text
    assert "**Not provided.**" not in text


def test_report_worst_cases_show_evidence_and_escape_pipes() -> None:
    text = _render()
    assert "**r2** (hard): 0.000" in text
    assert "- Retrieved: `d9`" in text
    assert "q2 \\| pipe" in text
    assert "- **Error:** TimeoutError: slow" in text
    assert '- Made calls: `f({"x": 2})`' in text
    # Faithfulness worst case lists each claim's verdict and the judge's rationale.
    assert "  - **unsupported**: No idea. (0/2 content tokens in context (0.00))" in text


def test_report_withholds_tail_latency() -> None:
    assert "| p95 | withheld | needs n >= 20, have 4 |" in _render()


# --- CLI -------------------------------------------------------------------------


def test_report_command_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    run = tmp_path / "run"
    dataset = str(ROOT / "datasets/rag/questions.jsonl")
    assert main(["run", "--dataset", dataset, "--system", "bm25", "--out", str(run)]) == 0
    labels = str(ROOT / "examples/faithfulness_calibration.jsonl")
    out = tmp_path / "reports" / "rag.md"
    argv = ["report", str(run), "--judge", "overlap", "--calibration-labels", labels]
    assert main([*argv, "--out", str(out), "--k", "1", "5"]) == 0
    text = out.read_text()
    assert "| Retrieval recall@1 | 0.863 |" in text
    assert "| paraphrase | 8 |" in text
    assert "kappa 0.17 is *slight* agreement" in text
    assert (run / "scores.json").exists()


def test_report_calibration_requires_judge(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["report", str(tmp_path), "--calibration-labels", "x.jsonl"]) == 2
    assert "requires --judge" in capsys.readouterr().err
