import json
from pathlib import Path

import pytest

from llm_eval.cli import main
from llm_eval.runner import run_eval, write_run
from llm_eval.scoring import ScoringError, score_run
from llm_eval.serialization import RunFormatError, read_results
from llm_eval.systems.mock import ScriptedSystem
from llm_eval.types import CaseResult, EvalCase, SystemResponse, ToolCall, Usage

CASES = [
    EvalCase(id="perfect", input="q", expected_doc_ids=("a",)),
    EvalCase(id="miss", input="q", expected_doc_ids=("a", "b")),
    EvalCase(id="errored", input="q", expected_doc_ids=("a",)),
    EvalCase(id="no-gold", input="q"),
]
RESULTS = [
    CaseResult("perfect", 1.0, SystemResponse("x", retrieved_doc_ids=("a", "z"))),
    CaseResult("miss", 1.0, SystemResponse("x", retrieved_doc_ids=("z", "a"))),
    CaseResult("errored", 1.0, error="Timeout"),
    CaseResult("no-gold", 1.0, SystemResponse("x", retrieved_doc_ids=("a",))),
]


def test_scores_only_labelled_cases_and_zeroes_errors() -> None:
    card = score_run(CASES, RESULTS, ks=[1, 2])
    by_id = {c.case_id: c for c in card.cases}
    assert by_id["no-gold"].retrieval is None
    assert by_id["errored"].errored
    assert by_id["errored"].retrieval is not None
    assert all(v == 0.0 for v in by_id["errored"].retrieval.values())

    recall1 = card.retrieval["recall@1"]
    assert (recall1.n, recall1.n_errored) == (3, 1)
    # perfect=1, miss=0 (z at rank 1), errored=0
    assert recall1.mean == pytest.approx(1 / 3)
    # recall@2: perfect=1, miss=1/2, errored=0
    assert card.retrieval["recall@2"].mean == pytest.approx(0.5)
    # mrr: 1, 1/2, 0
    assert card.retrieval["mrr"].mean == pytest.approx(0.5)


def test_no_labelled_cases_gives_empty_aggregate() -> None:
    card = score_run(CASES[3:], RESULTS[3:])
    assert card.retrieval == {}


def test_mismatched_ids_rejected() -> None:
    with pytest.raises(ScoringError, match=r"missing=\['no-gold'\] extra=\[\]"):
        score_run(CASES, RESULTS[:3])
    with pytest.raises(ScoringError, match="duplicate"):
        score_run(CASES, [*RESULTS, RESULTS[0]])


def test_results_round_trip(tmp_path: Path) -> None:
    resp = SystemResponse(
        "ans",
        retrieved_doc_ids=("a",),
        retrieved_contexts=("ctx",),
        tool_calls=(ToolCall("t", {"k": [1, {"n": None}]}),),
        usage=Usage(3, 4),
        model="m",
    )
    run = run_eval(ScriptedSystem({"a": resp}), [EvalCase("a", "q"), EvalCase("b", "q")])
    write_run(run, tmp_path)
    assert read_results(tmp_path) == list(run.results)


def test_read_results_reports_bad_lines(tmp_path: Path) -> None:
    (tmp_path / "results.jsonl").write_text('{"case_id": "a", "latency_ms": "slow"}\n')
    with pytest.raises(RunFormatError, match=r"results\.jsonl:1: expected number"):
        read_results(tmp_path)


def _write_dataset(path: Path) -> None:
    path.write_text(
        '{"id": "a", "input": "q", "expected_doc_ids": ["d1"]}\n'
        '{"id": "b", "input": "q", "expected_doc_ids": ["d2"]}\n'
    )


def _run(dataset: Path, out: Path) -> None:
    system = "tests.test_scoring:FixtureSystem"
    assert main(["run", "--dataset", str(dataset), "--system", system, "--out", str(out)]) == 0


class FixtureSystem(ScriptedSystem):
    def __init__(self) -> None:
        super().__init__(
            {
                "a": SystemResponse(
                    "x", retrieved_doc_ids=("d1",), usage=Usage(1000, 100), model="mock-small"
                ),
                "b": SystemResponse("x", retrieved_doc_ids=("d9", "d2")),
            },
            name="fixture",
        )


def test_score_command_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset, out = tmp_path / "d.jsonl", tmp_path / "run"
    _write_dataset(dataset)
    _run(dataset, out)
    capsys.readouterr()

    assert main(["score", str(out), "--k", "1", "2"]) == 0
    printed = capsys.readouterr().out
    assert "retrieval (n=2, errored=0" in printed
    assert "recall@1" in printed

    scores = json.loads((out / "scores.json").read_text())
    assert scores["retrieval"]["recall@1"] == {"mean": 0.5, "n": 2, "n_errored": 0}
    assert scores["retrieval"]["recall@2"]["mean"] == 1.0
    assert scores["retrieval"]["mrr"]["mean"] == 0.75
    assert scores["latency"]["n"] == 2
    assert scores["cost"] is None
    assert "cost: skipped" in printed
    assert "tool calls: no cases with expected_tool_calls" in printed


def test_score_command_with_pricing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset, out = tmp_path / "d.jsonl", tmp_path / "run"
    _write_dataset(dataset)
    _run(dataset, out)
    pricing = Path(__file__).parent.parent / "examples" / "pricing.example.toml"
    capsys.readouterr()

    assert main(["score", str(out), "--pricing", str(pricing)]) == 0
    printed = capsys.readouterr().out
    assert "prices as of 2026-11-24" in printed
    scores = json.loads((out / "scores.json").read_text())
    # FixtureSystem: "a" uses mock-small (1000 in, 100 out), "b" reports no usage.
    # 1000 * 0.50 / 1M + 100 * 2.00 / 1M = 0.0005 + 0.0002
    assert scores["cost"]["n_priced"] == 1
    assert scores["cost"]["n_unpriced"] == 1
    assert scores["cost"]["total"] == pytest.approx(0.0007)


def test_score_command_rejects_changed_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset, out = tmp_path / "d.jsonl", tmp_path / "run"
    _write_dataset(dataset)
    _run(dataset, out)
    dataset.write_text(dataset.read_text().replace("d2", "d3"))
    capsys.readouterr()

    assert main(["score", str(out)]) == 2
    assert "does not match the dataset this run used" in capsys.readouterr().err


def test_score_command_missing_run_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["score", str(tmp_path / "nope")]) == 2
    assert "manifest.json: not found" in capsys.readouterr().err
