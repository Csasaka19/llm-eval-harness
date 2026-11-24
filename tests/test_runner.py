import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from llm_eval.runner import run_eval, write_run
from llm_eval.system import System, load_system
from llm_eval.systems.mock import EchoSystem, ScriptedSystem
from llm_eval.types import CaseResult, EvalCase, SystemResponse, ToolCall, Usage

CASES = [EvalCase(id="a", input="hello"), EvalCase(id="b", input="world")]


def fake_clock(step: float = 0.25) -> Iterator[float]:
    t = 0.0
    while True:
        yield t
        t += step


def test_runs_every_case_and_measures_latency() -> None:
    ticks = fake_clock()
    run = run_eval(EchoSystem(), CASES, clock=lambda: next(ticks))
    assert run.system_name == "echo"
    assert [r.case_id for r in run.results] == ["a", "b"]
    assert [r.response.answer for r in run.results if r.response] == ["hello", "world"]
    assert all(r.latency_ms == 250.0 for r in run.results)
    assert run.error_count == 0


def test_system_errors_are_recorded_not_raised() -> None:
    system = ScriptedSystem({"a": SystemResponse(answer="ok")})
    seen: list[CaseResult] = []
    run = run_eval(system, CASES, on_result=seen.append)
    assert run.error_count == 1
    assert run.results[0].ok
    assert not run.results[1].ok
    assert run.results[1].response is None
    assert run.results[1].error is not None
    assert "no scripted response for case 'b'" in run.results[1].error
    assert seen == list(run.results)


def test_write_run_round_trips(tmp_path: Path) -> None:
    response = SystemResponse(
        answer="42",
        retrieved_doc_ids=("d1", "d2"),
        retrieved_contexts=("ctx1", "ctx2"),
        tool_calls=(ToolCall("calc", {"expr": "6*7"}),),
        usage=Usage(input_tokens=10, output_tokens=2),
        model="mock-1",
    )
    run = run_eval(ScriptedSystem({"a": response}), CASES[:1])
    out = write_run(run, tmp_path / "run", extra_manifest={"dataset": "x.jsonl"})

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["system"] == "scripted"
    assert manifest["case_count"] == 1
    assert manifest["error_count"] == 0
    assert manifest["dataset"] == "x.jsonl"

    [line] = (out / "results.jsonl").read_text().splitlines()
    row = json.loads(line)
    assert row["case_id"] == "a"
    assert row["error"] is None
    assert row["response"]["retrieved_doc_ids"] == ["d1", "d2"]
    assert row["response"]["tool_calls"] == [{"name": "calc", "arguments": {"expr": "6*7"}}]
    assert row["response"]["usage"] == {"input_tokens": 10, "output_tokens": 2}


def test_load_system_builtin_and_import_path() -> None:
    assert isinstance(load_system("echo"), EchoSystem)
    system = load_system("llm_eval.systems.mock:EchoSystem")
    assert isinstance(system, System)


@pytest.mark.parametrize("spec", ["nope", "llm_eval.systems.mock:", ":EchoSystem"])
def test_load_system_rejects_bad_specs(spec: str) -> None:
    with pytest.raises(ValueError, match="invalid system spec"):
        load_system(spec)


def test_load_system_rejects_non_systems() -> None:
    with pytest.raises(TypeError, match="does not implement"):
        load_system("pathlib:PurePath")
