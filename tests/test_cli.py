import json
from pathlib import Path

import pytest

from llm_eval import __version__
from llm_eval.cli import main


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_args_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "llm-eval" in capsys.readouterr().out


EXAMPLES = Path(__file__).parent.parent / "examples"


def test_run_command_writes_results(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "run"
    code = main(
        ["run", "--dataset", str(EXAMPLES / "smoke.jsonl"), "--system", "echo", "--out", str(out)]
    )
    assert code == 0
    assert "3 cases, 0 errors" in capsys.readouterr().out
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["system_spec"] == "echo"
    assert len(manifest["dataset_sha256"]) == 64
    assert len((out / "results.jsonl").read_text().splitlines()) == 3


def test_run_command_reports_bad_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n")
    assert main(["run", "--dataset", str(bad), "--system", "echo"]) == 2
    assert "invalid JSON" in capsys.readouterr().err
