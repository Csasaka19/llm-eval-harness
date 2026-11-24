"""Run a system over a dataset and persist raw, per-case results.

Raw results are the source of truth: every metric and report is computed from
``results.jsonl`` after the fact, so scoring can change without re-running systems.
"""

from __future__ import annotations

import dataclasses
import json
import platform
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from llm_eval import __version__
from llm_eval.system import System
from llm_eval.types import CaseResult, EvalCase, JSONObject

Clock = Callable[[], float]


@dataclass(frozen=True)
class RunResult:
    system_name: str
    started_at: str
    finished_at: str
    results: tuple[CaseResult, ...]

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results if not r.ok)


def run_eval(
    system: System,
    cases: Sequence[EvalCase],
    *,
    clock: Clock = time.perf_counter,
    on_result: Callable[[CaseResult], None] | None = None,
) -> RunResult:
    """Run every case sequentially. Exceptions are recorded per case, never raised."""
    started_at = _now_iso()
    results: list[CaseResult] = []
    for case in cases:
        t0 = clock()
        try:
            response = system.run(case)
        except Exception as exc:
            result = CaseResult(
                case_id=case.id,
                latency_ms=(clock() - t0) * 1000.0,
                error=f"{type(exc).__name__}: {exc}",
            )
        else:
            result = CaseResult(
                case_id=case.id, latency_ms=(clock() - t0) * 1000.0, response=response
            )
        results.append(result)
        if on_result is not None:
            on_result(result)
    return RunResult(
        system_name=system.name,
        started_at=started_at,
        finished_at=_now_iso(),
        results=tuple(results),
    )


def write_run(
    run: RunResult, out_dir: str | Path, *, extra_manifest: JSONObject | None = None
) -> Path:
    """Write ``manifest.json`` and ``results.jsonl`` into ``out_dir`` (created if needed)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.jsonl").open("w", encoding="utf-8") as fh:
        for result in run.results:
            fh.write(json.dumps(dataclasses.asdict(result), ensure_ascii=False) + "\n")
    manifest: JSONObject = {
        "harness_version": __version__,
        "python_version": platform.python_version(),
        "system": run.system_name,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "case_count": len(run.results),
        "error_count": run.error_count,
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out


def default_run_dir(root: str | Path = "runs") -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path(root) / stamp


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
