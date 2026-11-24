"""Read raw run artifacts back into typed objects."""

from __future__ import annotations

import json
from pathlib import Path

from llm_eval.types import CaseResult, JSONObject, JSONValue, SystemResponse, ToolCall, Usage


class RunFormatError(ValueError):
    """Raised when a run directory is missing files or contains malformed records."""


def read_manifest(run_dir: str | Path) -> JSONObject:
    path = Path(run_dir) / "manifest.json"
    try:
        obj: JSONValue = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise RunFormatError(f"{path}: not found") from None
    except json.JSONDecodeError as exc:
        raise RunFormatError(f"{path}: invalid JSON: {exc.msg}") from exc
    if not isinstance(obj, dict):
        raise RunFormatError(f"{path}: must be a JSON object")
    return obj


def read_results(run_dir: str | Path) -> list[CaseResult]:
    path = Path(run_dir) / "results.jsonl"
    if not path.exists():
        raise RunFormatError(f"{path}: not found")
    results: list[CaseResult] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            where = f"{path}:{lineno}"
            try:
                obj: JSONValue = json.loads(line)
                results.append(_case_result(obj))
            except (json.JSONDecodeError, TypeError, KeyError) as exc:
                raise RunFormatError(f"{where}: {exc}") from exc
    return results


def _case_result(obj: JSONValue) -> CaseResult:
    o = _obj(obj)
    response = o.get("response")
    error = o.get("error")
    return CaseResult(
        case_id=_str(o["case_id"]),
        latency_ms=_num(o["latency_ms"]),
        response=None if response is None else _response(response),
        error=None if error is None else _str(error),
    )


def _response(obj: JSONValue) -> SystemResponse:
    o = _obj(obj)
    usage = o.get("usage")
    model = o.get("model")
    return SystemResponse(
        answer=_str(o["answer"]),
        retrieved_doc_ids=tuple(_str(x) for x in _list(o.get("retrieved_doc_ids", []))),
        retrieved_contexts=tuple(_str(x) for x in _list(o.get("retrieved_contexts", []))),
        tool_calls=tuple(_tool_call(x) for x in _list(o.get("tool_calls", []))),
        usage=None if usage is None else _usage(usage),
        model=None if model is None else _str(model),
    )


def _tool_call(obj: JSONValue) -> ToolCall:
    o = _obj(obj)
    return ToolCall(name=_str(o["name"]), arguments=_obj(o.get("arguments", {})))


def _usage(obj: JSONValue) -> Usage:
    o = _obj(obj)
    return Usage(
        input_tokens=_int(o.get("input_tokens", 0)), output_tokens=_int(o.get("output_tokens", 0))
    )


def _obj(v: JSONValue) -> JSONObject:
    if not isinstance(v, dict):
        raise TypeError(f"expected object, got {type(v).__name__}")
    return v


def _list(v: JSONValue) -> list[JSONValue]:
    if not isinstance(v, list):
        raise TypeError(f"expected list, got {type(v).__name__}")
    return v


def _str(v: JSONValue) -> str:
    if not isinstance(v, str):
        raise TypeError(f"expected string, got {type(v).__name__}")
    return v


def _int(v: JSONValue) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise TypeError(f"expected integer, got {type(v).__name__}")
    return v


def _num(v: JSONValue) -> float:
    if isinstance(v, bool) or not isinstance(v, int | float):
        raise TypeError(f"expected number, got {type(v).__name__}")
    return float(v)
