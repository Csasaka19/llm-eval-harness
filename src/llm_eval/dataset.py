"""Load evaluation cases from JSONL, with line-numbered validation errors."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from llm_eval.types import EvalCase, JSONObject, JSONValue, ToolCall

_KNOWN_KEYS = frozenset(
    {"id", "input", "expected_doc_ids", "expected_answer", "expected_tool_calls", "metadata"}
)


class DatasetError(ValueError):
    """Raised when a dataset file is malformed. The message includes the line number."""


def load_jsonl(path: str | Path) -> list[EvalCase]:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        return parse_lines(fh, source=str(path))


def parse_lines(lines: Iterable[str], source: str = "<input>") -> list[EvalCase]:
    cases: list[EvalCase] = []
    seen: dict[str, int] = {}
    for lineno, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        where = f"{source}:{lineno}"
        try:
            obj: JSONValue = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{where}: invalid JSON: {exc.msg}") from exc
        case = _parse_case(obj, where)
        if case.id in seen:
            raise DatasetError(
                f"{where}: duplicate id {case.id!r} (first seen on line {seen[case.id]})"
            )
        seen[case.id] = lineno
        cases.append(case)
    return cases


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _parse_case(obj: JSONValue, where: str) -> EvalCase:
    if not isinstance(obj, dict):
        raise DatasetError(f"{where}: each line must be a JSON object")
    unknown = set(obj) - _KNOWN_KEYS
    if unknown:
        raise DatasetError(f"{where}: unknown keys {sorted(unknown)}")

    case_id = _require_str(obj, "id", where)
    text = _require_str(obj, "input", where)

    doc_ids_raw = obj.get("expected_doc_ids", [])
    if not isinstance(doc_ids_raw, list) or not all(isinstance(d, str) for d in doc_ids_raw):
        raise DatasetError(f"{where}: 'expected_doc_ids' must be a list of strings")
    doc_ids = tuple(d for d in doc_ids_raw if isinstance(d, str))

    answer = obj.get("expected_answer")
    if answer is not None and not isinstance(answer, str):
        raise DatasetError(f"{where}: 'expected_answer' must be a string or null")

    calls_raw = obj.get("expected_tool_calls", [])
    if not isinstance(calls_raw, list):
        raise DatasetError(f"{where}: 'expected_tool_calls' must be a list")
    calls = tuple(
        _parse_tool_call(c, f"{where}: expected_tool_calls[{i}]") for i, c in enumerate(calls_raw)
    )

    metadata = obj.get("metadata", {})
    if not isinstance(metadata, dict):
        raise DatasetError(f"{where}: 'metadata' must be an object")

    return EvalCase(
        id=case_id,
        input=text,
        expected_doc_ids=doc_ids,
        expected_answer=answer,
        expected_tool_calls=calls,
        metadata=metadata,
    )


def _parse_tool_call(obj: JSONValue, where: str) -> ToolCall:
    if not isinstance(obj, dict):
        raise DatasetError(f"{where}: must be an object")
    name = _require_str(obj, "name", where)
    args = obj.get("arguments", {})
    if not isinstance(args, dict):
        raise DatasetError(f"{where}: 'arguments' must be an object")
    return ToolCall(name=name, arguments=args)


def _require_str(obj: JSONObject, key: str, where: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value:
        raise DatasetError(f"{where}: {key!r} is required and must be a non-empty string")
    return value
