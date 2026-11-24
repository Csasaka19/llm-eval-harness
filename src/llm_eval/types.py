"""Core data types shared by the dataset loader, systems, runner, and metrics.

All types are immutable and JSON-serialisable via :func:`dataclasses.asdict`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

JSONValue: TypeAlias = "str | int | float | bool | list[JSONValue] | dict[str, JSONValue] | None"
JSONObject: TypeAlias = dict[str, JSONValue]


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation, either expected (gold) or produced by a system."""

    name: str
    arguments: JSONObject = field(default_factory=dict)


@dataclass(frozen=True)
class EvalCase:
    """One evaluation example.

    Every ``expected_*`` field is optional: a case only contributes to the metrics
    whose gold labels it provides.
    """

    id: str
    input: str
    expected_doc_ids: tuple[str, ...] = ()
    expected_answer: str | None = None
    expected_tool_calls: tuple[ToolCall, ...] = ()
    metadata: JSONObject = field(default_factory=dict)


@dataclass(frozen=True)
class Usage:
    """Token usage reported by the system under test (used for cost)."""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class SystemResponse:
    """What a system under test returns for a single case."""

    answer: str
    retrieved_doc_ids: tuple[str, ...] = ()
    retrieved_contexts: tuple[str, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage | None = None
    model: str | None = None


@dataclass(frozen=True)
class CaseResult:
    """Outcome of running one case: a response or an error, plus wall-clock latency."""

    case_id: str
    latency_ms: float
    response: SystemResponse | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None
