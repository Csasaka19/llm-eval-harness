"""Tool-call correctness at three argument-matching strictness levels.

Each level is reported separately so readers can see how much a score depends on
leniency:

* ``exact``  - same tool name and arguments deep-equal (no extra or missing keys).
* ``subset`` - same tool name and every expected argument present with an equal value;
  extra arguments from the system are allowed.
* ``name``   - same tool name; arguments ignored.

Per level, for one case:

* ``precision`` - matched calls / calls made (1.0 when no calls were made).
* ``recall``    - matched calls / calls expected (1.0 when none were expected).
* ``f1``        - harmonic mean of the two (0.0 when both are 0).
* ``ordered_recall`` - longest common subsequence of expected vs made calls, divided by
  calls expected: credit only for calls made in the right relative order.
* ``sequence_match`` - 1.0 iff the made calls equal the expected calls one-to-one, in order.

Matching between expected and made calls is a maximum bipartite matching, not a greedy
scan, so a lenient match cannot "steal" a call that a later expectation needed.

Values compare with JSON semantics: ``1 == 1.0`` but ``True != 1`` (unlike Python).
No normalisation (case, whitespace, units) is applied; that is deliberate, and
dataset-specific normalisation belongs in the system adapter or a custom metric.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

from llm_eval.types import JSONValue, ToolCall

MatchMode = Literal["exact", "subset", "name"]
MATCH_MODES: tuple[MatchMode, ...] = ("exact", "subset", "name")
TOOL_CALL_METRICS: tuple[str, ...] = (
    "precision",
    "recall",
    "f1",
    "ordered_recall",
    "sequence_match",
)


def json_equal(a: JSONValue, b: JSONValue) -> bool:
    """Deep equality with JSON semantics: bools are not numbers, ints equal floats."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, int | float) and isinstance(b, int | float):
        return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(json_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(json_equal(x, y) for x, y in zip(a, b, strict=True))
    return type(a) is type(b) and a == b


def calls_match(expected: ToolCall, actual: ToolCall, mode: MatchMode) -> bool:
    if expected.name != actual.name:
        return False
    if mode == "name":
        return True
    if mode == "exact":
        return json_equal(expected.arguments, actual.arguments)
    return all(
        key in actual.arguments and json_equal(value, actual.arguments[key])
        for key, value in expected.arguments.items()
    )


def max_matching(n_left: int, n_right: int, edge: Callable[[int, int], bool]) -> int:
    """Size of a maximum bipartite matching (Kuhn's algorithm; inputs here are tiny)."""
    match_right: list[int | None] = [None] * n_right

    def augment(i: int, seen: list[bool]) -> bool:
        for j in range(n_right):
            if seen[j] or not edge(i, j):
                continue
            seen[j] = True
            owner = match_right[j]
            if owner is None or augment(owner, seen):
                match_right[j] = i
                return True
        return False

    return sum(1 for i in range(n_left) if augment(i, [False] * n_right))


def lcs_length(n_a: int, n_b: int, eq: Callable[[int, int], bool]) -> int:
    prev = [0] * (n_b + 1)
    for i in range(n_a):
        cur = [0] * (n_b + 1)
        for j in range(n_b):
            cur[j + 1] = prev[j] + 1 if eq(i, j) else max(prev[j + 1], cur[j])
        prev = cur
    return prev[n_b]


def tool_call_scores_for_mode(
    expected: Sequence[ToolCall], actual: Sequence[ToolCall], mode: MatchMode
) -> dict[str, float]:
    def edge(i: int, j: int) -> bool:
        return calls_match(expected[i], actual[j], mode)

    matched = max_matching(len(expected), len(actual), edge)
    precision = matched / len(actual) if actual else 1.0
    recall = matched / len(expected) if expected else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ordered = lcs_length(len(expected), len(actual), edge)
    ordered_recall = ordered / len(expected) if expected else 1.0
    sequence_match = float(len(expected) == len(actual) == ordered)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ordered_recall": ordered_recall,
        "sequence_match": sequence_match,
    }


def tool_call_scores(expected: Sequence[ToolCall], actual: Sequence[ToolCall]) -> dict[str, float]:
    """All modes and metrics for one case, keyed like ``subset/recall``."""
    scores: dict[str, float] = {}
    for mode in MATCH_MODES:
        for metric, value in tool_call_scores_for_mode(expected, actual, mode).items():
            scores[f"{mode}/{metric}"] = value
    return scores


def zero_tool_call_scores() -> dict[str, float]:
    """Scores for an errored case: a crash is never credited, even if no calls were expected."""
    return {f"{mode}/{metric}": 0.0 for mode in MATCH_MODES for metric in TOOL_CALL_METRICS}
