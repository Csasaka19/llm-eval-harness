"""Expected values are derived by hand in comments."""

import pytest

from llm_eval.metrics.tool_calls import (
    calls_match,
    json_equal,
    lcs_length,
    max_matching,
    tool_call_scores,
    tool_call_scores_for_mode,
)
from llm_eval.scoring import score_run
from llm_eval.types import CaseResult, EvalCase, JSONValue, SystemResponse, ToolCall

WEATHER = ToolCall("get_weather", {"city": "Nairobi", "unit": "celsius"})
SEARCH = ToolCall("search", {"q": "flights"})
BOOK = ToolCall("book", {"id": 7})


@pytest.mark.parametrize(
    ("a", "b", "equal"),
    [
        (1, 1.0, True),
        (True, 1, False),
        (False, 0, False),
        (True, True, True),
        ("1", 1, False),
        (None, None, True),
        ([1, 2], [1, 2], True),
        ([1, 2], [2, 1], False),
        ({"a": [1, {"b": True}]}, {"a": [1.0, {"b": True}]}, True),
        ({"a": 1}, {"a": 1, "b": 2}, False),
        ({"a": True}, {"a": 1}, False),
    ],
)
def test_json_equal(a: JSONValue, b: JSONValue, equal: bool) -> None:
    assert json_equal(a, b) is equal


def test_match_modes() -> None:
    extra_arg = ToolCall("get_weather", {"city": "Nairobi", "unit": "celsius", "days": 3})
    wrong_val = ToolCall("get_weather", {"city": "Nairobi", "unit": "fahrenheit"})
    missing = ToolCall("get_weather", {"city": "Nairobi"})
    wrong_name = ToolCall("weather", WEATHER.arguments)

    assert calls_match(WEATHER, WEATHER, "exact")
    assert not calls_match(WEATHER, extra_arg, "exact")
    assert calls_match(WEATHER, extra_arg, "subset")
    assert not calls_match(WEATHER, wrong_val, "subset")
    assert not calls_match(WEATHER, missing, "subset")
    assert calls_match(WEATHER, wrong_val, "name")
    assert not calls_match(WEATHER, wrong_name, "name")


def test_max_matching_is_not_greedy() -> None:
    # Left 0 can match right 0 or 1; left 1 can only match right 0.
    # Greedy (0->0) would leave left 1 unmatched; the maximum matching is 2.
    edges = {(0, 0), (0, 1), (1, 0)}
    assert max_matching(2, 2, lambda i, j: (i, j) in edges) == 2


def test_subset_matching_does_not_steal() -> None:
    # Expected: generic search, then a specific one. Actual: specific, then generic.
    # Under "subset", the first expectation matches both actual calls. A greedy scan would
    # take actual[0] for it and leave expected[1] unmatched. Recall must be 1.0.
    generic = ToolCall("search", {})
    specific = ToolCall("search", {"q": "x"})
    actual = [ToolCall("search", {"q": "x"}), ToolCall("search", {"q": "y"})]
    scores = tool_call_scores_for_mode([generic, specific], actual, "subset")
    assert scores["recall"] == 1.0
    assert scores["precision"] == 1.0


def test_lcs() -> None:
    a, b = "ABCBDAB", "BDCABA"  # classic example, LCS length 4
    assert lcs_length(len(a), len(b), lambda i, j: a[i] == b[j]) == 4


def test_perfect_sequence() -> None:
    s = tool_call_scores_for_mode([SEARCH, BOOK], [SEARCH, BOOK], "exact")
    assert s == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "ordered_recall": 1.0,
        "sequence_match": 1.0,
    }


def test_right_calls_wrong_order() -> None:
    s = tool_call_scores_for_mode([SEARCH, BOOK], [BOOK, SEARCH], "exact")
    assert (s["precision"], s["recall"], s["f1"]) == (1.0, 1.0, 1.0)
    assert s["ordered_recall"] == 0.5  # LCS is 1 of 2
    assert s["sequence_match"] == 0.0


def test_extra_and_missing_calls() -> None:
    # Expected [SEARCH, BOOK]; made [SEARCH, WEATHER]. 1 match.
    s = tool_call_scores_for_mode([SEARCH, BOOK], [SEARCH, WEATHER], "exact")
    assert s["precision"] == 0.5
    assert s["recall"] == 0.5
    assert s["f1"] == 0.5
    assert s["sequence_match"] == 0.0


def test_duplicate_calls_only_match_once() -> None:
    # Calling the same tool twice when it was expected once: precision 1/2.
    s = tool_call_scores_for_mode([SEARCH], [SEARCH, SEARCH], "exact")
    assert (s["precision"], s["recall"]) == (0.5, 1.0)
    assert s["f1"] == pytest.approx(2 / 3)


def test_no_calls_expected_and_none_made() -> None:
    s = tool_call_scores_for_mode([], [], "exact")
    assert all(v == 1.0 for v in s.values())


def test_no_calls_expected_but_some_made() -> None:
    s = tool_call_scores_for_mode([], [SEARCH], "exact")
    assert s["precision"] == 0.0
    assert s["recall"] == 1.0  # vacuous
    assert s["f1"] == 0.0
    assert s["sequence_match"] == 0.0


def test_calls_expected_but_none_made() -> None:
    s = tool_call_scores_for_mode([SEARCH], [], "exact")
    assert s["precision"] == 1.0  # no wrong calls made; recall carries the penalty
    assert s["recall"] == 0.0
    assert s["f1"] == 0.0


def test_leniency_shows_up_per_mode() -> None:
    made = ToolCall("get_weather", {"city": "Nairobi", "unit": "fahrenheit"})
    scores = tool_call_scores([WEATHER], [made])
    assert scores["exact/recall"] == 0.0
    assert scores["subset/recall"] == 0.0
    assert scores["name/recall"] == 1.0
    assert len(scores) == 15  # 3 modes x 5 metrics


def test_scoring_unlabelled_vs_no_calls_vs_errored() -> None:
    cases = [
        EvalCase("unlabelled", "q"),
        EvalCase("no-tools", "q", expected_tool_calls=()),
        EvalCase("crashed", "q", expected_tool_calls=()),
        EvalCase("one", "q", expected_tool_calls=(SEARCH,)),
    ]
    results = [
        CaseResult("unlabelled", 1.0, SystemResponse("x", tool_calls=(SEARCH,))),
        CaseResult("no-tools", 1.0, SystemResponse("x")),
        CaseResult("crashed", 1.0, error="Boom"),
        CaseResult("one", 1.0, SystemResponse("x", tool_calls=(SEARCH, BOOK))),
    ]
    card = score_run(cases, results)
    by_id = {c.case_id: c for c in card.cases}
    assert by_id["unlabelled"].tool_calls is None
    assert by_id["no-tools"].tool_calls is not None
    assert by_id["no-tools"].tool_calls["exact/f1"] == 1.0
    assert by_id["crashed"].tool_calls is not None
    assert set(by_id["crashed"].tool_calls.values()) == {0.0}

    f1 = card.tool_calls["exact/f1"]
    assert (f1.n, f1.n_errored) == (3, 1)
    # no-tools=1, crashed=0, one: p=1/2 r=1 -> f1=2/3
    assert f1.mean == pytest.approx((1 + 0 + 2 / 3) / 3)
