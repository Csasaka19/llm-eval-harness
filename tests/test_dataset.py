from pathlib import Path

import pytest

from llm_eval.dataset import DatasetError, load_jsonl, parse_lines
from llm_eval.types import ToolCall

EXAMPLES = Path(__file__).parent.parent / "examples"


def test_loads_example_dataset() -> None:
    cases = load_jsonl(EXAMPLES / "smoke.jsonl")
    assert [c.id for c in cases] == ["rag-001", "rag-002", "agent-001"]
    assert cases[0].expected_doc_ids == ("eiffel",)
    assert cases[0].expected_answer == "1889"
    assert cases[2].expected_tool_calls == (
        ToolCall("get_weather", {"city": "Nairobi", "unit": "celsius"}),
    )


def test_minimal_case_has_empty_defaults() -> None:
    [case] = parse_lines(['{"id": "a", "input": "q"}'])
    assert case.expected_doc_ids == ()
    assert case.expected_answer is None
    assert case.expected_tool_calls == ()
    assert case.metadata == {}


def test_skips_blank_and_comment_lines() -> None:
    cases = parse_lines(["", "// comment", '{"id": "a", "input": "q"}', "   "])
    assert len(cases) == 1


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("not json", "invalid JSON"),
        ("[1, 2]", "must be a JSON object"),
        ('{"input": "q"}', "'id' is required"),
        ('{"id": "a", "input": ""}', "'input' is required"),
        ('{"id": "a", "input": "q", "bogus": 1}', "unknown keys ['bogus']"),
        ('{"id": "a", "input": "q", "expected_doc_ids": [1]}', "list of strings"),
        ('{"id": "a", "input": "q", "expected_answer": 5}', "'expected_answer'"),
        ('{"id": "a", "input": "q", "expected_tool_calls": [{"arguments": {}}]}', "'name'"),
        ('{"id": "a", "input": "q", "metadata": []}', "'metadata'"),
    ],
)
def test_validation_errors_include_location(line: str, message: str) -> None:
    with pytest.raises(DatasetError, match=r"data\.jsonl:2: ") as exc:
        parse_lines(['{"id": "ok", "input": "q"}', line], source="data.jsonl")
    assert message in str(exc.value)


def test_duplicate_ids_rejected() -> None:
    with pytest.raises(DatasetError, match="duplicate id 'a' \\(first seen on line 1\\)"):
        parse_lines(['{"id": "a", "input": "q"}', '{"id": "a", "input": "r"}'])
