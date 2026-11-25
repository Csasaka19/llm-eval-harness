import json
from pathlib import Path

import pytest

from llm_eval.cli import main
from llm_eval.corpus import CorpusError, Document, load_corpus
from llm_eval.dataset import load_jsonl
from llm_eval.system import load_system
from llm_eval.systems.bm25 import BM25Index, BM25RagSystem
from llm_eval.systems.rules_agent import RulesAgent
from llm_eval.types import EvalCase, ToolCall

ROOT = Path(__file__).parent.parent
CORPUS = ROOT / "datasets" / "rag" / "corpus.jsonl"
RAG = ROOT / "datasets" / "rag" / "questions.jsonl"
AGENT = ROOT / "datasets" / "agent" / "tasks.jsonl"


# --- shipped datasets ------------------------------------------------------------


def test_shipped_datasets_are_consistent() -> None:
    doc_ids = {d.id for d in load_corpus(CORPUS)}
    questions = load_jsonl(RAG)
    tasks = load_jsonl(AGENT)
    assert len(doc_ids) == 30
    assert len(questions) == 40
    assert len(tasks) == 20
    for case in questions:
        assert case.expected_doc_ids, case.id
        assert set(case.expected_doc_ids) <= doc_ids, case.id
        assert case.expected_answer, case.id
        assert case.metadata.get("type") in {"direct", "paraphrase", "multi-doc", "distractor"}
    for case in tasks:
        assert case.expected_tool_calls is not None, case.id
        assert case.metadata.get("type") in {"single", "multi", "no-tool", "paraphrase"}
    tools = {
        t["name"] for t in json.loads((ROOT / "datasets/agent/tools.json").read_text())["tools"]
    }
    used = {c.name for case in tasks for c in case.expected_tool_calls or ()}
    assert used <= tools


# --- corpus ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('{"text": "x"}', "'id'"),
        ('{"id": "a"}', "'text'"),
        ('{"id": "a", "text": "x"}\n{"id": "a", "text": "y"}', "duplicate"),
        ("// nothing", "empty"),
    ],
)
def test_corpus_validation(tmp_path: Path, body: str, message: str) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text(body)
    with pytest.raises(CorpusError, match=message):
        load_corpus(path)


# --- BM25 ------------------------------------------------------------------------

DOCS = [
    Document("cats", "Cats are small furry pets that purr."),
    Document("dogs", "Dogs are loyal pets that bark."),
    Document("rocks", "Granite is an igneous rock."),
]


def test_bm25_ranks_matching_doc_first() -> None:
    index = BM25Index(DOCS)
    assert index.search("why do cats purr?", 3)[0][0].id == "cats"
    assert [d.id for d, _ in index.search("granite rock", 3)] == ["rocks"]


def test_bm25_rare_terms_outweigh_common_ones() -> None:
    # "pets" appears in 2 docs, "bark" in 1: the doc with the rarer term wins.
    hits = BM25Index(DOCS).search("pets bark", 3)
    assert hits[0][0].id == "dogs"
    assert hits[0][1] > hits[1][1] > 0


def test_bm25_no_match_returns_nothing() -> None:
    assert BM25Index(DOCS).search("quantum chromodynamics", 3) == []


def test_bm25_system_response_shape(tmp_path: Path) -> None:
    corpus = tmp_path / "c.jsonl"
    corpus.write_text("\n".join(json.dumps({"id": d.id, "text": d.text}) for d in DOCS))
    system = BM25RagSystem(corpus, top_k=2, context_k=1)
    resp = system.run(EvalCase("q", "What sound do dogs make? Do they bark?"))
    assert resp.retrieved_doc_ids[0] == "dogs"
    assert len(resp.retrieved_doc_ids) <= 2
    assert resp.retrieved_contexts == ("Dogs are loyal pets that bark.",)
    assert resp.answer == "Dogs are loyal pets that bark."
    assert system.run(EvalCase("q", "zzz")).answer == "I don't know."


def test_bm25_corpus_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus = tmp_path / "c.jsonl"
    corpus.write_text('{"id": "only", "text": "the only document"}')
    monkeypatch.setenv("LLM_EVAL_CORPUS", str(corpus))
    system = load_system("bm25")
    assert isinstance(system, BM25RagSystem)
    assert system.run(EvalCase("q", "only document")).retrieved_doc_ids == ("only",)


# --- rules agent -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "calls"),
    [
        (
            "What's the weather in Lisbon in fahrenheit?",
            [ToolCall("get_weather", {"city": "Lisbon", "unit": "fahrenheit"})],
        ),
        (
            "Convert 12.5 usd to eur",
            [ToolCall("convert_currency", {"amount": 12.5, "from": "USD", "to": "EUR"})],
        ),
        (
            "Search the docs for the parental leave policy, then get the flight log for KX-1.",
            [
                ToolCall("search_docs", {"query": "parental leave policy"}),
                ToolCall("get_flight_log", {"drone_id": "KX-1"}),
            ],
        ),
        (
            "Open a ticket titled 'Broken prop' please",
            [ToolCall("create_ticket", {"title": "Broken prop", "priority": "medium"})],
        ),
        ("Tell me about the weather today", []),
    ],
)
def test_rules_agent(text: str, calls: list[ToolCall]) -> None:
    assert list(RulesAgent().run(EvalCase("t", text)).tool_calls) == calls


def test_rules_agent_emits_calls_in_request_order() -> None:
    text = "Get the flight log for KX-9 and then convert 5 EUR to USD."
    names = [c.name for c in RulesAgent().run(EvalCase("t", text)).tool_calls]
    assert names == ["get_flight_log", "convert_currency"]


# --- bench -----------------------------------------------------------------------


def test_bench_command_runs_and_scores(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "run"
    argv = ["bench", "--dataset", str(AGENT), "--system", "rules-agent", "--out", str(out)]
    assert main(argv) == 0
    printed = capsys.readouterr().out
    assert "20 cases, 0 errors" in printed
    assert "tool calls (n=20" in printed
    scores = json.loads((out / "scores.json").read_text())
    # Pinned so baseline changes are deliberate; see README for the per-type breakdown.
    assert scores["tool_calls"]["exact/f1"]["mean"] == pytest.approx(0.8)


def test_bench_rag_baseline_pinned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    out = tmp_path / "run"
    argv = ["bench", "--dataset", str(RAG), "--system", "bm25", "--out", str(out), "--k", "1", "3"]
    assert main(argv) == 0
    scores = json.loads((out / "scores.json").read_text())
    assert scores["retrieval"]["recall@1"]["mean"] == pytest.approx(0.8625)
    assert scores["retrieval"]["mrr"]["mean"] == pytest.approx(0.9292, abs=1e-4)


def test_bench_stops_if_system_cannot_load(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["bench", "--dataset", str(AGENT), "--system", "nope:Nope", "--out", str(tmp_path)]
    assert main(argv) == 2
    assert "could not load system" in capsys.readouterr().err
