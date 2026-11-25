"""The interface every system under test implements, plus loading by import path."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from llm_eval.plugins import load_plugin
from llm_eval.types import EvalCase, SystemResponse


@runtime_checkable
class System(Protocol):
    """A RAG pipeline, agent, or anything else that maps an :class:`EvalCase` to a response.

    Implementations should raise on failure rather than returning a sentinel; the runner
    records the exception and keeps going.
    """

    @property
    def name(self) -> str: ...

    def run(self, case: EvalCase) -> SystemResponse: ...


BUILTIN_SYSTEMS: dict[str, str] = {
    "echo": "llm_eval.systems.mock:EchoSystem",
    "bm25": "llm_eval.systems.bm25:BM25RagSystem",
    "rules-agent": "llm_eval.systems.rules_agent:RulesAgent",
}


def load_system(spec: str) -> System:
    """Instantiate a system from a builtin alias or a ``package.module:ClassName`` spec.

    The class must be constructible with no arguments.
    """
    instance = load_plugin(spec, BUILTIN_SYSTEMS, "system")
    if not isinstance(instance, System):
        raise TypeError(f"{spec} does not implement the System protocol (name, run)")
    return instance
