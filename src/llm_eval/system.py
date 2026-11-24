"""The interface every system under test implements, plus loading by import path."""

from __future__ import annotations

import importlib
from typing import Protocol, runtime_checkable

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
}


def load_system(spec: str) -> System:
    """Instantiate a system from a builtin alias or a ``package.module:ClassName`` spec.

    The class must be constructible with no arguments.
    """
    target = BUILTIN_SYSTEMS.get(spec, spec)
    module_name, sep, attr = target.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(
            f"invalid system spec {spec!r}: expected a builtin "
            f"({', '.join(sorted(BUILTIN_SYSTEMS))}) or 'module:ClassName'"
        )
    module = importlib.import_module(module_name)
    factory = getattr(module, attr)
    instance = factory()
    if not isinstance(instance, System):
        raise TypeError(f"{target} does not implement the System protocol (name, run)")
    return instance
