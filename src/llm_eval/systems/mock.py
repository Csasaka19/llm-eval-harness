"""Deterministic systems for tests, smoke runs, and CI. No network, no API keys."""

from __future__ import annotations

from collections.abc import Mapping

from llm_eval.types import EvalCase, SystemResponse


class EchoSystem:
    """Returns the input as the answer. Useful as a floor baseline and for smoke tests."""

    name = "echo"

    def run(self, case: EvalCase) -> SystemResponse:
        return SystemResponse(answer=case.input, model="echo")


class ScriptedSystem:
    """Returns pre-recorded responses keyed by case id; raises for unknown cases."""

    def __init__(self, responses: Mapping[str, SystemResponse], name: str = "scripted") -> None:
        self._responses = dict(responses)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def run(self, case: EvalCase) -> SystemResponse:
        try:
            return self._responses[case.id]
        except KeyError:
            raise KeyError(f"no scripted response for case {case.id!r}") from None
