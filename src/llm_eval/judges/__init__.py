"""Faithfulness judges: decide whether a claim is supported by retrieved context.

No judge's scores should be trusted until its agreement with human labels has been
measured with ``llm-eval calibrate``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from llm_eval.plugins import load_plugin


@dataclass(frozen=True)
class ClaimVerdict:
    claim: str
    supported: bool
    rationale: str | None = None


@runtime_checkable
class Judge(Protocol):
    """Decides whether a single claim is supported by the given contexts."""

    @property
    def name(self) -> str: ...

    def judge_claim(self, claim: str, contexts: Sequence[str]) -> ClaimVerdict: ...


BUILTIN_JUDGES: dict[str, str] = {
    "overlap": "llm_eval.judges.overlap:TokenOverlapJudge",
}


def load_judge(spec: str) -> Judge:
    instance = load_plugin(spec, BUILTIN_JUDGES, "judge")
    if not isinstance(instance, Judge):
        raise TypeError(f"{spec} does not implement the Judge protocol (name, judge_claim)")
    return instance
