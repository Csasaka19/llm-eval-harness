"""Answer faithfulness: the share of an answer's claims supported by retrieved context.

Claims are produced by splitting the answer into sentences. This is a crude but
transparent extractor; a smarter one can be swapped in later without changing the metric.

* ``faithfulness`` = supported claims / claims. An answer with no claims (for example an
  empty string) scores 1.0 vacuously, so ``n_claims`` is reported alongside: a system
  that says nothing should not look faithful by accident.
* Faithfulness is not correctness. A claim can be faithful to a wrong document.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from llm_eval.judges import ClaimVerdict, Judge

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def split_claims(answer: str) -> list[str]:
    claims = []
    for part in _SENTENCE_END.split(answer):
        text = _BULLET.sub("", part).strip()
        if text:
            claims.append(text)
    return claims


def judge_answer(
    answer: str, contexts: Sequence[str], judge: Judge
) -> tuple[dict[str, float], tuple[ClaimVerdict, ...]]:
    verdicts = tuple(judge.judge_claim(c, contexts) for c in split_claims(answer))
    supported = sum(1 for v in verdicts if v.supported)
    score = supported / len(verdicts) if verdicts else 1.0
    return {"faithfulness": score, "n_claims": float(len(verdicts))}, verdicts


def zero_faithfulness_scores() -> dict[str, float]:
    return {"faithfulness": 0.0, "n_claims": 0.0}
