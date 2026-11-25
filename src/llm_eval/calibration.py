"""Measure a judge's agreement with human labels before trusting its scores.

Input: JSONL of human-labelled claims::

    {"id": "c1", "claim": "...", "contexts": ["..."], "label": "supported",
     "note": "optional: why this example is tricky"}

Output: confusion matrix, accuracy, Cohen's kappa, and how well the judge catches
unsupported claims (the class that matters: missed hallucinations).

Accuracy alone is misleading when labels are imbalanced: a judge that always says
"supported" scores 80% on a set that is 80% supported. Kappa corrects for that, and is
0 for a judge no better than chance.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from llm_eval.judges import ClaimVerdict, Judge
from llm_eval.types import JSONValue

LABELS = ("supported", "unsupported")


class CalibrationError(ValueError):
    """Raised for malformed calibration files."""


@dataclass(frozen=True)
class LabelledClaim:
    id: str
    claim: str
    contexts: tuple[str, ...]
    supported: bool
    note: str | None = None


@dataclass(frozen=True)
class Disagreement:
    id: str
    claim: str
    human_supported: bool
    judge: ClaimVerdict
    note: str | None


@dataclass(frozen=True)
class CalibrationReport:
    judge: str
    n: int
    # Confusion matrix, rows = human, cols = judge.
    both_supported: int
    human_supported_judge_unsupported: int
    human_unsupported_judge_supported: int
    both_unsupported: int
    accuracy: float
    cohens_kappa: float | None
    unsupported_recall: float | None
    unsupported_precision: float | None
    disagreements: tuple[Disagreement, ...]


def load_labels(path: str | Path) -> list[LabelledClaim]:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        return parse_labels(fh, source=str(path))


def parse_labels(lines: Iterable[str], source: str = "<input>") -> list[LabelledClaim]:
    out: list[LabelledClaim] = []
    seen: set[str] = set()
    for lineno, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        where = f"{source}:{lineno}"
        try:
            obj: JSONValue = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CalibrationError(f"{where}: invalid JSON: {exc.msg}") from exc
        if not isinstance(obj, dict):
            raise CalibrationError(f"{where}: each line must be a JSON object")
        cid, claim, label = obj.get("id"), obj.get("claim"), obj.get("label")
        contexts, note = obj.get("contexts"), obj.get("note")
        if not isinstance(cid, str) or not cid:
            raise CalibrationError(f"{where}: 'id' must be a non-empty string")
        if cid in seen:
            raise CalibrationError(f"{where}: duplicate id {cid!r}")
        if not isinstance(claim, str) or not claim:
            raise CalibrationError(f"{where}: 'claim' must be a non-empty string")
        if label not in LABELS:
            raise CalibrationError(f"{where}: 'label' must be one of {list(LABELS)}")
        if not isinstance(contexts, list) or not all(isinstance(c, str) for c in contexts):
            raise CalibrationError(f"{where}: 'contexts' must be a list of strings")
        if note is not None and not isinstance(note, str):
            raise CalibrationError(f"{where}: 'note' must be a string")
        seen.add(cid)
        out.append(
            LabelledClaim(
                id=cid,
                claim=claim,
                contexts=tuple(c for c in contexts if isinstance(c, str)),
                supported=label == "supported",
                note=note,
            )
        )
    if not out:
        raise CalibrationError(f"{source}: no labelled claims")
    return out


def cohens_kappa(human: Sequence[bool], judge: Sequence[bool]) -> float | None:
    """Cohen's kappa for two binary raters; ``None`` when undefined (chance agreement = 1)."""
    n = len(human)
    if n == 0 or n != len(judge):
        raise ValueError("need two equal-length, non-empty label sequences")
    observed = sum(1 for h, j in zip(human, judge, strict=True) if h == j) / n
    p_h, p_j = sum(human) / n, sum(judge) / n
    expected = p_h * p_j + (1 - p_h) * (1 - p_j)
    if expected == 1:
        return None
    return (observed - expected) / (1 - expected)


def calibrate(judge: Judge, labelled: Sequence[LabelledClaim]) -> CalibrationReport:
    verdicts = [judge.judge_claim(item.claim, item.contexts) for item in labelled]
    human = [item.supported for item in labelled]
    machine = [v.supported for v in verdicts]

    ss = sum(1 for h, m in zip(human, machine, strict=True) if h and m)
    su = sum(1 for h, m in zip(human, machine, strict=True) if h and not m)
    us = sum(1 for h, m in zip(human, machine, strict=True) if not h and m)
    uu = sum(1 for h, m in zip(human, machine, strict=True) if not h and not m)

    disagreements = tuple(
        Disagreement(item.id, item.claim, item.supported, v, item.note)
        for item, v in zip(labelled, verdicts, strict=True)
        if item.supported != v.supported
    )
    return CalibrationReport(
        judge=judge.name,
        n=len(labelled),
        both_supported=ss,
        human_supported_judge_unsupported=su,
        human_unsupported_judge_supported=us,
        both_unsupported=uu,
        accuracy=(ss + uu) / len(labelled),
        cohens_kappa=cohens_kappa(human, machine),
        unsupported_recall=uu / (uu + us) if uu + us else None,
        unsupported_precision=uu / (uu + su) if uu + su else None,
        disagreements=disagreements,
    )
