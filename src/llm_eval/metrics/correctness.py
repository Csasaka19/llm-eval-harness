"""Answer correctness against a reference answer (``expected_answer``).

Deterministic, SQuAD-style token metrics:

* ``exact_match``  - normalised answer equals normalised reference.
* ``token_f1``     - F1 over the multiset of normalised tokens.
* ``token_recall`` - share of reference tokens present in the answer. F1 penalises long
  answers even when they contain the right fact; recall does not, so both are reported.

Normalisation: lower-case, drop articles (a, an, the) and punctuation, but keep numbers
intact. Unlike SQuAD, "1.8" stays "1.8" (not "18") and "1,000" becomes "1000".

Limitations: these metrics reward word overlap, not meaning. "Yes" vs "Yes, you can" is
partial credit; a correct paraphrase can score low; a wrong answer that repeats the
reference's words can score high. They are a floor, not a verdict.
"""

from __future__ import annotations

import re
from collections import Counter

_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*")
_ARTICLES = frozenset({"a", "an", "the"})

CORRECTNESS_METRICS: tuple[str, ...] = ("exact_match", "token_f1", "token_recall")


def normalize_tokens(text: str) -> list[str]:
    text = _THOUSANDS.sub("", text.lower())
    return [t for t in _TOKEN.findall(text) if t not in _ARTICLES]


def correctness_scores(answer: str, reference: str) -> dict[str, float]:
    pred, gold = normalize_tokens(answer), normalize_tokens(reference)
    exact = float(pred == gold)
    if not pred or not gold:
        return {"exact_match": exact, "token_f1": exact, "token_recall": exact}
    common = sum((Counter(pred) & Counter(gold)).values())
    precision, recall = common / len(pred), common / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if common else 0.0
    return {"exact_match": exact, "token_f1": f1, "token_recall": recall}


def zero_correctness_scores() -> dict[str, float]:
    return dict.fromkeys(CORRECTNESS_METRICS, 0.0)
