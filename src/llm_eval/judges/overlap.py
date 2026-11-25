"""A deterministic, dependency-free baseline judge based on token overlap.

It is deliberately simple and has known blind spots, which calibration exposes:

* Negation: "X is not Y" overlaps almost fully with "X is Y".
* Paraphrase: a correct claim in different words can fall below the threshold.
* Entity swaps where both entities appear somewhere in the context.

It does require every number in the claim to appear in the context, because changed
figures are among the most common and most harmful hallucinations.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from llm_eval.judges import ClaimVerdict

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_NUMBER = re.compile(r"^[0-9][0-9.,]*$")
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "she",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "was",
        "were",
        "which",
        "who",
        "will",
        "with",
    ]
)


def content_tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class TokenOverlapJudge:
    """Supported iff all numbers appear in the context and at least ``threshold`` of the
    claim's content tokens appear in the (concatenated) contexts."""

    def __init__(self, threshold: float = 0.75) -> None:
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        self.threshold = threshold

    @property
    def name(self) -> str:
        return f"overlap(threshold={self.threshold})"

    def judge_claim(self, claim: str, contexts: Sequence[str]) -> ClaimVerdict:
        claim_tokens = content_tokens(claim)
        if not claim_tokens:
            return ClaimVerdict(claim, True, "no content tokens")
        context_tokens = set(content_tokens(" ".join(contexts)))

        missing_numbers = [t for t in claim_tokens if _NUMBER.match(t) and t not in context_tokens]
        if missing_numbers:
            return ClaimVerdict(
                claim, False, f"numbers not in context: {', '.join(missing_numbers)}"
            )

        present = sum(1 for t in claim_tokens if t in context_tokens)
        ratio = present / len(claim_tokens)
        return ClaimVerdict(
            claim,
            ratio >= self.threshold,
            f"{present}/{len(claim_tokens)} content tokens in context ({ratio:.2f})",
        )
