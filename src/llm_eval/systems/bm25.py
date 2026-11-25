"""Baseline RAG system: BM25 retrieval plus an extractive answer. No model, no network.

The answer is the sentence from the top-ranked document that shares the most words with
the question. Because it is copied from retrieved context, this baseline is faithful by
construction; its faithfulness score mostly measures the judge, not the system.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from llm_eval.corpus import Document, load_corpus
from llm_eval.judges.overlap import content_tokens
from llm_eval.metrics.faithfulness import split_claims
from llm_eval.types import EvalCase, SystemResponse

DEFAULT_CORPUS = "datasets/rag/corpus.jsonl"
CORPUS_ENV = "LLM_EVAL_CORPUS"
_PUNCT = re.compile(r"[^\w\s-]")


class BM25Index:
    """Okapi BM25 over title + text, using the same tokeniser as the overlap judge."""

    def __init__(self, docs: Sequence[Document], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = list(docs)
        self.k1, self.b = k1, b
        self._tfs = [Counter(content_tokens(f"{d.title} {d.text}")) for d in self.docs]
        self._lens = [sum(tf.values()) for tf in self._tfs]
        self._avg_len = sum(self._lens) / len(self._lens)
        df: Counter[str] = Counter()
        for tf in self._tfs:
            df.update(tf.keys())
        n = len(self.docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        terms = content_tokens(_PUNCT.sub(" ", query))
        scored = []
        for doc, tf, length in zip(self.docs, self._tfs, self._lens, strict=True):
            score = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * length / self._avg_len)
                    score += self._idf[t] * f * (self.k1 + 1) / norm
            if score > 0:
                scored.append((doc, score))
        scored.sort(key=lambda pair: (-pair[1], pair[0].id))
        return scored[:k]


class BM25RagSystem:
    """Retrieves ``top_k`` docs, passes ``context_k`` as context, answers extractively.

    Corpus path: ``$LLM_EVAL_CORPUS`` or ``datasets/rag/corpus.jsonl`` (relative to cwd).
    """

    def __init__(
        self, corpus_path: str | Path | None = None, top_k: int = 10, context_k: int = 3
    ) -> None:
        path = corpus_path or os.environ.get(CORPUS_ENV) or DEFAULT_CORPUS
        self.index = BM25Index(load_corpus(path))
        self.top_k, self.context_k = top_k, context_k

    @property
    def name(self) -> str:
        return f"bm25-extractive(top_k={self.top_k}, context_k={self.context_k})"

    def run(self, case: EvalCase) -> SystemResponse:
        hits = self.index.search(case.input, self.top_k)
        contexts = tuple(doc.text for doc, _ in hits[: self.context_k])
        answer = _best_sentence(case.input, hits[0][0].text) if hits else "I don't know."
        return SystemResponse(
            answer=answer,
            retrieved_doc_ids=tuple(doc.id for doc, _ in hits),
            retrieved_contexts=contexts,
            model="bm25",
        )


def _best_sentence(question: str, text: str) -> str:
    q = set(content_tokens(question))
    sentences = split_claims(text)
    return max(sentences, key=lambda s: len(q & set(content_tokens(s))))
