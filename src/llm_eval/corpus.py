"""Document corpora for retrieval systems (JSONL: one {"id", "title", "text"} per line)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from llm_eval.types import JSONValue


class CorpusError(ValueError):
    """Raised for malformed corpus files."""


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    title: str = ""


def load_corpus(path: str | Path) -> list[Document]:
    path = Path(path)
    docs: list[Document] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("//"):
                continue
            where = f"{path}:{lineno}"
            try:
                obj: JSONValue = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusError(f"{where}: invalid JSON: {exc.msg}") from exc
            if not isinstance(obj, dict):
                raise CorpusError(f"{where}: each line must be a JSON object")
            doc_id, text, title = obj.get("id"), obj.get("text"), obj.get("title", "")
            if not isinstance(doc_id, str) or not doc_id:
                raise CorpusError(f"{where}: 'id' must be a non-empty string")
            if doc_id in seen:
                raise CorpusError(f"{where}: duplicate id {doc_id!r}")
            if not isinstance(text, str) or not text:
                raise CorpusError(f"{where}: 'text' must be a non-empty string")
            if not isinstance(title, str):
                raise CorpusError(f"{where}: 'title' must be a string")
            seen.add(doc_id)
            docs.append(Document(id=doc_id, text=text, title=title))
    if not docs:
        raise CorpusError(f"{path}: corpus is empty")
    return docs
