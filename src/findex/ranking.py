from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from findex.index import Index, Posting

ScorerName = Literal["bm25", "tfidf"]


class Scorer(Protocol):
    def score(self, term: str, posting: Posting, index: Index) -> float: ...


@dataclass(frozen=True, slots=True)
class TfIdf:
    def score(self, term: str, posting: Posting, index: Index) -> float:
        df = index.df(term)
        if df == 0 or posting.tf == 0:
            return 0.0
        idf = math.log(index.num_docs / df)
        tf_weight = 1.0 + math.log(posting.tf)
        return tf_weight * idf


@dataclass(frozen=True, slots=True)
class BM25:
    k1: float = 1.5
    b: float = 0.75

    def score(self, term: str, posting: Posting, index: Index) -> float:
        df = index.df(term)
        if df == 0 or posting.tf == 0:
            return 0.0
        n = index.num_docs
        idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
        avgdl = index.avg_doc_length or 1.0
        dl = index.doc_length(posting.doc_id)
        denom = posting.tf + self.k1 * (1.0 - self.b + self.b * (dl / avgdl))
        tf_norm = (posting.tf * (self.k1 + 1.0)) / denom
        return idf * tf_norm


def make_scorer(name: ScorerName) -> Scorer:
    if name == "tfidf":
        return TfIdf()
    return BM25()


@dataclass(order=True, frozen=True, slots=True)
class SearchResult:
    score: float
    doc_id: int
    title: str
    snippet: str = ""

    def __str__(self) -> str:
        return f"[{self.score:.4f}] ({self.doc_id}) {self.title}: {self.snippet}"


def extract_snippet(
    doc_path: str,
    terms: list[str],
    *,
    window: int = 80,
) -> str:
    path = Path(doc_path)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return ""

    lowered = compact.casefold()
    best_pos = -1
    for term in terms:
        pos = lowered.find(term.casefold())
        if pos != -1 and (best_pos == -1 or pos < best_pos):
            best_pos = pos

    if best_pos == -1:
        snippet = compact[: window * 2]
    else:
        start = max(0, best_pos - window)
        end = min(len(compact), best_pos + window)
        snippet = compact[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(compact):
            snippet = snippet + "..."

    unique_terms = sorted({t for t in terms if t}, key=len, reverse=True)
    if unique_terms:
        pattern = re.compile(
            r"\b(" + "|".join(re.escape(t) for t in unique_terms) + r")\b",
            re.IGNORECASE,
        )
        snippet = pattern.sub(r"[\1]", snippet)

    return snippet
