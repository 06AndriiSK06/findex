from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CorpusStats:
    num_documents: int
    total_tokens: int
    vocab_size: int
    top_terms: list[tuple[str, int]]


def compute_stats(
    root: Path,
    *,
    limit: int | None = None,
    top_n: int = 20,
) -> CorpusStats:
    counts: Counter[str] = Counter()
    num_docs = 0
    total_tokens = 0

    for doc_id, doc in enumerate(iter_documents(root)):
        if limit is not None and doc_id >= limit:
            break
        num_docs += 1
        for token in tokenize(doc.text):
            counts[token] += 1
            total_tokens += 1

    log.info("Computed stats for %d documents (%d tokens)", num_docs, total_tokens)
    return CorpusStats(
        num_documents=num_docs,
        total_tokens=total_tokens,
        vocab_size=len(counts),
        top_terms=counts.most_common(top_n),
    )
