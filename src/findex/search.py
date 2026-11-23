from __future__ import annotations

import functools
import heapq
import logging
from operator import itemgetter

from findex.index import Index
from findex.query import parse_query
from findex.ranking import BM25, Scorer, SearchResult, extract_snippet
from findex.timing import timed

log = logging.getLogger(__name__)

_INDEX_REGISTRY: dict[int, Index] = {}


@functools.lru_cache(maxsize=256)
def _cached_match_ids(
    index_id: int,
    num_docs: int,
    num_terms: int,
    raw_query: str,
) -> tuple[int, ...]:
    log.debug("[cache MISS] parsing and evaluating query=%r", raw_query)
    index = _INDEX_REGISTRY[index_id]
    tree = parse_query(raw_query)
    matched = tree.evaluate(index)
    return tuple(sorted(matched))


def clear_query_cache() -> None:
    _cached_match_ids.cache_clear()


def get_matched_doc_ids(index: Index, raw_query: str) -> tuple[int, ...]:
    idx_id = id(index)
    _INDEX_REGISTRY[idx_id] = index
    before = _cached_match_ids.cache_info().hits
    res = _cached_match_ids(idx_id, index.num_docs, len(index), raw_query)
    after = _cached_match_ids.cache_info().hits
    if after > before:
        log.debug("[cache HIT] query=%r", raw_query)
    return res


@timed
def search(
    index: Index,
    query: str,
    *,
    scorer: Scorer | None = None,
    k: int = 10,
    with_snippets: bool = True,
) -> list[SearchResult]:
    active_scorer: Scorer = scorer if scorer is not None else BM25()
    matched_ids = set(get_matched_doc_ids(index, query))
    if not matched_ids:
        return []

    tree = parse_query(query)
    positive_terms = list(dict.fromkeys(tree.terms()))
    scores: dict[int, float] = {doc_id: 0.0 for doc_id in matched_ids}

    for term in positive_terms:
        plist = index.postings.get(term, [])
        for posting in plist:
            if posting.doc_id in scores:
                scores[posting.doc_id] += active_scorer.score(term, posting, index)

    top_pairs = heapq.nlargest(k, scores.items(), key=itemgetter(1))
    results: list[SearchResult] = []
    for doc_id, score in top_pairs:
        meta = index.doc_meta[doc_id]
        snippet = extract_snippet(meta.path, positive_terms) if with_snippets else ""
        results.append(
            SearchResult(
                score=score,
                doc_id=doc_id,
                title=meta.title,
                snippet=snippet,
            )
        )
    return results
