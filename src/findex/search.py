from __future__ import annotations

import functools
import heapq
import logging
from operator import itemgetter
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from findex.index import Index, Int32Array
from findex.query import parse_query
from findex.ranking import BM25, Scorer, SearchResult, extract_snippet
from findex.timing import timed

log = logging.getLogger(__name__)

_INDEX_REGISTRY: dict[int, Index] = {}


@runtime_checkable
class _ArrayScorer(Protocol):
    def score_array(
        self,
        term: str,
        doc_ids: Int32Array,
        tfs: Int32Array,
        index: Index,
    ) -> npt.NDArray[np.float64]: ...


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


def search_scalar(
    index: Index,
    query: str,
    *,
    scorer: Scorer,
    k: int = 10,
    with_snippets: bool = True,
) -> list[SearchResult]:
    """Reference Lab 3 scalar retrieval path used for equivalence verification and benchmarks."""
    matched_ids = set(get_matched_doc_ids(index, query))
    if not matched_ids:
        return []

    tree = parse_query(query)
    positive_terms = list(dict.fromkeys(tree.terms()))
    scores: dict[int, float] = {doc_id: 0.0 for doc_id in sorted(matched_ids)}

    for term in positive_terms:
        plist = index.postings.get(term, [])
        for posting in plist:
            if posting.doc_id in scores:
                scores[posting.doc_id] += scorer.score(term, posting, index)

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


@timed
def search(
    index: Index,
    query: str,
    *,
    scorer: Scorer | None = None,
    k: int = 10,
    with_snippets: bool = True,
) -> list[SearchResult]:
    if k <= 0:
        return []
    active_scorer: Scorer = scorer if scorer is not None else BM25()
    if not isinstance(active_scorer, _ArrayScorer):
        return search_scalar(
            index,
            query,
            scorer=active_scorer,
            k=k,
            with_snippets=with_snippets,
        )

    matched_tuple = get_matched_doc_ids(index, query)
    if not matched_tuple:
        return []

    tree = parse_query(query)
    positive_terms = list(dict.fromkeys(tree.terms()))

    arr_size = int(index.doc_lengths_arr.size)
    if arr_size == 0:
        return []

    all_scores = np.zeros(arr_size, dtype=np.float64)
    for term in positive_terms:
        doc_ids = index.postings_doc_ids.get(term)
        if doc_ids is not None and doc_ids.size > 0:
            tfs = index.postings_tfs[term]
            all_scores[doc_ids] += active_scorer.score_array(term, doc_ids, tfs, index)

    matched_arr = np.asarray(matched_tuple, dtype=np.int32)
    matched_scores = all_scores[matched_arr]

    n_matched = int(matched_arr.size)
    top_n = min(k, n_matched)
    if top_n < n_matched:
        # Sort by (-score, doc_id) before partitioning so ties preserve ascending doc_id order
        order = np.lexsort((matched_arr, -matched_scores))
        part_idx = order[:top_n]
    else:
        part_idx = np.arange(n_matched)

    cand_docs = [int(x) for x in matched_arr[part_idx].tolist()]
    cand_scores = [float(x) for x in matched_scores[part_idx].tolist()]
    ranked_pairs = sorted(
        zip(cand_docs, cand_scores, strict=True),
        key=lambda pair: (-pair[1], pair[0]),
    )

    results: list[SearchResult] = []
    for doc_id, score in ranked_pairs:
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
