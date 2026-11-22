from __future__ import annotations

import argparse
import functools
import heapq
import logging
import time
import tracemalloc
from operator import itemgetter
from pathlib import Path

from findex.index import Index, Posting
from findex.query import parse_query
from findex.ranking import BM25, Scorer, SearchResult, TfIdf, extract_snippet
from findex.store import open_index
from findex.timing import timed

logger = logging.getLogger("findex.search")

_INDEX_REGISTRY: dict[int, Index] = {}


@functools.lru_cache(maxsize=256)
def _cached_match_ids(
    index_id: int,
    num_docs: int,
    num_terms: int,
    raw_query: str,
) -> tuple[int, ...]:
    logger.info("[cache MISS] parsing and evaluating query=%r", raw_query)
    index = _INDEX_REGISTRY[index_id]
    tree = parse_query(raw_query)
    matched = tree.evaluate(index)
    return tuple(sorted(matched))


def get_matched_doc_ids(index: Index, raw_query: str) -> tuple[int, ...]:
    idx_id = id(index)
    _INDEX_REGISTRY[idx_id] = index
    before = _cached_match_ids.cache_info().hits
    res = _cached_match_ids(idx_id, index.num_docs, len(index), raw_query)
    after = _cached_match_ids.cache_info().hits
    if after > before:
        logger.info("[cache HIT] query=%r", raw_query)
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
        snippet = (
            extract_snippet(meta.path, positive_terms)
            if with_snippets
            else ""
        )
        results.append(
            SearchResult(
                score=score,
                doc_id=doc_id,
                title=meta.title,
                snippet=snippet,
            )
        )
    return results


def run_ranking_evaluation(index: Index) -> None:
    bm25 = BM25()
    tfidf = TfIdf()

    print("\n=== 1. Three Ranking Sanity Checks (BM25) ===")

    # Check 1: Rare term outranks common term
    sorted_terms = sorted(index.postings.items(), key=lambda kv: len(kv[1]))
    rare_term = next(t for t, pl in sorted_terms if len(pl) == 2 and t.isalpha() and len(t) > 4)
    common_term = "the"
    res_check1 = search(index, f"{rare_term} OR {common_term}", scorer=bm25, k=3, with_snippets=False)
    print(f"\n[Check 1] Rare term '{rare_term}' (df=2) vs common '{common_term}' (df={index.df(common_term)}):")
    for r in res_check1:
        has_rare = any(p.doc_id == r.doc_id for p in index[rare_term])
        print(f"  doc_id={r.doc_id} | score={r.score:.4f} | has_rare={has_rare} | {r.title}")

    # Check 2: Saturation (20th repetition adds almost nothing)
    sat_term = "generator"
    p_tf1 = next(p for p in index[sat_term] if p.tf == 1)

    s1 = bm25.score(sat_term, Posting(doc_id=p_tf1.doc_id, tf=1), index)
    s19 = bm25.score(sat_term, Posting(doc_id=p_tf1.doc_id, tf=19), index)
    s20 = bm25.score(sat_term, Posting(doc_id=p_tf1.doc_id, tf=20), index)
    print(f"\n[Check 2] BM25 saturation on '{sat_term}' (doc_id={p_tf1.doc_id}):")
    print(f"  tf=1:  {s1:.4f}")
    print(f"  tf=19: {s19:.4f}")
    print(f"  tf=20: {s20:.4f} (marginal gain of 20th rep: +{s20 - s19:.6f})")

    # Check 3: Short doc with 1 match outranks very long doc with 1 match
    single_hits = [p for p in index["asyncio"] if p.tf == 1]
    shortest = min(single_hits, key=lambda p: index.doc_length(p.doc_id))
    longest = max(single_hits, key=lambda p: index.doc_length(p.doc_id))
    s_short = bm25.score("asyncio", shortest, index)
    s_long = bm25.score("asyncio", longest, index)
    print("\n[Check 3] Length normalization on 'asyncio' (both tf=1):")
    print(f"  Short doc [{shortest.doc_id}] (len={index.doc_length(shortest.doc_id)}): score={s_short:.4f}")
    print(f"  Long  doc [{longest.doc_id}] (len={index.doc_length(longest.doc_id)}): score={s_long:.4f}")

    # Precision@5 evaluation on 10 queries with title-keyword relevance ground truth
    eval_queries = [
        ("asyncio", ["asyncio"]),
        ("sqlite3", ["sqlite3"]),
        ("itertools", ["itertools"]),
        ("pathlib", ["pathlib"]),
        ("subprocess", ["subprocess"]),
        ("collections", ["collections"]),
        ("threading", ["threading"]),
        ("unittest", ["unittest"]),
        ("json", ["json"]),
        ("argparse", ["argparse"]),
    ]

    print("\n=== 2. Precision@5 Evaluation (10 Queries: TF-IDF vs BM25) ===")
    print("| Query | TF-IDF P@5 | BM25 P@5 | Top-1 BM25 Document |")
    print("| :--- | ---: | ---: | :--- |")

    tfidf_scores: list[float] = []
    bm25_scores: list[float] = []

    for q, rel_keywords in eval_queries:
        r_tfidf = search(index, q, scorer=tfidf, k=5, with_snippets=False)
        r_bm25 = search(index, q, scorer=bm25, k=5, with_snippets=False)

        p5_tfidf = sum(
            1 for r in r_tfidf if any(kw in r.title.casefold() for kw in rel_keywords)
        ) / 5.0
        p5_bm25 = sum(
            1 for r in r_bm25 if any(kw in r.title.casefold() for kw in rel_keywords)
        ) / 5.0

        tfidf_scores.append(p5_tfidf)
        bm25_scores.append(p5_bm25)
        top1_title = r_bm25[0].title if r_bm25 else "-"
        print(f"| `{q}` | {p5_tfidf:.2f} | {p5_bm25:.2f} | {top1_title} |")

    mean_tfidf = sum(tfidf_scores) / len(tfidf_scores)
    mean_bm25 = sum(bm25_scores) / len(bm25_scores)
    print(f"| **Mean P@5** | **{mean_tfidf:.2f}** | **{mean_bm25:.2f}** | — |")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Ranked search over inverted index.")
    parser.add_argument("index_path", type=Path, help="Path to saved index file")
    parser.add_argument("query", nargs="?", default="", help="Search query")
    parser.add_argument(
        "--scorer",
        choices=("bm25", "tfidf"),
        default="bm25",
        help="Ranking algorithm (default: bm25)",
    )
    parser.add_argument("--top", type=int, default=5, help="Top-k results to display")
    parser.add_argument(
        "--repeat",
        action="store_true",
        help="Run the query twice to demonstrate lru_cache hit",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run sanity checks and 10-query Precision@5 evaluation",
    )
    args = parser.parse_args()

    tracemalloc.start()
    with open_index(args.index_path) as index:
        if args.eval:
            run_ranking_evaluation(index)
            tracemalloc.stop()
            return

        scorer: Scorer = BM25() if args.scorer == "bm25" else TfIdf()
        t0 = time.perf_counter()
        results = search(index, args.query, scorer=scorer, k=args.top)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if args.repeat:
            _ = search(index, args.query, scorer=scorer, k=args.top)

        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"\nIndex:   {index!r}")
        print(f"Query:   {args.query!r} (scorer={args.scorer})")
        print(f"Results: {len(results)} (top-{args.top}) in {elapsed_ms:.2f} ms")
        print(f"Peak memory: {peak_bytes / (1024 * 1024):.2f} MiB\n")

        for rank, res in enumerate(results, start=1):
            print(f"{rank}. {res}")


if __name__ == "__main__":
    main()