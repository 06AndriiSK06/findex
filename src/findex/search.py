from __future__ import annotations

import argparse
import time
import tracemalloc
from pathlib import Path

from findex.index import InvertedIndex
from findex.store import load_index
from findex.tokenize import tokenize


def intersect_merge(a: list[int], b: list[int]) -> list[int]:
    result: list[int] = []
    i = 0
    j = 0
    len_a = len(a)
    len_b = len(b)
    while i < len_a and j < len_b:
        if a[i] == b[j]:
            result.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return result


def union_merge(a: list[int], b: list[int]) -> list[int]:
    result: list[int] = []
    i = 0
    j = 0
    len_a = len(a)
    len_b = len(b)
    while i < len_a and j < len_b:
        if a[i] == b[j]:
            result.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    if i < len_a:
        result.extend(a[i:])
    if j < len_b:
        result.extend(b[j:])
    return result


def difference_merge(a: list[int], b: list[int]) -> list[int]:
    result: list[int] = []
    i = 0
    j = 0
    len_a = len(a)
    len_b = len(b)
    while i < len_a and j < len_b:
        if a[i] == b[j]:
            i += 1
            j += 1
        elif a[i] < b[j]:
            result.append(a[i])
            i += 1
        else:
            j += 1
    if i < len_a:
        result.extend(a[i:])
    return result


def _term_doc_ids(index: InvertedIndex, raw_term: str) -> list[int]:
    tokens = list(tokenize(raw_term))
    if not tokens:
        return []
    plist = index.postings.get(tokens[0], [])
    return [p.doc_id for p in plist]


def search(
    index: InvertedIndex,
    query: str,
    *,
    engine: str = "merge",
) -> list[int]:
    raw_parts = query.strip().split()
    if not raw_parts:
        return []

    all_docs = sorted(index.doc_meta.keys())
    tokens: list[str] = []
    for part in raw_parts:
        upper = part.upper()
        if upper in {"AND", "OR", "NOT"}:
            tokens.append(upper)
        else:
            norm = list(tokenize(part))
            if norm:
                tokens.append(norm[0])

    if not tokens:
        return []

    if engine == "set":
        return _search_set(index, tokens, all_docs)
    return _search_merge(index, tokens, all_docs)


def _search_merge(
    index: InvertedIndex,
    tokens: list[str],
    all_docs: list[int],
) -> list[int]:
    idx = 0
    negate_first = False
    if tokens[idx] == "NOT":
        negate_first = True
        idx += 1
    if idx >= len(tokens):
        return []

    current = _term_doc_ids(index, tokens[idx])
    if negate_first:
        current = difference_merge(all_docs, current)
    idx += 1

    while idx < len(tokens):
        op = "AND"
        if tokens[idx] in {"AND", "OR"}:
            op = tokens[idx]
            idx += 1
        if idx >= len(tokens):
            break

        negate = False
        if tokens[idx] == "NOT":
            negate = True
            idx += 1
        if idx >= len(tokens):
            break

        rhs = _term_doc_ids(index, tokens[idx])
        if negate:
            rhs = difference_merge(all_docs, rhs)

        if op == "OR":
            current = union_merge(current, rhs)
        else:
            current = intersect_merge(current, rhs)
        idx += 1

    return current


def _search_set(
    index: InvertedIndex,
    tokens: list[str],
    all_docs: list[int],
) -> list[int]:
    all_set = set(all_docs)
    idx = 0
    negate_first = False
    if tokens[idx] == "NOT":
        negate_first = True
        idx += 1
    if idx >= len(tokens):
        return []

    current = set(_term_doc_ids(index, tokens[idx]))
    if negate_first:
        current = all_set - current
    idx += 1

    while idx < len(tokens):
        op = "AND"
        if tokens[idx] in {"AND", "OR"}:
            op = tokens[idx]
            idx += 1
        if idx >= len(tokens):
            break

        negate = False
        if tokens[idx] == "NOT":
            negate = True
            idx += 1
        if idx >= len(tokens):
            break

        rhs = set(_term_doc_ids(index, tokens[idx]))
        if negate:
            rhs = all_set - rhs

        if op == "OR":
            current = current | rhs
        else:
            current = current & rhs
        idx += 1

    return sorted(current)


def run_engine_benchmark(index: InvertedIndex, iterations: int = 2000) -> None:
    sorted_terms = sorted(index.postings.items(), key=lambda kv: len(kv[1]))
    rarest = [sorted_terms[0][0], sorted_terms[1][0]]
    most_common = [sorted_terms[-1][0], sorted_terms[-2][0]]

    cases = [
        ("2 most common (AND)", f"{most_common[0]} {most_common[1]}"),
        ("2 most common (OR)", f"{most_common[0]} OR {most_common[1]}"),
        ("2 rarest (AND)", f"{rarest[0]} {rarest[1]}"),
        ("2 rarest (OR)", f"{rarest[0]} OR {rarest[1]}"),
    ]

    print(f"\nMost common terms: {most_common[0]} ({len(index.postings[most_common[0]])} docs), "
          f"{most_common[1]} ({len(index.postings[most_common[1]])} docs)")
    print(f"Rarest terms:      {rarest[0]} ({len(index.postings[rarest[0]])} docs), "
          f"{rarest[1]} ({len(index.postings[rarest[1]])} docs)")

    print(f"\n--- Boolean Engine Benchmark ({iterations} iterations) ---")
    print("| Query type | Terms | merge | set |")
    print("| :--- | :--- | ---: | ---: |")

    for label, query in cases:
        t0 = time.perf_counter()
        for _ in range(iterations):
            res_merge = search(index, query, engine="merge")
        t_merge = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        for _ in range(iterations):
            res_set = search(index, query, engine="set")
        t_set = (time.perf_counter() - t0) * 1000

        assert res_merge == res_set
        print(f"| {label} | `{query}` | {t_merge:.2f} ms | {t_set:.2f} ms |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search persisted inverted index.")
    parser.add_argument("index_path", type=Path, help="Path to saved index file")
    parser.add_argument("query", nargs="?", default="", help="Boolean search query")
    parser.add_argument(
        "--engine",
        choices=("merge", "set"),
        default="merge",
        help="Boolean intersection/union engine",
    )
    parser.add_argument("--top", type=int, default=10, help="Max results to display")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark merge vs set on most common and rarest terms",
    )
    args = parser.parse_args()

    tracemalloc.start()
    t_load0 = time.perf_counter()
    index = load_index(args.index_path)
    load_elapsed = time.perf_counter() - t_load0

    if args.benchmark:
        run_engine_benchmark(index)
        tracemalloc.stop()
        return

    t_search0 = time.perf_counter()
    doc_ids = search(index, args.query, engine=args.engine)
    search_elapsed = time.perf_counter() - t_search0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Query:         {args.query!r} (engine={args.engine})")
    print(f"Matches:       {len(doc_ids)}")
    print(f"Load time:     {load_elapsed:.4f} s")
    print(f"Search time:   {search_elapsed * 1000:.3f} ms")
    print(f"Peak memory:   {peak_bytes / (1024 * 1024):.2f} MiB\n")

    for rank, doc_id in enumerate(doc_ids[: args.top], start=1):
        meta = index.doc_meta[doc_id]
        print(f"{rank}. [{doc_id}] {meta.title} ({meta.path})")


if __name__ == "__main__":
    main()