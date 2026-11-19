from __future__ import annotations

import argparse
import itertools
import time
import tracemalloc
from collections import Counter
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize


def run_lazy_stats(root: Path, limit: int | None = None) -> tuple[int, int, Counter[str]]:
    docs = iter_documents(root)
    if limit is not None:
        docs = itertools.islice(docs, limit)

    doc_count = 0
    total_tokens = 0
    vocab: Counter[str] = Counter()

    for doc in docs:
        doc_count += 1
        for token in tokenize(doc.text):
            total_tokens += 1
            vocab[token] += 1

    return doc_count, total_tokens, vocab


def run_eager_stats(root: Path, limit: int | None = None) -> tuple[int, int, Counter[str]]:
    """Eager baseline using materialized lists for memory comparison."""
    docs = list(iter_documents(root))
    if limit is not None:
        docs = docs[:limit]

    tokenized_docs: list[list[str]] = [list(tokenize(doc.text)) for doc in docs]
    all_tokens: list[str] = [tok for doc_tokens in tokenized_docs for tok in doc_tokens]

    vocab = Counter(all_tokens)
    return len(docs), len(all_tokens), vocab


def measure(fn, root: Path, limit: int | None):
    tracemalloc.start()
    t0 = time.perf_counter()
    doc_count, total_tokens, vocab = fn(root, limit)
    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return doc_count, total_tokens, vocab, peak_bytes / (1024 * 1024), elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute corpus statistics for findex.")
    parser.add_argument("corpus_dir", type=Path, help="Path to corpus directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run both eager and lazy pipelines and print comparison table",
    )
    args = parser.parse_args()

    doc_count, total_tokens, vocab, peak_mb, elapsed = measure(
        run_lazy_stats, args.corpus_dir, args.limit
    )

    print(f"Documents:       {doc_count}")
    print(f"Total tokens:    {total_tokens}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Elapsed time:    {elapsed:.3f} s")
    print(f"Peak memory:     {peak_mb:.2f} MiB")
    print("\nTop 50 terms:")
    for rank, (term, freq) in enumerate(vocab.most_common(50), start=1):
        print(f"{rank:>2}. {term:<20} {freq}")

    if args.benchmark:
        e_docs, _, _, e_peak_mb, e_elapsed = measure(
            run_eager_stats, args.corpus_dir, args.limit
        )
        print("\n--- Benchmark Table ---")
        print("| Version | Documents | Peak memory | Elapsed |")
        print("| :--- | ---: | ---: | ---: |")
        print(f"| eager (lists) | {e_docs} | {e_peak_mb:.2f} MiB | {e_elapsed:.3f} s |")
        print(f"| lazy (generators) | {doc_count} | {peak_mb:.2f} MiB | {elapsed:.3f} s |")


if __name__ == "__main__":
    main()