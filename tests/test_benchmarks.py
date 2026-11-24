from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from findex.index import Index, build_index
from findex.ranking import BM25, BM25Scalar, TfIdf, TfIdfScalar
from findex.search import clear_query_cache, search, search_scalar


@pytest.mark.parametrize(
    "query",
    [
        "python",
        "quaternion",
        "python async",
        "python OR asyncio",
        "python NOT java",
        '"event loop"',
    ],
)
def test_numpy_scorer_matches_lab03_ranking_exactly(built_index: Index, query: str) -> None:
    clear_query_cache()
    vec_bm25 = search(built_index, query, scorer=BM25(), k=10, with_snippets=False)
    clear_query_cache()
    ref_bm25 = search_scalar(built_index, query, scorer=BM25Scalar(), k=10, with_snippets=False)

    assert [r.doc_id for r in vec_bm25] == [r.doc_id for r in ref_bm25]
    for v_res, r_res in zip(vec_bm25, ref_bm25, strict=True):
        assert math.isclose(v_res.score, r_res.score, rel_tol=1e-12, abs_tol=1e-12)

    clear_query_cache()
    vec_tfidf = search(built_index, query, scorer=TfIdf(), k=10, with_snippets=False)
    clear_query_cache()
    ref_tfidf = search_scalar(built_index, query, scorer=TfIdfScalar(), k=10, with_snippets=False)

    assert [r.doc_id for r in vec_tfidf] == [r.doc_id for r in ref_tfidf]
    for v_res, r_res in zip(vec_tfidf, ref_tfidf, strict=True):
        assert math.isclose(v_res.score, r_res.score, rel_tol=1e-12, abs_tol=1e-12)


@pytest.fixture
def bench_corpus_dir(tmp_path: Path) -> Path:
    corpus = tmp_path / "bench_corpus"
    corpus.mkdir()
    for i in range(150):
        (corpus / f"doc_{i:03d}.txt").write_text(
            f"Document {i}: Python asyncio event loop, NumPy vectorization, and index ranking.",
            encoding="utf-8",
        )
    return corpus


def test_benchmark_build_index(benchmark: Any, bench_corpus_dir: Path) -> None:
    idx = benchmark(build_index, bench_corpus_dir, workers=1, with_positions=False)
    assert idx.num_docs == 150


def test_benchmark_search(benchmark: Any, bench_corpus_dir: Path) -> None:
    idx = build_index(bench_corpus_dir, workers=1, with_positions=False)

    def run_query() -> list[Any]:
        clear_query_cache()
        return search(idx, "python asyncio index", k=10, with_snippets=False)

    results = benchmark(run_query)
    assert len(results) == 10
