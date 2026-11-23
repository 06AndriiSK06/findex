from __future__ import annotations

import logging
from pathlib import Path

import pytest

from findex.index import Index, Posting, build_index
from findex.query import And, Not, Or, Phrase, QuerySyntaxError, Term, parse_query
from findex.ranking import BM25, SearchResult, TfIdf
from findex.search import clear_query_cache, search
from findex.store import open_index, save_array_binary


def test_index_dunders_and_properties(built_index: Index) -> None:
    assert len(built_index) > 10
    assert "python" in built_index
    assert "nonexistent_xyz" not in built_index
    assert built_index.num_docs == 5
    assert built_index.avg_doc_length > 0
    assert built_index.df("python") == 3
    assert built_index.df("missing") == 0
    assert "Index(terms=" in repr(built_index)

    with pytest.raises(KeyError):
        _ = built_index["missing"]


def test_open_index_context_manager_closes_on_exception(saved_index_path: Path) -> None:
    with pytest.raises(RuntimeError), open_index(saved_index_path) as loaded:
        assert not loaded.closed
        raise RuntimeError("boom")

    assert loaded.closed


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a OR b c", Or(Term("a"), And(Term("b"), Term("c")))),
        ("NOT a", Not(Term("a"))),
        ('"event loop"', Phrase(("event", "loop"))),
        (
            'python AND (async OR await) NOT java "event loop"',
            And(
                And(
                    And(Term("python"), Or(Term("async"), Term("await"))),
                    Not(Term("java")),
                ),
                Phrase(("event", "loop")),
            ),
        ),
    ],
)
def test_query_parser_trees(raw: str, expected: object) -> None:
    assert parse_query(raw) == expected


@pytest.mark.parametrize("bad_query", ["", "   ", "(a AND b", "a AND", '""'])
def test_query_parser_invalid_syntax(bad_query: str) -> None:
    with pytest.raises(QuerySyntaxError):
        parse_query(bad_query)


def test_query_node_operator_overloading() -> None:
    composed = Term("python") & (Term("async") | Term("await")) & ~Term("java")
    assert composed == And(
        And(Term("python"), Or(Term("async"), Term("await"))),
        Not(Term("java")),
    )


def test_phrase_query_evaluation(built_index: Index) -> None:
    assert parse_query('"event loop"').evaluate(built_index) == {0}
    assert parse_query('"gimbal lock" OR generators').evaluate(built_index) == {2, 4}


def test_phrase_query_without_positions_raises(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("event loop", encoding="utf-8")
    idx = build_index(tmp_path, with_positions=False)
    with pytest.raises(QuerySyntaxError):
        parse_query('"event loop"').evaluate(idx)


def test_ranking_sanity_checks_and_snippets(
    built_index: Index,
    caplog: pytest.LogCaptureFixture,
) -> None:
    bm25 = BM25()
    tfidf = TfIdf()

    # 1. Rare term ('quaternion') outranks common term ('python')
    res = search(built_index, "quaternion OR python", scorer=bm25, k=2)
    assert res[0].title == "rare topic"
    assert "[Quaternion]" in res[0].snippet

    # 2. Saturation check (20th repetition adds almost nothing)
    s1 = bm25.score("python", Posting(doc_id=0, tf=1), built_index)
    s19 = bm25.score("python", Posting(doc_id=0, tf=19), built_index)
    s20 = bm25.score("python", Posting(doc_id=0, tf=20), built_index)
    assert (s20 - s19) < 0.05 * s1

    # 3. Length normalization check (short doc > long doc for same tf=1)
    res_len = search(built_index, "asyncio", scorer=bm25, k=2)
    assert res_len[0].title == "async intro"
    assert res_len[1].title == "long manual"

    # 4. Swappable protocol & ordering
    res_tfidf = search(built_index, "asyncio", scorer=tfidf, k=2)
    assert len(res_tfidf) == 2
    assert min([SearchResult(0.1, 1, "b"), SearchResult(0.9, 0, "a")]).score == 0.1

    # 5. @timed and lru_cache hit logging
    clear_query_cache()
    with caplog.at_level(logging.DEBUG):
        _ = search(built_index, "quaternion", scorer=bm25, k=1)
        _ = search(built_index, "quaternion", scorer=bm25, k=1)
    assert "[cache MISS]" in caplog.text
    assert "[cache HIT]" in caplog.text
    assert "[timed] search took" in caplog.text


@pytest.mark.slow
def test_bulk_corpus_build_and_persistence_benchmark(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "bulk"
    corpus_dir.mkdir()
    for i in range(400):
        (corpus_dir / f"doc_{i:04d}.txt").write_text(
            f"Document {i} discusses python indexing, ranking, and search algorithms.",
            encoding="utf-8",
        )
    idx = build_index(corpus_dir, with_positions=True)
    out = tmp_path / "bulk.bin"
    save_array_binary(idx, out)
    with open_index(out) as loaded:
        assert loaded.num_docs == 400
