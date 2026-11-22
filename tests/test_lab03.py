import logging
from pathlib import Path

import pytest

from findex.index import Posting, build_index
from findex.query import And, Not, Or, Phrase, Term, parse_query
from findex.ranking import BM25, SearchResult, TfIdf
from findex.search import _cached_match_ids, search
from findex.store import open_index, save_array_binary


def test_index_dunders_and_properties(tmp_path: Path) -> None:
    (tmp_path / "d0.txt").write_text("python streams generators", encoding="utf-8")
    (tmp_path / "d1.txt").write_text("python dicts", encoding="utf-8")

    idx = build_index(tmp_path)
    assert len(idx) == 4
    assert "python" in idx
    assert "rust" not in idx
    assert list(idx) == ["python", "streams", "generators", "dicts"]
    assert idx.num_docs == 2
    assert idx.avg_doc_length == pytest.approx(2.5)
    assert idx.df("python") == 2
    assert idx.df("missing") == 0
    assert repr(idx) == "Index(terms=4, docs=2)"

    with pytest.raises(KeyError):
        _ = idx["missing"]


def test_open_index_context_manager_closes_on_exception(tmp_path: Path) -> None:
    (tmp_path / "d0.txt").write_text("hello world", encoding="utf-8")
    idx = build_index(tmp_path)
    bin_path = tmp_path / "idx.bin"
    save_array_binary(idx, bin_path)

    with pytest.raises(RuntimeError), open_index(bin_path) as loaded:
        assert not loaded.closed
        raise RuntimeError("boom")

    assert loaded.closed


def test_query_parser_trees_and_operators() -> None:
    assert parse_query("a OR b c") == Or(Term("a"), And(Term("b"), Term("c")))

    tree = parse_query('python AND (async OR await) NOT java "event loop"')
    expected = And(
        And(
            And(Term("python"), Or(Term("async"), Term("await"))),
            Not(Term("java")),
        ),
        Phrase(("event", "loop")),
    )
    assert tree == expected

    composed = Term("python") & (Term("async") | Term("await")) & ~Term("java")
    assert composed == And(
        And(Term("python"), Or(Term("async"), Term("await"))),
        Not(Term("java")),
    )


def test_phrase_query_evaluation(tmp_path: Path) -> None:
    (tmp_path / "d0.txt").write_text("async event loop in python", encoding="utf-8")
    (tmp_path / "d1.txt").write_text("event happens in a loop", encoding="utf-8")

    idx = build_index(tmp_path, with_positions=True)
    assert parse_query('"event loop"').evaluate(idx) == {0}
    assert parse_query('"event loop" OR loop').evaluate(idx) == {0, 1}


def test_ranking_sanity_checks_and_snippets(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    (tmp_path / "short.txt").write_text("rareterm commonword", encoding="utf-8")
    (tmp_path / "long.txt").write_text(
        "commonword " + " ".join(f"filler{i}" for i in range(60)),
        encoding="utf-8",
    )

    idx = build_index(tmp_path, with_positions=True)
    bm25 = BM25()
    tfidf = TfIdf()

    # 1. Rare term outranks common term
    res = search(idx, "rareterm OR commonword", scorer=bm25, k=2)
    assert res[0].title == "short"
    assert "[rareterm]" in res[0].snippet

    # 2. Saturation check
    s1 = bm25.score("commonword", Posting(doc_id=0, tf=1), idx)
    s19 = bm25.score("commonword", Posting(doc_id=0, tf=19), idx)
    s20 = bm25.score("commonword", Posting(doc_id=0, tf=20), idx)
    assert (s20 - s19) < 0.05 * s1

    # 3. Length normalization check (short doc > long doc for same tf=1)
    res_len = search(idx, "commonword", scorer=bm25, k=2)
    assert res_len[0].title == "short"
    assert res_len[1].title == "long"

    # 4. Swappable protocol & ordering
    res_tfidf = search(idx, "commonword", scorer=tfidf, k=2)
    assert len(res_tfidf) == 2
    assert min([SearchResult(0.1, 1, "b"), SearchResult(0.9, 0, "a")]).score == 0.1

    # 5. @timed and lru_cache hit logging
    _cached_match_ids.cache_clear()
    with caplog.at_level(logging.INFO):
        _ = search(idx, "rareterm", scorer=bm25, k=1)
        _ = search(idx, "rareterm", scorer=bm25, k=1)
    assert "[cache MISS]" in caplog.text
    assert "[cache HIT]" in caplog.text
    assert "[timed] search took" in caplog.text