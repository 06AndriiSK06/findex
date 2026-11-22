from pathlib import Path

import pytest

from findex.index import DocMeta, Posting, build_index
from findex.query import parse_query
from findex.store import load_array_binary, load_pickle, save_array_binary, save_pickle


def test_posting_and_docmeta_are_hashable() -> None:
    p1 = Posting(doc_id=1, tf=3)
    p2 = Posting(doc_id=1, tf=3)
    m1 = DocMeta(doc_id=1, path="a.txt", title="a")
    m2 = DocMeta(doc_id=1, path="a.txt", title="a")

    assert p1 == p2
    assert hash(p1) == hash(p2)
    assert len({p1, p2}) == 1
    assert len({m1, m2}) == 1

    with pytest.raises(AttributeError):
        p1.tf = 5  # type: ignore[misc]


def test_build_index_and_boolean_queries(tmp_path: Path) -> None:
    (tmp_path / "doc0.txt").write_text("Python generators stream data fast", encoding="utf-8")
    (tmp_path / "doc1.txt").write_text("Python dictionaries use hash tables", encoding="utf-8")
    (tmp_path / "doc2.txt").write_text("Inverted index maps terms to postings", encoding="utf-8")

    idx = build_index(tmp_path, with_positions=True)

    assert len(idx.doc_meta) == 3
    assert idx.doc_lengths[0] == 5
    assert idx.postings["python"] == [
        Posting(doc_id=0, tf=1, positions=(0,)),
        Posting(doc_id=1, tf=1, positions=(0,)),
    ]

    for query, expected in [
        ("python generators", [0]),
        ("generators OR index", [0, 2]),
        ("python NOT generators", [1]),
        ("NOT python", [2]),
    ]:
        assert sorted(parse_query(query).evaluate(idx)) == expected


def test_store_roundtrip_pickle_and_binary(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello world hello", encoding="utf-8")
    (tmp_path / "b.txt").write_text("world of python", encoding="utf-8")

    idx = build_index(tmp_path)

    pkl_path = tmp_path / "index.pkl"
    save_pickle(idx, pkl_path)
    loaded_pkl = load_pickle(pkl_path)
    assert loaded_pkl.postings == idx.postings
    assert loaded_pkl.doc_lengths == idx.doc_lengths
    assert loaded_pkl.doc_meta == idx.doc_meta

    bin_path = tmp_path / "index.bin"
    save_array_binary(idx, bin_path)
    loaded_bin = load_array_binary(bin_path)
    assert loaded_bin.postings == idx.postings
    assert loaded_bin.doc_lengths == idx.doc_lengths
    assert loaded_bin.doc_meta == idx.doc_meta