from __future__ import annotations

import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from findex.index import build_index, intersect_merge, union_merge
from findex.query import parse_query
from findex.store import load_array_binary, save_array_binary
from findex.tokenize import tokenize


@given(st.text())
def test_tokenize_never_yields_empty_and_is_idempotent(text: str) -> None:
    tokens = list(tokenize(text))
    assert all(tok != "" for tok in tokens)
    assert list(tokenize(" ".join(tokens))) == tokens


@given(
    st.lists(st.integers(min_value=0, max_value=500), unique=True),
    st.lists(st.integers(min_value=0, max_value=500), unique=True),
)
def test_merge_matches_set_oracle(a: list[int], b: list[int]) -> None:
    sa = sorted(a)
    sb = sorted(b)
    assert intersect_merge(sa, sb) == sorted(set(sa) & set(sb))
    assert union_merge(sa, sb) == sorted(set(sa) | set(sb))


@given(
    st.lists(
        st.text(alphabet="abcde ", min_size=1, max_size=40),
        min_size=1,
        max_size=6,
    )
)
@settings(max_examples=40)
def test_index_roundtrip_and_sorted_postings(docs: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for i, content in enumerate(docs):
            (root / f"d{i}.txt").write_text(content, encoding="utf-8")

        idx = build_index(root, with_positions=True)
        for plist in idx.postings.values():
            doc_ids = [p.doc_id for p in plist]
            assert doc_ids == sorted(doc_ids)

        bin_path = root / "roundtrip.bin"
        save_array_binary(idx, bin_path)
        reloaded = load_array_binary(bin_path)
        assert reloaded.postings == idx.postings
        assert reloaded.doc_lengths == idx.doc_lengths
        assert reloaded.doc_meta == idx.doc_meta


@given(st.lists(st.sampled_from(["alpha", "beta", "gamma", "delta"]), min_size=1, max_size=5))
def test_query_tree_equivalence(terms: list[str]) -> None:
    q = " AND ".join(terms)
    t1 = parse_query(q)
    t2 = parse_query(" ".join(terms))
    assert t1 == t2
