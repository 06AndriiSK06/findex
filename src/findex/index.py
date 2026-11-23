from __future__ import annotations

import logging
from array import array
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from findex.corpus import iter_documents
from findex.timing import timed
from findex.tokenize import tokenize

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    path: str
    title: str


class Index(Mapping[str, list[Posting]]):
    def __init__(
        self,
        postings: dict[str, list[Posting]],
        doc_lengths: dict[int, int],
        doc_meta: dict[int, DocMeta],
        *,
        with_positions: bool = False,
    ) -> None:
        self._postings = postings
        self._doc_lengths = doc_lengths
        self._doc_meta = doc_meta
        self.with_positions = with_positions
        self.closed = False

    @property
    def postings(self) -> dict[str, list[Posting]]:
        return self._postings

    @property
    def doc_lengths(self) -> dict[int, int]:
        return self._doc_lengths

    @property
    def doc_meta(self) -> dict[int, DocMeta]:
        return self._doc_meta

    @property
    def num_docs(self) -> int:
        return len(self._doc_lengths)

    @cached_property
    def avg_doc_length(self) -> float:
        if not self._doc_lengths:
            return 0.0
        return sum(self._doc_lengths.values()) / self.num_docs

    def doc_length(self, doc_id: int) -> int:
        return self._doc_lengths[doc_id]

    def df(self, term: str) -> int:
        plist = self._postings.get(term)
        return len(plist) if plist is not None else 0

    def close(self) -> None:
        self.closed = True

    def __len__(self) -> int:
        return len(self._postings)

    def __contains__(self, term: object) -> bool:
        return term in self._postings

    def __getitem__(self, term: str) -> list[Posting]:
        return self._postings[term]

    def __iter__(self) -> Iterator[str]:
        return iter(self._postings)

    def __repr__(self) -> str:
        return f"Index(terms={len(self):_}, docs={self.num_docs:_})"


def intersect_merge(a: Iterable[int], b: Iterable[int]) -> list[int]:
    list_a = list(a)
    list_b = list(b)
    result: list[int] = []
    i = 0
    j = 0
    while i < len(list_a) and j < len(list_b):
        if list_a[i] == list_b[j]:
            result.append(list_a[i])
            i += 1
            j += 1
        elif list_a[i] < list_b[j]:
            i += 1
        else:
            j += 1
    return result


def union_merge(a: Iterable[int], b: Iterable[int]) -> list[int]:
    list_a = list(a)
    list_b = list(b)
    result: list[int] = []
    i = 0
    j = 0
    while i < len(list_a) and j < len(list_b):
        if list_a[i] == list_b[j]:
            result.append(list_a[i])
            i += 1
            j += 1
        elif list_a[i] < list_b[j]:
            result.append(list_a[i])
            i += 1
        else:
            result.append(list_b[j])
            j += 1
    if i < len(list_a):
        result.extend(list_a[i:])
    if j < len(list_b):
        result.extend(list_b[j:])
    return result


@timed
def build_index(
    root: Path,
    *,
    limit: int | None = None,
    with_positions: bool = False,
    on_doc: Callable[[int], None] | None = None,
) -> Index:
    postings_map: dict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}

    for doc_id, doc in enumerate(iter_documents(root)):
        if limit is not None and doc_id >= limit:
            break

        title = doc.path.stem.replace("_", " ")
        doc_meta[doc_id] = DocMeta(doc_id=doc_id, path=str(doc.path), title=title)

        if with_positions:
            pos_map: dict[str, list[int]] = defaultdict(list)
            total_tokens = 0
            for pos, token in enumerate(tokenize(doc.text)):
                pos_map[token].append(pos)
                total_tokens += 1
            doc_lengths[doc_id] = total_tokens
            for term, positions in pos_map.items():
                postings_map[term].append(
                    Posting(doc_id=doc_id, tf=len(positions), positions=tuple(positions))
                )
        else:
            counts: Counter[str] = Counter()
            total_tokens = 0
            for token in tokenize(doc.text):
                counts[token] += 1
                total_tokens += 1
            doc_lengths[doc_id] = total_tokens
            for term, tf in counts.items():
                postings_map[term].append(Posting(doc_id=doc_id, tf=tf))

        if on_doc is not None:
            on_doc(doc_id + 1)

    log.info("Indexed %d documents (%d terms)", len(doc_meta), len(postings_map))
    return Index(
        postings=dict(postings_map),
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
        with_positions=with_positions,
    )


def pack_postings_arrays(plist: Iterable[Posting]) -> tuple[array[int], array[int]]:
    doc_ids = array("I")
    tfs = array("I")
    for p in plist:
        doc_ids.append(p.doc_id)
        tfs.append(p.tf)
    return doc_ids, tfs
