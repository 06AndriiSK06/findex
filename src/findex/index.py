from __future__ import annotations

import logging
import time
from array import array
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, cast

import numpy as np
import numpy.typing as npt

from findex.corpus import Document, collect_document_paths, iter_documents
from findex.timing import timed
from findex.tokenize import tokenize

log = logging.getLogger(__name__)

ExecutorKind = Literal["serial", "threads", "processes"]
Int32Array = npt.NDArray[np.int32]


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


@dataclass(frozen=True, slots=True)
class PartialIndex:
    postings: dict[str, list[Posting]]
    doc_lengths: dict[int, int]
    doc_meta: dict[int, DocMeta]
    with_positions: bool = False


class Index(Mapping[str, list[Posting]]):
    def __init__(
        self,
        postings: dict[str, list[Posting]],
        doc_lengths: dict[int, int],
        doc_meta: dict[int, DocMeta],
        *,
        with_positions: bool = False,
        merge_seconds: float = 0.0,
        postings_doc_ids: dict[str, Int32Array] | None = None,
        postings_tfs: dict[str, Int32Array] | None = None,
        doc_lengths_arr: Int32Array | None = None,
    ) -> None:
        self._postings = postings
        self._doc_lengths = doc_lengths
        self._doc_meta = doc_meta
        self.with_positions = with_positions
        self.merge_seconds = merge_seconds
        self.closed = False

        if doc_lengths_arr is not None:
            self._doc_lengths_arr = doc_lengths_arr
        elif doc_lengths:
            max_id = max(doc_lengths.keys())
            arr = np.zeros(max_id + 1, dtype=np.int32)
            for did, dlen in doc_lengths.items():
                arr[did] = dlen
            self._doc_lengths_arr = arr
        else:
            self._doc_lengths_arr = np.empty(0, dtype=np.int32)

        if postings_doc_ids is not None and postings_tfs is not None:
            self._postings_doc_ids = postings_doc_ids
            self._postings_tfs = postings_tfs
        else:
            p_ids: dict[str, Int32Array] = {}
            p_tfs: dict[str, Int32Array] = {}
            for term, plist in postings.items():
                p_ids[term] = np.fromiter(
                    (p.doc_id for p in plist), dtype=np.int32, count=len(plist)
                )
                p_tfs[term] = np.fromiter((p.tf for p in plist), dtype=np.int32, count=len(plist))
            self._postings_doc_ids = p_ids
            self._postings_tfs = p_tfs

    @property
    def postings(self) -> dict[str, list[Posting]]:
        return self._postings

    @property
    def postings_doc_ids(self) -> dict[str, Int32Array]:
        return self._postings_doc_ids

    @property
    def postings_tfs(self) -> dict[str, Int32Array]:
        return self._postings_tfs

    @property
    def doc_lengths_arr(self) -> Int32Array:
        return self._doc_lengths_arr

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
        if self._doc_lengths_arr.size == 0 or self.num_docs == 0:
            return 0.0
        return float(self._doc_lengths_arr.sum()) / self.num_docs

    def doc_length(self, doc_id: int) -> int:
        return self._doc_lengths[doc_id]

    def df(self, term: str) -> int:
        arr = self._postings_doc_ids.get(term)
        return int(arr.size) if arr is not None else 0

    def close(self) -> None:
        self.closed = True

    def __len__(self) -> int:
        return len(self._postings_doc_ids)

    def __contains__(self, term: object) -> bool:
        return term in self._postings_doc_ids

    def __getitem__(self, term: str) -> list[Posting]:
        return self._postings[term]

    def __iter__(self) -> Iterator[str]:
        return iter(self._postings_doc_ids)

    def __repr__(self) -> str:
        return f"Index(terms={len(self):_}, docs={self.num_docs:_})"


def _to_int32_array(values: Iterable[int]) -> Int32Array:
    if isinstance(values, np.ndarray):
        return cast(Int32Array, np.fromiter(values.tolist(), dtype=np.int32))
    return np.fromiter(values, dtype=np.int32)


def intersect_merge(a: Iterable[int], b: Iterable[int]) -> list[int]:
    arr_a = _to_int32_array(a)
    arr_b = _to_int32_array(b)
    if arr_a.size == 0 or arr_b.size == 0:
        return []
    merged: Int32Array = np.intersect1d(arr_a, arr_b, assume_unique=True)
    return [int(x) for x in merged.tolist()]


def union_merge(a: Iterable[int], b: Iterable[int]) -> list[int]:
    arr_a = _to_int32_array(a)
    arr_b = _to_int32_array(b)
    if arr_a.size == 0:
        return [int(x) for x in arr_b.tolist()]
    if arr_b.size == 0:
        return [int(x) for x in arr_a.tolist()]
    merged: Int32Array = np.union1d(arr_a, arr_b)
    return [int(x) for x in merged.tolist()]


def _build_partial_from_docs(
    docs: Sequence[Document],
    *,
    start_doc_id: int = 0,
    with_positions: bool = False,
) -> PartialIndex:
    postings_map: dict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}

    for offset, doc in enumerate(docs):
        doc_id = start_doc_id + offset
        tokens = list(tokenize(doc.text))
        doc_lengths[doc_id] = len(tokens)
        title = doc.title if doc.title is not None else doc.path.stem.replace("_", " ")
        doc_meta[doc_id] = DocMeta(doc_id=doc_id, path=str(doc.path), title=title)

        if with_positions:
            pos_by_term: dict[str, list[int]] = defaultdict(list)
            for pos, tok in enumerate(tokens):
                pos_by_term[tok].append(pos)
            for tok, positions in pos_by_term.items():
                postings_map[tok].append(
                    Posting(doc_id=doc_id, tf=len(positions), positions=tuple(positions))
                )
        else:
            counts = Counter(tokens)
            for tok, tf in counts.items():
                postings_map[tok].append(Posting(doc_id=doc_id, tf=tf))

    return PartialIndex(
        postings=dict(postings_map),
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
        with_positions=with_positions,
    )


def build_partial(
    doc_paths: list[Path],
    *,
    start_doc_id: int = 0,
    with_positions: bool = False,
) -> PartialIndex:
    docs: list[Document] = []
    for path in doc_paths:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            log.warning("Skipping undecodable file: %s", path)
            continue
        docs.append(Document(path=path, text=text, title=path.stem.replace("_", " ")))

    return _build_partial_from_docs(
        docs,
        start_doc_id=start_doc_id,
        with_positions=with_positions,
    )


def merge(partials: Iterable[PartialIndex]) -> Index:
    t0 = time.perf_counter()
    partial_list = list(partials)
    if not partial_list:
        return Index(postings={}, doc_lengths={}, doc_meta={}, with_positions=False)

    with_positions = partial_list[0].with_positions
    merged_postings: dict[str, list[Posting]] = defaultdict(list)
    merged_lengths: dict[int, int] = {}
    merged_meta: dict[int, DocMeta] = {}

    for part in partial_list:
        merged_lengths.update(part.doc_lengths)
        merged_meta.update(part.doc_meta)
        for term, plist in part.postings.items():
            merged_postings[term].extend(plist)

    for plist in merged_postings.values():
        if len(plist) > 1:
            plist.sort(key=lambda p: p.doc_id)

    sorted_postings = dict(sorted(merged_postings.items()))
    sorted_lengths = dict(sorted(merged_lengths.items()))
    sorted_meta = dict(sorted(merged_meta.items()))

    p_ids: dict[str, Int32Array] = {}
    p_tfs: dict[str, Int32Array] = {}
    for term, plist in sorted_postings.items():
        p_ids[term] = np.fromiter((p.doc_id for p in plist), dtype=np.int32, count=len(plist))
        p_tfs[term] = np.fromiter((p.tf for p in plist), dtype=np.int32, count=len(plist))

    merge_elapsed = time.perf_counter() - t0

    return Index(
        postings=sorted_postings,
        doc_lengths=sorted_lengths,
        doc_meta=sorted_meta,
        with_positions=with_positions,
        merge_seconds=merge_elapsed,
        postings_doc_ids=p_ids,
        postings_tfs=p_tfs,
    )


def _chunk_paths(paths: list[Path], num_chunks: int) -> list[tuple[int, list[Path]]]:
    if not paths:
        return []
    n = max(1, min(num_chunks, len(paths)))
    base_size, remainder = divmod(len(paths), n)
    chunks: list[tuple[int, list[Path]]] = []
    offset = 0
    for i in range(n):
        size = base_size + (1 if i < remainder else 0)
        chunks.append((offset, paths[offset : offset + size]))
        offset += size
    return chunks


@timed
def build_index(
    root: Path,
    *,
    limit: int | None = None,
    with_positions: bool = False,
    workers: int = 1,
    executor: ExecutorKind = "processes",
    on_doc: Callable[[int], None] | None = None,
) -> Index:
    if workers < 1:
        raise ValueError("workers must be >= 1")

    if root.is_file() and root.suffix.lower() == ".jsonl":
        docs = list(iter_documents(root, limit=limit))
        if not docs:
            return Index(postings={}, doc_lengths={}, doc_meta={}, with_positions=with_positions)
        part = _build_partial_from_docs(docs, start_doc_id=0, with_positions=with_positions)
        if on_doc is not None:
            on_doc(len(part.doc_meta))
        return merge([part])

    paths = collect_document_paths(root, limit=limit)
    if not paths:
        return Index(postings={}, doc_lengths={}, doc_meta={}, with_positions=with_positions)

    if executor == "serial" or workers == 1:
        part = build_partial(paths, start_doc_id=0, with_positions=with_positions)
        if on_doc is not None:
            on_doc(len(part.doc_meta))
        idx = merge([part])
        log.info(
            "Indexed %d documents (%d terms) via serial in merge=%.4fs",
            idx.num_docs,
            len(idx),
            idx.merge_seconds,
        )
        return idx

    chunks = _chunk_paths(paths, workers)
    partials: list[PartialIndex] = []
    completed_docs = 0

    pool_cls = ThreadPoolExecutor if executor == "threads" else ProcessPoolExecutor
    with pool_cls(max_workers=workers) as pool:
        futures = [
            pool.submit(
                build_partial,
                chunk_paths,
                start_doc_id=start_id,
                with_positions=with_positions,
            )
            for start_id, chunk_paths in chunks
        ]
        for fut in as_completed(futures):
            part = fut.result()
            partials.append(part)
            completed_docs += len(part.doc_meta)
            if on_doc is not None:
                on_doc(completed_docs)

    idx = merge(partials)
    log.info(
        "Indexed %d documents (%d terms) via %s (workers=%d, merge=%.4fs)",
        idx.num_docs,
        len(idx),
        executor,
        workers,
        idx.merge_seconds,
    )
    return idx


def pack_postings_arrays(plist: Iterable[Posting]) -> tuple[array[int], array[int]]:
    doc_ids = array("I")
    tfs = array("I")
    for p in plist:
        doc_ids.append(p.doc_id)
        tfs.append(p.tf)
    return doc_ids, tfs
