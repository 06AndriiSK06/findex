from __future__ import annotations

import logging
import time
from array import array
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal

from findex.corpus import collect_document_paths
from findex.timing import timed
from findex.tokenize import tokenize

log = logging.getLogger(__name__)

ExecutorKind = Literal["serial", "threads", "processes"]


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
    ) -> None:
        self._postings = postings
        self._doc_lengths = doc_lengths
        self._doc_meta = doc_meta
        self.with_positions = with_positions
        self.merge_seconds = merge_seconds
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


def build_partial(
    doc_paths: list[Path],
    *,
    start_doc_id: int = 0,
    with_positions: bool = False,
) -> PartialIndex:
    postings_map: dict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}

    next_id = start_doc_id
    for path in doc_paths:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            log.warning("Skipping undecodable file: %s", path)
            continue

        doc_id = next_id
        next_id += 1
        title = path.stem.replace("_", " ")
        doc_meta[doc_id] = DocMeta(doc_id=doc_id, path=str(path), title=title)

        if with_positions:
            pos_map: dict[str, list[int]] = defaultdict(list)
            total_tokens = 0
            for pos, token in enumerate(tokenize(text)):
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
            for token in tokenize(text):
                counts[token] += 1
                total_tokens += 1
            doc_lengths[doc_id] = total_tokens
            for term, tf in counts.items():
                postings_map[term].append(Posting(doc_id=doc_id, tf=tf))

    return PartialIndex(
        postings=dict(postings_map),
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
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
    merge_elapsed = time.perf_counter() - t0

    return Index(
        postings=sorted_postings,
        doc_lengths=sorted_lengths,
        doc_meta=sorted_meta,
        with_positions=with_positions,
        merge_seconds=merge_elapsed,
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
