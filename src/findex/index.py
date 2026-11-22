from __future__ import annotations

import argparse
import functools
import pickle
import shutil
import time
import tracemalloc
from array import array
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from findex.corpus import iter_documents
from findex.timing import timed
from findex.tokenize import tokenize


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True)
class PlainPosting:
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

    @functools.cached_property
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


InvertedIndex = Index


@timed
def build_index(
    root: Path,
    *,
    limit: int | None = None,
    with_positions: bool = False,
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

    return Index(
        postings=dict(postings_map),
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
        with_positions=with_positions,
    )


def _build_plain_postings(root: Path, limit: int | None = None) -> dict[str, list[PlainPosting]]:
    postings_map: dict[str, list[PlainPosting]] = defaultdict(list)
    for doc_id, doc in enumerate(iter_documents(root)):
        if limit is not None and doc_id >= limit:
            break
        counts = Counter(tokenize(doc.text))
        for term, tf in counts.items():
            postings_map[term].append(PlainPosting(doc_id=doc_id, tf=tf))
    return dict(postings_map)


def _build_slotted_postings(root: Path, limit: int | None = None) -> dict[str, list[Posting]]:
    postings_map: dict[str, list[Posting]] = defaultdict(list)
    for doc_id, doc in enumerate(iter_documents(root)):
        if limit is not None and doc_id >= limit:
            break
        counts = Counter(tokenize(doc.text))
        for term, tf in counts.items():
            postings_map[term].append(Posting(doc_id=doc_id, tf=tf))
    return dict(postings_map)


def _build_array_postings(
    root: Path, limit: int | None = None
) -> dict[str, tuple[array[int], array[int]]]:
    raw_map: dict[str, tuple[array[int], array[int]]] = defaultdict(
        lambda: (array("I"), array("I"))
    )
    for doc_id, doc in enumerate(iter_documents(root)):
        if limit is not None and doc_id >= limit:
            break
        counts = Counter(tokenize(doc.text))
        for term, tf in counts.items():
            doc_ids, tfs = raw_map[term]
            doc_ids.append(doc_id)
            tfs.append(tf)
    return dict(raw_map)


def run_memory_benchmark(root: Path, out_dir: Path, limit: int | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, float, float, float]] = []

    tracemalloc.start()
    plain = _build_plain_postings(root, limit=limit)
    _, peak_plain = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    plain_path = out_dir / "bench_plain.pkl"
    plain_path.write_bytes(pickle.dumps(plain, protocol=pickle.HIGHEST_PROTOCOL))
    t0 = time.perf_counter()
    _ = pickle.loads(plain_path.read_bytes())
    load_plain = time.perf_counter() - t0
    results.append(
        (
            "list[Posting] (plain @dataclass)",
            peak_plain / (1024 * 1024),
            plain_path.stat().st_size / (1024 * 1024),
            load_plain,
        )
    )
    del plain

    tracemalloc.start()
    slotted = _build_slotted_postings(root, limit=limit)
    _, peak_slotted = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    slotted_path = out_dir / "bench_slotted.pkl"
    slotted_path.write_bytes(pickle.dumps(slotted, protocol=pickle.HIGHEST_PROTOCOL))
    t0 = time.perf_counter()
    _ = pickle.loads(slotted_path.read_bytes())
    load_slotted = time.perf_counter() - t0
    results.append(
        (
            "list[Posting] (slots=True)",
            peak_slotted / (1024 * 1024),
            slotted_path.stat().st_size / (1024 * 1024),
            load_slotted,
        )
    )
    del slotted

    tracemalloc.start()
    arr_postings = _build_array_postings(root, limit=limit)
    _, peak_arr = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    arr_path = out_dir / "bench_array.pkl"
    arr_path.write_bytes(pickle.dumps(arr_postings, protocol=pickle.HIGHEST_PROTOCOL))
    t0 = time.perf_counter()
    _ = pickle.loads(arr_path.read_bytes())
    load_arr = time.perf_counter() - t0
    results.append(
        (
            "array('I') pairs (doc_ids, tfs)",
            peak_arr / (1024 * 1024),
            arr_path.stat().st_size / (1024 * 1024),
            load_arr,
        )
    )
    del arr_postings

    print("\n--- Memory & Storage Table ---")
    print("| Postings representation | Peak memory (build) | Index file size | Load time |")
    print("| :--- | ---: | ---: | ---: |")
    for label, mem_mib, size_mib, load_s in results:
        print(f"| {label} | {mem_mib:.2f} MiB | {size_mib:.2f} MiB | {load_s:.4f} s |")

    for p in (plain_path, slotted_path, arr_path):
        p.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and persist inverted index.")
    parser.add_argument("corpus_dir", type=Path, help="Path to corpus directory")
    parser.add_argument("--out", type=Path, default=Path("index.bin"), help="Output index file")
    parser.add_argument(
        "--format",
        choices=("binary", "pickle"),
        default="binary",
        help="Serialization format (default: binary)",
    )
    parser.add_argument("--positions", action="store_true", help="Record token positions")
    parser.add_argument("--limit", type=int, default=None, help="Max documents to index")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run 3-way memory and storage benchmark",
    )
    args = parser.parse_args()

    if args.benchmark:
        run_memory_benchmark(args.corpus_dir, Path(".bench_tmp"), limit=args.limit)
        shutil.rmtree(".bench_tmp", ignore_errors=True)
        return

    from findex.store import save_array_binary, save_pickle

    tracemalloc.start()
    t0 = time.perf_counter()
    index = build_index(args.corpus_dir, limit=args.limit, with_positions=args.positions)
    build_elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t_save0 = time.perf_counter()
    if args.format == "pickle":
        save_pickle(index, args.out)
    else:
        save_array_binary(index, args.out)
    save_elapsed = time.perf_counter() - t_save0

    size_mib = args.out.stat().st_size / (1024 * 1024)
    total_postings = sum(len(plist) for plist in index.postings.values())

    print(f"Built {index!r}")
    print(f"Total postings:    {total_postings}")
    print(f"Positions stored:  {index.with_positions}")
    print(f"Build time:        {build_elapsed:.3f} s")
    print(f"Save time ({args.format}): {save_elapsed:.4f} s")
    print(f"Index file size:   {size_mib:.2f} MiB")
    print(f"Peak memory:       {peak_bytes / (1024 * 1024):.2f} MiB")


if __name__ == "__main__":
    main()