from __future__ import annotations

import json
import logging
import pickle
import struct
from array import array
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TypedDict

from findex.index import DocMeta, Index, Posting
from findex.timing import timed

log = logging.getLogger(__name__)

MAGIC_V1 = b"FIDX0001"
MAGIC_V2 = b"FIDX0002"


class _IndexHeader(TypedDict):
    with_positions: bool
    doc_lengths: dict[str, int]
    doc_meta: dict[str, list[str]]
    terms: dict[str, list[int]]


def save_pickle(index: Index, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path) -> Index:
    # SECURITY WARNING: pickle.load executes arbitrary Python bytecode via __reduce__.
    # Never call load_pickle on untrusted or network-downloaded files.
    if not path.exists():
        raise FileNotFoundError(f"Index file not found: {path}")
    with path.open("rb") as f:
        obj: object = pickle.load(f)
    if not isinstance(obj, Index):
        raise TypeError("Invalid pickle payload: expected Index")
    return obj


def save_array_binary(index: Index, path: Path) -> None:
    terms_meta: dict[str, list[int]] = {}
    payload_chunks: list[bytes] = []
    offset = 0

    for term, plist in index.postings.items():
        count = len(plist)
        doc_ids = array("I", (p.doc_id for p in plist))
        tfs = array("I", (p.tf for p in plist))
        if index.with_positions:
            flat_pos = array("I")
            for p in plist:
                flat_pos.extend(p.positions)
            raw = doc_ids.tobytes() + tfs.tobytes() + flat_pos.tobytes()
        else:
            raw = doc_ids.tobytes() + tfs.tobytes()

        payload_chunks.append(raw)
        terms_meta[term] = [offset, count]
        offset += len(raw)

    header: _IndexHeader = {
        "with_positions": index.with_positions,
        "doc_lengths": {str(k): v for k, v in index.doc_lengths.items()},
        "doc_meta": {str(k): [m.path, m.title] for k, m in index.doc_meta.items()},
        "terms": terms_meta,
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")

    with path.open("wb") as f:
        f.write(MAGIC_V2)
        f.write(struct.pack("<I", len(header_bytes)))
        f.write(header_bytes)
        for chunk in payload_chunks:
            f.write(chunk)


def load_array_binary(path: Path) -> Index:
    if not path.exists():
        raise FileNotFoundError(f"Index file not found: {path}")
    data = path.read_bytes()
    if len(data) < 12 or data[:8] not in {MAGIC_V1, MAGIC_V2}:
        raise ValueError(f"Invalid binary index header in {path}")

    header_len: int = struct.unpack("<I", data[8:12])[0]
    header_end = 12 + header_len
    header: _IndexHeader = json.loads(data[12:header_end].decode("utf-8"))
    payload = memoryview(data)[header_end:]

    with_positions = bool(header["with_positions"])
    doc_lengths = {int(k): int(v) for k, v in header["doc_lengths"].items()}
    doc_meta = {
        int(k): DocMeta(doc_id=int(k), path=v[0], title=v[1]) for k, v in header["doc_meta"].items()
    }

    postings: dict[str, list[Posting]] = {}
    for term, meta_pair in header["terms"].items():
        offset, count = meta_pair[0], meta_pair[1]
        half = count * 4
        doc_ids = array("I")
        tfs = array("I")
        doc_ids.frombytes(payload[offset : offset + half])
        tfs.frombytes(payload[offset + half : offset + 2 * half])

        if with_positions:
            total_pos = sum(tfs)
            pos_start = offset + 2 * half
            flat_pos = array("I")
            flat_pos.frombytes(payload[pos_start : pos_start + total_pos * 4])
            plist: list[Posting] = []
            cursor = 0
            for d, tf in zip(doc_ids, tfs, strict=True):
                plist.append(
                    Posting(
                        doc_id=d,
                        tf=tf,
                        positions=tuple(flat_pos[cursor : cursor + tf]),
                    )
                )
                cursor += tf
            postings[term] = plist
        else:
            postings[term] = [Posting(doc_id=d, tf=t) for d, t in zip(doc_ids, tfs, strict=True)]

    return Index(
        postings=postings,
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
        with_positions=with_positions,
    )


@timed
def load_index(path: Path) -> Index:
    if not path.exists():
        raise FileNotFoundError(f"Index file not found: {path}")
    with path.open("rb") as f:
        head = f.read(8)
    log.info("Loading index from %s", path)
    if head in {MAGIC_V1, MAGIC_V2}:
        return load_array_binary(path)
    return load_pickle(path)


@contextmanager
def open_index(path: Path) -> Generator[Index, None, None]:
    index = load_index(path)
    try:
        yield index
    finally:
        index.close()
