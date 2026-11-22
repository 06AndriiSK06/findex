from __future__ import annotations

import json
import pickle
import struct
from array import array
from pathlib import Path

from findex.index import DocMeta, InvertedIndex, Posting

MAGIC = b"FIDX0001"


def save_pickle(index: InvertedIndex, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path) -> InvertedIndex:
    # SECURITY WARNING: pickle.load executes arbitrary Python bytecode embedded in the
    # stream during unpickling (__reduce__). Never call load_pickle on untrusted or
    # network-downloaded files; only load index files generated locally.
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, InvertedIndex):
        raise TypeError("Invalid pickle payload: expected InvertedIndex")
    return obj


def save_array_binary(index: InvertedIndex, path: Path) -> None:
    terms_meta: dict[str, list[int]] = {}
    payload_chunks: list[bytes] = []
    offset = 0

    for term, plist in index.postings.items():
        count = len(plist)
        doc_ids = array("I", (p.doc_id for p in plist))
        tfs = array("I", (p.tf for p in plist))
        raw = doc_ids.tobytes() + tfs.tobytes()
        payload_chunks.append(raw)
        terms_meta[term] = [offset, count]
        offset += len(raw)

    header = {
        "with_positions": False,
        "doc_lengths": {str(k): v for k, v in index.doc_lengths.items()},
        "doc_meta": {
            str(k): [m.path, m.title] for k, m in index.doc_meta.items()
        },
        "terms": terms_meta,
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")

    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(header_bytes)))
        f.write(header_bytes)
        for chunk in payload_chunks:
            f.write(chunk)


def load_array_binary(path: Path) -> InvertedIndex:
    data = path.read_bytes()
    if data[:8] != MAGIC:
        raise ValueError("Invalid binary index magic header")

    (header_len,) = struct.unpack("<I", data[8:12])
    header_end = 12 + header_len
    header = json.loads(data[12:header_end].decode("utf-8"))
    payload = memoryview(data)[header_end:]

    doc_lengths = {int(k): int(v) for k, v in header["doc_lengths"].items()}
    doc_meta = {
        int(k): DocMeta(doc_id=int(k), path=v[0], title=v[1])
        for k, v in header["doc_meta"].items()
    }

    postings: dict[str, list[Posting]] = {}
    for term, (offset, count) in header["terms"].items():
        half = count * 4
        doc_ids = array("I")
        tfs = array("I")
        doc_ids.frombytes(payload[offset : offset + half])
        tfs.frombytes(payload[offset + half : offset + 2 * half])
        postings[term] = [Posting(doc_id=d, tf=t) for d, t in zip(doc_ids, tfs, strict=True)]

    return InvertedIndex(
        postings=postings,
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
        with_positions=bool(header.get("with_positions", False)),
    )


def load_index(path: Path) -> InvertedIndex:
    with path.open("rb") as f:
        head = f.read(8)
    if head == MAGIC:
        return load_array_binary(path)
    return load_pickle(path)