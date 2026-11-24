from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Document:
    path: Path
    text: str


def collect_document_paths(root: Path, *, limit: int | None = None) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Corpus path does not exist: {root}")
    files = [root] if root.is_file() else sorted(root.rglob("*.txt"))
    if limit is not None:
        return files[:limit]
    return files


def iter_documents(root: Path) -> Iterator[Document]:
    for path in collect_document_paths(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            log.warning("Skipping undecodable file: %s", path)
            continue
        yield Document(path=path, text=text)
