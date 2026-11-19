from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    path: Path
    text: str


def iter_documents(root: Path) -> Iterator[Document]:
    """Lazily yield Document objects from all .txt files under `root`."""
    if not root.exists():
        raise FileNotFoundError(f"Corpus path does not exist: {root}")

    for path in sorted(root.rglob("*.txt")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            logger.warning("Encoding issue in %s (%s); reading with replacement", path, exc)
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.error("Failed to read %s: %s", path, exc)
            continue

        yield Document(
            doc_id=path.relative_to(root).as_posix(),
            path=path,
            text=text,
        )