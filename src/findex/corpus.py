from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Document:
    path: Path
    text: str
    title: str | None = None
    url: str | None = None


def collect_document_paths(root: Path, *, limit: int | None = None) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Corpus path does not exist: {root}")
    files = [root] if root.is_file() else sorted(root.rglob("*.txt"))
    if limit is not None:
        return files[:limit]
    return files


def iter_documents(root: Path, *, limit: int | None = None) -> Iterator[Document]:
    if not root.exists():
        raise FileNotFoundError(f"Corpus path does not exist: {root}")

    if root.is_file() and root.suffix.lower() == ".jsonl":
        yielded = 0
        with root.open("r", encoding="utf-8") as fh:
            for line_no, raw_line in enumerate(fh, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("Skipping malformed JSONL line %d in %s", line_no, root)
                    continue
                text = str(record.get("text", ""))
                url = str(record.get("url", f"{root}#{line_no}"))
                title = str(record.get("title") or url)
                yield Document(path=Path(url), text=text, title=title, url=url)
                yielded += 1
                if limit is not None and yielded >= limit:
                    return
        return

    for path in collect_document_paths(root, limit=limit):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            log.warning("Skipping undecodable file: %s", path)
            continue
        yield Document(path=path, text=text, title=path.stem.replace("_", " "))
