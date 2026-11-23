from __future__ import annotations

from pathlib import Path

import pytest

from findex.index import Index, build_index
from findex.store import save_array_binary


@pytest.fixture(scope="session")
def mini_corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("corpus")
    docs = {
        "async_intro.txt": (
            "Python asyncio uses an event loop to schedule coroutines and tasks. "
            "Async and await make concurrent code readable."
        ),
        "generators.txt": (
            "Python generators yield items lazily one at a time. "
            "A generator saves memory on large streams."
        ),
        "dicts_hashing.txt": (
            "Python dictionaries use open addressing hash tables. "
            "Hashable keys require consistent equality and hash."
        ),
        "long_manual.txt": (
            "asyncio " + " ".join(f"section_{i} documentation text" for i in range(120))
        ),
        "rare_topic.txt": ("Quaternion rotations avoid gimbal lock in three-dimensional graphics."),
    }
    for name, content in docs.items():
        (root / name).write_text(content, encoding="utf-8")
    return root


@pytest.fixture(scope="session")
def built_index(mini_corpus_dir: Path) -> Index:
    return build_index(mini_corpus_dir, with_positions=True)


@pytest.fixture(scope="session")
def saved_index_path(built_index: Index, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("idx") / "test_index.bin"
    save_array_binary(built_index, out)
    return out
