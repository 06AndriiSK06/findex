from __future__ import annotations

import pytest

from findex.tokenize import tokenize


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Hello, World!", ["hello", "world"]),
        ("state-of-the-art utf-8", ["state-of-the-art", "utf-8"]),
        ("don't 'quoted' п'ять", ["don't", "quoted", "п'ять"]),
        ("Straße", ["strasse"]),
        ("café vs cafe\u0301", ["café", "vs", "café"]),
        ("Київ 2026 Python3.14", ["київ", "2026", "python3", "14"]),
        ("   \n\t\r  ", []),
        ("--- ''' !!!", []),
    ],
)
def test_tokenize_parametrized(raw: str, expected: list[str]) -> None:
    assert list(tokenize(raw)) == expected
