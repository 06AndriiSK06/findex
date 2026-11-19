from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator

# Tokenization rules:
# - Unicode NFC normalization is applied prior to casefolding.
# - str.casefold() is used for locale-independent caseless matching.
# - Internal apostrophes (' and ’) are kept (e.g., don't, п'ять), leading/trailing stripped.
# - Internal hyphens (-) are kept as single compound tokens (e.g., state-of-the-art).
# - Digits are retained inside tokens (e.g., 2026, utf-8, mp3).

_TOKEN_RE = re.compile(r"\w+(?:['’\-]\w+)*", re.UNICODE)


def tokenize(text: str) -> Iterator[str]:
    """Stream normalized tokens from `text` without materializing a list."""
    normalized = unicodedata.normalize("NFC", text).casefold()
    for match in _TOKEN_RE.finditer(normalized):
        yield match.group(0)