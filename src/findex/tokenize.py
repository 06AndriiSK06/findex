from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator

_TOKEN_RE = re.compile(r"\w+(?:[-'’]\w+)*", re.UNICODE)


def tokenize(text: str) -> Iterator[str]:
    normalized = unicodedata.normalize("NFC", text).casefold()
    for match in _TOKEN_RE.finditer(normalized):
        yield match.group(0)
