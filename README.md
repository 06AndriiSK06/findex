# findex — Search Engine (Lab 01: Streams, Not Lists)

A streaming text-processing intake pipeline built with Python generators, `pathlib`, `re`, and Unicode normalization.

## Corpus

The corpus consists of **1,015 plain-text documentation files** from the official Python 3.12 and 3.13 documentation archives (`python-3.12-docs-text.zip` and `python-3.13-docs-text.zip` from `docs.python.org`), totaling ~3.52M tokens and 55,764 unique terms.

To reproduce the corpus locally (stored in `data/`, excluded via `.gitignore`):

```powershell
uv run python scripts/fetch_corpus.py
```

## Setup & Usage

Install dependencies and run tests/linter:

```powershell
uv sync
uv run pytest
uv run ruff check .
```

Run corpus statistics on the full dataset or a limited slice:

```powershell
uv run python -m findex.stats data/
uv run python -m findex.stats data/ --limit 100
uv run python -m findex.stats data/ --benchmark
```

## Tokenization Rules

1. **Unicode Normalization:** Input text is normalized using `unicodedata.normalize("NFC", text)` so composed (`é`) and decomposed (`e\u0301`) forms map to identical code points.
2. **Case Folding:** `str.casefold()` is applied for aggressive, locale-independent caseless matching (e.g., `Straße` -> `strasse`).
3. **Apostrophes (`'` and `’`):** Retained inside words (e.g., `don't`, `п'ять`), stripped when leading or trailing.
4. **Hyphens (`-`):** Compound hyphenated words are kept as single tokens (e.g., `state-of-the-art`, `utf-8`).
5. **Digits:** Retained inside tokens (`2026`, `mp3`, `3`), matched via Unicode-aware `\w`.

## Benchmark: Eager vs. Lazy Pipeline

Measured on the full 1,015-document corpus (3,521,578 tokens) using `tracemalloc` and `time.perf_counter()`:

| Version           | Documents | Peak memory | Elapsed |
| :---------------- | --------: | ----------: | ------: |
| eager (lists)     |      1015 |  243.49 MiB | 6.506 s |
| lazy (generators) |      1015 |   36.45 MiB | 7.561 s |

In the **eager** implementation, peak memory spikes to **243.49 MiB** because the pipeline materializes three large collections in RAM simultaneously: the full list of `Document` objects containing every raw file text, a nested `list[list[str]]` of per-document token lists, and a flattened `list[str]` of all 3.52 million token strings (plus Python list pointer overhead). In contrast, the **lazy** generator pipeline peaks at **36.45 MiB** because documents and tokens are streamed one at a time and immediately discarded after updating the `Counter`; memory usage is bounded solely by the single largest document in flight and the resulting 55,764-term vocabulary table.
