# findex — Search Engine (Lab 01: Streams, Not Lists)

A streaming text-processing intake pipeline built with Python generators, `pathlib`, `re`, and Unicode normalization.

## Corpus

The corpus consists of **1,015 plain-text documentation files** from the official Python 3.12 and 3.13 documentation archives (`python-3.12-docs-text.zip` and `python-3.13-docs-text.zip` from `docs.python.org`), totaling ~3.52M tokens and 55,764 unique terms.

To reproduce the corpus locally (stored in `data/`, excluded via `.gitignore`):

```powershell
uv run python scripts/fetch_corpus.py
```
