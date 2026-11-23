# findex — Search Engine

[![CI](https://github.com/06AndriiSK06/findex/actions/workflows/ci.yml/badge.svg)](https://github.com/06AndriiSK06/findex/actions/workflows/ci.yml)

An incremental full-text search engine built from scratch in Python using streaming generators, a Pythonic `Index` data model, recursive-descent query parsing, TF-IDF / BM25 ranking, and a `typer` + `rich` CLI.

## Installation & CLI Usage

### Install from Wheel or Local Repository

Install the packaged CLI tool directly into your user `PATH` using `uv`:

```powershell
# From a built wheel or GitHub Release asset:
uv tool install dist/findex-0.4.0-py3-none-any.whl

# Or directly from the repository root:
uv tool install .
```

Run without installing via `uvx`:

```powershell
uvx --from dist/findex-0.4.0-py3-none-any.whl findex --help
```

### CLI Commands

```powershell
# Build a binary index with positional postings and a rich progress bar
findex index data/ --out index_pos.bin --positions

# Ranked search with a rich results table and highlighted snippets
findex search index_pos.bin 'python AND (async OR await) NOT java \"event loop\"' --k 5

# JSON lines output to stdout for piping (diagnostics go to stderr via -v / -vv)
findex -vv search index_pos.bin "generator AND iterator" --scorer tfidf --json

# Corpus or index statistics
findex stats index_pos.bin --top 10
```

Example output of `findex search`:

```text
                    Search Results (BM25) — 'python AND (async OR await) NOT java "event loop"'
┏━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ # ┃   Score ┃ Title                                         ┃ Snippet                                            ┃
┡━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1 │ 18.6475 │ py3.13 howto a-conceptual-overview-of-asyncio │ A Conceptual Overview of "asyncio"                 │
│   │         │                                               │ ********************************** This HOWTO      │
│   │         │                                               │ article seeks to help you...                       │
│ 2 │ 18.5059 │ py3.13 library asyncio-dev                    │ Developing with asyncio ***********************    │
│   │         │                                               │ Asynchronous programming is different from         │
│   │         │                                               │ class...                                           │
│ 3 │ 18.4173 │ py3.12 library asyncio-task                   │ Coroutines and Tasks ******************** This     │
│   │         │                                               │ section outlines high-level asyncio APIs to work   │
│   │         │                                               │ with coroutines and Tasks. * Coroutines *          │
│   │         │                                               │ Awaitables * Crea...                               │
│ 4 │ 18.3875 │ py3.13 library asyncio-task                   │ Coroutines and tasks ******************** This     │
│   │         │                                               │ section outlines high-level asyncio APIs to work   │
│   │         │                                               │ with coroutines and Tasks. * Coroutines *          │
│   │         │                                               │ Awaitables * Crea...                               │
│ 5 │ 17.8322 │ py3.12 library asyncio-eventloop              │ Event Loop ********** **Source code:**             │
│   │         │                                               │ Lib/asyncio/events.py, Lib/asyncio/base_e...       │
└───┴─────────┴───────────────────────────────────────────────┴────────────────────────────────────────────────────┘
```

## Corpus

The corpus consists of **1,015 plain-text documentation files** from the official Python 3.12 and 3.13 documentation archives (`python-3.12-docs-text.zip` and `python-3.13-docs-text.zip` from `docs.python.org`), totaling **3,521,578 tokens**, **55,764 unique terms**, and **645,774 postings**.

To reproduce the corpus locally (stored in `data/`, excluded via `.gitignore`):

```powershell
uv run python scripts/fetch_corpus.py
```

---

## Lab 04: Typing, Testing, Packaging & CLI

### Static Type Checking (`pyright` Strict Mode)

- Configured `[tool.pyright] typeCheckingMode = "strict"` in `pyproject.toml` across `src/` and `tests/` (**0 errors, 0 warnings**).
- `Scorer` is defined as a structural `typing.Protocol`, verified statically against both `BM25` and `TfIdf` without inheritance.
- Generator functions (`iter_documents`, `tokenize`) return `Iterator[...]`, merge helpers accept `Iterable[...]`, and scorer options are constrained via `Literal["bm25", "tfidf"]`.

### Test Suite & Code Coverage (`94%`)

The test suite contains **39 pytest tests**, including parametrized Unicode tokenizer cases, session-scoped fixtures in `tests/conftest.py`, AST parser tree-equality tests, BM25/TF-IDF ranking sanity checks, `CliRunner` integration tests, `@pytest.mark.slow` bulk benchmarks, and **4 Hypothesis property-based tests** (`tokenize` idempotence, `merge` vs. `set` oracle, `load(save(index))` round-trip & sorted postings invariant, and query tree equivalence).

```text
Name                     Stmts   Miss  Cover   Missing
------------------------------------------------------
src\findex\__init__.py       2      1    50%   2
src\findex\cli.py          113      4    96%   69, 173-174, 240
src\findex\corpus.py        21      3    86%   24-26
src\findex\index.py        137      8    94%   66, 151, 190-195
src\findex\query.py        185     12    94%   16, 19, 38, 51, 53, 61, 75, 121, 153, 193, 236, 246
src\findex\ranking.py       73      8    89%   23, 37, 49, 61, 72, 76, 86, 92
src\findex\search.py        50      1    98%   59
src\findex\stats.py         27      1    96%   34
src\findex\store.py        106      5    95%   38, 42, 85, 88, 147
src\findex\timing.py        15      0   100%
src\findex\tokenize.py       9      0   100%
------------------------------------------------------
TOTAL                      738     43    94%
```

**Intentionally uncovered branches (6%):**

- `if __name__ == "__main__":` CLI entrypoint guards (`cli.py:240`).
- Defensive fallback branches for empty corpora/missing files on disk (`avg_doc_length == 0.0` in `index.py:66`, missing file guard in `extract_snippet` at `ranking.py:72`, `UnicodeDecodeError` skip in `corpus.py:24-26`).
- Abstract `NotImplementedError` stubs on the base `QueryNode` class (`query.py:16, 19`).

---

## Lab 01: Streams, Not Lists

### Tokenization Rules

1. **Unicode Normalization:** Input text is normalized using `unicodedata.normalize("NFC", text)` so composed (`é`) and decomposed (`e\u0301`) forms map to identical code points.
2. **Case Folding:** `str.casefold()` is applied for aggressive, locale-independent caseless matching (e.g., `Straße` -> `strasse`).
3. **Apostrophes (`'` and `’`):** Retained inside words (e.g., `don't`, `п'ять`), stripped when leading or trailing.
4. **Hyphens (`-`):** Compound hyphenated words are kept as single tokens (e.g., `state-of-the-art`, `utf-8`).
5. **Digits:** Retained inside tokens (`2026`, `mp3`, `3`), matched via Unicode-aware `\w`.

### Benchmark: Eager vs. Lazy Pipeline

Measured on the full 1,015-document corpus (3,521,578 tokens) using `tracemalloc` and `time.perf_counter()`:

| Version           | Documents | Peak memory | Elapsed |
| :---------------- | --------: | ----------: | ------: |
| eager (lists)     |      1015 |  243.49 MiB | 6.506 s |
| lazy (generators) |      1015 |   36.45 MiB | 7.561 s |

---

## Lab 02: Inverted Index, Hashing & Memory

### Memory & Storage Study (Postings Representations)

Measured across all 1,015 documents (55,764 terms, 645,774 postings) under `tracemalloc`:

| Postings representation               | Peak memory (build) | Index file size | Load time |
| :------------------------------------ | ------------------: | --------------: | --------: |
| `list[Posting]` (plain `@dataclass`)  |          101.81 MiB |       14.33 MiB |  0.4162 s |
| `list[Posting]` (`slots=True`)        |           78.25 MiB |       10.63 MiB |  2.7764 s |
| `array('I')` pairs (`doc_ids`, `tfs`) |           53.74 MiB |        7.42 MiB |  0.1105 s |

> **Security Note on `pickle`:** `pickle.load()` is a stack-based virtual machine that can instantiate arbitrary classes and execute code via `__reduce__` during deserialization. It is safe only for locally generated index files and must never be called on untrusted or network-downloaded files.

### Boolean Query Benchmark: `merge` vs. `set`

| Query type            | Terms                          | `merge` (two-pointer) | `set` (hash set) |
| :-------------------- | :----------------------------- | --------------------: | ---------------: |
| 2 most common (`AND`) | `the in`                       |             839.48 ms |        305.72 ms |
| 2 most common (`OR`)  | `the OR in`                    |             851.87 ms |        298.76 ms |
| 2 rarest (`AND`)      | `deconstruct provisionally`    |              66.89 ms |         89.74 ms |
| 2 rarest (`OR`)       | `deconstruct OR provisionally` |              72.77 ms |         92.03 ms |

---

## Lab 03: Ranking, Object Model, Decorators & Query Language

### Three Ranking Sanity Checks (BM25 on Corpus)

1. **Rare term outranks common term** (`mature` with $df=2$ vs. `the` with $df=1011$):
   - `doc_id=502` (`py3.13 bugs`, contains `mature`): **score = `9.5264`**
   - `doc_id=1` (`py3.12 bugs`, contains `mature`): **score = `9.4354`**
   - `doc_id=903` (`py3.13 library tkinter`, contains only `the`): **score = `0.0111`**
2. **Term-frequency saturation** (`generator` in `doc_id=0`):
   - `tf = 1`: **`3.2573`**, `tf = 19`: **`4.5572`**, `tf = 20`: **`4.5622`** (marginal gain of the 20th occurrence: **`+0.005057`**)
3. **Document-length normalization** (`asyncio`, both with `tf = 1`):
   - Short document `[151]` (`len = 67` tokens): **score = `3.8160`** vs. Long document `[811]` (`len = 16,718` tokens): **score = `0.7843`**

### Precision@5 Evaluation (10 Hand-Labeled Queries: TF-IDF vs. BM25)

| Query         | TF-IDF P@5 | BM25 P@5 | Top-1 BM25 Document                             |
| :------------ | ---------: | -------: | :---------------------------------------------- |
| `asyncio`     |       0.60 |     1.00 | `py3.12 library asyncio-task`                   |
| `sqlite3`     |       0.40 |     0.40 | `py3.12 library sqlite3`                        |
| `itertools`   |       0.20 |     0.40 | `py3.12 howto functional`                       |
| `pathlib`     |       0.40 |     0.40 | `py3.13 library pathlib`                        |
| `subprocess`  |       0.40 |     0.80 | `py3.12 library asyncio-subprocess`             |
| `collections` |       0.00 |     0.40 | `py3.12 library collections.abc`                |
| `threading`   |       0.40 |     0.60 | `py3.12 deprecations pending-removal-in-future` |
| `unittest`    |       0.60 |     0.40 | `py3.12 library unittest`                       |
| `json`        |       0.40 |     0.40 | `py3.13 library json`                           |
| `argparse`    |       0.60 |     1.00 | `py3.13 howto argparse`                         |
| **Mean P@5**  |   **0.40** | **0.58** | —                                               |
