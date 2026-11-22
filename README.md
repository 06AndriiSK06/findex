# findex — Search Engine

An incremental full-text search engine built from scratch in Python using streaming generators, custom inverted indexes, and compact binary storage.

## Corpus

The corpus consists of **1,015 plain-text documentation files** from the official Python 3.12 and 3.13 documentation archives (`python-3.12-docs-text.zip` and `python-3.13-docs-text.zip` from `docs.python.org`), totaling **3,521,578 tokens**, **55,764 unique terms**, and **645,774 postings**.

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

### Lab 01: Corpus Statistics

```powershell
uv run python -m findex.stats data/
uv run python -m findex.stats data/ --limit 100
uv run python -m findex.stats data/ --benchmark
```

### Lab 02: Build & Search Inverted Index

Build and persist the inverted index (`binary` array format or `pickle`, with optional token positions):

```powershell
uv run python -m findex.index data/ --out index.bin --format binary
uv run python -m findex.index data/ --out index.pkl --format pickle
uv run python -m findex.index data/ --out index_pos.pkl --format pickle --positions
```

Run Boolean queries (`AND` implicit or explicit, `OR`, `NOT`) using two-pointer `merge` or `set` engine:

```powershell
uv run python -m findex.search index.bin "generator AND iterator"
uv run python -m findex.search index.bin "dictionary OR hash NOT list" --engine merge
uv run python -m findex.search index.bin --benchmark
uv run python -m findex.index data/ --benchmark
```

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

In the **eager** implementation, peak memory spikes to **243.49 MiB** because the pipeline materializes three large collections in RAM simultaneously: the full list of `Document` objects containing every raw file text, a nested `list[list[str]]` of per-document token lists, and a flattened `list[str]` of all 3.52 million token strings (plus Python list pointer overhead). In contrast, the **lazy** generator pipeline peaks at **36.45 MiB** because documents and tokens are streamed one at a time and immediately discarded after updating the `Counter`; memory usage is bounded solely by the single largest document in flight and the resulting 55,764-term vocabulary table.

---

## Lab 02: Inverted Index, Hashing & Memory

### Memory & Storage Study (Postings Representations)

Measured across all 1,015 documents (55,764 terms, 645,774 postings) under `tracemalloc`:

| Postings representation               | Peak memory (build) | Index file size | Load time |
| :------------------------------------ | ------------------: | --------------: | --------: |
| `list[Posting]` (plain `@dataclass`)  |          101.81 MiB |       14.33 MiB |  0.4162 s |
| `list[Posting]` (`slots=True`)        |           78.25 MiB |       10.63 MiB |  2.7764 s |
| `array('I')` pairs (`doc_ids`, `tfs`) |           53.74 MiB |        7.42 MiB |  0.1105 s |

**Where the bytes go:** In a plain `@dataclass`, every single `Posting` instance allocates a dynamic `__dict__` hash table (~104 bytes of overhead on 64-bit CPython) plus object header and references to separate 28-byte Python `int` objects, inflating peak RAM to **101.81 MiB** across 645,774 postings. Adding `slots=True` eliminates the per-instance `__dict__` by storing fixed C-level attribute pointers directly inside the object struct, saving ~23.5 MiB in RAM and shrinking the serialized file from **14.33 MiB** to **10.63 MiB**. Switching to paired `array('I')` buffers removes per-posting Python object headers and 8-byte list pointers altogether, packing `doc_id` and `tf` as contiguous 4-byte unsigned C integers—cutting peak memory nearly in half (**53.74 MiB**) and dropping load time to **0.1105 s**.

### Serialization Formats & Optional Positions

| Format / Mode                       | Positions | File size | Save time | Load time |
| :---------------------------------- | :-------: | --------: | --------: | --------: |
| Binary (`array('I')` + JSON header) |    No     |  6.34 MiB |  0.2173 s |  3.2459 s |
| `pickle` (`HIGHEST_PROTOCOL`)       |    No     | 10.73 MiB |  2.1321 s |  2.7764 s |
| `pickle` (`HIGHEST_PROTOCOL`)       |    Yes    | 22.07 MiB |  3.0463 s |         — |

Enabling `--positions` stores exact token offsets (`tuple[int, ...]`) for every occurrence of every term, roughly doubling the serialized index size (**10.73 MiB → 22.07 MiB**) and increasing build peak RAM from **78.32 MiB** to **222.93 MiB** to unlock phrase queries in Lab 03.

> **Security Note on `pickle`:** `pickle.load()` is a stack-based virtual machine that can instantiate arbitrary classes and execute code via `__reduce__` during deserialization. It is safe only for locally generated index files and must never be called on untrusted or network-downloaded files.

### Boolean Query Benchmark: `merge` vs. `set`

Measured over **2,000 iterations** on the two most frequent terms (`the` in 1,011 docs, `in` in 982 docs) and the two rarest terms (`deconstruct` in 1 doc, `provisionally` in 1 doc):

| Query type            | Terms                          | `merge` (two-pointer) | `set` (hash set) |
| :-------------------- | :----------------------------- | --------------------: | ---------------: |
| 2 most common (`AND`) | `the in`                       |             839.48 ms |        305.72 ms |
| 2 most common (`OR`)  | `the OR in`                    |             851.87 ms |        298.76 ms |
| 2 rarest (`AND`)      | `deconstruct provisionally`    |              66.89 ms |         89.74 ms |
| 2 rarest (`OR`)       | `deconstruct OR provisionally` |              72.77 ms |         92.03 ms |

On **rare terms** (short postings lists), the two-pointer `merge` wins (**66.89 ms vs. 89.74 ms**) because it scans sorted lists directly in $O(|A| + |B|)$ without paying the overhead of allocating hash tables and sorting the result afterwards. On **very frequent terms** (~1,000 postings each), CPython's C-level `set` intersection/union outperforms a pure-Python `while` loop due to vectorized C hash-table probing, though `merge` remains essential for streaming disk-backed postings and positional phrase matching.
