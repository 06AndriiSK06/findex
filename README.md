# findex — Search Engine

An incremental full-text search engine built from scratch in Python using streaming generators, a Pythonic `Index` data model, recursive-descent query parsing, and TF-IDF / BM25 ranking.

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

### Lab 02 & 03: Build, Persist & Rank Search

Build and persist the inverted index (with optional `--positions` for phrase queries):

```powershell
uv run python -m findex.index data/ --out index.bin --format binary
uv run python -m findex.index data/ --out index_pos.bin --format binary --positions
uv run python -m findex.index data/ --benchmark
```

Run ranked queries (`BM25` or `TF-IDF`) with Boolean operators, parentheses, `"exact phrases"`, snippets, and caching:

```powershell
uv run python -m findex.search index_pos.bin 'python AND (async OR await) NOT java \"event loop\"' --repeat
uv run python -m findex.search index_pos.bin "generator AND iterator" --scorer tfidf
uv run python -m findex.search index_pos.bin --eval
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

In the **eager** implementation, peak memory spikes to **243.49 MiB** because the pipeline materializes three large collections in RAM simultaneously: the full list of `Document` objects containing every raw file text, a nested `list[list[str]]` of per-document token lists, and a flattened `list[str]` of all 3.52 million token strings. In contrast, the **lazy** generator pipeline peaks at **36.45 MiB** because documents and tokens are streamed one at a time and immediately discarded after updating the `Counter`.

---

## Lab 02: Inverted Index, Hashing & Memory

### Memory & Storage Study (Postings Representations)

Measured across all 1,015 documents (55,764 terms, 645,774 postings) under `tracemalloc`:

| Postings representation               | Peak memory (build) | Index file size | Load time |
| :------------------------------------ | ------------------: | --------------: | --------: |
| `list[Posting]` (plain `@dataclass`)  |          101.81 MiB |       14.33 MiB |  0.4162 s |
| `list[Posting]` (`slots=True`)        |           78.25 MiB |       10.63 MiB |  2.7764 s |
| `array('I')` pairs (`doc_ids`, `tfs`) |           53.74 MiB |        7.42 MiB |  0.1105 s |

**Where the bytes go:** In a plain `@dataclass`, every `Posting` instance allocates a dynamic `__dict__` hash table (~104 bytes of overhead on 64-bit CPython) plus object headers and references to 28-byte Python `int` objects, inflating peak RAM to **101.81 MiB**. Adding `slots=True` eliminates the per-instance `__dict__` by storing fixed C-level attribute pointers directly inside the object struct, saving ~23.5 MiB in RAM and shrinking the serialized file to **10.63 MiB**. Switching to paired `array('I')` buffers removes per-posting Python object headers and list pointers altogether, packing `doc_id` and `tf` as contiguous 4-byte unsigned C integers (**53.74 MiB** peak RAM, **0.1105 s** load).

> **Security Note on `pickle`:** `pickle.load()` is a stack-based virtual machine that can instantiate arbitrary classes and execute code via `__reduce__` during deserialization. It is safe only for locally generated index files and must never be called on untrusted or network-downloaded files.

### Boolean Query Benchmark: `merge` vs. `set`

Measured over **2,000 iterations** on the two most frequent terms (`the` in 1,011 docs, `in` in 982 docs) and the two rarest terms (`deconstruct` in 1 doc, `provisionally` in 1 doc):

| Query type            | Terms                          | `merge` (two-pointer) | `set` (hash set) |
| :-------------------- | :----------------------------- | --------------------: | ---------------: |
| 2 most common (`AND`) | `the in`                       |             839.48 ms |        305.72 ms |
| 2 most common (`OR`)  | `the OR in`                    |             851.87 ms |        298.76 ms |
| 2 rarest (`AND`)      | `deconstruct provisionally`    |              66.89 ms |         89.74 ms |
| 2 rarest (`OR`)       | `deconstruct OR provisionally` |              72.77 ms |         92.03 ms |

---

## Lab 03: Ranking, Object Model, Decorators & Query Language

### Data Model, Protocols & Context Management

- **`Index(collections.abc.Mapping[str, list[Posting]])`:** Implements `__len__`, `__contains__`, `__getitem__`, `__iter__`, `__repr__` (`Index(terms=55_764, docs=1_015)`), `num_docs`, `df(term)`, and `@functools.cached_property` for `avg_doc_length`.
- **`open_index(path)`:** `@contextmanager` wrapping `load_index` that guarantees `index.close()` runs in `finally`, even if an exception occurs inside the `with` block.
- **`Scorer` Protocol:** Structural typing (`typing.Protocol`) allowing `TfIdf` and `BM25(k1=1.5, b=0.75)` to be swapped interchangeably; top-`k` results are extracted in $O(n \log k)$ via `heapq.nlargest` into `@dataclass(order=True)` `SearchResult` records.
- **Recursive-Descent Query Parser:** Parses implicit/explicit `AND`, `OR`, `NOT`, parentheses, and `"quoted phrases"` (evaluated via positional postings) into composable AST nodes (`Term`, `Phrase`, `And`, `Or`, `Not`) supporting `&`, `|`, and `~`.

### Three Ranking Sanity Checks (BM25 on Corpus)

1. **Rare term outranks common term** (`mature` with $df=2$ vs. `the` with $df=1011$):
   - `doc_id=502` (`py3.13 bugs`, contains `mature`): **score = `9.5264`**
   - `doc_id=1` (`py3.12 bugs`, contains `mature`): **score = `9.4354`**
   - `doc_id=903` (`py3.13 library tkinter`, contains only `the`): **score = `0.0111`**
2. **Term-frequency saturation** (`generator` in `doc_id=0`):
   - `tf = 1`: **`3.2573`**
   - `tf = 19`: **`4.5572`**
   - `tf = 20`: **`4.5622`** (marginal gain of the 20th occurrence is only **`+0.005057`**)
3. **Document-length normalization** (`asyncio`, both with `tf = 1`):
   - Short document `[151]` (`len = 67` tokens): **score = `3.8160`**
   - Long document `[811]` (`len = 16,718` tokens): **score = `0.7843`**

### Precision@5 Evaluation (10 Hand-Labeled Queries: TF-IDF vs. BM25)

Relevance ground truth requires top-5 results to belong to the corresponding module/guide documentation pages:

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

BM25 outperforms TF-IDF (**0.58 vs. 0.40 Mean P@5**) because document-length normalization prevents massive index/changelog files from dominating short, focused module reference pages.

### `@timed` Decorator & `@lru_cache` Evidence

Running `uv run python -m findex.search index_pos.bin 'python AND (async OR await) NOT java \"event loop\"' --repeat`:

```text
[timed] load_index took 16128.02 ms
[cache MISS] parsing and evaluating query='python AND (async OR await) NOT java "event loop"'
[timed] search took 46.38 ms
[cache HIT] query='python AND (async OR await) NOT java "event loop"'
[timed] search took 36.66 ms

Index:   Index(terms=55_764, docs=1_015)
Query:   'python AND (async OR await) NOT java "event loop"' (scorer=bm25)
Results: 5 (top-5) in 46.65 ms
Peak memory: 249.09 MiB

1. [18.6475] (609) py3.13 howto a-conceptual-overview-of-asyncio: A Conceptual Overview of "asyncio" ********************************** This HOWTO article seeks to help you...
2. [18.5059] (650) py3.13 library asyncio-dev: Developing with asyncio *********************** Asynchronous programming is different from class...
3. [18.4173] (149) py3.12 library asyncio-task: Coroutines and Tasks ******************** This section outlines high-level asyncio APIs to work with coroutines and Tasks. * Coroutines * Awaitables * Crea...
4. [18.3875] (664) py3.13 library asyncio-task: Coroutines and tasks ******************** This section outlines high-level asyncio APIs to work with coroutines and Tasks. * Coroutines * Awaitables * Crea...
5. [17.8322] (136) py3.12 library asyncio-eventloop: [Event] [Loop] ********** **Source code:** Lib/asyncio/events.py, Lib/asyncio/base_e...
```
