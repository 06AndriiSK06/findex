# findex — Fast Local & Web Search Engine in Python

`findex` is a modular, type-checked (`pyright --strict`) command-line search engine written in Python 3.14. It supports polite asynchronous web crawling, streaming JSONL ingestion, multi-process/multi-threaded inverted index construction, binary index persistence, and ranked retrieval (TF-IDF, BM25, boolean, and positional phrase queries).

---

## Features & Architecture

- **Polite Async Web Crawler (`findex.crawler`):**
  - Built on `asyncio` (`TaskGroup`, bounded `Queue`, `Semaphore`) and `httpx.AsyncClient`.
  - Strict `robots.txt` compliance (`urllib.robotparser`) with per-origin caching and `Crawl-delay` support.
  - Per-host concurrency limits and request pacing (`_HostRateLimiter`).
  - URL canonicalization (scheme/host normalization, fragment stripping, query parameter sorting, domain allowlisting).
  - Exponential backoff with jitter and `Retry-After` handling on transient HTTP `429`/`503`/network errors.
  - Streaming JSONL output (`url`, `title`, `text`, `fetched_at`) and structured per-URL `crawl.log`.
- **Corpus & Tokenization (`findex.corpus`, `findex.tokenize`):**
  - Streaming document reader (`iter_documents`) supporting both local directory trees (`.txt`, `.md`) and crawled `.jsonl` files without loading entire datasets into RAM.
  - Unicode-aware regex tokenization with lowercasing and stop-word filtering.
- **Inverted Index & Parallel Indexing (`findex.index`, `findex.store`):**
  - Inverted index with term frequencies (`tf`) and optional positional postings (`positions`).
  - Map-reduce style parallel index building (`serial`, `threads`, `processes`) with deterministic partial index merging.
  - Compact binary serialization (`array('I')` / `struct` framing) for fast load and query times.
- **Query Engine & Ranking (`findex.query`, `findex.ranking`, `findex.search`):**
  - Boolean (`AND`, `OR`, `NOT`), phrase (`"exact phrase"`), and free-text queries.
  - Okapi BM25 and TF-IDF scoring with document metadata and snippets.
- **CLI (`findex.cli`):**
  - Built with `typer` and `rich` live progress/status tables (`findex crawl`, `findex index`, `findex search`, `findex stats`).

---

## Installation & Development

Requires **Python 3.14+** and [`uv`](https://docs.astral.sh/uv/).

```powershell
# Install dependencies
uv sync

# Run linter, formatter, strict type checker, and test suite
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

---

## Quickstart CLI Usage

### 1. Crawl a Website to Streaming JSONL

```powershell
uv run findex crawl https://docs.python.org/3/ `
  --max-pages 200 `
  --concurrency 20 `
  --per-host 5 `
  --min-delay 0.05 `
  --out data/lab06_bench/crawl_c20.jsonl `
  --log data/lab06_bench/crawl_c20.log
```

### 2. Build a Positional Inverted Index

```powershell
# From crawled JSONL
uv run findex index data/lab06_bench/crawl_c20.jsonl --out data/lab06_bench/crawl_c20.bin --positions
# Indexed 200 documents (15147 terms) via processes (workers=8) -> data\lab06_bench\crawl_c20.bin

# Or from a local directory of text/markdown files
uv run findex index data/corpus --out data/index.bin --positions --executor processes --workers 8
```

### 3. Search the Index

```powershell
uv run findex search data/lab06_bench/crawl_c20.bin "asyncio" --k 3
```

---

## Lab 06 — Async Crawler (`asyncio` + `httpx`)

### Target Site & Politeness Policy

- **Seed URL:** `https://docs.python.org/3/` (official Python 3 documentation; `robots.txt` explicitly allows automated indexing of `/3/`).
- **Identification & Limits:**
  - Custom `User-Agent`: `findex-crawler/0.4 (+https://github.com/student/findex)`.
  - Per-origin `robots.txt` check before every request (`RobotsCache`), honoring `Disallow` rules and `Crawl-delay`.
  - Per-host semaphore (`per_host <= 5`) and minimum delay between requests to the same host (`min_delay = 0.05s`).
  - Bounded retries with exponential backoff + jitter on `429`, `502`, `503`, `504`, and transport timeouts (`Retry-After` respected).

### Concurrency Benchmark (`max_pages = 200`)

Measured via `scripts/bench_lab06.py` (`docs/bench_lab06.json`) against `https://docs.python.org/3/` across concurrency levels `1`, `5`, and `20`:

| Concurrency | Pages | Wall (s) | Pages/s | Errors | Peak RSS (MB) |
| ----------- | ----: | -------: | ------: | -----: | ------------: |
| 1           |   200 |   29.774 |    6.72 |      0 |         408.7 |
| 5           |   200 |   18.290 |   10.93 |      0 |         269.1 |
| 20          |   200 |   18.109 |   11.04 |      0 |         414.9 |

**Observations:**

- Increasing concurrency from `1` to `5` reduces wall-clock time by **38.6%** (`29.774s → 18.290s`) and raises throughput from `6.72` to `10.93` pages/s by overlapping network I/O latency across concurrent tasks.
- Increasing global concurrency from `5` to `20` produces nearly identical throughput (`11.04` pages/s, `18.109s`) because all 200 pages reside on a single host (`docs.python.org`), where the per-host semaphore (`per_host = 5`) and pacing delay (`min_delay = 0.05s`) intentionally throttle per-domain load.

### Error Handling & Log Sample

Permanent HTTP errors (such as `404 Not Found`) are logged immediately without wasting retry budget and without interrupting active workers:

```text
2026-11-24T00:51:34+00:00 | status=404 | bytes=9548 | 0.163s | https://docs.python.org/3/nonexistent-lab06-check | HTTP 404
```

### Verification Summary

- **Unit & Integration Tests:** `48 passed` (`pytest`), including offline `httpx.MockTransport` tests for URL canonicalization, `429`/`503` backoff retries, `404` non-retry behavior, `robots.txt` disallow caching, and end-to-end JSONL streaming into `build_index`.
- **Static Analysis:** `0 errors` under `pyright` (`strict` mode) and `ruff`.
