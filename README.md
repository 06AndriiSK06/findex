# findex — Fast Local & Web Search Engine in Python

**Live Production Deployment (Render):** https://findex-a9ma.onrender.com

- **Web Search UI:** https://findex-a9ma.onrender.com/
- **Interactive OpenAPI / Swagger Docs:** https://findex-a9ma.onrender.com/docs
- **Health Check:** https://findex-a9ma.onrender.com/health
- **Index Stats:** https://findex-a9ma.onrender.com/stats

`findex` is a modular, type-checked (`pyright --strict`) command-line and web search engine written in Python 3.14. It supports polite asynchronous web crawling, streaming JSONL ingestion, multi-process/multi-threaded inverted index construction, binary index persistence, ranked retrieval (TF-IDF, BM25, boolean, and positional phrase queries), and a production FastAPI web service.

---

## Web Service & HTTP API (`findex.web`)

`findex` exposes an asynchronous REST API and server-rendered HTML search interface built with **FastAPI**, **Pydantic v2** (`pydantic-settings`), **Jinja2**, and **Uvicorn**.

### Endpoints

| Method | Path             | Description                                                                                                         |
| :----- | :--------------- | :------------------------------------------------------------------------------------------------------------------ |
| `GET`  | `/`              | Server-rendered HTML search UI with `<mark>` snippet highlighting and pagination                                    |
| `GET`  | `/doc/{doc_id}`  | HTML document viewer with query context back-navigation                                                             |
| `GET`  | `/search`        | JSON search endpoint (`q`, `k`, `page`, `scorer=bm25\|tfidf`)                                                       |
| `GET`  | `/docs/{doc_id}` | Full JSON document payload (`doc_id`, `title`, `path`, `content`, `token_count`)                                    |
| `GET`  | `/stats`         | Index statistics (`num_docs`, `vocab_size`, `total_tokens`, `avg_doc_length`, `index_size_bytes`, `uptime_seconds`) |
| `GET`  | `/health`        | Readiness/liveness probe (`200 OK` when index is loaded, `503 Service Unavailable` otherwise)                       |
| `GET`  | `/docs`          | Auto-generated OpenAPI 3.1 Swagger UI                                                                               |

### Running Locally & via Docker

```powershell
# Start web server directly via CLI
uv run findex serve --index data/index.bin --host 0.0.0.0 --port 8000

# Or build and run multi-stage non-root Docker image
docker build -t findex:0.7.0 .
docker run -d -p 8000:8000 -e INDEX_PATH=/app/data/index.bin findex:0.7.0
```

### Concurrency Architecture & Load Benchmarks

- **Thread-Pool Offloading (`def` vs `async def`):** Query parsing, inverted-index intersection, and BM25/TF-IDF scoring are synchronous CPU/memory-bound operations. Route handlers (`/search`, `/docs/{doc_id}`, `/`, `/doc/{doc_id}`) are intentionally defined as plain `def` functions rather than `async def`. FastAPI/Starlette automatically dispatches `def` endpoints onto AnyIO's worker thread pool (`40` threads by default), preventing long searches from blocking the `asyncio` event loop while `/health` and `/stats` remain responsive.
- **GIL & Multi-Worker Scaling:** Under Python's GIL, thread-pool offloading prevents event-loop stalls and overlaps any file I/O (reading document contents on `/docs/{doc_id}`), while pure-Python CPU parallelism scales across OS processes via `--workers N` (`uvicorn --workers 4`). Because the loaded `Index` is read-only after startup, each worker process holds its own in-memory snapshot with zero lock contention.

| Environment                                 | Workload                                                                 | Throughput                                    | p50 Latency | p95 Latency | p99 Latency |
| :------------------------------------------ | :----------------------------------------------------------------------- | :-------------------------------------------- | :---------- | :---------- | :---------- |
| **Localhost (Windows, 1 worker)**           | 200 reqs, concurrency 20 (`/search?q=python&k=10`)                       | **206.2 req/s**                               | `64.87 ms`  | `226.05 ms` | `324.24 ms` |
| **Cloud Production (Render Free, US-West)** | 100 reqs, concurrency 10 (`/search?q=python&k=10`, remote RTT from Kyiv) | **35.5 req/s** (`0.63 ms` server search time) | `234.01 ms` | `338.02 ms` | `714.48 ms` |

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
- **CLI & Web Server (`findex.cli`, `findex.web`):**
  - Built with `typer` and `rich` live progress/status tables (`findex crawl`, `findex index`, `findex search`, `findex stats`, `findex serve`).

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
