from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.live import Live
from rich.markup import escape
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from findex.crawler import CrawlStats, crawl
from findex.index import ExecutorKind, build_index
from findex.query import QuerySyntaxError
from findex.ranking import ScorerName, make_scorer
from findex.search import search as run_search
from findex.semantic import (
    SearchMode,
    SentenceTransformerEncoder,
    build_semantic_index,
    hybrid_search,
    load_semantic_index,
    save_semantic_index,
    semantic_search,
)
from findex.stats import compute_stats
from findex.store import open_index, save_array_binary

app = typer.Typer(
    name="findex",
    help="Full-text and semantic search engine CLI.",
    no_args_is_help=True,
    add_completion=False,
)

log = logging.getLogger(__name__)


def _stdout() -> Console:
    return Console(file=sys.stdout)


def _stderr() -> Console:
    return Console(file=sys.stderr)


def _configure_logging(verbosity: int) -> None:
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )


def _fail(message: str) -> typer.Exit:
    _stderr().print(f"[bold red]Error:[/bold red] {escape(message)}")
    return typer.Exit(code=1)


def _format_snippet_rich(snippet: str) -> str:
    parts = snippet.split("[")
    out: list[str] = [escape(parts[0])]
    for part in parts[1:]:
        if "]" in part:
            term, rest = part.split("]", 1)
            out.append(f"[bold yellow]{escape(term)}[/bold yellow]{escape(rest)}")
        else:
            out.append(escape("[" + part))
    return "".join(out)


@app.callback()
def main_callback(
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Increase diagnostic verbosity (-v for INFO, -vv for DEBUG).",
        ),
    ] = 0,
) -> None:
    _configure_logging(verbose)


@app.command("crawl")
def crawl_cmd(
    seed_url: Annotated[str, typer.Argument(help="Initial seed URL to crawl")],
    max_pages: Annotated[
        int,
        typer.Option("--max-pages", min=1, help="Maximum number of pages to fetch"),
    ] = 200,
    concurrency: Annotated[
        int,
        typer.Option("--concurrency", "-c", min=1, help="Max concurrent in-flight requests"),
    ] = 10,
    per_host: Annotated[
        int,
        typer.Option("--per-host", min=1, help="Max concurrent requests per host"),
    ] = 2,
    min_delay: Annotated[
        float,
        typer.Option("--min-delay", min=0.0, help="Minimum delay (s) between requests per host"),
    ] = 0.1,
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Streaming JSONL output file"),
    ] = Path("data/crawl.jsonl"),
    log_file: Annotated[
        Path,
        typer.Option("--log", help="Per-URL crawl log path"),
    ] = Path("crawl.log"),
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Run asyncio event loop in debug mode"),
    ] = False,
) -> None:
    stats = CrawlStats()
    out.parent.mkdir(parents=True, exist_ok=True)

    def _render_status() -> Table:
        tbl = Table(title="findex async crawler", show_header=True)
        tbl.add_column("Pages", justify="right", style="bold green")
        tbl.add_column("Pages/s", justify="right", style="cyan")
        tbl.add_column("In-flight", justify="right", style="yellow")
        tbl.add_column("Errors", justify="right", style="red")
        tbl.add_column("Queue", justify="right", style="magenta")
        tbl.add_row(
            f"{stats.pages_fetched} / {max_pages}",
            f"{stats.pages_per_sec:.2f}",
            str(stats.in_flight),
            str(stats.errors),
            str(stats.queue_depth),
        )
        return tbl

    async def _run_crawl() -> None:
        with (
            out.open("w", encoding="utf-8") as jsonl_fh,
            Live(_render_status(), console=_stderr(), refresh_per_second=8, transient=True) as live,
        ):
            async for page in crawl(
                [seed_url],
                max_pages=max_pages,
                concurrency=concurrency,
                per_host=per_host,
                min_delay=min_delay,
                log_path=log_file,
                stats=stats,
            ):
                record = {
                    "url": page.url,
                    "title": page.title,
                    "text": page.text,
                    "fetched_at": page.fetched_at,
                }
                jsonl_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl_fh.flush()
                live.update(_render_status())

    try:
        asyncio.run(_run_crawl(), debug=debug)
    except (OSError, ValueError) as exc:
        raise _fail(str(exc)) from None

    _stdout().print(
        f"[bold green]Crawled[/bold green] {stats.pages_fetched} pages "
        f"in {stats.elapsed:.2f}s ({stats.pages_per_sec:.2f} pages/s, errors={stats.errors}) "
        f"-> [cyan]{escape(str(out))}[/cyan]"
    )


@app.command("index")
def index_cmd(
    corpus: Annotated[Path, typer.Argument(help="Path to corpus directory or file")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output binary index path")] = Path(
        "index.bin"
    ),
    positions: Annotated[
        bool,
        typer.Option("--positions", help="Store token positions for phrase queries"),
    ] = False,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Maximum number of documents to index"),
    ] = None,
    workers: Annotated[
        int,
        typer.Option("--workers", "-w", min=1, help="Number of worker threads/processes"),
    ] = max(1, min(8, os.cpu_count() or 4)),
    executor: Annotated[
        ExecutorKind,
        typer.Option(
            "--executor",
            "-e",
            help="Execution backend (serial, threads, processes)",
        ),
    ] = "processes",
) -> None:
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed} docs"),
            TimeElapsedColumn(),
            console=_stderr(),
            transient=True,
        ) as progress:
            task_id = progress.add_task(
                f"Indexing ({executor}, workers={workers})...",
                total=limit,
            )

            def _advance(count: int) -> None:
                progress.update(task_id, completed=count)

            idx = build_index(
                corpus,
                limit=limit,
                with_positions=positions,
                workers=workers,
                executor=executor,
                on_doc=_advance,
            )
            save_array_binary(idx, out)

        _stdout().print(
            f"[bold green]Indexed[/bold green] {idx.num_docs} documents "
            f"({len(idx)} terms) via [magenta]{executor}[/magenta] "
            f"(workers={workers}) -> [cyan]{escape(str(out))}[/cyan]"
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise _fail(str(exc)) from None


@app.command("embed")
def embed_cmd(
    index_path: Annotated[Path, typer.Argument(help="Path to saved binary index file")],
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Sentence-Transformers model name"),
    ] = "all-MiniLM-L6-v2",
    corpus: Annotated[
        Path | None,
        typer.Option("--corpus", "-c", help="Optional original corpus path (.jsonl or dir)"),
    ] = None,
    chunk_words: Annotated[
        int,
        typer.Option("--chunk-words", min=32, help="Words per text chunk"),
    ] = 220,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", min=1, help="Embedding batch size"),
    ] = 32,
) -> None:
    try:
        with (
            open_index(index_path) as idx,
            Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total} chunks"),
                TimeElapsedColumn(),
                console=_stderr(),
                transient=True,
            ) as progress,
        ):
            encoder = SentenceTransformerEncoder(model)
            task_id = progress.add_task(f"Embedding chunks ({model})...", total=None)

            def _on_batch(done: int, total: int) -> None:
                progress.update(task_id, completed=done, total=total)

            sem_idx = build_semantic_index(
                idx,
                encoder=encoder,
                model_name=model,
                corpus_path=corpus,
                chunk_words=chunk_words,
                batch_size=batch_size,
                on_batch=_on_batch,
            )
            emb_path, docs_path = save_semantic_index(sem_idx, index_path)

        _stdout().print(
            f"[bold green]Embedded[/bold green] {sem_idx.num_chunks} chunks "
            f"(dim={sem_idx.dim}) using [magenta]{escape(model)}[/magenta] "
            f"-> [cyan]{escape(str(emb_path))}[/cyan], [cyan]{escape(str(docs_path))}[/cyan]"
        )
    except (FileNotFoundError, OSError, ValueError, ImportError) as exc:
        raise _fail(str(exc)) from None


@app.command("search")
def search_cmd(
    index_path: Annotated[Path, typer.Argument(help="Path to saved index file")],
    query: Annotated[str, typer.Argument(help="Search query string")],
    k: Annotated[int, typer.Option("--k", "-k", min=1, help="Number of top results")] = 10,
    scorer: Annotated[
        ScorerName,
        typer.Option("--scorer", "-s", help="Ranking scorer (bm25 or tfidf)"),
    ] = "bm25",
    mode: Annotated[
        SearchMode,
        typer.Option("--mode", "-m", help="Retrieval mode (keyword, semantic, or hybrid)"),
    ] = "keyword",
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON lines to stdout for piping"),
    ] = False,
) -> None:
    try:
        with open_index(index_path) as idx:
            if mode == "keyword":
                results = run_search(
                    idx,
                    query,
                    scorer=make_scorer(scorer),
                    k=k,
                    with_snippets=True,
                )
            else:
                sem_idx = load_semantic_index(index_path)
                if sem_idx is None:
                    raise ValueError(
                        f"Semantic embeddings not found for {index_path}. "
                        f"Run `findex embed {index_path}` first."
                    )
                encoder = SentenceTransformerEncoder(sem_idx.model_name)
                if mode == "semantic":
                    results = semantic_search(
                        idx,
                        sem_idx,
                        query,
                        encoder=encoder,
                        k=k,
                        with_snippets=True,
                    )
                else:
                    results = hybrid_search(
                        idx,
                        sem_idx,
                        query,
                        encoder=encoder,
                        scorer=make_scorer(scorer),
                        k=k,
                        with_snippets=True,
                    )
    except (FileNotFoundError, QuerySyntaxError, ValueError, TypeError, OSError) as exc:
        raise _fail(str(exc)) from None

    if as_json:
        for res in results:
            payload = {
                "doc_id": res.doc_id,
                "score": round(res.score, 6),
                "title": res.title,
                "snippet": res.snippet,
                "mode": mode,
            }
            sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        return

    if not results:
        _stdout().print(f"[yellow]No documents matched query:[/yellow] {escape(repr(query))}")
        return

    label = mode.upper() if mode != "keyword" else scorer.upper()
    table = Table(title=f"Search Results ({label}) — {escape(repr(query))}")
    table.add_column("#", justify="right", style="dim", no_wrap=True)
    table.add_column("Score", justify="right", style="cyan", no_wrap=True)
    table.add_column("Title", style="bold green")
    table.add_column("Snippet")

    for rank, res in enumerate(results, start=1):
        table.add_row(
            str(rank),
            f"{res.score:.4f}",
            escape(res.title),
            _format_snippet_rich(res.snippet),
        )

    _stdout().print(table)


@app.command("stats")
def stats_cmd(
    target: Annotated[Path, typer.Argument(help="Path to corpus directory or index file")],
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Limit documents when scanning a corpus"),
    ] = None,
    top: Annotated[int, typer.Option("--top", help="Number of top terms to display")] = 10,
) -> None:
    try:
        if target.is_file() and target.suffix in {".bin", ".pkl"}:
            with open_index(target) as idx:
                total_tokens = sum(idx.doc_lengths.values())
                top_terms = sorted(
                    ((term, sum(p.tf for p in plist)) for term, plist in idx.items()),
                    key=lambda item: item[1],
                    reverse=True,
                )[:top]
                num_docs = idx.num_docs
                vocab_size = len(idx)
        else:
            st = compute_stats(target, limit=limit, top_n=top)
            num_docs = st.num_documents
            total_tokens = st.total_tokens
            vocab_size = st.vocab_size
            top_terms = st.top_terms
    except (FileNotFoundError, ValueError, TypeError, OSError) as exc:
        raise _fail(str(exc)) from None

    summary = Table(title=f"Corpus / Index Statistics — {escape(str(target))}")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", justify="right", style="cyan")
    summary.add_row("Documents", f"{num_docs:,}")
    summary.add_row("Total tokens", f"{total_tokens:,}")
    summary.add_row("Vocabulary size", f"{vocab_size:,}")
    _stdout().print(summary)

    if top_terms:
        terms_table = Table(title=f"Top {len(top_terms)} Terms")
        terms_table.add_column("Term", style="bold green")
        terms_table.add_column("Frequency", justify="right", style="cyan")
        for term, freq in top_terms:
            terms_table.add_row(escape(term), f"{freq:,}")
        _stdout().print(terms_table)


@app.command("serve")
def serve_cmd(
    index_path: Annotated[
        Path | None,
        typer.Option("--index", "-i", help="Path to serialized index file (overrides INDEX_PATH)"),
    ] = None,
    host: Annotated[
        str,
        typer.Option("--host", help="Host interface to bind"),
    ] = "0.0.0.0",
    port: Annotated[
        int,
        typer.Option("--port", "-p", min=1, max=65535, help="Port to bind"),
    ] = 8000,
    workers: Annotated[
        int,
        typer.Option("--workers", "-w", min=1, help="Number of uvicorn worker processes"),
    ] = 1,
    reload: Annotated[
        bool,
        typer.Option("--reload", help="Enable auto-reload for development"),
    ] = False,
) -> None:
    import uvicorn

    if index_path is not None:
        os.environ["INDEX_PATH"] = str(index_path)

    uvicorn.run(
        "findex.web:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
    )


if __name__ == "__main__":
    app()
