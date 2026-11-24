from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qsl, urldefrag, urlencode, urljoin, urlsplit, urlunsplit

import httpx

from findex.crawler.fetch import DEFAULT_USER_AGENT, fetch
from findex.crawler.robots import RobotsCache

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Page:
    url: str
    title: str
    text: str
    fetched_at: str


@dataclass(slots=True)
class CrawlStats:
    pages_fetched: int = 0
    errors: int = 0
    in_flight: int = 0
    queue_depth: int = 0
    bytes_total: int = 0
    started_at: float = 0.0

    @property
    def elapsed(self) -> float:
        if self.started_at <= 0.0:
            return 0.0
        return max(1e-6, time.perf_counter() - self.started_at)

    @property
    def pages_per_sec(self) -> float:
        return self.pages_fetched / max(1e-6, self.elapsed)


def canonicalize_url(raw_url: str, *, base_url: str | None = None) -> str | None:
    joined = urljoin(base_url, raw_url.strip()) if base_url else raw_url.strip()
    no_frag, _ = urldefrag(joined)
    parts = urlsplit(no_frag)
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"} or not parts.netloc:
        return None

    host = (parts.hostname or "").lower()
    if not host:
        return None

    port = parts.port
    default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    netloc = host if (port is None or default_port) else f"{host}:{port}"

    path = parts.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    sorted_query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((scheme, netloc, path, sorted_query, ""))


class _LinkAndTextParser(HTMLParser):
    _SKIP_TAGS = {"script", "style", "noscript", "svg", "head"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self._title_parts: list[str] = []
        self._text_parts: list[str] = []
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower = tag.lower()
        if lower == "title":
            self._in_title = True
        if lower in self._SKIP_TAGS:
            self._skip_depth += 1
        if lower == "a":
            for k, v in attrs:
                if k.lower() == "href" and v:
                    self.links.append(v)

    def handle_endtag(self, tag: str) -> None:
        lower = tag.lower()
        if lower == "title":
            self._in_title = False
        if lower in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        stripped = data.strip()
        if not stripped:
            return
        if self._in_title:
            self._title_parts.append(stripped)
        if self._skip_depth == 0:
            self._text_parts.append(stripped)

    @property
    def title(self) -> str:
        return " ".join(self._title_parts).strip()

    @property
    def text(self) -> str:
        return " ".join(self._text_parts).strip()


def extract_links_and_text(html: str, *, base_url: str) -> tuple[list[str], str, str]:
    parser = _LinkAndTextParser()
    parser.feed(html)
    parser.close()

    canonical_links: list[str] = []
    for href in parser.links:
        canon = canonicalize_url(href, base_url=base_url)
        if canon is not None:
            canonical_links.append(canon)
    return canonical_links, parser.title, parser.text


class _HostRateLimiter:
    def __init__(self, per_host_concurrency: int, min_delay: float) -> None:
        self._per_host_concurrency = per_host_concurrency
        self._min_delay = min_delay
        self._sems: dict[str, asyncio.Semaphore] = {}
        self._pace_locks: dict[str, asyncio.Lock] = {}
        self._last_request: dict[str, float] = {}

    def semaphore(self, host: str) -> asyncio.Semaphore:
        sem = self._sems.get(host)
        if sem is None:
            sem = asyncio.Semaphore(self._per_host_concurrency)
            self._sems[host] = sem
        return sem

    async def wait_turn(self, host: str, extra_delay: float | None = None) -> None:
        delay = max(self._min_delay, extra_delay or 0.0)
        if delay <= 0.0:
            return
        lock = self._pace_locks.setdefault(host, asyncio.Lock())
        async with lock:
            now = time.perf_counter()
            prev = self._last_request.get(host)
            if prev is not None:
                wait_for = delay - (now - prev)
                if wait_for > 0.0:
                    await asyncio.sleep(wait_for)
            self._last_request[host] = time.perf_counter()


def _domain_allowed(host: str, allowed_domains: set[str]) -> bool:
    return any(host == d or host.endswith(f".{d}") for d in allowed_domains)


async def crawl(
    seeds: Sequence[str],
    *,
    max_pages: int = 200,
    concurrency: int = 10,
    per_host: int = 2,
    min_delay: float = 0.1,
    timeout: float = 10.0,
    allowed_domains: set[str] | None = None,
    user_agent: str = DEFAULT_USER_AGENT,
    log_path: Path | None = None,
    stats: CrawlStats | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncIterator[Page]:
    if max_pages < 1:
        return

    crawl_stats = stats if stats is not None else CrawlStats()
    crawl_stats.started_at = time.perf_counter()

    canonical_seeds: list[str] = []
    for s in seeds:
        c = canonicalize_url(s)
        if c is not None:
            canonical_seeds.append(c)

    if not canonical_seeds:
        return

    domains: set[str] = (
        {d.lower().lstrip(".") for d in allowed_domains}
        if allowed_domains is not None
        else {urlsplit(s).hostname or "" for s in canonical_seeds}
    )
    domains.discard("")

    frontier: asyncio.Queue[str] = asyncio.Queue()
    seen: set[str] = set()
    for seed in canonical_seeds:
        host = urlsplit(seed).hostname or ""
        if _domain_allowed(host, domains) and seed not in seen:
            seen.add(seed)
            frontier.put_nowait(seed)
    crawl_stats.queue_depth = frontier.qsize()

    if frontier.empty():
        return

    out_queue: asyncio.Queue[Page | None] = asyncio.Queue()
    global_sem = asyncio.Semaphore(concurrency)
    host_limiter = _HostRateLimiter(per_host_concurrency=per_host, min_delay=min_delay)
    stop_event = asyncio.Event()

    log_file = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")

    def _write_log(url: str, status: int, nbytes: int, elapsed: float, note: str = "") -> None:
        if log_file is None:
            return
        ts = datetime.now(UTC).isoformat(timespec="seconds")
        suffix = f" | {note}" if note else ""
        line = f"{ts} | status={status} | bytes={nbytes} | {elapsed:.3f}s | {url}{suffix}\n"
        log_file.write(line)
        log_file.flush()

    client_kwargs: dict[str, object] = {
        "headers": {"User-Agent": user_agent},
        "timeout": httpx.Timeout(timeout),
    }
    if transport is not None:
        client_kwargs["transport"] = transport

    async with httpx.AsyncClient(**client_kwargs) as client:  # type: ignore[arg-type]
        robots = RobotsCache(client, user_agent=user_agent, timeout=timeout)

        async def _worker() -> None:
            while not stop_event.is_set():
                url = await frontier.get()
                crawl_stats.queue_depth = frontier.qsize()
                try:
                    if stop_event.is_set():
                        continue

                    host = urlsplit(url).hostname or ""
                    if not _domain_allowed(host, domains):
                        _write_log(url, 0, 0, 0.0, "disallowed_domain")
                        continue

                    if not await robots.allowed(url):
                        _write_log(url, 0, 0, 0.0, "disallowed_by_robots")
                        continue

                    async with global_sem, host_limiter.semaphore(host):
                        if stop_event.is_set():
                            continue
                        await host_limiter.wait_turn(host, robots.crawl_delay(url))
                        crawl_stats.in_flight += 1
                        try:
                            res = await fetch(client, url, timeout=timeout)
                        finally:
                            crawl_stats.in_flight -= 1

                    _write_log(
                        url,
                        res.status,
                        res.bytes_received,
                        res.elapsed,
                        res.error or "ok",
                    )

                    if res.error is not None or res.status >= 400:
                        crawl_stats.errors += 1
                        continue

                    crawl_stats.bytes_total += res.bytes_received
                    if "html" not in res.content_type.lower() and res.content_type != "":
                        continue

                    links, title, text = await asyncio.to_thread(
                        extract_links_and_text,
                        res.text,
                        base_url=res.final_url,
                    )

                    if not stop_event.is_set():
                        for link in links:
                            lhost = urlsplit(link).hostname or ""
                            if _domain_allowed(lhost, domains) and link not in seen:
                                seen.add(link)
                                frontier.put_nowait(link)
                        crawl_stats.queue_depth = frontier.qsize()

                        if crawl_stats.pages_fetched < max_pages:
                            crawl_stats.pages_fetched += 1
                            page = Page(
                                url=res.final_url,
                                title=title or res.final_url,
                                text=text,
                                fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
                            )
                            await out_queue.put(page)
                            if crawl_stats.pages_fetched >= max_pages:
                                stop_event.set()
                finally:
                    frontier.task_done()

        async def _coordinator() -> None:
            join_task = asyncio.create_task(frontier.join())
            stop_task = asyncio.create_task(stop_event.wait())
            try:
                async with asyncio.TaskGroup() as tg:
                    workers = [tg.create_task(_worker()) for _ in range(concurrency)]
                    await asyncio.wait(
                        {join_task, stop_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    stop_event.set()
                    for w in workers:
                        if not w.done():
                            w.cancel()
            finally:
                for task in (join_task, stop_task):
                    if not task.done():
                        task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                await out_queue.put(None)

        coord_task = asyncio.create_task(_coordinator())
        try:
            while True:
                item = await out_queue.get()
                if item is None:
                    break
                yield item
        finally:
            stop_event.set()
            if not coord_task.done():
                coord_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await coord_task
            if log_file is not None:
                log_file.close()
