from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from findex.corpus import iter_documents
from findex.crawler.crawl import Page, canonicalize_url, crawl
from findex.crawler.fetch import fetch
from findex.crawler.robots import RobotsCache
from findex.index import build_index


def test_canonicalize_url() -> None:
    assert (
        canonicalize_url("HTTP://Example.COM:80/docs/intro/#section?b=2&a=1")
        == "http://example.com/docs/intro"
    )
    assert (
        canonicalize_url("https://docs.example.com/api/?z=9&a=1#top")
        == "https://docs.example.com/api?a=1&z=9"
    )
    assert canonicalize_url("mailto:test@example.com") is None


@pytest.mark.asyncio
async def test_fetch_retries_429_with_retry_after_and_503() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "0.01"}, text="rate limited")
        if calls == 2:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, text="ok payload", headers={"content-type": "text/html"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        res = await fetch(client, "https://example.com/page", backoff_base=0.01, backoff_cap=0.05)

    assert res.status == 200
    assert res.text == "ok payload"
    assert res.attempts == 3
    assert calls == 3


@pytest.mark.asyncio
async def test_fetch_does_not_retry_404() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404, text="not found")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        res = await fetch(client, "https://example.com/missing", backoff_base=0.01)

    assert res.status == 404
    assert res.attempts == 1
    assert calls == 1


@pytest.mark.asyncio
async def test_robots_cache_obeys_disallow_and_caches() -> None:
    robots_hits = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal robots_hits
        if request.url.path == "/robots.txt":
            robots_hits += 1
            return httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")
        return httpx.Response(200, text="ok")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        cache = RobotsCache(client)
        assert await cache.allowed("https://example.com/public/page") is True
        assert await cache.allowed("https://example.com/private/secret") is False
        assert await cache.allowed("https://example.com/other") is True

    assert robots_hits == 1


@pytest.mark.asyncio
async def test_crawl_respects_max_pages_robots_and_streams_jsonl(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /blocked\n")
        if path == "/":
            html = (
                "<html><head><title>Home</title></head><body>"
                "<h1>Welcome</h1>"
                '<a href="/p1#frag">P1</a>'
                '<a href="/p1/">P1 dup</a>'
                '<a href="/p2">P2</a>'
                '<a href="/blocked">Blocked</a>'
                '<a href="https://other.org/ext">External</a>'
                "</body></html>"
            )
            return httpx.Response(200, text=html, headers={"content-type": "text/html"})
        if path in {"/p1", "/p2", "/p3"}:
            html = f"<html><head><title>{path}</title></head><body>Body {path}</body></html>"
            return httpx.Response(200, text=html, headers={"content-type": "text/html"})
        return httpx.Response(404, text="missing")

    transport = httpx.MockTransport(handler)
    log_path = tmp_path / "crawl.log"
    jsonl_path = tmp_path / "crawl.jsonl"

    pages: list[Page] = []
    with jsonl_path.open("w", encoding="utf-8") as fh:
        async for page in crawl(
            ["https://example.com/"],
            max_pages=2,
            concurrency=2,
            per_host=2,
            min_delay=0.0,
            log_path=log_path,
            transport=transport,
        ):
            pages.append(page)
            fh.write(
                json.dumps(
                    {
                        "url": page.url,
                        "title": page.title,
                        "text": page.text,
                        "fetched_at": page.fetched_at,
                    }
                )
                + "\n"
            )

    assert len(pages) == 2
    assert log_path.exists()
    docs = list(iter_documents(jsonl_path))
    assert len(docs) == 2

    idx = build_index(jsonl_path, with_positions=True, workers=1, executor="serial")
    assert idx.num_docs == 2
