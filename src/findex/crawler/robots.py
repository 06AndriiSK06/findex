from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from findex.crawler.fetch import DEFAULT_USER_AGENT

log = logging.getLogger(__name__)


class RobotsCache:
    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = 10.0,
    ) -> None:
        self._client = client
        self._user_agent = user_agent
        self._timeout = timeout
        self._parsers: dict[str, RobotFileParser | None] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _host_key(self, url: str) -> tuple[str, str]:
        parts = urlsplit(url)
        scheme = (parts.scheme or "https").lower()
        netloc = parts.netloc.lower()
        return f"{scheme}://{netloc}", f"{scheme}://{netloc}/robots.txt"

    async def _load_parser(self, origin: str, robots_url: str) -> RobotFileParser | None:
        try:
            async with asyncio.timeout(self._timeout):
                resp = await self._client.get(robots_url, follow_redirects=True)
        except (TimeoutError, httpx.HTTPError) as exc:
            log.warning("Failed to fetch %s (%s); assuming allowed", robots_url, exc)
            return None

        if resp.status_code in {401, 403}:
            parser = RobotFileParser()
            parser.parse(["User-agent: *", "Disallow: /"])
            return parser

        if resp.status_code >= 400:
            return None

        parser = RobotFileParser()
        parser.set_url(robots_url)
        parser.parse(resp.text.splitlines())
        return parser

    async def allowed(self, url: str) -> bool:
        origin, robots_url = self._host_key(url)
        if origin not in self._parsers:
            lock = self._locks.setdefault(origin, asyncio.Lock())
            async with lock:
                if origin not in self._parsers:
                    self._parsers[origin] = await self._load_parser(origin, robots_url)

        parser = self._parsers[origin]
        if parser is None:
            return True
        return parser.can_fetch(self._user_agent, url)

    def crawl_delay(self, url: str) -> float | None:
        origin, _ = self._host_key(url)
        parser = self._parsers.get(origin)
        if parser is None:
            return None
        delay = parser.crawl_delay(self._user_agent)
        return float(delay) if delay is not None else None
