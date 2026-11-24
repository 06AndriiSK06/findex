from __future__ import annotations

import asyncio
import email.utils
import logging
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

log = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "findex-crawler/0.6 (+https://github.com/06AndriiSK06/findex)"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    final_url: str
    status: int
    text: str
    content_type: str
    bytes_received: int
    elapsed: float
    attempts: int
    error: str | None = None


def parse_retry_after(header_val: str | None) -> float | None:
    if not header_val:
        return None
    raw = header_val.strip()
    try:
        seconds = float(raw)
        return max(0.0, seconds)
    except ValueError:
        pass
    try:
        dt = email.utils.parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        delta = (dt - datetime.now(UTC)).total_seconds()
        return max(0.0, delta)
    except (TypeError, ValueError):
        return None


def _compute_backoff(
    attempt: int,
    *,
    base: float = 0.5,
    cap: float = 10.0,
    retry_after: float | None = None,
) -> float:
    jitter = random.uniform(0.0, 0.25)
    exp_delay = min(cap, base * (2**attempt)) + jitter
    if retry_after is not None:
        return min(cap, max(retry_after, exp_delay))
    return exp_delay


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    max_retries: int = 3,
    backoff_base: float = 0.5,
    backoff_cap: float = 10.0,
) -> FetchResult:
    t0 = time.perf_counter()
    last_status = 0
    last_error: str | None = None

    for attempt in range(max_retries + 1):
        try:
            async with asyncio.timeout(timeout):
                resp = await client.get(url, follow_redirects=True)
            elapsed = time.perf_counter() - t0
            raw_bytes = resp.content
            status = resp.status_code
            content_type = resp.headers.get("content-type", "")

            if status in RETRYABLE_STATUS_CODES:
                last_status = status
                last_error = f"HTTP {status}"
                if attempt < max_retries:
                    retry_after = (
                        parse_retry_after(resp.headers.get("Retry-After"))
                        if status == 429
                        else None
                    )
                    delay = _compute_backoff(
                        attempt,
                        base=backoff_base,
                        cap=backoff_cap,
                        retry_after=retry_after,
                    )
                    log.debug(
                        "Retrying %s after status %d (attempt %d/%d, sleep=%.2fs)",
                        url,
                        status,
                        attempt + 1,
                        max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                return FetchResult(
                    url=url,
                    final_url=str(resp.url),
                    status=status,
                    text="",
                    content_type=content_type,
                    bytes_received=len(raw_bytes),
                    elapsed=elapsed,
                    attempts=attempt + 1,
                    error=last_error,
                )

            if status >= 400:
                return FetchResult(
                    url=url,
                    final_url=str(resp.url),
                    status=status,
                    text="",
                    content_type=content_type,
                    bytes_received=len(raw_bytes),
                    elapsed=elapsed,
                    attempts=attempt + 1,
                    error=f"HTTP {status}",
                )

            return FetchResult(
                url=url,
                final_url=str(resp.url),
                status=status,
                text=resp.text,
                content_type=content_type,
                bytes_received=len(raw_bytes),
                elapsed=elapsed,
                attempts=attempt + 1,
                error=None,
            )

        except TimeoutError:
            last_status = 0
            last_error = "timeout"
            if attempt < max_retries:
                delay = _compute_backoff(attempt, base=backoff_base, cap=backoff_cap)
                log.debug(
                    "Retrying %s after timeout (attempt %d/%d, sleep=%.2fs)",
                    url,
                    attempt + 1,
                    max_retries,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
        except httpx.TooManyRedirects:
            return FetchResult(
                url=url,
                final_url=url,
                status=0,
                text="",
                content_type="",
                bytes_received=0,
                elapsed=time.perf_counter() - t0,
                attempts=attempt + 1,
                error="too_many_redirects",
            )
        except httpx.HTTPError as exc:
            last_status = 0
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                delay = _compute_backoff(attempt, base=backoff_base, cap=backoff_cap)
                await asyncio.sleep(delay)
                continue

    return FetchResult(
        url=url,
        final_url=url,
        status=last_status,
        text="",
        content_type="",
        bytes_received=0,
        elapsed=time.perf_counter() - t0,
        attempts=max_retries + 1,
        error=last_error or "fetch_failed",
    )
