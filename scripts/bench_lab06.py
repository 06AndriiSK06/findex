from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import psutil

from findex.crawler import CrawlStats, crawl

SEED_URL = "https://docs.python.org/3/"
MAX_PAGES = 200
CONCURRENCY_LEVELS = [1, 5, 20]


async def run_single(concurrency: int, out_dir: Path) -> dict[str, object]:
    proc = psutil.Process(os.getpid())
    stats = CrawlStats()
    jsonl_path = out_dir / f"crawl_c{concurrency}.jsonl"
    log_path = out_dir / f"crawl_c{concurrency}.log"

    peak_rss = proc.memory_info().rss
    t0 = time.perf_counter()

    with jsonl_path.open("w", encoding="utf-8") as fh:
        async for page in crawl(
            [SEED_URL],
            max_pages=MAX_PAGES,
            concurrency=concurrency,
            per_host=min(concurrency, 5),
            min_delay=0.05,
            timeout=10.0,
            log_path=log_path,
            stats=stats,
        ):
            fh.write(
                json.dumps(
                    {
                        "url": page.url,
                        "title": page.title,
                        "text": page.text,
                        "fetched_at": page.fetched_at,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            rss_now = proc.memory_info().rss
            if rss_now > peak_rss:
                peak_rss = rss_now

    wall_s = time.perf_counter() - t0
    pages_per_s = stats.pages_fetched / max(1e-6, wall_s)
    peak_rss_mb = round(peak_rss / (1024 * 1024), 1)

    failure_lines: list[str] = []
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if "status=200" not in line and "disallowed" not in line:
                failure_lines.append(line)
                if len(failure_lines) >= 5:
                    break

    return {
        "concurrency": concurrency,
        "pages": stats.pages_fetched,
        "wall_s": round(wall_s, 3),
        "pages_per_s": round(pages_per_s, 2),
        "errors": stats.errors,
        "peak_rss_mb": peak_rss_mb,
        "sample_failures": failure_lines,
    }


async def main() -> None:
    out_dir = Path("data/lab06_bench")
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []

    for c in CONCURRENCY_LEVELS:
        print(f"Running crawl benchmark: concurrency={c}, max_pages={MAX_PAGES} ...")
        row = await run_single(c, out_dir)
        results.append(row)
        print(
            f"  concurrency={row['concurrency']:2d} | pages={row['pages']} | "
            f"wall={row['wall_s']}s | pages/s={row['pages_per_s']} | "
            f"errors={row['errors']} | peak_rss={row['peak_rss_mb']} MB"
        )

    out_json = Path("docs/bench_lab06.json")
    out_json.write_text(
        json.dumps(
            {
                "seed_url": SEED_URL,
                "max_pages": MAX_PAGES,
                "rows": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved -> {out_json}")


if __name__ == "__main__":
    asyncio.run(main())
