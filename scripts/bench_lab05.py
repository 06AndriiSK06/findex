from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import threading
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

try:
    import psutil  # type: ignore[import-untyped]
except ImportError:
    psutil = None

from findex.corpus import collect_document_paths
from findex.index import ExecutorKind, build_index
from findex.tokenize import tokenize

RESULTS_JSON = Path("docs/bench_results.json")
PLOT_PNG = Path("docs/speedup_lab05.png")


def _gil_status() -> bool:
    fn = getattr(sys, "_is_gil_enabled", None)
    if callable(fn):
        return bool(fn())
    return True


def _cpu_model() -> str:
    try:
        import winreg  # type: ignore[import-not-found]

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(val).strip()
    except Exception:
        return platform.processor() or "Unknown CPU"


def _measure_single_run(
    corpus: Path,
    *,
    executor: ExecutorKind,
    workers: int,
) -> dict[str, float]:
    if psutil is not None:
        proc = psutil.Process(os.getpid())
        rss_before = proc.memory_info().rss / (1024 * 1024)
    else:
        tracemalloc.start()
        rss_before = 0.0

    t_wall_0 = time.perf_counter()
    t_cpu_0 = time.process_time()

    idx = build_index(
        corpus,
        with_positions=True,
        workers=workers,
        executor=executor,
    )

    wall_s = time.perf_counter() - t_wall_0
    cpu_s = time.process_time() - t_cpu_0

    if psutil is not None:
        proc = psutil.Process(os.getpid())
        rss_after = proc.memory_info().rss / (1024 * 1024)
        peak_mb = max(rss_before, rss_after)
    else:
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak_bytes / (1024 * 1024)

    return {
        "wall_s": wall_s,
        "cpu_s": cpu_s,
        "peak_rss_mb": peak_mb,
        "merge_s": idx.merge_seconds,
    }


def run_benchmark(corpus: Path, repeats: int = 3) -> list[dict[str, Any]]:
    gil_on = _gil_status()
    py_tag = f"{sys.version_info.major}.{sys.version_info.minor}{'t' if not gil_on else ''}"
    cpu_count = os.cpu_count() or 8
    worker_grid = sorted({1, 2, 4, 8, cpu_count})

    # Warm-up
    _ = build_index(corpus, limit=20, with_positions=True, workers=1, executor="serial")

    rows: list[dict[str, Any]] = []

    configs: list[tuple[str, ExecutorKind, int]] = []
    if gil_on:
        configs.append(("serial", "serial", 1))
    for w in worker_grid:
        if w == 1:
            continue
        label_t = "threads" if gil_on else f"threads ({py_tag}, no-GIL)"
        configs.append((label_t, "threads", w))
    if gil_on:
        for w in worker_grid:
            if w == 1:
                continue
            configs.append(("processes", "processes", w))

    for label, executor, workers in configs:
        samples = [
            _measure_single_run(corpus, executor=executor, workers=workers) for _ in range(repeats)
        ]
        row = {
            "label": label,
            "executor": executor,
            "python": py_tag,
            "gil_enabled": gil_on,
            "workers": workers,
            "wall_s": round(statistics.median(s["wall_s"] for s in samples), 4),
            "cpu_s": round(statistics.median(s["cpu_s"] for s in samples), 4),
            "peak_rss_mb": round(max(s["peak_rss_mb"] for s in samples), 2),
            "merge_s": round(statistics.median(s["merge_s"] for s in samples), 4),
        }
        rows.append(row)
        print(
            f"{row['label']:24s} | workers={row['workers']:2d} | "
            f"wall={row['wall_s']:.4f}s | cpu={row['cpu_s']:.4f}s | "
            f"RSS/Peak={row['peak_rss_mb']:.1f}MB | merge={row['merge_s']:.4f}s"
        )

    return rows


def run_race_demo(corpus: Path) -> dict[str, int]:
    paths = collect_document_paths(corpus, limit=250)
    expected_total = sum(len(list(tokenize(p.read_text(encoding="utf-8")))) for p in paths)

    unsafe_counter = 0

    def _unsafe_worker(chunk: list[Path]) -> None:
        nonlocal unsafe_counter
        for p in chunk:
            for _ in tokenize(p.read_text(encoding="utf-8")):
                unsafe_counter += 1

    safe_counter = 0
    lock = threading.Lock()

    def _safe_worker(chunk: list[Path]) -> None:
        nonlocal safe_counter
        local_count = 0
        for p in chunk:
            local_count += len(list(tokenize(p.read_text(encoding="utf-8"))))
        with lock:
            safe_counter += local_count

    chunks = [paths[i::8] for i in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_unsafe_worker, chunks))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_safe_worker, chunks))

    return {
        "expected": expected_total,
        "unsafe_shared_counter": unsafe_counter,
        "safe_locked_counter": safe_counter,
    }


def run_io_counter_benchmark(corpus: Path) -> dict[str, float]:
    paths = collect_document_paths(corpus)

    def _read_bytes(p: Path) -> int:
        return len(p.read_bytes())

    t0 = time.perf_counter()
    _ = [_read_bytes(p) for p in paths]
    serial_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        _ = list(pool.map(_read_bytes, paths))
    threads_s = time.perf_counter() - t1

    return {
        "io_serial_s": round(serial_s, 4),
        "io_threads_8_s": round(threads_s, 4),
        "io_speedup": round(serial_s / max(threads_s, 1e-9), 2),
    }


def save_and_plot(new_rows: list[dict[str, Any]], extra: dict[str, Any]) -> None:
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {"rows": []}
    if RESULTS_JSON.exists():
        existing = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))

    merged_map: dict[tuple[str, int], dict[str, Any]] = {
        (r["label"], r["workers"]): r for r in existing.get("rows", [])
    }
    for r in new_rows:
        merged_map[(r["label"], r["workers"])] = r

    all_rows = list(merged_map.values())
    serial_wall = next(
        (r["wall_s"] for r in all_rows if r["executor"] == "serial" and r["gil_enabled"]),
        all_rows[0]["wall_s"],
    )
    for r in all_rows:
        r["speedup"] = round(serial_wall / r["wall_s"], 2)

    merged_extra = {
        k: v for k, v in existing.items() if k not in {"cpu_model", "logical_cores", "rows"}
    }
    merged_extra.update(extra)

    payload: dict[str, Any] = {
        "cpu_model": _cpu_model(),
        "logical_cores": os.cpu_count(),
        "rows": all_rows,
        **merged_extra,
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\nSaved JSON -> {RESULTS_JSON} (skipping plot: matplotlib not installed)")
        return

    fig: Any
    ax: Any
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=150)  # type: ignore[no-untyped-call]
    max_w = max(int(r["workers"]) for r in all_rows)
    ideal_x = list(range(1, max_w + 1))
    ax.plot(ideal_x, ideal_x, "--", color="gray", label="Ideal linear speedup")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in all_rows:
        if r["executor"] == "serial":
            continue
        grouped.setdefault(str(r["label"]), []).append(r)

    for label, grp in grouped.items():
        grp_sorted = sorted(grp, key=lambda x: int(x["workers"]))
        xs = [1] + [int(x["workers"]) for x in grp_sorted]
        ys = [1.0] + [float(x["speedup"]) for x in grp_sorted]
        ax.plot(xs, ys, marker="o", linewidth=2, label=label)

    ax.set_title(f"findex Parallel Indexing Speedup ({_cpu_model()}, {os.cpu_count()} threads)")
    ax.set_xlabel("Workers")
    ax.set_ylabel("Speedup vs. Serial (1 worker)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PNG)
    print(f"\nSaved plot -> {PLOT_PNG}")


def main() -> None:
    corpus = Path("data")
    gil_on = _gil_status()
    print(f"Python {sys.version.split()[0]} | sys._is_gil_enabled() = {gil_on}")
    print(f"CPU: {_cpu_model()} | Logical cores: {os.cpu_count()}\n")

    rows = run_benchmark(corpus, repeats=3)
    race = run_race_demo(corpus)
    io_bench = run_io_counter_benchmark(corpus)

    print("\nRace condition check:", race)
    print("I/O-bound counter-benchmark:", io_bench)
    save_and_plot(
        rows,
        extra={
            f"race_gil_{gil_on}": race,
            f"io_bench_gil_{gil_on}": io_bench,
        },
    )


if __name__ == "__main__":
    main()
