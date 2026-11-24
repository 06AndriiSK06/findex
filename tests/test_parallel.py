from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.corpus import collect_document_paths
from findex.index import build_index, build_partial, merge
from findex.store import save_array_binary

runner = CliRunner()


def test_serial_partial_merge_matches_lab04_index(mini_corpus_dir: Path) -> None:
    all_paths = collect_document_paths(mini_corpus_dir)
    via_partial = merge([build_partial(all_paths, start_doc_id=0, with_positions=True)])
    via_serial = build_index(
        mini_corpus_dir,
        with_positions=True,
        workers=1,
        executor="serial",
    )
    assert via_partial.postings == via_serial.postings
    assert via_partial.doc_lengths == via_serial.doc_lengths
    assert via_partial.doc_meta == via_serial.doc_meta


def test_parallel_executors_produce_byte_identical_indexes(
    mini_corpus_dir: Path,
    tmp_path: Path,
) -> None:
    idx_serial = build_index(
        mini_corpus_dir,
        with_positions=True,
        workers=1,
        executor="serial",
    )
    idx_threads = build_index(
        mini_corpus_dir,
        with_positions=True,
        workers=3,
        executor="threads",
    )
    idx_procs = build_index(
        mini_corpus_dir,
        with_positions=True,
        workers=3,
        executor="processes",
    )

    assert idx_threads.postings == idx_serial.postings
    assert idx_threads.doc_lengths == idx_serial.doc_lengths
    assert idx_threads.doc_meta == idx_serial.doc_meta

    assert idx_procs.postings == idx_serial.postings
    assert idx_procs.doc_lengths == idx_serial.doc_lengths
    assert idx_procs.doc_meta == idx_serial.doc_meta

    p_serial = tmp_path / "serial.bin"
    p_threads = tmp_path / "threads.bin"
    p_procs = tmp_path / "procs.bin"
    save_array_binary(idx_serial, p_serial)
    save_array_binary(idx_threads, p_threads)
    save_array_binary(idx_procs, p_procs)

    assert p_serial.read_bytes() == p_threads.read_bytes() == p_procs.read_bytes()


def test_worker_exception_surfaces_loudly(tmp_path: Path) -> None:
    good = tmp_path / "good.txt"
    good.write_text("valid text", encoding="utf-8")
    missing = tmp_path / "missing_file.txt"

    with (
        pytest.raises(FileNotFoundError),
        ProcessPoolExecutor(max_workers=2) as pool,
    ):
        fut = pool.submit(build_partial, [good, missing], start_doc_id=0)
        _ = fut.result()


def test_cli_index_executors_and_workers(mini_corpus_dir: Path, tmp_path: Path) -> None:
    for executor in ("serial", "threads", "processes"):
        out = tmp_path / f"{executor}.bin"
        res = runner.invoke(
            app,
            [
                "index",
                str(mini_corpus_dir),
                "--out",
                str(out),
                "--positions",
                "--workers",
                "2",
                "--executor",
                executor,
            ],
        )
        assert res.exit_code == 0
        assert out.exists()