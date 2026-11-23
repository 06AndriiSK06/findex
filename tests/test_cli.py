from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    assert "index" in res.stdout
    assert "search" in res.stdout
    assert "stats" in res.stdout


def test_cli_index_and_search_table(mini_corpus_dir: Path, tmp_path: Path) -> None:
    out_bin = tmp_path / "cli_idx.bin"
    res_idx = runner.invoke(
        app,
        ["-v", "index", str(mini_corpus_dir), "--out", str(out_bin), "--positions"],
    )
    assert res_idx.exit_code == 0
    assert out_bin.exists()

    res_search = runner.invoke(
        app,
        ["-vv", "search", str(out_bin), 'python AND (async OR await) "event loop"'],
    )
    assert res_search.exit_code == 0
    assert "async intro" in res_search.stdout


def test_cli_search_json_output_is_clean_for_piping(saved_index_path: Path) -> None:
    res = runner.invoke(
        app,
        ["-vv", "search", str(saved_index_path), "generators", "--json"],
    )
    assert res.exit_code == 0
    lines = [line for line in res.stdout.strip().splitlines() if line.strip()]
    assert len(lines) >= 1
    first_obj = json.loads(lines[0])
    assert first_obj["title"] == "generators"
    assert "score" in first_obj


def test_cli_stats_on_corpus_and_index(mini_corpus_dir: Path, saved_index_path: Path) -> None:
    res_corpus = runner.invoke(app, ["stats", str(mini_corpus_dir), "--top", "5"])
    assert res_corpus.exit_code == 0
    assert "Documents" in res_corpus.stdout

    res_idx = runner.invoke(app, ["stats", str(saved_index_path), "--top", "5"])
    assert res_idx.exit_code == 0
    assert "Vocabulary size" in res_idx.stdout


@pytest.mark.parametrize(
    "args",
    [
        ["search", "nonexistent_index.bin", "python"],
        ["index", "nonexistent_corpus_dir"],
        ["stats", "missing_path"],
    ],
)
def test_cli_expected_errors_exit_nonzero_without_traceback(args: list[str]) -> None:
    res = runner.invoke(app, args)
    assert res.exit_code == 1
    assert "Error:" in res.stderr
    assert "Traceback" not in res.output


def test_cli_broken_query_syntax_exits_nonzero(saved_index_path: Path) -> None:
    res = runner.invoke(app, ["search", str(saved_index_path), "(unclosed AND python"])
    assert res.exit_code == 1
    assert "Error:" in res.stderr
    assert "Traceback" not in res.output
