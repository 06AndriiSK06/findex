# `findex` — Full-Text Search Engine in Python

![CI](https://github.com/06AndriiSK06/findex/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12%2B%20%7C%203.13t-blue)
![Typing](https://img.shields.io/badge/pyright-strict-brightgreen)
![Tests](https://img.shields.io/badge/pytest-43%20passed-brightgreen)

`findex` is a modular, strictly typed full-text search engine built in Python across Labs 01–05. It supports deterministic tokenization, positional inverted indexing, binary serialization (`array('I')` + `zlib`), boolean and phrase queries, TF-IDF and BM25 ranking, a `typer` + `rich` CLI, and parallel Map-Reduce index construction (`serial`, `threads`, `processes`).

---

## Installation

Install directly from the release wheel or run locally via `uv`:

```bash
uv tool install [https://github.com/06AndriiSK06/findex/releases/download/v0.4.0/findex-0.4.0-py3-none-any.whl](https://github.com/06AndriiSK06/findex/releases/download/v0.4.0/findex-0.4.0-py3-none-any.whl)
findex --help
```
