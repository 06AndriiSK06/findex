# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from findex.index import DocMeta, Index, Posting
from findex.web.app import create_app
from findex.web.config import Settings
from findex.web.deps import get_index


def _json_dict(resp: Any) -> dict[str, Any]:
    return cast(dict[str, Any], resp.json())


def _build_fixture_index(tmp_path: Path) -> Index:
    docs_text = {
        0: ("Asyncio Guide", "Asyncio event loop tasks and coroutines in Python web servers."),
        1: ("FastAPI Overview", "FastAPI uses Pydantic validation and ASGI uvicorn workers."),
        2: ("Python Typing", "Type hints and dataclasses make Python code reliable and fast."),
        3: ("Search Ranking", "BM25 and TF-IDF ranking models score Python search queries."),
        4: ("Docker Deployment", "Multi-stage Docker containers run FastAPI apps in production."),
    }
    postings: dict[str, list[Posting]] = {}
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}

    for doc_id, (title, text) in docs_text.items():
        file_path = tmp_path / f"doc_{doc_id}.txt"
        file_path.write_text(text, encoding="utf-8")
        tokens = [t.strip(".,").lower() for t in text.split()]
        doc_lengths[doc_id] = len(tokens)
        doc_meta[doc_id] = DocMeta(doc_id=doc_id, path=str(file_path), title=title)
        counts: dict[str, int] = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1
        for tok, tf in counts.items():
            postings.setdefault(tok, []).append(Posting(doc_id=doc_id, tf=tf))

    return Index(postings=postings, doc_lengths=doc_lengths, doc_meta=doc_meta)


@pytest.fixture
def fixture_index(tmp_path: Path) -> Index:
    return _build_fixture_index(tmp_path)


@pytest.fixture
def client(fixture_index: Index) -> TestClient:
    app = create_app(
        Settings(index_path=Path("nonexistent_fixture.bin")),
        load_on_startup=False,
    )
    app.state.index = fixture_index
    app.state.index_size_bytes = 2048
    app.dependency_overrides[get_index] = lambda: fixture_index
    return TestClient(app)


def test_health_unloaded_returns_503() -> None:
    app = create_app(
        Settings(index_path=Path("does_not_exist.bin")),
        load_on_startup=False,
    )
    with TestClient(app) as tc:
        resp: httpx.Response = tc.get("/health")
    assert resp.status_code == 503
    data = _json_dict(resp)
    assert data["status"] == "unavailable"
    assert data["index_loaded"] is False


def test_health_loaded_returns_200(client: TestClient) -> None:
    resp: httpx.Response = client.get("/health")
    assert resp.status_code == 200
    assert _json_dict(resp) == {"status": "ok", "index_loaded": True, "num_docs": 5}


def test_search_success_and_pagination(client: TestClient) -> None:
    resp_p1: httpx.Response = client.get(
        "/search", params={"q": "python", "k": 2, "page": 1, "scorer": "bm25"}
    )
    assert resp_p1.status_code == 200
    body_p1 = _json_dict(resp_p1)
    assert body_p1["query"] == "python"
    assert body_p1["scorer"] == "bm25"
    assert body_p1["total"] == 3
    assert body_p1["total_pages"] == 2
    assert body_p1["page"] == 1
    p1_results = cast(list[dict[str, Any]], body_p1["results"])
    assert len(p1_results) == 2
    assert "[Python]" in str(p1_results[0]["snippet"])

    resp_p2: httpx.Response = client.get(
        "/search", params={"q": "python", "k": 2, "page": 2, "scorer": "bm25"}
    )
    assert resp_p2.status_code == 200
    body_p2 = _json_dict(resp_p2)
    assert body_p2["page"] == 2
    p2_results = cast(list[dict[str, Any]], body_p2["results"])
    assert len(p2_results) == 1
    p1_ids = {int(r["doc_id"]) for r in p1_results}
    assert int(p2_results[0]["doc_id"]) not in p1_ids

    resp_tfidf: httpx.Response = client.get(
        "/search", params={"q": "python", "k": 5, "scorer": "tfidf"}
    )
    assert resp_tfidf.status_code == 200
    assert _json_dict(resp_tfidf)["scorer"] == "tfidf"


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"q": ""},
        {"q": "python", "k": "abc"},
        {"q": "python", "k": 0},
        {"q": "python", "k": 500},
        {"q": "python", "page": 0},
        {"q": "python", "scorer": "invalid_scorer"},
        {"q": "(unclosed"},
    ],
)
def test_search_validation_422(client: TestClient, params: dict[str, object]) -> None:
    resp: httpx.Response = client.get("/search", params=params)
    assert resp.status_code == 422


def test_get_document_success_and_404(client: TestClient) -> None:
    ok_resp: httpx.Response = client.get("/docs/1")
    assert ok_resp.status_code == 200
    doc = _json_dict(ok_resp)
    assert doc["doc_id"] == 1
    assert doc["title"] == "FastAPI Overview"
    assert "Pydantic validation" in str(doc["content"])

    missing_resp: httpx.Response = client.get("/docs/9999")
    assert missing_resp.status_code == 404
    assert "not found" in str(_json_dict(missing_resp)["detail"]).lower()


def test_stats_endpoint(client: TestClient) -> None:
    resp: httpx.Response = client.get("/stats")
    assert resp.status_code == 200
    data = _json_dict(resp)
    assert data["num_docs"] == 5
    assert int(data["vocab_size"]) > 10
    assert data["index_size_bytes"] == 2048
    assert float(data["uptime_seconds"]) >= 0.0


def test_unhandled_exception_returns_clean_500_with_request_id(fixture_index: Index) -> None:
    app = create_app(Settings(index_path=Path("none.bin")), load_on_startup=False)
    app.dependency_overrides[get_index] = lambda: fixture_index

    @app.get("/boom")
    async def boom(_idx: Index = Depends(get_index)) -> dict[str, str]:  # noqa: B008
        raise RuntimeError("secret internal failure details")

    with TestClient(app, raise_server_exceptions=False) as tc:
        resp: httpx.Response = tc.get("/boom", headers={"X-Request-ID": "req-test-123"})

    assert resp.status_code == 500
    assert resp.headers["X-Request-ID"] == "req-test-123"
    body = _json_dict(resp)
    assert body == {"detail": "Internal server error", "request_id": "req-test-123"}
    assert "RuntimeError" not in str(resp.text)


def test_web_ui_routes_and_static_css(client: TestClient) -> None:
    home_resp: httpx.Response = client.get("/", params={"q": "python", "k": 2, "page": 1})
    assert home_resp.status_code == 200
    home_html = str(home_resp.text)
    assert "<mark>Python</mark>" in home_html
    assert "Next →" in home_html

    doc_resp: httpx.Response = client.get("/doc/0", params={"q": "python"})
    assert doc_resp.status_code == 200
    doc_html = str(doc_resp.text)
    assert "Asyncio Guide" in doc_html
    assert "coroutines" in doc_html

    css_resp: httpx.Response = client.get("/static/style.css")
    assert css_resp.status_code == 200
    assert ".result-card" in str(css_resp.text)


async def test_sync_vs_async_concurrency_20_requests() -> None:
    work_delay = 0.025

    blocking_app = FastAPI()

    @blocking_app.get("/search")
    async def blocking_search() -> dict[str, str]:
        time.sleep(work_delay)
        return {"mode": "blocking_async_def"}

    threaded_app = FastAPI()

    @threaded_app.get("/search")
    def threaded_search() -> dict[str, str]:
        time.sleep(work_delay)
        return {"mode": "threadpool_def"}

    async def _burst(target_app: FastAPI, n: int = 20) -> float:
        transport = httpx.ASGITransport(app=target_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            t0 = time.perf_counter()
            responses = await asyncio.gather(*(ac.get("/search") for _ in range(n)))
            elapsed = time.perf_counter() - t0
        assert all(r.status_code == 200 for r in responses)
        return elapsed

    blocking_elapsed = await _burst(blocking_app, n=20)
    threaded_elapsed = await _burst(threaded_app, n=20)

    assert blocking_elapsed >= 0.45
    assert threaded_elapsed < blocking_elapsed * 0.5
