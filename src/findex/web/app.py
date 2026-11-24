from __future__ import annotations

import asyncio
import html
import logging
import math
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup

from findex.index import Index
from findex.query import QuerySyntaxError, parse_query
from findex.ranking import ScorerName, make_scorer
from findex.search import search as run_search
from findex.store import load_index
from findex.web.config import Settings
from findex.web.deps import get_index, get_settings, get_start_time
from findex.web.schemas import (
    DocumentResponse,
    ErrorResponse,
    HealthResponse,
    SearchResponse,
    SearchResultItem,
    StatsResponse,
)

log = logging.getLogger(__name__)

_WEB_DIR = Path(__file__).resolve().parent
_TEMPLATES_DIR = _WEB_DIR / "templates"
_STATIC_DIR = _WEB_DIR / "static"


def _configure_web_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=False,
    )


def _highlight_snippet_html(snippet: str) -> Markup:
    parts = snippet.split("[")
    out: list[str] = [html.escape(parts[0])]
    for part in parts[1:]:
        if "]" in part:
            term, rest = part.split("]", 1)
            out.append(f"<mark>{html.escape(term)}</mark>{html.escape(rest)}")
        else:
            out.append(html.escape("[" + part))
    return Markup("".join(out))


def _read_document_content(doc_path: str) -> str:
    if doc_path.startswith(("http://", "https://")):
        return f"External crawled URL: {doc_path}"
    path = Path(doc_path)
    if not path.exists() or not path.is_file():
        return "(Full source file is not mounted on disk; indexed metadata only.)"
    return path.read_text(encoding="utf-8", errors="replace")


def _execute_search_sync(
    index: Index,
    query: str,
    scorer_name: ScorerName,
    k: int,
    page: int,
) -> SearchResponse:
    t0 = time.perf_counter()
    try:
        tree = parse_query(query)
        matched_ids = tree.evaluate(index)
    except QuerySyntaxError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Invalid search query: {exc}",
        ) from exc

    total = len(matched_ids)
    total_pages = math.ceil(total / k) if total > 0 else 0
    fetch_limit = min(total, page * k) if total > 0 else k

    scorer = make_scorer(scorer_name)
    raw_results = run_search(
        index,
        query,
        scorer=scorer,
        k=max(1, fetch_limit),
        with_snippets=True,
    )

    offset = (page - 1) * k
    page_slice = raw_results[offset : offset + k]

    items: list[SearchResultItem] = []
    for item in page_slice:
        meta = index.doc_meta.get(item.doc_id)
        doc_path = meta.path if meta is not None else ""
        snippet = item.snippet or (f"Source: {doc_path}" if doc_path else "")
        items.append(
            SearchResultItem(
                doc_id=item.doc_id,
                title=item.title,
                score=round(item.score, 6),
                snippet=snippet,
                path=doc_path,
            )
        )

    took_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return SearchResponse(
        query=query,
        scorer=scorer_name,
        page=page,
        k=k,
        total=total,
        total_pages=total_pages,
        took_ms=took_ms,
        results=items,
    )


def create_app(settings: Settings | None = None, *, load_on_startup: bool = True) -> FastAPI:
    active_settings = settings if settings is not None else Settings()
    _configure_web_logging(active_settings.log_level)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
        application.state.settings = active_settings
        application.state.started_at = time.perf_counter()
        application.state.index = None
        application.state.index_size_bytes = 0

        if load_on_startup:
            idx_path = active_settings.index_path
            if idx_path.exists():
                application.state.index = load_index(idx_path)
                application.state.index_size_bytes = idx_path.stat().st_size
                log.info(
                    "Loaded index from %s (%d docs, %d terms)",
                    idx_path,
                    application.state.index.num_docs,
                    len(application.state.index),
                )
            else:
                log.warning(
                    "Index path %s does not exist at startup; /health will return 503",
                    idx_path,
                )

        try:
            yield
        finally:
            loaded: Index | None = getattr(application.state, "index", None)
            if loaded is not None and not loaded.closed:
                loaded.close()
                log.info("Closed search index on shutdown")

    application = FastAPI(
        title="findex Search API",
        description="Full-text search engine HTTP API and web interface.",
        version="0.7.0",
        lifespan=lifespan,
    )
    application.state.settings = active_settings
    application.state.started_at = time.perf_counter()
    application.state.index = None
    application.state.index_size_bytes = 0

    application.add_middleware(
        CORSMiddleware,
        allow_origins=active_settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    templates.env.filters["highlight_snippet"] = _highlight_snippet_html
    application.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @application.middleware("http")
    async def request_id_and_error_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = req_id
        try:
            response = await call_next(request)
        except Exception:
            log.exception(
                "Unhandled error while processing request_id=%s path=%s",
                req_id,
                request.url.path,
            )
            payload = ErrorResponse(detail="Internal server error", request_id=req_id)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=payload.model_dump(),
                headers={"X-Request-ID": req_id},
            )
        response.headers["X-Request-ID"] = req_id
        return response

    @application.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health(request: Request) -> HealthResponse | JSONResponse:
        idx: Index | None = getattr(request.app.state, "index", None)
        if idx is None or idx.closed:
            payload = HealthResponse(status="unavailable", index_loaded=False, num_docs=0)
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=payload.model_dump(),
            )
        return HealthResponse(status="ok", index_loaded=True, num_docs=idx.num_docs)

    @application.get("/stats", response_model=StatsResponse, tags=["ops"])
    async def stats(
        request: Request,
        index: Annotated[Index, Depends(get_index)],
        started_at: Annotated[float, Depends(get_start_time)],
    ) -> StatsResponse:
        total_tokens = sum(index.doc_lengths.values())
        size_bytes = int(getattr(request.app.state, "index_size_bytes", 0))
        uptime = round(max(0.0, time.perf_counter() - started_at), 3)
        return StatsResponse(
            num_docs=index.num_docs,
            vocab_size=len(index),
            total_tokens=total_tokens,
            avg_doc_length=round(index.avg_doc_length, 2),
            with_positions=index.with_positions,
            index_size_bytes=size_bytes,
            uptime_seconds=uptime,
        )

    @application.get("/search", response_model=SearchResponse, tags=["search"])
    async def api_search(
        q: Annotated[str, Query(min_length=1, max_length=200, description="Search query")],
        k: Annotated[int, Query(ge=1, le=100, description="Results per page")] = 10,
        scorer: Annotated[Literal["bm25", "tfidf"], Query(description="Ranking scorer")] = "bm25",
        page: Annotated[int, Query(ge=1, le=1000, description="1-based page number")] = 1,
        index: Annotated[Index, Depends(get_index)] = ...,  # type: ignore[assignment]
        _settings: Annotated[Settings, Depends(get_settings)] = ...,  # type: ignore[assignment]
    ) -> SearchResponse:
        return await asyncio.to_thread(_execute_search_sync, index, q, scorer, k, page)

    @application.get("/docs/{doc_id}", response_model=DocumentResponse, tags=["documents"])
    async def api_get_document(
        doc_id: int,
        index: Annotated[Index, Depends(get_index)],
    ) -> DocumentResponse:
        meta = index.doc_meta.get(doc_id)
        if meta is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {doc_id} not found",
            )
        content = await asyncio.to_thread(_read_document_content, meta.path)
        return DocumentResponse(
            doc_id=meta.doc_id,
            title=meta.title,
            path=meta.path,
            token_count=index.doc_lengths.get(doc_id, 0),
            content=content,
        )

    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def ui_home(
        request: Request,
        q: Annotated[str | None, Query(max_length=200)] = None,
        k: Annotated[int, Query(ge=1, le=50)] = 10,
        scorer: Annotated[Literal["bm25", "tfidf"], Query()] = "bm25",
        page: Annotated[int, Query(ge=1, le=1000)] = 1,
    ) -> HTMLResponse:
        idx: Index | None = getattr(request.app.state, "index", None)
        search_res: SearchResponse | None = None
        error_msg: str | None = None

        if q is not None and q.strip():
            if idx is None or idx.closed:
                error_msg = "Search index is currently unavailable."
            else:
                try:
                    search_res = await asyncio.to_thread(
                        _execute_search_sync,
                        idx,
                        q.strip(),
                        scorer,
                        k,
                        page,
                    )
                except HTTPException as exc:
                    error_msg = str(exc.detail)

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "query": q or "",
                "k": k,
                "scorer": scorer,
                "page": page,
                "search_res": search_res,
                "error_msg": error_msg,
                "num_docs": idx.num_docs if idx is not None and not idx.closed else 0,
                "vocab_size": len(idx) if idx is not None and not idx.closed else 0,
            },
        )

    @application.get("/doc/{doc_id}", response_class=HTMLResponse, include_in_schema=False)
    async def ui_document(
        request: Request,
        doc_id: int,
        q: Annotated[str, Query(max_length=200)] = "",
        index: Annotated[Index, Depends(get_index)] = ...,  # type: ignore[assignment]
    ) -> HTMLResponse:
        meta = index.doc_meta.get(doc_id)
        if meta is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {doc_id} not found",
            )
        content = await asyncio.to_thread(_read_document_content, meta.path)
        return templates.TemplateResponse(
            request=request,
            name="doc.html",
            context={
                "doc": meta,
                "token_count": index.doc_lengths.get(doc_id, 0),
                "content": content,
                "query": q,
            },
        )

    return application


app = create_app()
