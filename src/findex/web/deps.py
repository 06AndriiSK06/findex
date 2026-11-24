from __future__ import annotations

from fastapi import HTTPException, Request, status

from findex.index import Index
from findex.web.config import Settings


def get_settings(request: Request) -> Settings:
    settings: Settings | None = getattr(request.app.state, "settings", None)
    if settings is None:
        settings = Settings()
        request.app.state.settings = settings
    return settings


def get_index(request: Request) -> Index:
    index: Index | None = getattr(request.app.state, "index", None)
    if index is None or index.closed:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search index is not loaded.",
        )
    return index


def get_start_time(request: Request) -> float:
    return float(getattr(request.app.state, "started_at", 0.0))
