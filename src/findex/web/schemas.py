from __future__ import annotations

from pydantic import BaseModel, Field


class SearchResultItem(BaseModel):
    doc_id: int
    title: str
    score: float
    snippet: str = ""
    path: str = ""


class SearchResponse(BaseModel):
    query: str
    scorer: str
    page: int = Field(ge=1)
    k: int = Field(ge=1)
    total: int = Field(ge=0)
    total_pages: int = Field(ge=0)
    took_ms: float = Field(ge=0.0)
    results: list[SearchResultItem]


class DocumentResponse(BaseModel):
    doc_id: int
    title: str
    path: str
    token_count: int
    content: str


class StatsResponse(BaseModel):
    num_docs: int
    vocab_size: int
    total_tokens: int
    avg_doc_length: float
    with_positions: bool
    index_size_bytes: int
    uptime_seconds: float


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    num_docs: int


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None
