from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
import numpy.typing as npt

from findex.corpus import iter_documents
from findex.index import Index, Int32Array
from findex.query import QuerySyntaxError
from findex.ranking import Scorer, SearchResult, extract_snippet, make_scorer
from findex.search import search as keyword_search

log = logging.getLogger(__name__)

SearchMode = Literal["keyword", "semantic", "hybrid"]
Float32Array = npt.NDArray[np.float32]


class TextEncoder(Protocol):
    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> Float32Array: ...


class SentenceTransformerEncoder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model: Any = SentenceTransformer(model_name)

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> Float32Array:
        if not sentences:
            return np.empty((0, 0), dtype=np.float32)
        raw = self._model.encode(
            list(sentences),
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
        return np.asarray(raw, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class SemanticIndex:
    embeddings: Float32Array
    chunk_doc_ids: Int32Array
    model_name: str = "all-MiniLM-L6-v2"

    @property
    def num_chunks(self) -> int:
        return int(self.embeddings.shape[0]) if self.embeddings.ndim == 2 else 0

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1]) if self.embeddings.ndim == 2 else 0


def embedding_paths(index_path: Path) -> tuple[Path, Path, Path]:
    base = index_path.with_suffix("")
    return (
        Path(f"{base}.emb.npy"),
        Path(f"{base}.emb_docs.npy"),
        Path(f"{base}.emb_meta.json"),
    )


def save_semantic_index(sem: SemanticIndex, index_path: Path) -> tuple[Path, Path]:
    emb_path, docs_path, meta_path = embedding_paths(index_path)
    np.save(emb_path, sem.embeddings.astype(np.float32, copy=False))
    np.save(docs_path, sem.chunk_doc_ids.astype(np.int32, copy=False))
    meta_path.write_text(
        json.dumps(
            {
                "model_name": sem.model_name,
                "num_chunks": sem.num_chunks,
                "dim": sem.dim,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return emb_path, docs_path


def load_semantic_index(index_path: Path) -> SemanticIndex | None:
    emb_path, docs_path, meta_path = embedding_paths(index_path)
    if not emb_path.exists() or not docs_path.exists():
        return None
    embeddings = cast(Float32Array, np.load(emb_path).astype(np.float32, copy=False))
    chunk_doc_ids = cast(Int32Array, np.load(docs_path).astype(np.int32, copy=False))
    model_name = "all-MiniLM-L6-v2"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        model_name = str(meta.get("model_name", model_name))
    return SemanticIndex(
        embeddings=embeddings,
        chunk_doc_ids=chunk_doc_ids,
        model_name=model_name,
    )


def chunk_text(text: str, *, chunk_words: int = 220, overlap_words: int = 40) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [" ".join(words)]
    step = max(1, chunk_words - overlap_words)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        piece = words[start : start + chunk_words]
        if piece:
            chunks.append(" ".join(piece))
        if start + chunk_words >= len(words):
            break
    return chunks


def l2_normalize(vectors: Float32Array) -> Float32Array:
    if vectors.size == 0:
        return vectors.astype(np.float32, copy=False)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    safe_norms = np.where(norms > 1e-12, norms, 1.0).astype(np.float32)
    return (vectors / safe_norms).astype(np.float32, copy=False)


def _collect_texts_for_index(index: Index, corpus_path: Path | None = None) -> dict[int, str]:
    texts_by_doc: dict[int, str] = {}
    if corpus_path is not None and corpus_path.exists():
        for doc_id, doc in enumerate(iter_documents(corpus_path)):
            if doc_id in index.doc_meta:
                title = doc.title or index.doc_meta[doc_id].title
                texts_by_doc[doc_id] = f"{title}. {doc.text}".strip()

    for doc_id, meta in index.doc_meta.items():
        if doc_id in texts_by_doc:
            continue
        p = Path(meta.path)
        if p.exists() and p.is_file():
            raw = p.read_text(encoding="utf-8", errors="replace")
            texts_by_doc[doc_id] = f"{meta.title}. {raw}".strip()
        else:
            texts_by_doc[doc_id] = meta.title
    return texts_by_doc


def build_semantic_index(
    index: Index,
    *,
    encoder: TextEncoder,
    model_name: str = "all-MiniLM-L6-v2",
    corpus_path: Path | None = None,
    chunk_words: int = 220,
    overlap_words: int = 40,
    batch_size: int = 32,
    on_batch: Callable[[int, int], None] | None = None,
) -> SemanticIndex:
    texts_by_doc = _collect_texts_for_index(index, corpus_path=corpus_path)
    all_chunks: list[str] = []
    doc_ids: list[int] = []

    for doc_id in sorted(index.doc_meta.keys()):
        doc_text = texts_by_doc.get(doc_id, "")
        chunks = chunk_text(doc_text, chunk_words=chunk_words, overlap_words=overlap_words)
        if not chunks and index.doc_meta[doc_id].title:
            chunks = [index.doc_meta[doc_id].title]
        for ch in chunks:
            all_chunks.append(ch)
            doc_ids.append(doc_id)

    total = len(all_chunks)
    if total == 0:
        return SemanticIndex(
            embeddings=np.empty((0, 0), dtype=np.float32),
            chunk_doc_ids=np.empty(0, dtype=np.int32),
            model_name=model_name,
        )

    encoded_batches: list[Float32Array] = []
    for start in range(0, total, batch_size):
        batch = all_chunks[start : start + batch_size]
        vec = encoder.encode(batch, batch_size=batch_size, show_progress_bar=False)
        encoded_batches.append(np.asarray(vec, dtype=np.float32))
        if on_batch is not None:
            on_batch(min(start + len(batch), total), total)

    matrix = np.vstack(encoded_batches).astype(np.float32, copy=False)
    normalized = l2_normalize(matrix)
    return SemanticIndex(
        embeddings=normalized,
        chunk_doc_ids=np.asarray(doc_ids, dtype=np.int32),
        model_name=model_name,
    )


def semantic_search(
    index: Index,
    sem_index: SemanticIndex,
    query: str,
    *,
    encoder: TextEncoder,
    k: int = 10,
    with_snippets: bool = True,
) -> list[SearchResult]:
    if k <= 0 or sem_index.num_chunks == 0 or index.doc_lengths_arr.size == 0:
        return []
    cleaned = query.strip()
    if not cleaned:
        return []

    q_raw = encoder.encode([cleaned], batch_size=1, show_progress_bar=False)
    q_norm = l2_normalize(np.asarray(q_raw, dtype=np.float32))[0]

    chunk_scores = sem_index.embeddings @ q_norm

    arr_size = int(index.doc_lengths_arr.size)
    best_doc_scores = np.full(arr_size, -np.inf, dtype=np.float32)
    np.maximum.at(best_doc_scores, sem_index.chunk_doc_ids, chunk_scores)

    valid_docs = np.flatnonzero(np.isfinite(best_doc_scores)).astype(np.int32)
    if valid_docs.size == 0:
        return []

    valid_scores = best_doc_scores[valid_docs]
    top_n = min(k, int(valid_docs.size))
    order = np.lexsort((valid_docs, -valid_scores))[:top_n]

    query_terms = [w.casefold() for w in cleaned.split() if w.isalpha()]
    results: list[SearchResult] = []
    for idx_pos in order.tolist():
        doc_id = int(valid_docs[idx_pos])
        score = float(valid_scores[idx_pos])
        meta = index.doc_meta[doc_id]
        snippet = extract_snippet(meta.path, query_terms) if with_snippets else ""
        if not snippet:
            snippet = f"Source: {meta.path}" if meta.path else meta.title
        results.append(
            SearchResult(
                score=score,
                doc_id=doc_id,
                title=meta.title,
                snippet=snippet,
            )
        )
    return results


def reciprocal_rank_fusion(
    keyword_results: Sequence[SearchResult],
    semantic_results: Sequence[SearchResult],
    *,
    index: Index,
    k: int = 10,
    rrf_k: int = 60,
) -> list[SearchResult]:
    if k <= 0:
        return []

    fused_scores: dict[int, float] = {}
    snippets_by_doc: dict[int, str] = {}

    for rank, item in enumerate(keyword_results, start=1):
        fused_scores[item.doc_id] = fused_scores.get(item.doc_id, 0.0) + 1.0 / (rrf_k + rank)
        if item.snippet:
            snippets_by_doc[item.doc_id] = item.snippet

    for rank, item in enumerate(semantic_results, start=1):
        fused_scores[item.doc_id] = fused_scores.get(item.doc_id, 0.0) + 1.0 / (rrf_k + rank)
        if item.doc_id not in snippets_by_doc and item.snippet:
            snippets_by_doc[item.doc_id] = item.snippet

    ranked = sorted(fused_scores.items(), key=lambda pair: (-pair[1], pair[0]))[:k]
    results: list[SearchResult] = []
    for doc_id, score in ranked:
        meta = index.doc_meta[doc_id]
        results.append(
            SearchResult(
                score=score,
                doc_id=doc_id,
                title=meta.title,
                snippet=snippets_by_doc.get(doc_id, ""),
            )
        )
    return results


def hybrid_search(
    index: Index,
    sem_index: SemanticIndex,
    query: str,
    *,
    encoder: TextEncoder,
    scorer: Scorer | None = None,
    k: int = 10,
    rrf_k: int = 60,
    with_snippets: bool = True,
) -> list[SearchResult]:
    pool_k = max(k * 3, 30)
    try:
        kw_res = keyword_search(
            index,
            query,
            scorer=scorer if scorer is not None else make_scorer("bm25"),
            k=pool_k,
            with_snippets=with_snippets,
        )
    except QuerySyntaxError:
        kw_res = []

    sem_res = semantic_search(
        index,
        sem_index,
        query,
        encoder=encoder,
        k=pool_k,
        with_snippets=with_snippets,
    )
    return reciprocal_rank_fusion(kw_res, sem_res, index=index, k=k, rrf_k=rrf_k)
