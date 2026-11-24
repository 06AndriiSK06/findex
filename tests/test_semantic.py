from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from findex.index import Index, build_index
from findex.ranking import SearchResult
from findex.semantic import (
    Float32Array,
    build_semantic_index,
    chunk_text,
    hybrid_search,
    l2_normalize,
    load_semantic_index,
    reciprocal_rank_fusion,
    save_semantic_index,
    semantic_search,
)
from findex.store import save_array_binary
from findex.web.app import create_app
from findex.web.config import Settings


class FakeSemanticEncoder:
    """Deterministic lightweight encoder mapping concepts to 4D orthogonal axes."""

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> Float32Array:
        rows: list[list[float]] = []
        for s in sentences:
            low = s.casefold()
            # Axis 0: performance / speed / optimization
            v0 = float(
                sum(
                    w in low
                    for w in ("fast", "faster", "speed", "performance", "optimization", "numpy")
                )
            )
            # Axis 1: async / concurrency / event loop
            v1 = float(sum(w in low for w in ("async", "asyncio", "event", "loop", "concurrent")))
            # Axis 2: rotation / quaternion / 3d
            v2 = float(sum(w in low for w in ("quaternion", "rotation", "gimbal", "3d")))
            # Axis 3: fallback baseline
            v3 = 0.1
            rows.append([v0, v1, v2, v3])
        return np.asarray(rows, dtype=np.float32)


def test_chunk_text_and_l2_normalize() -> None:
    words = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_text(words, chunk_words=220, overlap_words=40)
    assert len(chunks) == 3
    assert len(chunks[0].split()) == 220

    mat = np.asarray([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    normed = l2_normalize(mat)
    assert math.isclose(float(normed[0, 0] ** 2 + normed[0, 1] ** 2), 1.0, rel_tol=1e-6)
    assert float(normed[1, 0]) == 0.0 and float(normed[1, 1]) == 0.0


def test_semantic_best_chunk_per_doc_and_persistence(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    # Doc 0 has an irrelevant intro chunk followed by a performance optimization chunk
    intro = " ".join(["general"] * 230)
    tail = "Python performance optimization techniques using vectorization"
    (corpus / "perf_guide.txt").write_text(f"{intro} {tail}", encoding="utf-8")
    (corpus / "rotation.txt").write_text(
        "Quaternion math avoids gimbal lock in 3D", encoding="utf-8"
    )

    idx = build_index(corpus, workers=1)
    idx_path = tmp_path / "test.bin"
    save_array_binary(idx, idx_path)

    encoder = FakeSemanticEncoder()
    sem_idx = build_semantic_index(idx, encoder=encoder, chunk_words=220, overlap_words=40)
    assert sem_idx.num_chunks >= 3

    save_semantic_index(sem_idx, idx_path)
    loaded_sem = load_semantic_index(idx_path)
    assert loaded_sem is not None
    assert loaded_sem.num_chunks == sem_idx.num_chunks

    # Paraphrase query without overlapping words finds perf_guide via concept axis 0
    results = semantic_search(
        idx,
        loaded_sem,
        "how do I make my code faster",
        encoder=encoder,
        k=2,
        with_snippets=False,
    )
    assert results[0].title == "perf guide"


def test_reciprocal_rank_fusion_and_hybrid_search(built_index: Index) -> None:
    encoder = FakeSemanticEncoder()
    sem_idx = build_semantic_index(built_index, encoder=encoder)

    r1 = [SearchResult(10.0, 0, "doc0"), SearchResult(8.0, 1, "doc1")]
    r2 = [SearchResult(0.95, 1, "doc1"), SearchResult(0.80, 2, "doc2")]
    fused = reciprocal_rank_fusion(r1, r2, index=built_index, k=3, rrf_k=60)
    # doc1 appears in both lists (rank 2 + rank 1), so it outranks doc0 (rank 1 only)
    assert fused[0].doc_id == 1
    assert fused[1].doc_id == 0
    assert fused[2].doc_id == 2

    hyb = hybrid_search(
        built_index,
        sem_idx,
        "asyncio event loop",
        encoder=encoder,
        k=3,
        with_snippets=False,
    )
    assert len(hyb) >= 1


def test_web_api_semantic_and_hybrid_modes(tmp_path: Path) -> None:
    corpus = tmp_path / "web_corpus"
    corpus.mkdir()
    (corpus / "speed.txt").write_text(
        "Performance optimization and NumPy speedups", encoding="utf-8"
    )
    (corpus / "async.txt").write_text("Asyncio event loop concurrency", encoding="utf-8")

    idx = build_index(corpus, workers=1)
    idx_path = tmp_path / "web.bin"
    save_array_binary(idx, idx_path)

    encoder = FakeSemanticEncoder()
    sem_idx = build_semantic_index(idx, encoder=encoder)
    save_semantic_index(sem_idx, idx_path)

    app = create_app(Settings(index_path=idx_path), load_on_startup=True)
    app.state.text_encoder = encoder
    with TestClient(app) as client:
        app.state.text_encoder = encoder
        res_sem: Any = client.get("/search", params={"q": "make code faster", "mode": "semantic"})
        assert res_sem.status_code == 200
        body_sem = res_sem.json()
        assert body_sem["mode"] == "semantic"
        assert body_sem["results"][0]["title"] == "speed"

        res_hyb: Any = client.get("/search", params={"q": "async loop", "mode": "hybrid"})
        assert res_hyb.status_code == 200
        assert res_hyb.json()["mode"] == "hybrid"
