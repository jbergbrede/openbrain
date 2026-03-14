from __future__ import annotations

import asyncio
from uuid import UUID

import asyncpg

from .config import Settings
from .embeddings import EmbeddingProvider
from .models import ChunkSearchResult, Memory, SearchResult
from . import repository


def get_weights(query: str) -> tuple[float, float]:
    """Returns (semantic_weight, keyword_weight) based on word count."""
    word_count = len(query.strip().split())
    if word_count <= 2:
        return 0.2, 0.8
    elif word_count <= 5:
        return 0.5, 0.5
    else:
        return 0.7, 0.3


def detect_low_spread(scores: list[float], threshold: float = 0.05) -> bool:
    """Returns True if scores are too tightly clustered (low signal).
    Requires at least 2 scores — a single result has no clustering to detect."""
    if len(scores) < 2:
        return False
    return (max(scores) - min(scores)) < threshold


def _promote_chunks_to_memories(
    chunk_results: list[ChunkSearchResult],
) -> list[SearchResult]:
    """MAX similarity per memory; track top-scoring chunk per memory."""
    best: dict[UUID, tuple[float, ChunkSearchResult]] = {}
    for cr in chunk_results:
        mid = cr.memory.id
        if mid not in best or cr.similarity > best[mid][0]:
            best[mid] = (cr.similarity, cr)

    promoted = [
        SearchResult(
            memory=cr.memory,
            similarity=sim,
            score=sim,
            chunk_content=cr.chunk.content,
            chunk_id=cr.chunk.id,
        )
        for sim, cr in best.values()
    ]
    promoted.sort(key=lambda r: r.similarity, reverse=True)
    return promoted


def rrf_merge(
    semantic_results: list[SearchResult],
    keyword_results: list[SearchResult],
    weights: tuple[float, float],
    k: int = 60,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion with connection-count boost."""
    semantic_weight, keyword_weight = weights
    scores: dict[UUID, float] = {}
    memory_map: dict[UUID, Memory] = {}
    chunk_map: dict[UUID, tuple[str | None, UUID | None]] = {}

    for rank, result in enumerate(semantic_results):
        mid = result.memory.id
        scores[mid] = scores.get(mid, 0.0) + semantic_weight / (k + rank + 1)
        memory_map[mid] = result.memory
        chunk_map[mid] = (result.chunk_content, result.chunk_id)

    for rank, result in enumerate(keyword_results):
        mid = result.memory.id
        scores[mid] = scores.get(mid, 0.0) + keyword_weight / (k + rank + 1)
        if mid not in memory_map:
            memory_map[mid] = result.memory
        # Don't overwrite chunk_map if semantic leg already set it

    for mid, memory in memory_map.items():
        conn_boost = 1.0 + len(memory.connections) * 0.01
        scores[mid] *= conn_boost

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    chunk_content, chunk_id = None, None
    return [
        SearchResult(
            memory=memory_map[mid],
            similarity=score,
            score=score,
            chunk_content=chunk_map.get(mid, (None, None))[0],
            chunk_id=chunk_map.get(mid, (None, None))[1],
        )
        for mid, score in ranked
    ]


async def hybrid_search(
    pool: asyncpg.Pool,
    embedder: EmbeddingProvider,
    query: str,
    settings: Settings,
    limit: int = 10,
) -> list[SearchResult]:
    weights = get_weights(query)

    embedding, keyword_results = await asyncio.gather(
        embedder.embed(query),
        repository.keyword_search_memories(pool, query, limit=limit),
    )

    chunk_results = await repository.search_chunks(
        pool=pool,
        embedding=embedding,
        limit=limit * 3,
        threshold=settings.search.similarity_threshold,
    )

    semantic_results = _promote_chunks_to_memories(chunk_results)

    if settings.search.adaptive_weights and detect_low_spread(
        [r.similarity for r in semantic_results],
        threshold=settings.search.score_spread_threshold,
    ):
        effective_semantic: list[SearchResult] = []
        weights = (0.0, 1.0)
    else:
        effective_semantic = semantic_results

    merged = rrf_merge(
        effective_semantic,
        keyword_results,
        weights=weights,
        k=settings.search.rrf_k,
    )
    return merged[:limit]
