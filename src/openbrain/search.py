from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import UUID

import asyncpg

from . import repository
from .config import Settings
from .embeddings import EmbeddingProvider
from .models import ChunkSearchResult, Memory, SearchResult


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
    """MAX similarity per memory; track top-scoring chunk per memory.
    Synthetic chunks participate in scoring but the returned chunk_content
    comes from the best real (non-synthetic) chunk."""
    best_score: dict[UUID, float] = {}
    best_real: dict[UUID, ChunkSearchResult] = {}
    memory_map: dict[UUID, Memory] = {}

    for cr in chunk_results:
        mid = cr.memory.id
        memory_map[mid] = cr.memory

        # Track overall best score (synthetic or not)
        if mid not in best_score or cr.similarity > best_score[mid]:
            best_score[mid] = cr.similarity

        # Track best non-synthetic chunk for display
        if not cr.chunk.is_synthetic:
            if mid not in best_real or cr.similarity > best_real[mid].similarity:
                best_real[mid] = cr

    promoted = []
    for mid, sim in best_score.items():
        real = best_real.get(mid)
        promoted.append(
            SearchResult(
                memory=memory_map[mid],
                similarity=sim,
                score=sim,
                chunk_content=real.chunk.content if real else None,
                chunk_id=real.chunk.id if real else None,
            )
        )
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


@dataclass
class SearchDebugInfo:
    query: str
    weights: tuple[float, float]
    effective_weights: tuple[float, float]
    low_spread_detected: bool
    semantic_hits: list[dict] = field(default_factory=list)
    keyword_hits: list[dict] = field(default_factory=list)
    top_semantic_below_threshold: dict | None = None
    expanded_query: str | None = None


async def hybrid_search(
    pool: asyncpg.Pool,
    embedder: EmbeddingProvider,
    query: str,
    settings: Settings,
    limit: int = 10,
    debug: bool = False,
) -> list[SearchResult] | tuple[list[SearchResult], SearchDebugInfo]:
    # Adaptive weighting always uses the ORIGINAL query length
    weights = get_weights(query)

    # Optionally expand the query for the semantic leg
    expanded_query = None
    if settings.search.query_expansion:
        from .query_expansion import expand_for_semantic

        expanded_query = await expand_for_semantic(query, model=settings.search.query_expansion_model)
        semantic_query = expanded_query
    else:
        semantic_query = query

    # Keyword leg always uses original query; semantic leg uses expanded query
    embedding, keyword_results = await asyncio.gather(
        embedder.embed(semantic_query),
        repository.keyword_search_memories(pool, query, limit=limit),
    )

    # In debug mode, fetch all results to find top below-threshold
    threshold_for_query = 0.0 if debug else settings.search.similarity_threshold
    all_chunk_results = await repository.search_chunks(
        pool=pool,
        embedding=embedding,
        limit=limit * 3,
        threshold=threshold_for_query,
    )

    # Filter to threshold for actual results
    chunk_results = [r for r in all_chunk_results if r.similarity >= settings.search.similarity_threshold]
    semantic_results = _promote_chunks_to_memories(chunk_results)

    # For debug: find top result even if below threshold
    top_below_threshold = None
    if debug and not semantic_results and all_chunk_results:
        best = max(all_chunk_results, key=lambda r: r.similarity)
        top_below_threshold = {
            "memory_id": str(best.memory.id),
            "similarity": round(best.similarity, 4),
            "chunk_content": (best.chunk.content or "")[:120],
            "threshold": round(settings.search.similarity_threshold, 4),
        }

    low_spread = settings.search.adaptive_weights and detect_low_spread(
        [r.similarity for r in semantic_results],
        threshold=settings.search.score_spread_threshold,
    )

    if low_spread:
        effective_semantic: list[SearchResult] = []
        effective_weights = (0.0, 1.0)
    else:
        effective_semantic = semantic_results
        effective_weights = weights

    merged = rrf_merge(
        effective_semantic,
        keyword_results,
        weights=effective_weights,
        k=settings.search.rrf_k,
    )
    results = merged[:limit]

    if not debug:
        return results

    debug_info = SearchDebugInfo(
        query=query,
        weights=weights,
        effective_weights=effective_weights,
        low_spread_detected=low_spread,
        semantic_hits=[
            {
                "memory_id": str(r.memory.id),
                "similarity": round(r.similarity, 4),
                "chunk_content": (r.chunk_content or "")[:120],
            }
            for r in semantic_results
        ],
        keyword_hits=[
            {
                "memory_id": str(r.memory.id),
                "kw_rank": round(r.similarity, 4),
                "summary": r.memory.summary,
            }
            for r in keyword_results
        ],
        top_semantic_below_threshold=top_below_threshold,
        expanded_query=expanded_query,
    )
    return results, debug_info
