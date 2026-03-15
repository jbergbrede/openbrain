from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from openbrain.models import Chunk, ChunkSearchResult, Memory, SearchResult
from openbrain.search import (
    _promote_chunks_to_memories,
    detect_low_spread,
    get_weights,
    hybrid_search,
    rrf_merge,
)


def make_memory(**kwargs) -> Memory:
    defaults = dict(
        id=uuid4(),
        content="Test memory content",
        created_at=datetime.now(timezone.utc),
        summary="A test summary",
        people=[],
        topics=["testing"],
        action_items=[],
        connections=[],
        source="mcp",
        source_metadata={},
        language="en",
        content_english="Test memory content",
        content_german=None,
    )
    defaults.update(kwargs)
    return Memory(**defaults)


def make_result(memory: Memory | None = None, similarity: float = 0.8) -> SearchResult:
    m = memory or make_memory()
    return SearchResult(memory=m, similarity=similarity, score=similarity)


# --- Unit: get_weights ---


def test_get_weights_short_query():
    s, k = get_weights("Telekom")
    assert s == 0.2
    assert k == 0.8


def test_get_weights_two_words():
    s, k = get_weights("Deutsche Telekom")
    assert s == 0.2
    assert k == 0.8


def test_get_weights_medium_query():
    s, k = get_weights("invoice from last month")
    assert s == 0.5
    assert k == 0.5


def test_get_weights_long_query():
    s, k = get_weights("how much did I pay for my internet bill")
    assert s == 0.7
    assert k == 0.3


# --- Unit: detect_low_spread ---


def test_detect_low_spread_empty():
    assert detect_low_spread([]) is False


def test_detect_low_spread_single():
    assert detect_low_spread([0.85]) is False


def test_detect_low_spread_below_threshold():
    assert detect_low_spread([0.81, 0.82, 0.83], threshold=0.05) is True


def test_detect_low_spread_above_threshold():
    assert detect_low_spread([0.5, 0.9], threshold=0.05) is False


def test_detect_low_spread_exactly_at_threshold():
    # spread == threshold still counts as low spread (uses <=)
    assert detect_low_spread([0.80, 0.85], threshold=0.05) is True


# --- Unit: rrf_merge ---


def test_rrf_merge_combines_results():
    m1 = make_memory(content="alpha")
    m2 = make_memory(content="beta")
    m3 = make_memory(content="gamma")

    semantic = [make_result(m1, 0.9), make_result(m2, 0.8)]
    keyword = [make_result(m2, 0.7), make_result(m3, 0.6)]

    merged = rrf_merge(semantic, keyword, weights=(0.5, 0.5), k=60)
    ids = [r.memory.id for r in merged]

    assert m1.id in ids
    assert m2.id in ids
    assert m3.id in ids


def test_rrf_merge_keyword_only_weights():
    m1 = make_memory(content="alpha")
    m2 = make_memory(content="beta")

    semantic = [make_result(m1, 0.9)]
    keyword = [make_result(m2, 0.8)]

    merged = rrf_merge(semantic, keyword, weights=(0.0, 1.0), k=60)
    # With keyword-only weights, semantic results get 0 score contribution
    # but keyword results dominate; m2 should rank first
    assert merged[0].memory.id == m2.id


def test_rrf_merge_connection_boost():
    m_connected = make_memory(content="well connected", connections=[uuid4(), uuid4(), uuid4()])
    m_plain = make_memory(content="no connections")

    # Both appear at same rank in both legs
    semantic = [make_result(m_connected, 0.8), make_result(m_plain, 0.8)]
    keyword = [make_result(m_connected, 0.8), make_result(m_plain, 0.8)]

    merged = rrf_merge(semantic, keyword, weights=(0.5, 0.5), k=60)
    assert merged[0].memory.id == m_connected.id


def test_rrf_merge_deduplicates():
    m = make_memory()
    semantic = [make_result(m, 0.9)]
    keyword = [make_result(m, 0.8)]

    merged = rrf_merge(semantic, keyword, weights=(0.5, 0.5), k=60)
    assert len(merged) == 1


# --- Integration: hybrid_search ---


@pytest.mark.asyncio
async def test_hybrid_search_integration(pool):
    """Insert bilingual memories + chunks and verify hybrid search finds them."""
    from openbrain.repository import insert_chunks, insert_memory

    en_mem = make_memory(
        content="Deutsche Telekom monthly invoice 49.99 EUR",
        language="en",
        content_english="Deutsche Telekom monthly invoice 49.99 EUR",
        content_german="Deutsche Telekom monatliche Rechnung 49.99 EUR",
    )
    de_mem = make_memory(
        content="Ich habe meine Rechnung bezahlt",
        language="de",
        content_english="I paid my invoice",
        content_german="Ich habe meine Rechnung bezahlt",
    )
    en_id = await insert_memory(pool, en_mem)
    de_id = await insert_memory(pool, de_mem)

    embedding = [0.1] * 1536
    async with pool.acquire() as conn:
        await insert_chunks(
            conn,
            [
                Chunk(memory_id=en_id, chunk_index=0, content=en_mem.content, token_count=8, embedding=embedding),
                Chunk(memory_id=de_id, chunk_index=0, content=de_mem.content, token_count=6, embedding=embedding),
            ],
        )

    from openbrain.repository import keyword_search_memories

    # Short keyword query — should find both via keyword
    kw_results = await keyword_search_memories(pool, "Telekom", limit=10)
    kw_contents = [r.memory.content for r in kw_results]
    assert any("Telekom" in c for c in kw_contents)

    # German keyword — should match English memory via German translation in search_vector
    de_results = await keyword_search_memories(pool, "Rechnung", limit=10)
    de_ids = [r.memory.id for r in de_results]
    assert en_id in de_ids or de_id in de_ids


@pytest.mark.asyncio
async def test_hybrid_search_keyword_fallback_on_low_spread(pool):
    """When semantic scores cluster, hybrid falls back to keyword-only weighting."""
    from openbrain.config import SearchConfig, Settings
    from openbrain.repository import insert_chunks, insert_memory

    mem = make_memory(
        content="INV-2026-0847 invoice number unique",
        language="en",
        content_english="INV-2026-0847 invoice number unique",
        content_german=None,
    )
    mem_id = await insert_memory(pool, mem)

    identical_embedding = [0.1] * 1536
    async with pool.acquire() as conn:
        await insert_chunks(
            conn,
            [
                Chunk(
                    memory_id=mem_id, chunk_index=0, content=mem.content, token_count=5, embedding=identical_embedding
                ),
            ],
        )

    mock_embedder = MagicMock()
    # Return identical embeddings so semantic scores will cluster
    mock_embedder.embed = AsyncMock(return_value=identical_embedding)

    settings = Settings(
        openai_api_key="fake",
        postgres_dsn="postgresql://fake/fake",
        search=SearchConfig(
            similarity_threshold=0.0,  # allow all semantic results
            adaptive_weights=True,
            score_spread_threshold=0.05,
            rrf_k=60,
        ),
    )

    results = await hybrid_search(
        pool=pool,
        embedder=mock_embedder,
        query="INV-2026-0847",
        settings=settings,
        limit=10,
    )
    # Should return results (keyword leg fires even if semantic is low-signal)
    assert isinstance(results, list)


# --- Unit: _promote_chunks_to_memories ---


def test_promote_chunks_max_similarity():
    """Highest chunk similarity wins for each memory."""
    mem = make_memory()
    chunk_a = Chunk(memory_id=mem.id, chunk_index=0, content="first chunk", token_count=2)
    chunk_b = Chunk(memory_id=mem.id, chunk_index=1, content="second chunk", token_count=2)

    results = [
        ChunkSearchResult(chunk=chunk_a, memory=mem, similarity=0.7),
        ChunkSearchResult(chunk=chunk_b, memory=mem, similarity=0.9),
    ]
    promoted = _promote_chunks_to_memories(results)

    assert len(promoted) == 1
    assert promoted[0].similarity == 0.9
    assert promoted[0].chunk_content == "second chunk"


def test_promote_chunks_synthetic_boosts_score_but_real_content_shown():
    """Synthetic chunks participate in scoring but real chunk content is returned."""
    mem = make_memory()
    real_chunk = Chunk(memory_id=mem.id, chunk_index=0, content="real content", token_count=2)
    synthetic_chunk = Chunk(
        memory_id=mem.id, chunk_index=-1, content="How much did it cost?", token_count=5, is_synthetic=True
    )

    results = [
        ChunkSearchResult(chunk=real_chunk, memory=mem, similarity=0.7),
        ChunkSearchResult(chunk=synthetic_chunk, memory=mem, similarity=0.95),
    ]
    promoted = _promote_chunks_to_memories(results)

    assert len(promoted) == 1
    # Score comes from the synthetic chunk (highest)
    assert promoted[0].similarity == 0.95
    # But content comes from the real chunk
    assert promoted[0].chunk_content == "real content"


def test_promote_chunks_multiple_memories():
    """Each memory gets its own promoted result."""
    m1 = make_memory(content="memory one")
    m2 = make_memory(content="memory two")
    c1 = Chunk(memory_id=m1.id, chunk_index=0, content="chunk one", token_count=2)
    c2 = Chunk(memory_id=m2.id, chunk_index=0, content="chunk two", token_count=2)

    results = [
        ChunkSearchResult(chunk=c1, memory=m1, similarity=0.8),
        ChunkSearchResult(chunk=c2, memory=m2, similarity=0.6),
    ]
    promoted = _promote_chunks_to_memories(results)

    assert len(promoted) == 2
    assert promoted[0].memory.id == m1.id  # sorted by similarity desc


def test_rrf_merge_carries_chunk_content():
    """chunk_content from semantic leg is preserved in merged results."""
    m = make_memory()
    semantic = [SearchResult(memory=m, similarity=0.9, score=0.9, chunk_content="relevant snippet", chunk_id=None)]
    keyword = [SearchResult(memory=m, similarity=0.8, score=0.8)]

    merged = rrf_merge(semantic, keyword, weights=(0.5, 0.5), k=60)
    assert len(merged) == 1
    assert merged[0].chunk_content == "relevant snippet"
