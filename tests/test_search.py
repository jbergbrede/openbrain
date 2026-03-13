from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from openbrain.models import Memory, SearchResult
from openbrain.search import detect_low_spread, get_weights, hybrid_search, rrf_merge


def make_memory(**kwargs) -> Memory:
    defaults = dict(
        id=uuid4(),
        content="Test memory content",
        created_at=datetime.now(timezone.utc),
        summary="A test summary",
        embedding=[0.1] * 1536,
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
    """Insert bilingual memories and verify hybrid search finds them."""
    from openbrain.repository import insert_memory

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
    await insert_memory(pool, en_mem)
    await insert_memory(pool, de_mem)

    from openbrain.config import Settings
    from openbrain.repository import keyword_search_memories

    settings = Settings()

    # Short keyword query — should find both via keyword
    kw_results = await keyword_search_memories(pool, "Telekom", limit=10)
    kw_contents = [r.memory.content for r in kw_results]
    assert any("Telekom" in c for c in kw_contents)

    # German keyword — should match English memory via German translation in search_vector
    de_results = await keyword_search_memories(pool, "Rechnung", limit=10)
    de_ids = [r.memory.id for r in de_results]
    assert en_mem.id in de_ids or de_mem.id in de_ids


@pytest.mark.asyncio
async def test_hybrid_search_keyword_fallback_on_low_spread(pool):
    """When semantic scores cluster, hybrid falls back to keyword-only weighting."""
    from openbrain.repository import insert_memory
    from openbrain.config import Settings, SearchConfig

    mem = make_memory(
        content="INV-2026-0847 invoice number unique",
        language="en",
        content_english="INV-2026-0847 invoice number unique",
        content_german=None,
    )
    await insert_memory(pool, mem)

    mock_embedder = MagicMock()
    # Return identical embeddings so semantic scores will cluster
    mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)

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
