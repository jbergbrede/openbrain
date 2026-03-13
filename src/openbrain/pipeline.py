from __future__ import annotations

import asyncio
import json
from uuid import UUID

import asyncpg

from .config import Settings
from .embeddings import EmbeddingProvider
from .enrichment import enrich
from .models import Memory
from .repository import (
    get_distinct_topics,
    insert_memory_with_conn,
    search_memories,
    update_connections,
)


async def save_memory(
    pool: asyncpg.Pool,
    embedder: EmbeddingProvider,
    settings: Settings,
    content: str,
    source: str = "mcp",
    source_metadata: dict | None = None,
) -> Memory:
    existing_topics = await get_distinct_topics(pool)

    embedding, enrichment = await asyncio.gather(
        embedder.embed(content),
        enrich(
            content=content,
            existing_topics=existing_topics,
        ),
    )

    candidate_results = await search_memories(
        pool=pool,
        embedding=embedding,
        limit=settings.connection_finding.max_connections,
        threshold=settings.connection_finding.similarity_threshold,
    )
    connected_ids = [r.memory.id for r in candidate_results]

    new_memory = Memory(
        id=UUID("00000000-0000-0000-0000-000000000000"),  # placeholder, DB assigns
        content=content,
        created_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        summary=enrichment.summary,
        embedding=embedding,
        people=enrichment.people,
        topics=enrichment.topics,
        action_items=enrichment.action_items,
        connections=connected_ids,
        source=source,
        source_metadata=source_metadata or {},
        language=enrichment.language,
        content_english=enrichment.content_english or None,
        content_german=enrichment.content_german or None,
    )

    async with pool.acquire() as conn:
        async with conn.transaction():
            new_id = await insert_memory_with_conn(conn, new_memory)
            await update_connections(conn, new_id, connected_ids)

    new_memory.id = new_id
    return new_memory
