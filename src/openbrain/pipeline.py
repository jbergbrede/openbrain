from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import UUID

import asyncpg

from .chunker import chunk_content, count_tokens
from .config import Settings
from .embeddings import EmbeddingProvider
from .enrichment import enrich
from .models import Chunk, Memory
from .repository import (
    get_distinct_topics,
    insert_chunks,
    insert_memory_with_conn,
    search_chunks,
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

    # Enrich and chunk in parallel (both are independent of each other)
    enrichment, raw_chunks = await asyncio.gather(
        enrich(content=content, existing_topics=existing_topics),
        asyncio.to_thread(chunk_content, content),
    )

    # Embed content chunks (with contextual headers) + synthetic questions + summary
    header = f"[{enrichment.summary} | {', '.join(enrichment.topics)}]"
    content_embed_texts = [f"{header} {c.content}" for c in raw_chunks]
    question_texts = enrichment.questions or []
    all_embed_texts = content_embed_texts + question_texts
    all_embeddings, summary_embedding = await asyncio.gather(
        embedder.embed_batch(all_embed_texts),
        embedder.embed(enrichment.summary),
    )
    chunk_embeddings = all_embeddings[: len(raw_chunks)]
    question_embeddings = all_embeddings[len(raw_chunks) :]

    # Find connections using summary embedding (dedupe to memory level)
    candidate_results = await search_chunks(
        pool=pool,
        embedding=summary_embedding,
        limit=settings.connection_finding.max_connections,
        threshold=settings.connection_finding.similarity_threshold,
    )
    seen: set[UUID] = set()
    connected_ids: list[UUID] = []
    for r in candidate_results:
        if r.memory.id not in seen:
            seen.add(r.memory.id)
            connected_ids.append(r.memory.id)

    new_memory = Memory(
        id=UUID("00000000-0000-0000-0000-000000000000"),  # placeholder, DB assigns
        content=content,
        created_at=datetime.now(timezone.utc),
        summary=enrichment.summary,
        people=enrichment.people,
        topics=enrichment.topics,
        action_items=enrichment.action_items,
        connections=connected_ids,
        source=source,
        source_metadata=source_metadata or {},
        language=enrichment.language,
        content_english=enrichment.content_english or None,
        content_german=enrichment.content_german or None,
        keywords=enrichment.keywords,
        questions=enrichment.questions,
    )

    async with pool.acquire() as conn:
        async with conn.transaction():
            new_id = await insert_memory_with_conn(conn, new_memory)
            # Content chunks
            chunk_models = [
                Chunk(
                    memory_id=new_id,
                    chunk_index=raw.index,
                    content=raw.content,
                    token_count=raw.token_count,
                    embedding=emb,
                )
                for raw, emb in zip(raw_chunks, chunk_embeddings)
            ]
            # Synthetic question chunks
            for q_text, q_emb in zip(question_texts, question_embeddings):
                chunk_models.append(
                    Chunk(
                        memory_id=new_id,
                        chunk_index=-1,
                        content=q_text,
                        token_count=count_tokens(q_text),
                        embedding=q_emb,
                        is_synthetic=True,
                    )
                )
            await insert_chunks(conn, chunk_models)
            await update_connections(conn, new_id, connected_ids)
            for cid in connected_ids:
                await update_connections(conn, cid, [new_id])

    new_memory.id = new_id
    return new_memory
