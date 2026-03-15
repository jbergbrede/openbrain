#!/usr/bin/env python3
"""
Batch migration: re-enrich all memories (Phase 2) and re-embed chunks
with contextual headers (Phase 1).

Usage:
    uv run scripts/migrate_enrichment.py

Run AFTER applying migration 005_keywords_questions.sql.

What it does for each memory:
1. Re-enrich with updated prompt → populates keywords + questions
2. Update keywords, questions columns + rebuild search_vector
3. Re-embed all content chunks with contextual headers (title + topics prefix)
4. Generate + embed + insert synthetic question chunks

Memories that already have keywords populated are skipped.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import asyncpg

from openbrain.chunker import chunk_content, count_tokens
from openbrain.config import load_config
from openbrain.db import _init_conn
from openbrain.embeddings import get_embedder
from openbrain.enrichment import enrich
from openbrain.models import Chunk
from openbrain.repository import get_distinct_topics

BATCH_SIZE = 5


async def get_memories_to_migrate(conn: asyncpg.Connection) -> list[dict]:
    """Get memories that haven't been re-enriched yet (no keywords)."""
    rows = await conn.fetch(
        """
        SELECT id, content, summary, topics
        FROM memories
        WHERE keywords = '{}' OR keywords IS NULL
        ORDER BY created_at ASC
        """
    )
    return [
        {
            "id": row["id"],
            "content": row["content"],
            "summary": row["summary"],
            "topics": list(row["topics"] or []),
        }
        for row in rows
    ]


async def migrate_memory(
    conn: asyncpg.Connection,
    embedder,
    mem: dict,
    existing_topics: list[str],
) -> None:
    memory_id = mem["id"]
    content = mem["content"]

    # 1. Re-enrich with updated prompt
    enrichment = await enrich(content=content, existing_topics=existing_topics)

    # 2. Update memory with new keywords, questions, and rebuilt search_vector
    keywords_text = " ".join(enrichment.keywords) if enrichment.keywords else ""
    await conn.execute(
        """
        UPDATE memories SET
            keywords = $2,
            questions = $3,
            search_vector =
                setweight(to_tsvector('english', COALESCE(content_english, '')), 'A') ||
                setweight(to_tsvector('german',  COALESCE(content_german, '')),  'A') ||
                setweight(to_tsvector('english', COALESCE(summary, '')),         'B') ||
                setweight(to_tsvector('simple',  COALESCE($4, '')),              'B')
        WHERE id = $1
        """,
        str(memory_id),
        enrichment.keywords,
        enrichment.questions,
        keywords_text,
    )

    # 3. Re-embed content chunks with contextual headers
    # Use the enrichment summary/topics for the header
    summary = enrichment.summary or mem["summary"] or ""
    topics = enrichment.topics or mem["topics"] or []
    header = f"[{summary} | {', '.join(topics)}]"

    # Get existing content chunks
    chunk_rows = await conn.fetch(
        """
        SELECT id, chunk_index, content, token_count
        FROM chunks
        WHERE memory_id = $1 AND NOT is_synthetic
        ORDER BY chunk_index
        """,
        str(memory_id),
    )

    if chunk_rows:
        # Re-embed with contextual headers
        embed_texts = [f"{header} {row['content']}" for row in chunk_rows]
        embeddings = await embedder.embed_batch(embed_texts)

        for row, emb in zip(chunk_rows, embeddings):
            await conn.execute(
                "UPDATE chunks SET embedding = $1::vector WHERE id = $2",
                str(emb),
                row["id"],
            )

    # 4. Delete old synthetic chunks and create new ones
    await conn.execute(
        "DELETE FROM chunks WHERE memory_id = $1 AND is_synthetic",
        str(memory_id),
    )

    question_texts = enrichment.questions or []
    if question_texts:
        question_embeddings = await embedder.embed_batch(question_texts)
        await conn.executemany(
            """
            INSERT INTO chunks (memory_id, chunk_index, content, embedding, token_count, is_synthetic)
            VALUES ($1, -1, $2, $3::vector, $4, true)
            """,
            [
                (
                    str(memory_id),
                    q_text,
                    str(q_emb),
                    count_tokens(q_text),
                )
                for q_text, q_emb in zip(question_texts, question_embeddings)
            ],
        )


async def main() -> None:
    settings = load_config()
    embedder = get_embedder(settings.embedding, settings.model_dump())
    pool = await asyncpg.create_pool(settings.postgres_dsn, init=_init_conn)

    try:
        async with pool.acquire() as conn:
            memories = await get_memories_to_migrate(conn)
            existing_topics = await get_distinct_topics(pool)

        total = len(memories)
        print(f"Found {total} memories to migrate")

        if total == 0:
            print("Nothing to do.")
            return

        for i in range(0, total, BATCH_SIZE):
            batch = memories[i : i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            print(f"Batch {batch_num} ({i + 1}-{min(i + len(batch), total)} of {total})...")

            for mem in batch:
                async with pool.acquire() as conn:
                    await migrate_memory(conn, embedder, mem, existing_topics)
                print(f"  migrated {mem['id']}")

        print("Migration complete.")
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
