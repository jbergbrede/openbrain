#!/usr/bin/env python3
"""
Backfill chunks for existing memories that have none.

Usage:
    uv run scripts/backfill_chunks.py

Run AFTER applying migration 004_chunking.sql.
Run BEFORE applying migration 005_drop_embedding.sql.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import asyncpg

from openbrain.chunker import chunk_content
from openbrain.config import load_config
from openbrain.embeddings import get_embedder
from openbrain.models import Chunk
from openbrain.repository import insert_chunks

BATCH_SIZE = 10


async def get_memories_without_chunks(conn: asyncpg.Connection) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT id, content FROM memories
        WHERE id NOT IN (SELECT DISTINCT memory_id FROM chunks)
        ORDER BY created_at ASC
        """
    )
    return [{"id": row["id"], "content": row["content"]} for row in rows]


async def backfill(pool: asyncpg.Pool, embedder) -> None:
    async with pool.acquire() as conn:
        memories = await get_memories_without_chunks(conn)

    print(f"Found {len(memories)} memories to backfill")

    for i in range(0, len(memories), BATCH_SIZE):
        batch = memories[i : i + BATCH_SIZE]
        print(f"Processing batch {i // BATCH_SIZE + 1} ({len(batch)} memories)...")

        for mem in batch:
            raw_chunks = chunk_content(mem["content"])
            chunk_texts = [c.content for c in raw_chunks]
            embeddings = await embedder.embed_batch(chunk_texts)

            chunk_models = [
                Chunk(
                    memory_id=mem["id"],
                    chunk_index=raw.index,
                    content=raw.content,
                    token_count=raw.token_count,
                    embedding=emb,
                )
                for raw, emb in zip(raw_chunks, embeddings)
            ]

            async with pool.acquire() as conn:
                await insert_chunks(conn, chunk_models)

            print(f"  memory {mem['id']}: {len(chunk_models)} chunk(s)")

    print("Backfill complete.")
    print("You can now safely run migrations/005_drop_embedding.sql")


async def main() -> None:
    settings = load_config()
    embedder = get_embedder(settings.embedding, settings)
    pool = await asyncpg.create_pool(settings.postgres_dsn)
    try:
        await backfill(pool, embedder)
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
