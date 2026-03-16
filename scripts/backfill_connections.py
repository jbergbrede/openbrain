#!/usr/bin/env python3
"""Backfill connections for all memories using the improved multi-signal algorithm."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openbrain.config import load_config
from openbrain.db import get_pool
from openbrain.embeddings import get_embedder
from openbrain.repository import find_connection_candidates, list_memories, update_connections


async def main() -> None:
    settings = load_config()
    pool = await get_pool(settings.postgres_dsn)
    embedder = get_embedder(
        settings.embedding,
        {"openai_api_key": settings.openai_api_key, "google_api_key": settings.google_api_key},
    )

    memories = await list_memories(pool, limit=10000)
    print(f"Loaded {len(memories)} memories")

    cfg = settings.connection_finding
    total_new = 0

    for i, memory in enumerate(memories):
        summary = memory.summary or memory.content[:200]
        embedding = await embedder.embed(summary)

        candidates = await find_connection_candidates(
            pool=pool,
            embedding=embedding,
            topics=memory.topics,
            keywords=memory.keywords,
            people=memory.people,
            limit=cfg.max_connections,
            threshold=cfg.similarity_threshold,
            topic_boost=cfg.topic_boost,
            people_boost=cfg.people_boost,
            keyword_boost=cfg.keyword_boost,
            composite_threshold=cfg.composite_threshold,
        )

        new_ids = [
            r.memory.id for r in candidates if r.memory.id != memory.id and r.memory.id not in set(memory.connections)
        ]

        if new_ids:
            async with pool.acquire() as conn:
                await update_connections(conn, memory.id, new_ids)
                for cid in new_ids:
                    await update_connections(conn, cid, [memory.id])
            total_new += len(new_ids)
            print(f"  [{i + 1}/{len(memories)}] {memory.id}: +{len(new_ids)} new connections")
        else:
            if (i + 1) % 50 == 0:
                print(f"  [{i + 1}/{len(memories)}] ...")

    print(f"\nDone. Total new connections added: {total_new}")
    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
