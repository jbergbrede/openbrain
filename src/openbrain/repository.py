from __future__ import annotations

from uuid import UUID

import asyncpg

from .models import ActionItem, Memory, SearchResult


def _row_to_memory(row: asyncpg.Record) -> Memory:
    action_items = [
        ActionItem(
            text=ai["text"],
            status=ai.get("status", "open"),
            due_date=ai.get("due_date"),
        )
        for ai in (row["action_items"] or [])
    ]
    keys = row.keys()
    return Memory(
        id=row["id"],
        content=row["content"],
        created_at=row["created_at"],
        summary=row["summary"],
        embedding=None,  # not returned by default
        people=list(row["people"] or []),
        topics=list(row["topics"] or []),
        action_items=action_items,
        connections=[UUID(str(c)) for c in (row["connections"] or [])],
        source=row["source"],
        source_metadata=dict(row["source_metadata"] or {}),
        language=row["language"] if "language" in keys else "en",
        content_english=row["content_english"] if "content_english" in keys else None,
        content_german=row["content_german"] if "content_german" in keys else None,
    )


async def insert_memory(pool: asyncpg.Pool, memory: Memory) -> UUID:
    embedding_list = memory.embedding or []
    action_items = [
        {"text": ai.text, "status": ai.status, "due_date": ai.due_date}
        for ai in memory.action_items
    ]

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO memories
                (content, summary, embedding, people, topics, action_items,
                 connections, source, source_metadata,
                 language, content_english, content_german, search_vector)
            VALUES ($1, $2, $3::vector, $4, $5, $6::jsonb, $7::uuid[], $8, $9::jsonb,
                    $10, $11, $12,
                    setweight(to_tsvector('english', COALESCE($11, '')), 'A') ||
                    setweight(to_tsvector('german',  COALESCE($12, '')), 'A') ||
                    setweight(to_tsvector('english', COALESCE($2, '')), 'B'))
            RETURNING id
            """,
            memory.content,
            memory.summary,
            str(embedding_list) if embedding_list else None,
            memory.people,
            memory.topics,
            action_items,
            [str(c) for c in memory.connections],
            memory.source,
            memory.source_metadata,
            memory.language,
            memory.content_english,
            memory.content_german,
        )
        return UUID(str(row["id"]))


async def insert_memory_with_conn(conn: asyncpg.Connection, memory: Memory) -> UUID:
    embedding_list = memory.embedding or []
    action_items = [
        {"text": ai.text, "status": ai.status, "due_date": ai.due_date}
        for ai in memory.action_items
    ]

    row = await conn.fetchrow(
        """
        INSERT INTO memories
            (content, summary, embedding, people, topics, action_items,
             connections, source, source_metadata,
             language, content_english, content_german, search_vector)
        VALUES ($1, $2, $3::vector, $4, $5, $6::jsonb, $7::uuid[], $8, $9::jsonb,
                $10, $11, $12,
                setweight(to_tsvector('english', COALESCE($11, '')), 'A') ||
                setweight(to_tsvector('german',  COALESCE($12, '')), 'A') ||
                setweight(to_tsvector('english', COALESCE($2, '')), 'B'))
        RETURNING id
        """,
        memory.content,
        memory.summary,
        str(embedding_list) if embedding_list else None,
        memory.people,
        memory.topics,
        action_items,
        [str(c) for c in memory.connections],
        memory.source,
        memory.source_metadata,
        memory.language,
        memory.content_english,
        memory.content_german,
    )
    return UUID(str(row["id"]))


async def get_memory(pool: asyncpg.Pool, memory_id: UUID) -> Memory | None:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM memories WHERE id = $1",
            str(memory_id),
        )
        if row is None:
            return None
        return _row_to_memory(row)


async def list_memories(
    pool: asyncpg.Pool,
    limit: int = 20,
    offset: int = 0,
    filter_topics: list[str] | None = None,
    filter_people: list[str] | None = None,
) -> list[Memory]:
    conditions = []
    params: list = []
    idx = 1

    if filter_topics:
        conditions.append(f"topics && ${idx}::text[]")
        params.append(filter_topics)
        idx += 1

    if filter_people:
        conditions.append(f"people && ${idx}::text[]")
        params.append(filter_people)
        idx += 1

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.extend([limit, offset])

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"SELECT * FROM memories {where} ORDER BY created_at DESC LIMIT ${idx} OFFSET ${idx+1}",
            *params,
        )
        return [_row_to_memory(row) for row in rows]


async def delete_memory(pool: asyncpg.Pool, memory_id: UUID) -> bool:
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM memories WHERE id = $1",
            str(memory_id),
        )
        return result == "DELETE 1"


async def search_memories(
    pool: asyncpg.Pool,
    embedding: list[float],
    limit: int = 10,
    threshold: float = 0.7,
) -> list[SearchResult]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT *,
                1 - (embedding <=> $1::vector) AS similarity,
                (1 - (embedding <=> $1::vector)) * (1 + COALESCE(array_length(connections, 1), 0)) AS score
            FROM memories
            WHERE embedding IS NOT NULL
              AND 1 - (embedding <=> $1::vector) >= $2
            ORDER BY score DESC
            LIMIT $3
            """,
            str(embedding),
            threshold,
            limit,
        )
        return [
            SearchResult(
                memory=_row_to_memory(row),
                similarity=float(row["similarity"]),
                score=float(row["score"]),
            )
            for row in rows
        ]


async def keyword_search_memories(
    pool: asyncpg.Pool,
    query: str,
    limit: int = 10,
    min_rank: float = 0.0,
) -> list[SearchResult]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT *,
                GREATEST(
                    ts_rank(search_vector, websearch_to_tsquery('english', $1)),
                    ts_rank(search_vector, websearch_to_tsquery('german', $1))
                ) AS kw_rank
            FROM memories
            WHERE search_vector IS NOT NULL
              AND (
                search_vector @@ websearch_to_tsquery('english', $1)
                OR search_vector @@ websearch_to_tsquery('german', $1)
              )
              AND GREATEST(
                    ts_rank(search_vector, websearch_to_tsquery('english', $1)),
                    ts_rank(search_vector, websearch_to_tsquery('german', $1))
                  ) > $2
            ORDER BY kw_rank DESC
            LIMIT $3
            """,
            query,
            min_rank,
            limit,
        )
        return [
            SearchResult(
                memory=_row_to_memory(row),
                similarity=float(row["kw_rank"]),
                score=float(row["kw_rank"]),
            )
            for row in rows
        ]


async def update_connections(
    conn: asyncpg.Connection, memory_id: UUID, connected_ids: list[UUID]
) -> None:
    if not connected_ids:
        return
    for cid in connected_ids:
        await conn.execute(
            """
            UPDATE memories
            SET connections = array_append(connections, $1::uuid)
            WHERE id = $2 AND NOT ($1::uuid = ANY(connections))
            """,
            str(cid),
            str(memory_id),
        )


async def find_memory_by_slack_ts(
    pool: asyncpg.Pool, channel: str, ts: str
) -> UUID | None:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id FROM memories
            WHERE source = 'slack'
              AND source_metadata->>'channel' = $1
              AND source_metadata->>'thread_ts' = $2
            LIMIT 1
            """,
            channel,
            ts,
        )
        return UUID(str(row["id"])) if row else None


async def link_memories(pool: asyncpg.Pool, id1: UUID, id2: UUID) -> None:
    async with pool.acquire() as conn:
        async with conn.transaction():
            await update_connections(conn, id1, [id2])
            await update_connections(conn, id2, [id1])


async def get_distinct_topics(pool: asyncpg.Pool) -> list[str]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT DISTINCT unnest(topics) AS topic FROM memories ORDER BY topic"
        )
        return [row["topic"] for row in rows]
