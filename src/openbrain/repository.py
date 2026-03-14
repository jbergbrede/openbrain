from __future__ import annotations

from uuid import UUID

import asyncpg
from nltk.corpus import stopwords

from .models import ActionItem, Chunk, ChunkSearchResult, Memory, SearchResult

_STOPWORDS = frozenset(stopwords.words("english") + stopwords.words("german"))


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
    action_items = [{"text": ai.text, "status": ai.status, "due_date": ai.due_date} for ai in memory.action_items]

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO memories
                (content, summary, people, topics, action_items,
                 connections, source, source_metadata,
                 language, content_english, content_german, search_vector)
            VALUES ($1, $2, $3, $4, $5::jsonb, $6::uuid[], $7, $8::jsonb,
                    $9, $10, $11,
                    setweight(to_tsvector('english', COALESCE($10, '')), 'A') ||
                    setweight(to_tsvector('german',  COALESCE($11, '')), 'A') ||
                    setweight(to_tsvector('english', COALESCE($2, '')), 'B'))
            RETURNING id
            """,
            memory.content,
            memory.summary,
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
    action_items = [{"text": ai.text, "status": ai.status, "due_date": ai.due_date} for ai in memory.action_items]

    row = await conn.fetchrow(
        """
        INSERT INTO memories
            (content, summary, people, topics, action_items,
             connections, source, source_metadata,
             language, content_english, content_german, search_vector)
        VALUES ($1, $2, $3, $4, $5::jsonb, $6::uuid[], $7, $8::jsonb,
                $9, $10, $11,
                setweight(to_tsvector('english', COALESCE($10, '')), 'A') ||
                setweight(to_tsvector('german',  COALESCE($11, '')), 'A') ||
                setweight(to_tsvector('english', COALESCE($2, '')), 'B'))
        RETURNING id
        """,
        memory.content,
        memory.summary,
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


async def insert_chunks(conn: asyncpg.Connection, chunks: list[Chunk]) -> None:
    await conn.executemany(
        """
        INSERT INTO chunks (memory_id, chunk_index, content, embedding, token_count)
        VALUES ($1, $2, $3, $4::vector, $5)
        """,
        [
            (
                str(c.memory_id),
                c.chunk_index,
                c.content,
                str(c.embedding) if c.embedding else None,
                c.token_count,
            )
            for c in chunks
        ],
    )


async def search_chunks(
    pool: asyncpg.Pool,
    embedding: list[float],
    threshold: float,
    limit: int = 10,
) -> list[ChunkSearchResult]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                c.id AS chunk_id,
                c.memory_id,
                c.chunk_index,
                c.content AS chunk_content,
                c.token_count,
                1 - (c.embedding <=> $1::vector) AS similarity,
                m.id,
                m.content,
                m.created_at,
                m.summary,
                m.people,
                m.topics,
                m.action_items,
                m.connections,
                m.source,
                m.source_metadata,
                m.language,
                m.content_english,
                m.content_german
            FROM chunks c
            JOIN memories m ON c.memory_id = m.id
            WHERE c.embedding IS NOT NULL
              AND 1 - (c.embedding <=> $1::vector) >= $2
            ORDER BY similarity DESC
            LIMIT $3
            """,
            str(embedding),
            threshold,
            limit,
        )
        results = []
        for row in rows:
            memory = Memory(
                id=row["id"],
                content=row["content"],
                created_at=row["created_at"],
                summary=row["summary"],
                people=list(row["people"] or []),
                topics=list(row["topics"] or []),
                action_items=[
                    ActionItem(
                        text=ai["text"],
                        status=ai.get("status", "open"),
                        due_date=ai.get("due_date"),
                    )
                    for ai in (row["action_items"] or [])
                ],
                connections=[UUID(str(c)) for c in (row["connections"] or [])],
                source=row["source"],
                source_metadata=dict(row["source_metadata"] or {}),
                language=row["language"],
                content_english=row["content_english"],
                content_german=row["content_german"],
            )
            chunk = Chunk(
                id=UUID(str(row["chunk_id"])),
                memory_id=UUID(str(row["memory_id"])),
                chunk_index=row["chunk_index"],
                content=row["chunk_content"],
                token_count=row["token_count"],
            )
            results.append(
                ChunkSearchResult(
                    chunk=chunk,
                    memory=memory,
                    similarity=float(row["similarity"]),
                )
            )
        return results


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
            f"SELECT * FROM memories {where} ORDER BY created_at DESC LIMIT ${idx} OFFSET ${idx + 1}",
            *params,
        )
        return [_row_to_memory(row) for row in rows]


async def delete_memory(pool: asyncpg.Pool, memory_id: UUID) -> bool:
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "UPDATE memories SET connections = array_remove(connections, $1::uuid)"
                " WHERE $1::uuid = ANY(connections)",
                str(memory_id),
            )
            result = await conn.execute(
                "DELETE FROM memories WHERE id = $1",
                str(memory_id),
            )
        return result == "DELETE 1"


async def keyword_search_memories(
    pool: asyncpg.Pool,
    query: str,
    limit: int = 10,
    min_rank: float = 1e-4,
) -> list[SearchResult]:
    # OR-based query for matching: any word hit surfaces a result.
    # German compound words (e.g. "Bundesautobahn") won't match a query for
    # "autobahn" with AND semantics, so we match permissively and let ts_rank
    # (computed against the original AND query) handle scoring.
    or_query = (
        " OR ".join(w for w in query.split() if w and w.lower() not in _STOPWORDS) or query
    )  # fallback to raw query if all tokens are stopwords
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
                search_vector @@ websearch_to_tsquery('english', $2)
                OR search_vector @@ websearch_to_tsquery('german', $2)
              )
              AND GREATEST(
                    ts_rank(search_vector, websearch_to_tsquery('english', $1)),
                    ts_rank(search_vector, websearch_to_tsquery('german', $1))
                  ) > $3
            ORDER BY kw_rank DESC
            LIMIT $4
            """,
            query,
            or_query,
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


async def update_connections(conn: asyncpg.Connection, memory_id: UUID, connected_ids: list[UUID]) -> None:
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


async def find_memory_by_slack_ts(pool: asyncpg.Pool, channel: str, ts: str) -> UUID | None:
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
        rows = await conn.fetch("SELECT DISTINCT unnest(topics) AS topic FROM memories ORDER BY topic")
        return [row["topic"] for row in rows]
