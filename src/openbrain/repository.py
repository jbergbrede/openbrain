from __future__ import annotations

from uuid import UUID

import asyncpg
from nltk.corpus import stopwords

from .models import ActionItem, Chunk, ChunkSearchResult, Memory, SearchResult  # ActionItem used in insert_action_items

_STOPWORDS = frozenset(stopwords.words("english") + stopwords.words("german"))


def _row_to_memory(row: asyncpg.Record) -> Memory:
    keys = row.keys()
    return Memory(
        id=row["id"],
        content=row["content"],
        created_at=row["created_at"],
        summary=row["summary"],
        people=list(row["people"] or []),
        topics=list(row["topics"] or []),
        connections=[UUID(str(c)) for c in (row["connections"] or [])],
        source=row["source"],
        source_metadata=dict(row["source_metadata"] or {}),
        language=row["language"] if "language" in keys else "en",
        content_english=row["content_english"] if "content_english" in keys else None,
        content_german=row["content_german"] if "content_german" in keys else None,
        keywords=list(row["keywords"] or []) if "keywords" in keys else [],
        questions=list(row["questions"] or []) if "questions" in keys else [],
    )


async def insert_memory(pool: asyncpg.Pool, memory: Memory) -> UUID:
    keywords_text = " ".join(memory.keywords) if memory.keywords else ""

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO memories
                (content, summary, people, topics,
                 connections, source, source_metadata,
                 language, content_english, content_german,
                 keywords, questions, search_vector)
            VALUES ($1, $2, $3, $4, $5::uuid[], $6, $7::jsonb,
                    $8, $9, $10, $11, $12,
                    setweight(to_tsvector('english', COALESCE($9, '')), 'A') ||
                    setweight(to_tsvector('german',  COALESCE($10, '')), 'A') ||
                    setweight(to_tsvector('english', COALESCE($2, '')),  'B') ||
                    setweight(to_tsvector('simple',  COALESCE($13, '')), 'B'))
            RETURNING id
            """,
            memory.content,
            memory.summary,
            memory.people,
            memory.topics,
            [str(c) for c in memory.connections],
            memory.source,
            memory.source_metadata,
            memory.language,
            memory.content_english,
            memory.content_german,
            memory.keywords,
            memory.questions,
            keywords_text,
        )
        return UUID(str(row["id"]))


async def insert_memory_with_conn(conn: asyncpg.Connection, memory: Memory) -> UUID:
    keywords_text = " ".join(memory.keywords) if memory.keywords else ""

    row = await conn.fetchrow(
        """
        INSERT INTO memories
            (content, summary, people, topics,
             connections, source, source_metadata,
             language, content_english, content_german,
             keywords, questions, search_vector)
        VALUES ($1, $2, $3, $4, $5::uuid[], $6, $7::jsonb,
                $8, $9, $10, $11, $12,
                setweight(to_tsvector('english', COALESCE($9, '')), 'A') ||
                setweight(to_tsvector('german',  COALESCE($10, '')), 'A') ||
                setweight(to_tsvector('english', COALESCE($2, '')),  'B') ||
                setweight(to_tsvector('simple',  COALESCE($13, '')), 'B'))
        RETURNING id
        """,
        memory.content,
        memory.summary,
        memory.people,
        memory.topics,
        [str(c) for c in memory.connections],
        memory.source,
        memory.source_metadata,
        memory.language,
        memory.content_english,
        memory.content_german,
        memory.keywords,
        memory.questions,
        keywords_text,
    )
    return UUID(str(row["id"]))


async def insert_chunks(conn: asyncpg.Connection, chunks: list[Chunk]) -> None:
    await conn.executemany(
        """
        INSERT INTO chunks (memory_id, chunk_index, content, embedding, token_count, is_synthetic)
        VALUES ($1, $2, $3, $4::vector, $5, $6)
        """,
        [
            (
                str(c.memory_id),
                c.chunk_index,
                c.content,
                str(c.embedding) if c.embedding else None,
                c.token_count,
                c.is_synthetic,
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
                c.is_synthetic,
                1 - (c.embedding <=> $1::vector) AS similarity,
                m.id,
                m.content,
                m.created_at,
                m.summary,
                m.people,
                m.topics,
                m.connections,
                m.source,
                m.source_metadata,
                m.language,
                m.content_english,
                m.content_german,
                m.keywords,
                m.questions
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
                connections=[UUID(str(c)) for c in (row["connections"] or [])],
                source=row["source"],
                source_metadata=dict(row["source_metadata"] or {}),
                language=row["language"],
                content_english=row["content_english"],
                content_german=row["content_german"],
                keywords=list(row["keywords"] or []),
                questions=list(row["questions"] or []),
            )
            chunk = Chunk(
                id=UUID(str(row["chunk_id"])),
                memory_id=UUID(str(row["memory_id"])),
                chunk_index=row["chunk_index"],
                content=row["chunk_content"],
                token_count=row["token_count"],
                is_synthetic=row["is_synthetic"],
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
    created_after: str | None = None,
    created_before: str | None = None,
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

    if created_after:
        conditions.append(f"created_at >= ${idx}::timestamptz")
        params.append(created_after)
        idx += 1

    if created_before:
        conditions.append(f"created_at <= ${idx}::timestamptz")
        params.append(created_before)
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


async def insert_action_items(conn: asyncpg.Connection, memory_id: UUID, items: list[ActionItem]) -> list[UUID]:
    rows = await conn.fetch(
        """
        INSERT INTO action_items (memory_id, text, status, due_date)
        SELECT $1::uuid, t, s, CASE WHEN d IS NOT NULL THEN d::date ELSE NULL END
        FROM unnest($2::text[], $3::text[], $4::text[]) AS u(t, s, d)
        RETURNING id
        """,
        str(memory_id),
        [ai.text for ai in items],
        [ai.status for ai in items],
        [ai.due_date if ai.due_date else None for ai in items],
    )
    return [UUID(str(row["id"])) for row in rows]


async def list_action_items(
    pool: asyncpg.Pool,
    status: str | None = "open",
    due_before: str | None = None,
    due_after: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[dict]:
    conditions = []
    params: list = []
    idx = 1

    if status and status != "all":
        conditions.append(f"ai.status = ${idx}")
        params.append(status)
        idx += 1

    if due_before:
        conditions.append(f"ai.due_date <= ${idx}::date")
        params.append(due_before)
        idx += 1

    if due_after:
        conditions.append(f"ai.due_date >= ${idx}::date")
        params.append(due_after)
        idx += 1

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.extend([limit, offset])

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT ai.id, ai.memory_id, ai.text, ai.status, ai.due_date, ai.created_at,
                   m.people, m.topics, m.source,
                   LEFT(m.summary, 80) as memory_title
            FROM action_items ai
            JOIN memories m ON ai.memory_id = m.id
            {where}
            ORDER BY ai.due_date ASC NULLS LAST, ai.created_at DESC
            LIMIT ${idx} OFFSET ${idx + 1}
            """,
            *params,
        )
        return [
            {
                "id": str(row["id"]),
                "memory_id": str(row["memory_id"]),
                "text": row["text"],
                "status": row["status"],
                "due_date": row["due_date"].isoformat() if row["due_date"] else None,
                "created_at": row["created_at"].isoformat(),
                "memory_title": row["memory_title"],
                "people": list(row["people"] or []),
                "topics": list(row["topics"] or []),
                "source": row["source"],
            }
            for row in rows
        ]


async def update_action_item(
    pool: asyncpg.Pool,
    item_id: UUID,
    status: str | None = None,
    due_date: str | None = None,
    clear_due_date: bool = False,
) -> dict | None:
    sets = []
    params: list = []
    idx = 1

    if status is not None:
        sets.append(f"status = ${idx}")
        params.append(status)
        idx += 1

    if clear_due_date:
        sets.append("due_date = NULL")
    elif due_date is not None:
        sets.append(f"due_date = ${idx}::date")
        params.append(due_date)
        idx += 1

    if not sets:
        return None

    params.append(str(item_id))
    row = None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"UPDATE action_items SET {', '.join(sets)} WHERE id = ${idx} RETURNING id, text, status, due_date",
            *params,
        )
    if row is None:
        return None
    return {
        "id": str(row["id"]),
        "text": row["text"],
        "status": row["status"],
        "due_date": row["due_date"].isoformat() if row["due_date"] else None,
    }


async def delete_action_item(pool: asyncpg.Pool, item_id: UUID) -> bool:
    async with pool.acquire() as conn:
        result = await conn.execute("DELETE FROM action_items WHERE id = $1", str(item_id))
    return result == "DELETE 1"


async def find_connection_candidates(
    pool: asyncpg.Pool,
    embedding: list[float],
    topics: list[str],
    keywords: list[str],
    people: list[str],
    limit: int,
    threshold: float,
    topic_boost: float = 0.1,
    people_boost: float = 0.05,
    keyword_boost: float = 0.05,
    composite_threshold: float = 0.5,
) -> list[ChunkSearchResult]:
    chunk_results = await search_chunks(pool=pool, embedding=embedding, threshold=threshold, limit=limit * 3)

    topics_set = set(t.lower() for t in topics)
    people_set = set(p.lower() for p in people)
    keywords_set = set(k.lower() for k in keywords)

    # Dedupe to memory level, keeping max composite score
    best: dict[UUID, tuple[float, ChunkSearchResult]] = {}
    for r in chunk_results:
        base = r.similarity
        boost = 0.0
        mem = r.memory
        if topics_set & set(t.lower() for t in mem.topics):
            boost += topic_boost
        if people_set & set(p.lower() for p in mem.people):
            boost += people_boost
        mem_kw = set(k.lower() for k in mem.keywords)
        if len(keywords_set & mem_kw) >= 2:
            boost += keyword_boost
        composite = base + boost
        mid = mem.id
        if mid not in best or composite > best[mid][0]:
            best[mid] = (composite, r)

    results = [
        r for score, r in sorted(best.values(), key=lambda x: x[0], reverse=True) if score >= composite_threshold
    ]
    return results[:limit]


async def find_memory_by_paperless_id(pool: asyncpg.Pool, doc_id: int) -> UUID | None:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id FROM memories
            WHERE source = 'paperless'
              AND (source_metadata->>'paperless_document_id')::int = $1
            LIMIT 1
            """,
            doc_id,
        )
        return UUID(str(row["id"])) if row else None


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
