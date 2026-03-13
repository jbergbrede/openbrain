import pytest
from datetime import datetime, timezone
from uuid import UUID, uuid4

from openbrain.models import Memory, ActionItem
from openbrain.repository import (
    insert_memory,
    get_memory,
    list_memories,
    delete_memory,
    get_distinct_topics,
    update_connections,
    insert_memory_with_conn,
)


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
    )
    defaults.update(kwargs)
    return Memory(**defaults)


@pytest.mark.asyncio
async def test_insert_and_get(pool):
    mem = make_memory(content="Hello world", topics=["greetings"])
    inserted_id = await insert_memory(pool, mem)
    assert isinstance(inserted_id, UUID)

    fetched = await get_memory(pool, inserted_id)
    assert fetched is not None
    assert fetched.content == "Hello world"
    assert "greetings" in fetched.topics


@pytest.mark.asyncio
async def test_list_memories(pool):
    mem = make_memory(content="Listed memory", topics=["listing"])
    await insert_memory(pool, mem)

    results = await list_memories(pool, limit=50)
    contents = [m.content for m in results]
    assert "Listed memory" in contents


@pytest.mark.asyncio
async def test_list_filter_topics(pool):
    mem = make_memory(content="Topic filtered", topics=["unique_topic_xyz"])
    await insert_memory(pool, mem)

    results = await list_memories(pool, filter_topics=["unique_topic_xyz"])
    assert all("unique_topic_xyz" in m.topics for m in results)
    assert any(m.content == "Topic filtered" for m in results)


@pytest.mark.asyncio
async def test_delete(pool):
    mem = make_memory(content="To be deleted")
    mid = await insert_memory(pool, mem)

    deleted = await delete_memory(pool, mid)
    assert deleted is True

    result = await get_memory(pool, mid)
    assert result is None


@pytest.mark.asyncio
async def test_get_distinct_topics(pool):
    mem = make_memory(content="Topic test", topics=["alpha", "beta"])
    await insert_memory(pool, mem)

    topics = await get_distinct_topics(pool)
    assert "alpha" in topics
    assert "beta" in topics


@pytest.mark.asyncio
async def test_bidirectional_connections(pool):
    mem_a = make_memory(content="Memory A")
    mem_b = make_memory(content="Memory B")

    id_a = await insert_memory(pool, mem_a)
    id_b = await insert_memory(pool, mem_b)

    async with pool.acquire() as conn:
        async with conn.transaction():
            # When saving B, connect to A, and update A's connections array
            await conn.execute(
                "UPDATE memories SET connections = array_append(connections, $1::uuid) WHERE id = $2",
                str(id_a), str(id_b),
            )
            await update_connections(conn, id_b, [id_a])

    fetched_a = await get_memory(pool, id_a)
    fetched_b = await get_memory(pool, id_b)

    assert id_b in fetched_a.connections
    assert id_a in fetched_b.connections
