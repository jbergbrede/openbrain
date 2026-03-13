import asyncio
import pytest
import asyncpg
from testcontainers.postgres import PostgresContainer


@pytest.fixture(scope="session")
def postgres_container():
    with PostgresContainer("pgvector/pgvector:pg17") as pg:
        yield pg


@pytest.fixture(scope="session")
async def pool(postgres_container):
    from openbrain.db import get_pool, close_pool
    dsn = postgres_container.get_connection_url().replace("psycopg2", "asyncpg").replace("+asyncpg", "")
    p = await get_pool(dsn)
    yield p
    await close_pool()
