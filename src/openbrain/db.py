from __future__ import annotations

import json
import re
from pathlib import Path

import asyncpg

_pool: asyncpg.Pool | None = None

MIGRATIONS_DIR = Path(__file__).parent.parent.parent / "migrations"


async def _init_conn(conn: asyncpg.Connection) -> None:
    for typename in ("json", "jsonb"):
        await conn.set_type_codec(
            typename,
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
        )


async def get_pool(dsn: str) -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(dsn, init=_init_conn)
        await run_migrations(_pool)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def run_migrations(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS _migrations (
                filename TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)

        applied = {row["filename"] for row in await conn.fetch("SELECT filename FROM _migrations")}

        migration_files = sorted(f for f in MIGRATIONS_DIR.glob("*.sql") if re.match(r"^\d+_", f.name))

        for migration_file in migration_files:
            if migration_file.name not in applied:
                sql = migration_file.read_text()
                await conn.execute(sql)
                await conn.execute("INSERT INTO _migrations (filename) VALUES ($1)", migration_file.name)
