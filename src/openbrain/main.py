from __future__ import annotations

import asyncio
import signal
import sys
import logging
import argparse

from .config import load_config
from .db import close_pool, get_pool
from .embeddings import get_embedder

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Openbrain memory system")
    parser.add_argument(
        "--mode",
        choices=["mcp", "slack", "both"],
        default="mcp",
        help="Which interface(s) to run (default: mcp)",
    )
    parser.add_argument("--config", default=None, help="Path to config YAML")
    return parser.parse_args()


async def run_mcp(pool, embedder, settings) -> None:
    from .mcp_server import create_mcp_server
    mcp = create_mcp_server(pool, embedder, settings)
    await mcp.run_async(transport="stdio")


async def run_slack(pool, embedder, settings) -> None:
    from .slack_bot import create_slack_app
    _, handler = create_slack_app(pool, embedder, settings)
    logger.info("Starting Slack bot in socket mode")
    try:
        await handler.start_async()
    except asyncio.CancelledError:
        await handler.close_async()
        raise


async def async_main(mode: str, config_path: str | None) -> None:
    settings = load_config(config_path)

    logger.info("Connecting to Postgres...")
    try:
        pool = await get_pool(settings.postgres_dsn)
        async with pool.acquire() as conn:
            version = await conn.fetchval("SELECT version()")
        logger.info("Postgres OK — %s", version.split(",")[0])
    except Exception as e:
        logger.error("Postgres connection failed: %s", e)
        raise

    embedder = get_embedder(settings.embedding, {
        "openai_api_key": settings.openai_api_key,
        "google_api_key": settings.google_api_key,
    })
    logger.info("Embedder: %s / %s", settings.embedding.provider, settings.embedding.model)

    try:
        if mode == "mcp":
            await run_mcp(pool, embedder, settings)
        elif mode == "slack":
            await run_slack(pool, embedder, settings)
        elif mode == "both":
            await asyncio.gather(
                run_mcp(pool, embedder, settings),
                run_slack(pool, embedder, settings),
            )
    finally:
        await close_pool()


def main() -> None:
    args = parse_args()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    task = loop.create_task(async_main(args.mode, args.config))

    def _shutdown():
        logger.info("Shutting down...")
        task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown)

    try:
        loop.run_until_complete(task)
    except asyncio.CancelledError:
        pass
    finally:
        loop.close()


if __name__ == "__main__":
    main()
