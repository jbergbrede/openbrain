from __future__ import annotations

import asyncpg
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler

from .config import Settings
from .embeddings import EmbeddingProvider
from .pipeline import save_memory


def create_slack_app(
    pool: asyncpg.Pool,
    embedder: EmbeddingProvider,
    settings: Settings,
) -> tuple[AsyncApp, AsyncSocketModeHandler]:
    app = AsyncApp(token=settings.slack_bot_token)

    @app.event("reaction_added")
    async def handle_reaction(event, client, logger):
        if event.get("reaction") != "brain":
            return
        item = event.get("item", {})
        if item.get("type") != "message":
            return
        channel = item["channel"]
        ts = item["ts"]
        try:
            result = await client.conversations_history(
                channel=channel, latest=ts, limit=1, inclusive=True
            )
            messages = result.get("messages", [])
            if not messages:
                return
            message = messages[0]
            content = message.get("text", "")
            if not content:
                return
            author = message.get("user", "unknown")
            thread_ts = message.get("thread_ts", ts)
            await save_memory(
                pool=pool,
                embedder=embedder,
                settings=settings,
                content=content,
                source="slack",
                source_metadata={
                    "channel": channel,
                    "thread_ts": thread_ts,
                    "author": author,
                },
            )
            await client.reactions_add(channel=channel, timestamp=ts, name="white_check_mark")
        except Exception as e:
            logger.error(f"Failed to save memory from reaction: {e}")

    @app.event("message")
    async def handle_dm(event, client, logger):
        if event.get("channel_type") != "im":
            return
        if event.get("subtype"):
            return
        content = event.get("text", "")
        if not content:
            return
        channel = event["channel"]
        ts = event["ts"]
        user = event.get("user", "unknown")
        try:
            await save_memory(
                pool=pool,
                embedder=embedder,
                settings=settings,
                content=content,
                source="slack",
                source_metadata={
                    "channel": channel,
                    "thread_ts": ts,
                    "author": user,
                },
            )
            await client.reactions_add(channel=channel, timestamp=ts, name="white_check_mark")
        except Exception as e:
            logger.error(f"Failed to save memory from DM: {e}")

    handler = AsyncSocketModeHandler(app, settings.slack_app_token)
    return app, handler
