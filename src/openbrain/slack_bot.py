from __future__ import annotations

import re

import asyncpg
from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp
from slack_sdk.errors import SlackApiError

from .config import Settings
from .embeddings import EmbeddingProvider
from .pipeline import save_memory
from .repository import delete_memory, find_memory_by_slack_ts


def create_slack_app(
    pool: asyncpg.Pool,
    embedder: EmbeddingProvider,
    settings: Settings,
) -> tuple[AsyncApp, AsyncSocketModeHandler]:
    app = AsyncApp(token=settings.slack_bot_token)

    async def mark_done(client, channel, ts):
        try:
            await client.reactions_add(channel=channel, timestamp=ts, name="white_check_mark")
        except SlackApiError as e:
            if e.response["error"] != "already_reacted":
                raise

    async def save_or_update_thread(channel, thread_ts, messages, author) -> bool:
        content = "\n\n".join(
            re.sub(r"<@[A-Z0-9]+>\s*", "", m.get("text", "")).strip() for m in messages if m.get("text")
        )
        if not content:
            return False
        existing_id = await find_memory_by_slack_ts(pool, channel, thread_ts)
        if existing_id:
            await delete_memory(pool, existing_id)
        await save_memory(
            pool=pool,
            embedder=embedder,
            settings=settings,
            content=content,
            source="slack",
            source_metadata={"channel": channel, "thread_ts": thread_ts, "author": author},
        )
        return True

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
            result = await client.conversations_history(channel=channel, latest=ts, limit=1, inclusive=True)
            messages = result.get("messages", [])
            if not messages:
                return
            message = messages[0]
            author = message.get("user", "unknown")
            thread_ts = message.get("thread_ts", ts)

            if message.get("reply_count", 0) > 0 or message.get("thread_ts"):
                # thread root with replies OR a reply — fetch full thread, upsert
                thread_result = await client.conversations_replies(channel=channel, ts=thread_ts)
                saved = await save_or_update_thread(channel, thread_ts, thread_result.get("messages", []), author)
                if saved:
                    await mark_done(client, channel, thread_ts)
            else:
                # standalone message
                content = message.get("text", "")
                if not content:
                    return
                await save_memory(
                    pool=pool,
                    embedder=embedder,
                    settings=settings,
                    content=content,
                    source="slack",
                    source_metadata={"channel": channel, "thread_ts": ts, "author": author},
                )
                await mark_done(client, channel, ts)
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

    @app.event("app_mention")
    async def handle_mention(event, client, logger):
        thread_ts = event.get("thread_ts")
        channel = event["channel"]
        ts = event["ts"]
        user = event.get("user", "unknown")

        text = event.get("text", "")
        content = re.sub(r"<@[A-Z0-9]+>\s*", "", text, count=1).strip()

        try:
            if not thread_ts:
                # top-level mention → save this message
                if not content:
                    return
                await save_memory(
                    pool=pool,
                    embedder=embedder,
                    settings=settings,
                    content=content,
                    source="slack",
                    source_metadata={"channel": channel, "thread_ts": ts, "author": user},
                )
                await mark_done(client, channel, ts)
                return
            else:
                # any thread mention → fetch full thread, upsert
                result = await client.conversations_replies(channel=channel, ts=thread_ts)
                saved = await save_or_update_thread(channel, thread_ts, result.get("messages", []), user)
                if saved:
                    await mark_done(client, channel, thread_ts)
                return
        except Exception as e:
            logger.error(f"Failed to save memory from mention: {e}")

    handler = AsyncSocketModeHandler(app, settings.slack_app_token)
    return app, handler
