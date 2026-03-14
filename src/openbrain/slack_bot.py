from __future__ import annotations

import logging
import re

import asyncpg
from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp
from slack_sdk.errors import SlackApiError

from .config import Settings
from .embeddings import EmbeddingProvider
from .pipeline import save_memory
from .repository import delete_memory, find_memory_by_slack_ts
from .search import hybrid_search
from .synthesis import synthesize
from .transcribe import is_audio_file, transcribe_slack_file

log = logging.getLogger(__name__)


def create_slack_app(
    pool: asyncpg.Pool,
    embedder: EmbeddingProvider,
    settings: Settings,
) -> tuple[AsyncApp, AsyncSocketModeHandler]:
    app = AsyncApp(token=settings.slack_bot_token)

    async def mark_processing(client, channel, ts):
        try:
            await client.reactions_add(channel=channel, timestamp=ts, name="hourglass_flowing_sand")
        except SlackApiError as e:
            if e.response["error"] != "already_reacted":
                raise

    async def mark_done(client, channel, ts):
        try:
            await client.reactions_remove(channel=channel, timestamp=ts, name="hourglass_flowing_sand")
        except SlackApiError:
            pass
        try:
            await client.reactions_add(channel=channel, timestamp=ts, name="white_check_mark")
        except SlackApiError as e:
            if e.response["error"] != "already_reacted":
                raise

    async def mark_error(client, channel, ts):
        try:
            await client.reactions_remove(channel=channel, timestamp=ts, name="hourglass_flowing_sand")
        except SlackApiError:
            pass
        try:
            await client.reactions_add(channel=channel, timestamp=ts, name="x")
        except SlackApiError as e:
            if e.response["error"] != "already_reacted":
                raise

    async def handle_retrieval(query: str, client, channel: str, thread_ts: str | None):
        results = await hybrid_search(pool, embedder, query, settings, limit=settings.slack.retrieval_result_limit)
        if not results:
            text = "_I couldn't find anything about that._"
        else:
            text = await synthesize(query, results)
        await client.chat_postMessage(channel=channel, text=text, thread_ts=thread_ts)

    async def extract_audio(client, message: dict) -> tuple[str, float] | None:
        """Return (transcript, duration_seconds) if message has an audio file, else None."""
        if not settings.openai_api_key:
            return None
        for f in message.get("files", []):
            log.info(
                "Checking file: id=%s name=%s mime=%s is_audio=%s",
                f.get("id"),
                f.get("name"),
                f.get("mimetype"),
                is_audio_file(f),
            )
            if is_audio_file(f):
                # Event file objects are partial; fetch full file info for a working download URL
                log.info(f"Audio file detected: id={f.get('id')} name={f.get('name')} mime={f.get('mimetype')}")
                file_info_resp = await client.files_info(file=f["id"])
                full_file = file_info_resp["file"]
                log.info(
                    f"files.info url_private_download={full_file.get('url_private_download')} "
                    f"url_private={full_file.get('url_private')}"
                )
                transcript, duration = await transcribe_slack_file(
                    bot_token=settings.slack_bot_token,
                    file_info=full_file,
                    openai_api_key=settings.openai_api_key,
                )
                return transcript, duration
        return None

    async def build_message_content(client, message: dict) -> tuple[str, dict]:
        """Return (content, extra_metadata) for a single message, combining text and audio."""
        text = re.sub(r"<@[A-Z0-9]+>\s*", "", message.get("text", "")).strip()
        extra: dict = {}

        audio_result = await extract_audio(client, message)
        if audio_result:
            transcript, duration = audio_result
            parts = [p for p in [text, transcript] if p]
            text = "\n".join(parts)
            extra["source_type"] = "audio"
            extra["duration_seconds"] = duration

        return text, extra

    async def save_or_update_thread(client, channel, thread_ts, messages, author) -> bool:
        parts: list[str] = []
        audio_meta: dict = {}

        for m in messages:
            content, extra = await build_message_content(client, m)
            if content:
                parts.append(content)
            if extra.get("source_type") == "audio":
                audio_meta["source_type"] = "audio"
                audio_meta["duration_seconds"] = audio_meta.get("duration_seconds", 0.0) + extra.get(
                    "duration_seconds", 0.0
                )

        if not parts:
            return False

        full_content = "\n\n".join(parts)
        source_metadata = {"channel": channel, "thread_ts": thread_ts, "author": author, **audio_meta}

        existing_id = await find_memory_by_slack_ts(pool, channel, thread_ts)
        if existing_id:
            await delete_memory(pool, existing_id)
        await save_memory(
            pool=pool,
            embedder=embedder,
            settings=settings,
            content=full_content,
            source="slack",
            source_metadata=source_metadata,
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
        await mark_processing(client, channel, ts)
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
                saved = await save_or_update_thread(
                    client, channel, thread_ts, thread_result.get("messages", []), author
                )
                if saved:
                    await mark_done(client, channel, thread_ts)
            else:
                # standalone message
                content, extra_meta = await build_message_content(client, message)
                if not content:
                    return
                await save_memory(
                    pool=pool,
                    embedder=embedder,
                    settings=settings,
                    content=content,
                    source="slack",
                    source_metadata={"channel": channel, "thread_ts": ts, "author": author, **extra_meta},
                )
                await mark_done(client, channel, ts)
        except Exception as e:
            logger.error(f"Failed to save memory from reaction: {e}")
            await mark_error(client, channel, ts)

    @app.event("message")
    async def handle_dm(event, client, logger):
        if event.get("channel_type") != "im":
            return
        if event.get("subtype") and event.get("subtype") != "file_share":
            return
        channel = event["channel"]
        ts = event["ts"]
        user = event.get("user", "unknown")
        text = event.get("text", "")

        if text.startswith("?"):
            try:
                await handle_retrieval(text[1:].strip(), client, channel, thread_ts=None)
            except Exception as e:
                logger.error(f"Failed to handle DM retrieval: {e}")
            return

        file_meta = [
            {"id": f.get("id"), "name": f.get("name"), "mime": f.get("mimetype")} for f in event.get("files", [])
        ]
        log.info(f"handle_dm: subtype={event.get('subtype')} files={file_meta}")
        await mark_processing(client, channel, ts)
        try:
            content, extra_meta = await build_message_content(client, event)
            log.info(f"build_message_content: content_len={len(content)} extra_meta={extra_meta}")
            if not content:
                await mark_error(client, channel, ts)
                return
            await save_memory(
                pool=pool,
                embedder=embedder,
                settings=settings,
                content=content,
                source="slack",
                source_metadata={"channel": channel, "thread_ts": ts, "author": user, **extra_meta},
            )
            await mark_done(client, channel, ts)
        except Exception as e:
            logger.error(f"Failed to handle DM: {e}")
            await mark_error(client, channel, ts)

    @app.event("app_mention")
    async def handle_mention(event, client, logger):
        thread_ts = event.get("thread_ts")
        channel = event["channel"]
        ts = event["ts"]
        user = event.get("user", "unknown")

        text = event.get("text", "")
        content_text = re.sub(r"<@[A-Z0-9]+>\s*", "", text, count=1).strip()

        if content_text.startswith("?"):
            try:
                query_text = content_text[1:].strip()
                reply_ts = thread_ts or ts
                await handle_retrieval(query_text, client, channel, thread_ts=reply_ts)
            except Exception as e:
                logger.error(f"Failed to handle mention retrieval: {e}")
            return

        await mark_processing(client, channel, ts)
        try:
            if not thread_ts:
                # top-level mention → save this message
                content, extra_meta = await build_message_content(client, event)
                if not content:
                    return
                await save_memory(
                    pool=pool,
                    embedder=embedder,
                    settings=settings,
                    content=content,
                    source="slack",
                    source_metadata={"channel": channel, "thread_ts": ts, "author": user, **extra_meta},
                )
                await mark_done(client, channel, ts)
            else:
                # any thread mention → fetch full thread, upsert
                result = await client.conversations_replies(channel=channel, ts=thread_ts)
                saved = await save_or_update_thread(client, channel, thread_ts, result.get("messages", []), user)
                if saved:
                    await mark_done(client, channel, thread_ts)
        except Exception as e:
            logger.error(f"Failed to handle mention: {e}")
            await mark_error(client, channel, ts)

    handler = AsyncSocketModeHandler(app, settings.slack_app_token)
    return app, handler
