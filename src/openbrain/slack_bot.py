from __future__ import annotations

import logging
import re

import asyncpg
from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp
from slack_sdk.errors import SlackApiError

from .config import Settings
from .embeddings import EmbeddingProvider
from .paperless import fetch_paperless_document
from .pipeline import save_memory
from .models import SearchResult
from .repository import delete_memory, find_memory_by_paperless_id, find_memory_by_slack_ts
from .repository import list_memories as list_memories_db
from .search import SearchDebugInfo, hybrid_search
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

    async def handle_retrieval(query: str, client, channel: str, thread_ts: str | None, debug: bool = False):
        if debug:
            results, dbg = await hybrid_search(
                pool, embedder, query, settings, limit=settings.slack.retrieval_result_limit, debug=True
            )
        else:
            results = await hybrid_search(pool, embedder, query, settings, limit=settings.slack.retrieval_result_limit)

        if not results:
            text = "_I couldn't find anything about that._"
        else:
            text = await synthesize(query, results)

        if debug:
            text += "\n\n" + _format_debug(dbg)

        await client.chat_postMessage(channel=channel, text=text, thread_ts=thread_ts)

    def _format_debug(dbg: SearchDebugInfo) -> str:
        lines = [
            "```",
            f"[debug] weights: semantic={dbg.weights[0]}, keyword={dbg.weights[1]}",
            f"[debug] effective_weights: semantic={dbg.effective_weights[0]}, keyword={dbg.effective_weights[1]}",
            f"[debug] low_spread_detected: {dbg.low_spread_detected}",
            "",
            f"[semantic hits: {len(dbg.semantic_hits)}]",
        ]
        for h in dbg.semantic_hits:
            lines.append(f"  {h['similarity']:.4f}  {h['chunk_content'][:80]!r}")
        if dbg.top_semantic_below_threshold:
            t = dbg.top_semantic_below_threshold
            lines.append(
                f"  (top below threshold={t['threshold']}: {t['similarity']:.4f}  {t['chunk_content'][:80]!r})"
            )
        lines.append(f"\n[keyword hits: {len(dbg.keyword_hits)}]")
        for h in dbg.keyword_hits:
            lines.append(f"  rank={h['kw_rank']:.4f}  {h['summary'] or ''}")
        lines.append("```")
        return "\n".join(lines)

    async def extract_audio(client, message: dict) -> tuple[str, float] | None:
        """Return (transcript, duration_seconds) if message has an audio file, else None."""
        if not settings.openai_api_key:
            return None

        # Collect candidate file objects: top-level files + files nested in attachments
        candidates: list[dict] = list(message.get("files", []))
        for attachment in message.get("attachments", []):
            candidates.extend(attachment.get("files", []))

        for f in candidates:
            file_id = f.get("id")
            if not file_id:
                continue
            # Always fetch full file info: event objects are partial and may lack mimetype/URLs
            file_info_resp = await client.files_info(file=file_id)
            full_file = file_info_resp["file"]
            if is_audio_file(full_file):
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

    async def handle_paperless_message(event, client, logger):
        channel = event["channel"]
        ts = event["ts"]
        subtype = event.get("subtype")
        if subtype and subtype != "bot_message":
            return
        text = event.get("text", "")
        m = re.search(r"/documents/(\d+)/", text)
        if not m:
            log.debug("Paperless channel message has no [id:N] token, skipping")
            return
        doc_id = int(m.group(1))
        await mark_processing(client, channel, ts)
        try:
            doc = await fetch_paperless_document(settings.paperless.base_url, settings.paperless_api_token, doc_id)
            content_parts = []
            if doc["title"]:
                content_parts.append(f"Title: {doc['title']}")
            if doc["correspondent"]:
                content_parts.append(f"Correspondent: {doc['correspondent']}")
            if doc["tags"]:
                content_parts.append(f"Tags: {', '.join(doc['tags'])}")
            if doc["content"]:
                content_parts.append(doc["content"])
            content = "\n".join(content_parts)
            if not content.strip():
                await mark_error(client, channel, ts)
                return
            source_metadata = {
                "paperless_document_id": doc_id,
                "paperless_title": doc["title"],
                "paperless_correspondent": doc["correspondent"],
                "paperless_tags": doc["tags"],
            }
            existing_id = await find_memory_by_paperless_id(pool, doc_id)
            if existing_id:
                await delete_memory(pool, existing_id)
            await save_memory(
                pool=pool,
                embedder=embedder,
                settings=settings,
                content=content,
                source="paperless",
                source_metadata=source_metadata,
            )
            await mark_done(client, channel, ts)
        except Exception as e:
            logger.error(f"Failed to capture Paperless document {doc_id}: {e}")
            await mark_error(client, channel, ts)

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

        if text.startswith("??"):
            try:
                await handle_retrieval(text[2:].strip(), client, channel, thread_ts=None, debug=True)
            except Exception as e:
                logger.error(f"Failed to handle DM retrieval: {e}")
            return
        if text.startswith("?"):
            try:
                await handle_retrieval(text[1:].strip(), client, channel, thread_ts=None)
            except Exception as e:
                logger.error(f"Failed to handle DM retrieval: {e}")
            return

        await mark_processing(client, channel, ts)
        try:
            content, extra_meta = await build_message_content(client, event)
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

        if settings.paperless.base_url and settings.paperless.base_url in text:
            return await handle_paperless_message(event, client, logger)

        if content_text.startswith("??"):
            try:
                query_text = content_text[2:].strip()
                reply_ts = thread_ts or ts
                await handle_retrieval(query_text, client, channel, thread_ts=reply_ts, debug=True)
            except Exception as e:
                logger.error(f"Failed to handle mention retrieval: {e}")
            return
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

    @app.command("/list_memories")
    async def handle_list_memories(ack, body, client, logger):
        await ack()
        text = (body.get("text") or "").strip()
        default_limit = settings.slack.list_result_limit
        try:
            limit = int(text) if text else default_limit
            limit = max(1, min(limit, 50))
        except ValueError:
            limit = default_limit
        channel = body["channel_id"]
        user = body["user_id"]
        try:
            memories = await list_memories_db(pool=pool, limit=limit)
            if not memories:
                out = "_No memories stored yet._"
            else:
                results = [
                    SearchResult(
                        memory=m,
                        similarity=1.0,
                        score=1.0,
                        chunk_content=m.summary or m.content,
                        chunk_id=None,
                    )
                    for m in memories
                ]
                out = await synthesize(f"Summarize my {limit} most recent memories", results)
            await client.chat_postEphemeral(channel=channel, user=user, text=out)
        except Exception as e:
            logger.error(f"/list_memories failed: {e}")
            await client.chat_postEphemeral(channel=channel, user=user, text="_Failed to retrieve memories._")

    @app.command("/search_memories")
    async def handle_search_memories(ack, body, client, logger):
        await ack()
        query = (body.get("text") or "").strip()
        channel = body["channel_id"]
        user = body["user_id"]
        if not query:
            await client.chat_postEphemeral(channel=channel, user=user, text="_Usage: /search_memories <query>_")
            return
        try:
            results = await hybrid_search(
                pool, embedder, query, settings, limit=settings.slack.retrieval_result_limit
            )
            if not results:
                out = "_I couldn't find anything about that._"
            else:
                out = await synthesize(query, results)
            await client.chat_postEphemeral(channel=channel, user=user, text=out)
        except Exception as e:
            logger.error(f"/search_memories failed: {e}")
            await client.chat_postEphemeral(channel=channel, user=user, text="_Search failed._")

    handler = AsyncSocketModeHandler(app, settings.slack_app_token)
    return app, handler
