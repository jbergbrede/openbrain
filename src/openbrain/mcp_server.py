from __future__ import annotations

from uuid import UUID

import asyncpg
import fastmcp

from .config import Settings
from .embeddings import EmbeddingProvider
from .models import Memory
from .pipeline import save_memory as pipeline_save
from .repository import delete_action_item as repo_delete_action_item
from .repository import delete_memory as repo_delete_memory
from .repository import get_memory as repo_get_memory
from .repository import link_memories as repo_link_memories
from .repository import list_action_items as repo_list_action_items
from .repository import list_memories as repo_list_memories
from .repository import update_action_item as repo_update_action_item
from .search import hybrid_search


def create_mcp_server(
    pool: asyncpg.Pool,
    embedder: EmbeddingProvider,
    settings: Settings,
) -> fastmcp.FastMCP:
    mcp = fastmcp.FastMCP("openbrain")

    def _memory_dict(memory: Memory) -> dict:
        return {
            "id": str(memory.id),
            "content": memory.content,
            "summary": memory.summary,
            "people": memory.people,
            "topics": memory.topics,
            "connections": [str(c) for c in memory.connections],
            "source": memory.source,
            "source_metadata": memory.source_metadata,
            "created_at": memory.created_at.isoformat(),
        }

    @mcp.tool()
    async def save_memory(content: str) -> dict:
        """Save something to the user's personal memory store. Use when the user wants to remember something — a note, event, invoice, conversation, decision, or any personal information. Automatically enriches with metadata (people, topics, summary)."""  # noqa: E501
        memory = await pipeline_save(
            pool=pool,
            embedder=embedder,
            settings=settings,
            content=content,
            source="mcp",
        )
        return _memory_dict(memory)

    @mcp.tool()
    async def search_memories(query: str, limit: int = 10, debug: bool = False) -> dict:
        """Search the user's personal memory store using hybrid semantic + keyword search. Use this proactively whenever the user asks about anything personal — past events, people, places, finances, appointments, invoices, health, travel, or any information they may have previously stored. Returns the matching chunk and metadata per result — use get_memory(id) to fetch the full content of a specific memory. Set debug=true for explainability info.

        Query guidelines: Pass all relevant details from the user's question — including names, dates, amounts, and specific terms. Do not omit details, but also do not pad the query with filler. For example, prefer "electricity bill January Stadtwerke" over just "electricity bill", but do not wrap it in a full sentence."""  # noqa: E501
        outcome = await hybrid_search(
            pool=pool,
            embedder=embedder,
            query=query,
            settings=settings,
            limit=limit,
            debug=debug,
        )
        if debug:
            results, debug_info = outcome
        else:
            results = outcome

        hits = [
            {
                "id": str(r.memory.id),
                "summary": r.memory.summary,
                "people": r.memory.people,
                "topics": r.memory.topics,
                "connections": [str(c) for c in r.memory.connections],
                "similarity": r.similarity,
                "score": r.score,
                "created_at": r.memory.created_at.isoformat(),
                "matched_chunk": r.chunk_content,
                "chunk_id": str(r.chunk_id) if r.chunk_id else None,
            }
            for r in results
        ]

        if not debug:
            return {"results": hits}

        return {
            "results": hits,
            "debug": {
                "query": debug_info.query,
                "similarity_threshold": settings.search.similarity_threshold,
                "weights": {
                    "semantic": debug_info.effective_weights[0],
                    "keyword": debug_info.effective_weights[1],
                },
                "initial_weights": {
                    "semantic": debug_info.weights[0],
                    "keyword": debug_info.weights[1],
                },
                "low_spread_detected": debug_info.low_spread_detected,
                "expanded_query": debug_info.expanded_query,
                "semantic_hits": debug_info.semantic_hits,
                "top_semantic_below_threshold": debug_info.top_semantic_below_threshold,
                "keyword_hits": debug_info.keyword_hits,
            },
        }

    @mcp.tool()
    async def get_memory(id: str) -> dict | None:
        """Fetch the full content of a specific memory by its ID. Use to retrieve complete details after finding a memory via search, or to follow connection IDs returned by other tools."""  # noqa: E501
        memory = await repo_get_memory(pool, UUID(id))
        if memory is None:
            return None
        return _memory_dict(memory)

    @mcp.tool()
    async def list_memories(
        limit: int = 20,
        offset: int = 0,
        filter_topics: list[str] | None = None,
        filter_people: list[str] | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
    ) -> list[dict]:
        """Browse the user's most recent memories, optionally filtered by topic, person, or date range. Use when the user wants to see what's stored, review recent entries, or explore memories without a specific query in mind."""  # noqa: E501
        memories = await repo_list_memories(
            pool=pool,
            limit=limit,
            offset=offset,
            filter_topics=filter_topics,
            filter_people=filter_people,
            created_after=created_after,
            created_before=created_before,
        )
        return [_memory_dict(m) for m in memories]

    @mcp.tool()
    async def delete_memory(id: str) -> dict:
        """Permanently delete a memory by ID. Use when the user asks to forget or remove something from their memory store."""  # noqa: E501
        deleted = await repo_delete_memory(pool, UUID(id))
        return {"deleted": deleted, "id": id}

    @mcp.tool()
    async def link_memories(id1: str, id2: str) -> dict:
        """Manually connect two related memories bidirectionally."""
        await repo_link_memories(pool, UUID(id1), UUID(id2))
        return {"linked": True, "id1": id1, "id2": id2}

    @mcp.tool()
    async def list_action_items(
        status: str = "open",
        due_before: str | None = None,
        due_after: str | None = None,
        limit: int = 20,
    ) -> dict:
        """List action items across all memories. Filter by status and due date range. status: 'open', 'done', or 'all'."""  # noqa: E501
        items = await repo_list_action_items(
            pool=pool,
            status=status,
            due_before=due_before,
            due_after=due_after,
            limit=limit,
        )
        return {"items": items, "count": len(items)}

    @mcp.tool()
    async def update_action_item(
        id: str,
        status: str | None = None,
        due_date: str | None = None,
    ) -> dict:
        """Update an action item's status or due date. Pass empty string for due_date to clear it."""
        clear_due = due_date == ""
        result = await repo_update_action_item(
            pool=pool,
            item_id=UUID(id),
            status=status,
            due_date=due_date if not clear_due else None,
            clear_due_date=clear_due,
        )
        if result is None:
            return {"updated": False, "id": id}
        return {"updated": True, **result}

    @mcp.tool()
    async def delete_action_item(id: str) -> dict:
        """Delete a specific action item (e.g., irrelevant auto-extracted items)."""
        deleted = await repo_delete_action_item(pool, UUID(id))
        return {"deleted": deleted, "id": id}

    return mcp
