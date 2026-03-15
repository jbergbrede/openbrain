from __future__ import annotations

import logging

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

log = logging.getLogger(__name__)

_EXPANSION_PROMPT = """\
Rewrite this search query as a detailed paragraph that captures the intent, \
synonyms, and related concepts. Keep it under 100 words. \
Do NOT answer the question — just expand the search terms. \
Include both English and German terms where relevant.

Query: {query}

Expanded search:"""


async def expand_for_semantic(search_query: str, model: str = "haiku") -> str:
    """Expand a search query for the semantic leg. Returns original query on failure."""
    try:
        prompt = _EXPANSION_PROMPT.format(query=search_query)
        result_text = None
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(max_turns=1, model=model),
        ):
            if isinstance(message, ResultMessage):
                if message.is_error:
                    log.warning("Query expansion error: %s", message.result)
                    return search_query
                result_text = message.result
        expanded = (result_text or "").strip()
        return expanded if expanded else search_query
    except Exception:
        log.warning("Query expansion failed, using original query", exc_info=True)
        return search_query
