from __future__ import annotations

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

from .models import SearchResult

SYSTEM_PROMPT = """\
You are a personal memory assistant. Answer the user's question using ONLY the memories provided below.
Do not use any outside knowledge. If the memories do not contain enough information to answer, say so briefly.
Be concise and direct.\
"""


def _format_results(results: list[SearchResult]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        ts = r.memory.created_at.strftime("%Y-%m-%d")
        content = r.chunk_content or r.memory.content
        parts.append(f"[{i}] ({ts}) {content}")
    return "\n\n".join(parts)


async def synthesize(query_text: str, results: list[SearchResult]) -> str:
    memories_block = _format_results(results)
    prompt = f"{SYSTEM_PROMPT}\n\n---\nMemories:\n{memories_block}\n---\n\nQuestion: {query_text}"

    result_text = None
    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            max_turns=1,
            model="haiku",
        ),
    ):
        if isinstance(message, ResultMessage):
            if message.is_error:
                raise RuntimeError(f"Claude agent error: {message.result}")
            result_text = message.result

    return (result_text or "").strip()
