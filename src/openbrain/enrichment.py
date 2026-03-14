from __future__ import annotations

import json

from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

from .models import ActionItem, EnrichmentResult

SYSTEM_PROMPT = """\
You are an AI assistant that extracts structured information from personal notes and thoughts.
Extract the following from the user's text and return valid JSON only.

Fields:
- summary: concise 1-2 sentence summary
- people: list of people mentioned (first/last names, use empty list if none)
- topics: list of relevant topics/themes (prefer reusing existing topics from the provided list)
- action_items: list of tasks/todos, each with "text" and optional "due_date" (ISO date string or null)
- language: detected language of the input, either "en" or "de"
- content_english: the content in English (original if input is English, translation if input is German)
- content_german: the content in German (original if input is German, translation if input is English)

Return ONLY a JSON object with these exact keys. No markdown, no explanation.\
"""


def _build_prompt(content: str, existing_topics: list[str]) -> str:
    topics_hint = (
        f"\n\nExisting topics to reuse where applicable: {', '.join(existing_topics)}"
        if existing_topics
        else ""
    )
    return SYSTEM_PROMPT + "\n\n" + content + topics_hint


def _parse(data: dict, original_content: str) -> EnrichmentResult:
    action_items = [
        ActionItem(
            text=ai["text"],
            status=ai.get("status", "open"),
            due_date=ai.get("due_date"),
        )
        for ai in (data.get("action_items") or [])
    ]
    language = data.get("language", "en")
    content_english = data.get("content_english") or (original_content if language == "en" else "")
    content_german = data.get("content_german") or (original_content if language == "de" else "")
    return EnrichmentResult(
        summary=data.get("summary", ""),
        people=data.get("people") or [],
        topics=data.get("topics") or [],
        action_items=action_items,
        language=language,
        content_english=content_english,
        content_german=content_german,
    )


async def enrich(content: str, existing_topics: list[str]) -> EnrichmentResult:
    prompt = _build_prompt(content, existing_topics)
    result_text = None
    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            output_format={"type": "json_object"},
            max_turns=1,
            model="haiku",
        ),
    ):
        if isinstance(message, ResultMessage):
            result_text = message.result
    if not result_text:
        raise RuntimeError("No result from claude agent")
    # strip markdown code fences if present
    text = result_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return _parse(json.loads(text), original_content=content)
