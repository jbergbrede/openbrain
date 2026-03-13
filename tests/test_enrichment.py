import json
import pytest
from unittest.mock import MagicMock, patch
from typing import AsyncGenerator

from claude_agent_sdk import ResultMessage
from openbrain.enrichment import enrich
from openbrain.models import EnrichmentResult


def _make_query_mock(json_data: dict):
    result = MagicMock(spec=ResultMessage)
    result.result = json.dumps(json_data)

    async def _gen(*args, **kwargs) -> AsyncGenerator:
        yield result

    return _gen


@pytest.mark.asyncio
async def test_enrich_parses_response():
    mock_query = _make_query_mock({
        "summary": "A test summary",
        "people": ["Alice", "Bob"],
        "topics": ["testing", "python"],
        "action_items": [{"text": "Write tests", "due_date": None}],
    })
    with patch("openbrain.enrichment.query", new=mock_query):
        result = await enrich(
            content="Met with Alice and Bob about testing Python code.",
            existing_topics=["python"],
        )

    assert isinstance(result, EnrichmentResult)
    assert result.summary == "A test summary"
    assert "Alice" in result.people
    assert "testing" in result.topics
    assert len(result.action_items) == 1
    assert result.action_items[0].text == "Write tests"


@pytest.mark.asyncio
async def test_enrich_handles_empty_fields():
    mock_query = _make_query_mock({
        "summary": "Just a thought",
        "people": [],
        "topics": ["general"],
        "action_items": [],
    })
    with patch("openbrain.enrichment.query", new=mock_query):
        result = await enrich(
            content="Just a random thought",
            existing_topics=[],
        )

    assert result.people == []
    assert result.action_items == []
    assert result.summary == "Just a thought"
