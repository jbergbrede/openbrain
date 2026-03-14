import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
from uuid import uuid4

from openbrain.config import Settings
from openbrain.models import Memory
from openbrain.mcp_server import create_mcp_server


def make_mock_memory() -> Memory:
    return Memory(
        id=uuid4(),
        content="Test content",
        created_at=datetime.now(timezone.utc),
        summary="Test summary",
        people=[],
        topics=["test"],
        action_items=[],
        connections=[],
        source="mcp",
        source_metadata={},
    )


@pytest.fixture
def mock_pool():
    return MagicMock()


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 3)
    return embedder


@pytest.fixture
def mock_settings():
    return Settings(
        openai_api_key="fake",
        postgres_dsn="postgresql://fake/fake",
    )


@pytest.mark.asyncio
async def test_save_memory_tool(mock_pool, mock_embedder, mock_settings):
    memory = make_mock_memory()

    with patch("openbrain.mcp_server.pipeline_save", new=AsyncMock(return_value=memory)):
        mcp = create_mcp_server(mock_pool, mock_embedder, mock_settings)
        tools = {t.name: t for t in await mcp.list_tools()}
        assert "save_memory" in tools


@pytest.mark.asyncio
async def test_mcp_has_all_tools(mock_pool, mock_embedder, mock_settings):
    mcp = create_mcp_server(mock_pool, mock_embedder, mock_settings)
    tools = {t.name for t in await mcp.list_tools()}
    assert tools == {"save_memory", "search_memories", "get_memory", "list_memories", "delete_memory"}
