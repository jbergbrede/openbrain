from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID


@dataclass
class ActionItem:
    text: str
    status: str = "open"
    due_date: str | None = None
    id: UUID | None = None
    memory_id: UUID | None = None
    created_at: datetime | None = None


@dataclass
class Memory:
    id: UUID
    content: str
    created_at: datetime
    summary: str | None = None
    people: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    action_items: list[ActionItem] = field(default_factory=list)
    connections: list[UUID] = field(default_factory=list)
    source: str = "mcp"
    source_metadata: dict = field(default_factory=dict)
    language: str = "en"
    content_english: str | None = None
    content_german: str | None = None
    keywords: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    memory_id: UUID
    chunk_index: int
    content: str
    token_count: int
    embedding: list[float] | None = None
    id: UUID | None = None
    is_synthetic: bool = False


@dataclass
class ChunkSearchResult:
    chunk: Chunk
    memory: Memory
    similarity: float


@dataclass
class SearchResult:
    memory: Memory
    similarity: float
    score: float
    chunk_content: str | None = None
    chunk_id: UUID | None = None


@dataclass
class EnrichmentResult:
    summary: str
    people: list[str]
    topics: list[str]
    action_items: list[ActionItem]
    language: str = "en"
    content_english: str = ""
    content_german: str = ""
    keywords: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
