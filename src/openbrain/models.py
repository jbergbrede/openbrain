from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID


@dataclass
class ActionItem:
    text: str
    status: str = "open"
    due_date: str | None = None


@dataclass
class Memory:
    id: UUID
    content: str
    created_at: datetime
    summary: str | None = None
    embedding: list[float] | None = None
    people: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    action_items: list[ActionItem] = field(default_factory=list)
    connections: list[UUID] = field(default_factory=list)
    source: str = "mcp"
    source_metadata: dict = field(default_factory=dict)
    language: str = "en"
    content_english: str | None = None
    content_german: str | None = None


@dataclass
class SearchResult:
    memory: Memory
    similarity: float
    score: float


@dataclass
class EnrichmentResult:
    summary: str
    people: list[str]
    topics: list[str]
    action_items: list[ActionItem]
    language: str = "en"
    content_english: str = ""
    content_german: str = ""
