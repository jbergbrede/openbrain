# Openbrain

A personal memory RAG system — a "second brain" that captures free-form thoughts, enriches them with LLM-extracted metadata, and makes them retrievable via semantic search.

## System Diagram

```mermaid
graph TB
    subgraph Interfaces
        MCP[AI Client<br/>Claude etc.]
        SLACK[Slack<br/>🧠 react / DM]
    end

    subgraph App["Python App (single process)"]
        MCPS[MCP Server<br/>FastMCP]
        SLACKB[Slack Bot<br/>slack-bolt]
        CORE[Core Pipeline<br/>embed + enrich + connect]
        EMBED[Embedding Provider<br/>OpenAI / Google / Ollama]
        LLM[LLM Enrichment<br/>Claude Agent SDK]
    end

    subgraph Storage
        PG[(Postgres<br/>+ pgvector)]
    end

    MCP -->|stdio / MCP protocol| MCPS
    SLACK -->|socket mode| SLACKB
    MCPS --> CORE
    SLACKB --> CORE
    CORE --> EMBED
    CORE --> LLM
    CORE -->|read/write| PG
    EMBED -->|vectors| PG
```

### Capture Pipeline

```mermaid
sequenceDiagram
    participant Client
    participant Pipeline
    participant Embedder
    participant LLM
    participant DB

    Client->>Pipeline: save_memory(content)
    Pipeline->>DB: get_distinct_topics()
    par parallel
        Pipeline->>Embedder: embed(content)
        Pipeline->>LLM: enrich(content, existing_topics)
    end
    Embedder-->>Pipeline: embedding vector
    LLM-->>Pipeline: summary, people, topics, action_items
    Pipeline->>DB: search_memories(embedding) → connections
    Pipeline->>DB: transaction: insert memory + update bidirectional connections
    DB-->>Pipeline: memory id
    Pipeline-->>Client: Memory
```

## Setup

```bash
cp config.example.yaml config.yaml
cp .env.example .env
# fill in API keys and Postgres credentials in .env
```

## Running

**MCP server** (for use with Claude Desktop / Claude Code):

```bash
uv run openbrain --mode mcp
```

Add to your MCP client config:
```json
{
  "openbrain": {
    "command": "uv",
    "args": ["run", "--project", "/path/to/Openbrain", "openbrain", "--mode", "mcp"]
  }
}
```

**Slack bot:**

```bash
uv run openbrain --mode slack
```

**Both:**

```bash
uv run openbrain --mode both
```

## Docker

```bash
docker-compose up
```

Runs Postgres+pgvector and the app (Slack mode by default). Migrations run automatically on startup.

## MCP Tools

| Tool | Description |
|---|---|
| `save_memory` | Capture a thought. Enriches with metadata, embedding, and connections before storing. |
| `search_memories` | Semantic search. Returns matches with connection IDs. |
| `get_memory` | Fetch a memory by ID. Use to follow connections. |
| `list_memories` | Browse recent memories. Filterable by topics/people. |
| `delete_memory` | Remove a memory by ID. |

## Configuration

**`config.yaml`** — non-sensitive settings:

```yaml
embedding:
  provider: openai           # openai | google | ollama
  model: text-embedding-3-small

search:
  similarity_threshold: 0.7
  max_results: 10

connection_finding:
  similarity_threshold: 0.75
  max_connections: 5
```

**`.env`** — secrets:

```
OPENAI_API_KEY=sk-...         # only needed for OpenAI embeddings
POSTGRES_DSN=postgresql://openbrain:changeme@localhost:5432/openbrain
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
```

> **Enrichment** uses the Claude Agent SDK — no separate API key needed. It runs via the same `claude` CLI session you're already authenticated with.

## Slack Capture

- **Emoji react** — react with 🧠 on any message to save it as a memory. Bot confirms with ✅.
- **Direct message** — DM the bot with a thought. Bot confirms with ✅.

## Data Model

Memories are stored in a single `memories` table:

| Field | Type | Description |
|---|---|---|
| `id` | UUID | Primary key |
| `content` | TEXT | Raw thought |
| `summary` | TEXT | Claude-generated summary |
| `embedding` | VECTOR | Semantic search vector |
| `people` | TEXT[] | Extracted people mentions |
| `topics` | TEXT[] | Extracted topics (reused across memories) |
| `action_items` | JSONB | Extracted tasks with status |
| `connections` | UUID[] | Bidirectional links to related memories |
| `source` | TEXT | `mcp` or `slack` |
| `source_metadata` | JSONB | Channel, thread URL, author, etc. |
| `created_at` | TIMESTAMPTZ | Capture time |

Search ranking: `similarity × (1 + connection_count)` — well-connected "hub" memories rank higher.

## Development

```bash
uv run pytest tests/test_enrichment.py tests/test_mcp_tools.py   # unit tests (no Docker)
uv run pytest tests/test_repository.py                            # integration tests (requires Docker)
```

### Linting

[Ruff](https://docs.astral.sh/ruff/) is used for linting and formatting. Install the pre-commit hook once after cloning:

```bash
uv run pre-commit install
```

Ruff will then run automatically on every commit. To run manually:

```bash
uv run ruff check src/ tests/   # lint
uv run ruff format src/ tests/  # format
```
