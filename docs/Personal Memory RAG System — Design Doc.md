# Personal Memory RAG System — Design Doc

**Date:** March 4, 2026
**Status:** Draft

---

## 1. Overview

A personal memory system that acts as a "second brain" — capturing free-form thoughts, enriching them with LLM-extracted metadata, and making them retrievable via semantic search. The system is designed to be pluggable across embedding providers, LLM providers, and capture sources.

**Core stack:**
- **Python** — single process application
- **Postgres + pgvector** — storage and semantic search
- **MCP server** — primary interface for capture and retrieval
- **Slack bot** — secondary capture interface

---

## 2. Architecture

```
┌─────────────┐     ┌─────────────────────────────────────┐     ┌──────────────┐
│  AI Client   │◄──►│  Python App (single process)        │     │   Postgres   │
│ (Claude etc) │MCP  │  ┌───────────┐  ┌───────────────┐  │     │  + pgvector  │
└─────────────┘     │  │ MCP Server │  │   Slack Bot    │  │     └──────┬───────┘
                    │  └─────┬──────┘  └───────┬────────┘  │            │
                    │        └──────┬──────────┘           │            │
                    │          ┌────▼─────┐                │            │
                    │          │   Core   │────────────────┼────────────┘
                    │          │ Library  │                 │
                    │          └────┬─────┘                 │
                    │          ┌────▼─────┐                 │
                    │          │ Embedding│                 │
                    │          │ Provider │                 │
                    │          └──────────┘                 │
                    └─────────────────────────────────────┘
```

- **MCP Server** — thin interface exposing tools for capture and retrieval
- **Slack Bot** — thin interface for capturing thoughts via emoji react or DM
- **Core Library** — shared logic for enrichment, embedding, connection-finding, and all Postgres reads/writes
- **Embedding Provider** — pluggable adapter supporting cloud (OpenAI, Google) and local (Ollama) models

The single-process design keeps deployment simple (one Docker container). Since there is no shared in-memory state, both interfaces write to Postgres independently without conflict. If separation is needed later, the shared core library makes splitting into separate processes straightforward.

---

## 3. Data Model

**`memories` table:**

- `id` — UUID, primary key
- `content` — TEXT, the raw thought or note
- `summary` — TEXT, LLM-generated summary
- `embedding` — VECTOR(dims), semantic search vector (dimension depends on configured embedding model)
- `people` — TEXT[], extracted people mentions
- `topics` — TEXT[], extracted topics
- `action_items` — JSONB, extracted action items (text, status, optional due date)
- `connections` — UUID[], bidirectional links to related memories
- `source` — TEXT, `mcp` or `slack`
- `source_metadata` — JSONB, source-specific data (e.g., Slack channel, thread URL)
- `created_at` — TIMESTAMPTZ, when captured

**Indexes:**
- GIN index on `people`, `topics` for filtered queries
- HNSW or IVFFlat index on `embedding` for vector search
- GIN index on `connections` for bidirectional lookups

**Design decisions:**
- Arrays over junction tables — simpler, sufficient for a personal system, GIN-indexable
- Connections are bidirectional — when thought B connects to thought A, both arrays are updated
- Action items as JSONB — flexible structure without a separate table
- No `relevance_score` column — connection count is computed dynamically via `array_length(connections, 1)`

---

## 4. MCP Tools

| Tool | Description | Key Params |
|---|---|---|
| `save_memory` | Capture a thought. Core library enriches before storing. | `content` |
| `search_memories` | Semantic search. Returns matches with connection IDs (not content). | `query`, `limit` |
| `get_memory` | Fetch a specific memory by ID. Used to follow connections. | `id` |
| `list_memories` | Browse recent memories with optional filters. | `limit`, `offset`, `filter_topics`, `filter_people` |
| `delete_memory` | Remove a memory by ID. | `id` |

Connections are returned as IDs only. The AI client decides whether to follow them via `get_memory`, keeping token usage under its control.

---

## 5. Capture Pipeline

Triggered on every `save_memory` call (MCP or Slack):

1. **Generate embedding** — call configured embedding provider
2. **LLM enrichment** (runs in parallel with step 1) — single LLM call that extracts:
   - Summary
   - People
   - Topics
   - Action items
   - The enrichment prompt includes the list of existing topics so the LLM reuses them rather than creating duplicates
3. **Find connections** — semantic search existing memories using the new embedding, take top-N above a similarity threshold
4. **Update bidirectional connections** — append new memory's ID to each connected memory's connections array
5. **Store** — write everything to Postgres in one transaction

---

## 6. Search & Ranking

Search results are ranked by:

```
similarity * (1 + COALESCE(array_length(connections, 1), 0))
```

This boosts well-connected "hub" memories without overriding semantic relevance. No background job or precomputed scores needed — connection count is derived directly from the bidirectional connections array.

Action items are not tracked as tasks. Recency-weighted search lets the AI reason temporally: if you said "need to buy groceries" on Monday and "bought groceries" on Wednesday, the AI sees both chronologically and infers completion.

---

## 7. Slack Bot

Two capture modes:

- **Emoji react** — react with 🧠 on any message to save it as a memory
- **Direct message** — DM the bot with a thought

Both routes call the same core library pipeline (enrichment, embedding, connections, storage). The `source` field is set to `slack` and `source_metadata` captures channel, thread URL, and original author.

---

## 8. Configuration

**YAML config file** for non-sensitive settings:
- Embedding provider: name, model
- LLM provider: name, model
- Postgres: connection string
- Search defaults: similarity threshold, max results
- Connection finding: similarity threshold, max connections per memory

**Environment variables** for secrets:
- API keys (embedding, LLM)
- Slack bot token + app token
- Postgres password

---

## 9. Deployment

- Single Docker container running the Python app
- Postgres + pgvector in a separate container
- `docker-compose.yml` for local development and self-hosting

---

## 10. Future Considerations (Not in v1)

- **Attachments** — file/image storage alongside memories
- **Browser extension** — additional capture source
- **Email capture** — additional capture source
- **Process separation** — split MCP server and Slack bot into separate containers if needed
- **Self-improvement** — log which connections the AI follows via `get_memory` calls; use as signal to improve connection quality over time
- **PageRank** — evolve from simple connection count to a proper PageRank algorithm if the graph becomes complex enough to warrant itSTAMPTZ

**Design decisions:**

- **`people` and `topics` as Postgres arrays** with GIN indexes rather than separate junction tables. Simpler and sufficient for a personal system.
- **`connections` as UUID array** with bidirectional writes. When memory B connects to memory A, both arrays are updated. This keeps relevance boosting cheap at query time.
- **`action_items` as JSONB** for flexible structure without a separate table.
- **No `relevance_score` column.** Relevance boosting is computed at query time using `array_length(connections, 1)` — no background jobs needed.

**Indexes:**

- GIN index on `people`
- GIN index on `topics`
- HNSW or IVFFlat index on `embedding` for vector search

---

## 4. Capture Pipeline

When a memory is saved (via MCP or Slack), the core library runs the following pipeline:

**Step 1 — Generate embedding** (async)
Call the configured embedding provider to produce a vector for the raw content.

**Step 2 — LLM enrichment** (async, parallel with step 1)
A single LLM call that extracts:
- **Summary** — concise version of the thought
- **People** — mentioned individuals
- **Topics** — relevant themes, reusing existing topics from the database to avoid duplicates (the prompt receives the current list of distinct topics)
- **Action items** — any tasks or to-dos

**Step 3 — Find connections**
Semantic search existing memories using the new embedding. Select top-N results above a configurable similarity threshold as connections.

**Step 4 — Store**
In a single transaction:
- Insert the new memory with all enriched fields
- Update connected memories' `connections` arrays to include the new memory's ID (bidirectional linking)

Steps 1 and 2 run in parallel since they are independent.

---

## 5. MCP Tools

| Tool | Description |
|---|---|
| `save_memory` | Capture a thought. Core library enriches it (metadata, embedding, connections) before storing. |
| `search_memories` | Semantic search by natural language query. Returns matches with connection IDs but **not** connection content. Params: `query`, `limit`. |
| `get_memory` | Fetch a specific memory by ID. Used by the AI client to follow connections on demand. |
| `list_memories` | Browse recent memories. Params: `limit`, `offset`, `filter_topics`, `filter_people`. |
| `delete_memory` | Remove a memory by ID. |

**Connection retrieval is lazy by design.** `search_memories` returns connection IDs only. The AI client reads the memory, decides whether it needs more context, and calls `get_memory` for specific connections. This keeps token overhead under the AI's control.

**No `update_memory` tool** in v1. Delete and re-save if needed.

---

## 6. Search and Ranking

Search queries are embedded using the same provider as capture, then compared against stored vectors via cosine similarity.

**Ranking formula:**

```
score = similarity * (1 + COALESCE(array_length(connections, 1), 0))
```

Memories with more connections receive a natural boost, similar to PageRank. Because connections are bidirectional, heavily-referenced "hub" memories (core ideas, recurring themes) rank higher. Newer entries accumulate connections over time as subsequent thoughts link back to them.

Results are returned with clear timestamps so the AI client can reason temporally. For example, when asked about open action items, the AI sees both "need to buy groceries" (Monday) and "bought groceries" (Wednesday) and infers the task is complete.

---

## 7. Slack Bot

Two capture modes:

- **Emoji react** — React with 🧠 on any message to save it as a memory. The message content (and thread context if applicable) is passed through the capture pipeline.
- **Direct message** — DM the bot with a thought. The message content is passed through the capture pipeline.

Both modes use the same core library as MCP, ensuring consistent enrichment and storage.

---

## 8. Configuration

**Secrets (env vars):**
- Embedding provider API key
- LLM provider API key
- Postgres connection string
- Slack bot token + app token

**Settings (YAML config file):**
- Embedding provider: provider name, model name
- LLM provider: provider name, model name
- Search defaults: similarity threshold, max results
- Connection finding: similarity threshold, max connections per memory

---

## 9. Topic Consistency

To prevent topic drift and duplication (e.g., "ML" vs "machine learning"), the enrichment prompt receives the current list of distinct topics from the database. The LLM is instructed to reuse existing topics where applicable and only create new ones when no match exists.

This lightweight approach avoids the need for a separate review or consolidation process.

---

## 10. What This System Is Not

- **Not a task manager.** Action items are extracted as metadata for retrieval, but there is no completion tracking workflow. The AI reasons about task status by reading memories chronologically.
- **Not a graph database.** Connections are stored as adjacency lists in Postgres arrays. At personal scale (~10-18k memories/year), this is more than sufficient.
- **Not self-improving.** There is no reinforcement learning or automated feedback loop. Prompt tuning is done manually based on usage experience. The data to build feedback loops later (e.g., tracking which connections the AI follows via `get_memory` calls) is naturally captured.

---

## 11. Future Considerations

- **Attachments** — file/image support can be added later with a separate `attachments` table linked to memories
- **Additional capture sources** — browser extension, email, etc. Each is a thin interface calling into the core library
- **Process separation** — if needed, MCP server and Slack bot can be split into separate containers since they share no in-memory state
- **Self-improvement** — once enough usage data exists, connection quality and enrichment prompts could be improved based on which connections the AI actually follows