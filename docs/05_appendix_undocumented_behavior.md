# Appendix: Additional System Behavior

This appendix covers implemented behavior not described in the other design docs.

---

## @Mention Capture

In addition to emoji react and DM capture (doc 01), the Slack bot captures memories via channel @mentions.

**Top-level mention** (not in a thread): the bot strips its own mention tag, saves the remaining text as a memory with `channel` and `thread_ts` metadata.

**Thread mention**: the bot fetches the full thread via `conversations_replies`, strips all mention tags, concatenates all messages, and saves the combined content as a single memory.

---

## Thread Upsert

When saving from a Slack thread (via emoji react on a threaded message or @mention inside a thread), the bot performs a delete-and-replace:

1. Look up existing memory by `(channel, thread_ts)`
2. If found, delete the old memory and its chunks (CASCADE)
3. Save the full thread content as a new memory through the standard pipeline

Re-reacting or re-mentioning in a thread after new replies replaces the old memory with updated content.

---

## CLI Modes

The application runs via `openbrain --mode {mcp|slack|both} [--config PATH]`.

- `mcp` — runs the MCP server only (stdio transport)
- `slack` — runs the Slack bot only (Socket Mode)
- `both` — runs MCP server and Slack bot concurrently via `asyncio.gather`

Default mode is `mcp`. Config path defaults to `config.yaml` or the `OPENBRAIN_CONFIG` environment variable.

---

## Auto-Migrations

On startup, the database pool runs all pending SQL migrations from the `migrations/` directory. Files must match the pattern `^\d+_*.sql` and are executed in lexicographic order. Applied migrations are tracked in a `_migrations` table with filename and timestamp. No manual migration step is needed.

---

## Embedding Providers

Three providers are available behind an abstract `EmbeddingProvider` interface with `embed(text)` and `embed_batch(texts)` methods:

| Provider | Default Model | Notes |
|----------|--------------|-------|
| OpenAI | `text-embedding-3-small` | Uses `AsyncOpenAI` client |
| Google | `text-embedding-004` | Uses `genai.aio` async client |
| Ollama | `nomic-embed-text` | HTTP to `localhost:11434`, local inference |

Selected via `embedding.provider` in config. A factory function `get_embedder()` instantiates the correct provider with API keys from environment.

---

## Search Debug Mode

The `search_memories` MCP tool accepts a `debug: bool` parameter. When enabled, the response includes:

- Query text and similarity threshold
- Initial weight split (semantic vs keyword) and effective weights after low-spread detection
- Whether low-spread fallback to keyword-only triggered
- Scored semantic hits and keyword hits
- The top semantic result below threshold (if all were filtered out)

---

## Synthesis

When a retrieval query is triggered (via `?` prefix in Slack), the bot synthesizes an answer from search results using Claude Haiku via the Claude Agent SDK.

The synthesis prompt formats each result as `[index] (date) chunk_content` and instructs the model to answer using only the provided memories. If no search results exceed the similarity threshold, the bot replies with a static message and skips the LLM call entirely.
