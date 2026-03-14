

# Appendix: Chunking on Ingestion

## Overview

This appendix extends the core system design with chunking on ingestion — splitting longer memories into smaller pieces, each with its own embedding, to improve retrieval precision for memos, Slack threads, and pasted articles.

Without chunking, a long memory receives a single embedding of its full content. That embedding is diluted across too many topics to match specific queries reliably. Chunking ensures that a query about "Q3 budget" surfaces the paragraph that discusses it, not a vague match against a 2000-word memo.

---

## The Problem With Single-Embedding Memories

Short memories embed well — a single thought fits cleanly into an embedding vector. But longer content produces diluted embeddings:

```
Query: "Q3 budget"
Query embedding: [0.041, -0.023, ...]  ← focused signal

Memory embedding: [0.019, -0.011, ...]  ← 2000 words averaged together,
                                          budget paragraph's signal is buried
```

Cosine similarity scores cluster tightly, making ranking unreliable. The right paragraph exists in the memory — but the embedding can't surface it.

---

## Schema Changes

### New `chunks` table

- `id` — UUID, primary key
- `memory_id` — UUID, foreign key to `memories.id`, ON DELETE CASCADE
- `chunk_index` — INTEGER, 0-based position within the memory
- `content` — TEXT, the chunk text
- `embedding` — VECTOR(dims), chunk-level embedding (dimension matches the configured embedding model)
- `token_count` — INTEGER, for diagnostics and tuning

### Changes to `memories` table

- **Drop** the `embedding` column — all embeddings now live on chunks
- **Keep** the `search_vector` (tsvector) column — keyword search remains at memory level for bilingual coverage

---

## Chunking Strategy

1. **Short content bypass:** if content is ≤ ~200 tokens, create a single chunk with the full content — no splitting needed
2. **Structural split first:**
   - Split on markdown headers (`## ...`)
   - Then on double newlines (paragraph breaks)
   - Then on single newlines
3. **Token ceiling:** if any resulting section still exceeds ~500 tokens, split it on sentence boundaries
4. **Overlap:** ~50 tokens prepended from the previous chunk's tail to preserve continuity across chunk boundaries

This strategy respects the natural structure of the content — a paragraph about "Q3 budget" stays together rather than getting sliced mid-sentence — while the token ceiling acts as a safety net for oversized sections.

---

## Ingestion Flow

Ingestion is **synchronous** — the `save_memory` call blocks until all chunks are created and embedded. Given the low write volume of personal use, the added latency is negligible, and the memory is fully searchable the moment `save_memory` returns.

**Step-by-step:**

1. **Insert memory** into `memories` table (no embedding column — embeddings now live on chunks)
2. **LLM enrichment** — extract title, tags, category, connections, and translation (English ↔ German) — unchanged from current pipeline
3. **Build combined tsvector on `memories`** from both language versions of the content (unchanged)
4. **Chunk the original content** using the chunking strategy above → produces N chunks
5. **For each chunk:**
   - Compute embedding from the original language chunk text via the embedding provider
   - Insert into `chunks` table with `memory_id`, `chunk_index`, `content`, `embedding`, and `token_count`

**Key decisions:**

- **Translation stays at memory level.** The LLM enrichment already processes the full memory to extract tags, title, connections — translation happens there in one call. Translating per-chunk would mean N additional LLM calls for marginal benefit.
- **tsvector stays at memory level.** The memory-level tsvector already covers both language versions. Adding chunk-level tsvectors would produce duplicates without adding retrieval value, since the bilingual memory-level tsvector matches the same keywords and more.
- **Embeddings use the original language.** Modern embedding models are multilingual, so a German query will match a German chunk effectively.

---

## Search Changes

Search becomes a two-leg hybrid with mixed granularity:

- **Semantic leg** → queries `chunks.embedding` → returns chunk IDs with cosine similarity scores
- **Keyword leg** → queries `memories.search_vector` → returns memory IDs with ts_rank scores (unchanged)

### Score Reconciliation

The two legs operate at different granularities — chunks for semantic, memories for keyword. Before merging, chunk scores are promoted to memory level: for each memory, take the **max cosine similarity score** across all its chunks. This produces a single semantic score per memory that can be merged directly with the keyword leg's ts_rank score using the existing adaptive weighting formula.

After merging and ranking memories, the **top-scoring chunk(s)** from each winning memory are returned as the result — giving the caller a focused snippet rather than the full memory.

### MCP Tool Interface

- The `search` tool response includes chunk content plus `memory_id`, so the LLM can call `get_memory` for full context if needed
- The LLM can also traverse connections from the parent memory to explore related content

---

## What Doesn't Change

- **`memories` table structure** (aside from dropping the `embedding` column)
- **LLM enrichment pipeline** — tags, connections, translation still operate on the full memory
- **`get_memory` tool** — still returns the full memory content
- **Connection graph** — connections remain between memories, not chunks
- **Hybrid search weighting logic** — the adaptive weighting formula stays the same, just applied after chunk-to-memory score promotion

---

## Cost Considerations

Embedding API calls scale with the number of chunks. A 2000-word memo might produce 5–6 chunks, meaning 5–6 embedding calls instead of 1. For personal use volume this is negligible, but worth monitoring if ingestion volume grows significantly.

---

## Future Improvements

- **Language-aware keyword search at chunk level** — detect language per chunk and apply language-specific tsvector configs for better stemming. Worth its own design pass.
- **Chunk preference boost** — add a weighting axis that favors chunk-level precision over memory-level keyword matches. Can be layered on later if retrieval quality feels off; easier to add than to remove.
- **Language-aware tsvector at chunk level** — detect language per chunk and apply language-specific tsvector configs for better stemming. Worth its own design pass if keyword recall on long multilingual memories feels weak.