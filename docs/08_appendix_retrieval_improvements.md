# Appendix 08: Retrieval Quality Improvements

## Overview

This appendix describes three improvements to retrieval quality, addressing a class of failures where relevant memories exist but are not surfaced by the current search pipeline. The root causes are vocabulary mismatch (query terms differ from stored terms) and weak query signal (short queries produce diluted embeddings).

The improvements are ordered by implementation phase and can be adopted incrementally. Each phase is independently useful.

---

## The Problem

Two failure modes dominate:

**1. Vocabulary mismatch**

A dental cleaning invoice contains `"Zahnreinigung - Dr. Müller - €89,00"` but the user searches for `"dentist"`. Neither search leg finds it:

- **Semantic:** the embedding for `"dentist"` is not close enough to `"Zahnreinigung"` — they are related but not synonymous in embedding space
- **Keyword:** `"dentist"` does not appear in the content or its German translation

The information exists, but no query phrasing the user would naturally try will surface it.

**2. Weak query signal**

Short queries like `"electricity bill"` produce embeddings that lack sufficient dimensionality to rank results meaningfully. Cosine similarity scores cluster tightly, and the correct result may rank below unrelated content. The keyword leg helps for exact matches, but cannot bridge conceptual gaps.

---

## Phase 1: MCP Tool Description + Contextual Chunk Headers

### MCP Tool Description

The current `search_memories` tool description does not guide the AI client on query formulation. Clients (Claude, etc.) frequently compress natural language questions into 2–3 keyword fragments before calling the tool, degrading both search legs.

**Current description:**

```
Search the user's personal memory store using hybrid semantic + keyword search.
Use this proactively whenever the user asks about anything personal — past events,
people, places, finances, appointments, invoices, health, travel, or any information
they may have previously stored. Returns matching memories. Set debug=true for
explainability info.
```

**Updated description:**

```
Search the user's personal memory store using hybrid semantic + keyword search.
Use this proactively whenever the user asks about anything personal — past events,
people, places, finances, appointments, invoices, health, travel, or any information
they may have previously stored. Returns matching memories. Set debug=true for
explainability info.

Query guidelines: Pass all relevant details from the user's question — including
names, dates, amounts, and specific terms. Do not omit details, but also do not
pad the query with filler. For example, prefer "electricity bill January Stadtwerke"
over just "electricity bill", but do not wrap it in a full sentence.
```

This balances both search legs: the keyword leg gets precise, high-signal terms, and the semantic leg gets enough context for a meaningful embedding. The adaptive weighting stays balanced instead of being forced toward semantic by artificially long queries.

### Contextual Chunk Headers

Chunks from longer memories often lack context about what the parent memory is about. A chunk containing `"€127.43 — higher than last month"` embeds poorly because it has no grounding signal.

**Change:** Before computing the chunk embedding, prepend the memory's title and tags as a brief prefix:

```
[Electricity Bill - January 2026 | finances, utilities] €127.43 — higher than last month...
```

The prefix is used **only for embedding computation** — the stored `chunks.content` remains unchanged. This ensures the embedding captures what the chunk is about, not just what it literally says.

**Migration:** Existing chunks must be re-embedded with the new prefix. This is a one-time batch job — iterate all chunks, prepend their parent memory's title and tags, recompute embeddings, and update in place.

---

## Phase 2: Enrichment Pipeline — Inferred Keywords and Synthetic Questions

Two new fields are added to the LLM enrichment output schema, addressing vocabulary mismatch at ingestion time rather than query time.

### Inferred Keywords

Keywords that are **not already present** in the content but that a user might search for. These bridge vocabulary gaps by capturing synonyms, broader categories, colloquial terms, and cross-language equivalents.

**Added to enrichment prompt:**

```
- keywords: list of 10-15 search terms NOT already in the content that someone
  might use to find this information. Include synonyms, broader categories,
  colloquial terms, and related concepts in both English and German.
```

**Example — dental cleaning invoice:**

```json
"keywords": [
  "dentist", "Zahnarzt", "dental", "teeth", "Zähne",
  "health", "Gesundheit", "medical", "invoice", "Rechnung",
  "healthcare", "Vorsorge", "checkup", "hygiene"
]
```

**Storage:** Keywords are appended to the `search_vector` tsvector on the `memories` table, making them searchable via the keyword leg without any search-side changes:

```sql
setweight(to_tsvector('english', COALESCE(content_english, '')), 'A') ||
setweight(to_tsvector('german',  COALESCE(content_german,  '')), 'A') ||
setweight(to_tsvector('simple',  COALESCE(keywords_text,   '')), 'B')
```

Keywords use the `simple` text search config (no stemming) and weight `B` to rank slightly below direct content matches.

### Synthetic Questions

Natural language questions that the memory would answer. These are embedded as additional vectors pointing to the same chunk, enabling question-to-question matching — which produces significantly higher cosine similarity than question-to-document matching.

**Added to enrichment prompt:**

```
- questions: list of 3-5 natural language questions this content would answer.
  Include both English and German questions. Be specific — reference concrete
  details from the content rather than generating generic questions.
```

**Example — dental cleaning invoice:**

```json
"questions": [
  "How much did the dental cleaning cost?",
  "Wie viel hat die Zahnreinigung gekostet?",
  "When was my last dentist appointment?",
  "Wann war mein letzter Zahnarzttermin?",
  "What did Dr. Müller charge?"
]
```

### Schema Changes

New columns on `memories`:

- `keywords` — TEXT[], the inferred keyword list
- `questions` — TEXT[], the synthetic question list

New rows in `chunks`: for each synthetic question, an additional chunk row is inserted with:

- `memory_id` — same parent memory
- `chunk_index` — set to `-1` (or a dedicated flag) to distinguish from content chunks
- `content` — the question text
- `embedding` — embedding of the question text
- `is_synthetic` — BOOLEAN, `true`

Synthetic question chunks participate in the semantic leg of search like any other chunk. The max-score-per-memory promotion handles deduplication — if a synthetic question chunk scores highest, it surfaces the parent memory. The actual content chunks are returned to the caller, not the synthetic question.

### Updated Enrichment Prompt

```
You are an AI assistant that extracts structured information from personal notes
and thoughts. Extract the following from the user's text and return valid JSON only.

Fields:
- summary: concise 1-2 sentence summary
- people: list of people mentioned (first/last names, use empty list if none)
- topics: list of relevant topics/themes (prefer reusing existing topics from the
  provided list)
- action_items: list of tasks/todos, each with "text" and optional "due_date"
  (ISO date string or null)
- language: detected language of the input, either "en" or "de"
- content_english: the content in English (original if input is English, translation
  if input is German)
- content_german: the content in German (original if input is German, translation
  if input is English)
- keywords: list of 10-15 search terms NOT already in the content that someone
  might use to find this information. Include synonyms, broader categories,
  colloquial terms, and related concepts in both English and German.
- questions: list of 3-5 natural language questions this content would answer.
  Include both English and German questions. Be specific — reference concrete
  details from the content rather than generating generic questions.

Return ONLY a JSON object with these exact keys. No markdown, no explanation.
```

### Cost Considerations

The enrichment LLM call already processes the full memory content. Adding `keywords` and `questions` to the output schema increases output tokens by ~100–200 per memory — negligible cost increase. Synthetic question embeddings add 3–5 embedding API calls per memory on top of the content chunk embeddings.

### Migration

Existing memories must be re-enriched to populate `keywords` and `questions`. This is a batch job:

1. For each memory, call the updated enrichment prompt with the original content
2. Store the new `keywords` and `questions` fields
3. Rebuild `search_vector` to include keywords
4. Generate and embed synthetic question chunks

---

## Phase 3: Split Query Paths

If retrieval gaps remain after phases 1 and 2, the search pipeline can be extended with per-leg query optimization. Instead of passing the same query to both search legs, each leg receives a query tailored to its strengths.

### Design

```python
async def search_memories(query: str, limit: int = 5):
    # Keyword leg: use original query as-is
    keyword_results = await keyword_search(query, limit)

    # Semantic leg: use LLM-expanded query
    enriched_query = await expand_for_semantic(query)
    semantic_results = await semantic_search(enriched_query, limit)

    # Adaptive weighting uses ORIGINAL query length
    sem_weight, kw_weight = get_weights(query)

    # Merge with RRF as before
    return rrf_merge(semantic_results, keyword_results, sem_weight, kw_weight)
```

The expansion function generates a richer search paragraph for the semantic leg:

```python
async def expand_for_semantic(query: str) -> str:
    prompt = """Rewrite this search query as a detailed paragraph that captures
    the intent, synonyms, and related concepts. Keep it under 100 words.
    Do NOT answer the question — just expand the search terms.
    Include both English and German terms where relevant.

    Query: {query}

    Expanded search:"""

    return await llm_call(prompt.format(query=query))
```

### Key Design Decisions

- **Adaptive weighting uses original query length.** The expanded query is always 6+ words, which would force 70/30 semantic/keyword weighting. But a user who typed `"Telekom"` still needs heavy keyword weighting — the expansion is there to help the semantic leg, not to override the weighting logic.
- **Keyword leg is never modified.** The original query goes directly to BM25 without any LLM processing. This preserves exact-match behavior for identifiers, invoice numbers, and proper nouns.
- **Expansion failure falls back gracefully.** If the LLM call fails, the original query is used for both legs. Search degrades to current behavior rather than breaking.

### Latency Impact

Each search now includes an LLM call (~0.5–2 seconds depending on model and provider). For the Slack `?` path, this is acceptable — users expect a brief delay for a synthesized answer, and the synthesis step already includes an LLM call. For the MCP path, the added latency compounds with the client's own LLM processing, which may be noticeable.

A cheap, fast model (Claude Haiku, GPT-4o-mini) is recommended for expansion to minimize latency.

### Configuration

```yaml
search:
  query_expansion: true
  query_expansion_model: "haiku"  # cheap and fast
```

---

## What Does Not Change

- **Chunking strategy** — token ceiling, overlap, structural splitting all remain as-is
- **RRF merging** — same algorithm, same `k` constant
- **Score thresholding** — low-spread detection and keyword-only fallback unchanged
- **Connection graph** — connections remain between memories, not chunks
- **Capture pipeline** — emoji react, DM, @mention, and Paperless-ngx capture flows unchanged (enrichment output schema is extended, not replaced)
- **MCP tool signatures** — `search_memories` still accepts `query`, `limit`, and `debug`

---

## Summary

| Phase | Change | Effort | Latency Impact | Addresses |
|-------|--------|--------|----------------|-----------|
| 1 | MCP tool description | 5 minutes | None | Keyword compression by AI clients |
| 1 | Contextual chunk headers | ~1 hour + migration | None | Context-poor chunk embeddings |
| 2 | Inferred keywords | ~2 hours + migration | None (ingestion only) | Vocabulary mismatch (keyword leg) |
| 2 | Synthetic questions | ~2 hours + migration | None (ingestion only) | Vocabulary mismatch (semantic leg) |
| 3 | Split query paths | ~3 hours | +0.5–2s per search | Weak query signal for semantic leg |
