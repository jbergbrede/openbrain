# Appendix: Hybrid Search

## Overview

This appendix extends the core system design with hybrid search — combining the existing semantic search with BM25-style keyword search via Postgres full-text search. This addresses a known weakness of pure semantic search: short or keyword-heavy queries (e.g., "Telekom", "INV-2026-0847") produce noisy, low-signal embeddings that fail to surface the right results.

Hybrid search solves this by running two search legs in parallel and merging results with adaptive weighting.

---

## The Problem With Pure Semantic Search

Short queries produce embeddings that lack sufficient signal for meaningful ranking:

```
Query: "Telekom"
Query embedding: [0.023, -0.041, ...]  ← 768 dimensions encoding just one word

Chunk embedding:  [0.019, -0.033, ...]  ← 768 dimensions encoding 500 tokens
                                          of rich context
```

The cosine similarity scores for all results cluster tightly together (e.g., 0.34–0.38), making the ranking effectively random. The semantic model cannot determine whether you mean a Telekom invoice, a Telekom contract, or a Telekom complaint.

---

## Two-Leg Hybrid Search

Every query runs two search legs in parallel:

```
Query
  ├── Semantic leg: pgvector cosine similarity (existing, unchanged)
  └── Keyword leg:  ts_rank against search_vector (new)
```

**Keyword search dominates for short queries** because literal string matching produces clear score separation between matches and non-matches. **Semantic search dominates for natural language queries** because embeddings capture intent and synonyms that keyword search misses.

### How Different Query Types Behave

| Query | Semantic | Keyword | Who wins |
|---|---|---|---|
| `"Telekom"` | Noisy, useless | Exact match | Keyword |
| `"INV-2026-0847"` | Meaningless | Exact match | Keyword |
| `"electricity bill"` | Decent | Good | Balanced |
| `"how much did I pay for internet last year"` | Understands intent | Partial matches | Semantic |
| `"documents about disputing a charge"` | Gets the concept | May miss "disputing" | Semantic |
| `"Telekom bill from January over 100 euros"` | Good | Good | Both |

---

## Schema Changes

Three additions to the `memories` table:

```sql
language         VARCHAR(2) NOT NULL DEFAULT 'en'  -- 'en' or 'de'
content_english  TEXT                               -- original or translated
content_german   TEXT                               -- original or translated
search_vector    TSVECTOR
```

A GIN index on `search_vector`:

```sql
CREATE INDEX idx_memories_search_vector ON memories USING GIN (search_vector);
```

The `search_vector` always combines both language columns — no branching on language required:

```sql
setweight(to_tsvector('english', COALESCE(content_english, '')), 'A') ||
setweight(to_tsvector('german',  COALESCE(content_german,  '')), 'A')
```

This makes every memory keyword-searchable in both languages from a single tsvector column.

---

## Language Detection

Language detection is handled by the **existing LLM enrichment step**. The structured output schema is extended to include a `language` field (`"en"` or `"de"`). No new dependencies are introduced — the LLM is more accurate than lightweight libraries, especially for short or mixed-language input.

### Cross-Language Retrieval

Every memory is translated to the other language during the existing LLM enrichment step. The detected language determines which column holds the original (`content_english` or `content_german`) and which holds the translation. Both are included in the `search_vector`, so a keyword search for "Rechnung" will match an English memory whose German translation contains that word — and vice versa.

This approach is:
- **Deterministic** — no reliance on the AI client expanding queries correctly
- **Complete** — every term is translated, not just the ones an agent thinks to expand
- **Free at query time** — translation is already done at ingestion
- **Negligible cost at ingestion** — piggybacks on the existing LLM enrichment call

---

## Adaptive Weighting

Weights adjust dynamically based on query length:

```python
def get_weights(query: str):
    word_count = len(query.split())
    if word_count <= 2:
        return 0.2, 0.8   # trust keywords
    elif word_count <= 5:
        return 0.5, 0.5   # balanced
    else:
        return 0.7, 0.3   # trust semantics
```

| Query length | Semantic weight | Keyword weight |
|---|---|---|
| 1–2 words | 20% | 80% |
| 3–5 words | 50% | 50% |
| 6+ words | 70% | 30% |

---

## Score Thresholding

When semantic scores cluster tightly together, the semantic leg has no useful signal and is discarded:

```python
score_spread = max(semantic_scores) - min(semantic_scores)

if score_spread < 0.05:
    # Fall back to keyword-only results
```

---

## Result Merging With Reciprocal Rank Fusion

Each leg returns its top-N results. Results are merged using Reciprocal Rank Fusion (RRF):

```
rrf_score(doc) = semantic_weight / (k + semantic_rank)
               + keyword_weight / (k + keyword_rank)
```

Where `k = 60` (standard RRF constant). After RRF scoring, the existing connection-count boost is applied on top for final ranking.

---

## Configuration

Added to the YAML config under search defaults:

```yaml
search:
  adaptive_weights: true
  score_spread_threshold: 0.05
  rrf_k: 60
```

---

## What Does Not Change

- **MCP tool signatures** — `search_memories` still accepts `query` and `limit`
- **Capture pipeline** — only addition is `search_vector` population alongside the existing insert
- **Slack bot** — no changes
- **Embedding provider** — no changes
- **Enrichment pipeline** — additions are `language`, `content_english`, and `content_german` in the structured output schema
