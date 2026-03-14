# Appendix 07: Paperless-ngx Document Capture

## Overview

This appendix describes how scanned and ingested documents from a **Paperless-ngx** instance are automatically captured into the memory system via Slack. It introduces no new pipeline logic — documents are fed through the existing capture flow (chunking, enrichment, embedding, storage) — but adds a new event listener and an upsert mechanism keyed on the Paperless document ID.

---

## Architecture

The integration consists of two pieces:

1. **Paperless-ngx → Slack (push):** A webhook configured in Paperless-ngx posts a message to a dedicated Slack channel whenever a document is consumed or updated.
2. **Slack Bot → Paperless API (pull):** The bot detects messages in that channel, extracts the document ID, fetches the full OCR text and metadata from the Paperless API, and hands it off to the existing capture pipeline.

### Sequence

```
Paperless-ngx
  │
  ├── Webhook POST ──► Slack Channel (dedicated #paperless channel)
  │                         │
  │                         ├── Bot auto-triggers (every message in this channel)
  │                         │
  │                         ├── Parses document ID from message
  │                         │
  │                         ├── GET /api/documents/{id}/ ──► Paperless API
  │                         │       (returns: title, content, tags, correspondent)
  │                         │
  │                         ├── Upsert check (does a memory with this doc ID exist?)
  │                         │       YES → delete existing memory + chunks (CASCADE)
  │                         │       NO  → continue
  │                         │
  │                         ├── Feed OCR text + metadata into capture pipeline
  │                         │       (chunking → enrichment → embedding → storage)
  │                         │
  │                         └── React with ✅ on success, ❌ on error
```

---

## Paperless-ngx Webhook Configuration

In **Paperless-ngx → Settings → Workflows**, create a workflow:

- **Trigger:** Document Added / Consumed (and optionally Document Updated, for tag changes)
- **Action:** Webhook
- **URL:** The Slack **Incoming Webhook URL** for the dedicated channel
- **Method:** POST
- **Headers:** `Content-Type: application/json`
- **Body** (Slack-compatible format):

```json
{
  "text": "New document consumed: *{{title}}* [id:{{document_id}}]",
  "attachments": [{
    "fields": [
      { "title": "ID", "value": "{{document_id}}", "short": true },
      { "title": "Correspondent", "value": "{{correspondent}}", "short": true },
      { "title": "Tags", "value": "{{tags}}", "short": false }
    ]
  }]
}
```

> **Note:** The `[id:{{document_id}}]` token in the text is intentional — the bot uses a simple regex to extract it. Check which placeholders your Paperless-ngx version supports. At minimum `document_id` is always available to fetch everything else via the API.

---

## Bot-Side Logic

### Trigger

The bot listens for **every message** in the configured Paperless channel. No emoji react or mention is required — every message in this channel represents a document event.

### Steps

**1. Parse the document ID**

Extract the document ID from the Slack message text using a regex pattern like `\[id:(\d+)\]`.

**2. Fetch document content and metadata from the Paperless API**

```
GET /api/documents/{id}/
Authorization: Token <PAPERLESS_API_TOKEN>
```

This returns the full OCR-extracted text in the `content` field, along with title, tags, correspondent, created date, and other metadata. No PDF download is needed — the `content` field already contains the extracted text.

**3. Upsert check**

Query `source_metadata` for an existing memory with `paperless_document_id` matching the current document ID.

- If found: delete the existing memory and its chunks (CASCADE), then proceed to save a fresh version.
- If not found: proceed directly to save.

This is the same delete-and-replace pattern used for Slack thread upsert, where the key is `(channel, thread_ts)`. Here the key is simply `paperless_document_id`.

**4. Feed into the capture pipeline**

Hand the OCR text and metadata to the existing capture pipeline — identical to any other memory source. The text is chunked, enriched by the LLM (which will pick up the Paperless tags as topics), embedded, and stored in pgvector.

**5. React on the Slack message**

Add a ✅ emoji on success, ❌ on error. No reply message is posted. This is consistent with the audio capture appendix.

---

## Why Upsert Matters

The typical Paperless-ngx workflow is:

1. Scan a document — it lands in Paperless with the tag `inbox`
2. At some later point, review the document and assign proper tags, correspondent, document type, etc.

Without upsert, the memory captured at step 1 would contain raw OCR text but no meaningful tags. The memory would never be updated when the document is properly categorized at step 2.

With upsert keyed on `paperless_document_id`:

- The first webhook (document consumed) creates the initial memory — even if it only has the `inbox` tag
- The second webhook (document updated, tags changed) replaces the memory with a fresh version that includes the curated tags
- The final memory always reflects the latest state of the document in Paperless-ngx

This means Paperless tags directly influence the `topics` field assigned during LLM enrichment, improving retrieval quality.

---

## Source Metadata

Memories created from Paperless-ngx documents carry the following `source_metadata`:

- `source_type`: `"paperless"`
- `paperless_document_id`: the integer document ID from Paperless
- `paperless_title`: the document title
- `paperless_correspondent`: the assigned correspondent (if any)
- `paperless_tags`: list of tag names assigned in Paperless

---

## Configuration Additions

```yaml
slack:
  paperless_channel_id: "C0123456789"

paperless:
  base_url: "https://paperless.example.com"
  api_token: ""  # sourced from env var: PAPERLESS_API_TOKEN
```

The `api_token` must be stored as an environment variable or in the bot's secrets store — never embedded in Slack messages or committed to source control.

---

## What Does Not Change

This appendix introduces:

- A new event listener (messages in the Paperless channel)
- A new API client (Paperless-ngx REST API)
- An upsert check before saving

It does **not** change:

- The capture pipeline (chunking, enrichment, embedding, storage)
- The database schema (source_metadata already supports arbitrary keys)
- The MCP tools or search behavior
- Any other capture method (emoji react, DM, audio)

Paperless-ngx is simply another source that feeds into the same pipeline.