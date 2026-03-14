# Appendix: Slack Retrieval

## Overview

This appendix describes the retrieval extension to the existing Slack bot, which currently supports only memory capture. The bot is extended to support natural language querying of stored memories, triggered by a `?` prefix. The core library, search logic, and existing capture flows all remain unchanged.

---

## Trigger Logic

The bot distinguishes between **capture** and **retrieval** based on a single-character prefix:

- **No `?` prefix** → capture (save the message as a memory)
- **`?` prefix** → retrieval (search memories and return a synthesized answer)

This applies uniformly across both DMs and channel mentions.

---

## Context-Specific Behavior

**Direct Message (DM)**

- Capture: bot saves the message content as a memory (existing behavior, unchanged)
- Retrieval: bot replies directly in the DM conversation

**Channel Mention (`@memorybot`)**

- Capture: bot strips its own mention from the message text and saves the remainder as a memory, including channel and thread URL as source metadata
- Retrieval: bot strips its own mention, runs the retrieval pipeline, and replies **in a thread** under the original message

---

## Event Handling

The bot listens to two Slack event types:

- `message` events in DMs (existing)
- `app_mention` events in channels (new)

Routing logic:

```python
# DM handler
if message.startswith("?"):
    handle_retrieval(message[1:].strip(), reply_to=dm)
else:
    handle_capture(message)

# Channel mention handler
text = strip_mention(event["text"])
if text.startswith("?"):
    handle_retrieval(text[1:].strip(), reply_to=thread(event["ts"], event["channel"]))
else:
    handle_capture(text, source_metadata={channel, thread_url})
```

The `handle_retrieval` and `handle_capture` functions are shared across both contexts. The only difference is the reply target.

---

## Retrieval Pipeline

When a retrieval is triggered, the following steps execute in sequence:

**1. Strip prefix**

Remove the leading `?` and any surrounding whitespace to extract the raw query string.

**2. Hybrid search**

Call the existing `search_memories` core library function. This runs semantic and keyword search with adaptive weighting and Reciprocal Rank Fusion (RRF) merging. The top N results are returned, where N is configurable.

**3. Check results**

If no results exceed the similarity threshold, the bot replies:

> *"I couldn't find anything about that."*

No LLM call is made in this case.

**4. Synthesize**

Pass the query and the top results (chunk content and timestamps) to the LLM. The synthesis prompt instructs the model to answer the question using **only** the retrieved memories and nothing else.

**5. Reply**

Post the synthesized answer to the appropriate reply target, either a DM conversation or a channel thread.

---

## Configuration

A single new key is added to the YAML configuration:

```yaml
slack:
  retrieval_result_limit: 5
```

This controls how many top search results are passed to the synthesis step.

---

## What Does Not Change

- **Emoji react (🧠) capture** — untouched, continues to work as before
- **DM capture (no `?` prefix)** — untouched
- **Core library** — no modifications
- **MCP tools** — no modifications
- **Search logic** (semantic search, keyword search, RRF, adaptive weighting) — no modifications

The retrieval extension is purely additive. It introduces a new code path triggered by the `?` prefix and a new Slack event listener for `app_mention`. All existing functionality is preserved.