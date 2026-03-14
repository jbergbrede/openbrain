# Appendix: Audio Capture

## Overview

Extend the Slack bot to support audio file attachments as memory input. When a captured message (via emoji react or DM) contains an audio file, the bot downloads it, transcribes it using the OpenAI Whisper API, and feeds the transcript into the existing memory pipeline as raw content.

## Flow

1. **Acknowledge start** — React with ⏳ to indicate processing has begun
2. **Detection** — Check if the message contains a `files` attachment with an audio MIME type
3. **Download** — Use the Slack file URL and bot token to download the audio file into memory (bytes, not disk)
4. **Transcribe** — Send the audio bytes to the OpenAI Whisper API (`/v1/audio/transcriptions`)
5. **Save** — Pass the transcript as raw content into the existing memory save pipeline (enrichment, chunking, embedding, storage)
6. **Metadata** — Add `source_type: "audio"` and `duration_seconds` to the memory metadata
7. **Acknowledge end** — Remove ⏳ and react with ✅ on success, or ❌ on error

## Supported Sources

- **Slack recorded audio** — produces `.mp4` or `.webm`
- **iPhone voice memos** — uploaded as `.m4a`
- Any other format natively supported by the Whisper API: `mp3`, `mp4`, `mpeg`, `mpga`, `m4a`, `wav`, `webm`

No format conversion step is needed. Unsupported formats are rejected by the API and handled as errors.

## Mixed Content Handling

### Single message with text and audio

Both are captured. The text content comes first, followed by the audio transcript, separated by a newline.

### Threads with mixed content

The existing thread upsert logic applies. All messages in the thread are processed in chronological order:

- Text messages contribute their text
- Audio messages contribute their transcript
- Messages with both contribute text followed by transcript

The result is concatenated into a single memory. Delete-and-replace behavior is unchanged: re-reacting rebuilds the memory from the full thread.

## Audio File Lifecycle

The audio file is held in memory as bytes only for the duration of processing. It is never written to disk or stored permanently. After transcription, the bytes are discarded. The transcript is the only artifact that persists.

The original file remains available in Slack subject to the workspace's file retention policy.

## Error Handling

If any error occurs during capture — audio or otherwise — the bot reacts with ❌ instead of ✅. This applies to:

- **Whisper API rejection** — unsupported format
- **File too large** — Whisper API has a 25MB limit
- **Whisper API failure** — network or service errors
- **Enrichment failure** — LLM errors during the enrichment step
- **Storage failure** — database errors during save

Both the ⏳ → ✅/❌ reaction flow and the ❌ error reaction are introduced globally for all memory capture, not just audio.

## What Doesn't Change

- **Enrichment pipeline** — the transcript is plain text input, same as any other memory
- **Chunking, embedding, storage** — all unchanged
- **MCP retrieval** — audio-originated memories are searchable like any other

## Metadata

Audio-originated memories include two additional metadata fields:

- `source_type: "audio"` — distinguishes them from text-originated memories
- `duration_seconds` — the duration of the original audio file

## Dependencies

- `openai` Python package (already in use for embeddings) — provides the Whisper API client
- No new infrastructure, services, or libraries required