from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken


def _get_encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_get_encoding().encode(text))


def _decode_tokens(tokens: list[int]) -> str:
    return _get_encoding().decode(tokens)


def _encode(text: str) -> list[int]:
    return _get_encoding().encode(text)


@dataclass
class RawChunk:
    index: int
    content: str
    token_count: int


def _split_structural(text: str) -> list[str]:
    """Split on markdown headers, then double newlines. Preserves multi-line blocks."""
    # Split on markdown headers (## or higher)
    header_parts = re.split(r"(?m)^(?=#{1,6} )", text)
    sections: list[str] = []
    for part in header_parts:
        if not part.strip():
            continue
        # Split on double newlines — preserves lists, code blocks, tables
        para_parts = re.split(r"\n\n+", part)
        for para in para_parts:
            if para.strip():
                sections.append(para.strip())
    return sections


def _split_sentences(text: str) -> list[str]:
    """Split text on sentence boundaries."""
    # Split on '. ', '? ', '! ', keeping delimiter with preceding sentence
    parts = re.split(r"(?<=[.?!])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _add_overlap(chunks: list[str], overlap_tokens: int) -> list[str]:
    """Prepend tail tokens from previous chunk to each chunk."""
    if len(chunks) <= 1:
        return chunks
    result = [chunks[0]]
    enc = _get_encoding()
    for i in range(1, len(chunks)):
        prev_tokens = enc.encode(chunks[i - 1])
        overlap = prev_tokens[-overlap_tokens:] if len(prev_tokens) > overlap_tokens else prev_tokens
        overlap_text = enc.decode(overlap)
        result.append(overlap_text + " " + chunks[i])
    return result


def chunk_content(
    content: str,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
    short_threshold: int = 200,
) -> list[RawChunk]:
    """Split content into chunks for embedding."""
    if not content.strip():
        return [RawChunk(index=0, content=content, token_count=count_tokens(content))]

    total_tokens = count_tokens(content)
    if total_tokens <= short_threshold:
        return [RawChunk(index=0, content=content, token_count=total_tokens)]

    # Structural split into candidate sections
    sections = _split_structural(content)

    # Merge small adjacent sections, split oversized ones
    merged: list[str] = []
    current = ""
    current_tokens = 0

    for section in sections:
        section_tokens = count_tokens(section)

        if section_tokens > max_tokens:
            # Flush current buffer first
            if current.strip():
                merged.append(current.strip())
                current = ""
                current_tokens = 0
            # Sentence-split the oversized section
            sentences = _split_sentences(section)
            sent_buf = ""
            sent_buf_tokens = 0
            for sent in sentences:
                sent_tokens = count_tokens(sent)
                if sent_buf_tokens + sent_tokens > max_tokens and sent_buf.strip():
                    merged.append(sent_buf.strip())
                    sent_buf = sent
                    sent_buf_tokens = sent_tokens
                else:
                    sent_buf = (sent_buf + " " + sent).strip() if sent_buf else sent
                    sent_buf_tokens += sent_tokens
            if sent_buf.strip():
                merged.append(sent_buf.strip())
        elif current_tokens + section_tokens > max_tokens:
            if current.strip():
                merged.append(current.strip())
            current = section
            current_tokens = section_tokens
        else:
            current = (current + "\n\n" + section).strip() if current else section
            current_tokens += section_tokens

    if current.strip():
        merged.append(current.strip())

    # Add overlap — token_count reflects final content including overlap
    with_overlap = _add_overlap(merged, overlap_tokens)

    enc = tiktoken.get_encoding("cl100k_base")
    return [
        RawChunk(index=i, content=text, token_count=len(enc.encode(text)))
        for i, text in enumerate(with_overlap)
    ]
