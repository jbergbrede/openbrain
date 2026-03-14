from __future__ import annotations

from openbrain.chunker import chunk_content, count_tokens


def test_short_content_single_chunk():
    text = "A short thought."
    chunks = chunk_content(text, short_threshold=200)
    assert len(chunks) == 1
    assert chunks[0].content == text
    assert chunks[0].index == 0


def test_empty_content_single_chunk():
    chunks = chunk_content("", short_threshold=200)
    assert len(chunks) == 1


def test_token_count_accuracy():
    text = "Hello world"
    chunks = chunk_content(text, short_threshold=200)
    assert chunks[0].token_count == count_tokens(text)


def test_markdown_header_split():
    text = "## Section One\n\n" + ("alpha " * 60) + "\n\n## Section Two\n\n" + ("beta " * 60)
    chunks = chunk_content(text, max_tokens=100, short_threshold=10)
    assert len(chunks) >= 2
    # Section One content should appear in an early chunk
    full = " ".join(c.content for c in chunks)
    assert "alpha" in full
    assert "beta" in full


def test_paragraph_split():
    # Two paragraphs that together exceed max_tokens but each fits individually
    para1 = "word " * 60  # ~60 tokens
    para2 = "thing " * 60  # ~60 tokens
    text = para1.strip() + "\n\n" + para2.strip()

    chunks = chunk_content(text, max_tokens=80, short_threshold=10)
    assert len(chunks) >= 2


def test_sentence_split_on_long_section():
    # One long run-on paragraph with sentence breaks
    sentences = ["This is sentence number %d." % i for i in range(50)]
    text = " ".join(sentences)
    chunks = chunk_content(text, max_tokens=100, short_threshold=10)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.token_count <= 200  # reasonable ceiling with overlap


def test_overlap_present():
    # Each section is around 80 tokens; with overlap the next chunk starts with tail of prev
    section1 = "alpha " * 80
    section2 = "beta " * 80
    text = section1.strip() + "\n\n" + section2.strip()

    chunks = chunk_content(text, max_tokens=120, overlap_tokens=10, short_threshold=10)
    assert len(chunks) >= 2
    # Second chunk should begin with overlap from first chunk (contains "alpha")
    assert "alpha" in chunks[1].content


def test_chunk_indices_are_sequential():
    text = "\n\n".join(["word " * 60 for _ in range(5)])
    chunks = chunk_content(text, max_tokens=80, short_threshold=10)
    for i, chunk in enumerate(chunks):
        assert chunk.index == i


def test_all_content_preserved():
    """All original words appear somewhere across chunks."""
    unique_words = ["uniqueword%d" % i for i in range(10)]
    sections = [word + " " + ("filler " * 55) for word in unique_words]
    text = "\n\n".join(sections)

    chunks = chunk_content(text, max_tokens=100, short_threshold=10)
    all_text = " ".join(c.content for c in chunks)
    for word in unique_words:
        assert word in all_text
