"""Chunker tests.

The two properties worth protecting: chunks stay near the size budget, and every
chunk keeps the page it came from. Losing the second silently breaks citations
while everything else still appears to work.
"""

from __future__ import annotations

import pytest

from app.rag.chunker import chunk_pages, chunk_text


def test_short_text_is_a_single_chunk() -> None:
    chunks = chunk_text("A short sentence.", chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert chunks[0].content == "A short sentence."
    assert chunks[0].index == 0


def test_empty_and_whitespace_input_produce_nothing() -> None:
    assert chunk_text("", chunk_size=100, chunk_overlap=10) == []
    assert chunk_text("   \n\n  \t ", chunk_size=100, chunk_overlap=10) == []


def test_chunks_are_indexed_consecutively_from_zero() -> None:
    text = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_text(text, chunk_size=120, chunk_overlap=20)
    assert len(chunks) > 1
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunks_respect_the_size_budget() -> None:
    text = "\n\n".join(f"Paragraph number {i}. " * 8 for i in range(30))
    size, overlap = 300, 50
    chunks = chunk_text(text, chunk_size=size, chunk_overlap=overlap)

    # A chunk may exceed `size` only by the overlap carried across the seam.
    assert all(len(c.content) <= size + overlap for c in chunks)


def test_consecutive_chunks_overlap() -> None:
    text = " ".join(f"token{i}" for i in range(300))
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=60)
    assert len(chunks) >= 2

    # The tail of one chunk must reappear at the head of the next, otherwise a
    # fact straddling the boundary becomes unretrievable.
    first_tail = chunks[0].content[-30:]
    assert first_tail.split()[0] in chunks[1].content


def test_no_content_is_dropped_between_chunks() -> None:
    words = [f"w{i}" for i in range(500)]
    chunks = chunk_text(" ".join(words), chunk_size=150, chunk_overlap=30)
    joined = " ".join(c.content for c in chunks)
    assert all(word in joined for word in words)


def test_a_single_oversized_token_is_hard_split() -> None:
    chunks = chunk_text("x" * 1000, chunk_size=100, chunk_overlap=0)
    assert len(chunks) == 10
    assert all(len(c.content) == 100 for c in chunks)


def test_page_numbers_are_carried_onto_chunks() -> None:
    pages = [
        (1, "Alpha content about badgers. " * 10),
        (2, "Beta content about otters. " * 10),
        (3, "Gamma content about herons. " * 10),
    ]
    chunks = chunk_pages(pages, chunk_size=150, chunk_overlap=20)

    assert {c.page for c in chunks} == {1, 2, 3}
    # Pages are concatenated in order, so page numbers must never go backwards.
    seen = [c.page for c in chunks]
    assert seen == sorted(seen)

    for chunk in chunks:
        expected = {1: "Alpha", 2: "Beta", 3: "Gamma"}[chunk.page]
        assert expected in chunk.content or chunk.content.startswith(("content", "about"))


def test_unpaginated_sources_have_no_page() -> None:
    chunks = chunk_text("Some plain text file content.", chunk_size=100, chunk_overlap=0)
    assert all(c.page is None for c in chunks)


def test_empty_pages_are_skipped() -> None:
    chunks = chunk_pages(
        [(1, ""), (2, "Real content here."), (3, "   ")],
        chunk_size=100,
        chunk_overlap=0,
    )
    assert len(chunks) == 1
    assert chunks[0].page == 2


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_text("anything", chunk_size=100, chunk_overlap=100)


def test_zero_overlap_is_allowed() -> None:
    chunks = chunk_text(
        " ".join(f"w{i}" for i in range(200)), chunk_size=100, chunk_overlap=0
    )
    assert len(chunks) > 1
