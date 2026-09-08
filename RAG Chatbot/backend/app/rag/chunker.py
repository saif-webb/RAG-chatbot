"""Recursive character chunking that preserves page attribution.

Two properties matter here and neither is free:

1. **Chunks respect natural boundaries.** The splitter tries paragraph breaks
   first, then lines, then sentences, then words, and only falls back to a hard
   character cut when a single token is genuinely longer than `chunk_size`.
2. **Every chunk knows which page it came from.** Pages are concatenated once,
   with their character ranges recorded, so a chunk's page is looked up from the
   offset where it starts. Without this there is nothing to cite.

Everything in this module is pure — no I/O, no config, no network — which is
what makes it directly unit-testable.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", "; ", " ", "")

# Pages are joined with a blank line so a paragraph split is always available at
# a page boundary, and so offsets stay easy to reason about.
_PAGE_JOINER = "\n\n"


@dataclass(slots=True, frozen=True)
class Chunk:
    index: int
    content: str
    page: int | None


def chunk_pages(
    pages: Sequence[tuple[int | None, str]],
    *,
    chunk_size: int,
    chunk_overlap: int,
    separators: Sequence[str] = DEFAULT_SEPARATORS,
) -> list[Chunk]:
    """Split `(page_number, page_text)` pairs into overlapping chunks.

    A chunk is attributed to the page containing its first character, so a chunk
    spanning a page break cites where it started.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text, boundaries, page_numbers = _flatten(pages)
    if not text.strip():
        return []

    pieces = _split_recursive(text, 0, chunk_size, tuple(separators))
    merged = _merge(pieces, chunk_size, chunk_overlap)

    chunks: list[Chunk] = []
    for offset, body in merged:
        stripped = body.strip()
        if not stripped:
            continue
        start = offset + (len(body) - len(body.lstrip()))
        chunks.append(
            Chunk(
                index=len(chunks),
                content=stripped,
                page=_page_at(start, boundaries, page_numbers),
            )
        )
    return chunks


def chunk_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Convenience wrapper for unpaginated sources (plain text, Markdown)."""
    return chunk_pages([(None, text)], chunk_size=chunk_size, chunk_overlap=chunk_overlap)


# --- internals -------------------------------------------------------------


def _flatten(
    pages: Sequence[tuple[int | None, str]],
) -> tuple[str, list[int], list[int | None]]:
    """Concatenate pages, returning the text plus a searchable offset index."""
    parts: list[str] = []
    boundaries: list[int] = []
    page_numbers: list[int | None] = []
    cursor = 0

    for page, body in pages:
        if not body:
            continue
        if parts:
            cursor += len(_PAGE_JOINER)
            parts.append(_PAGE_JOINER)
        boundaries.append(cursor)
        page_numbers.append(page)
        parts.append(body)
        cursor += len(body)

    return "".join(parts), boundaries, page_numbers


def _page_at(
    offset: int, boundaries: list[int], page_numbers: list[int | None]
) -> int | None:
    if not boundaries:
        return None
    # Rightmost page whose start is <= offset.
    i = bisect.bisect_right(boundaries, offset) - 1
    return page_numbers[max(i, 0)]


def _split_recursive(
    text: str, offset: int, size: int, separators: tuple[str, ...]
) -> list[tuple[int, str]]:
    """Break `text` into pieces of at most `size` chars, keeping start offsets.

    Recurses through progressively finer separators; the empty separator is the
    terminal case and cuts on raw character count.
    """
    if len(text) <= size:
        return [(offset, text)] if text else []

    separator = separators[0] if separators else ""
    remaining = separators[1:] if separators else ()

    if separator == "":
        return [(offset + i, text[i : i + size]) for i in range(0, len(text), size)]

    if separator not in text:
        return _split_recursive(text, offset, size, remaining)

    out: list[tuple[int, str]] = []
    cursor = offset
    parts = text.split(separator)
    for i, part in enumerate(parts):
        # Re-attach the separator so offsets stay aligned with the source text.
        piece = part if i == len(parts) - 1 else part + separator
        if not piece:
            continue
        if len(piece) <= size:
            out.append((cursor, piece))
        else:
            out.extend(_split_recursive(piece, cursor, size, remaining))
        cursor += len(piece)
    return out


def _merge(
    pieces: list[tuple[int, str]], size: int, overlap: int
) -> list[tuple[int, str]]:
    """Greedily pack pieces into chunks, carrying `overlap` chars across the seam."""
    chunks: list[tuple[int, str]] = []
    current = ""
    current_offset = 0

    for offset, piece in pieces:
        if current and len(current) + len(piece) > size:
            chunks.append((current_offset, current))
            if overlap > 0:
                tail = current[-overlap:]
                current_offset = current_offset + len(current) - len(tail)
                current = tail + piece
            else:
                current_offset, current = offset, piece
        else:
            if not current:
                current_offset = offset
            current += piece

    if current:
        chunks.append((current_offset, current))
    return chunks
