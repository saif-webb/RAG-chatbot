"""Internal record types passed between the store and the RAG pipeline.

These are deliberately plain dataclasses rather than Pydantic models: they never
touch the wire. The API layer converts them into `app.schemas.*` models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID


@dataclass(slots=True)
class Document:
    id: UUID
    filename: str
    content_hash: str
    mime_type: str | None
    size_bytes: int
    status: str
    error: str | None
    chunk_count: int
    created_at: datetime


@dataclass(slots=True)
class ChunkRecord:
    """A chunk ready to be persisted.

    `embedding` must have exactly as many floats as the embedding model reports,
    which is the width the pgvector column was created with.
    """

    chunk_index: int
    content: str
    page: int | None
    embedding: list[float]


@dataclass(slots=True)
class SearchHit:
    chunk_id: UUID
    document_id: UUID
    filename: str
    page: int | None
    content: str
    score: float  # cosine similarity in [-1, 1]; higher is more similar


@dataclass(slots=True)
class MessageRecord:
    role: str  # "user" | "assistant"
    content: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime | None = None
