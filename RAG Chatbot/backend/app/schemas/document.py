"""Wire models for the document endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

DocumentStatus = Literal["pending", "processing", "ready", "failed"]


class DocumentOut(BaseModel):
    # Built from the `app.db.models.Document` dataclass.
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    mime_type: str | None = None
    size_bytes: int = 0
    status: DocumentStatus
    error: str | None = None
    chunk_count: int = 0
    created_at: datetime

    # True when this upload matched an existing document by content hash and
    # was therefore not re-embedded.
    duplicate: bool = False


class DocumentListOut(BaseModel):
    documents: list[DocumentOut] = Field(default_factory=list)


class DeleteResultOut(BaseModel):
    id: UUID
    deleted: bool
