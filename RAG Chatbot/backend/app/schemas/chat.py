"""Wire models for the chat endpoint."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """One retrieved chunk, as shown to the user beneath an answer.

    `index` is the bracket number the model is told to cite: `[1]`, `[2]`, ...
    """

    index: int
    document_id: UUID
    filename: str
    page: int | None = None
    score: float
    snippet: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    conversation_id: UUID | None = None


class ChatResponse(BaseModel):
    """Non-streaming shape. The SSE stream emits these fields as separate events."""

    conversation_id: UUID
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool = True
