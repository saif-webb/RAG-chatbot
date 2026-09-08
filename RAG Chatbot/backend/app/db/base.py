"""The storage contract.

`PostgresStore` in `database.py` is the only production implementation. The
protocol exists so the test suite can substitute an in-memory double and run
without a database, and so the pipeline never depends on driver details.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from app.db.models import ChunkRecord, Document, MessageRecord, SearchHit


class StoreError(RuntimeError):
    """Raised for storage problems that should surface as a clean 5xx."""


class DimensionMismatchError(StoreError):
    """The persisted vector dimension does not match the loaded embedding model.

    Recovering means dropping the chunks table and re-ingesting, so this is
    raised at startup rather than being discovered at the first insert.
    """


@runtime_checkable
class Store(Protocol):
    # --- lifecycle ---------------------------------------------------------
    async def connect(self) -> None:
        """Open the pool and create the schema if absent. Idempotent."""

    async def close(self) -> None: ...

    async def health(self) -> bool:
        """Cheap round-trip used by /healthz."""

    # --- documents ---------------------------------------------------------
    async def get_document(self, document_id: UUID) -> Document | None: ...

    async def get_document_by_hash(self, content_hash: str) -> Document | None: ...

    async def create_document(
        self,
        *,
        document_id: UUID,
        filename: str,
        content_hash: str,
        mime_type: str | None,
        size_bytes: int,
    ) -> Document: ...

    async def list_documents(self) -> list[Document]: ...

    async def set_document_status(
        self,
        document_id: UUID,
        status: str,
        *,
        error: str | None = None,
        chunk_count: int | None = None,
    ) -> None: ...

    async def delete_document(self, document_id: UUID) -> bool:
        """Delete the document and, by cascade, all of its chunks."""

    # --- chunks ------------------------------------------------------------
    async def replace_chunks(self, document_id: UUID, chunks: list[ChunkRecord]) -> int:
        """Atomically swap in a fresh set of chunks for one document.

        Replacing rather than appending makes re-ingestion idempotent instead of
        duplicating every chunk.
        """

    async def search(self, embedding: list[float], top_k: int) -> list[SearchHit]:
        """Nearest neighbours by cosine similarity, across ready documents only."""

    # --- conversations -----------------------------------------------------
    async def create_conversation(self, conversation_id: UUID) -> None: ...

    async def conversation_exists(self, conversation_id: UUID) -> bool: ...

    async def get_recent_messages(
        self, conversation_id: UUID, limit: int
    ) -> list[MessageRecord]:
        """The most recent `limit` messages, returned oldest-first."""

    async def add_message(
        self,
        conversation_id: UUID,
        role: str,
        content: str,
        citations: list[dict[str, Any]] | None = None,
    ) -> None: ...
