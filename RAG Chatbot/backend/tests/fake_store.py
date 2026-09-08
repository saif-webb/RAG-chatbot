"""In-memory implementation of the `Store` protocol, for tests.

Mirrors the Postgres store's observable behaviour — unique content hashes,
cascade on delete, chunk replacement, ready-only search, cosine ranking — so the
pipeline tests exercise real logic without needing a database.

`tests/test_store.py` runs the same expectations against a real Postgres when
`TEST_DATABASE_URL` is set, which is what keeps this double honest.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import numpy as np

from app.db.base import StoreError
from app.db.models import ChunkRecord, Document, MessageRecord, SearchHit


class DuplicateContentHashError(StoreError):
    """Mirrors the UNIQUE constraint on documents.content_hash."""


class FakeStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._documents: dict[UUID, Document] = {}
        self._chunks: dict[UUID, list[tuple[UUID, ChunkRecord]]] = {}
        self._conversations: set[UUID] = set()
        self._messages: list[tuple[UUID, MessageRecord]] = []
        self._connected = False

    # --- lifecycle ---------------------------------------------------------

    async def connect(self) -> None:
        self._connected = True

    async def close(self) -> None:
        self._connected = False

    async def health(self) -> bool:
        return self._connected

    # --- documents ---------------------------------------------------------

    async def get_document(self, document_id: UUID) -> Document | None:
        return self._documents.get(document_id)

    async def get_document_by_hash(self, content_hash: str) -> Document | None:
        return next(
            (d for d in self._documents.values() if d.content_hash == content_hash),
            None,
        )

    async def create_document(
        self,
        *,
        document_id: UUID,
        filename: str,
        content_hash: str,
        mime_type: str | None,
        size_bytes: int,
    ) -> Document:
        if await self.get_document_by_hash(content_hash):
            raise DuplicateContentHashError(
                f"documents.content_hash is unique; {content_hash!r} already exists."
            )
        document = Document(
            id=document_id,
            filename=filename,
            content_hash=content_hash,
            mime_type=mime_type,
            size_bytes=size_bytes,
            status="pending",
            error=None,
            chunk_count=0,
            created_at=datetime.now(timezone.utc),
        )
        self._documents[document_id] = document
        return document

    async def list_documents(self) -> list[Document]:
        return sorted(self._documents.values(), key=lambda d: d.created_at, reverse=True)

    async def set_document_status(
        self,
        document_id: UUID,
        status: str,
        *,
        error: str | None = None,
        chunk_count: int | None = None,
    ) -> None:
        current = self._documents.get(document_id)
        if current is None:
            return
        self._documents[document_id] = replace(
            current,
            status=status,
            error=error,
            chunk_count=(current.chunk_count if chunk_count is None else chunk_count),
        )

    async def delete_document(self, document_id: UUID) -> bool:
        if document_id not in self._documents:
            return False
        del self._documents[document_id]
        self._chunks.pop(document_id, None)  # ON DELETE CASCADE
        return True

    # --- chunks ------------------------------------------------------------

    async def replace_chunks(self, document_id: UUID, chunks: list[ChunkRecord]) -> int:
        for chunk in chunks:
            if len(chunk.embedding) != self.dim:
                raise StoreError(
                    f"Embedding has {len(chunk.embedding)} dimensions but the "
                    f"store expects {self.dim}."
                )
        self._chunks[document_id] = [(uuid4(), c) for c in chunks]
        return len(chunks)

    async def search(self, embedding: list[float], top_k: int) -> list[SearchHit]:
        rows = [
            (document_id, chunk_id, chunk)
            for document_id, entries in self._chunks.items()
            if (doc := self._documents.get(document_id)) and doc.status == "ready"
            for chunk_id, chunk in entries
        ]
        if not rows:
            return []

        matrix = np.array([c.embedding for _, _, c in rows], dtype=np.float32)
        query = np.asarray(embedding, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1) * float(np.linalg.norm(query))
        scores = (matrix @ query) / np.maximum(norms, 1e-12)

        hits = [
            SearchHit(
                chunk_id=rows[i][1],
                document_id=rows[i][0],
                filename=self._documents[rows[i][0]].filename,
                page=rows[i][2].page,
                content=rows[i][2].content,
                score=float(scores[i]),
            )
            for i in np.argsort(-scores)[:top_k]
        ]
        return hits

    # --- conversations -----------------------------------------------------

    async def create_conversation(self, conversation_id: UUID) -> None:
        self._conversations.add(conversation_id)

    async def conversation_exists(self, conversation_id: UUID) -> bool:
        return conversation_id in self._conversations

    async def get_recent_messages(
        self, conversation_id: UUID, limit: int
    ) -> list[MessageRecord]:
        if limit <= 0:
            return []
        owned = [m for cid, m in self._messages if cid == conversation_id]
        return owned[-limit:]

    async def add_message(
        self,
        conversation_id: UUID,
        role: str,
        content: str,
        citations: list[dict[str, Any]] | None = None,
    ) -> None:
        self._messages.append(
            (
                conversation_id,
                MessageRecord(
                    role=role,
                    content=content,
                    citations=citations or [],
                    created_at=datetime.now(timezone.utc),
                ),
            )
        )
