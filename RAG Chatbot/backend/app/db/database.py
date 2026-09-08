"""Postgres + pgvector — the only store.

Uses asyncpg directly rather than an ORM. The schema is four tables of
hand-written SQL, and going driver-level sidesteps the well-known friction
between SQLAlchemy's asyncpg dialect, pgvector's type codec, and PgBouncer.

The vector dimension is supplied by the loaded embedding model, not by
configuration, so the column and the vectors written into it cannot disagree.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import UUID, uuid4

import asyncpg
import numpy as np

from app.core.config import PROJECT_ROOT, Settings
from app.db.base import DimensionMismatchError, StoreError
from app.db.models import ChunkRecord, Document, MessageRecord, SearchHit

logger = logging.getLogger(__name__)

SCHEMA_PATH = PROJECT_ROOT / "scripts" / "init_db.sql"

_DOC_COLUMNS = (
    "id, filename, content_hash, mime_type, size_bytes, "
    "status, error, chunk_count, created_at"
)

# asyncpg takes these as connect() keyword arguments, not as DSN query
# parameters, and raises on ones it does not recognise. Managed providers hand
# you a URL containing several of them, so they are stripped here.
_DSN_PARAMS_TO_STRIP = {
    "sslmode",
    "channel_binding",
    "options",
    "target_session_attrs",
    "connect_timeout",
    "application_name",
    "gssencmode",
    "pgbouncer",
}


def normalize_dsn(raw: str) -> tuple[str, bool]:
    """Return `(dsn, use_ssl)` with driver-specific query parameters removed.

    Managed providers hand out URLs like
    ``postgresql://u:p@host/db?sslmode=require&channel_binding=require``.
    asyncpg rejects those parameters, so they are translated into the `ssl`
    connect argument. A ``postgresql+asyncpg://`` scheme (SQLAlchemy style) is
    also accepted and normalised.
    """
    parts = urlsplit(raw.strip())

    scheme = parts.scheme.split("+", 1)[0]
    if scheme == "postgres":
        scheme = "postgresql"
    if scheme != "postgresql":
        raise StoreError(
            f"DATABASE_URL must be a postgresql:// URL, got {parts.scheme!r}://"
        )

    kept: list[tuple[str, str]] = []
    sslmode = "prefer"
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key.lower() == "sslmode":
            sslmode = value
        elif key.lower() not in _DSN_PARAMS_TO_STRIP:
            kept.append((key, value))

    dsn = urlunsplit((scheme, parts.netloc, parts.path, urlencode(kept), ""))
    return dsn, sslmode not in {"disable", "allow", "prefer"}


def _warn_if_unpooled(dsn: str) -> None:
    host = urlsplit(dsn).hostname or ""
    if "neon.tech" in host and "-pooler." not in host:
        logger.warning(
            "DATABASE_URL points at a direct Neon endpoint. Use the pooled one "
            "(insert '-pooler' into the endpoint id) so the app survives "
            "connection churn."
        )
    elif "supabase" in host and "pooler.supabase.com" not in host:
        logger.warning(
            "DATABASE_URL points at a direct Supabase endpoint, which is IPv6-only "
            "without the paid IPv4 add-on. Use the transaction pooler on port 6543."
        )


class PostgresStore:
    def __init__(self, settings: Settings, embed_dim: int) -> None:
        self._settings = settings
        self._dim = embed_dim
        self._pool: asyncpg.Pool | None = None

    @property
    def dim(self) -> int:
        return self._dim

    # --- lifecycle ---------------------------------------------------------

    async def connect(self) -> None:
        dsn, use_ssl = normalize_dsn(self._settings.database_url)
        _warn_if_unpooled(dsn)

        # The pgvector codec can only be registered once the extension exists,
        # so the schema is created on a throwaway connection first.
        bootstrap = await asyncpg.connect(dsn=dsn, ssl=use_ssl or None)
        try:
            await self._apply_schema(bootstrap)
            await self._verify_dimension(bootstrap)
        finally:
            await bootstrap.close()

        self._pool = await asyncpg.create_pool(
            dsn=dsn,
            ssl=use_ssl or None,
            min_size=self._settings.db_pool_min,
            max_size=self._settings.db_pool_max,
            # PgBouncer in transaction mode cannot see server-side prepared
            # statements across pooled connections; disabling the cache is the
            # documented way to stay compatible. Harmless on a direct connection.
            statement_cache_size=0,
            init=_register_vector_codec,
            command_timeout=60,
        )
        logger.info("Connected to Postgres (vector dim=%d)", self._dim)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def health(self) -> bool:
        async with self._acquire() as conn:
            return await conn.fetchval("SELECT 1") == 1

    async def _apply_schema(self, conn: asyncpg.Connection) -> None:
        sql = Path(SCHEMA_PATH).read_text(encoding="utf-8")
        try:
            await conn.execute(sql.replace("{EMBED_DIM}", str(self._dim)))
        except asyncpg.InsufficientPrivilegeError as exc:
            raise StoreError(
                "Could not create the pgvector extension. Ask a superuser to run "
                "`CREATE EXTENSION vector;` on this database once, then restart. "
                f"({exc})"
            ) from exc

    async def _verify_dimension(self, conn: asyncpg.Connection) -> None:
        """Fail loudly when the stored dimension no longer matches the model.

        pgvector keeps a column's dimension in `atttypmod`. Silently continuing
        would either error on every insert or, worse, mix incompatible vectors.
        """
        actual = await conn.fetchval(
            """
            SELECT atttypmod
            FROM pg_attribute
            WHERE attrelid = 'chunks'::regclass
              AND attname = 'embedding'
              AND NOT attisdropped
            """
        )
        if actual is not None and actual > 0 and actual != self._dim:
            raise DimensionMismatchError(
                f"chunks.embedding is vector({actual}) but the embedding model "
                f"{self._settings.embedding_model} produces {self._dim} dimensions. "
                f"A pgvector column's dimension is fixed at creation. To switch "
                f"models, re-ingest from scratch:\n"
                f"    DROP TABLE chunks;\n"
                f"then restart the app and re-upload your documents."
            )

    def _acquire(self) -> Any:
        if self._pool is None:
            raise StoreError("Store is not connected; call connect() first.")
        return self._pool.acquire()

    # --- documents ---------------------------------------------------------

    async def get_document(self, document_id: UUID) -> Document | None:
        async with self._acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT {_DOC_COLUMNS} FROM documents WHERE id = $1", document_id
            )
        return _row_to_document(row) if row else None

    async def get_document_by_hash(self, content_hash: str) -> Document | None:
        async with self._acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT {_DOC_COLUMNS} FROM documents WHERE content_hash = $1",
                content_hash,
            )
        return _row_to_document(row) if row else None

    async def create_document(
        self,
        *,
        document_id: UUID,
        filename: str,
        content_hash: str,
        mime_type: str | None,
        size_bytes: int,
    ) -> Document:
        async with self._acquire() as conn:
            row = await conn.fetchrow(
                f"""
                INSERT INTO documents
                    (id, filename, content_hash, mime_type, size_bytes, status)
                VALUES ($1, $2, $3, $4, $5, 'pending')
                RETURNING {_DOC_COLUMNS}
                """,
                document_id,
                filename,
                content_hash,
                mime_type,
                size_bytes,
            )
        return _row_to_document(row)

    async def list_documents(self) -> list[Document]:
        async with self._acquire() as conn:
            rows = await conn.fetch(
                f"SELECT {_DOC_COLUMNS} FROM documents ORDER BY created_at DESC"
            )
        return [_row_to_document(r) for r in rows]

    async def set_document_status(
        self,
        document_id: UUID,
        status: str,
        *,
        error: str | None = None,
        chunk_count: int | None = None,
    ) -> None:
        async with self._acquire() as conn:
            await conn.execute(
                """
                UPDATE documents
                   SET status = $2,
                       error = $3,
                       chunk_count = COALESCE($4, chunk_count)
                 WHERE id = $1
                """,
                document_id,
                status,
                error,
                chunk_count,
            )

    async def delete_document(self, document_id: UUID) -> bool:
        async with self._acquire() as conn:
            result = await conn.execute(
                "DELETE FROM documents WHERE id = $1", document_id
            )
        return result.rsplit(" ", 1)[-1] != "0"

    # --- chunks ------------------------------------------------------------

    async def replace_chunks(self, document_id: UUID, chunks: list[ChunkRecord]) -> int:
        rows = [
            (
                uuid4(),
                document_id,
                c.chunk_index,
                c.content,
                c.page,
                _as_vector(c.embedding, self._dim),
            )
            for c in chunks
        ]
        async with self._acquire() as conn, conn.transaction():
            await conn.execute("DELETE FROM chunks WHERE document_id = $1", document_id)
            if rows:
                await conn.executemany(
                    """
                    INSERT INTO chunks
                        (id, document_id, chunk_index, content, page, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    rows,
                )
        return len(rows)

    async def search(self, embedding: list[float], top_k: int) -> list[SearchHit]:
        vector = _as_vector(embedding, self._dim)
        async with self._acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT c.id,
                       c.document_id,
                       d.filename,
                       c.page,
                       c.content,
                       1 - (c.embedding <=> $1) AS score
                  FROM chunks c
                  JOIN documents d ON d.id = c.document_id
                 WHERE d.status = 'ready'
                 ORDER BY c.embedding <=> $1
                 LIMIT $2
                """,
                vector,
                top_k,
            )
        return [
            SearchHit(
                chunk_id=r["id"],
                document_id=r["document_id"],
                filename=r["filename"],
                page=r["page"],
                content=r["content"],
                score=float(r["score"]),
            )
            for r in rows
        ]

    # --- conversations -----------------------------------------------------

    async def create_conversation(self, conversation_id: UUID) -> None:
        async with self._acquire() as conn:
            await conn.execute(
                "INSERT INTO conversations (id) VALUES ($1) ON CONFLICT DO NOTHING",
                conversation_id,
            )

    async def conversation_exists(self, conversation_id: UUID) -> bool:
        async with self._acquire() as conn:
            return await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM conversations WHERE id = $1)",
                conversation_id,
            )

    async def get_recent_messages(
        self, conversation_id: UUID, limit: int
    ) -> list[MessageRecord]:
        if limit <= 0:
            return []
        async with self._acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT role, content, citations, created_at
                  FROM messages
                 WHERE conversation_id = $1
                 ORDER BY created_at DESC, id DESC
                 LIMIT $2
                """,
                conversation_id,
                limit,
            )
        return [
            MessageRecord(
                role=r["role"],
                content=r["content"],
                citations=json.loads(r["citations"]) if r["citations"] else [],
                created_at=r["created_at"],
            )
            for r in reversed(rows)
        ]

    async def add_message(
        self,
        conversation_id: UUID,
        role: str,
        content: str,
        citations: list[dict[str, Any]] | None = None,
    ) -> None:
        async with self._acquire() as conn:
            await conn.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content, citations)
                VALUES ($1, $2, $3, $4, $5)
                """,
                uuid4(),
                conversation_id,
                role,
                content,
                json.dumps(citations) if citations else None,
            )


def create_store(settings: Settings, embed_dim: int) -> PostgresStore:
    return PostgresStore(settings, embed_dim)


async def _register_vector_codec(conn: asyncpg.Connection) -> None:
    from pgvector.asyncpg import register_vector

    await register_vector(conn)


def _row_to_document(row: asyncpg.Record) -> Document:
    return Document(
        id=row["id"],
        filename=row["filename"],
        content_hash=row["content_hash"],
        mime_type=row["mime_type"],
        size_bytes=row["size_bytes"],
        status=row["status"],
        error=row["error"],
        chunk_count=row["chunk_count"],
        created_at=row["created_at"],
    )


def _as_vector(embedding: list[float], expected_dim: int) -> np.ndarray:
    if len(embedding) != expected_dim:
        raise StoreError(
            f"Embedding has {len(embedding)} dimensions but the schema expects "
            f"{expected_dim}."
        )
    return np.asarray(embedding, dtype=np.float32)
