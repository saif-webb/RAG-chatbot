-- Schema for Postgres + pgvector.
--
-- Idempotent on purpose: app/db/database.py runs this at startup, so a fresh
-- deploy against an empty database provisions itself with no manual migration
-- step. You can also apply it by hand:
--
--     psql "$DATABASE_URL" -f scripts/init_db.sql      -- after substituting 384
--
-- {EMBED_DIM} is substituted at runtime with the dimension reported by the
-- loaded embedding model (384 for all-MiniLM-L6-v2). A pgvector column's
-- dimension is fixed at creation time, so switching to a model with a different
-- output size requires `DROP TABLE chunks;` and a re-ingest. Startup compares
-- the live column against the model and refuses to run on a mismatch.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id            UUID PRIMARY KEY,
    filename      TEXT        NOT NULL,
    content_hash  TEXT        NOT NULL UNIQUE,
    mime_type     TEXT,
    size_bytes    BIGINT      NOT NULL DEFAULT 0,
    status        TEXT        NOT NULL DEFAULT 'pending',
    error         TEXT,
    chunk_count   INTEGER     NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id           UUID PRIMARY KEY,
    document_id  UUID    NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INTEGER NOT NULL,
    content      TEXT    NOT NULL,
    page         INTEGER,
    embedding    VECTOR({EMBED_DIM}) NOT NULL
);

-- ON DELETE CASCADE above is what makes deleting a document actually remove its
-- vectors. Without it, deleted documents keep polluting search results.
CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (document_id);

-- HNSW with cosine distance, matching the `<=>` operator used by the search
-- query. Embeddings are stored unit-normalised, so cosine and inner product
-- rank identically; cosine is kept for readability of the 1 - distance score.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS conversations (
    id         UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
    id               UUID PRIMARY KEY,
    conversation_id  UUID        NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role             TEXT        NOT NULL,
    content          TEXT        NOT NULL,
    citations        JSONB,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS messages_conversation_idx
    ON messages (conversation_id, created_at);
