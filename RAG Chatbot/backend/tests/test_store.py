"""Integration tests for the real Postgres + pgvector store.

Skipped unless `TEST_DATABASE_URL` points at a Postgres with the pgvector
extension available. Locally:

    docker compose up -d
    set TEST_DATABASE_URL=postgresql://rag:rag@localhost:5432/rag   # Windows
    export TEST_DATABASE_URL=postgresql://rag:rag@localhost:5432/rag

These assert the same behaviour the in-memory `FakeStore` promises, which is
what stops the double from drifting away from the real thing.

⚠ The tables are dropped and recreated for each test. Point this at a scratch
database, never at one holding data you care about.
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest
import pytest_asyncio

from app.core.config import Settings
from app.db.base import DimensionMismatchError
from app.db.database import PostgresStore, normalize_dsn
from app.db.models import ChunkRecord

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="Set TEST_DATABASE_URL to run the Postgres integration tests.",
)

DIM = 8


def _unit(*values: float) -> list[float]:
    """Pad to DIM and normalise, so scores are readable cosine similarities."""
    vector = list(values) + [0.0] * (DIM - len(values))
    norm = sum(v * v for v in vector) ** 0.5
    return [v / norm for v in vector]


A = _unit(1.0, 0.0, 0.0)
B = _unit(0.0, 1.0, 0.0)
A_ISH = _unit(0.9, 0.1, 0.0)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        database_url=TEST_DATABASE_URL,
        gemini_api_key="test-key-not-used",
    )


@pytest_asyncio.fixture
async def store(settings: Settings):
    import asyncpg

    dsn, use_ssl = normalize_dsn(TEST_DATABASE_URL)
    connection = await asyncpg.connect(dsn=dsn, ssl=use_ssl or None)
    try:
        await connection.execute(
            "DROP TABLE IF EXISTS messages, chunks, conversations, documents CASCADE"
        )
    finally:
        await connection.close()

    instance = PostgresStore(settings, embed_dim=DIM)
    await instance.connect()
    try:
        yield instance
    finally:
        await instance.close()


async def _ready_document(store: PostgresStore, filename: str, chunks: list[ChunkRecord]):
    document = await store.create_document(
        document_id=uuid4(),
        filename=filename,
        content_hash=f"hash-{filename}",
        mime_type="text/plain",
        size_bytes=100,
    )
    await store.replace_chunks(document.id, chunks)
    await store.set_document_status(document.id, "ready", chunk_count=len(chunks))
    return document


def _chunk(index: int, text: str, embedding: list[float], page: int | None = 1):
    return ChunkRecord(chunk_index=index, content=text, page=page, embedding=embedding)


# --- schema ----------------------------------------------------------------


async def test_connect_creates_the_schema_and_extension(store: PostgresStore) -> None:
    assert await store.health() is True
    assert await store.list_documents() == []


async def test_dimension_mismatch_is_caught_at_startup(settings: Settings) -> None:
    """A pgvector column's dimension is fixed; changing models must fail loudly."""
    wrong = PostgresStore(settings, embed_dim=DIM + 1)
    with pytest.raises(DimensionMismatchError, match="DROP TABLE chunks"):
        await wrong.connect()
    await wrong.close()


# --- documents -------------------------------------------------------------


async def test_document_round_trip(store: PostgresStore) -> None:
    created = await store.create_document(
        document_id=uuid4(),
        filename="a.txt",
        content_hash="hash-a",
        mime_type="text/plain",
        size_bytes=42,
    )
    assert created.status == "pending"

    fetched = await store.get_document(created.id)
    assert fetched is not None
    assert fetched.filename == "a.txt"
    assert fetched.size_bytes == 42
    assert (await store.get_document_by_hash("hash-a")).id == created.id
    assert await store.get_document_by_hash("nope") is None


async def test_content_hash_is_unique(store: PostgresStore) -> None:
    import asyncpg

    await store.create_document(
        document_id=uuid4(),
        filename="a.txt",
        content_hash="same",
        mime_type=None,
        size_bytes=1,
    )
    with pytest.raises(asyncpg.UniqueViolationError):
        await store.create_document(
            document_id=uuid4(),
            filename="b.txt",
            content_hash="same",
            mime_type=None,
            size_bytes=1,
        )


async def test_status_update_preserves_chunk_count_when_omitted(
    store: PostgresStore,
) -> None:
    document = await _ready_document(store, "a.txt", [_chunk(0, "text", A)])
    await store.set_document_status(document.id, "processing")

    refreshed = await store.get_document(document.id)
    assert refreshed.status == "processing"
    assert refreshed.chunk_count == 1


# --- chunks and search -----------------------------------------------------


async def test_search_ranks_by_cosine_similarity(store: PostgresStore) -> None:
    await _ready_document(
        store,
        "doc.txt",
        [_chunk(0, "orthogonal", B), _chunk(1, "near", A_ISH), _chunk(2, "exact", A)],
    )

    hits = await store.search(A, top_k=3)
    assert [h.content for h in hits] == ["exact", "near", "orthogonal"]
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)
    assert hits[2].score == pytest.approx(0.0, abs=1e-4)


async def test_search_returns_page_and_filename_for_citations(
    store: PostgresStore,
) -> None:
    await _ready_document(store, "guide.pdf", [_chunk(0, "body", A, page=7)])
    hit = (await store.search(A, top_k=1))[0]
    assert hit.filename == "guide.pdf"
    assert hit.page == 7


async def test_search_ignores_documents_that_are_not_ready(
    store: PostgresStore,
) -> None:
    document = await _ready_document(store, "doc.txt", [_chunk(0, "body", A)])
    assert len(await store.search(A, top_k=5)) == 1

    await store.set_document_status(document.id, "processing")
    assert await store.search(A, top_k=5) == []


async def test_replace_chunks_swaps_rather_than_appends(store: PostgresStore) -> None:
    document = await _ready_document(
        store, "doc.txt", [_chunk(0, "first", A), _chunk(1, "second", A)]
    )
    assert len(await store.search(A, top_k=10)) == 2

    await store.replace_chunks(document.id, [_chunk(0, "only", A)])
    hits = await store.search(A, top_k=10)
    assert [h.content for h in hits] == ["only"]


async def test_wrong_dimension_is_rejected_before_insert(store: PostgresStore) -> None:
    from app.db.base import StoreError

    document = await store.create_document(
        document_id=uuid4(),
        filename="a.txt",
        content_hash="h",
        mime_type=None,
        size_bytes=1,
    )
    with pytest.raises(StoreError, match="dimensions"):
        await store.replace_chunks(document.id, [_chunk(0, "bad", [1.0, 2.0])])


# --- cascade ---------------------------------------------------------------


async def test_deleting_a_document_cascades_to_chunks(store: PostgresStore) -> None:
    """The whole point of the FK: deleted documents must stop being retrievable."""
    document = await _ready_document(store, "doc.txt", [_chunk(0, "body", A)])
    assert len(await store.search(A, top_k=5)) == 1

    assert await store.delete_document(document.id) is True
    assert await store.search(A, top_k=5) == []
    assert await store.get_document(document.id) is None


async def test_deleting_an_unknown_document_reports_false(
    store: PostgresStore,
) -> None:
    assert await store.delete_document(uuid4()) is False


# --- conversations ---------------------------------------------------------


async def test_conversation_and_message_round_trip(store: PostgresStore) -> None:
    conversation_id = uuid4()
    await store.create_conversation(conversation_id)
    assert await store.conversation_exists(conversation_id) is True

    await store.add_message(conversation_id, "user", "question")
    await store.add_message(
        conversation_id, "assistant", "answer", [{"index": 1, "filename": "a.txt"}]
    )

    messages = await store.get_recent_messages(conversation_id, 10)
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[1].citations[0]["filename"] == "a.txt"


async def test_create_conversation_is_idempotent(store: PostgresStore) -> None:
    conversation_id = uuid4()
    await store.create_conversation(conversation_id)
    await store.create_conversation(conversation_id)  # must not raise


async def test_recent_messages_are_capped_and_oldest_first(
    store: PostgresStore,
) -> None:
    conversation_id = uuid4()
    await store.create_conversation(conversation_id)
    for i in range(6):
        await store.add_message(conversation_id, "user", f"m{i}")

    messages = await store.get_recent_messages(conversation_id, 3)
    assert [m.content for m in messages] == ["m3", "m4", "m5"]
    assert await store.get_recent_messages(conversation_id, 0) == []
