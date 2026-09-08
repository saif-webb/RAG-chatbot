"""End-to-end pipeline tests.

Loading, chunking, retrieval, thresholding, citation assembly and orchestration
are the production code paths. The store, the embedding model and Gemini are
substituted, so the suite needs no database, no network and no API key.

The Postgres store itself is covered by `test_store.py`.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.rag.pipeline import RagPipeline
from app.rag.prompt import NO_CONTEXT_ANSWER
from tests.conftest import StubEmbedder, StubLLM
from tests.fake_store import DuplicateContentHashError, FakeStore

PHOTOSYNTHESIS = (
    b"Photosynthesis converts light energy into chemical energy. Chlorophyll in "
    b"the chloroplast absorbs sunlight. The light dependent reactions split water "
    b"and release oxygen. The Calvin cycle then fixes carbon dioxide into glucose. "
)

BOOKKEEPING = (
    b"Double entry bookkeeping records every transaction twice, as a debit and a "
    b"credit. The ledger must balance. Depreciation spreads the cost of an asset "
    b"across its useful life. "
)


async def _ingest(pipeline: RagPipeline, store: FakeStore, name: str, body: bytes):
    document = await store.create_document(
        document_id=uuid4(),
        filename=name,
        content_hash=f"hash-{name}",
        mime_type="text/plain",
        size_bytes=len(body),
    )
    await pipeline.ingest(document.id, name, body)
    return await store.get_document(document.id)


# --- ingestion -------------------------------------------------------------


async def test_ingest_marks_the_document_ready_with_chunks(
    pipeline: RagPipeline, store: FakeStore
) -> None:
    document = await _ingest(pipeline, store, "biology.txt", PHOTOSYNTHESIS)
    assert document.status == "ready"
    assert document.chunk_count > 0
    assert document.error is None


async def test_ingest_records_failure_instead_of_raising(
    pipeline: RagPipeline, store: FakeStore
) -> None:
    """A background task that raised would leave the row stuck on 'processing'."""
    document = await _ingest(pipeline, store, "photo.jpeg", b"\xff\xd8\xff binary")
    assert document.status == "failed"
    assert "supported" in (document.error or "").lower()


async def test_ingest_of_an_unreadable_file_fails_cleanly(
    pipeline: RagPipeline, store: FakeStore
) -> None:
    document = await _ingest(pipeline, store, "blank.txt", b"    \n\n   ")
    assert document.status == "failed"
    assert document.chunk_count == 0


async def test_reingesting_replaces_chunks_rather_than_duplicating(
    pipeline: RagPipeline, store: FakeStore, embedder: StubEmbedder
) -> None:
    document = await _ingest(pipeline, store, "biology.txt", PHOTOSYNTHESIS)
    first = document.chunk_count

    await pipeline.ingest(document.id, "biology.txt", PHOTOSYNTHESIS)
    again = await store.get_document(document.id)
    assert again.chunk_count == first

    # The row count must match too — a bug that appended instead of replacing
    # would keep chunk_count right while doubling the stored vectors.
    query = await embedder.embed_query("chlorophyll sunlight oxygen glucose")
    assert len(await store.search(query, 100)) == first


async def test_documents_are_embedded_with_the_document_task(
    pipeline: RagPipeline, store: FakeStore, embedder: StubEmbedder
) -> None:
    await _ingest(pipeline, store, "biology.txt", PHOTOSYNTHESIS)
    assert embedder.document_calls, "documents should go through embed_documents"
    assert not embedder.query_calls, "ingestion must not use the query embedding"


# --- retrieval and answering ----------------------------------------------


async def test_relevant_question_is_answered_with_citations(
    pipeline: RagPipeline, store: FakeStore, llm: StubLLM
) -> None:
    await _ingest(pipeline, store, "biology.txt", PHOTOSYNTHESIS)

    result = await pipeline.answer("What does chlorophyll absorb during photosynthesis?")

    assert result.grounded
    assert result.answer
    assert result.citations
    assert result.citations[0].index == 1
    assert result.citations[0].filename == "biology.txt"
    assert llm.stream_prompts, "a grounded question should reach the LLM"


async def test_irrelevant_question_is_refused_without_calling_the_llm(
    pipeline: RagPipeline, store: FakeStore, llm: StubLLM
) -> None:
    """The threshold guard is the main anti-hallucination lever."""
    await _ingest(pipeline, store, "biology.txt", PHOTOSYNTHESIS)

    result = await pipeline.answer("What were the quarterly tariffs on imported tin?")

    assert not result.grounded
    assert result.answer == NO_CONTEXT_ANSWER
    assert result.citations == []
    assert not llm.stream_prompts, "the LLM must not be called with no context"


async def test_questions_with_no_documents_at_all_are_refused(
    pipeline: RagPipeline, llm: StubLLM
) -> None:
    result = await pipeline.answer("Anything at all?")
    assert not result.grounded
    assert not llm.stream_prompts


async def test_retrieval_prefers_the_matching_document(
    pipeline: RagPipeline, store: FakeStore
) -> None:
    await _ingest(pipeline, store, "biology.txt", PHOTOSYNTHESIS)
    await _ingest(pipeline, store, "accounting.txt", BOOKKEEPING)

    result = await pipeline.answer("Explain debit credit ledger depreciation")

    assert result.grounded
    assert result.citations[0].filename == "accounting.txt"


async def test_citations_are_numbered_consecutively(
    pipeline: RagPipeline, store: FakeStore
) -> None:
    await _ingest(pipeline, store, "biology.txt", PHOTOSYNTHESIS)
    result = await pipeline.answer("chlorophyll sunlight oxygen carbon glucose")
    assert [c.index for c in result.citations] == list(
        range(1, len(result.citations) + 1)
    )


async def test_only_ready_documents_are_searchable(
    pipeline: RagPipeline, store: FakeStore
) -> None:
    document = await _ingest(pipeline, store, "biology.txt", PHOTOSYNTHESIS)
    await store.set_document_status(document.id, "processing")

    result = await pipeline.answer("What does chlorophyll absorb?")
    assert not result.grounded, "a half-ingested document must not leak into answers"


# --- conversation memory ---------------------------------------------------


async def test_first_question_is_not_condensed(
    pipeline: RagPipeline, store: FakeStore, llm: StubLLM
) -> None:
    await _ingest(pipeline, store, "biology.txt", PHOTOSYNTHESIS)
    await pipeline.answer("What does chlorophyll do?")
    assert not llm.complete_prompts, "nothing to condense against on turn one"


async def test_follow_up_is_condensed_into_a_standalone_question(
    pipeline: RagPipeline, store: FakeStore, llm: StubLLM
) -> None:
    """Without this, 'what about that?' embeds into a meaningless vector."""
    await _ingest(pipeline, store, "biology.txt", PHOTOSYNTHESIS)

    first = await pipeline.answer("What does chlorophyll absorb?")
    llm.condensed = "What does the Calvin cycle fix carbon dioxide into?"

    second = await pipeline.answer(
        "And what about that other stage?", first.conversation_id
    )

    assert llm.complete_prompts, "the follow-up should trigger condensing"
    assert "What does chlorophyll absorb?" in llm.complete_prompts[0]
    assert second.grounded, "condensing should recover retrievable wording"


async def test_conversation_history_is_persisted(
    pipeline: RagPipeline, store: FakeStore
) -> None:
    await _ingest(pipeline, store, "biology.txt", PHOTOSYNTHESIS)
    result = await pipeline.answer("What does chlorophyll absorb?")

    messages = await store.get_recent_messages(result.conversation_id, 10)
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[0].content == "What does chlorophyll absorb?"


async def test_condense_failure_falls_back_to_the_raw_message(
    pipeline: RagPipeline, store: FakeStore, llm: StubLLM
) -> None:
    from app.rag.llm import LLMError

    await _ingest(pipeline, store, "biology.txt", PHOTOSYNTHESIS)
    first = await pipeline.answer("What does chlorophyll absorb?")

    async def boom(*args, **kwargs):
        raise LLMError("provider down")

    llm.complete = boom  # type: ignore[method-assign]

    result = await pipeline.answer("chlorophyll sunlight", first.conversation_id)
    assert result.grounded, "a condensing failure must not break the answer"


# --- deletion --------------------------------------------------------------


async def test_deleting_a_document_removes_its_vectors(
    pipeline: RagPipeline, store: FakeStore
) -> None:
    """ON DELETE CASCADE is what stops deleted documents polluting results."""
    document = await _ingest(pipeline, store, "biology.txt", PHOTOSYNTHESIS)

    before = await pipeline.answer("What does chlorophyll absorb?")
    assert before.grounded

    assert await store.delete_document(document.id) is True

    after = await pipeline.answer("What does chlorophyll absorb?")
    assert not after.grounded
    assert await store.get_document(document.id) is None


async def test_deleting_an_unknown_document_reports_false(
    store: FakeStore,
) -> None:
    assert await store.delete_document(uuid4()) is False


# --- deduplication ---------------------------------------------------------


async def test_content_hash_is_unique_per_document(store: FakeStore) -> None:
    await store.create_document(
        document_id=uuid4(),
        filename="a.txt",
        content_hash="same",
        mime_type="text/plain",
        size_bytes=10,
    )
    found = await store.get_document_by_hash("same")
    assert found is not None and found.filename == "a.txt"

    with pytest.raises(DuplicateContentHashError):
        await store.create_document(
            document_id=uuid4(),
            filename="b.txt",
            content_hash="same",
            mime_type="text/plain",
            size_bytes=10,
        )
