"""The RAG orchestrator — the only module the API layer talks to.

Two flows live here:

**Ingest** (background): load -> chunk -> embed -> store. Runs detached from the
upload request so the HTTP call returns immediately, and records `failed` with a
message rather than leaving a document stuck on `processing`.

**Answer** (streamed): condense -> embed -> search -> guard -> generate. The
condense and guard steps are what separate this from naive RAG:

* *Condense* rewrites a follow-up into a standalone question. Embedding "what
  about the second one?" verbatim produces a vector that means nothing.
* *Guard* returns a grounded refusal without calling the LLM when nothing clears
  the similarity threshold. A model handed irrelevant passages will cheerfully
  invent an answer from them.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import UUID, uuid4

from app.core.config import Settings
from app.db.base import Store
from app.db.models import ChunkRecord
from app.rag import prompt as prompts
from app.rag.chunker import chunk_pages
from app.rag.embeddings import Embedder, EmbeddingError
from app.rag.llm import LLMClient, LLMError
from app.rag.loader import EmptyDocumentError, UnsupportedFileError, load_pages
from app.rag.retriever import Retriever
from app.schemas.chat import Citation

logger = logging.getLogger(__name__)

EventType = Literal["meta", "token", "citations", "done", "error"]


@dataclass(slots=True)
class StreamEvent:
    type: EventType
    data: Any


@dataclass(slots=True)
class Answer:
    conversation_id: UUID
    answer: str
    citations: list[Citation] = field(default_factory=list)
    grounded: bool = True


class RagPipeline:
    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        llm: LLMClient,
        settings: Settings,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._llm = llm
        self._settings = settings
        self._retriever = Retriever(store, embedder, settings)

    # --- ingestion ---------------------------------------------------------

    async def ingest(self, document_id: UUID, filename: str, data: bytes) -> None:
        """Parse, chunk, embed and store one document. Never raises.

        Failures are written to the document row instead, so the UI can show why
        an upload did not work and the user can delete and retry.
        """
        try:
            await self._store.set_document_status(document_id, "processing")

            pages = load_pages(filename, data)
            chunks = chunk_pages(
                pages,
                chunk_size=self._settings.chunk_size,
                chunk_overlap=self._settings.chunk_overlap,
            )
            if not chunks:
                raise EmptyDocumentError(f"'{filename}' produced no usable chunks.")

            logger.info(
                "Embedding %d chunks from '%s' (%d pages)",
                len(chunks),
                filename,
                len(pages),
            )
            vectors = await self._embedder.embed_documents([c.content for c in chunks])
            if len(vectors) != len(chunks):
                raise EmbeddingError(
                    f"Embedder returned {len(vectors)} vectors for {len(chunks)} chunks."
                )

            stored = await self._store.replace_chunks(
                document_id,
                [
                    ChunkRecord(
                        chunk_index=chunk.index,
                        content=chunk.content,
                        page=chunk.page,
                        embedding=vector,
                    )
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ],
            )
            await self._store.set_document_status(
                document_id, "ready", error=None, chunk_count=stored
            )
            logger.info("Ingested '%s' as %d chunks", filename, stored)

        except (UnsupportedFileError, EmptyDocumentError) as exc:
            await self._fail(document_id, str(exc))
        except (EmbeddingError, LLMError) as exc:
            await self._fail(document_id, f"Embedding failed: {exc}")
        except Exception as exc:
            logger.exception("Ingestion of '%s' failed", filename)
            await self._fail(document_id, f"Unexpected error: {exc}")

    async def _fail(self, document_id: UUID, message: str) -> None:
        logger.warning("Document %s failed: %s", document_id, message)
        try:
            await self._store.set_document_status(
                document_id, "failed", error=message[:2000], chunk_count=0
            )
        except Exception:
            logger.exception("Could not record failure for document %s", document_id)

    # --- answering ---------------------------------------------------------

    async def answer_stream(
        self, message: str, conversation_id: UUID | None = None
    ) -> AsyncIterator[StreamEvent]:
        conversation_id = await self._ensure_conversation(conversation_id)
        yield StreamEvent("meta", {"conversation_id": str(conversation_id)})

        history = await self._store.get_recent_messages(
            conversation_id, self._settings.history_turns
        )
        question = await self._condense(history, message)
        if question != message:
            logger.info("Condensed follow-up to: %s", question)
            yield StreamEvent("meta", {"standalone_question": question})

        retrieval = await self._retriever.retrieve(question)
        await self._store.add_message(conversation_id, "user", message)

        if not retrieval.grounded:
            await self._store.add_message(
                conversation_id, "assistant", prompts.NO_CONTEXT_ANSWER
            )
            yield StreamEvent("token", prompts.NO_CONTEXT_ANSWER)
            yield StreamEvent("citations", [])
            yield StreamEvent("done", {"grounded": False})
            return

        yield StreamEvent(
            "citations", [c.model_dump(mode="json") for c in retrieval.citations]
        )

        parts: list[str] = []
        try:
            async for piece in self._llm.stream(
                prompts.ANSWER_SYSTEM_PROMPT,
                prompts.build_answer_prompt(question, retrieval.context),
                temperature=0.2,
            ):
                parts.append(piece)
                yield StreamEvent("token", piece)
        except LLMError as exc:
            logger.warning("Answer generation failed: %s", exc)
            yield StreamEvent("error", str(exc))
            return

        text = "".join(parts).strip()
        if text:
            await self._store.add_message(
                conversation_id,
                "assistant",
                text,
                [c.model_dump(mode="json") for c in retrieval.citations],
            )
        yield StreamEvent("done", {"grounded": True})

    async def answer(self, message: str, conversation_id: UUID | None = None) -> Answer:
        """Non-streaming convenience wrapper over `answer_stream`."""
        conversation = conversation_id or uuid4()
        parts: list[str] = []
        citations: list[Citation] = []
        grounded = True

        async for event in self.answer_stream(message, conversation_id):
            if event.type == "meta" and "conversation_id" in event.data:
                conversation = UUID(event.data["conversation_id"])
            elif event.type == "token":
                parts.append(event.data)
            elif event.type == "citations":
                citations = [Citation(**c) for c in event.data]
            elif event.type == "done":
                grounded = bool(event.data.get("grounded", True))
            elif event.type == "error":
                raise LLMError(event.data)

        return Answer(
            conversation_id=conversation,
            answer="".join(parts).strip(),
            citations=citations,
            grounded=grounded,
        )

    async def _ensure_conversation(self, conversation_id: UUID | None) -> UUID:
        if conversation_id is None:
            conversation_id = uuid4()
        await self._store.create_conversation(conversation_id)
        return conversation_id

    async def _condense(self, history: list, message: str) -> str:
        if not prompts.needs_condensing(history):
            return message
        try:
            raw = await self._llm.complete(
                prompts.CONDENSE_SYSTEM_PROMPT,
                prompts.build_condense_prompt(history, message),
                temperature=0.0,
            )
        except LLMError as exc:
            # Condensing is an optimisation; retrieving on the raw message is a
            # worse but working fallback.
            logger.warning("Condensing failed, using the raw message: %s", exc)
            return message
        return prompts.clean_condensed(raw, message)
