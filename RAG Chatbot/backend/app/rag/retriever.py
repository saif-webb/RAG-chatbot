"""Retrieval: embed the question, search, filter, and turn hits into citations."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.core.config import Settings
from app.db.base import Store
from app.db.models import SearchHit
from app.rag.embeddings import Embedder
from app.rag.prompt import format_context
from app.schemas.chat import Citation

logger = logging.getLogger(__name__)

SNIPPET_CHARS = 320


@dataclass(slots=True)
class Retrieval:
    context: str
    citations: list[Citation]
    hits: list[SearchHit]

    @property
    def grounded(self) -> bool:
        """False when nothing cleared the threshold — answer without the LLM."""
        return bool(self.hits)


class Retriever:
    def __init__(self, store: Store, embedder: Embedder, settings: Settings) -> None:
        self._store = store
        self._embedder = embedder
        self._settings = settings

    async def retrieve(self, question: str) -> Retrieval:
        embedding = await self._embedder.embed_query(question)
        hits = await self._store.search(embedding, self._settings.top_k)

        threshold = self._settings.similarity_threshold
        relevant = [h for h in hits if h.score >= threshold]

        if hits and not relevant:
            logger.info(
                "All %d hits fell below SIMILARITY_THRESHOLD=%.2f (best %.3f); "
                "answering with the grounded refusal.",
                len(hits),
                threshold,
                hits[0].score,
            )

        context, used = format_context(
            relevant, max_chars=self._settings.max_context_chars
        )
        return Retrieval(context=context, citations=to_citations(used), hits=used)


def to_citations(hits: list[SearchHit]) -> list[Citation]:
    """Number citations from 1 so they line up with the `[n]` markers in context."""
    return [
        Citation(
            index=i,
            document_id=hit.document_id,
            filename=hit.filename,
            page=hit.page,
            score=round(hit.score, 4),
            snippet=_snippet(hit.content),
        )
        for i, hit in enumerate(hits, start=1)
    ]


def _snippet(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= SNIPPET_CHARS:
        return collapsed
    return collapsed[:SNIPPET_CHARS].rsplit(" ", 1)[0] + "…"
