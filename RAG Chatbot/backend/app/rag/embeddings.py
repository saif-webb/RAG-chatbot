"""Local embeddings via sentence-transformers (`all-MiniLM-L6-v2`).

384 dimensions, ~90 MB of weights, and no API key, no quota and no per-request
cost — ingesting a large corpus is bounded by CPU rather than by a daily limit.

Two consequences worth knowing:

* **Memory.** sentence-transformers pulls PyTorch. Expect roughly 1 GB of
  installed dependencies and 400-600 MB of resident memory once the model is
  loaded. It does not fit a 512 MB instance.
* **Symmetry.** Unlike Gemini's embeddings, MiniLM uses one encoder for both
  passages and questions, so there is no document/query task type to set — and
  no silent recall loss from getting it wrong.

The model is loaded once at startup. Encoding is CPU-bound and synchronous, so
it runs in a worker thread to keep the event loop responsive.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol, runtime_checkable

from app.core.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Embedding failed in a way the caller should surface."""


@runtime_checkable
class Embedder(Protocol):
    """What the pipeline needs from an embedder.

    Exists so the tests can substitute a deterministic double; there is one
    real implementation.
    """

    name: str
    dim: int

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...

    async def aclose(self) -> None: ...


class SentenceTransformerEmbedder:
    name = "sentence-transformers"

    def __init__(self, settings: Settings) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - install-time failure
            raise EmbeddingError(
                "sentence-transformers is not installed:\n"
                "    pip install -r backend/requirements.txt"
            ) from exc

        self.model_name = settings.embedding_model
        self._batch_size = settings.embedding_batch_size

        logger.info(
            "Loading embedding model %s (this can take a moment)…", self.model_name
        )
        self._model = SentenceTransformer(
            self.model_name, device=settings.embedding_device
        )

        # Read the dimension from the model rather than from configuration, so
        # the pgvector column can never disagree with what is being stored.
        dim = self._model.get_sentence_embedding_dimension()
        if not dim:
            raise EmbeddingError(
                f"Could not determine the output dimension of {self.model_name}."
            )
        self.dim = int(dim)
        logger.info("Embedding model ready: %s (%d dims)", self.model_name, self.dim)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode, texts)

    async def embed_query(self, text: str) -> list[float]:
        return (await asyncio.to_thread(self._encode, [text]))[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        try:
            vectors = self._model.encode(
                texts,
                batch_size=self._batch_size,
                # Unit-length vectors make the cosine distance operator and the
                # stored data consistent.
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingError(f"Encoding {len(texts)} texts failed: {exc}") from exc

        if vectors.shape[1] != self.dim:
            raise EmbeddingError(
                f"{self.model_name} returned {vectors.shape[1]} dimensions, "
                f"expected {self.dim}."
            )
        return vectors.astype("float32").tolist()

    async def aclose(self) -> None:
        return None


def create_embedder(settings: Settings) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(settings)
