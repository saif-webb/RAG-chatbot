"""Test fixtures.

The suite runs offline: no database, no network, no API key. Three things are
substituted — the store, the embedding model and Gemini — and everything else
(loader, chunker, retriever, prompt builder, pipeline) is production code.

`tests/test_store.py` covers the real Postgres store separately, and is skipped
unless `TEST_DATABASE_URL` points at a database.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from app.core.config import Settings
from app.rag.pipeline import RagPipeline
from tests.fake_store import FakeStore

STUB_DIM = 512
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Without this, every document shares "the"/"and"/"is" with every question and
# nothing is ever orthogonal — the refusal path would become untestable.
_STOPWORDS = frozenset(
    """a an and are as at be been but by can did do does for from had has have
    how in into is it its of on or that the then there these this to was were
    what when where which who why will with you your about across every must
    during their them they it's other another""".split()
)


class StubEmbedder:
    """Deterministic lexical embedding: one dimension per distinct token.

    Not semantic, but it has the property the tests need — documents sharing
    meaningful words score high, unrelated ones score exactly zero. A vocabulary
    dictionary is used rather than a hash so there are no collisions to make
    results depend on which words happen to co-occur.

    The real embedder (`all-MiniLM-L6-v2`) needs ~90 MB of weights and PyTorch,
    which is not something a unit test should download.
    """

    name = "stub"
    dim = STUB_DIM
    model_name = "stub"

    def __init__(self) -> None:
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []
        self._vocab: dict[str, int] = {}

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        return [self._vector(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self._vector(text)

    async def aclose(self) -> None:
        return None

    def _vector(self, text: str) -> list[float]:
        buckets = [0.0] * STUB_DIM
        for token in _TOKEN_RE.findall(text.lower()):
            if token in _STOPWORDS:
                continue
            index = self._vocab.setdefault(token, len(self._vocab) % STUB_DIM)
            buckets[index] += 1.0
        norm = sum(v * v for v in buckets) ** 0.5
        return [v / norm for v in buckets] if norm else buckets


class StubLLM:
    """Records prompts and returns canned text, standing in for Gemini."""

    name = "stub"

    def __init__(self, answer: str = "Stub answer [1].") -> None:
        self.answer = answer
        self.condensed: str | None = None
        self.complete_prompts: list[str] = []
        self.stream_prompts: list[str] = []

    async def complete(
        self, system: str, prompt: str, *, temperature: float = 0.0
    ) -> str:
        self.complete_prompts.append(prompt)
        return self.condensed if self.condensed is not None else "condensed question"

    async def stream(
        self, system: str, prompt: str, *, temperature: float = 0.2
    ) -> AsyncIterator[str]:
        self.stream_prompts.append(prompt)
        for word in self.answer.split(" "):
            yield word + " "

    async def aclose(self) -> None:
        return None


@pytest.fixture
def settings() -> Settings:
    return Settings(
        # Both are required by the config validator; nothing connects to them
        # because the store and LLM are substituted.
        database_url="postgresql://test:test@localhost:5432/test",
        gemini_api_key="test-key-not-used",
        chunk_size=200,
        chunk_overlap=40,
        top_k=5,
        # Tuned to the stub embedder, NOT to production. Bag-of-words cosine
        # between a short question and a long passage lands around 0.1-0.4 even
        # for a clear match, while an unrelated question scores exactly 0.0.
        # The shipped default of 0.35 is calibrated for all-MiniLM-L6-v2.
        similarity_threshold=0.08,
        history_turns=6,
    )


@pytest_asyncio.fixture
async def store() -> AsyncIterator[FakeStore]:
    instance = FakeStore(dim=STUB_DIM)
    await instance.connect()
    try:
        yield instance
    finally:
        await instance.close()


@pytest.fixture
def embedder() -> StubEmbedder:
    return StubEmbedder()


@pytest.fixture
def llm() -> StubLLM:
    return StubLLM()


@pytest.fixture
def pipeline(
    store: FakeStore,
    embedder: StubEmbedder,
    llm: StubLLM,
    settings: Settings,
) -> RagPipeline:
    return RagPipeline(store, embedder, llm, settings)
