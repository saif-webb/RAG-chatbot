"""Gemini chat client.

Chosen over a locally-run model because it needs no weights on disk, no GPU and
no RAM — the machine's memory budget is spent on the embedding model instead.

Note the free tier's terms: Google uses free-tier content to improve their
products. Fine for a demo, not for confidential documents.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from app.core.config import Settings

logger = logging.getLogger(__name__)

MAX_OUTPUT_TOKENS = 2048


class LLMError(RuntimeError):
    """A provider failure that should reach the user as a clean message."""


class RateLimitError(LLMError):
    """Quota exhausted (HTTP 429). Worth surfacing distinctly — free-tier daily
    caps are the most common way this fails, and the fix is to wait, not retry."""


def is_rate_limit(exc: Exception) -> bool:
    """Detect quota exhaustion across the shapes the SDK reports it in."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in (429, "429"):
        return True
    text = str(exc).upper()
    return "RESOURCE_EXHAUSTED" in text or "429" in text or "QUOTA" in text


def _wrap(exc: Exception, action: str) -> LLMError:
    if is_rate_limit(exc):
        return RateLimitError(
            f"Gemini quota exhausted while {action}. Check your limits at "
            f"https://aistudio.google.com/rate-limit and try again shortly."
        )
    return LLMError(f"Gemini {action} failed: {exc}")


@runtime_checkable
class LLMClient(Protocol):
    """What the pipeline needs from a chat model.

    Exists so the tests can substitute a double; there is one real
    implementation.
    """

    name: str

    async def complete(
        self, system: str, prompt: str, *, temperature: float = 0.0
    ) -> str: ...

    def stream(
        self, system: str, prompt: str, *, temperature: float = 0.2
    ) -> AsyncIterator[str]: ...

    async def aclose(self) -> None: ...


class GeminiClient:
    name = "gemini"

    def __init__(self, settings: Settings) -> None:
        from google import genai
        from google.genai import types

        self._model = settings.gemini_chat_model
        self._client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(
                timeout=settings.gemini_timeout_seconds * 1000  # milliseconds
            ),
        )

    def _config(self, system: str, temperature: float) -> Any:
        from google.genai import types

        return types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )

    async def complete(
        self, system: str, prompt: str, *, temperature: float = 0.0
    ) -> str:
        """One-shot completion. Used to condense follow-up questions."""
        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=prompt,
                config=self._config(system, temperature),
            )
        except Exception as exc:
            raise _wrap(exc, "generating a response") from exc
        return (response.text or "").strip()

    async def stream(
        self, system: str, prompt: str, *, temperature: float = 0.2
    ) -> AsyncIterator[str]:
        """Yield the answer incrementally so the UI can render as it arrives."""
        try:
            stream = self._client.aio.models.generate_content_stream(
                model=self._model,
                contents=prompt,
                config=self._config(system, temperature),
            )
            # Across google-genai versions this is either an async iterator or a
            # coroutine resolving to one.
            if inspect.isawaitable(stream):
                stream = await stream

            async for chunk in stream:
                if text := getattr(chunk, "text", None):
                    yield text
        except Exception as exc:
            raise _wrap(exc, "streaming a response") from exc

    async def aclose(self) -> None:
        return None


def create_llm(settings: Settings) -> GeminiClient:
    logger.info("LLM: Gemini %s", settings.gemini_chat_model)
    return GeminiClient(settings)
