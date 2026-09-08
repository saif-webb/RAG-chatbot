"""Chat endpoint — answers stream back as Server-Sent Events."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.deps import PipelineDep
from app.rag.pipeline import RagPipeline
from app.schemas.chat import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # Without this, nginx-style proxies buffer the whole response and the
    # answer arrives in one lump instead of streaming.
    "X-Accel-Buffering": "no",
}


@router.post("", summary="Ask a question (SSE stream)")
async def chat(request: ChatRequest, pipeline: PipelineDep) -> StreamingResponse:
    """Stream an answer.

    Event types, each carrying a JSON payload:

    * `meta`      — `conversation_id`, and `standalone_question` when a
                    follow-up was rewritten
    * `citations` — the passages the answer may cite, sent *before* the text so
                    the UI can render sources as the answer arrives
    * `token`     — a fragment of the answer
    * `done`      — `grounded: false` when the question was refused for lack of
                    supporting context
    * `error`     — generation failed; the message is safe to display
    """
    return StreamingResponse(
        _events(pipeline, request),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post(
    "/sync",
    response_model=ChatResponse,
    summary="Ask a question (non-streaming)",
)
async def chat_sync(request: ChatRequest, pipeline: PipelineDep) -> ChatResponse:
    """Same pipeline, buffered. Convenient for scripts and curl."""
    result = await pipeline.answer(request.message, request.conversation_id)
    return ChatResponse(
        conversation_id=result.conversation_id,
        answer=result.answer,
        citations=result.citations,
        grounded=result.grounded,
    )


async def _events(pipeline: RagPipeline, request: ChatRequest) -> AsyncIterator[str]:
    try:
        async for event in pipeline.answer_stream(
            request.message, request.conversation_id
        ):
            payload = event.data if event.type != "token" else {"text": event.data}
            yield _sse(event.type, payload)
    except Exception as exc:
        logger.exception("Chat stream failed")
        yield _sse("error", {"message": f"Something went wrong: {exc}"})


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"
