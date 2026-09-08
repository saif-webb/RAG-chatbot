"""Application entry point.

Run locally:

    docker compose up -d          # Postgres with pgvector
    cd backend
    uvicorn app.main:app --reload

The static frontend is served by this same app, mounted at `/`. That is
deliberate: one service, one origin, one URL — and therefore no CORS
configuration to get wrong, and nothing extra to deploy.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import chat, documents
from app.core.config import Settings, get_settings
from app.db.database import create_store
from app.rag.embeddings import create_embedder
from app.rag.llm import create_llm
from app.rag.pipeline import RagPipeline

logger = logging.getLogger(__name__)


def configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    # These log every request at INFO, drowning out everything else.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    configure_logging(settings)

    # Order matters: the embedding model is loaded first because it is what
    # declares the vector dimension, and the schema is built from that. This is
    # why there is no EMBED_DIM setting to get out of step with the model.
    embedder = create_embedder(settings)

    store = create_store(settings, embedder.dim)
    # Creates the schema if missing and verifies the live column dimension still
    # matches the model, so a fresh deploy provisions itself and a mismatched one
    # fails here rather than at the first insert.
    await store.connect()

    llm = create_llm(settings)

    app.state.settings = settings
    app.state.store = store
    app.state.embedder = embedder
    app.state.llm = llm
    app.state.pipeline = RagPipeline(store, embedder, llm, settings)

    logger.info(
        "%s ready — postgres · %s · %s (%d dims)",
        settings.app_name,
        settings.gemini_chat_model,
        embedder.model_name,
        embedder.dim,
    )
    try:
        yield
    finally:
        await llm.aclose()
        await embedder.aclose()
        await store.close()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="Retrieval-augmented chat over your own documents.",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.include_router(documents.router)
    app.include_router(chat.router)

    @app.get("/healthz", tags=["ops"], summary="Liveness and store check")
    async def healthz() -> JSONResponse:
        ok = False
        detail = "not started"
        try:
            ok = await app.state.store.health()
            detail = "ok" if ok else "database unreachable"
        except Exception as exc:  # noqa: BLE001 - health must report, never raise
            detail = str(exc)
        return JSONResponse(
            status_code=200 if ok else 503,
            content={
                "status": "ok" if ok else "degraded",
                "store": "postgres+pgvector",
                "llm": settings.gemini_chat_model,
                "embeddings": settings.embedding_model,
                "embed_dim": getattr(app.state, "embedder", None)
                and app.state.embedder.dim,
                "detail": detail,
            },
        )

    # Mounted last so it never shadows /api or /healthz. html=True serves
    # index.html for `/`.
    if settings.frontend_dir.is_dir():
        app.mount(
            "/",
            StaticFiles(directory=settings.frontend_dir, html=True),
            name="frontend",
        )
    else:
        logger.warning(
            "No frontend directory at %s; serving the API only.", settings.frontend_dir
        )

    return app


app = create_app()
