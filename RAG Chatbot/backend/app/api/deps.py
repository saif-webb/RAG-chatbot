"""Shared request dependencies.

Everything expensive — the store's connection pool, the embedder, the LLM
client — is built once during app startup and hung off `app.state`. These
accessors just hand it to the routes.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.core.config import Settings, get_settings
from app.db.base import Store
from app.rag.pipeline import RagPipeline


def get_store(request: Request) -> Store:
    return request.app.state.store


def get_pipeline(request: Request) -> RagPipeline:
    return request.app.state.pipeline


StoreDep = Annotated[Store, Depends(get_store)]
PipelineDep = Annotated[RagPipeline, Depends(get_pipeline)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
