"""Application settings — the single source of truth for every tunable.

Loaded once from the environment, falling back to the project `.env` file.
Nothing else in the codebase reads `os.environ` directly.

The stack is fixed rather than pluggable: Gemini for chat, sentence-transformers
for embeddings, Postgres + pgvector for storage.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> core -> app -> backend -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "RAG Chatbot"
    log_level: str = "INFO"

    # --- database ----------------------------------------------------------
    # Postgres with the pgvector extension. Locally: docker compose up -d.
    # Deployed: a managed instance (Neon, Supabase, RDS...).
    database_url: str = ""
    db_pool_min: int = Field(default=1, ge=1, le=20)
    db_pool_max: int = Field(default=5, ge=1, le=50)

    # --- llm ---------------------------------------------------------------
    gemini_api_key: str = ""
    gemini_chat_model: str = "gemini-2.5-flash-lite"
    gemini_timeout_seconds: int = Field(default=120, ge=5, le=600)

    # --- embeddings --------------------------------------------------------
    # all-MiniLM-L6-v2 is 384-dimensional and symmetric: the same encoder is used
    # for passages and questions, so there is no document/query task distinction
    # to get wrong.
    #
    # The vector dimension is read from the loaded model rather than configured,
    # which removes a whole class of "EMBED_DIM does not match the column" bugs.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(default=32, ge=1, le=512)

    # --- retrieval ---------------------------------------------------------
    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=150, ge=0)
    top_k: int = Field(default=8, ge=1, le=50)
    # Cosine floor. For all-MiniLM-L6-v2, unrelated text lands near 0.0-0.2 and a
    # genuine match near 0.4-0.7, so 0.35 separates them with a little headroom.
    similarity_threshold: float = Field(default=0.35, ge=-1.0, le=1.0)
    max_context_chars: int = Field(default=12_000, ge=500)
    history_turns: int = Field(default=6, ge=0, le=40)

    # --- uploads -----------------------------------------------------------
    max_upload_mb: int = Field(default=20, ge=1, le=200)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def frontend_dir(self) -> Path:
        return PROJECT_ROOT / "frontend"

    @model_validator(mode="after")
    def _check_coherent(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be smaller than "
                f"CHUNK_SIZE ({self.chunk_size}); otherwise chunking never advances."
            )

        if self.db_pool_min > self.db_pool_max:
            raise ValueError("DB_POOL_MIN cannot exceed DB_POOL_MAX.")

        if not self.database_url:
            raise ValueError(
                "DATABASE_URL is required. Start a local Postgres with "
                "`docker compose up -d`, or paste a managed connection string."
            )

        if not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is required. Get one free at "
                "https://aistudio.google.com/apikey"
            )

        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
