"""Connection-string normalisation.

Managed Postgres providers hand out URLs carrying parameters asyncpg rejects
outright. Getting this wrong produces a confusing `unexpected keyword argument`
at startup, so it is worth pinning down. Pure functions — these always run, with
no database.
"""

from __future__ import annotations

import pytest

from app.db.base import StoreError
from app.db.database import normalize_dsn


def test_strips_parameters_asyncpg_rejects() -> None:
    """Neon and Supabase both append these to the copy-paste connection string."""
    dsn, ssl = normalize_dsn(
        "postgresql://u:p@host/db?sslmode=require&channel_binding=require"
    )
    assert "sslmode" not in dsn
    assert "channel_binding" not in dsn
    assert ssl is True


def test_keeps_unknown_parameters() -> None:
    dsn, _ = normalize_dsn("postgresql://u:p@host/db?search_path=public")
    assert "search_path=public" in dsn


def test_accepts_sqlalchemy_and_postgres_schemes() -> None:
    assert normalize_dsn("postgresql+asyncpg://u:p@h/db")[0].startswith("postgresql://")
    assert normalize_dsn("postgres://u:p@h/db")[0].startswith("postgresql://")


def test_ssl_is_off_for_a_plain_local_url() -> None:
    """A local `docker compose` Postgres has no certificate; requiring SSL fails."""
    assert normalize_dsn("postgresql://rag:rag@localhost:5432/rag")[1] is False


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("disable", False),
        ("allow", False),
        ("prefer", False),
        ("require", True),
        ("verify-full", True),
    ],
)
def test_sslmode_maps_to_the_ssl_flag(mode: str, expected: bool) -> None:
    assert normalize_dsn(f"postgresql://u:p@h/db?sslmode={mode}")[1] is expected


def test_rejects_a_non_postgres_url() -> None:
    with pytest.raises(StoreError, match="postgresql://"):
        normalize_dsn("mysql://u:p@h/db")


def test_credentials_and_path_survive() -> None:
    dsn, _ = normalize_dsn(
        "postgresql://user:p%40ss@ep-x-pooler.aws.neon.tech/neondb?sslmode=require"
    )
    assert dsn == "postgresql://user:p%40ss@ep-x-pooler.aws.neon.tech/neondb"
