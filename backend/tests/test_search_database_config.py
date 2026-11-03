import pytest

from backend.search.repository import create_search_database_engine


def test_search_database_requires_an_explicit_url(monkeypatch):
    monkeypatch.delenv("SEARCH_DATABASE_URL", raising=False)
    monkeypatch.delenv("CATALOG_DATABASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="explicitly configured"):
        create_search_database_engine()


def test_search_database_rejects_non_postgresql_url(monkeypatch):
    monkeypatch.setenv("SEARCH_DATABASE_URL", "sqlite:///temporary.db")

    with pytest.raises(RuntimeError, match="requires a PostgreSQL"):
        create_search_database_engine()


def test_search_database_uses_explicit_search_url(monkeypatch):
    monkeypatch.setenv("SEARCH_DATABASE_URL", "postgresql+psycopg2://user:secret@db.example.test/search")
    monkeypatch.setenv("CATALOG_DATABASE_URL", "postgresql+psycopg2://other:secret@db.example.test/catalog")
    engine = create_search_database_engine()
    try:
        assert engine.url.database == "search"
        assert engine.url.host == "db.example.test"
    finally:
        engine.dispose()
