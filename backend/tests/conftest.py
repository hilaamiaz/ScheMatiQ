"""Pytest fixtures for backend tests."""

import pytest

from app.services import session_manager


@pytest.fixture
def session_manager_fixture():
    return session_manager


@pytest.fixture(autouse=True)
def _no_conversion_cache(monkeypatch):
    """Keep tests from reading or filling the developer's conversion cache."""
    from app.services import conversion_cache
    monkeypatch.setattr(conversion_cache, "CACHE_DIR", None)
