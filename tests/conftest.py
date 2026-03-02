"""Pytest configuration and fixtures."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "requires_spoon_core: mark test as requiring spoon-core SDK"
    )
    config.addinivalue_line(
        "markers", "requires_api_key: mark test as requiring a live API key"
    )


def _spoon_core_available() -> bool:
    """Check if spoon-core SDK is importable."""
    try:
        import spoon_ai  # noqa: F401
        return True
    except ImportError:
        return False


def pytest_collection_modifyitems(config, items):
    """Skip tests that require spoon-core if it's not installed."""
    has_core = _spoon_core_available()

    skip_spoon_core = pytest.mark.skip(
        reason="spoon-core SDK not installed (pip install spoon-ai)"
    )
    skip_api_key = pytest.mark.skip(
        reason="No API key configured for live tests"
    )

    for item in items:
        # Skip tests with explicit marker
        if "requires_spoon_core" in item.keywords and not has_core:
            item.add_marker(skip_spoon_core)

        # Skip tests needing a live API key
        if "requires_api_key" in item.keywords:
            if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
                item.add_marker(skip_api_key)


@pytest.fixture
def workspace_dir(tmp_path: Path) -> Path:
    """Provide a temporary workspace directory for tests."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def test_env(monkeypatch, workspace_dir: Path):
    """Set up a minimal test environment with required env vars."""
    monkeypatch.setenv("SPOON_BOT_WORKSPACE_PATH", str(workspace_dir))
    monkeypatch.setenv("SPOON_BOT_LOG_LEVEL", "DEBUG")
    return {"workspace": workspace_dir}
