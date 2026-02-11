"""Pytest configuration and fixtures."""

import sys

import pytest


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "requires_spoon_core: mark test as requiring spoon-core SDK"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests that require spoon-core if it's not installed."""
    try:
        import spoon_ai  # noqa: F401
        spoon_core_available = True
    except ImportError:
        spoon_core_available = False

    skip_spoon_core = pytest.mark.skip(
        reason="spoon-core SDK not installed (pip install spoon-ai)"
    )

    for item in items:
        # Skip tests with explicit marker
        if "requires_spoon_core" in item.keywords:
            if not spoon_core_available:
                item.add_marker(skip_spoon_core)

        # Also skip if test imports from modules that require spoon-core
        # Check if the test module has any ImportError due to spoon-core
        if not spoon_core_available:
            try:
                # Try to import the test module to check for import errors
                pass
            except ImportError:
                item.add_marker(skip_spoon_core)
