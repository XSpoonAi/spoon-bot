"""QA regression tests for gateway fixes."""

import pytest
from spoon_bot.gateway.errors import TimeoutCode, build_timeout_error_detail


class TestTimeoutErrorCodeStandardization:
    """P0-2.3: Verify three timeout codes exist and have proper structure."""

    def test_timeout_upstream_code(self):
        assert TimeoutCode.TIMEOUT_UPSTREAM == "TIMEOUT_UPSTREAM"

    def test_timeout_tool_code(self):
        assert TimeoutCode.TIMEOUT_TOOL == "TIMEOUT_TOOL"

    def test_timeout_total_code(self):
        assert TimeoutCode.TIMEOUT_TOTAL == "TIMEOUT_TOTAL"

    def test_build_timeout_upstream_detail(self):
        detail = build_timeout_error_detail("TIMEOUT_UPSTREAM", elapsed_ms=5000, limit_ms=3000)
        assert detail.code == "TIMEOUT_UPSTREAM"
        assert "timed out" in detail.message.lower() or "upstream" in detail.message.lower()
        assert detail.details["elapsed_ms"] == 5000
        assert detail.details["limit_ms"] == 3000

    def test_build_timeout_tool_detail(self):
        detail = build_timeout_error_detail("TIMEOUT_TOOL", elapsed_ms=15000, limit_ms=10000, context="shell")
        assert detail.code == "TIMEOUT_TOOL"
        assert detail.details["context"] == "shell"

    def test_build_timeout_total_detail(self):
        detail = build_timeout_error_detail("TIMEOUT_TOTAL", elapsed_ms=120000, limit_ms=120000)
        assert detail.code == "TIMEOUT_TOTAL"
