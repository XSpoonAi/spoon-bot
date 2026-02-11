"""Tests for smart fallback error handling in gateway."""

import pytest
from spoon_bot.gateway.errors import (
    TimeoutCode,
    GatewayErrorCode,
    build_timeout_error_detail,
    build_error_response,
)
from spoon_bot.gateway.models.responses import ErrorDetail, ErrorResponse


class TestSmartFallback:
    """Verify error responses are well-structured and include trace info."""

    def test_error_response_includes_trace_id(self):
        detail = ErrorDetail(code="TEST_ERROR", message="test")
        resp = build_error_response(detail, request_id="req_123", trace_id="trc_abc")
        assert resp.meta.trace_id == "trc_abc"
        assert resp.meta.request_id == "req_123"

    def test_error_response_includes_timing(self):
        detail = ErrorDetail(code="TEST_ERROR", message="test")
        timing = {"total_elapsed_ms": 500}
        resp = build_error_response(detail, request_id="req_123", timing=timing)
        assert resp.meta.timing == timing

    def test_timeout_error_response_full(self):
        error_detail = build_timeout_error_detail(
            TimeoutCode.TIMEOUT_UPSTREAM, elapsed_ms=5000, limit_ms=3000
        )
        resp = build_error_response(
            error_detail,
            request_id="req_456",
            trace_id="trc_def",
            timing={"total_elapsed_ms": 5000},
        )
        assert resp.success is False
        assert resp.error.code == "TIMEOUT_UPSTREAM"
        assert resp.meta.trace_id == "trc_def"

    def test_all_gateway_error_codes_defined(self):
        codes = [e.value for e in GatewayErrorCode]
        assert "TIMEOUT_UPSTREAM" in codes
        assert "TIMEOUT_TOOL" in codes
        assert "TIMEOUT_TOTAL" in codes
        assert "BUDGET_EXHAUSTED" in codes
        assert "CANCELLED" in codes
