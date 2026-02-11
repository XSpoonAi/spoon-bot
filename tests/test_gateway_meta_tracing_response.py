"""Tests for response meta tracing fields."""

from datetime import datetime
from spoon_bot.gateway.models.responses import MetaInfo, APIResponse, ErrorResponse, ErrorDetail


class TestMetaInfoTracing:
    def test_trace_id_optional_default_none(self):
        meta = MetaInfo(request_id="req_123")
        assert meta.trace_id is None

    def test_trace_id_set(self):
        meta = MetaInfo(request_id="req_123", trace_id="trc_abc123")
        assert meta.trace_id == "trc_abc123"

    def test_timing_optional_default_none(self):
        meta = MetaInfo(request_id="req_123")
        assert meta.timing is None

    def test_timing_set(self):
        timing_data = {"total_elapsed_ms": 150, "started_at": "2024-01-01T00:00:00Z"}
        meta = MetaInfo(request_id="req_123", timing=timing_data)
        assert meta.timing == timing_data

    def test_backward_compatible(self):
        """Existing fields still work without trace/timing."""
        meta = MetaInfo(request_id="req_123", duration_ms=100)
        assert meta.request_id == "req_123"
        assert meta.duration_ms == 100
        assert meta.trace_id is None
        assert meta.timing is None

    def test_serialization_includes_trace(self):
        meta = MetaInfo(
            request_id="req_123",
            trace_id="trc_abc",
            timing={"total_elapsed_ms": 50},
        )
        data = meta.model_dump()
        assert data["trace_id"] == "trc_abc"
        assert data["timing"]["total_elapsed_ms"] == 50

    def test_api_response_with_trace_meta(self):
        meta = MetaInfo(
            request_id="req_123",
            trace_id="trc_test",
            timing={"total_elapsed_ms": 200},
        )
        resp = APIResponse(success=True, data={"msg": "ok"}, meta=meta)
        assert resp.meta.trace_id == "trc_test"

    def test_error_response_with_trace_meta(self):
        meta = MetaInfo(
            request_id="req_123",
            trace_id="trc_err",
        )
        err = ErrorResponse(
            error=ErrorDetail(code="TEST", message="test error"),
            meta=meta,
        )
        assert err.meta.trace_id == "trc_err"
