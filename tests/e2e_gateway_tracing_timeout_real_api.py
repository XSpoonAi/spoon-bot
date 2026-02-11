"""End-to-end tests for gateway tracing, timing, and timeout features.

These tests validate the full integration of P0-1 (tracing/timing) and P0-2
(budgets/cancellation/timeout codes) using a test FastAPI app with mocked agent.

For real API testing, set SPOON_E2E_BASE_URL environment variable.
"""

import asyncio
import json
import os
import time
from unittest.mock import AsyncMock, patch

import pytest

from spoon_bot.gateway.observability.tracing import new_trace_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_e2e_app():
    """Create a fully configured test FastAPI app."""
    from spoon_bot.gateway.api.v1.agent import router
    from spoon_bot.gateway.config import GatewayConfig, BudgetConfig

    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router, prefix="/v1/agent")

    # Skip auth
    from spoon_bot.gateway.auth.dependencies import get_current_user

    class FakeUser:
        user_id = "e2e_test_user"
        session_key = "e2e_default"

    app.dependency_overrides[get_current_user] = lambda: FakeUser()
    return app


def _mock_agent(response_text="E2E test response", stream_chunks=None, delay=0):
    """Create a mock agent with configurable behavior."""
    agent = AsyncMock()

    async def mock_process(**kwargs):
        if delay:
            await asyncio.sleep(delay)
        return response_text

    async def mock_process_with_thinking(**kwargs):
        if delay:
            await asyncio.sleep(delay)
        return (response_text, "thinking about it...")

    agent.process = mock_process
    agent.process_with_thinking = mock_process_with_thinking

    async def mock_stream(**kwargs):
        chunks = stream_chunks or [
            {"type": "content", "delta": "E2E ", "metadata": {}},
            {"type": "content", "delta": "test ", "metadata": {}},
            {"type": "content", "delta": "response", "metadata": {}},
            {"type": "done", "delta": "", "metadata": {}},
        ]
        for chunk in chunks:
            if delay:
                await asyncio.sleep(delay / len(chunks))
            yield chunk

    agent.stream = mock_stream
    return agent


def _parse_sse_events(body: str) -> list[dict]:
    """Parse SSE event stream into structured events."""
    events = []
    current = {}
    for line in body.split("\n"):
        line = line.strip()
        if line.startswith("event:"):
            current["event"] = line[len("event:"):].strip()
        elif line.startswith("data:"):
            current["data"] = line[len("data:"):].strip()
            events.append(current)
            current = {}
        elif line == "" and current:
            events.append(current)
            current = {}
    return events


# ===========================================================================
# E2E Scenario 1: REST Non-streaming -- trace_id + timing
# ===========================================================================

class TestE2ERestNonStreaming:
    """Validate trace_id and timing in REST JSON responses."""

    def test_json_response_has_trace_id(self):
        from fastapi.testclient import TestClient

        app = _create_e2e_app()
        agent = _mock_agent()

        with patch("spoon_bot.gateway.api.v1.agent.get_agent", return_value=agent):
            client = TestClient(app)
            resp = client.post("/v1/agent/chat", json={
                "message": "What is 2+2?",
                "options": {"stream": False},
            })

        assert resp.status_code == 200
        data = resp.json()

        # Verify trace_id
        assert data["meta"]["trace_id"] is not None
        assert data["meta"]["trace_id"].startswith("trc_")
        assert len(data["meta"]["trace_id"]) == 20  # "trc_" + 16 hex chars

    def test_json_response_has_timing(self):
        from fastapi.testclient import TestClient

        app = _create_e2e_app()
        agent = _mock_agent()

        with patch("spoon_bot.gateway.api.v1.agent.get_agent", return_value=agent):
            client = TestClient(app)
            resp = client.post("/v1/agent/chat", json={"message": "hello"})

        data = resp.json()
        timing = data["meta"]["timing"]

        assert timing is not None
        assert "total_elapsed_ms" in timing
        assert timing["total_elapsed_ms"] >= 0
        assert "started_at" in timing
        assert "T" in timing["started_at"]  # ISO 8601

    def test_trace_id_is_unique_per_request(self):
        from fastapi.testclient import TestClient

        app = _create_e2e_app()
        agent = _mock_agent()

        with patch("spoon_bot.gateway.api.v1.agent.get_agent", return_value=agent):
            client = TestClient(app)
            r1 = client.post("/v1/agent/chat", json={"message": "first"})
            r2 = client.post("/v1/agent/chat", json={"message": "second"})

        assert r1.json()["meta"]["trace_id"] != r2.json()["meta"]["trace_id"]

    def test_backward_compatible_fields(self):
        from fastapi.testclient import TestClient

        app = _create_e2e_app()
        agent = _mock_agent()

        with patch("spoon_bot.gateway.api.v1.agent.get_agent", return_value=agent):
            client = TestClient(app)
            resp = client.post("/v1/agent/chat", json={"message": "hello"})

        data = resp.json()
        # Old fields still present
        assert data["success"] is True
        assert "response" in data["data"]
        assert data["meta"]["request_id"].startswith("req_")
        assert data["meta"]["duration_ms"] >= 0


# ===========================================================================
# E2E Scenario 2: REST SSE -- trace + timing side-channels + [DONE]
# ===========================================================================

class TestE2ERestSSE:
    """Validate SSE side-channel events for trace and timing."""

    def test_sse_has_trace_event_at_start(self):
        from fastapi.testclient import TestClient

        app = _create_e2e_app()
        agent = _mock_agent()

        with patch("spoon_bot.gateway.api.v1.agent.get_agent", return_value=agent):
            client = TestClient(app)
            resp = client.post("/v1/agent/chat", json={
                "message": "stream test",
                "options": {"stream": True},
            })

        events = _parse_sse_events(resp.text)
        trace_events = [e for e in events if e.get("event") == "trace"]

        assert len(trace_events) >= 1
        trace_data = json.loads(trace_events[0]["data"])
        assert "trace_id" in trace_data
        assert trace_data["trace_id"].startswith("trc_")
        assert "request_id" in trace_data

    def test_sse_has_timing_event_before_done(self):
        from fastapi.testclient import TestClient

        app = _create_e2e_app()
        agent = _mock_agent()

        with patch("spoon_bot.gateway.api.v1.agent.get_agent", return_value=agent):
            client = TestClient(app)
            resp = client.post("/v1/agent/chat", json={
                "message": "stream test",
                "options": {"stream": True},
            })

        events = _parse_sse_events(resp.text)
        timing_events = [e for e in events if e.get("event") == "timing"]

        assert len(timing_events) >= 1
        timing_data = json.loads(timing_events[0]["data"])
        assert "total_elapsed_ms" in timing_data
        assert timing_data["total_elapsed_ms"] >= 0

    def test_sse_ends_with_done(self):
        from fastapi.testclient import TestClient

        app = _create_e2e_app()
        agent = _mock_agent()

        with patch("spoon_bot.gateway.api.v1.agent.get_agent", return_value=agent):
            client = TestClient(app)
            resp = client.post("/v1/agent/chat", json={
                "message": "stream test",
                "options": {"stream": True},
            })

        events = _parse_sse_events(resp.text)
        data_events = [e for e in events if "data" in e]
        assert data_events[-1]["data"] == "[DONE]"

    def test_sse_has_trace_id_header(self):
        from fastapi.testclient import TestClient

        app = _create_e2e_app()
        agent = _mock_agent()

        with patch("spoon_bot.gateway.api.v1.agent.get_agent", return_value=agent):
            client = TestClient(app)
            resp = client.post("/v1/agent/chat", json={
                "message": "stream test",
                "options": {"stream": True},
            })

        assert "x-trace-id" in resp.headers
        assert resp.headers["x-trace-id"].startswith("trc_")

    def test_sse_content_chunks_between_trace_and_timing(self):
        from fastapi.testclient import TestClient

        app = _create_e2e_app()
        agent = _mock_agent()

        with patch("spoon_bot.gateway.api.v1.agent.get_agent", return_value=agent):
            client = TestClient(app)
            resp = client.post("/v1/agent/chat", json={
                "message": "stream test",
                "options": {"stream": True},
            })

        events = _parse_sse_events(resp.text)

        # Find trace, content, timing, done events in order
        trace_idx = next(i for i, e in enumerate(events) if e.get("event") == "trace")
        timing_idx = next(i for i, e in enumerate(events) if e.get("event") == "timing")
        done_idx = next(i for i, e in enumerate(events) if e.get("data") == "[DONE]")

        # Content chunks should be between trace and timing
        content_indices = [
            i for i, e in enumerate(events)
            if "data" in e and e["data"] not in ("[DONE]",)
            and e.get("event") not in ("trace", "timing")
        ]

        assert trace_idx < min(content_indices) if content_indices else True
        assert timing_idx < done_idx
        if content_indices:
            assert max(content_indices) < timing_idx


# ===========================================================================
# E2E Scenario 3: Timeout Error Codes
# ===========================================================================

class TestE2ETimeoutCodes:
    """Validate standardized timeout error codes."""

    def test_upstream_timeout_returns_504(self):
        from fastapi.testclient import TestClient

        app = _create_e2e_app()
        agent = _mock_agent()
        agent.process = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch("spoon_bot.gateway.api.v1.agent.get_agent", return_value=agent):
            with patch("spoon_bot.gateway.api.v1.agent.get_config") as mock_cfg:
                from spoon_bot.gateway.config import GatewayConfig, BudgetConfig
                cfg = GatewayConfig()
                cfg.budget = BudgetConfig(request_timeout_ms=1000)
                mock_cfg.return_value = cfg

                client = TestClient(app)
                resp = client.post("/v1/agent/chat", json={"message": "timeout test"})

        assert resp.status_code == 504
        detail = resp.json()["detail"]
        assert detail["code"] == "TIMEOUT_UPSTREAM"
        assert "elapsed_ms" in detail.get("details", {})
        assert "limit_ms" in detail.get("details", {})

    def test_timeout_error_codes_are_consistent(self):
        """Verify all three timeout codes exist in the enum."""
        from spoon_bot.gateway.errors import TimeoutCode

        assert TimeoutCode.TIMEOUT_UPSTREAM.value == "TIMEOUT_UPSTREAM"
        assert TimeoutCode.TIMEOUT_TOOL.value == "TIMEOUT_TOOL"
        assert TimeoutCode.TIMEOUT_TOTAL.value == "TIMEOUT_TOTAL"


# ===========================================================================
# E2E Scenario 4: Cancellation
# ===========================================================================

class TestE2ECancellation:
    """Validate cancellation propagation."""

    @pytest.mark.asyncio
    async def test_cancel_event_stops_sse_stream(self):
        """Cancel event should stop SSE stream before all chunks are emitted."""
        from spoon_bot.gateway.api.v1.agent import _stream_sse

        chunks_yielded = []
        cancel_event = asyncio.Event()

        agent = AsyncMock()

        async def slow_stream(**kwargs):
            for i in range(20):
                await asyncio.sleep(0.01)
                yield {"type": "content", "delta": f"chunk{i}", "metadata": {}}
            yield {"type": "done", "delta": "", "metadata": {}}

        agent.stream = slow_stream

        # Set cancel after a short delay
        async def set_cancel():
            await asyncio.sleep(0.05)
            cancel_event.set()

        asyncio.create_task(set_cancel())

        async for chunk in _stream_sse(
            agent, "test", None, False,
            trace_id="trc_e2e_cancel_test",
            request_id="req_e2e_cancel",
            cancel_event=cancel_event,
        ):
            chunks_yielded.append(chunk)

        # Should have stopped well before all 20 content chunks
        content_chunks = [c for c in chunks_yielded if "chunk" in c and '"content"' in c]
        assert len(content_chunks) < 20

        # Should still have timing and DONE
        text = "".join(chunks_yielded)
        assert "event: timing" in text
        assert "[DONE]" in text


# ===========================================================================
# E2E Scenario 5: Trace ID Correlation
# ===========================================================================

class TestE2ETraceCorrelation:
    """Validate that trace_id can be used to correlate across events."""

    def test_sse_trace_id_matches_header(self):
        from fastapi.testclient import TestClient

        app = _create_e2e_app()
        agent = _mock_agent()

        with patch("spoon_bot.gateway.api.v1.agent.get_agent", return_value=agent):
            client = TestClient(app)
            resp = client.post("/v1/agent/chat", json={
                "message": "correlation test",
                "options": {"stream": True},
            })

        # Get trace_id from header
        header_trace_id = resp.headers.get("x-trace-id")

        # Get trace_id from event
        events = _parse_sse_events(resp.text)
        trace_events = [e for e in events if e.get("event") == "trace"]
        event_trace_id = json.loads(trace_events[0]["data"])["trace_id"]

        # They should match
        assert header_trace_id == event_trace_id

    @pytest.mark.asyncio
    async def test_ws_trace_id_consistent_across_events(self):
        """WS handler should use same trace_id for all events in one chat."""
        from spoon_bot.gateway.websocket.handler import WebSocketHandler

        handler = WebSocketHandler("e2e_conn")
        agent = _mock_agent()

        class FakeConn:
            session_key = "default"

        class FakeMgr:
            def __init__(self):
                self.sent = []

            def get_connection(self, _):
                return FakeConn()

            async def send_message(self, _, msg):
                if hasattr(msg, 'to_dict'):
                    self.sent.append(msg.to_dict())
                else:
                    self.sent.append(msg)

        mgr = FakeMgr()

        with patch("spoon_bot.gateway.websocket.handler.get_agent", return_value=agent):
            with patch("spoon_bot.gateway.websocket.handler.get_connection_manager", return_value=mgr):
                with patch("spoon_bot.gateway.websocket.handler.get_config") as mock_cfg:
                    from spoon_bot.gateway.config import GatewayConfig
                    mock_cfg.return_value = GatewayConfig()
                    result = await handler._handle_chat({
                        "message": "correlation ws test",
                        "stream": True,
                    })

        # All events should share the same trace_id
        trace_ids = set()
        for msg in mgr.sent:
            if isinstance(msg, dict) and "data" in msg:
                data = msg["data"]
                if isinstance(data, dict) and "trace_id" in data:
                    trace_ids.add(data["trace_id"])

        assert len(trace_ids) == 1, f"Expected 1 unique trace_id, got {trace_ids}"
        assert result["trace_id"] in trace_ids
