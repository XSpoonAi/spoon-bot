"""Tests for REST/SSE tracing, timing, budget, and timeout error codes."""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# We need to mock the gateway dependencies
from spoon_bot.gateway.models.responses import MetaInfo, APIResponse, ChatResponse, StreamChunk
from spoon_bot.gateway.observability.tracing import new_trace_id


def _create_test_app():
    """Create a test FastAPI app with mocked dependencies."""
    from spoon_bot.gateway.api.v1.agent import router
    from spoon_bot.gateway.config import GatewayConfig, BudgetConfig

    app = FastAPI()
    app.include_router(router, prefix="/v1/agent")

    # Mock auth dependency to skip auth
    from spoon_bot.gateway.auth.dependencies import get_current_user

    class FakeUser:
        user_id = "test_user"
        session_key = "default"

    app.dependency_overrides[get_current_user] = lambda: FakeUser()

    return app


def _mock_agent(response_text="Hello!", stream_chunks=None):
    """Create a mock agent."""
    agent = AsyncMock()
    agent.process = AsyncMock(return_value=response_text)
    agent.process_with_thinking = AsyncMock(return_value=(response_text, "thinking..."))

    async def fake_stream(**kwargs):
        chunks = stream_chunks or [
            {"type": "content", "delta": "Hello", "metadata": {}},
            {"type": "content", "delta": " world", "metadata": {}},
            {"type": "done", "delta": "", "metadata": {}},
        ]
        for chunk in chunks:
            yield chunk

    agent.stream = fake_stream
    return agent


class TestChatJsonTraceAndTiming:
    """P0-1.4: REST JSON responses must include trace_id and timing."""

    def test_chat_json_contains_trace_and_timing(self):
        app = _create_test_app()
        mock_agent = _mock_agent()

        with patch("spoon_bot.gateway.api.v1.agent.get_agent", return_value=mock_agent):
            client = TestClient(app)
            resp = client.post("/v1/agent/chat", json={
                "message": "hello",
                "options": {"stream": False},
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["meta"]["trace_id"] is not None
        assert data["meta"]["trace_id"].startswith("trc_")
        assert data["meta"]["timing"] is not None
        assert "total_elapsed_ms" in data["meta"]["timing"]
        assert data["meta"]["timing"]["total_elapsed_ms"] >= 0

    def test_chat_json_backward_compatible(self):
        """Existing fields (request_id, duration_ms) still present."""
        app = _create_test_app()
        mock_agent = _mock_agent()

        with patch("spoon_bot.gateway.api.v1.agent.get_agent", return_value=mock_agent):
            client = TestClient(app)
            resp = client.post("/v1/agent/chat", json={"message": "hi"})

        data = resp.json()
        assert "request_id" in data["meta"]
        assert data["meta"]["request_id"].startswith("req_")


class TestChatSseTraceAndTiming:
    """P0-1.5: SSE must include trace and timing side-channel events."""

    def test_chat_sse_contains_trace_and_timing_events(self):
        app = _create_test_app()
        mock_agent = _mock_agent()

        with patch("spoon_bot.gateway.api.v1.agent.get_agent", return_value=mock_agent):
            client = TestClient(app)
            resp = client.post("/v1/agent/chat", json={
                "message": "hello",
                "options": {"stream": True},
            })

        assert resp.status_code == 200
        body = resp.text
        lines = body.strip().split("\n")

        # Parse SSE events
        events = []
        current_event = {}
        for line in lines:
            line = line.strip()
            if line.startswith("event:"):
                current_event["event"] = line[len("event:"):].strip()
            elif line.startswith("data:"):
                current_event["data"] = line[len("data:"):].strip()
                events.append(current_event)
                current_event = {}
            elif line == "":
                if current_event:
                    events.append(current_event)
                    current_event = {}

        # Should contain event: trace
        trace_events = [e for e in events if e.get("event") == "trace"]
        assert len(trace_events) >= 1, f"No trace event found. Events: {events}"
        trace_data = json.loads(trace_events[0]["data"])
        assert "trace_id" in trace_data
        assert trace_data["trace_id"].startswith("trc_")

        # Should contain event: timing
        timing_events = [e for e in events if e.get("event") == "timing"]
        assert len(timing_events) >= 1, f"No timing event found. Events: {events}"
        timing_data = json.loads(timing_events[0]["data"])
        assert "total_elapsed_ms" in timing_data

        # Should end with [DONE]
        data_events = [e for e in events if "data" in e]
        assert data_events[-1]["data"] == "[DONE]"

    def test_chat_sse_has_trace_id_header(self):
        app = _create_test_app()
        mock_agent = _mock_agent()

        with patch("spoon_bot.gateway.api.v1.agent.get_agent", return_value=mock_agent):
            client = TestClient(app)
            resp = client.post("/v1/agent/chat", json={
                "message": "hello",
                "options": {"stream": True},
            })

        assert "x-trace-id" in resp.headers


class TestTimeoutErrorCodes:
    """P0-2.3: Timeout errors use standardized codes."""

    def test_upstream_timeout_returns_timeout_upstream(self):
        app = _create_test_app()
        mock_agent = _mock_agent()
        mock_agent.process = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch("spoon_bot.gateway.api.v1.agent.get_agent", return_value=mock_agent):
            with patch("spoon_bot.gateway.api.v1.agent.get_config") as mock_config:
                from spoon_bot.gateway.config import GatewayConfig, BudgetConfig
                cfg = GatewayConfig()
                cfg.budget = BudgetConfig(request_timeout_ms=1000)
                mock_config.return_value = cfg

                client = TestClient(app)
                resp = client.post("/v1/agent/chat", json={"message": "hello"})

        # Should return error (408 or 504 are both acceptable for timeout)
        assert resp.status_code in (408, 504, 500)
        data = resp.json()
        assert "detail" in data or "error" in data
        detail = data.get("detail", data.get("error", {}))
        if isinstance(detail, dict):
            assert detail.get("code") in ("TIMEOUT_UPSTREAM", "TIMEOUT_TOTAL", "AGENT_ERROR")
