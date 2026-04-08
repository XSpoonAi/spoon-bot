"""
Gateway tracing, timing, budget, error-code, and cancellation tests.

Consolidated from:
  - test_gateway_tracing_utils.py
  - test_gateway_meta_tracing_response.py
  - test_gateway_agent_rest_tracing.py
  - test_gateway_ws_tracing.py
  - test_gateway_execution_budget.py
  - test_gateway_qa_fixes.py
  - test_prompt_and_gateway_smart_fallback.py
  - test_gateway_cancel_propagation.py
  - test_toolkit_adapter_timeout.py

NOTE: Some of these tests depend on tracing / budget features that are still
      under development.  They are kept as a reference and regression suite
      for when the features land.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from spoon_bot.gateway.observability.tracing import (
    new_trace_id,
    now_ms,
    TimerSpan,
    build_timing_payload,
)


# ============================================================================
# §1  Tracing utilities  (was test_gateway_tracing_utils.py)
# ============================================================================


class TestNewTraceId:
    def test_returns_string(self):
        assert isinstance(new_trace_id(), str)

    def test_has_prefix(self):
        assert new_trace_id().startswith("trc_")

    def test_unique(self):
        assert len({new_trace_id() for _ in range(100)}) == 100

    def test_length(self):
        # "trc_" + 16 hex chars = 20
        assert len(new_trace_id()) == 20


class TestNowMs:
    def test_returns_int(self):
        assert isinstance(now_ms(), int)

    def test_reasonable_value(self):
        assert now_ms() > 1_704_067_200_000  # after 2024-01-01

    def test_monotonically_increasing(self):
        a = now_ms()
        time.sleep(0.01)
        assert now_ms() >= a


class TestTimerSpan:
    def test_elapsed_grows(self):
        span = TimerSpan("test")
        time.sleep(0.05)
        assert span.elapsed_ms >= 40

    def test_stop_freezes_elapsed(self):
        span = TimerSpan("test")
        time.sleep(0.05)
        elapsed = span.stop()
        time.sleep(0.05)
        assert span.elapsed_ms == elapsed

    def test_context_manager(self):
        with TimerSpan("ctx") as span:
            time.sleep(0.05)
        assert span.elapsed_ms >= 40

    def test_name(self):
        assert TimerSpan("my_span").name == "my_span"

    def test_default_name(self):
        assert TimerSpan().name == "total"

    def test_start_ms_is_epoch(self):
        before = int(time.time() * 1000)
        span = TimerSpan()
        after = int(time.time() * 1000)
        assert before <= span.start_ms <= after


class TestBuildTimingPayload:
    def test_contains_required_fields(self):
        with TimerSpan("req") as span:
            time.sleep(0.01)
        p = build_timing_payload(span)
        assert "total_elapsed_ms" in p
        assert "started_at" in p
        assert "span" in p

    def test_total_elapsed_matches_span(self):
        with TimerSpan("req") as span:
            time.sleep(0.05)
        assert build_timing_payload(span)["total_elapsed_ms"] == span.elapsed_ms

    def test_span_name(self):
        with TimerSpan("my_span") as span:
            pass
        assert build_timing_payload(span)["span"] == "my_span"

    def test_started_at_is_iso(self):
        with TimerSpan() as span:
            pass
        assert "T" in build_timing_payload(span)["started_at"]

    def test_extra_fields_merged(self):
        with TimerSpan() as span:
            pass
        p = build_timing_payload(span, extra={"tool_name": "shell", "steps": 3})
        assert p["tool_name"] == "shell"
        assert p["steps"] == 3


# ============================================================================
# §2  Meta / response tracing  (was test_gateway_meta_tracing_response.py)
# ============================================================================


class TestMetaInfoTracing:
    def test_trace_id_optional_default_none(self):
        from spoon_bot.gateway.models.responses import MetaInfo
        assert MetaInfo(request_id="r").trace_id is None

    def test_trace_id_set(self):
        from spoon_bot.gateway.models.responses import MetaInfo
        assert MetaInfo(request_id="r", trace_id="trc_abc").trace_id == "trc_abc"

    def test_timing_optional_default_none(self):
        from spoon_bot.gateway.models.responses import MetaInfo
        assert MetaInfo(request_id="r").timing is None

    def test_timing_set(self):
        from spoon_bot.gateway.models.responses import MetaInfo
        m = MetaInfo(request_id="r", timing={"total_elapsed_ms": 150, "started_at": "2024-01-01T00:00:00Z"})
        assert m.timing["total_elapsed_ms"] == 150

    def test_backward_compatible(self):
        from spoon_bot.gateway.models.responses import MetaInfo
        m = MetaInfo(request_id="r", duration_ms=100)
        assert m.duration_ms == 100 and m.trace_id is None and m.timing is None

    def test_serialization_includes_trace(self):
        from spoon_bot.gateway.models.responses import MetaInfo
        d = MetaInfo(request_id="r", trace_id="trc_x", timing={"total_elapsed_ms": 50}).model_dump()
        assert d["trace_id"] == "trc_x"
        assert d["timing"]["total_elapsed_ms"] == 50

    def test_api_response_with_trace_meta(self):
        from spoon_bot.gateway.models.responses import MetaInfo, APIResponse
        meta = MetaInfo(request_id="r", trace_id="trc_test", timing={"total_elapsed_ms": 200})
        assert APIResponse(success=True, data={"msg": "ok"}, meta=meta).meta.trace_id == "trc_test"

    def test_error_response_with_trace_meta(self):
        from spoon_bot.gateway.models.responses import MetaInfo, ErrorResponse, ErrorDetail
        meta = MetaInfo(request_id="r", trace_id="trc_err")
        err = ErrorResponse(error=ErrorDetail(code="TEST", message="test error"), meta=meta)
        assert err.meta.trace_id == "trc_err"


# ============================================================================
# §3  QA error-code regression  (was test_gateway_qa_fixes.py)
# ============================================================================


class TestTimeoutErrorCodeStandardization:
    def test_timeout_upstream_code(self):
        from spoon_bot.gateway.errors import TimeoutCode
        assert TimeoutCode.TIMEOUT_UPSTREAM == "TIMEOUT_UPSTREAM"

    def test_timeout_tool_code(self):
        from spoon_bot.gateway.errors import TimeoutCode
        assert TimeoutCode.TIMEOUT_TOOL == "TIMEOUT_TOOL"

    def test_timeout_total_code(self):
        from spoon_bot.gateway.errors import TimeoutCode
        assert TimeoutCode.TIMEOUT_TOTAL == "TIMEOUT_TOTAL"

    def test_build_timeout_upstream_detail(self):
        from spoon_bot.gateway.errors import build_timeout_error_detail
        d = build_timeout_error_detail("TIMEOUT_UPSTREAM", elapsed_ms=5000, limit_ms=3000)
        assert d.code == "TIMEOUT_UPSTREAM"
        assert d.details["elapsed_ms"] == 5000
        assert d.details["limit_ms"] == 3000

    def test_build_timeout_tool_detail_context(self):
        from spoon_bot.gateway.errors import build_timeout_error_detail
        d = build_timeout_error_detail("TIMEOUT_TOOL", elapsed_ms=15000, limit_ms=10000, context="shell")
        assert d.code == "TIMEOUT_TOOL"
        assert d.details["context"] == "shell"

    def test_build_timeout_total_detail(self):
        from spoon_bot.gateway.errors import build_timeout_error_detail
        d = build_timeout_error_detail("TIMEOUT_TOTAL", elapsed_ms=120000, limit_ms=120000)
        assert d.code == "TIMEOUT_TOTAL"


# ============================================================================
# §4  Smart-fallback error handling  (was test_prompt_and_gateway_smart_fallback.py)
# ============================================================================


class TestSmartFallback:
    def test_error_response_includes_trace_id(self):
        from spoon_bot.gateway.errors import build_error_response
        from spoon_bot.gateway.models.responses import ErrorDetail
        resp = build_error_response(
            ErrorDetail(code="TEST_ERROR", message="test"),
            request_id="req_123", trace_id="trc_abc",
        )
        assert resp.meta.trace_id == "trc_abc"
        assert resp.meta.request_id == "req_123"

    def test_error_response_includes_timing(self):
        from spoon_bot.gateway.errors import build_error_response
        from spoon_bot.gateway.models.responses import ErrorDetail
        resp = build_error_response(
            ErrorDetail(code="TEST_ERROR", message="test"),
            request_id="req_123", timing={"total_elapsed_ms": 500},
        )
        assert resp.meta.timing["total_elapsed_ms"] == 500

    def test_timeout_error_response_full(self):
        from spoon_bot.gateway.errors import TimeoutCode, build_timeout_error_detail, build_error_response
        ed = build_timeout_error_detail(TimeoutCode.TIMEOUT_UPSTREAM, elapsed_ms=5000, limit_ms=3000)
        resp = build_error_response(
            ed, request_id="req_456", trace_id="trc_def",
            timing={"total_elapsed_ms": 5000},
        )
        assert resp.success is False
        assert resp.error.code == "TIMEOUT_UPSTREAM"
        assert resp.meta.trace_id == "trc_def"

    def test_all_gateway_error_codes_defined(self):
        from spoon_bot.gateway.errors import GatewayErrorCode
        codes = [e.value for e in GatewayErrorCode]
        for c in ("TIMEOUT_UPSTREAM", "TIMEOUT_TOOL", "TIMEOUT_TOTAL", "BUDGET_EXHAUSTED", "CANCELLED"):
            assert c in codes


# ============================================================================
# §5  Execution budget  (was test_gateway_execution_budget.py)
# ============================================================================


class TestExecutionBudget:
    def test_defaults_unlimited(self):
        from spoon_bot.gateway.observability.budget import ExecutionBudget
        b = ExecutionBudget()
        assert b.request_ms == 0 and b.tool_ms == 0 and b.stream_ms == 0
        assert b.is_unlimited()

    def test_custom_values(self):
        from spoon_bot.gateway.observability.budget import ExecutionBudget
        b = ExecutionBudget(request_ms=30000, tool_ms=10000, stream_ms=60000)
        assert b.request_ms == 30000 and not b.is_unlimited()

    def test_partial_unlimited(self):
        from spoon_bot.gateway.observability.budget import ExecutionBudget
        assert not ExecutionBudget(request_ms=30000).is_unlimited()


class TestCheckBudget:
    def test_within_budget_no_error(self):
        from spoon_bot.gateway.observability.budget import check_budget
        check_budget("request", limit_ms=30000, elapsed_ms=1000)

    def test_unlimited_no_error(self):
        from spoon_bot.gateway.observability.budget import check_budget
        check_budget("request", limit_ms=0, elapsed_ms=999999)

    def test_exceeded_raises(self):
        from spoon_bot.gateway.observability.budget import check_budget, BudgetExhaustedError
        with pytest.raises(BudgetExhaustedError) as exc:
            check_budget("request", limit_ms=30000, elapsed_ms=30000)
        assert exc.value.budget_type == "request"

    def test_exceeded_message(self):
        from spoon_bot.gateway.observability.budget import check_budget, BudgetExhaustedError
        with pytest.raises(BudgetExhaustedError, match="request budget exhausted"):
            check_budget("request", limit_ms=5000, elapsed_ms=6000)

    def test_tool_budget(self):
        from spoon_bot.gateway.observability.budget import check_budget, BudgetExhaustedError
        with pytest.raises(BudgetExhaustedError) as exc:
            check_budget("tool", limit_ms=10000, elapsed_ms=15000)
        assert exc.value.budget_type == "tool"

    def test_stream_budget(self):
        from spoon_bot.gateway.observability.budget import check_budget, BudgetExhaustedError
        with pytest.raises(BudgetExhaustedError) as exc:
            check_budget("stream", limit_ms=60000, elapsed_ms=61000)
        assert exc.value.budget_type == "stream"

    def test_just_under_limit_ok(self):
        from spoon_bot.gateway.observability.budget import check_budget
        check_budget("request", limit_ms=30000, elapsed_ms=29999)


class TestBudgetConfig:
    def test_defaults(self):
        from spoon_bot.gateway.config import BudgetConfig
        b = BudgetConfig()
        assert b.request_timeout_ms == 0
        assert b.tool_timeout_ms > 0
        assert b.stream_timeout_ms == 0

    def test_custom_values(self):
        from spoon_bot.gateway.config import BudgetConfig
        b = BudgetConfig(request_timeout_ms=5000, tool_timeout_ms=2000)
        assert b.request_timeout_ms == 5000


# ============================================================================
# §6  WS tracing / cancel / timeout  (was test_gateway_ws_tracing.py)
# ============================================================================


class _FakeConnection:
    def __init__(self):
        self.session_key = "default"
        self.messages = []


class _FakeManager:
    def __init__(self):
        self.sent_messages = []
        self._connection = _FakeConnection()

    def get_connection(self, conn_id):
        return self._connection

    async def send_message(self, conn_id, message):
        if hasattr(message, "to_dict"):
            self.sent_messages.append(message.to_dict())
        elif isinstance(message, dict):
            self.sent_messages.append(message)
        else:
            self.sent_messages.append(message)


def _make_mock_agent(response="Hello!", stream_chunks=None):
    agent = AsyncMock()
    agent.process = AsyncMock(return_value=response)
    agent.process_with_thinking = AsyncMock(return_value=(response, "thinking..."))

    async def fake_stream(**kwargs):
        for chunk in (stream_chunks or [
            {"type": "content", "delta": "Hello", "metadata": {}},
            {"type": "content", "delta": " world", "metadata": {}},
            {"type": "done", "delta": "", "metadata": {}},
        ]):
            yield chunk

    agent.stream = fake_stream
    return agent


class TestWsTracingInStreamingEvents:
    """WS streaming / complete / error events should contain trace_id."""

    @pytest.mark.asyncio
    async def test_streaming_events_contain_trace_id(self):
        from spoon_bot.gateway.websocket.handler import WebSocketHandler
        handler = WebSocketHandler("conn_test")
        fm = _FakeManager()
        with patch("spoon_bot.gateway.websocket.handler.get_agent", return_value=_make_mock_agent()):
            with patch("spoon_bot.gateway.websocket.handler.get_connection_manager", return_value=fm):
                with patch("spoon_bot.gateway.websocket.handler.get_config") as mc:
                    from spoon_bot.gateway.config import GatewayConfig
                    mc.return_value = GatewayConfig()
                    await handler._handle_chat({"message": "hi", "stream": True})
        traced = [m for m in fm.sent_messages
                  if isinstance(m, dict) and isinstance(m.get("data"), dict)
                  and "trace_id" in m.get("data", {})]
        assert len(traced) > 0

    @pytest.mark.asyncio
    async def test_streaming_chunk_events_contain_trace_id(self):
        from spoon_bot.gateway.websocket.handler import WebSocketHandler
        handler = WebSocketHandler("conn_test")
        fm = _FakeManager()
        with patch("spoon_bot.gateway.websocket.handler.get_agent", return_value=_make_mock_agent()):
            with patch("spoon_bot.gateway.websocket.handler.get_connection_manager", return_value=fm):
                with patch("spoon_bot.gateway.websocket.handler.get_config") as mc:
                    from spoon_bot.gateway.config import GatewayConfig
                    mc.return_value = GatewayConfig()
                    await handler._handle_chat({"message": "hi", "stream": True})
        chunks = [m for m in fm.sent_messages
                  if isinstance(m, dict) and m.get("event") == "agent.stream.chunk"]
        assert len(chunks) >= 1
        for ev in chunks:
            assert "trace_id" in ev["data"]

    @pytest.mark.asyncio
    async def test_stream_done_event_contains_trace_id(self):
        from spoon_bot.gateway.websocket.handler import WebSocketHandler
        handler = WebSocketHandler("conn_test")
        fm = _FakeManager()
        with patch("spoon_bot.gateway.websocket.handler.get_agent", return_value=_make_mock_agent()):
            with patch("spoon_bot.gateway.websocket.handler.get_connection_manager", return_value=fm):
                with patch("spoon_bot.gateway.websocket.handler.get_config") as mc:
                    from spoon_bot.gateway.config import GatewayConfig
                    mc.return_value = GatewayConfig()
                    await handler._handle_chat({"message": "hi", "stream": True})
        done = [m for m in fm.sent_messages
                if isinstance(m, dict) and m.get("event") == "agent.stream.done"]
        assert len(done) >= 1
        assert done[0]["data"]["trace_id"].startswith("trc_")

    @pytest.mark.asyncio
    async def test_complete_event_contains_trace_id(self):
        from spoon_bot.gateway.websocket.handler import WebSocketHandler
        handler = WebSocketHandler("conn_test")
        fm = _FakeManager()
        with patch("spoon_bot.gateway.websocket.handler.get_agent", return_value=_make_mock_agent()):
            with patch("spoon_bot.gateway.websocket.handler.get_connection_manager", return_value=fm):
                with patch("spoon_bot.gateway.websocket.handler.get_config") as mc:
                    from spoon_bot.gateway.config import GatewayConfig
                    mc.return_value = GatewayConfig()
                    await handler._handle_chat({"message": "hi", "stream": False})
        complete = [m for m in fm.sent_messages
                    if isinstance(m, dict) and m.get("event") == "agent.complete"]
        assert len(complete) >= 1
        assert complete[0]["data"]["trace_id"].startswith("trc_")

    @pytest.mark.asyncio
    async def test_complete_event_contains_timing(self):
        from spoon_bot.gateway.websocket.handler import WebSocketHandler
        handler = WebSocketHandler("conn_test")
        fm = _FakeManager()
        with patch("spoon_bot.gateway.websocket.handler.get_agent", return_value=_make_mock_agent()):
            with patch("spoon_bot.gateway.websocket.handler.get_connection_manager", return_value=fm):
                with patch("spoon_bot.gateway.websocket.handler.get_config") as mc:
                    from spoon_bot.gateway.config import GatewayConfig
                    mc.return_value = GatewayConfig()
                    await handler._handle_chat({"message": "hi", "stream": False})
        complete = [m for m in fm.sent_messages
                    if isinstance(m, dict) and m.get("event") == "agent.complete"]
        assert len(complete) >= 1
        assert "timing" in complete[0]["data"]
        assert "total_elapsed_ms" in complete[0]["data"]["timing"]

    @pytest.mark.asyncio
    async def test_result_contains_trace_id(self):
        from spoon_bot.gateway.websocket.handler import WebSocketHandler
        handler = WebSocketHandler("conn_test")
        fm = _FakeManager()
        with patch("spoon_bot.gateway.websocket.handler.get_agent", return_value=_make_mock_agent()):
            with patch("spoon_bot.gateway.websocket.handler.get_connection_manager", return_value=fm):
                with patch("spoon_bot.gateway.websocket.handler.get_config") as mc:
                    from spoon_bot.gateway.config import GatewayConfig
                    mc.return_value = GatewayConfig()
                    result = await handler._handle_chat({"message": "hi", "stream": False})
        assert result["trace_id"].startswith("trc_")

    @pytest.mark.asyncio
    async def test_result_contains_timing(self):
        from spoon_bot.gateway.websocket.handler import WebSocketHandler
        handler = WebSocketHandler("conn_test")
        fm = _FakeManager()
        with patch("spoon_bot.gateway.websocket.handler.get_agent", return_value=_make_mock_agent()):
            with patch("spoon_bot.gateway.websocket.handler.get_connection_manager", return_value=fm):
                with patch("spoon_bot.gateway.websocket.handler.get_config") as mc:
                    from spoon_bot.gateway.config import GatewayConfig
                    mc.return_value = GatewayConfig()
                    result = await handler._handle_chat({"message": "hi", "stream": False})
        assert "timing" in result
        assert "total_elapsed_ms" in result["timing"]
        assert "started_at" in result["timing"]
        assert result["timing"]["span"] == "ws_chat"

    @pytest.mark.asyncio
    async def test_thinking_event_contains_trace_id(self):
        from spoon_bot.gateway.websocket.handler import WebSocketHandler
        handler = WebSocketHandler("conn_test")
        fm = _FakeManager()
        with patch("spoon_bot.gateway.websocket.handler.get_agent", return_value=_make_mock_agent()):
            with patch("spoon_bot.gateway.websocket.handler.get_connection_manager", return_value=fm):
                with patch("spoon_bot.gateway.websocket.handler.get_config") as mc:
                    from spoon_bot.gateway.config import GatewayConfig
                    mc.return_value = GatewayConfig()
                    await handler._handle_chat({"message": "hi", "stream": False})
        thinking = [m for m in fm.sent_messages
                    if isinstance(m, dict) and m.get("event") == "agent.thinking"]
        assert len(thinking) >= 1
        assert thinking[0]["data"]["trace_id"].startswith("trc_")


class TestWsCancellationPropagation:
    """Client disconnect / cancel should cancel running tasks."""

    @pytest.mark.asyncio
    async def test_cancel_sets_flag_and_cancels_task(self):
        from spoon_bot.gateway.websocket.handler import WebSocketHandler
        handler = WebSocketHandler("conn_test")
        handler._current_task = asyncio.create_task(asyncio.sleep(100))
        result = await handler._handle_cancel({})
        assert result["cancelled"] is True
        assert handler._cancel_requested is True
        await asyncio.sleep(0.01)
        assert handler._current_task is None or handler._current_task.cancelled()

    @pytest.mark.asyncio
    async def test_cancel_without_running_task(self):
        from spoon_bot.gateway.websocket.handler import WebSocketHandler
        handler = WebSocketHandler("conn_test")
        handler._current_task = None
        result = await handler._handle_cancel({})
        assert result["cancelled"] is True
        assert handler._cancel_requested is True

    @pytest.mark.asyncio
    async def test_cancel_returns_task_id(self):
        from spoon_bot.gateway.websocket.handler import WebSocketHandler
        handler = WebSocketHandler("conn_test")
        handler._current_task_id = "task_abc12345"
        result = await handler._handle_cancel({})
        assert result["task_id"] == "task_abc12345"

    @pytest.mark.asyncio
    async def test_init_has_current_task_none(self):
        from spoon_bot.gateway.websocket.handler import WebSocketHandler
        assert WebSocketHandler("conn_test")._current_task is None

    @pytest.mark.asyncio
    async def test_current_task_cleared_after_chat_completes(self):
        from spoon_bot.gateway.websocket.handler import WebSocketHandler
        handler = WebSocketHandler("conn_test")
        fm = _FakeManager()
        with patch("spoon_bot.gateway.websocket.handler.get_agent", return_value=_make_mock_agent()):
            with patch("spoon_bot.gateway.websocket.handler.get_connection_manager", return_value=fm):
                with patch("spoon_bot.gateway.websocket.handler.get_config") as mc:
                    from spoon_bot.gateway.config import GatewayConfig
                    mc.return_value = GatewayConfig()
                    await handler._handle_chat({"message": "hi", "stream": False})
        assert handler._current_task is None

    @pytest.mark.asyncio
    async def test_disconnect_cleanup_cancels_background_task(self):
        from spoon_bot.gateway.websocket.handler import WebSocketHandler

        handler = WebSocketHandler("conn_test")
        handler._current_task = asyncio.create_task(asyncio.sleep(100))

        cancelled = await handler._cancel_current_task_for_cleanup(timeout=0.2)
        assert cancelled is True
        assert handler._current_task is None

    @pytest.mark.asyncio
    async def test_stream_chat_continues_after_connection_is_gone(self):
        from spoon_bot.gateway.websocket.handler import WebSocketHandler

        class _DisconnectedManager(_FakeManager):
            def get_connection(self, conn_id):
                return None

            async def send_message(self, conn_id, message):
                return False

        fm = _DisconnectedManager()

        with patch("spoon_bot.gateway.websocket.handler.get_agent", return_value=_make_mock_agent()):
            handler = WebSocketHandler("conn_test", "persisted-session")
            with patch("spoon_bot.gateway.websocket.handler.get_connection_manager", return_value=fm):
                with patch("spoon_bot.gateway.websocket.handler.get_config") as mc:
                    from spoon_bot.gateway.config import GatewayConfig
                    mc.return_value = GatewayConfig()
                    result = await handler._handle_chat({"message": "hi", "stream": True})

        assert result["success"] is True
        assert result["content"] == "Hello world"
        assert result["session_key"] == "persisted-session"
        assert fm.sent_messages == []

    @pytest.mark.asyncio
    async def test_websocket_disconnect_does_not_cancel_background_chat(self):
        from fastapi import WebSocketDisconnect
        from spoon_bot.gateway.config import GatewayConfig
        from spoon_bot.gateway.websocket import handler as ws_handler_module

        fake_ws = MagicMock()
        fake_ws.client = None
        fake_ws.receive_json = AsyncMock(side_effect=WebSocketDisconnect())

        fake_manager = MagicMock()
        fake_manager.connect = AsyncMock(return_value="conn_test")
        fake_manager.send_message = AsyncMock(return_value=True)
        fake_manager.disconnect = AsyncMock()

        fake_handler = MagicMock()
        fake_handler._cleanup_resources = AsyncMock()
        fake_handler._cancel_current_task_for_cleanup = AsyncMock(return_value=True)

        with patch.object(ws_handler_module, "get_config", return_value=GatewayConfig()):
            with patch.object(ws_handler_module, "get_connection_manager", return_value=fake_manager):
                with patch.object(ws_handler_module, "is_auth_required", return_value=False):
                    with patch.object(ws_handler_module, "WebSocketHandler", return_value=fake_handler):
                        await ws_handler_module.websocket_endpoint(fake_ws)

        fake_handler._cancel_current_task_for_cleanup.assert_not_awaited()
        fake_handler._cleanup_resources.assert_awaited_once()
        fake_manager.disconnect.assert_awaited_once_with("conn_test")


class TestWsTimeoutErrorCodes:
    """WS agent.error events use standardized timeout codes."""

    @pytest.mark.asyncio
    async def test_timeout_emits_error_event_with_upstream_code(self):
        from spoon_bot.gateway.websocket.handler import WebSocketHandler
        handler = WebSocketHandler("conn_test")
        mock_agent = AsyncMock()
        mock_agent.process = AsyncMock(side_effect=asyncio.TimeoutError())
        fm = _FakeManager()
        with patch("spoon_bot.gateway.websocket.handler.get_agent", return_value=mock_agent):
            with patch("spoon_bot.gateway.websocket.handler.get_connection_manager", return_value=fm):
                with patch("spoon_bot.gateway.websocket.handler.get_config") as mc:
                    from spoon_bot.gateway.config import GatewayConfig, BudgetConfig
                    cfg = GatewayConfig()
                    cfg.budget = BudgetConfig(request_timeout_ms=1000)
                    mc.return_value = cfg
                    with pytest.raises(asyncio.TimeoutError):
                        await handler._handle_chat({"message": "hi", "stream": False})
        errors = [m for m in fm.sent_messages
                  if isinstance(m, dict) and m.get("event") == "agent.error"]
        assert len(errors) >= 1
        assert errors[0]["data"]["trace_id"].startswith("trc_")
        assert errors[0]["data"]["error"]["code"] == "TIMEOUT_UPSTREAM"

    @pytest.mark.asyncio
    async def test_timeout_error_event_contains_timing(self):
        from spoon_bot.gateway.websocket.handler import WebSocketHandler
        handler = WebSocketHandler("conn_test")
        mock_agent = AsyncMock()
        mock_agent.process = AsyncMock(side_effect=asyncio.TimeoutError())
        fm = _FakeManager()
        with patch("spoon_bot.gateway.websocket.handler.get_agent", return_value=mock_agent):
            with patch("spoon_bot.gateway.websocket.handler.get_connection_manager", return_value=fm):
                with patch("spoon_bot.gateway.websocket.handler.get_config") as mc:
                    from spoon_bot.gateway.config import GatewayConfig, BudgetConfig
                    cfg = GatewayConfig()
                    cfg.budget = BudgetConfig(request_timeout_ms=1000)
                    mc.return_value = cfg
                    with pytest.raises(asyncio.TimeoutError):
                        await handler._handle_chat({"message": "hi", "stream": False})
        errors = [m for m in fm.sent_messages
                  if isinstance(m, dict) and m.get("event") == "agent.error"]
        assert len(errors) >= 1
        assert "timing" in errors[0]["data"]
        assert "total_elapsed_ms" in errors[0]["data"]["timing"]

    @pytest.mark.asyncio
    async def test_budget_exhausted_emits_error_event(self):
        from spoon_bot.gateway.websocket.handler import WebSocketHandler
        from spoon_bot.gateway.observability.budget import BudgetExhaustedError
        handler = WebSocketHandler("conn_test")
        fm = _FakeManager()
        with patch("spoon_bot.gateway.websocket.handler.get_agent", return_value=_make_mock_agent()):
            with patch("spoon_bot.gateway.websocket.handler.get_connection_manager", return_value=fm):
                with patch("spoon_bot.gateway.websocket.handler.get_config") as mc:
                    from spoon_bot.gateway.config import GatewayConfig, BudgetConfig
                    cfg = GatewayConfig()
                    cfg.budget = BudgetConfig(request_timeout_ms=0, tool_timeout_ms=0, stream_timeout_ms=0)
                    mc.return_value = cfg
                    with patch(
                        "spoon_bot.gateway.websocket.handler.check_budget",
                        side_effect=BudgetExhaustedError("request", 1000, 1500),
                    ):
                        with pytest.raises(BudgetExhaustedError):
                            await handler._handle_chat({"message": "hi", "stream": False})
        errors = [m for m in fm.sent_messages
                  if isinstance(m, dict) and m.get("event") == "agent.error"]
        assert len(errors) >= 1
        ed = errors[0]["data"]
        assert ed["trace_id"].startswith("trc_")
        assert ed["error"]["code"] == "TIMEOUT_TOTAL"
        assert ed["error"]["budget_type"] == "request"
        assert ed["error"]["elapsed_ms"] == 1500
        assert ed["error"]["limit_ms"] == 1000

    @pytest.mark.asyncio
    async def test_stream_budget_exhausted_emits_error(self):
        from spoon_bot.gateway.websocket.handler import WebSocketHandler
        from spoon_bot.gateway.observability.budget import BudgetExhaustedError
        handler = WebSocketHandler("conn_test")
        fm = _FakeManager()
        with patch("spoon_bot.gateway.websocket.handler.get_agent", return_value=_make_mock_agent()):
            with patch("spoon_bot.gateway.websocket.handler.get_connection_manager", return_value=fm):
                with patch("spoon_bot.gateway.websocket.handler.get_config") as mc:
                    from spoon_bot.gateway.config import GatewayConfig, BudgetConfig
                    cfg = GatewayConfig()
                    cfg.budget = BudgetConfig(stream_timeout_ms=1)
                    mc.return_value = cfg

                    def side_effect_check(budget_type, limit_ms, elapsed_ms):
                        if budget_type == "stream":
                            raise BudgetExhaustedError("stream", 1, 50)

                    with patch(
                        "spoon_bot.gateway.websocket.handler.check_budget",
                        side_effect=side_effect_check,
                    ):
                        with pytest.raises(BudgetExhaustedError):
                            await handler._handle_chat({"message": "hi", "stream": True})
        errors = [m for m in fm.sent_messages
                  if isinstance(m, dict) and m.get("event") == "agent.error"]
        assert len(errors) >= 1
        assert errors[0]["data"]["error"]["code"] == "TIMEOUT_TOTAL"
        assert errors[0]["data"]["error"]["budget_type"] == "stream"

    @pytest.mark.asyncio
    async def test_current_task_cleared_on_timeout(self):
        from spoon_bot.gateway.websocket.handler import WebSocketHandler
        handler = WebSocketHandler("conn_test")
        mock_agent = AsyncMock()
        mock_agent.process = AsyncMock(side_effect=asyncio.TimeoutError())
        fm = _FakeManager()
        with patch("spoon_bot.gateway.websocket.handler.get_agent", return_value=mock_agent):
            with patch("spoon_bot.gateway.websocket.handler.get_connection_manager", return_value=fm):
                with patch("spoon_bot.gateway.websocket.handler.get_config") as mc:
                    from spoon_bot.gateway.config import GatewayConfig
                    mc.return_value = GatewayConfig()
                    with pytest.raises(asyncio.TimeoutError):
                        await handler._handle_chat({"message": "hi", "stream": False})
        assert handler._current_task is None

    @pytest.mark.asyncio
    async def test_current_task_id_cleared_on_timeout(self):
        from spoon_bot.gateway.websocket.handler import WebSocketHandler
        handler = WebSocketHandler("conn_test")
        mock_agent = AsyncMock()
        mock_agent.process = AsyncMock(side_effect=asyncio.TimeoutError())
        fm = _FakeManager()
        with patch("spoon_bot.gateway.websocket.handler.get_agent", return_value=mock_agent):
            with patch("spoon_bot.gateway.websocket.handler.get_connection_manager", return_value=fm):
                with patch("spoon_bot.gateway.websocket.handler.get_config") as mc:
                    from spoon_bot.gateway.config import GatewayConfig
                    mc.return_value = GatewayConfig()
                    with pytest.raises(asyncio.TimeoutError):
                        await handler._handle_chat({"message": "hi", "stream": False})
        assert handler._current_task_id is None


class TestWsTraceIdUniqueness:
    @pytest.mark.asyncio
    async def test_different_chats_get_different_trace_ids(self):
        from spoon_bot.gateway.websocket.handler import WebSocketHandler
        handler = WebSocketHandler("conn_test")
        fm = _FakeManager()
        ids = []
        with patch("spoon_bot.gateway.websocket.handler.get_agent", return_value=_make_mock_agent()):
            with patch("spoon_bot.gateway.websocket.handler.get_connection_manager", return_value=fm):
                with patch("spoon_bot.gateway.websocket.handler.get_config") as mc:
                    from spoon_bot.gateway.config import GatewayConfig
                    mc.return_value = GatewayConfig()
                    for _ in range(3):
                        r = await handler._handle_chat({"message": "hi", "stream": False})
                        ids.append(r["trace_id"])
        assert len(set(ids)) == 3


# ============================================================================
# §7  REST tracing  (was test_gateway_agent_rest_tracing.py)
# ============================================================================


def _create_rest_test_app():
    from fastapi import FastAPI
    from spoon_bot.gateway.api.v1.agent import router
    app = FastAPI()
    app.include_router(router, prefix="/v1/agent")
    from spoon_bot.gateway.auth.dependencies import get_current_user

    class FakeUser:
        user_id = "test_user"
        session_key = "default"

    app.dependency_overrides[get_current_user] = lambda: FakeUser()
    return app


class TestChatJsonTraceAndTiming:
    def test_chat_json_contains_trace_and_timing(self):
        from fastapi.testclient import TestClient
        app = _create_rest_test_app()
        with patch("spoon_bot.gateway.api.v1.agent.get_agent", return_value=_make_mock_agent()):
            c = TestClient(app)
            resp = c.post("/v1/agent/chat", json={"message": "hello", "options": {"stream": False}})
        data = resp.json()
        assert data["meta"]["trace_id"] is not None
        assert data["meta"]["trace_id"].startswith("trc_")
        assert "total_elapsed_ms" in data["meta"]["timing"]

    def test_chat_json_backward_compatible(self):
        from fastapi.testclient import TestClient
        app = _create_rest_test_app()
        with patch("spoon_bot.gateway.api.v1.agent.get_agent", return_value=_make_mock_agent()):
            c = TestClient(app)
            resp = c.post("/v1/agent/chat", json={"message": "hi"})
        data = resp.json()
        assert "request_id" in data["meta"]
        assert data["meta"]["request_id"].startswith("req_")


class TestChatSseTraceAndTiming:
    def test_chat_sse_contains_trace_and_timing_events(self):
        from fastapi.testclient import TestClient
        app = _create_rest_test_app()
        with patch("spoon_bot.gateway.api.v1.agent.get_agent", return_value=_make_mock_agent()):
            c = TestClient(app)
            resp = c.post("/v1/agent/chat", json={"message": "hello", "options": {"stream": True}})
        body = resp.text
        events = []
        current_event = {}
        for line in body.strip().split("\n"):
            line = line.strip()
            if line.startswith("event:"):
                current_event["event"] = line[len("event:"):].strip()
            elif line.startswith("data:"):
                current_event["data"] = line[len("data:"):].strip()
                events.append(current_event)
                current_event = {}
            elif line == "" and current_event:
                events.append(current_event)
                current_event = {}
        trace_events = [e for e in events if e.get("event") == "trace"]
        assert len(trace_events) >= 1
        td = json.loads(trace_events[0]["data"])
        assert td["trace_id"].startswith("trc_")
        timing_events = [e for e in events if e.get("event") == "timing"]
        assert len(timing_events) >= 1
        assert "total_elapsed_ms" in json.loads(timing_events[0]["data"])
        data_events = [e for e in events if "data" in e]
        assert data_events[-1]["data"] == "[DONE]"

    def test_chat_sse_has_trace_id_header(self):
        from fastapi.testclient import TestClient
        app = _create_rest_test_app()
        with patch("spoon_bot.gateway.api.v1.agent.get_agent", return_value=_make_mock_agent()):
            c = TestClient(app)
            resp = c.post("/v1/agent/chat", json={"message": "hello", "options": {"stream": True}})
        assert "x-trace-id" in resp.headers

    def test_chat_sse_done_metadata_content_falls_back(self):
        from fastapi.testclient import TestClient

        async def done_only_stream(**kwargs):
            yield {"type": "done", "delta": "", "metadata": {"content": "fallback from done metadata"}}

        app = _create_rest_test_app()
        agent = AsyncMock()
        agent.stream = done_only_stream

        with patch("spoon_bot.gateway.api.v1.agent.get_agent", return_value=agent):
            c = TestClient(app)
            resp = c.post("/v1/agent/chat", json={"message": "hello", "options": {"stream": True}})

        assert resp.status_code == 200
        assert "fallback from done metadata" in resp.text


class TestRestTimeoutErrorCodes:
    def test_upstream_timeout_returns_error(self):
        from fastapi.testclient import TestClient
        app = _create_rest_test_app()
        mock_agent = _make_mock_agent()
        mock_agent.process = AsyncMock(side_effect=asyncio.TimeoutError())
        with patch("spoon_bot.gateway.api.v1.agent.get_agent", return_value=mock_agent):
            with patch("spoon_bot.gateway.api.v1.agent.get_config") as mc:
                from spoon_bot.gateway.config import GatewayConfig, BudgetConfig
                cfg = GatewayConfig()
                cfg.budget = BudgetConfig(request_timeout_ms=1000)
                mc.return_value = cfg
                c = TestClient(app)
                resp = c.post("/v1/agent/chat", json={"message": "hello"})
        assert resp.status_code in (408, 500, 504)
        data = resp.json()
        detail = data.get("detail", data.get("error", {}))
        if isinstance(detail, dict):
            assert detail.get("code") in ("TIMEOUT_UPSTREAM", "TIMEOUT_TOTAL", "AGENT_ERROR")


# ============================================================================
# §8  REST cancellation propagation  (was test_gateway_cancel_propagation.py)
# ============================================================================


class TestRestCancellation:
    @pytest.mark.asyncio
    async def test_cancel_event_stops_streaming(self):
        from spoon_bot.gateway.api.v1.agent import _stream_sse

        cancel_event = asyncio.Event()
        mock_agent = AsyncMock()

        async def slow_stream(**kw):
            for i in range(10):
                await asyncio.sleep(0.01)
                yield {"type": "content", "delta": f"chunk{i}", "metadata": {}}
            yield {"type": "done", "delta": "", "metadata": {}}

        mock_agent.stream = slow_stream

        async def set_cancel():
            await asyncio.sleep(0.03)
            cancel_event.set()

        asyncio.create_task(set_cancel())

        chunks_yielded = []
        async for chunk in _stream_sse(
            mock_agent, "test", None, False,
            trace_id="trc_test", request_id="req_test",
            cancel_event=cancel_event,
        ):
            chunks_yielded.append(chunk)
        content_chunks = [c for c in chunks_yielded if "content" in c and "chunk" in c]
        assert len(content_chunks) < 10


# ============================================================================
# §9  Toolkit adapter timeout  (was test_toolkit_adapter_timeout.py)
# ============================================================================


class TestToolkitAdapterTimeout:
    @pytest.mark.asyncio
    async def test_toolkit_wrapper_moves_to_background_after_timeout(self):
        from spoon_bot.toolkit.adapter import ToolkitToolWrapper

        class _SlowAsyncTool:
            name = "slow_async"
            description = "slow tool"
            async def execute(self, **kwargs):
                await asyncio.sleep(0.2)
                return "done"

        tool = ToolkitToolWrapper(_SlowAsyncTool(), timeout_seconds=0.05)
        result = await tool.execute()
        assert "background" in result.lower()
        assert "job_id:" in result

    @pytest.mark.asyncio
    async def test_toolkit_background_job_returns_result_when_ready(self):
        from spoon_bot.toolkit.adapter import ToolkitToolWrapper

        class _SlowAsyncTool:
            name = "slow_async_result"
            description = "slow tool"

            async def execute(self, **kwargs):
                await asyncio.sleep(0.1)
                return "done"

        tool = ToolkitToolWrapper(_SlowAsyncTool(), timeout_seconds=0.01)
        background = await tool.execute()
        job_id = background.split("job_id:", 1)[1].splitlines()[0].strip()
        await asyncio.sleep(0.15)
        result = await tool.execute(action="job_output", job_id=job_id)
        assert result == "done"

    @pytest.mark.asyncio
    async def test_toolkit_background_jobs_are_scoped_by_owner(self):
        from spoon_bot.agent.tools.execution_context import bind_tool_owner
        from spoon_bot.toolkit.adapter import ToolkitToolWrapper

        class _SlowAsyncTool:
            name = "slow_async_owner"
            description = "slow tool"

            async def execute(self, **kwargs):
                await asyncio.sleep(0.15)
                return "done"

        tool = ToolkitToolWrapper(_SlowAsyncTool(), timeout_seconds=0.01)
        with bind_tool_owner("owner-a"):
            background = await tool.execute()
            job_id = background.split("job_id:", 1)[1].splitlines()[0].strip()

        with bind_tool_owner("owner-b"):
            denied = await tool.execute(action="job_status", job_id=job_id)

        assert "not found" in denied.lower()
