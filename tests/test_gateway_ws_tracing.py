"""Tests for WebSocket tracing, timing, cancellation propagation, and timeout error codes."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from spoon_bot.gateway.websocket.handler import WebSocketHandler
from spoon_bot.gateway.websocket.protocol import WSRequest, WSEvent, WSResponse, WSError


class FakeConnection:
    """Fake connection for testing."""
    def __init__(self):
        self.session_key = "default"
        self.messages = []


class FakeManager:
    """Fake connection manager for testing."""
    def __init__(self):
        self.sent_messages = []
        self._connection = FakeConnection()

    def get_connection(self, conn_id):
        return self._connection

    async def send_message(self, conn_id, message):
        if hasattr(message, 'to_dict'):
            self.sent_messages.append(message.to_dict())
        elif isinstance(message, dict):
            self.sent_messages.append(message)
        else:
            self.sent_messages.append(message)


def _make_mock_agent(response="Hello!", stream_chunks=None):
    """Create a mock agent with process/stream/process_with_thinking support."""
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
    """P0-1.6: WS streaming/complete/error events should contain trace_id."""

    @pytest.mark.asyncio
    async def test_streaming_events_contain_trace_id(self):
        handler = WebSocketHandler("conn_test")
        mock_agent = _make_mock_agent()
        fake_manager = FakeManager()

        with patch("spoon_bot.gateway.websocket.handler.get_agent", return_value=mock_agent):
            with patch("spoon_bot.gateway.websocket.handler.get_connection_manager", return_value=fake_manager):
                with patch("spoon_bot.gateway.websocket.handler.get_config") as mock_cfg:
                    from spoon_bot.gateway.config import GatewayConfig
                    mock_cfg.return_value = GatewayConfig()
                    result = await handler._handle_chat({
                        "message": "hello",
                        "stream": True,
                    })

        # Check that events were sent with trace_id
        events_with_trace = [
            m for m in fake_manager.sent_messages
            if isinstance(m, dict) and "data" in m and isinstance(m.get("data"), dict)
            and "trace_id" in m.get("data", {})
        ]
        assert len(events_with_trace) > 0, f"No events with trace_id found. Messages: {fake_manager.sent_messages}"

    @pytest.mark.asyncio
    async def test_streaming_chunk_events_contain_trace_id(self):
        handler = WebSocketHandler("conn_test")
        mock_agent = _make_mock_agent()
        fake_manager = FakeManager()

        with patch("spoon_bot.gateway.websocket.handler.get_agent", return_value=mock_agent):
            with patch("spoon_bot.gateway.websocket.handler.get_connection_manager", return_value=fake_manager):
                with patch("spoon_bot.gateway.websocket.handler.get_config") as mock_cfg:
                    from spoon_bot.gateway.config import GatewayConfig
                    mock_cfg.return_value = GatewayConfig()
                    await handler._handle_chat({
                        "message": "hello",
                        "stream": True,
                    })

        # Find agent.stream.chunk events specifically
        chunk_events = [
            m for m in fake_manager.sent_messages
            if isinstance(m, dict) and m.get("event") == "agent.stream.chunk"
        ]
        assert len(chunk_events) >= 1, f"No stream chunk events found. Messages: {fake_manager.sent_messages}"
        for ev in chunk_events:
            assert "trace_id" in ev["data"], f"Chunk event missing trace_id: {ev}"

    @pytest.mark.asyncio
    async def test_stream_done_event_contains_trace_id(self):
        handler = WebSocketHandler("conn_test")
        mock_agent = _make_mock_agent()
        fake_manager = FakeManager()

        with patch("spoon_bot.gateway.websocket.handler.get_agent", return_value=mock_agent):
            with patch("spoon_bot.gateway.websocket.handler.get_connection_manager", return_value=fake_manager):
                with patch("spoon_bot.gateway.websocket.handler.get_config") as mock_cfg:
                    from spoon_bot.gateway.config import GatewayConfig
                    mock_cfg.return_value = GatewayConfig()
                    await handler._handle_chat({
                        "message": "hello",
                        "stream": True,
                    })

        done_events = [
            m for m in fake_manager.sent_messages
            if isinstance(m, dict) and m.get("event") == "agent.stream.done"
        ]
        assert len(done_events) >= 1
        assert "trace_id" in done_events[0]["data"]
        assert done_events[0]["data"]["trace_id"].startswith("trc_")

    @pytest.mark.asyncio
    async def test_complete_event_contains_trace_id(self):
        handler = WebSocketHandler("conn_test")
        mock_agent = _make_mock_agent()
        fake_manager = FakeManager()

        with patch("spoon_bot.gateway.websocket.handler.get_agent", return_value=mock_agent):
            with patch("spoon_bot.gateway.websocket.handler.get_connection_manager", return_value=fake_manager):
                with patch("spoon_bot.gateway.websocket.handler.get_config") as mock_cfg:
                    from spoon_bot.gateway.config import GatewayConfig
                    mock_cfg.return_value = GatewayConfig()
                    result = await handler._handle_chat({
                        "message": "hello",
                        "stream": False,
                    })

        # Find agent.complete event
        complete_events = [
            m for m in fake_manager.sent_messages
            if isinstance(m, dict) and m.get("event") == "agent.complete"
        ]
        assert len(complete_events) >= 1, f"No agent.complete event found. Messages: {fake_manager.sent_messages}"
        assert "trace_id" in complete_events[0]["data"]
        assert complete_events[0]["data"]["trace_id"].startswith("trc_")

    @pytest.mark.asyncio
    async def test_complete_event_contains_timing(self):
        handler = WebSocketHandler("conn_test")
        mock_agent = _make_mock_agent()
        fake_manager = FakeManager()

        with patch("spoon_bot.gateway.websocket.handler.get_agent", return_value=mock_agent):
            with patch("spoon_bot.gateway.websocket.handler.get_connection_manager", return_value=fake_manager):
                with patch("spoon_bot.gateway.websocket.handler.get_config") as mock_cfg:
                    from spoon_bot.gateway.config import GatewayConfig
                    mock_cfg.return_value = GatewayConfig()
                    result = await handler._handle_chat({
                        "message": "hello",
                        "stream": False,
                    })

        complete_events = [
            m for m in fake_manager.sent_messages
            if isinstance(m, dict) and m.get("event") == "agent.complete"
        ]
        assert len(complete_events) >= 1
        data = complete_events[0]["data"]
        assert "timing" in data
        assert "total_elapsed_ms" in data["timing"]

    @pytest.mark.asyncio
    async def test_result_contains_trace_id(self):
        handler = WebSocketHandler("conn_test")
        mock_agent = _make_mock_agent()
        fake_manager = FakeManager()

        with patch("spoon_bot.gateway.websocket.handler.get_agent", return_value=mock_agent):
            with patch("spoon_bot.gateway.websocket.handler.get_connection_manager", return_value=fake_manager):
                with patch("spoon_bot.gateway.websocket.handler.get_config") as mock_cfg:
                    from spoon_bot.gateway.config import GatewayConfig
                    mock_cfg.return_value = GatewayConfig()
                    result = await handler._handle_chat({
                        "message": "hello",
                        "stream": False,
                    })

        assert "trace_id" in result
        assert result["trace_id"].startswith("trc_")

    @pytest.mark.asyncio
    async def test_result_contains_timing(self):
        handler = WebSocketHandler("conn_test")
        mock_agent = _make_mock_agent()
        fake_manager = FakeManager()

        with patch("spoon_bot.gateway.websocket.handler.get_agent", return_value=mock_agent):
            with patch("spoon_bot.gateway.websocket.handler.get_connection_manager", return_value=fake_manager):
                with patch("spoon_bot.gateway.websocket.handler.get_config") as mock_cfg:
                    from spoon_bot.gateway.config import GatewayConfig
                    mock_cfg.return_value = GatewayConfig()
                    result = await handler._handle_chat({
                        "message": "hello",
                        "stream": False,
                    })

        assert "timing" in result
        assert "total_elapsed_ms" in result["timing"]
        assert "started_at" in result["timing"]
        assert "span" in result["timing"]
        assert result["timing"]["span"] == "ws_chat"

    @pytest.mark.asyncio
    async def test_thinking_event_contains_trace_id(self):
        handler = WebSocketHandler("conn_test")
        mock_agent = _make_mock_agent()
        fake_manager = FakeManager()

        with patch("spoon_bot.gateway.websocket.handler.get_agent", return_value=mock_agent):
            with patch("spoon_bot.gateway.websocket.handler.get_connection_manager", return_value=fake_manager):
                with patch("spoon_bot.gateway.websocket.handler.get_config") as mock_cfg:
                    from spoon_bot.gateway.config import GatewayConfig
                    mock_cfg.return_value = GatewayConfig()
                    await handler._handle_chat({
                        "message": "hello",
                        "stream": False,
                    })

        thinking_events = [
            m for m in fake_manager.sent_messages
            if isinstance(m, dict) and m.get("event") == "agent.thinking"
        ]
        assert len(thinking_events) >= 1
        assert "trace_id" in thinking_events[0]["data"]
        assert thinking_events[0]["data"]["trace_id"].startswith("trc_")


class TestWsCancellationPropagation:
    """P0-2.2: Client disconnect should cancel running tasks."""

    @pytest.mark.asyncio
    async def test_cancel_sets_flag_and_cancels_task(self):
        handler = WebSocketHandler("conn_test")
        # Simulate a running task
        async def long_running():
            await asyncio.sleep(100)

        handler._current_task = asyncio.create_task(long_running())

        result = await handler._handle_cancel({})
        assert result["cancelled"] is True
        assert handler._cancel_requested is True

        # Task should be cancelled
        await asyncio.sleep(0.01)
        assert handler._current_task is None or handler._current_task.cancelled()

    @pytest.mark.asyncio
    async def test_cancel_without_running_task(self):
        handler = WebSocketHandler("conn_test")
        handler._current_task = None
        result = await handler._handle_cancel({})
        assert result["cancelled"] is True
        assert handler._cancel_requested is True

    @pytest.mark.asyncio
    async def test_cancel_returns_task_id(self):
        handler = WebSocketHandler("conn_test")
        handler._current_task_id = "task_abc12345"
        result = await handler._handle_cancel({})
        assert result["task_id"] == "task_abc12345"

    @pytest.mark.asyncio
    async def test_init_has_current_task_none(self):
        handler = WebSocketHandler("conn_test")
        assert handler._current_task is None

    @pytest.mark.asyncio
    async def test_current_task_cleared_after_chat_completes(self):
        handler = WebSocketHandler("conn_test")
        mock_agent = _make_mock_agent()
        fake_manager = FakeManager()

        with patch("spoon_bot.gateway.websocket.handler.get_agent", return_value=mock_agent):
            with patch("spoon_bot.gateway.websocket.handler.get_connection_manager", return_value=fake_manager):
                with patch("spoon_bot.gateway.websocket.handler.get_config") as mock_cfg:
                    from spoon_bot.gateway.config import GatewayConfig
                    mock_cfg.return_value = GatewayConfig()
                    await handler._handle_chat({
                        "message": "hello",
                        "stream": False,
                    })

        # After _handle_chat completes, _current_task should be None
        assert handler._current_task is None


class TestWsTimeoutErrorCodes:
    """P0-2.3: WS agent.error events use standardized timeout codes."""

    @pytest.mark.asyncio
    async def test_timeout_emits_error_event_with_upstream_code(self):
        handler = WebSocketHandler("conn_test")
        mock_agent = AsyncMock()
        mock_agent.process = AsyncMock(side_effect=asyncio.TimeoutError())
        fake_manager = FakeManager()

        with patch("spoon_bot.gateway.websocket.handler.get_agent", return_value=mock_agent):
            with patch("spoon_bot.gateway.websocket.handler.get_connection_manager", return_value=fake_manager):
                with patch("spoon_bot.gateway.websocket.handler.get_config") as mock_cfg:
                    from spoon_bot.gateway.config import GatewayConfig, BudgetConfig
                    cfg = GatewayConfig()
                    cfg.budget = BudgetConfig(request_timeout_ms=1000)
                    mock_cfg.return_value = cfg

                    with pytest.raises(asyncio.TimeoutError):
                        await handler._handle_chat({
                            "message": "hello",
                            "stream": False,
                        })

        # Check that an agent.error event was sent
        error_events = [
            m for m in fake_manager.sent_messages
            if isinstance(m, dict) and m.get("event") == "agent.error"
        ]
        assert len(error_events) >= 1, f"No agent.error event found. Messages: {fake_manager.sent_messages}"
        error_data = error_events[0]["data"]
        assert "trace_id" in error_data
        assert error_data["trace_id"].startswith("trc_")
        assert error_data["error"]["code"] == "TIMEOUT_UPSTREAM"

    @pytest.mark.asyncio
    async def test_timeout_error_event_contains_timing(self):
        handler = WebSocketHandler("conn_test")
        mock_agent = AsyncMock()
        mock_agent.process = AsyncMock(side_effect=asyncio.TimeoutError())
        fake_manager = FakeManager()

        with patch("spoon_bot.gateway.websocket.handler.get_agent", return_value=mock_agent):
            with patch("spoon_bot.gateway.websocket.handler.get_connection_manager", return_value=fake_manager):
                with patch("spoon_bot.gateway.websocket.handler.get_config") as mock_cfg:
                    from spoon_bot.gateway.config import GatewayConfig, BudgetConfig
                    cfg = GatewayConfig()
                    cfg.budget = BudgetConfig(request_timeout_ms=1000)
                    mock_cfg.return_value = cfg

                    with pytest.raises(asyncio.TimeoutError):
                        await handler._handle_chat({
                            "message": "hello",
                            "stream": False,
                        })

        error_events = [
            m for m in fake_manager.sent_messages
            if isinstance(m, dict) and m.get("event") == "agent.error"
        ]
        assert len(error_events) >= 1
        assert "timing" in error_events[0]["data"]
        assert "total_elapsed_ms" in error_events[0]["data"]["timing"]

    @pytest.mark.asyncio
    async def test_budget_exhausted_emits_error_event(self):
        handler = WebSocketHandler("conn_test")
        mock_agent = _make_mock_agent()
        fake_manager = FakeManager()

        with patch("spoon_bot.gateway.websocket.handler.get_agent", return_value=mock_agent):
            with patch("spoon_bot.gateway.websocket.handler.get_connection_manager", return_value=fake_manager):
                with patch("spoon_bot.gateway.websocket.handler.get_config") as mock_cfg:
                    from spoon_bot.gateway.config import GatewayConfig, BudgetConfig
                    # Set an extremely low request timeout so the budget check fires
                    cfg = GatewayConfig()
                    cfg.budget = BudgetConfig(request_timeout_ms=0, tool_timeout_ms=0, stream_timeout_ms=0)
                    mock_cfg.return_value = cfg

                    # With budget=0 (unlimited), check_budget won't fire.
                    # Instead, patch check_budget to raise.
                    from spoon_bot.gateway.observability.budget import BudgetExhaustedError
                    with patch(
                        "spoon_bot.gateway.websocket.handler.check_budget",
                        side_effect=BudgetExhaustedError("request", 1000, 1500),
                    ):
                        with pytest.raises(BudgetExhaustedError):
                            await handler._handle_chat({
                                "message": "hello",
                                "stream": False,
                            })

        error_events = [
            m for m in fake_manager.sent_messages
            if isinstance(m, dict) and m.get("event") == "agent.error"
        ]
        assert len(error_events) >= 1
        error_data = error_events[0]["data"]
        assert "trace_id" in error_data
        assert error_data["error"]["code"] == "TIMEOUT_TOTAL"
        assert error_data["error"]["budget_type"] == "request"
        assert error_data["error"]["elapsed_ms"] == 1500
        assert error_data["error"]["limit_ms"] == 1000

    @pytest.mark.asyncio
    async def test_stream_budget_exhausted_emits_error(self):
        handler = WebSocketHandler("conn_test")
        mock_agent = _make_mock_agent()
        fake_manager = FakeManager()

        with patch("spoon_bot.gateway.websocket.handler.get_agent", return_value=mock_agent):
            with patch("spoon_bot.gateway.websocket.handler.get_connection_manager", return_value=fake_manager):
                with patch("spoon_bot.gateway.websocket.handler.get_config") as mock_cfg:
                    from spoon_bot.gateway.config import GatewayConfig, BudgetConfig
                    cfg = GatewayConfig()
                    cfg.budget = BudgetConfig(stream_timeout_ms=1)  # 1ms - will exhaust immediately
                    mock_cfg.return_value = cfg

                    # Patch check_budget to raise on stream type
                    from spoon_bot.gateway.observability.budget import BudgetExhaustedError
                    call_count = 0

                    def side_effect_check(budget_type, limit_ms, elapsed_ms):
                        nonlocal call_count
                        call_count += 1
                        if budget_type == "stream":
                            raise BudgetExhaustedError("stream", 1, 50)

                    with patch(
                        "spoon_bot.gateway.websocket.handler.check_budget",
                        side_effect=side_effect_check,
                    ):
                        with pytest.raises(BudgetExhaustedError):
                            await handler._handle_chat({
                                "message": "hello",
                                "stream": True,
                            })

        error_events = [
            m for m in fake_manager.sent_messages
            if isinstance(m, dict) and m.get("event") == "agent.error"
        ]
        assert len(error_events) >= 1
        assert error_events[0]["data"]["error"]["code"] == "TIMEOUT_TOTAL"
        assert error_events[0]["data"]["error"]["budget_type"] == "stream"

    @pytest.mark.asyncio
    async def test_current_task_cleared_on_timeout(self):
        handler = WebSocketHandler("conn_test")
        mock_agent = AsyncMock()
        mock_agent.process = AsyncMock(side_effect=asyncio.TimeoutError())
        fake_manager = FakeManager()

        with patch("spoon_bot.gateway.websocket.handler.get_agent", return_value=mock_agent):
            with patch("spoon_bot.gateway.websocket.handler.get_connection_manager", return_value=fake_manager):
                with patch("spoon_bot.gateway.websocket.handler.get_config") as mock_cfg:
                    from spoon_bot.gateway.config import GatewayConfig
                    mock_cfg.return_value = GatewayConfig()

                    with pytest.raises(asyncio.TimeoutError):
                        await handler._handle_chat({
                            "message": "hello",
                            "stream": False,
                        })

        # _current_task should be cleared in finally block
        assert handler._current_task is None

    @pytest.mark.asyncio
    async def test_current_task_id_cleared_on_timeout(self):
        handler = WebSocketHandler("conn_test")
        mock_agent = AsyncMock()
        mock_agent.process = AsyncMock(side_effect=asyncio.TimeoutError())
        fake_manager = FakeManager()

        with patch("spoon_bot.gateway.websocket.handler.get_agent", return_value=mock_agent):
            with patch("spoon_bot.gateway.websocket.handler.get_connection_manager", return_value=fake_manager):
                with patch("spoon_bot.gateway.websocket.handler.get_config") as mock_cfg:
                    from spoon_bot.gateway.config import GatewayConfig
                    mock_cfg.return_value = GatewayConfig()

                    with pytest.raises(asyncio.TimeoutError):
                        await handler._handle_chat({
                            "message": "hello",
                            "stream": False,
                        })

        assert handler._current_task_id is None


class TestWsTraceIdUniqueness:
    """Verify that each chat invocation gets a unique trace_id."""

    @pytest.mark.asyncio
    async def test_different_chats_get_different_trace_ids(self):
        handler = WebSocketHandler("conn_test")
        mock_agent = _make_mock_agent()
        fake_manager = FakeManager()
        trace_ids = []

        with patch("spoon_bot.gateway.websocket.handler.get_agent", return_value=mock_agent):
            with patch("spoon_bot.gateway.websocket.handler.get_connection_manager", return_value=fake_manager):
                with patch("spoon_bot.gateway.websocket.handler.get_config") as mock_cfg:
                    from spoon_bot.gateway.config import GatewayConfig
                    mock_cfg.return_value = GatewayConfig()

                    for _ in range(3):
                        result = await handler._handle_chat({
                            "message": "hello",
                            "stream": False,
                        })
                        trace_ids.append(result["trace_id"])

        # All trace_ids should be unique
        assert len(set(trace_ids)) == 3, f"Expected 3 unique trace_ids, got {trace_ids}"
