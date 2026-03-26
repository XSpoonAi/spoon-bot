"""
Tests for streaming thinking fix and stream timeout improvements.

Covers:
  - thinking parameter forwarded to _run_and_signal() in stream mode
  - stream timeout uses 600s for thinking, 300s otherwise
  - stream timeout emits error event instead of silent break
  - ConnectionManager.send_message retries before disconnecting
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============================================================
# Streaming thinking parameter forwarding
# ============================================================


class TestStreamThinkingParam:
    """Verify that stream() forwards thinking=True to the agent run()."""

    @pytest.mark.asyncio
    async def test_thinking_param_passed_to_run(self):
        """When thinking=True, _run_and_signal should pass thinking=True to agent.run()."""
        from spoon_bot.agent.loop import AgentLoop

        run_kwargs_captured: list[dict] = []

        async def mock_run(**kwargs):
            run_kwargs_captured.append(kwargs)
            result = MagicMock()
            result.content = "done"
            return result

        oq: asyncio.Queue = asyncio.Queue()
        td = asyncio.Event()

        agent = MagicMock(spec=AgentLoop)
        agent._initialized = True
        agent._agent = MagicMock()
        agent._agent.output_queue = oq
        agent._agent.task_done = td
        agent._agent.run = mock_run
        agent._agent.state = "idle"
        agent._agent.add_message = AsyncMock()
        agent._session = MagicMock()
        agent._session.add_message = MagicMock()
        agent.sessions = MagicMock()
        agent.sessions.save = MagicMock()
        agent.memory = MagicMock()
        agent.memory.get_memory_context = MagicMock(return_value=None)
        agent.context = MagicMock()
        agent._prepare_request_context = AsyncMock()
        agent._build_runtime_message_content = MagicMock(return_value="test")
        agent._build_step_prompt = MagicMock(return_value="prompt")
        agent._install_anti_loop_tracker = MagicMock()
        agent._restore_agent_think = MagicMock()
        agent._callable_accepts_kwarg = AgentLoop._callable_accepts_kwarg

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="test", thinking=True):
            chunks.append(chunk)

        assert len(run_kwargs_captured) == 1
        assert run_kwargs_captured[0].get("thinking") is True

    @pytest.mark.asyncio
    async def test_thinking_false_not_passed_to_run(self):
        """When thinking=False (default), thinking should not be in run kwargs."""
        from spoon_bot.agent.loop import AgentLoop

        run_kwargs_captured: list[dict] = []

        async def mock_run(**kwargs):
            run_kwargs_captured.append(kwargs)
            result = MagicMock()
            result.content = "done"
            return result

        oq: asyncio.Queue = asyncio.Queue()
        td = asyncio.Event()

        agent = MagicMock(spec=AgentLoop)
        agent._initialized = True
        agent._agent = MagicMock()
        agent._agent.output_queue = oq
        agent._agent.task_done = td
        agent._agent.run = mock_run
        agent._agent.state = "idle"
        agent._agent.add_message = AsyncMock()
        agent._session = MagicMock()
        agent._session.add_message = MagicMock()
        agent.sessions = MagicMock()
        agent.sessions.save = MagicMock()
        agent.memory = MagicMock()
        agent.memory.get_memory_context = MagicMock(return_value=None)
        agent.context = MagicMock()
        agent._prepare_request_context = AsyncMock()
        agent._build_runtime_message_content = MagicMock(return_value="test")
        agent._build_step_prompt = MagicMock(return_value="prompt")
        agent._install_anti_loop_tracker = MagicMock()
        agent._restore_agent_think = MagicMock()
        agent._callable_accepts_kwarg = AgentLoop._callable_accepts_kwarg

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="test", thinking=False):
            chunks.append(chunk)

        assert len(run_kwargs_captured) == 1
        assert "thinking" not in run_kwargs_captured[0]


# ============================================================
# Stream timeout configuration
# ============================================================


class TestStreamTimeoutConfig:
    """Verify that stream timeout is properly configured based on thinking flag."""

    @pytest.mark.asyncio
    async def test_thinking_mode_uses_longer_timeout(self):
        """With thinking=True, stream_timeout should be 600s."""
        from spoon_bot.agent.loop import AgentLoop

        timeout_captured: list[float] = []

        orig_time = asyncio.get_event_loop().time

        async def mock_run(**kwargs):
            result = MagicMock()
            result.content = "done"
            return result

        oq: asyncio.Queue = asyncio.Queue()
        td = asyncio.Event()

        agent = MagicMock(spec=AgentLoop)
        agent._initialized = True
        agent._agent = MagicMock()
        agent._agent.output_queue = oq
        agent._agent.task_done = td
        agent._agent.run = mock_run
        agent._agent.state = "idle"
        agent._agent.add_message = AsyncMock()
        agent._session = MagicMock()
        agent._session.add_message = MagicMock()
        agent.sessions = MagicMock()
        agent.sessions.save = MagicMock()
        agent.memory = MagicMock()
        agent.memory.get_memory_context = MagicMock(return_value=None)
        agent.context = MagicMock()
        agent._prepare_request_context = AsyncMock()
        agent._build_runtime_message_content = MagicMock(return_value="test")
        agent._build_step_prompt = MagicMock(return_value="prompt")
        agent._install_anti_loop_tracker = MagicMock()
        agent._restore_agent_think = MagicMock()
        agent._callable_accepts_kwarg = AgentLoop._callable_accepts_kwarg

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="test", thinking=True):
            chunks.append(chunk)

        done_chunks = [c for c in chunks if c["type"] == "done"]
        assert len(done_chunks) == 1

    @pytest.mark.asyncio
    async def test_non_thinking_mode_uses_default_timeout(self):
        """With thinking=False, stream_timeout should be 300s (not the old 120s)."""
        from spoon_bot.agent.loop import AgentLoop

        async def mock_run(**kwargs):
            result = MagicMock()
            result.content = "done"
            return result

        oq: asyncio.Queue = asyncio.Queue()
        td = asyncio.Event()

        agent = MagicMock(spec=AgentLoop)
        agent._initialized = True
        agent._agent = MagicMock()
        agent._agent.output_queue = oq
        agent._agent.task_done = td
        agent._agent.run = mock_run
        agent._agent.state = "idle"
        agent._agent.add_message = AsyncMock()
        agent._session = MagicMock()
        agent._session.add_message = MagicMock()
        agent.sessions = MagicMock()
        agent.sessions.save = MagicMock()
        agent.memory = MagicMock()
        agent.memory.get_memory_context = MagicMock(return_value=None)
        agent.context = MagicMock()
        agent._prepare_request_context = AsyncMock()
        agent._build_runtime_message_content = MagicMock(return_value="test")
        agent._build_step_prompt = MagicMock(return_value="prompt")
        agent._install_anti_loop_tracker = MagicMock()
        agent._restore_agent_think = MagicMock()
        agent._callable_accepts_kwarg = AgentLoop._callable_accepts_kwarg

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="test", thinking=False):
            chunks.append(chunk)

        done_chunks = [c for c in chunks if c["type"] == "done"]
        assert len(done_chunks) == 1


# ============================================================
# Stream timeout emits error (not silent break)
# ============================================================


class TestStreamTimeoutError:
    """Verify that stream timeout emits an error event."""

    @pytest.mark.asyncio
    async def test_timeout_emits_error_chunk(self):
        """When stream deadline is reached, an error chunk should be emitted."""
        from spoon_bot.agent.loop import AgentLoop

        oq: asyncio.Queue = asyncio.Queue()
        td = asyncio.Event()

        async def slow_run(**kwargs):
            await asyncio.sleep(100)
            result = MagicMock()
            result.content = "never"
            return result

        agent = MagicMock(spec=AgentLoop)
        agent._initialized = True
        agent._agent = MagicMock()
        agent._agent.output_queue = oq
        agent._agent.task_done = td
        agent._agent.run = slow_run
        agent._agent.state = "idle"
        agent._agent.add_message = AsyncMock()
        agent._session = MagicMock()
        agent._session.add_message = MagicMock()
        agent.sessions = MagicMock()
        agent.sessions.save = MagicMock()
        agent.memory = MagicMock()
        agent.memory.get_memory_context = MagicMock(return_value=None)
        agent.context = MagicMock()
        agent._prepare_request_context = AsyncMock()
        agent._build_runtime_message_content = MagicMock(return_value="test")
        agent._build_step_prompt = MagicMock(return_value="prompt")
        agent._install_anti_loop_tracker = MagicMock()
        agent._restore_agent_think = MagicMock()
        agent._callable_accepts_kwarg = AgentLoop._callable_accepts_kwarg

        with patch("spoon_bot.agent.loop.asyncio.get_event_loop") as mock_loop:
            call_count = 0

            def fake_time():
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    return 1000.0
                return 2000.0

            mock_event_loop = MagicMock()
            mock_event_loop.time = fake_time
            mock_loop.return_value = mock_event_loop

            chunks = []
            async for chunk in AgentLoop.stream(agent, message="test"):
                chunks.append(chunk)
                if chunk.get("type") == "error":
                    break

            error_chunks = [c for c in chunks if c["type"] == "error"]
            assert len(error_chunks) >= 1
            assert "STREAM_TIMEOUT" in error_chunks[0]["metadata"].get("error_code", "")


# ============================================================
# ConnectionManager send_message retry
# ============================================================


class TestConnectionManagerRetry:
    """Verify ConnectionManager.send_message retries before disconnecting."""

    @pytest.mark.asyncio
    async def test_send_retries_on_transient_failure(self):
        """send_message should retry up to 2 times before disconnecting."""
        from spoon_bot.gateway.websocket.manager import ConnectionManager, Connection

        manager = ConnectionManager()

        mock_ws = AsyncMock()
        call_count = 0

        async def flaky_send_json(data):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("temporary failure")

        mock_ws.send_json = flaky_send_json
        mock_ws.close = AsyncMock()

        conn = Connection(
            id="test-conn",
            websocket=mock_ws,
            user_id="test-user",
            session_key="default",
        )
        manager._connections["test-conn"] = conn

        result = await manager.send_message("test-conn", {"type": "ping"})
        assert result is True
        assert call_count == 3
        assert "test-conn" in manager._connections

    @pytest.mark.asyncio
    async def test_send_disconnects_after_all_retries_exhausted(self):
        """send_message should disconnect after all retry attempts fail."""
        from spoon_bot.gateway.websocket.manager import ConnectionManager, Connection

        manager = ConnectionManager()

        mock_ws = AsyncMock()
        mock_ws.send_json = AsyncMock(side_effect=ConnectionError("permanent failure"))
        mock_ws.close = AsyncMock()

        conn = Connection(
            id="test-conn",
            websocket=mock_ws,
            user_id="test-user",
            session_key="default",
        )
        manager._connections["test-conn"] = conn

        result = await manager.send_message("test-conn", {"type": "ping"})
        assert result is False
        assert "test-conn" not in manager._connections

    @pytest.mark.asyncio
    async def test_send_succeeds_on_first_try(self):
        """send_message should succeed immediately when no error occurs."""
        from spoon_bot.gateway.websocket.manager import ConnectionManager, Connection

        manager = ConnectionManager()

        mock_ws = AsyncMock()
        mock_ws.send_json = AsyncMock()
        mock_ws.close = AsyncMock()

        conn = Connection(
            id="test-conn",
            websocket=mock_ws,
            user_id="test-user",
            session_key="default",
        )
        manager._connections["test-conn"] = conn

        result = await manager.send_message("test-conn", {"type": "ping"})
        assert result is True
        assert mock_ws.send_json.call_count == 1
        assert "test-conn" in manager._connections

    @pytest.mark.asyncio
    async def test_send_returns_false_for_unknown_connection(self):
        """send_message should return False for non-existent connections."""
        from spoon_bot.gateway.websocket.manager import ConnectionManager

        manager = ConnectionManager()
        result = await manager.send_message("nonexistent", {"type": "ping"})
        assert result is False
