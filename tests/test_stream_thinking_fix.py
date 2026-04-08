"""
Tests for streaming thinking behavior and resilient long-running streams.

Covers:
  - thinking parameter forwarded to _run_and_signal() in stream mode
  - stream loop no longer enforces a local deadline
  - delayed chunks still arrive without tripping a local stream timeout
  - ConnectionManager.send_message retries before disconnecting
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_mock_stream_agent(run_impl) -> tuple[object, asyncio.Queue, asyncio.Event]:
    from spoon_bot.agent.loop import AgentLoop

    oq: asyncio.Queue = asyncio.Queue()
    td = asyncio.Event()

    agent = MagicMock(spec=AgentLoop)
    agent._initialized = True
    agent._agent = MagicMock()
    agent._agent.output_queue = oq
    agent._agent.task_done = td
    agent._agent.run = run_impl
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
    return agent, oq, td


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

        agent, _, _ = _make_mock_stream_agent(mock_run)

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

        agent, _, _ = _make_mock_stream_agent(mock_run)

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="test", thinking=False):
            chunks.append(chunk)

        assert len(run_kwargs_captured) == 1
        assert "thinking" not in run_kwargs_captured[0]


# ============================================================
# Stream wait behavior
# ============================================================


class TestStreamWaitBehavior:
    """Verify that stream() waits for completion instead of timing out locally."""

    @pytest.mark.asyncio
    async def test_stream_waits_for_delayed_chunk_without_timeout(self):
        """A delayed chunk should still be delivered without a synthetic timeout error."""
        from spoon_bot.agent.loop import AgentLoop

        async def mock_run(**kwargs):
            await asyncio.sleep(0.05)
            await oq.put({"type": "content", "delta": "late chunk", "metadata": {}})
            result = MagicMock()
            result.content = "late chunk"
            return result

        agent, oq, _ = _make_mock_stream_agent(mock_run)

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="test"):
            chunks.append(chunk)

        assert [c for c in chunks if c["type"] == "error"] == []
        content_chunks = [c for c in chunks if c["type"] == "content"]
        done_chunks = [c for c in chunks if c["type"] == "done"]
        assert len(content_chunks) == 1
        assert content_chunks[0]["delta"] == "late chunk"
        assert len(done_chunks) == 1
        assert done_chunks[0]["metadata"]["content"] == "late chunk"


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
