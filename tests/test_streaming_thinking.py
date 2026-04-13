"""
Tests for streaming and thinking support.

Tests the new streaming SSE, thinking mode, and backward compatibility
for the spoon-bot gateway API.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

# ============================================================
# Model tests (no spoon-core needed)
# ============================================================


class TestChatOptionsModel:
    """Test ChatOptions model with new fields."""

    def test_default_values(self):
        from spoon_bot.gateway.models.requests import ChatOptions

        opts = ChatOptions()
        assert opts.stream is False
        assert opts.thinking is False
        assert opts.max_iterations == 20
        assert opts.model is None

    def test_stream_true(self):
        from spoon_bot.gateway.models.requests import ChatOptions

        opts = ChatOptions(stream=True)
        assert opts.stream is True
        assert opts.thinking is False

    def test_thinking_true(self):
        from spoon_bot.gateway.models.requests import ChatOptions

        opts = ChatOptions(thinking=True)
        assert opts.thinking is True
        assert opts.stream is False

    def test_both_enabled(self):
        from spoon_bot.gateway.models.requests import ChatOptions

        opts = ChatOptions(stream=True, thinking=True)
        assert opts.stream is True
        assert opts.thinking is True

    def test_serialization_roundtrip(self):
        from spoon_bot.gateway.models.requests import ChatOptions

        opts = ChatOptions(stream=True, thinking=True, max_iterations=50)
        data = opts.model_dump()
        restored = ChatOptions(**data)
        assert restored.stream is True
        assert restored.thinking is True
        assert restored.max_iterations == 50

    def test_json_roundtrip(self):
        from spoon_bot.gateway.models.requests import ChatOptions

        opts = ChatOptions(stream=True, thinking=True)
        json_str = opts.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["stream"] is True
        assert parsed["thinking"] is True


class TestStreamChunkModel:
    """Test StreamChunk model."""

    def test_content_chunk(self):
        from spoon_bot.gateway.models.responses import StreamChunk

        chunk = StreamChunk(type="content", delta="Hello")
        assert chunk.type == "content"
        assert chunk.delta == "Hello"
        assert chunk.metadata == {}

    def test_thinking_chunk(self):
        from spoon_bot.gateway.models.responses import StreamChunk

        chunk = StreamChunk(type="thinking", delta="Let me think...")
        assert chunk.type == "thinking"
        assert chunk.delta == "Let me think..."

    def test_done_chunk(self):
        from spoon_bot.gateway.models.responses import StreamChunk

        chunk = StreamChunk(type="done", delta="", metadata={"content": "full response"})
        assert chunk.type == "done"
        assert chunk.delta == ""
        assert chunk.metadata["content"] == "full response"

    def test_error_chunk(self):
        from spoon_bot.gateway.models.responses import StreamChunk

        chunk = StreamChunk(type="error", delta="", metadata={"error": "something broke"})
        assert chunk.type == "error"
        assert chunk.metadata["error"] == "something broke"

    def test_tool_call_chunk(self):
        from spoon_bot.gateway.models.responses import StreamChunk

        chunk = StreamChunk(
            type="tool_call",
            delta="",
            metadata={"tool": "shell", "args": {"command": "ls"}},
        )
        assert chunk.type == "tool_call"

    def test_serialization(self):
        from spoon_bot.gateway.models.responses import StreamChunk

        chunk = StreamChunk(type="content", delta="hello world")
        data = chunk.model_dump()
        assert data == {"type": "content", "delta": "hello world", "metadata": {}}

    def test_json_serialization(self):
        from spoon_bot.gateway.models.responses import StreamChunk

        chunk = StreamChunk(type="content", delta="hello")
        json_str = chunk.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["type"] == "content"
        assert parsed["delta"] == "hello"

    def test_deserialization(self):
        from spoon_bot.gateway.models.responses import StreamChunk

        data = {"type": "thinking", "delta": "hmm", "metadata": {"key": "val"}}
        chunk = StreamChunk(**data)
        assert chunk.type == "thinking"
        assert chunk.delta == "hmm"
        assert chunk.metadata == {"key": "val"}

    def test_default_delta(self):
        from spoon_bot.gateway.models.responses import StreamChunk

        chunk = StreamChunk(type="done")
        assert chunk.delta == ""
        assert chunk.metadata == {}

    def test_empty_metadata_default(self):
        from spoon_bot.gateway.models.responses import StreamChunk

        chunk = StreamChunk(type="content", delta="x")
        assert chunk.metadata == {}


class TestChatResponseWithThinking:
    """Test ChatResponse includes thinking_content field."""

    def test_without_thinking(self):
        from spoon_bot.gateway.models.responses import ChatResponse

        resp = ChatResponse(response="Hello!")
        assert resp.response == "Hello!"
        assert resp.thinking_content is None

    def test_with_thinking(self):
        from spoon_bot.gateway.models.responses import ChatResponse

        resp = ChatResponse(
            response="Hello!",
            thinking_content="Let me think about this...",
        )
        assert resp.response == "Hello!"
        assert resp.thinking_content == "Let me think about this..."

    def test_thinking_is_optional(self):
        from spoon_bot.gateway.models.responses import ChatResponse

        # Should work without thinking_content in input
        data = {"response": "test", "tool_calls": []}
        resp = ChatResponse(**data)
        assert resp.thinking_content is None

    def test_serialization_with_thinking(self):
        from spoon_bot.gateway.models.responses import ChatResponse

        resp = ChatResponse(response="Hi", thinking_content="Thinking...")
        data = resp.model_dump()
        assert data["thinking_content"] == "Thinking..."
        assert data["response"] == "Hi"

    def test_backward_compat_no_thinking_field_in_old_data(self):
        """Old responses without thinking_content should still deserialize."""
        from spoon_bot.gateway.models.responses import ChatResponse

        # Simulating old response data that has no thinking_content key
        data = {"response": "old response", "tool_calls": [], "usage": None}
        resp = ChatResponse(**data)
        assert resp.response == "old response"
        assert resp.thinking_content is None


class TestChatRequestModel:
    """Test ChatRequest model backward compatibility."""

    def test_basic_request_no_options(self):
        from spoon_bot.gateway.models.requests import ChatRequest

        req = ChatRequest(message="hello")
        assert req.message == "hello"
        assert req.options is None
        assert req.media == []

    def test_request_with_stream_option(self):
        from spoon_bot.gateway.models.requests import ChatRequest

        req = ChatRequest(
            message="hello",
            options={"stream": True},
        )
        assert req.options.stream is True
        assert req.options.thinking is False

    def test_request_with_thinking_option(self):
        from spoon_bot.gateway.models.requests import ChatRequest

        req = ChatRequest(
            message="hello",
            options={"thinking": True},
        )
        assert req.options.thinking is True
        assert req.options.stream is False

    def test_request_with_both_options(self):
        from spoon_bot.gateway.models.requests import ChatRequest

        req = ChatRequest(
            message="hello",
            options={"stream": True, "thinking": True},
        )
        assert req.options.stream is True
        assert req.options.thinking is True

    def test_old_format_still_works(self):
        """Existing request format (no options) must still work."""
        from spoon_bot.gateway.models.requests import ChatRequest

        req = ChatRequest(message="hello", session_key="test")
        assert req.message == "hello"
        assert req.session_key == "test"
        assert req.options is None


# ============================================================
# SSE format tests
# ============================================================


class TestSSEFormat:
    """Test SSE event format generation."""

    def test_sse_chunk_format(self):
        """SSE chunks must follow 'data: {json}\\n\\n' format."""
        from spoon_bot.gateway.models.responses import StreamChunk

        chunk = StreamChunk(type="content", delta="Hello")
        json_str = chunk.model_dump_json()
        sse_line = f"data: {json_str}\n\n"

        assert sse_line.startswith("data: ")
        assert sse_line.endswith("\n\n")
        # Parse back the JSON
        payload = json.loads(sse_line[6:].strip())
        assert payload["type"] == "content"
        assert payload["delta"] == "Hello"

    def test_sse_done_format(self):
        """SSE done signal must be 'data: [DONE]\\n\\n'."""
        done_line = "data: [DONE]\n\n"
        assert done_line == "data: [DONE]\n\n"

    def test_sse_thinking_chunk(self):
        from spoon_bot.gateway.models.responses import StreamChunk

        chunk = StreamChunk(type="thinking", delta="Let me think...")
        sse = f"data: {chunk.model_dump_json()}\n\n"
        parsed = json.loads(sse[6:].strip())
        assert parsed["type"] == "thinking"

    def test_sse_stream_sequence_no_done_chunk(self):
        """A complete SSE stream should have content chunks + [DONE] only (no done chunk)."""
        from spoon_bot.gateway.models.responses import StreamChunk

        events = []

        # Simulate chunk sequence (only content chunks — done is filtered in SSE)
        for text in ["He", "llo", " wo", "rld"]:
            chunk = StreamChunk(type="content", delta=text)
            events.append(f"data: {chunk.model_dump_json()}\n\n")

        # Terminal DONE (the only completion signal)
        events.append("data: [DONE]\n\n")

        # All events must be properly formatted
        for event in events:
            assert event.startswith("data: ")
            assert event.endswith("\n\n")

        # Last event should be [DONE]
        assert events[-1] == "data: [DONE]\n\n"

        # No done-type JSON chunk should appear
        for event in events[:-1]:
            parsed = json.loads(event[6:].strip())
            assert parsed["type"] != "done"

    def test_sse_error_chunk_format(self):
        """SSE error chunks should have type=error with error in metadata."""
        from spoon_bot.gateway.models.responses import StreamChunk

        chunk = StreamChunk(type="error", delta="", metadata={"error": "connection lost"})
        sse = f"data: {chunk.model_dump_json()}\n\n"
        parsed = json.loads(sse[6:].strip())
        assert parsed["type"] == "error"
        assert parsed["metadata"]["error"] == "connection lost"


# ============================================================
# WebSocket protocol tests
# ============================================================


class TestWebSocketProtocol:
    """Test WebSocket protocol additions."""

    def test_stream_chunk_event_exists(self):
        from spoon_bot.gateway.websocket.protocol import ServerEvent

        assert ServerEvent.AGENT_STREAM_CHUNK.value == "agent.stream.chunk"

    def test_stream_done_event_exists(self):
        from spoon_bot.gateway.websocket.protocol import ServerEvent

        assert ServerEvent.AGENT_STREAM_DONE.value == "agent.stream.done"

    def test_existing_events_unchanged(self):
        from spoon_bot.gateway.websocket.protocol import ServerEvent

        # Verify existing events are still present
        assert ServerEvent.AGENT_THINKING.value == "agent.thinking"
        assert ServerEvent.AGENT_COMPLETE.value == "agent.complete"
        assert ServerEvent.AGENT_STREAMING.value == "agent.streaming"
        assert ServerEvent.AGENT_TOOL_CALL.value == "agent.tool_call"
        assert ServerEvent.AGENT_ERROR.value == "agent.error"


# ============================================================
# AgentLoop streaming tests (with mocking)
# ============================================================


@pytest.mark.requires_spoon_core
class TestAgentLoopStream:
    """Test AgentLoop.stream() method."""

    @pytest.fixture
    def mock_agent_loop(self):
        """Create a mock AgentLoop with stream support."""
        loop = MagicMock()
        loop._initialized = True
        loop._session = MagicMock()
        loop._session.add_message = MagicMock()
        loop.sessions = MagicMock()
        loop.sessions.save = MagicMock()
        return loop

    @pytest.mark.asyncio
    async def test_stream_yields_typed_dicts(self):
        """stream() should yield dicts with type, delta, metadata keys."""
        from spoon_bot.agent.loop import AgentLoop

        # Mock the internal agent
        mock_chunk_1 = MagicMock()
        mock_chunk_1.type = "content"  # Won't match 'thinking' check
        mock_chunk_1.content = "Hello"
        mock_chunk_1.metadata = None  # Not a dict with 'type'

        mock_chunk_2 = MagicMock()
        mock_chunk_2.type = "content"
        mock_chunk_2.content = " World"
        mock_chunk_2.metadata = None

        async def mock_run(**kwargs):
            for c in [mock_chunk_1, mock_chunk_2]:
                await agent._agent.output_queue.put(c)

        agent = MagicMock(spec=AgentLoop)
        agent._initialized = True
        agent._agent = MagicMock()
        agent._agent.output_queue = asyncio.Queue()
        agent._agent.task_done = asyncio.Event()
        agent._agent.run = mock_run
        agent._agent.add_message = AsyncMock()
        agent._agent.state = "idle"
        agent._session = MagicMock()
        agent._session.add_message = MagicMock()
        agent.sessions = MagicMock()
        agent.sessions.save = MagicMock()
        agent.memory = MagicMock()
        agent.memory.get_memory_context = MagicMock(return_value=None)
        agent.context = MagicMock()
        agent._prepare_request_context = AsyncMock()
        agent._build_runtime_message_content = MagicMock(side_effect=lambda *args, **kwargs: args[1])
        agent._build_step_prompt = MagicMock(return_value="prompt")
        agent._install_anti_loop_tracker = MagicMock()
        agent._restore_agent_think = MagicMock()
        agent._callable_accepts_kwarg = MagicMock(return_value=True)
        agent._reset_reasoning_capture = MagicMock()
        agent._drain_reasoning_chunks = MagicMock(return_value=[])

        # Call the real stream method bound to our mock
        chunks = []
        async for chunk in AgentLoop.stream(agent, message="test"):
            chunks.append(chunk)

        # Should have content chunks + done
        content_chunks = [c for c in chunks if c["type"] == "content"]
        done_chunks = [c for c in chunks if c["type"] == "done"]

        assert len(content_chunks) == 2
        assert content_chunks[0]["delta"] == "Hello"
        assert content_chunks[1]["delta"] == " World"
        assert len(done_chunks) == 1
        assert done_chunks[0]["metadata"]["content"] == "Hello World"

    @pytest.mark.asyncio
    async def test_stream_handles_string_chunks(self):
        """stream() should handle plain string chunks."""
        from spoon_bot.agent.loop import AgentLoop

        async def mock_run(**kwargs):
            await agent._agent.output_queue.put("Hello")
            await agent._agent.output_queue.put(" World")

        agent = MagicMock(spec=AgentLoop)
        agent._initialized = True
        agent._agent = MagicMock()
        agent._agent.output_queue = asyncio.Queue()
        agent._agent.task_done = asyncio.Event()
        agent._agent.run = mock_run
        agent._agent.add_message = AsyncMock()
        agent._agent.state = "idle"
        agent._session = MagicMock()
        agent._session.add_message = MagicMock()
        agent.sessions = MagicMock()
        agent.sessions.save = MagicMock()
        agent.memory = MagicMock()
        agent.memory.get_memory_context = MagicMock(return_value=None)
        agent.context = MagicMock()
        agent._prepare_request_context = AsyncMock()
        agent._build_runtime_message_content = MagicMock(side_effect=lambda *args, **kwargs: args[1])
        agent._build_step_prompt = MagicMock(return_value="prompt")
        agent._install_anti_loop_tracker = MagicMock()
        agent._restore_agent_think = MagicMock()
        agent._callable_accepts_kwarg = MagicMock(return_value=True)
        agent._reset_reasoning_capture = MagicMock()
        agent._drain_reasoning_chunks = MagicMock(return_value=[])

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="test"):
            chunks.append(chunk)

        content_chunks = [c for c in chunks if c["type"] == "content"]
        assert len(content_chunks) == 2
        assert content_chunks[0]["delta"] == "Hello"
        assert content_chunks[1]["delta"] == " World"

    @pytest.mark.asyncio
    async def test_stream_handles_thinking_chunks(self):
        """stream() should identify and pass through thinking chunks."""
        from spoon_bot.agent.loop import AgentLoop

        thinking_chunk = MagicMock()
        thinking_chunk.type = "thinking"
        thinking_chunk.content = "Let me think..."

        content_chunk = MagicMock()
        content_chunk.type = "content"
        content_chunk.content = "Result"
        content_chunk.metadata = None

        async def mock_run(**kwargs):
            await agent._agent.output_queue.put(thinking_chunk)
            await agent._agent.output_queue.put(content_chunk)

        agent = MagicMock(spec=AgentLoop)
        agent._initialized = True
        agent._agent = MagicMock()
        agent._agent.output_queue = asyncio.Queue()
        agent._agent.task_done = asyncio.Event()
        agent._agent.run = mock_run
        agent._agent.add_message = AsyncMock()
        agent._agent.state = "idle"
        agent._session = MagicMock()
        agent._session.add_message = MagicMock()
        agent.sessions = MagicMock()
        agent.sessions.save = MagicMock()
        agent.memory = MagicMock()
        agent.memory.get_memory_context = MagicMock(return_value=None)
        agent.context = MagicMock()
        agent._prepare_request_context = AsyncMock()
        agent._build_runtime_message_content = MagicMock(side_effect=lambda *args, **kwargs: args[1])
        agent._build_step_prompt = MagicMock(return_value="prompt")
        agent._install_anti_loop_tracker = MagicMock()
        agent._restore_agent_think = MagicMock()
        agent._callable_accepts_kwarg = MagicMock(return_value=True)
        agent._reset_reasoning_capture = MagicMock()
        agent._drain_reasoning_chunks = MagicMock(return_value=[])

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="test", thinking=True):
            chunks.append(chunk)

        thinking_chunks = [c for c in chunks if c["type"] == "thinking"]
        content_chunks = [c for c in chunks if c["type"] == "content"]

        assert len(thinking_chunks) == 1
        assert thinking_chunks[0]["delta"] == "Let me think..."
        assert len(content_chunks) == 1
        assert content_chunks[0]["delta"] == "Result"

    @pytest.mark.asyncio
    async def test_stream_downgrades_to_content_when_no_tool_call_follows(self):
        """stream() should keep normal content when tool calls never follow."""
        from spoon_bot.agent.loop import AgentLoop

        async def mock_run(**kwargs):
            agent._capture_reasoning_text("Reasoning from tracked think")
            await agent._agent.output_queue.put({"content": "Answer"})

        agent = MagicMock(spec=AgentLoop)
        agent._initialized = True
        agent._agent = MagicMock()
        agent._agent.output_queue = asyncio.Queue()
        agent._agent.task_done = asyncio.Event()
        agent._agent.run = mock_run
        agent._agent.add_message = AsyncMock()
        agent._agent.state = "idle"
        agent._session = MagicMock()
        agent._session.add_message = MagicMock()
        agent.sessions = MagicMock()
        agent.sessions.save = MagicMock()
        agent.memory = MagicMock()
        agent.memory.get_memory_context = MagicMock(return_value=None)
        agent.context = MagicMock()
        agent._prepare_request_context = AsyncMock()
        agent._build_runtime_message_content = MagicMock(side_effect=lambda *args, **kwargs: args[1])
        agent._build_step_prompt = MagicMock(return_value="prompt")
        agent._install_anti_loop_tracker = MagicMock()
        agent._restore_agent_think = MagicMock()
        agent._callable_accepts_kwarg = MagicMock(return_value=True)
        agent._drain_reasoning_chunks = AgentLoop._drain_reasoning_chunks.__get__(agent, AgentLoop)
        agent._reset_reasoning_capture = AgentLoop._reset_reasoning_capture.__get__(agent, AgentLoop)
        agent._capture_reasoning_text = AgentLoop._capture_reasoning_text.__get__(agent, AgentLoop)
        agent._reset_reasoning_capture()

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="test", thinking=True):
            chunks.append(chunk)

        thinking_chunks = [c for c in chunks if c["type"] == "thinking"]
        content_chunks = [c for c in chunks if c["type"] == "content"]

        assert len(thinking_chunks) == 0
        assert len(content_chunks) == 1
        assert content_chunks[0]["delta"] == "Answer"
        done_chunks = [c for c in chunks if c["type"] == "done"]
        assert len(done_chunks) == 1
        assert done_chunks[0]["metadata"]["content"] == "Answer"

    @pytest.mark.asyncio
    async def test_stream_late_tracked_reasoning_still_preserves_content(self):
        """Late tracked reasoning should not strip normal content output."""
        from spoon_bot.agent.loop import AgentLoop

        async def mock_run(**kwargs):
            await agent._agent.output_queue.put({"content": "Answer"})
            await asyncio.sleep(0)
            agent._capture_reasoning_text("Late tracked reasoning")

        agent = MagicMock(spec=AgentLoop)
        agent._initialized = True
        agent._agent = MagicMock()
        agent._agent.output_queue = asyncio.Queue()
        agent._agent.task_done = asyncio.Event()
        agent._agent.run = mock_run
        agent._agent.add_message = AsyncMock()
        agent._agent.state = "idle"
        agent._session = MagicMock()
        agent._session.add_message = MagicMock()
        agent.sessions = MagicMock()
        agent.sessions.save = MagicMock()
        agent.memory = MagicMock()
        agent.memory.get_memory_context = MagicMock(return_value=None)
        agent.context = MagicMock()
        agent._prepare_request_context = AsyncMock()
        agent._build_runtime_message_content = MagicMock(side_effect=lambda *args, **kwargs: args[1])
        agent._build_step_prompt = MagicMock(return_value="prompt")
        agent._install_anti_loop_tracker = MagicMock()
        agent._restore_agent_think = MagicMock()
        agent._callable_accepts_kwarg = MagicMock(return_value=True)
        agent._drain_reasoning_chunks = AgentLoop._drain_reasoning_chunks.__get__(agent, AgentLoop)
        agent._reset_reasoning_capture = AgentLoop._reset_reasoning_capture.__get__(agent, AgentLoop)
        agent._capture_reasoning_text = AgentLoop._capture_reasoning_text.__get__(agent, AgentLoop)
        agent._reset_reasoning_capture()

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="test", thinking=True):
            chunks.append(chunk)

        emitted = [c for c in chunks if c["type"] in {"thinking", "content"}]
        assert len(emitted) == 1
        assert emitted[0]["type"] == "content"
        assert emitted[0]["delta"] == "Answer"
        done_chunks = [c for c in chunks if c["type"] == "done"]
        assert len(done_chunks) == 1
        assert done_chunks[0]["metadata"]["content"] == "Answer"

    @pytest.mark.asyncio
    async def test_stream_keeps_leading_content_streaming_when_tool_call_follows(self):
        """Plain content should stay streamable even if a tool call appears later."""
        from spoon_bot.agent.loop import AgentLoop

        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function = MagicMock()
        tool_call.function.name = "shell"
        tool_call.function.arguments = '{"command":"pwd"}'

        async def mock_run(**kwargs):
            await agent._agent.output_queue.put({"content": "Part A. "})
            await agent._agent.output_queue.put({"tool_calls": [tool_call]})
            await agent._agent.output_queue.put({"content": "Part B."})

        agent = MagicMock(spec=AgentLoop)
        agent._initialized = True
        agent._agent = MagicMock()
        agent._agent.output_queue = asyncio.Queue()
        agent._agent.task_done = asyncio.Event()
        agent._agent.run = mock_run
        agent._agent.add_message = AsyncMock()
        agent._agent.state = "idle"
        agent._session = MagicMock()
        agent._session.add_message = MagicMock()
        agent.sessions = MagicMock()
        agent.sessions.save = MagicMock()
        agent.memory = MagicMock()
        agent.memory.get_memory_context = MagicMock(return_value=None)
        agent.context = MagicMock()
        agent._prepare_request_context = AsyncMock()
        agent._build_runtime_message_content = MagicMock(return_value="placeholder")
        agent._build_step_prompt = MagicMock(return_value="prompt")
        agent._install_anti_loop_tracker = MagicMock()
        agent._restore_agent_think = MagicMock()
        agent._callable_accepts_kwarg = MagicMock(return_value=True)
        agent._reset_reasoning_capture = MagicMock()
        agent._drain_reasoning_chunks = MagicMock(return_value=[])

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="placeholder", thinking=True):
            chunks.append(chunk)

        emitted = [c for c in chunks if c["type"] in {"content", "tool_call"}]
        assert [c["type"] for c in emitted] == ["content", "tool_call", "content"]
        assert emitted[0]["delta"] == "Part A. "
        assert emitted[1]["metadata"]["name"] == "shell"
        assert emitted[2]["delta"] == "Part B."
        done_chunks = [c for c in chunks if c["type"] == "done"]
        assert done_chunks[0]["metadata"]["content"] == "Part A. Part B."

    @pytest.mark.asyncio
    async def test_stream_falls_back_to_run_result_after_pre_tool_content_without_final_chunk(self):
        """A tool preamble should not suppress the final run() answer fallback."""
        from spoon_bot.agent.loop import AgentLoop

        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function = MagicMock()
        tool_call.function.name = "shell"
        tool_call.function.arguments = '{"command":"pwd"}'

        async def mock_run(**kwargs):
            await agent._agent.output_queue.put({"content": "Need tool first. "})
            await agent._agent.output_queue.put({"tool_calls": [tool_call]})
            result = MagicMock()
            result.content = "Final answer."
            return result

        agent = MagicMock(spec=AgentLoop)
        agent._initialized = True
        agent._agent = MagicMock()
        agent._agent.output_queue = asyncio.Queue()
        agent._agent.task_done = asyncio.Event()
        agent._agent.run = mock_run
        agent._agent.add_message = AsyncMock()
        agent._agent.state = "idle"
        agent._session = MagicMock()
        agent._session.add_message = MagicMock()
        agent.sessions = MagicMock()
        agent.sessions.save = MagicMock()
        agent.memory = MagicMock()
        agent.memory.get_memory_context = MagicMock(return_value=None)
        agent.context = MagicMock()
        agent._prepare_request_context = AsyncMock()
        agent._build_runtime_message_content = MagicMock(return_value="placeholder")
        agent._build_step_prompt = MagicMock(return_value="prompt")
        agent._install_anti_loop_tracker = MagicMock()
        agent._restore_agent_think = MagicMock()
        agent._callable_accepts_kwarg = MagicMock(return_value=True)
        agent._reset_reasoning_capture = MagicMock()
        agent._drain_reasoning_chunks = MagicMock(return_value=[])
        agent._normalize_comparable_text = AgentLoop._normalize_comparable_text

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="placeholder", thinking=True):
            chunks.append(chunk)

        emitted = [c for c in chunks if c["type"] in {"content", "tool_call"}]
        assert [c["type"] for c in emitted] == ["content", "tool_call", "content"]
        assert emitted[0]["delta"] == "Need tool first. "
        assert emitted[2]["delta"] == "Final answer."
        assert emitted[2]["metadata"]["fallback"] == "run_result_after_tool_preamble"
        done_chunks = [c for c in chunks if c["type"] == "done"]
        assert done_chunks[0]["metadata"]["content"] == "Need tool first. Final answer."

    @pytest.mark.asyncio
    async def test_stream_passes_through_structured_tool_result_chunks(self):
        """Explicit tool_result chunks should be preserved for downstream UIs."""
        from spoon_bot.agent.loop import AgentLoop

        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function = MagicMock()
        tool_call.function.name = "shell"
        tool_call.function.arguments = '{"command":"pwd"}'

        async def mock_run(**kwargs):
            await agent._agent.output_queue.put({"tool_calls": [tool_call]})
            await agent._agent.output_queue.put(
                {
                    "type": "tool_result",
                    "metadata": {
                        "id": "call_1",
                        "name": "shell",
                    },
                    "result": {
                        "stdout": "/workspace",
                    },
                }
            )
            await agent._agent.output_queue.put({"content": "Done."})

        agent = MagicMock(spec=AgentLoop)
        agent._initialized = True
        agent._agent = MagicMock()
        agent._agent.output_queue = asyncio.Queue()
        agent._agent.task_done = asyncio.Event()
        agent._agent.run = mock_run
        agent._agent.add_message = AsyncMock()
        agent._agent.memory = MagicMock()
        agent._agent.memory.messages = []
        agent._agent.state = "idle"
        agent._session = MagicMock()
        agent._session.add_message = MagicMock()
        agent.sessions = MagicMock()
        agent.sessions.save = MagicMock()
        agent.memory = MagicMock()
        agent.memory.get_memory_context = MagicMock(return_value=None)
        agent.context = MagicMock()
        agent._prepare_request_context = AsyncMock()
        agent._build_runtime_message_content = MagicMock(return_value="placeholder")
        agent._build_step_prompt = MagicMock(return_value="prompt")
        agent._install_anti_loop_tracker = MagicMock()
        agent._restore_agent_think = MagicMock()
        agent._callable_accepts_kwarg = MagicMock(return_value=True)
        agent._reset_reasoning_capture = MagicMock()
        agent._drain_reasoning_chunks = MagicMock(return_value=[])

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="placeholder"):
            chunks.append(chunk)

        emitted = [c for c in chunks if c["type"] in {"tool_call", "tool_result", "content"}]
        assert [c["type"] for c in emitted] == ["tool_call", "tool_result", "content"]
        assert emitted[1]["metadata"]["id"] == "call_1"
        assert emitted[1]["metadata"]["name"] == "shell"
        assert emitted[1]["metadata"]["result"] == '{"stdout": "/workspace"}'

    @pytest.mark.asyncio
    async def test_stream_backfills_tool_result_from_runtime_memory(self):
        """Tool results stored only in runtime memory should still be emitted."""
        from spoon_bot.agent.loop import AgentLoop

        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function = MagicMock()
        tool_call.function.name = "shell"
        tool_call.function.arguments = '{"command":"pwd"}'

        async def mock_run(**kwargs):
            await agent._agent.output_queue.put({"tool_calls": [tool_call]})
            agent._agent.memory.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "shell",
                    "content": "/workspace",
                }
            )
            await agent._agent.output_queue.put({"content": "Done."})

        agent = MagicMock(spec=AgentLoop)
        agent._initialized = True
        agent._agent = MagicMock()
        agent._agent.output_queue = asyncio.Queue()
        agent._agent.task_done = asyncio.Event()
        agent._agent.run = mock_run
        agent._agent.add_message = AsyncMock()
        agent._agent.memory = MagicMock()
        agent._agent.memory.messages = []
        agent._agent.state = "idle"
        agent._session = MagicMock()
        agent._session.add_message = MagicMock()
        agent.sessions = MagicMock()
        agent.sessions.save = MagicMock()
        agent.memory = MagicMock()
        agent.memory.get_memory_context = MagicMock(return_value=None)
        agent.context = MagicMock()
        agent._prepare_request_context = AsyncMock()
        agent._build_runtime_message_content = MagicMock(return_value="placeholder")
        agent._build_step_prompt = MagicMock(return_value="prompt")
        agent._install_anti_loop_tracker = MagicMock()
        agent._restore_agent_think = MagicMock()
        agent._callable_accepts_kwarg = MagicMock(return_value=True)
        agent._reset_reasoning_capture = MagicMock()
        agent._drain_reasoning_chunks = MagicMock(return_value=[])

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="placeholder"):
            chunks.append(chunk)

        emitted = [c for c in chunks if c["type"] in {"tool_call", "tool_result", "content"}]
        assert [c["type"] for c in emitted] == ["tool_call", "tool_result", "content"]
        assert emitted[1]["metadata"]["id"] == "call_1"
        assert emitted[1]["metadata"]["name"] == "shell"
        assert emitted[1]["metadata"]["result"] == "/workspace"

    @pytest.mark.asyncio
    async def test_stream_uses_explicit_thinking_chunk_without_prompt_heuristic(self):
        """Explicit thinking chunks from core should drive pre-tool thinking display."""
        from spoon_bot.agent.loop import AgentLoop

        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function = MagicMock()
        tool_call.function.name = "shell"
        tool_call.function.arguments = '{"command":"pwd"}'

        async def mock_run(**kwargs):
            await agent._agent.output_queue.put(
                {
                    "type": "thinking",
                    "delta": "First I will inspect the workspace.",
                    "content": "First I will inspect the workspace.",
                    "metadata": {
                        "phase": "think",
                        "source": "toolcall_agent",
                    },
                }
            )
            await agent._agent.output_queue.put({"tool_calls": [tool_call]})
            await agent._agent.output_queue.put({"content": "Done."})

        agent = MagicMock(spec=AgentLoop)
        agent._initialized = True
        agent._agent = MagicMock()
        agent._agent.output_queue = asyncio.Queue()
        agent._agent.task_done = asyncio.Event()
        agent._agent.run = mock_run
        agent._agent.add_message = AsyncMock()
        agent._agent.state = "idle"
        agent._session = MagicMock()
        agent._session.add_message = MagicMock()
        agent.sessions = MagicMock()
        agent.sessions.save = MagicMock()
        agent.memory = MagicMock()
        agent.memory.get_memory_context = MagicMock(return_value=None)
        agent.context = MagicMock()
        agent._prepare_request_context = AsyncMock()
        agent._build_runtime_message_content = MagicMock(return_value="just answer directly")
        agent._build_step_prompt = MagicMock(return_value="prompt")
        agent._install_anti_loop_tracker = MagicMock()
        agent._restore_agent_think = MagicMock()
        agent._callable_accepts_kwarg = MagicMock(return_value=True)
        agent._reset_reasoning_capture = MagicMock()
        agent._drain_reasoning_chunks = MagicMock(return_value=[])

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="placeholder", thinking=True):
            chunks.append(chunk)

        emitted = [c for c in chunks if c["type"] in {"thinking", "tool_call", "content"}]
        assert [c["type"] for c in emitted] == ["thinking", "tool_call", "content"]
        assert emitted[0]["type"] == "thinking"
        assert emitted[0]["delta"] == "First I will inspect the workspace."
        assert emitted[0]["metadata"]["source"] == "toolcall_agent"
        assert emitted[1]["type"] == "tool_call"
        assert emitted[2]["type"] == "content"
        assert emitted[2]["delta"] == "Done."

    @pytest.mark.asyncio
    async def test_stream_explicit_thinking_without_tool_call_falls_back_once(self):
        """Explicit thinking chunks should not be replayed as content before done fallback."""
        from spoon_bot.agent.loop import AgentLoop

        async def mock_run(**kwargs):
            await agent._agent.output_queue.put(
                {
                    "type": "thinking",
                    "delta": "I will think before answering.",
                    "content": "I will think before answering.",
                    "metadata": {
                        "phase": "think",
                        "source": "toolcall_agent",
                    },
                }
            )
            return "Final answer."

        agent = MagicMock(spec=AgentLoop)
        agent._initialized = True
        agent._agent = MagicMock()
        agent._agent.output_queue = asyncio.Queue()
        agent._agent.task_done = asyncio.Event()
        agent._agent.run = mock_run
        agent._agent.add_message = AsyncMock()
        agent._agent.state = "idle"
        agent._session = MagicMock()
        agent._session.add_message = MagicMock()
        agent.sessions = MagicMock()
        agent.sessions.save = MagicMock()
        agent.memory = MagicMock()
        agent.memory.get_memory_context = MagicMock(return_value=None)
        agent.context = MagicMock()
        agent._prepare_request_context = AsyncMock()
        agent._build_runtime_message_content = MagicMock(return_value="just answer directly")
        agent._build_step_prompt = MagicMock(return_value="prompt")
        agent._install_anti_loop_tracker = MagicMock()
        agent._restore_agent_think = MagicMock()
        agent._callable_accepts_kwarg = MagicMock(return_value=True)
        agent._reset_reasoning_capture = MagicMock()
        agent._drain_reasoning_chunks = MagicMock(return_value=[])

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="placeholder", thinking=True):
            chunks.append(chunk)

        thinking_chunks = [c for c in chunks if c["type"] == "thinking"]
        content_chunks = [c for c in chunks if c["type"] == "content"]
        done_chunks = [c for c in chunks if c["type"] == "done"]

        assert [c["delta"] for c in thinking_chunks] == ["I will think before answering."]
        assert [c["delta"] for c in content_chunks] == ["Final answer."]
        assert done_chunks[0]["metadata"]["content"] == "Final answer."

    @pytest.mark.asyncio
    async def test_stream_handles_dict_thinking_chunks(self):
        """Structured dict thinking events should pass through unchanged."""
        from spoon_bot.agent.loop import AgentLoop

        async def mock_run(**kwargs):
            await agent._agent.output_queue.put(
                {
                    "type": "thinking",
                    "delta": "Inspecting candidate commands.",
                    "content": "Inspecting candidate commands.",
                    "metadata": {"phase": "think", "source": "provider"},
                }
            )
            await agent._agent.output_queue.put({"content": "Final answer."})

        agent = MagicMock(spec=AgentLoop)
        agent._initialized = True
        agent._agent = MagicMock()
        agent._agent.output_queue = asyncio.Queue()
        agent._agent.task_done = asyncio.Event()
        agent._agent.run = mock_run
        agent._agent.add_message = AsyncMock()
        agent._agent.state = "idle"
        agent._session = MagicMock()
        agent._session.add_message = MagicMock()
        agent.sessions = MagicMock()
        agent.sessions.save = MagicMock()
        agent.memory = MagicMock()
        agent.memory.get_memory_context = MagicMock(return_value=None)
        agent.context = MagicMock()
        agent._prepare_request_context = AsyncMock()
        agent._build_runtime_message_content = MagicMock(side_effect=lambda *args, **kwargs: args[1])
        agent._build_step_prompt = MagicMock(return_value="prompt")
        agent._install_anti_loop_tracker = MagicMock()
        agent._restore_agent_think = MagicMock()
        agent._callable_accepts_kwarg = MagicMock(return_value=True)
        agent._reset_reasoning_capture = MagicMock()
        agent._drain_reasoning_chunks = MagicMock(return_value=[])

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="test", thinking=True):
            chunks.append(chunk)

        emitted = [c for c in chunks if c["type"] in {"thinking", "content"}]
        assert emitted[0] == {
            "type": "thinking",
            "delta": "Inspecting candidate commands.",
            "metadata": {"phase": "think", "source": "provider"},
        }
        assert emitted[1]["type"] == "content"
        assert emitted[1]["delta"] == "Final answer."

    @pytest.mark.asyncio
    async def test_stream_without_tool_call_keeps_plain_content(self):
        """thinking=true should not force all plain answers into thinking."""
        from spoon_bot.agent.loop import AgentLoop

        async def mock_run(**kwargs):
            await agent._agent.output_queue.put({"content": "直接给答案"})

        agent = MagicMock(spec=AgentLoop)
        agent._initialized = True
        agent._agent = MagicMock()
        agent._agent.output_queue = asyncio.Queue()
        agent._agent.task_done = asyncio.Event()
        agent._agent.run = mock_run
        agent._agent.add_message = AsyncMock()
        agent._agent.state = "idle"
        agent._session = MagicMock()
        agent._session.add_message = MagicMock()
        agent.sessions = MagicMock()
        agent.sessions.save = MagicMock()
        agent.memory = MagicMock()
        agent.memory.get_memory_context = MagicMock(return_value=None)
        agent.context = MagicMock()
        agent._prepare_request_context = AsyncMock()
        agent._build_runtime_message_content = MagicMock(side_effect=lambda *args, **kwargs: args[1])
        agent._build_step_prompt = MagicMock(return_value="prompt")
        agent._install_anti_loop_tracker = MagicMock()
        agent._restore_agent_think = MagicMock()
        agent._callable_accepts_kwarg = MagicMock(return_value=True)
        agent._reset_reasoning_capture = MagicMock()
        agent._drain_reasoning_chunks = MagicMock(return_value=[])

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="test", thinking=True):
            chunks.append(chunk)

        thinking_chunks = [c for c in chunks if c["type"] == "thinking"]
        content_chunks = [c for c in chunks if c["type"] == "content"]
        assert thinking_chunks == []
        assert len(content_chunks) == 1
        assert content_chunks[0]["delta"] == "直接给答案"
        done_chunks = [c for c in chunks if c["type"] == "done"]
        assert len(done_chunks) == 1
        assert done_chunks[0]["metadata"]["content"] == "直接给答案"

    @pytest.mark.asyncio
    async def test_stream_without_tool_call_emits_first_content_before_run_finishes(self):
        """thinking=true should not buffer normal content until the very end."""
        from spoon_bot.agent.loop import AgentLoop

        release_run = asyncio.Event()

        async def mock_run(**kwargs):
            await agent._agent.output_queue.put({"content": "chunk-1 "})
            await asyncio.wait_for(release_run.wait(), timeout=0.2)
            await agent._agent.output_queue.put({"content": "chunk-2"})

        agent = MagicMock(spec=AgentLoop)
        agent._initialized = True
        agent._agent = MagicMock()
        agent._agent.output_queue = asyncio.Queue()
        agent._agent.task_done = asyncio.Event()
        agent._agent.run = mock_run
        agent._agent.add_message = AsyncMock()
        agent._agent.state = "idle"
        agent._session = MagicMock()
        agent._session.add_message = MagicMock()
        agent.sessions = MagicMock()
        agent.sessions.save = MagicMock()
        agent.memory = MagicMock()
        agent.memory.get_memory_context = MagicMock(return_value=None)
        agent.context = MagicMock()
        agent._prepare_request_context = AsyncMock()
        agent._build_runtime_message_content = MagicMock(side_effect=lambda *args, **kwargs: args[1])
        agent._build_step_prompt = MagicMock(return_value="prompt")
        agent._install_anti_loop_tracker = MagicMock()
        agent._restore_agent_think = MagicMock()
        agent._callable_accepts_kwarg = MagicMock(return_value=True)
        agent._reset_reasoning_capture = MagicMock()
        agent._drain_reasoning_chunks = MagicMock(return_value=[])

        stream_iter = AgentLoop.stream(agent, message="test", thinking=True)

        first_chunk = await asyncio.wait_for(anext(stream_iter), timeout=0.2)
        assert first_chunk == {"type": "content", "delta": "chunk-1 ", "metadata": {}}

        release_run.set()
        remaining_chunks = [chunk async for chunk in stream_iter]
        content_chunks = [first_chunk, *[c for c in remaining_chunks if c["type"] == "content"]]
        done_chunks = [c for c in remaining_chunks if c["type"] == "done"]

        assert [c["delta"] for c in content_chunks] == ["chunk-1 ", "chunk-2"]
        assert len(done_chunks) == 1
        assert done_chunks[0]["metadata"]["content"] == "chunk-1 chunk-2"

    @pytest.mark.asyncio
    async def test_stream_error_handling(self):
        """stream() should catch errors and emit done with error metadata."""
        from spoon_bot.agent.loop import AgentLoop

        async def mock_run(**kwargs):
            await agent._agent.output_queue.put("partial")
            raise RuntimeError("Connection lost")

        agent = MagicMock(spec=AgentLoop)
        agent._initialized = True
        agent._agent = MagicMock()
        agent._agent.output_queue = asyncio.Queue()
        agent._agent.task_done = asyncio.Event()
        agent._agent.run = mock_run
        agent._agent.add_message = AsyncMock()
        agent._agent.state = "idle"
        agent._session = MagicMock()
        agent._session.add_message = MagicMock()
        agent.sessions = MagicMock()
        agent.sessions.save = MagicMock()
        agent.memory = MagicMock()
        agent.memory.get_memory_context = MagicMock(return_value=None)
        agent.context = MagicMock()
        agent._prepare_request_context = AsyncMock()
        agent._build_runtime_message_content = MagicMock(side_effect=lambda *args, **kwargs: args[1])
        agent._build_step_prompt = MagicMock(return_value="prompt")
        agent._install_anti_loop_tracker = MagicMock()
        agent._restore_agent_think = MagicMock()
        agent._callable_accepts_kwarg = MagicMock(return_value=True)
        agent._reset_reasoning_capture = MagicMock()
        agent._drain_reasoning_chunks = MagicMock(return_value=[])

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="test"):
            chunks.append(chunk)

        # Should get the partial content + error done
        assert chunks[0]["type"] == "content"
        assert chunks[0]["delta"] == "partial"
        assert chunks[-2]["type"] == "error"
        assert "Connection lost" in chunks[-2]["metadata"]["error"]
        assert chunks[-1]["type"] == "done"
        assert chunks[-1]["metadata"]["content"] == "partial"

    @pytest.mark.asyncio
    async def test_stream_skips_session_save_on_error(self):
        """stream() should NOT save session when full_content is empty (error before any content)."""
        from spoon_bot.agent.loop import AgentLoop

        async def mock_stream(message, **kwargs):
            raise RuntimeError("Immediate failure")
            yield "never"  # pragma: no cover

        agent = MagicMock(spec=AgentLoop)
        agent._initialized = True
        agent._agent = MagicMock()
        agent._agent.stream = mock_stream
        agent._session = MagicMock()
        agent._session.add_message = MagicMock()
        agent.sessions = MagicMock()
        agent.sessions.save = MagicMock()
        agent.memory = MagicMock()

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="test"):
            chunks.append(chunk)

        # Should get error done chunk
        assert chunks[-1]["type"] == "done"
        assert "error" in chunks[-1]["metadata"]

        # Session should NOT be saved since full_content is empty
        agent.sessions.save.assert_not_called()
        agent._session.add_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_stream_saves_session_on_success(self):
        """stream() should save session after successful completion."""
        from spoon_bot.agent.loop import AgentLoop

        async def mock_run(**kwargs):
            await agent._agent.output_queue.put("hello")

        agent = MagicMock(spec=AgentLoop)
        agent._initialized = True
        agent._agent = MagicMock()
        agent._agent.output_queue = asyncio.Queue()
        agent._agent.task_done = asyncio.Event()
        agent._agent.run = mock_run
        agent._agent.add_message = AsyncMock()
        agent._agent.state = "idle"
        agent._session = MagicMock()
        agent._session.add_message = MagicMock()
        agent.sessions = MagicMock()
        agent.sessions.save = MagicMock()
        agent.memory = MagicMock()
        agent.memory.get_memory_context = MagicMock(return_value=None)
        agent.context = MagicMock()
        agent._prepare_request_context = AsyncMock()
        agent._build_runtime_message_content = MagicMock(side_effect=lambda *args, **kwargs: args[1])
        agent._build_step_prompt = MagicMock(return_value="prompt")
        agent._install_anti_loop_tracker = MagicMock()
        agent._restore_agent_think = MagicMock()
        agent._callable_accepts_kwarg = MagicMock(return_value=True)
        agent._reset_reasoning_capture = MagicMock()
        agent._drain_reasoning_chunks = MagicMock(return_value=[])

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="test message"):
            chunks.append(chunk)

        # Verify session was saved
        agent._session.add_message.assert_any_call("user", "test message")
        agent._session.add_message.assert_any_call("assistant", "hello")
        agent.sessions.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_close_cancels_background_run_and_skips_session_save(self):
        """Closing the stream should stop the background run and avoid persisting stale output."""
        from spoon_bot.agent.loop import AgentLoop

        run_cancelled = asyncio.Event()

        async def mock_run(**kwargs):
            await agent._agent.output_queue.put({"content": "hello"})
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                run_cancelled.set()
                raise

        agent = MagicMock(spec=AgentLoop)
        agent._initialized = True
        agent._agent = MagicMock()
        agent._agent.output_queue = asyncio.Queue()
        agent._agent.task_done = asyncio.Event()
        agent._agent.run = mock_run
        agent._agent.add_message = AsyncMock()
        agent._agent.state = "idle"
        agent._session = MagicMock()
        agent._session.add_message = MagicMock()
        agent.sessions = MagicMock()
        agent.sessions.save = MagicMock()
        agent.memory = MagicMock()
        agent.memory.get_memory_context = MagicMock(return_value=None)
        agent.context = MagicMock()
        agent._prepare_request_context = AsyncMock()
        agent._build_runtime_message_content = MagicMock(side_effect=lambda *args, **kwargs: args[1])
        agent._build_step_prompt = MagicMock(return_value="prompt")
        agent._install_anti_loop_tracker = MagicMock()
        agent._restore_agent_think = MagicMock()
        agent._callable_accepts_kwarg = MagicMock(return_value=True)
        agent._reset_reasoning_capture = MagicMock()
        agent._drain_reasoning_chunks = MagicMock(return_value=[])

        stream = AgentLoop.stream(agent, message="test message")
        first_chunk = await stream.__anext__()

        assert first_chunk["type"] == "content"
        assert first_chunk["delta"] == "hello"

        await stream.aclose()
        await asyncio.wait_for(run_cancelled.wait(), timeout=1.0)

        agent.sessions.save.assert_not_called()
        agent._session.add_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_install_anti_loop_tracker_does_not_stack_previous_request_prompt(self):
        """A new request should not inherit the previous request's anti-loop wrapper."""
        from pathlib import Path
        from spoon_bot.agent.loop import AgentLoop

        seen_prompts = []

        async def base_think():
            seen_prompts.append(agent._agent.next_step_prompt)
            return True

        tool_call = MagicMock()
        tool_call.function = MagicMock()
        tool_call.function.name = "shell"
        tool_call.function.arguments = '{"command":"cd /workspace && ls -la .agents/skills/pdf"}'

        agent = MagicMock(spec=AgentLoop)
        agent.workspace = Path("/workspace")
        agent._agent = MagicMock()
        agent._agent.think = base_think
        agent._agent._spoon_bot_base_think = base_think
        agent._agent.next_step_prompt = ""
        agent._agent.tool_calls = [tool_call]
        agent._agent.memory = MagicMock()
        agent._agent.memory.messages = []
        agent._compress_runtime_context = MagicMock(return_value=0)

        AgentLoop._install_anti_loop_tracker(agent, "prompt one")
        agent._agent.next_step_prompt = "prompt one"
        await agent._agent.think()

        AgentLoop._install_anti_loop_tracker(agent, "prompt two")
        agent._agent.next_step_prompt = "prompt two"
        await agent._agent.think()

        assert seen_prompts[-1] == "prompt two"


@pytest.mark.requires_spoon_core
class TestAgentLoopProcessWithThinking:
    """Test AgentLoop.process_with_thinking() method."""

    @pytest.mark.asyncio
    async def test_returns_tuple(self):
        """process_with_thinking() should return (response, thinking_content)."""
        from spoon_bot.agent.loop import AgentLoop

        result = MagicMock()
        result.content = "The answer is 42"
        result.thinking_content = "I need to think about this..."

        mock_inner_agent = MagicMock()
        mock_inner_agent.run = AsyncMock(return_value=result)
        mock_inner_agent.add_message = AsyncMock()

        agent = MagicMock(spec=AgentLoop)
        agent._initialized = True
        agent._agent = mock_inner_agent
        agent._session = MagicMock()
        agent._session.add_message = MagicMock()
        agent.sessions = MagicMock()
        agent.memory = MagicMock()
        agent.memory.get_memory_context = MagicMock(return_value=None)
        agent.context = MagicMock()
        agent._auto_commit = False
        agent._git = None
        agent._prepare_request_context = AsyncMock()
        agent._build_runtime_message_content = MagicMock(side_effect=lambda *args, **kwargs: args[1])
        agent._build_step_prompt = MagicMock(return_value="prompt")
        agent._install_anti_loop_tracker = MagicMock()
        agent._restore_agent_think = MagicMock()
        agent._callable_accepts_kwarg = MagicMock(return_value=True)
        agent._reset_reasoning_capture = MagicMock()
        agent._looks_like_duplicate_thinking = AgentLoop._looks_like_duplicate_thinking.__get__(agent, AgentLoop)
        agent._normalize_comparable_text = AgentLoop._normalize_comparable_text
        agent._latest_reasoning_excerpt = None

        response, thinking = await AgentLoop.process_with_thinking(agent, message="What is 6*7?")

        assert response == "The answer is 42"
        assert thinking == "I need to think about this..."

    @pytest.mark.asyncio
    async def test_process_with_thinking_binds_tool_owner(self):
        """process_with_thinking() should bind tool owner context for run()."""
        from contextlib import contextmanager
        from spoon_bot.agent.loop import AgentLoop

        result = MagicMock()
        result.content = "ok"
        result.thinking_content = None

        mock_inner_agent = MagicMock()
        mock_inner_agent.run = AsyncMock(return_value=result)
        mock_inner_agent.add_message = AsyncMock()

        agent = MagicMock(spec=AgentLoop)
        agent._initialized = True
        agent._agent = mock_inner_agent
        agent._session = MagicMock()
        agent._session.add_message = MagicMock()
        agent.sessions = MagicMock()
        agent.memory = MagicMock()
        agent.memory.get_memory_context = MagicMock(return_value=None)
        agent.context = MagicMock()
        agent._auto_commit = False
        agent._git = None
        agent._prepare_request_context = AsyncMock()
        agent._build_runtime_message_content = MagicMock(side_effect=lambda *args, **kwargs: args[1])
        agent._build_step_prompt = MagicMock(return_value="prompt")
        agent._install_anti_loop_tracker = MagicMock()
        agent._restore_agent_think = MagicMock()
        agent._callable_accepts_kwarg = MagicMock(return_value=True)
        agent._reset_reasoning_capture = MagicMock()
        agent._looks_like_duplicate_thinking = AgentLoop._looks_like_duplicate_thinking.__get__(agent, AgentLoop)
        agent._normalize_comparable_text = AgentLoop._normalize_comparable_text
        agent._latest_reasoning_excerpt = None
        agent._current_tool_owner_key = MagicMock(return_value="user:alice|session:default")

        owners: list[str] = []

        @contextmanager
        def _capture_owner(owner):
            owners.append(str(owner))
            yield owner

        with patch("spoon_bot.agent.loop.bind_tool_owner", side_effect=_capture_owner):
            response, thinking = await AgentLoop.process_with_thinking(agent, message="test")

        assert response == "ok"
        assert thinking is None
        assert owners == ["user:alice|session:default"]

    @pytest.mark.asyncio
    async def test_thinking_fallback_attributes(self):
        """Should try multiple attributes for thinking content."""
        from spoon_bot.agent.loop import AgentLoop

        # Test with .thinking attribute (not .thinking_content)
        result = MagicMock(spec=["content", "thinking"])
        result.content = "Answer"
        result.thinking = "My thought process"

        mock_inner_agent = MagicMock()
        mock_inner_agent.run = AsyncMock(return_value=result)
        mock_inner_agent.add_message = AsyncMock()

        agent = MagicMock(spec=AgentLoop)
        agent._initialized = True
        agent._agent = mock_inner_agent
        agent._session = MagicMock()
        agent._session.add_message = MagicMock()
        agent.sessions = MagicMock()
        agent.memory = MagicMock()
        agent.memory.get_memory_context = MagicMock(return_value=None)
        agent.context = MagicMock()
        agent._auto_commit = False
        agent._git = None
        agent._prepare_request_context = AsyncMock()
        agent._build_runtime_message_content = MagicMock(side_effect=lambda *args, **kwargs: args[1])
        agent._build_step_prompt = MagicMock(return_value="prompt")
        agent._install_anti_loop_tracker = MagicMock()
        agent._restore_agent_think = MagicMock()
        agent._callable_accepts_kwarg = MagicMock(return_value=True)
        agent._reset_reasoning_capture = MagicMock()
        agent._looks_like_duplicate_thinking = AgentLoop._looks_like_duplicate_thinking.__get__(agent, AgentLoop)
        agent._normalize_comparable_text = AgentLoop._normalize_comparable_text
        agent._latest_reasoning_excerpt = None

        response, thinking = await AgentLoop.process_with_thinking(agent, message="test")
        assert thinking == "My thought process"

    @pytest.mark.asyncio
    async def test_thinking_from_metadata(self):
        """Should extract thinking from metadata dict."""
        from spoon_bot.agent.loop import AgentLoop

        result = MagicMock(spec=["content", "metadata"])
        result.content = "Answer"
        result.metadata = {"thinking": "Metadata thought"}

        mock_inner_agent = MagicMock()
        mock_inner_agent.run = AsyncMock(return_value=result)
        mock_inner_agent.add_message = AsyncMock()

        agent = MagicMock(spec=AgentLoop)
        agent._initialized = True
        agent._agent = mock_inner_agent
        agent._session = MagicMock()
        agent._session.add_message = MagicMock()
        agent.sessions = MagicMock()
        agent.memory = MagicMock()
        agent.memory.get_memory_context = MagicMock(return_value=None)
        agent.context = MagicMock()
        agent._auto_commit = False
        agent._git = None
        agent._prepare_request_context = AsyncMock()
        agent._build_runtime_message_content = MagicMock(side_effect=lambda *args, **kwargs: args[1])
        agent._build_step_prompt = MagicMock(return_value="prompt")
        agent._install_anti_loop_tracker = MagicMock()
        agent._restore_agent_think = MagicMock()
        agent._callable_accepts_kwarg = MagicMock(return_value=True)
        agent._reset_reasoning_capture = MagicMock()
        agent._looks_like_duplicate_thinking = AgentLoop._looks_like_duplicate_thinking.__get__(agent, AgentLoop)
        agent._normalize_comparable_text = AgentLoop._normalize_comparable_text
        agent._latest_reasoning_excerpt = None

        response, thinking = await AgentLoop.process_with_thinking(agent, message="test")
        assert thinking == "Metadata thought"

    @pytest.mark.asyncio
    async def test_does_not_fall_back_to_tracked_reasoning_when_provider_omits_thinking_fields(self):
        """process_with_thinking() should not reuse tracked assistant output as thinking."""
        from spoon_bot.agent.loop import AgentLoop

        result = MagicMock(spec=["content"])
        result.content = "Answer"

        async def mock_run(**kwargs):
            agent._capture_reasoning_text("Tracked reasoning excerpt")
            return result

        mock_inner_agent = MagicMock()
        mock_inner_agent.run = mock_run
        mock_inner_agent.add_message = AsyncMock()

        agent = MagicMock(spec=AgentLoop)
        agent._initialized = True
        agent._agent = mock_inner_agent
        agent._session = MagicMock()
        agent._session.add_message = MagicMock()
        agent.sessions = MagicMock()
        agent.memory = MagicMock()
        agent.memory.get_memory_context = MagicMock(return_value=None)
        agent.context = MagicMock()
        agent._auto_commit = False
        agent._git = None
        agent._prepare_request_context = AsyncMock()
        agent._build_runtime_message_content = MagicMock(side_effect=lambda *args, **kwargs: args[1])
        agent._build_step_prompt = MagicMock(return_value="prompt")
        agent._install_anti_loop_tracker = MagicMock()
        agent._restore_agent_think = MagicMock()
        agent._callable_accepts_kwarg = MagicMock(return_value=True)
        agent._reset_reasoning_capture = AgentLoop._reset_reasoning_capture.__get__(agent, AgentLoop)
        agent._capture_reasoning_text = AgentLoop._capture_reasoning_text.__get__(agent, AgentLoop)
        agent._looks_like_duplicate_thinking = AgentLoop._looks_like_duplicate_thinking.__get__(agent, AgentLoop)
        agent._normalize_comparable_text = AgentLoop._normalize_comparable_text
        agent._reset_reasoning_capture()

        response, thinking = await AgentLoop.process_with_thinking(agent, message="test")

        assert response == "Answer"
        assert thinking is None

    @pytest.mark.asyncio
    async def test_process_with_thinking_suppresses_duplicate_metadata_thinking(self):
        """Provider thinking identical to final content should be omitted."""
        from spoon_bot.agent.loop import AgentLoop

        result = MagicMock(spec=["content", "metadata"])
        result.content = "Answer"
        result.metadata = {"thinking": "Answer"}

        mock_inner_agent = MagicMock()
        mock_inner_agent.run = AsyncMock(return_value=result)
        mock_inner_agent.add_message = AsyncMock()

        agent = MagicMock(spec=AgentLoop)
        agent._initialized = True
        agent._agent = mock_inner_agent
        agent._session = MagicMock()
        agent._session.add_message = MagicMock()
        agent.sessions = MagicMock()
        agent.memory = MagicMock()
        agent.memory.get_memory_context = MagicMock(return_value=None)
        agent.context = MagicMock()
        agent._auto_commit = False
        agent._git = None
        agent._prepare_request_context = AsyncMock()
        agent._build_runtime_message_content = MagicMock(side_effect=lambda *args, **kwargs: args[1])
        agent._build_step_prompt = MagicMock(return_value="prompt")
        agent._install_anti_loop_tracker = MagicMock()
        agent._restore_agent_think = MagicMock()
        agent._callable_accepts_kwarg = MagicMock(return_value=True)
        agent._reset_reasoning_capture = AgentLoop._reset_reasoning_capture.__get__(agent, AgentLoop)
        agent._looks_like_duplicate_thinking = AgentLoop._looks_like_duplicate_thinking.__get__(agent, AgentLoop)
        agent._normalize_comparable_text = AgentLoop._normalize_comparable_text
        agent._reset_reasoning_capture()

        response, thinking = await AgentLoop.process_with_thinking(agent, message="test")

        assert response == "Answer"
        assert thinking is None

    @pytest.mark.asyncio
    async def test_error_returns_friendly_message(self):
        """Should raise on agent error (caller handles it)."""
        from spoon_bot.agent.loop import AgentLoop

        mock_inner_agent = MagicMock()
        mock_inner_agent.run = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
        mock_inner_agent.add_message = AsyncMock()

        agent = MagicMock(spec=AgentLoop)
        agent._initialized = True
        agent._agent = mock_inner_agent
        agent._session = MagicMock()
        agent.sessions = MagicMock()
        agent.memory = MagicMock()
        agent.memory.get_memory_context = MagicMock(return_value=None)
        agent.context = MagicMock()
        agent._auto_commit = False
        agent._git = None
        agent._prepare_request_context = AsyncMock()
        agent._build_runtime_message_content = MagicMock(side_effect=lambda *args, **kwargs: args[1])
        agent._build_step_prompt = MagicMock(return_value="prompt")
        agent._install_anti_loop_tracker = MagicMock()
        agent._restore_agent_think = MagicMock()
        agent._callable_accepts_kwarg = MagicMock(return_value=True)
        agent._reset_reasoning_capture = MagicMock()
        agent._latest_reasoning_excerpt = None

        with pytest.raises(RuntimeError, match="LLM unavailable"):
            await AgentLoop.process_with_thinking(agent, message="test")


# ============================================================
# REST API endpoint tests
# ============================================================


def _create_test_app(mock_agent=None):
    """Create a test FastAPI app with mocked dependencies."""
    from spoon_bot.gateway.app import create_app, set_agent
    from spoon_bot.gateway.config import GatewayConfig
    import spoon_bot.gateway.app as app_module

    # Disable auth for testing
    app_module._auth_required = False

    config = GatewayConfig.from_env()
    app = create_app(config)

    if mock_agent:
        set_agent(mock_agent)

    return app


class TestRESTEndpointNonStreaming:
    """Test REST /chat endpoint in non-streaming mode."""

    def test_non_streaming_returns_json(self):
        """Non-streaming chat should return standard APIResponse."""
        mock_agent = MagicMock()
        mock_agent.process = AsyncMock(return_value="Hello from agent")
        mock_agent.sessions = MagicMock()
        mock_agent.tools = MagicMock()
        mock_agent.tools.list_tools = MagicMock(return_value=[])
        mock_agent.skills = []

        app = _create_test_app(mock_agent)
        client = TestClient(app)

        response = client.post(
            "/v1/agent/chat",
            json={"message": "hello"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["response"] == "Hello from agent"
        assert data["data"]["thinking_content"] is None

    def test_non_streaming_with_options(self):
        """Non-streaming with explicit options should work."""
        mock_agent = MagicMock()
        mock_agent.process = AsyncMock(return_value="Response")
        mock_agent.sessions = MagicMock()
        mock_agent.tools = MagicMock()
        mock_agent.tools.list_tools = MagicMock(return_value=[])
        mock_agent.skills = []

        app = _create_test_app(mock_agent)
        client = TestClient(app)

        response = client.post(
            "/v1/agent/chat",
            json={
                "message": "hello",
                "options": {"stream": False, "thinking": False},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_thinking_non_streaming(self):
        """Thinking mode without streaming should return thinking_content."""
        mock_agent = MagicMock()
        mock_agent.process_with_thinking = AsyncMock(
            return_value=("Answer", "My thinking process")
        )
        mock_agent.sessions = MagicMock()
        mock_agent.tools = MagicMock()
        mock_agent.tools.list_tools = MagicMock(return_value=[])
        mock_agent.skills = []

        app = _create_test_app(mock_agent)
        client = TestClient(app)

        response = client.post(
            "/v1/agent/chat",
            json={
                "message": "think about this",
                "options": {"thinking": True},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["response"] == "Answer"
        assert data["data"]["thinking_content"] == "My thinking process"


class TestRESTEndpointStreaming:
    """Test REST /chat endpoint in streaming mode."""

    def test_streaming_returns_sse(self):
        """Streaming chat should return text/event-stream with no done chunk (only [DONE])."""
        mock_agent = MagicMock()

        async def mock_stream(**kwargs):
            yield {"type": "content", "delta": "Hello", "metadata": {}}
            yield {"type": "content", "delta": " World", "metadata": {}}
            yield {"type": "done", "delta": "", "metadata": {"content": "Hello World"}}

        mock_agent.stream = mock_stream
        mock_agent.sessions = MagicMock()
        mock_agent.tools = MagicMock()
        mock_agent.tools.list_tools = MagicMock(return_value=[])
        mock_agent.skills = []

        app = _create_test_app(mock_agent)
        client = TestClient(app)

        response = client.post(
            "/v1/agent/chat",
            json={
                "message": "hello",
                "options": {"stream": True},
            },
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        # Parse SSE events
        lines = response.text.strip().split("\n\n")
        events = []
        for line in lines:
            if line.startswith("data: "):
                payload = line[6:]
                if payload == "[DONE]":
                    events.append({"_done": True})
                else:
                    events.append(json.loads(payload))

        # Should have content chunks and [DONE] — done chunks are filtered
        content_events = [e for e in events if e.get("type") == "content"]
        done_events = [e for e in events if e.get("type") == "done"]
        terminal = [e for e in events if e.get("_done")]

        assert len(content_events) == 2
        assert content_events[0]["delta"] == "Hello"
        assert content_events[1]["delta"] == " World"
        assert len(done_events) == 0  # done chunks are filtered out
        assert len(terminal) == 1  # only [DONE] terminal signal

    def test_streaming_has_correct_headers(self):
        """Streaming response should have Cache-Control and Connection headers."""
        mock_agent = MagicMock()

        async def mock_stream(**kwargs):
            yield {"type": "done", "delta": "", "metadata": {"content": ""}}

        mock_agent.stream = mock_stream
        mock_agent.sessions = MagicMock()
        mock_agent.tools = MagicMock()
        mock_agent.tools.list_tools = MagicMock(return_value=[])
        mock_agent.skills = []

        app = _create_test_app(mock_agent)
        client = TestClient(app)

        response = client.post(
            "/v1/agent/chat",
            json={
                "message": "hello",
                "options": {"stream": True},
            },
        )

        assert response.headers.get("cache-control") == "no-cache"
        assert "x-request-id" in response.headers

    def test_streaming_with_thinking(self):
        """Streaming with thinking should include thinking chunks."""
        mock_agent = MagicMock()

        async def mock_stream(**kwargs):
            yield {"type": "thinking", "delta": "Let me think...", "metadata": {}}
            yield {"type": "content", "delta": "Answer", "metadata": {}}
            yield {"type": "done", "delta": "", "metadata": {"content": "Answer"}}

        mock_agent.stream = mock_stream
        mock_agent.sessions = MagicMock()
        mock_agent.tools = MagicMock()
        mock_agent.tools.list_tools = MagicMock(return_value=[])
        mock_agent.skills = []

        app = _create_test_app(mock_agent)
        client = TestClient(app)

        response = client.post(
            "/v1/agent/chat",
            json={
                "message": "think and answer",
                "options": {"stream": True, "thinking": True},
            },
        )

        assert response.status_code == 200
        lines = response.text.strip().split("\n\n")
        events = []
        for line in lines:
            if line.startswith("data: ") and line[6:] != "[DONE]":
                events.append(json.loads(line[6:]))

        thinking_events = [e for e in events if e.get("type") == "thinking"]
        assert len(thinking_events) >= 1
        assert thinking_events[0]["delta"] == "Let me think..."

        # Verify no done chunks leaked through
        done_events = [e for e in events if e.get("type") == "done"]
        assert len(done_events) == 0

    def test_streaming_sse_error_handling(self):
        """SSE should emit error chunk when agent.stream() raises."""
        mock_agent = MagicMock()

        async def mock_stream(**kwargs):
            yield {"type": "content", "delta": "partial", "metadata": {}}
            raise RuntimeError("LLM connection lost")

        mock_agent.stream = mock_stream
        mock_agent.sessions = MagicMock()
        mock_agent.tools = MagicMock()
        mock_agent.tools.list_tools = MagicMock(return_value=[])
        mock_agent.skills = []

        app = _create_test_app(mock_agent)
        client = TestClient(app)

        response = client.post(
            "/v1/agent/chat",
            json={
                "message": "hello",
                "options": {"stream": True},
            },
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        # Parse SSE events
        lines = response.text.strip().split("\n\n")
        events = []
        for line in lines:
            if line.startswith("data: "):
                payload = line[6:]
                if payload == "[DONE]":
                    events.append({"_done": True})
                else:
                    events.append(json.loads(payload))

        # Should have partial content, error chunk, and [DONE]
        content_events = [e for e in events if e.get("type") == "content"]
        error_events = [e for e in events if e.get("type") == "error"]
        terminal = [e for e in events if e.get("_done")]

        assert len(content_events) == 1
        assert content_events[0]["delta"] == "partial"
        assert len(error_events) == 1
        assert "LLM connection lost" in error_events[0]["metadata"]["error"]
        assert len(terminal) == 1  # [DONE] still sent after error


class TestBackwardCompatibility:
    """Test that existing behavior is completely unchanged."""

    def test_request_without_options_field(self):
        """Request without 'options' at all should work as before."""
        mock_agent = MagicMock()
        mock_agent.process = AsyncMock(return_value="Old behavior works")
        mock_agent.sessions = MagicMock()
        mock_agent.tools = MagicMock()
        mock_agent.tools.list_tools = MagicMock(return_value=[])
        mock_agent.skills = []

        app = _create_test_app(mock_agent)
        client = TestClient(app)

        response = client.post(
            "/v1/agent/chat",
            json={"message": "hello"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["response"] == "Old behavior works"
        # process was called, not process_with_thinking
        mock_agent.process.assert_called_once()

    def test_request_with_empty_options(self):
        """Request with empty options dict should use defaults."""
        mock_agent = MagicMock()
        mock_agent.process = AsyncMock(return_value="Default behavior")
        mock_agent.sessions = MagicMock()
        mock_agent.tools = MagicMock()
        mock_agent.tools.list_tools = MagicMock(return_value=[])
        mock_agent.skills = []

        app = _create_test_app(mock_agent)
        client = TestClient(app)

        response = client.post(
            "/v1/agent/chat",
            json={"message": "hello", "options": {}},
        )

        assert response.status_code == 200
        mock_agent.process.assert_called_once()

    def test_old_response_format_preserved(self):
        """Non-streaming response should still match original schema."""
        mock_agent = MagicMock()
        mock_agent.process = AsyncMock(return_value="test response")
        mock_agent.sessions = MagicMock()
        mock_agent.tools = MagicMock()
        mock_agent.tools.list_tools = MagicMock(return_value=[])
        mock_agent.skills = []

        app = _create_test_app(mock_agent)
        client = TestClient(app)

        response = client.post(
            "/v1/agent/chat",
            json={"message": "hello"},
        )

        data = response.json()
        # Must have these exact keys
        assert "success" in data
        assert "data" in data
        assert "meta" in data
        assert "response" in data["data"]
        assert "tool_calls" in data["data"]
        assert "usage" in data["data"]
        # thinking_content is new but None by default
        assert "thinking_content" in data["data"]
        assert data["data"]["thinking_content"] is None

    def test_session_key_still_works(self):
        """Session key in request should still be handled."""
        mock_agent = MagicMock()
        mock_agent.process = AsyncMock(return_value="ok")
        mock_agent.sessions = MagicMock()
        mock_agent.tools = MagicMock()
        mock_agent.tools.list_tools = MagicMock(return_value=[])
        mock_agent.skills = []

        app = _create_test_app(mock_agent)
        client = TestClient(app)

        response = client.post(
            "/v1/agent/chat",
            json={"message": "hello", "session_key": "custom-session"},
        )

        assert response.status_code == 200

    def test_media_field_still_works(self, tmp_path: Path):
        """Media field should still work in non-streaming mode."""
        mock_agent = MagicMock()
        mock_agent.process = AsyncMock(return_value="processed with media")
        mock_agent.sessions = MagicMock()
        mock_agent.tools = MagicMock()
        mock_agent.tools.list_tools = MagicMock(return_value=[])
        mock_agent.skills = []
        mock_agent.workspace = tmp_path

        app = _create_test_app(mock_agent)
        client = TestClient(app)

        image_path = tmp_path / "image.png"
        image_path.write_bytes(b"png")

        response = client.post(
            "/v1/agent/chat",
            json={
                "message": "analyze this",
                "media": [str(image_path)],
            },
        )

        assert response.status_code == 200


# ============================================================
# Module import tests
# ============================================================


class TestModuleExports:
    """Test that new models are properly exported."""

    def test_stream_chunk_importable_from_models(self):
        from spoon_bot.gateway.models import StreamChunk

        assert StreamChunk is not None

    def test_stream_chunk_importable_from_responses(self):
        from spoon_bot.gateway.models.responses import StreamChunk

        assert StreamChunk is not None

    def test_chat_options_has_thinking(self):
        from spoon_bot.gateway.models.requests import ChatOptions

        opts = ChatOptions()
        assert hasattr(opts, "thinking")

    def test_chat_options_has_stream(self):
        from spoon_bot.gateway.models.requests import ChatOptions

        opts = ChatOptions()
        assert hasattr(opts, "stream")

    def test_chat_response_has_thinking_content(self):
        from spoon_bot.gateway.models.responses import ChatResponse

        resp = ChatResponse(response="test")
        assert hasattr(resp, "thinking_content")
