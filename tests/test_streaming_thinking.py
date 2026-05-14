"""
Tests for streaming and thinking support.

Tests the new streaming SSE, thinking mode, and backward compatibility
for the spoon-bot gateway API.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
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
        assert opts.reasoning_effort is None
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

    def test_reasoning_effort_roundtrip(self):
        from spoon_bot.gateway.models.requests import ChatOptions

        opts = ChatOptions(stream=True, thinking=True, reasoning_effort="high")
        data = opts.model_dump()
        restored = ChatOptions(**data)
        assert restored.reasoning_effort == "high"


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
        assert data == {
            "type": "content",
            "delta": "hello world",
            "metadata": {},
            "source": None,
        }

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

    def _make_stream_agent(self, items, *, run_result_text: str = "", run_error: Exception | None = None):
        from spoon_bot.agent.loop import AgentLoop

        queue: asyncio.Queue = asyncio.Queue()
        task_done = asyncio.Event()

        async def run(**kwargs):
            for item in items:
                await queue.put(item)
            if run_error:
                raise run_error
            if run_result_text:
                return MagicMock(content=run_result_text)
            return MagicMock(content=None)

        async def run_with_retry(label="agent", pre_retry_cleanup=None, **run_kwargs):
            return await agent._agent.run(**run_kwargs)

        agent = MagicMock(spec=AgentLoop)
        agent._initialized = True
        agent.provider = "anthropic"
        agent._chatbot = None
        agent._agent = MagicMock()
        agent._agent.add_message = AsyncMock()
        agent._agent.run = AsyncMock(side_effect=run)
        agent._agent.output_queue = queue
        agent._agent.task_done = task_done
        agent._agent.state = "idle"
        agent._agent.system_prompt = "base system"
        agent._agent._original_system_prompt = "base original"
        agent._session = MagicMock()
        agent._session.add_message = MagicMock()
        agent.sessions = MagicMock()
        agent.sessions.save = MagicMock()
        agent.memory = MagicMock()
        agent.memory.get_memory_context = MagicMock(return_value=None)
        agent.context = MagicMock()
        agent._prepare_request_context = AsyncMock()
        agent._add_current_turn_skill_zip_context = MagicMock(side_effect=lambda text: text)
        agent._build_runtime_message_content = MagicMock(side_effect=lambda *args, **kwargs: args[1])
        agent._build_step_prompt = MagicMock(return_value="prompt")
        agent._build_request_context_prompt = MagicMock(return_value="[USER REQUEST]: test")
        agent._install_anti_loop_tracker = MagicMock()
        agent._restore_agent_think = MagicMock()
        agent._callable_accepts_kwarg = MagicMock(return_value=True)
        agent._reset_reasoning_capture = MagicMock()
        agent._drain_reasoning_chunks = MagicMock(return_value=[])
        agent._run_agent_with_retry = AsyncMock(side_effect=run_with_retry)
        agent._select_next_step_prompt = AgentLoop._select_next_step_prompt.__get__(agent, AgentLoop)
        agent._normalize_comparable_text = AgentLoop._normalize_comparable_text
        agent._persist_turn = MagicMock()
        agent._current_tool_owner_key = MagicMock(return_value="user:test|session:default")
        agent.set_subagent_context = MagicMock()
        return agent

    def test_select_next_step_prompt_keeps_context_for_openrouter_thinking(self):
        """Thinking mode should use the same lightweight next-step prompt on OpenRouter."""
        from spoon_bot.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        loop.provider = "openrouter"
        loop._chatbot = None
        loop._build_step_prompt = MagicMock(return_value="context prompt")

        prompt = AgentLoop._select_next_step_prompt(loop, "inspect workspace", thinking=True)

        assert prompt == AgentLoop.DEFAULT_NEXT_STEP_PROMPT
        loop._build_step_prompt.assert_not_called()

    def test_select_next_step_prompt_keeps_context_for_other_providers(self):
        """Thinking mode should use the same lightweight next-step prompt for every provider."""
        from spoon_bot.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        loop.provider = "anthropic"
        loop._chatbot = None
        loop._build_step_prompt = MagicMock(return_value="context prompt")

        prompt = AgentLoop._select_next_step_prompt(loop, "inspect workspace", thinking=True)

        assert prompt == AgentLoop.DEFAULT_NEXT_STEP_PROMPT
        loop._build_step_prompt.assert_not_called()

    def test_apply_request_context_to_system_prompt_augments_and_restores_prompts(self):
        """Thinking runs should preserve request context via temporary system prompt augmentation."""
        from spoon_bot.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        loop._agent = MagicMock()
        loop._agent.system_prompt = "base system"
        loop._agent._original_system_prompt = "base original"
        loop._build_request_context_prompt = MagicMock(return_value="[USER REQUEST]: inspect workspace")

        original_prompt, original_base_prompt = AgentLoop._apply_request_context_to_system_prompt(
            loop,
            "inspect workspace",
            thinking=True,
        )

        assert original_prompt == "base system"
        assert original_base_prompt == "base original"
        assert "## Active Request Context" in loop._agent.system_prompt
        assert "[USER REQUEST]: inspect workspace" in loop._agent.system_prompt
        assert "## Active Request Context" in loop._agent._original_system_prompt

        AgentLoop._restore_request_context_system_prompt(
            loop,
            original_prompt,
            original_base_prompt,
        )

        assert loop._agent.system_prompt == "base system"
        assert loop._agent._original_system_prompt == "base original"

    def test_apply_request_context_to_system_prompt_also_augments_non_thinking_runs(self):
        """Non-thinking runs should receive the same active request context guardrails."""
        from spoon_bot.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        loop._agent = MagicMock()
        loop._agent.system_prompt = "base system"
        loop._build_request_context_prompt = MagicMock(return_value="[USER REQUEST]: keep only latest")

        original_prompt, original_base_prompt = AgentLoop._apply_request_context_to_system_prompt(
            loop,
            "keep only latest",
            thinking=False,
        )

        assert original_prompt == "base system"
        assert original_base_prompt is not None
        assert "## Active Request Context" in loop._agent.system_prompt
        assert "[USER REQUEST]: keep only latest" in loop._agent.system_prompt

    def test_build_request_context_prompt_preserves_tail_output_constraints(self):
        """Compact request scaffolding must keep the latest request ending verbatim."""
        from spoon_bot.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        loop.workspace = Path("/workspace")
        loop._extract_env_for_prompt = MagicMock(return_value="")

        message = (
            "Inspect the workspace carefully. "
            + ("middle context " * 30)
            + "When you are done, reply with exactly: FINAL_OK. Do not summarize."
        )

        prompt = AgentLoop._build_request_context_prompt(loop, message)

        assert "[CURRENT DATE]:" in prompt
        assert "[CURRENT TIMEZONE]:" in prompt
        assert "[... middle omitted to save tokens; preserve latest tail instructions ...]" in prompt
        assert "reply with exactly: final_ok" in prompt.lower()
        assert "Do not summarize." in prompt
        assert "[AUTHORITATIVE REQUEST ENDING]" not in prompt
        assert "[LATEST REQUEST ENDING]:" not in prompt

    def test_build_request_context_prompt_marks_latest_turn_authoritative(self):
        """Thinking scaffolding should explicitly supersede unfinished prior plans."""
        from spoon_bot.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        loop.workspace = Path("/workspace")
        loop._extract_env_for_prompt = MagicMock(return_value="")

        prompt = AgentLoop._build_request_context_prompt(loop, "Handle only this newest request.")

        assert "[TURN PRIORITY]:" in prompt
        assert "Execute only the newest user request." in prompt
        assert "stale tool sequence" in prompt

    def test_build_request_context_prompt_bounds_external_side_effects(self):
        """A single request must not turn into an unbounded external side-effect loop."""
        from spoon_bot.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        loop.workspace = Path("/workspace")
        loop._extract_env_for_prompt = MagicMock(return_value="")

        prompt = AgentLoop._build_request_context_prompt(loop, "Run the next external action.")

        assert "[EXTERNAL SIDE-EFFECT BOUNDARY]:" in prompt
        assert "at most one" in prompt
        assert "external" in prompt
        assert "explicit count" in prompt
        assert "Do not batch multiple alternative tool attempts" in prompt
        assert "run that command exactly as written" in prompt
        assert "never remove protective wrappers" in prompt
        assert "never convert a simulated command into a live" in prompt

    def test_build_request_context_prompt_explains_interrupted_previous_request_resolution(self):
        """Interrupted prior requests should be presented as amend-vs-replace context, not a second task."""
        from spoon_bot.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        loop.workspace = Path("/workspace")
        loop._extract_env_for_prompt = MagicMock(return_value="")
        loop._recent_turn_notice = (
            "The immediately previous user request was interrupted before completion.\n"
            "[INTERRUPTED PREVIOUS REQUEST]: Continue the report export with profile C\n"
            "Resolve it against the newest user message as follows:\n"
            "- If the newest user message is itself a standalone actionable request, treat it as replacing the interrupted request.\n"
            "- If the newest user message only adds constraints or details to the interrupted request, continue the interrupted request with the new constraints applied.\n"
            "- Do not execute both as separate tasks unless the newest user message explicitly asks for both."
        )

        prompt = AgentLoop._build_request_context_prompt(loop, "Continue the report export with profile A")

        assert "[PREVIOUS TURN STATUS]:" in prompt
        assert "[INTERRUPTED PREVIOUS REQUEST]: Continue the report export with profile C" in prompt
        assert "standalone actionable request" in prompt
        assert "Do not execute both as separate tasks" in prompt

    def test_build_request_context_prompt_keeps_latest_request_tail_without_contract_routing(self):
        """Long requests keep the latest tail verbatim without prompt-derived contracts."""
        from spoon_bot.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        loop.workspace = Path("/workspace")
        loop._extract_env_for_prompt = MagicMock(return_value="")

        transfer_tail = (
            "1. Finalize the audit checklist for workspace PROJECT-123"
        )
        message = (
            "Do not continue the earlier status records or overview."
            + ("stale context " * 80)
            + transfer_tail
        )

        prompt = AgentLoop._build_request_context_prompt(loop, message)

        assert "[TURN PRIORITY]:" in prompt
        assert "[REQUEST CONTRACT]:" not in prompt
        assert "[PREVIOUS ASSISTANT OFFER]:" not in prompt
        assert transfer_tail in prompt
    def test_build_step_prompt_marks_latest_turn_authoritative(self):
        """Per-step prompts should make the newest user task override prior execution plans."""
        from spoon_bot.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        loop.workspace = Path("/workspace")
        loop._extract_env_for_prompt = MagicMock(return_value="")

        prompt = AgentLoop._build_step_prompt(loop, "Fix the bug now.")

        assert "[TURN PRIORITY]:" in prompt
        assert "Execute only the newest user request." in prompt
        assert "[USER REQUEST]: Fix the bug now." in prompt

    def test_finalize_response_content_filters_execution_artifacts(self):
        """Finalization should not rewrite output based on prompt keywords."""
        from spoon_bot.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        content = "Step 1: Observed output of cmd: ignored\n\nActual answer"

        cleaned = AgentLoop._finalize_response_content(
            loop,
            "Reply with OK",
            content,
            turn_memory_start_index=0,
        )

        assert cleaned == "Actual answer"

    def test_finalize_response_content_strips_leaked_scratchpad_prefix(self):
        """Internal planning notes should not be part of the user-facing answer."""
        from spoon_bot.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        content = (
            "Need to inspect the generated file before answering. "
            "Here is the result: the script queries balances only."
        )

        cleaned = AgentLoop._finalize_response_content(
            loop,
            "Explain the generated script",
            content,
            turn_memory_start_index=0,
        )

        assert cleaned == "Here is the result: the script queries balances only."

    def test_finalize_response_content_strips_let_me_scratchpad_prefix(self):
        """English execution preambles should not be replayed as final content."""
        from spoon_bot.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        content = (
            "Let me start by fetching the repo to understand the skill structure."
            "The agent stopped the tool loop because status was already known."
        )

        cleaned = AgentLoop._finalize_response_content(
            loop,
            "Continue",
            content,
            turn_memory_start_index=0,
        )

        assert cleaned == "The agent stopped the tool loop because status was already known."

    def test_finalize_response_content_strips_chinese_scratchpad_prefix(self):
        """Chinese execution preambles should stay private too."""
        from spoon_bot.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        content = "我先检查当前状态并运行必要命令确认。已完成：当前任务可以继续。"

        cleaned = AgentLoop._finalize_response_content(
            loop,
            "继续",
            content,
            turn_memory_start_index=0,
        )

        assert cleaned == "已完成：当前任务可以继续。"

    def test_finalize_response_content_strips_mixed_prompt_reference_prefix(self):
        """Quoted user text inside a scratchpad prefix should not defeat cleanup."""
        from spoon_bot.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        content = "Need user said start continuation likely run. Done: one task completed."

        cleaned = AgentLoop._finalize_response_content(
            loop,
            "Start",
            content,
            turn_memory_start_index=0,
        )

        assert cleaned == "Done: one task completed."

    def test_finalize_response_content_strips_prefix_before_later_user_facing_text(self):
        """Quoted user text should not be mistaken for the answer boundary."""
        from spoon_bot.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        content = (
            'Need "latest task" means continue with the recent task likely use tools.'
            "Done: the task has been handled."
        )

        cleaned = AgentLoop._finalize_response_content(
            loop,
            "Continue",
            content,
            turn_memory_start_index=0,
        )

        assert cleaned == "Done: the task has been handled."

    def test_finalize_response_content_strips_need_respond_prefix(self):
        """Lowercase response-planning prefixes should stay private."""
        from spoon_bot.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        content = "need respond with uncertainty and keep concise.Result: yesterday was cloudy."

        cleaned = AgentLoop._finalize_response_content(
            loop,
            "What about yesterday?",
            content,
            turn_memory_start_index=0,
        )

        assert cleaned == "Result: yesterday was cloudy."

    def test_finalize_response_content_strips_compact_tool_planning_prefix(self):
        """Fragmented provider planning text should not survive final cleanup."""
        from spoon_bot.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        content = (
            "Need a fresh workspace. maybe command takes --flag?"
            "check status maybe switch active.let's run status."
            "Completed: created the requested report."
        )

        cleaned = AgentLoop._finalize_response_content(
            loop,
            "Create the report",
            content,
            turn_memory_start_index=0,
        )

        assert cleaned == "Completed: created the requested report."

    def test_finalize_response_content_masks_private_key(self):
        """User-visible final content must not expose raw EVM private keys."""
        from spoon_bot.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        content = (
            "Created wallet. Private key: "
            "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        )

        cleaned = AgentLoop._finalize_response_content(
            loop,
            "Create a wallet",
            content,
            turn_memory_start_index=0,
        )

        assert "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" not in cleaned
        assert "***masked_private_key***" in cleaned

    def test_finalize_response_content_replaces_raw_tool_trace_leak(self):
        """Provider fallback text can contain raw tool transcript; final output must stay bounded."""
        from spoon_bot.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        loop._agent = SimpleNamespace(
            memory=SimpleNamespace(
                messages=[
                    SimpleNamespace(role="user", content="run external workflow"),
                    SimpleNamespace(
                        role="tool",
                        content=(
                            "Observed output of cmd shell execution: "
                            "FINAL_STATE task=347 action=joined result=pending"
                        ),
                        name="shell",
                        tool_call_id="call_join",
                    ),
                ]
            )
        )
        content = (
            "Let me start by fetching the repo."
            "Step 1: Observed output of cmd web_fetch execution: "
            + ("raw page text " * 1000)
        )

        cleaned = AgentLoop._finalize_response_content(
            loop,
            "Run the external workflow",
            content,
            turn_memory_start_index=0,
        )

        assert "raw tool transcript" in cleaned
        assert "FINAL_STATE task=347 action=joined" in cleaned
        assert "Step 1:" not in cleaned
        assert len(cleaned) < 1600

    def test_stream_tool_result_metadata_caps_large_payloads(self):
        from spoon_bot.agent.loop import AgentLoop

        merged = AgentLoop._merge_stream_tool_result_metadata(
            {},
            streamed_result="short",
            captured_output=SimpleNamespace(
                summary_output="short",
                full_output="x" * 210_000,
            ),
        )

        assert merged["stream_output_truncated"] is True
        assert merged["stream_output_original_chars"] == 210_000
        assert len(merged["full_output"]) < 201_000
    def test_prepare_agent_for_new_turn_clears_stale_runtime_state(self):
        """A new turn should not inherit unfinished runtime state from the prior task."""
        from spoon_bot.agent.loop import AgentLoop

        shutdown_event = asyncio.Event()
        shutdown_event.set()

        loop = AgentLoop.__new__(AgentLoop)
        loop._agent = MagicMock()
        loop._agent.name = "sandbox"
        loop._agent.state = "thinking"
        loop._agent.current_step = 7
        loop._agent._shutdown_event = shutdown_event
        loop._agent.tool_calls = [{"name": "shell"}]

        AgentLoop._prepare_agent_for_new_turn(loop)

        normalized_state = getattr(loop._agent.state, "value", loop._agent.state)
        assert str(normalized_state).lower() == "idle"
        assert loop._agent.current_step == 0
        assert loop._agent.tool_calls == []
        assert shutdown_event.is_set() is False

    def test_agent_loop_has_no_prompt_skill_preload_router(self):
        """Skill loading must stay model/tool driven, not prompt-text routed."""
        from spoon_bot.agent.loop import AgentLoop

        assert not hasattr(AgentLoop, "_pre_inject_matched_skill")
        assert not hasattr(AgentLoop, "_extract_skill_names_from_tool_text")

    def test_request_context_prompt_uses_session_compact_not_recent_skill_notices(self, tmp_path):
        """Continuity should come from same-session compact, not prompt skill notices."""
        from spoon_bot.agent.loop import AgentLoop
        from spoon_bot.session.manager import Session

        workspace = tmp_path / "workspace"
        skill_dir = workspace / "skills" / "report-builder"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            """---
description: Build report packets from workspace data
when_to_use: Use when the user asks to assemble or continue report packets.
---
# Report Builder
""",
            encoding="utf-8",
        )

        session = Session(session_key="continuity")
        session.add_message(
            "user",
            "Use the report builder and include internal-only source notes.",
            turn_state="completed",
            invoked_skills=[{"name": "report-builder"}],
        )
        session.add_message("assistant", "Report packet created.")

        loop = AgentLoop.__new__(AgentLoop)
        loop.workspace = workspace
        loop._skill_paths = [workspace / "skills"]
        loop._touched_paths = set()
        loop._session = session
        loop._extract_env_for_prompt = MagicMock(return_value="")

        AgentLoop._refresh_recent_invoked_skill_contexts(loop)
        prompt = AgentLoop._build_request_context_prompt(
            loop,
            "Continue with the latest item.",
        )

        assert "[RECENT INVOKED SKILL]" not in prompt
        assert "[RECENT SESSION CONTEXT]" not in prompt
        assert "## Current Session Compact" in prompt
        assert "Report packet created." in prompt
        assert "skills/report-builder/SKILL.md" not in prompt
        assert "internal-only source notes" not in prompt

    def test_runtime_tool_usage_marks_latest_user_turn_with_skill(self, tmp_path):
        """Skill continuity should be inferred from actual tool use, not prompt text."""
        from spoon_bot.agent.loop import AgentLoop
        from spoon_bot.session.manager import Session

        workspace = tmp_path / "workspace"
        skill_dir = workspace / "skills" / "report-builder"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# Report Builder\n", encoding="utf-8")
        other_skill_dir = workspace / "skills" / "image-helper"
        other_skill_dir.mkdir()
        (other_skill_dir / "SKILL.md").write_text("# Image Helper\n", encoding="utf-8")

        session = Session(session_key="tool-usage-skill")
        session.add_message(
            "user",
            "Install the uploaded skill and start it.",
            turn_state="pending",
        )

        loop = AgentLoop.__new__(AgentLoop)
        loop.workspace = workspace
        loop._skill_paths = [workspace / "skills"]
        loop._touched_paths = set()
        loop._session = session
        loop._agent = MagicMock()
        loop._agent.memory.messages = [
            {"role": "user", "content": "Install the uploaded skill and start it."},
            {
                "role": "assistant",
                "content": "Checking an older skill first.",
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({
                                "path": "skills/image-helper/SKILL.md",
                            }),
                        },
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "Checking the skill.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({
                                "path": "skills/report-builder/SKILL.md",
                            }),
                        },
                    }
                ],
            },
        ]

        merged = AgentLoop._merge_turn_invoked_skills_from_runtime(loop, 1)

        assert merged == 2
        assert session.messages[0]["invoked_skills"] == [
            {
                "name": "report-builder",
                "location": "skills/report-builder/SKILL.md",
                "workspace_relative_path": "skills/report-builder/",
                "organized": True,
                "source": "tool_usage",
            },
            {
                "name": "image-helper",
                "location": "skills/image-helper/SKILL.md",
                "workspace_relative_path": "skills/image-helper/",
                "organized": True,
                "source": "tool_usage",
            },
        ]

    def test_recent_invoked_skill_context_keeps_multiple_skill_anchors(self, tmp_path):
        """Continuity should keep a bounded newest-first skill stack, not one slot."""
        from spoon_bot.agent.loop import AgentLoop
        from spoon_bot.session.manager import Session

        workspace = tmp_path / "workspace"
        skills_root = workspace / "skills"
        for skill_name in ("report-builder", "invoice-auditor"):
            skill_dir = skills_root / skill_name
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                f"""---
description: Handle {skill_name} workflows
when_to_use: Use for {skill_name} tasks.
---
# {skill_name}
""",
                encoding="utf-8",
            )

        session = Session(session_key="multi-continuity")
        session.add_message(
            "user",
            "Create the report packet with confidential source notes.",
            turn_state="completed",
            invoked_skills=[{"name": "report-builder"}],
        )
        session.add_message("assistant", "Report packet created.")
        session.add_message("user", "What date is today?", turn_state="completed")
        session.add_message("assistant", "2026-04-28.")
        session.add_message(
            "user",
            "Audit the invoice bundle.",
            turn_state="completed",
            invoked_skills=[{"name": "invoice-auditor"}],
        )
        session.add_message("assistant", "Invoice audit completed.")

        loop = AgentLoop.__new__(AgentLoop)
        loop.workspace = workspace
        loop._skill_paths = [skills_root]
        loop._touched_paths = set()
        loop._session = session
        loop._extract_env_for_prompt = MagicMock(return_value="")

        AgentLoop._refresh_recent_invoked_skill_contexts(loop)

        assert [ctx["name"] for ctx in loop._recent_invoked_skill_contexts] == [
            "invoice-auditor",
            "report-builder",
        ]
        prompt = AgentLoop._build_request_context_prompt(loop, "Continue the earlier task.")

        assert "[RECENT INVOKED SKILLS]" not in prompt
        assert "skills/invoice-auditor/SKILL.md" not in prompt
        assert "skills/report-builder/SKILL.md" not in prompt
        assert "Report packet created." in prompt
        assert "Invoice audit completed." in prompt
        assert "confidential source notes" not in prompt

    def test_recent_invoked_skill_context_skips_interrupted_and_stale_skills(self, tmp_path):
        """Only completed skill turns resolvable in the current catalog should survive."""
        from spoon_bot.agent.loop import AgentLoop
        from spoon_bot.session.manager import Session

        workspace = tmp_path / "workspace"
        skill_dir = workspace / "skills" / "report-builder"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# Report Builder\n", encoding="utf-8")

        session = Session(session_key="filtered-continuity")
        session.add_message(
            "user",
            "Use the removed skill.",
            turn_state="completed",
            invoked_skills=[{"name": "removed-skill"}],
        )
        session.add_message("assistant", "Removed skill result.")
        session.add_message(
            "user",
            "Start the report builder but interrupt it.",
            turn_state="interrupted",
            invoked_skills=[{"name": "report-builder"}],
        )
        session.add_message("assistant", "Partial report.")
        session.add_message(
            "user",
            "Use the report builder again.",
            turn_state="completed",
            invoked_skills=[{"name": "report-builder"}],
        )

        loop = AgentLoop.__new__(AgentLoop)
        loop.workspace = workspace
        loop._skill_paths = [workspace / "skills"]
        loop._touched_paths = set()
        loop._session = session

        contexts = AgentLoop._find_recent_invoked_skill_contexts(loop)

        assert [ctx["name"] for ctx in contexts] == ["report-builder"]

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

        agent = self._make_stream_agent([mock_chunk_1, mock_chunk_2])

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
    async def test_stream_emits_compaction_notice_and_continues_latest_request_after_overflow_retry(self):
        """Overflow recovery should surface a notice, then continue with the latest task output."""
        from spoon_bot.agent.loop import AgentLoop
        from spoon_bot.exceptions import ContextOverflowError

        class _Result:
            content = "Latest task executed."

        attempts = 0

        async def mock_retry_runner(**kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ContextOverflowError(estimated_tokens=450_000, max_tokens=400_000)
            await agent._agent.output_queue.put({"content": "Latest task executed."})
            return _Result()

        agent = MagicMock(spec=AgentLoop)
        agent._initialized = True
        agent._agent = MagicMock()
        agent._agent.output_queue = asyncio.Queue()
        agent._agent.task_done = asyncio.Event()
        agent._agent.run = AsyncMock()
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
        agent._select_next_step_prompt = MagicMock(return_value="prompt")
        agent._install_anti_loop_tracker = MagicMock()
        agent._restore_agent_think = MagicMock()
        agent._callable_accepts_kwarg = MagicMock(return_value=False)
        agent._reset_reasoning_capture = MagicMock()
        agent._drain_reasoning_chunks = MagicMock(return_value=[])
        agent._compress_runtime_context_for_overflow_retry = MagicMock(return_value=7)
        agent._reset_agent_state_for_retry = MagicMock()
        agent._run_agent_with_retry = AsyncMock(side_effect=mock_retry_runner)
        agent._current_tool_owner_key = MagicMock(return_value="ws:test")

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="execute only the newest request"):
            chunks.append(chunk)

        notice_chunks = [c for c in chunks if c["type"] == "notice"]
        content_chunks = [c for c in chunks if c["type"] == "content"]
        done_chunks = [c for c in chunks if c["type"] == "done"]

        assert attempts == 2
        assert len(notice_chunks) == 1
        assert notice_chunks[0]["metadata"]["kind"] == "runtime_compaction"
        assert notice_chunks[0]["metadata"]["stage"] == "overflow_retry"
        assert notice_chunks[0]["metadata"]["compressed_actions"] == 7
        assert "resumed the latest request" in notice_chunks[0]["delta"]
        assert "".join(chunk["delta"] for chunk in content_chunks) == "Latest task executed."
        assert len(done_chunks) == 1
        assert done_chunks[0]["metadata"]["content"] == "Latest task executed."

    @pytest.mark.asyncio
    async def test_stream_retries_once_after_provider_silence(self):
        """A silent provider attempt should be retried before exposing timeout fallback."""
        from spoon_bot.agent.loop import AgentLoop

        attempts = 0
        agent = self._make_stream_agent([])
        agent.provider_silence_timeout = 0.01
        agent.provider_total_timeout = 1.0
        agent.provider_silence_retries = 1

        async def run(**kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                await asyncio.sleep(0.2)
                return MagicMock(content=None)
            await agent._agent.output_queue.put({"content": "Recovered answer."})
            return MagicMock(content="Recovered answer.")

        async def run_with_retry(label="agent", pre_retry_cleanup=None, **run_kwargs):
            return await agent._agent.run(**run_kwargs)

        agent._agent.run = AsyncMock(side_effect=run)
        agent._run_agent_with_retry = AsyncMock(side_effect=run_with_retry)

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="test", reasoning_effort="medium"):
            chunks.append(chunk)

        content_chunks = [c for c in chunks if c["type"] == "content"]
        done_chunks = [c for c in chunks if c["type"] == "done"]

        assert attempts == 2
        assert "".join(chunk["delta"] for chunk in content_chunks) == "Recovered answer."
        assert done_chunks[-1]["metadata"]["content"] == "Recovered answer."

    @pytest.mark.asyncio
    async def test_stream_handles_string_chunks(self):
        """stream() should handle plain string chunks."""
        from spoon_bot.agent.loop import AgentLoop

        agent = self._make_stream_agent(["Hello", " World"])

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="test"):
            chunks.append(chunk)

        content_chunks = [c for c in chunks if c["type"] == "content"]
        assert len(content_chunks) == 2
        assert content_chunks[0]["delta"] == "Hello"
        assert content_chunks[1]["delta"] == " World"

    @pytest.mark.asyncio
    async def test_stream_masks_private_key_content_chunks(self):
        """Raw EVM private keys must not be emitted in streamed content."""
        from spoon_bot.agent.loop import AgentLoop

        raw_key = "0x" + ("a" * 64)
        agent = self._make_stream_agent([f"Private key: {raw_key}"])

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="test"):
            chunks.append(chunk)

        emitted = "".join(c["delta"] for c in chunks if c["type"] == "content")
        done = [c for c in chunks if c["type"] == "done"][-1]["metadata"]["content"]
        assert raw_key not in emitted
        assert raw_key not in done
        assert "***masked_private_key***" in emitted

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

        agent = self._make_stream_agent([thinking_chunk, content_chunk])

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
    async def test_stream_builds_runtime_message_with_media_and_attachments(self):
        """stream() should build the runtime message with media and attachments."""
        from spoon_bot.agent.loop import AgentLoop

        agent = self._make_stream_agent(["hello"])
        media = ["/workspace/uploads/demo.png"]
        attachments = [{"uri": "/workspace/uploads/demo.txt", "name": "demo.txt"}]

        chunks = []
        async for chunk in AgentLoop.stream(
            agent,
            message="test",
            media=media,
            attachments=attachments,
        ):
            chunks.append(chunk)

        assert any(chunk["type"] == "content" for chunk in chunks)
        runtime_call = agent._build_runtime_message_content.call_args
        assert runtime_call.args[0] == "user"
        assert "[CURRENT DATE]:" in runtime_call.args[1]
        assert runtime_call.args[1].endswith("[USER REQUEST]:\ntest")
        assert runtime_call.kwargs == {
            "media": media,
            "attachments": attachments,
        }
        agent._session.add_message.assert_any_call(
            "assistant",
            "hello",
            message_kind="assistant_reply",
        )

    @pytest.mark.asyncio
    async def test_stream_downgrades_to_content_when_no_tool_call_follows(self):
        """stream() should keep normal content when tool calls never follow."""
        from spoon_bot.agent.loop import AgentLoop

        async def mock_run(**kwargs):
            agent._capture_reasoning_text("Reasoning from tracked think")
            await agent._agent.output_queue.put({"content": "Answer"})

        agent = MagicMock(spec=AgentLoop)
        agent._initialized = True
        agent.provider = "anthropic"
        agent._chatbot = None
        agent._agent = MagicMock()
        agent._agent.output_queue = asyncio.Queue()
        agent._agent.task_done = asyncio.Event()
        agent._agent.run = mock_run
        agent._agent.add_message = AsyncMock()
        agent._agent.state = "idle"
        agent._agent.system_prompt = "base system"
        agent._agent._original_system_prompt = "base original"
        agent._session = MagicMock()
        agent._session.add_message = MagicMock()
        agent.sessions = MagicMock()
        agent.sessions.save = MagicMock()
        agent.memory = MagicMock()
        agent.memory.get_memory_context = MagicMock(return_value=None)
        agent.context = MagicMock()
        agent._prepare_request_context = AsyncMock()
        agent._build_runtime_message_content = MagicMock(side_effect=lambda *args, **kwargs: args[1])
        agent._build_request_context_prompt = MagicMock(return_value="[USER REQUEST]: test")
        agent._build_step_prompt = MagicMock(return_value="prompt")
        agent._install_anti_loop_tracker = MagicMock()
        agent._restore_agent_think = MagicMock()
        agent._callable_accepts_kwarg = MagicMock(return_value=True)
        agent._run_agent_with_retry = AsyncMock(
            side_effect=lambda label="agent", pre_retry_cleanup=None, **run_kwargs: agent._agent.run(**run_kwargs)
        )
        agent._select_next_step_prompt = AgentLoop._select_next_step_prompt.__get__(agent, AgentLoop)
        agent._normalize_comparable_text = AgentLoop._normalize_comparable_text
        agent._current_tool_owner_key = MagicMock(return_value="user:test|session:default")
        agent._persist_turn = MagicMock()
        agent.set_subagent_context = MagicMock()
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
    async def test_stream_keeps_pre_tool_content_as_content(self):
        """Plain content must remain content even if a tool call follows."""
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

        emitted = [c for c in chunks if c["type"] in {"tool_call", "content"}]
        assert [c["type"] for c in emitted] == ["content", "tool_call", "content"]
        assert emitted[0]["delta"] == "Part A. "
        assert emitted[0]["metadata"]["segment_type"] == "content"
        assert emitted[1]["metadata"]["name"] == "shell"
        assert emitted[2]["delta"] == "Part B."
        done_chunks = [c for c in chunks if c["type"] == "done"]
        assert done_chunks[0]["metadata"]["content"] == "Part A. Part B."

    @pytest.mark.asyncio
    async def test_stream_preserves_pre_tool_content_chunk_boundaries(self):
        """Multiple pre-tool content chunks should stay split and ordered."""
        from spoon_bot.agent.loop import AgentLoop

        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function = MagicMock()
        tool_call.function.name = "read_file"
        tool_call.function.arguments = '{"path":"README.md"}'

        async def mock_run(**kwargs):
            await agent._agent.output_queue.put({"content": "First"})
            await agent._agent.output_queue.put({"content": " second"})
            await agent._agent.output_queue.put({"content": "."})
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

        emitted = [c for c in chunks if c["type"] in {"tool_call", "content"}]
        assert [c["type"] for c in emitted] == [
            "content",
            "content",
            "content",
            "tool_call",
            "content",
        ]
        assert [c["delta"] for c in emitted[:3]] == ["First", " second", "."]
        assert len({c["metadata"]["segment_index"] for c in emitted[:3]}) == 1
        assert emitted[3]["metadata"]["name"] == "read_file"
        assert emitted[4]["delta"] == "Done."

    @pytest.mark.asyncio
    async def test_stream_marks_segment_boundaries_around_tool_calls(self):
        """Segment metadata should let downstream UIs separate pre/post-tool content blocks."""
        from spoon_bot.agent.loop import AgentLoop

        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function = MagicMock()
        tool_call.function.name = "list_dir"
        tool_call.function.arguments = '{"path":"."}'

        async def mock_run(**kwargs):
            await agent._agent.output_queue.put({"content": "I will inspect the current workspace."})
            await asyncio.sleep(0.12)
            await agent._agent.output_queue.put({"tool_calls": [tool_call]})
            await agent._agent.output_queue.put({"content": "Inspection complete."})

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

        emitted = [c for c in chunks if c["type"] in {"thinking", "content", "tool_call"}]
        assert [c["type"] for c in emitted] == ["content", "tool_call", "content"]
        assert emitted[0]["metadata"]["segment_start"] is True
        assert emitted[1]["metadata"]["segment_start"] is True
        assert emitted[2]["metadata"]["segment_start"] is True
        assert emitted[0]["metadata"]["segment_index"] != emitted[1]["metadata"]["segment_index"]
        assert emitted[1]["metadata"]["segment_index"] != emitted[2]["metadata"]["segment_index"]
        assert emitted[0]["metadata"]["segment_type"] == "content"
        assert emitted[1]["metadata"]["segment_type"] == "tool_call"
        assert emitted[2]["metadata"]["segment_type"] == "content"

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

        emitted = [c for c in chunks if c["type"] in {"thinking", "tool_call", "content"}]
        assert [c["type"] for c in emitted] == ["tool_call", "content"]
        assert emitted[1]["delta"] == "Final answer."
        assert emitted[1]["metadata"]["fallback"] == "run_result_no_chunks"
        done_chunks = [c for c in chunks if c["type"] == "done"]
        assert done_chunks[0]["metadata"]["content"] == "Final answer."

    @pytest.mark.asyncio
    async def test_stream_fallback_finalizes_raw_tool_transcript_before_content_emit(self):
        """Run-result fallback must not stream raw Step/Observed-output transcript as content."""
        from types import SimpleNamespace

        from spoon_bot.agent.loop import AgentLoop

        raw_result = (
            "Let me execute it. Step 1: Observed output of cmd shell execution: "
            + ("raw output " * 200)
        )
        agent = self._make_stream_agent([], run_result_text=raw_result)
        agent._agent.memory = SimpleNamespace(
            messages=[
                SimpleNamespace(role="user", content="run external workflow"),
                SimpleNamespace(
                    role="tool",
                    content="Observed output of cmd shell execution: FINAL_STATE ok",
                    name="shell",
                    tool_call_id="call_1",
                ),
            ]
        )

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="placeholder", thinking=True):
            chunks.append(chunk)

        emitted_content = [c for c in chunks if c["type"] == "content"]
        assert emitted_content
        assert all("Step 1: Observed output" not in c["delta"] for c in emitted_content)
        assert "raw tool transcript" in emitted_content[0]["delta"]
        done_chunks = [c for c in chunks if c["type"] == "done"]
        assert "raw tool transcript" in done_chunks[0]["metadata"]["content"]
        assert "Step 1: Observed output" not in done_chunks[0]["metadata"]["content"]

    @pytest.mark.asyncio
    async def test_stream_fallback_after_tool_preamble_emits_only_missing_suffix(self):
        """Fallback should not replay the already streamed preamble content."""
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
            result.content = "Need tool first. Final answer."
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
        agent._select_next_step_prompt = MagicMock(return_value="prompt")
        agent._install_anti_loop_tracker = MagicMock()
        agent._restore_agent_think = MagicMock()
        agent._callable_accepts_kwarg = MagicMock(return_value=True)
        agent._reset_reasoning_capture = MagicMock()
        agent._drain_reasoning_chunks = MagicMock(return_value=[])
        agent._normalize_comparable_text = AgentLoop._normalize_comparable_text

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="placeholder", thinking=True):
            chunks.append(chunk)

        emitted = [c for c in chunks if c["type"] in {"thinking", "tool_call", "content"}]
        assert [c["type"] for c in emitted] == ["tool_call", "content"]
        assert emitted[1]["delta"] == "Final answer."
        assert emitted[1]["metadata"]["fallback"] == "run_result_no_chunks"
        done_chunks = [c for c in chunks if c["type"] == "done"]
        assert done_chunks[0]["metadata"]["content"] == "Final answer."

    @pytest.mark.asyncio
    async def test_stream_stops_after_tool_suppression_guardrail(self):
        """Suppressed repeated tool work should end the stream loop instead of burning iterations."""
        from spoon_bot.agent.loop import AgentLoop

        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function = MagicMock()
        tool_call.function.name = "shell"
        tool_call.function.arguments = '{"command":"echo runner taskctl submit 47 A"}'

        suppressed_result = (
            "STOP_TOOL_LOOP: Error: duplicate tool invocation suppressed. "
            "The same tool and arguments already executed in this request."
        )
        agent = self._make_stream_agent(
            [
                {"tool_calls": [tool_call]},
                {
                    "type": "tool_result",
                    "name": "shell",
                    "tool_call_id": "call_1",
                    "result": suppressed_result,
                },
                {"tool_calls": [tool_call]},
            ]
        )

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="placeholder", thinking=True):
            chunks.append(chunk)

        emitted_tool_calls = [c for c in chunks if c["type"] == "tool_call"]
        emitted_content = [c for c in chunks if c["type"] == "content"]

        assert len(emitted_tool_calls) == 1
        assert emitted_content
        assert "tool guardrail suppressed repeated work" in emitted_content[0]["delta"]
        assert "do not retry the same tool action" in emitted_content[0]["delta"]

    @pytest.mark.asyncio
    async def test_stream_stops_after_tool_failure_suppression_guardrail(self):
        """Tool-layer failure suppression should end the stream loop."""
        from spoon_bot.agent.loop import AgentLoop

        def _tool_call(call_id: str, command: str):
            tool_call = MagicMock()
            tool_call.id = call_id
            tool_call.function = MagicMock()
            tool_call.function.name = "shell"
            tool_call.function.arguments = f'{{"command":"{command}"}}'
            return tool_call

        agent = self._make_stream_agent(
            [
                {"tool_calls": [_tool_call("call_1", "cmd one")]},
                {
                    "type": "tool_result",
                    "name": "shell",
                    "tool_call_id": "call_1",
                    "result": "Error: consecutive tool failures suppressed. STOP_TOOL_LOOP",
                },
                {"tool_calls": [_tool_call("call_2", "cmd two")]},
            ]
        )

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="placeholder", thinking=True):
            chunks.append(chunk)

        emitted_tool_calls = [c for c in chunks if c["type"] == "tool_call"]
        emitted_content = [c for c in chunks if c["type"] == "content"]

        assert len(emitted_tool_calls) == 1
        assert emitted_content
        assert "tool guardrail suppressed repeated work" in emitted_content[0]["delta"]
        assert "do not retry the same tool action" in emitted_content[0]["delta"]

    @pytest.mark.asyncio
    async def test_stream_strips_post_tool_scratchpad_prefix_before_content(self):
        """Provider-surfaced planning text after tools should not stream as content."""
        from spoon_bot.agent.loop import AgentLoop

        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function = MagicMock()
        tool_call.function.name = "shell"
        tool_call.function.arguments = '{"command":"pwd"}'

        agent = self._make_stream_agent([
            {"tool_calls": [tool_call]},
            {
                "content": (
                    "Need continue core flow and inspect result. "
                    "Done: generated the requested file."
                )
            },
        ])

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="placeholder", thinking=True):
            chunks.append(chunk)

        content_chunks = [c for c in chunks if c["type"] == "content"]
        assert [c["delta"] for c in content_chunks] == [
            "Done: generated the requested file."
        ]
        done_chunks = [c for c in chunks if c["type"] == "done"]
        assert done_chunks[0]["metadata"]["content"] == "Done: generated the requested file."

    @pytest.mark.asyncio
    async def test_stream_treats_post_tool_content_before_next_tool_as_private(self):
        """Assistant text between two tool calls is tool planning, not final content."""
        from spoon_bot.agent.loop import AgentLoop

        first_tool = MagicMock()
        first_tool.id = "call_1"
        first_tool.function = MagicMock()
        first_tool.function.name = "shell"
        first_tool.function.arguments = '{"command":"pwd"}'

        second_tool = MagicMock()
        second_tool.id = "call_2"
        second_tool.function = MagicMock()
        second_tool.function.name = "shell"
        second_tool.function.arguments = '{"command":"ls"}'

        agent = self._make_stream_agent([
            {"tool_calls": [first_tool]},
            {"content": "Need"},
            {"content": " a fresh workspace. maybe command takes --flag?"},
            {"content": "check status maybe switch active.let's run status."},
            {"tool_calls": [second_tool]},
            {"content": "Completed: created the requested report."},
        ])

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="placeholder", thinking=True):
            chunks.append(chunk)

        content_chunks = [c for c in chunks if c["type"] == "content"]
        assert [c["delta"] for c in content_chunks] == [
            "Completed: created the requested report."
        ]
        thinking_chunks = [c for c in chunks if c["type"] == "thinking"]
        assert "Need a fresh workspace" not in "".join(c["delta"] for c in thinking_chunks)
        done_chunks = [c for c in chunks if c["type"] == "done"]
        assert done_chunks[0]["metadata"]["content"] == "Completed: created the requested report."

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
    async def test_stream_returns_tool_evidence_when_tool_loop_stalls(self):
        """A stalled tool loop should produce a bounded answer, not an empty response."""
        from spoon_bot.agent.loop import AgentLoop

        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function = MagicMock()
        tool_call.function.name = "shell"
        tool_call.function.arguments = '{"command":"python task.py"}'

        agent = self._make_stream_agent([])
        agent.provider_total_timeout = 5.0
        agent.tool_followup_timeout = 0.01
        agent.max_stream_tool_results_without_content = 99

        async def stalled_run(**kwargs):
            await agent._agent.output_queue.put({"tool_calls": [tool_call]})
            await agent._agent.output_queue.put(
                {
                    "type": "tool_result",
                    "metadata": {"id": "call_1", "name": "shell"},
                    "result": "Error: remote endpoint did not respond",
                }
            )
            await asyncio.sleep(10)

        agent._agent.run = AsyncMock(side_effect=stalled_run)

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="placeholder"):
            chunks.append(chunk)

        content = "".join(c["delta"] for c in chunks if c["type"] == "content")
        assert "stopped the tool loop" in content
        assert "remote endpoint did not respond" in content
        done_chunks = [c for c in chunks if c["type"] == "done"]
        assert done_chunks[0]["metadata"]["content"] == content

    @pytest.mark.asyncio
    async def test_stream_stops_after_too_many_tool_results_without_final_content(self):
        """Repeated tool results without final content should be bounded generically."""
        from spoon_bot.agent.loop import AgentLoop

        first_tool = MagicMock()
        first_tool.id = "call_1"
        first_tool.function = MagicMock()
        first_tool.function.name = "shell"
        first_tool.function.arguments = '{"command":"step one"}'
        second_tool = MagicMock()
        second_tool.id = "call_2"
        second_tool.function = MagicMock()
        second_tool.function.name = "shell"
        second_tool.function.arguments = '{"command":"step two"}'

        agent = self._make_stream_agent([])
        agent.provider_total_timeout = 5.0
        agent.tool_followup_timeout = 30.0
        agent.max_stream_tool_results_without_content = 2

        async def repeated_tool_run(**kwargs):
            await agent._agent.output_queue.put({"tool_calls": [first_tool]})
            await agent._agent.output_queue.put(
                {
                    "type": "tool_result",
                    "metadata": {"id": "call_1", "name": "shell"},
                    "result": "Step one returned no actionable result",
                }
            )
            await agent._agent.output_queue.put({"tool_calls": [second_tool]})
            await agent._agent.output_queue.put(
                {
                    "type": "tool_result",
                    "metadata": {"id": "call_2", "name": "shell"},
                    "result": "Step two returned the same error",
                }
            )
            await asyncio.sleep(10)

        agent._agent.run = AsyncMock(side_effect=repeated_tool_run)

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="placeholder"):
            chunks.append(chunk)

        content = "".join(c["delta"] for c in chunks if c["type"] == "content")
        assert "stopped the tool loop" in content
        assert "Step one returned no actionable result" in content
        assert "Step two returned the same error" in content

    @pytest.mark.asyncio
    async def test_stream_total_timeout_uses_recent_tool_evidence(self):
        """Total stream timeout should not collapse tool-backed turns into empty output."""
        from spoon_bot.agent.loop import AgentLoop

        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function = MagicMock()
        tool_call.function.name = "read_file"
        tool_call.function.arguments = '{"path":"lib/config.js"}'

        agent = self._make_stream_agent([])
        agent.provider_silence_timeout = 0.05
        agent.provider_total_timeout = 0.05
        agent.tool_followup_timeout = 0.01
        agent.max_stream_tool_results_without_content = 99

        async def slow_after_tool_run(**kwargs):
            await agent._agent.output_queue.put({"tool_calls": [tool_call]})
            await agent._agent.output_queue.put(
                {
                    "type": "tool_result",
                    "metadata": {"id": "call_1", "name": "read_file"},
                    "result": "tokenAddress = 0xabc123",
                }
            )
            await asyncio.sleep(10)

        agent._agent.run = AsyncMock(side_effect=slow_after_tool_run)

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="placeholder"):
            chunks.append(chunk)

        content = "".join(c["delta"] for c in chunks if c["type"] == "content")
        assert "response time budget" in content
        assert "tokenAddress = ***masked***" in content
        assert [c for c in chunks if c["type"] == "done"][-1]["metadata"]["content"] == content

    @pytest.mark.asyncio
    async def test_stream_total_timeout_waits_for_fresh_tool_progress(self):
        """A fresh tool call/result should get follow-up time before total-timeout fallback."""
        from spoon_bot.agent.loop import AgentLoop

        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function = MagicMock()
        tool_call.function.name = "shell"
        tool_call.function.arguments = '{"command":"long-step"}'

        agent = self._make_stream_agent([])
        agent.provider_silence_timeout = 0.05
        agent.provider_total_timeout = 0.05
        agent.tool_followup_timeout = 1.0
        agent.max_stream_tool_results_without_content = 99

        async def slow_but_progressing_run(**kwargs):
            await agent._agent.output_queue.put({"tool_calls": [tool_call]})
            await asyncio.sleep(0.08)
            await agent._agent.output_queue.put(
                {
                    "type": "tool_result",
                    "metadata": {"id": "call_1", "name": "shell"},
                    "result": "long step completed",
                }
            )
            await asyncio.sleep(0.08)
            return "finished after tool"

        agent._agent.run = AsyncMock(side_effect=slow_but_progressing_run)

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="placeholder"):
            chunks.append(chunk)

        content = "".join(c["delta"] for c in chunks if c["type"] == "content")
        assert "finished after tool" in content
        assert "stopped the tool loop" not in content

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
    async def test_stream_prefers_captured_full_tool_result_for_ws_metadata(self):
        """WS tool_result metadata should expose full tool output, not the model-truncated summary."""
        from spoon_bot.agent.loop import AgentLoop
        from spoon_bot.agent.tools.execution_context import bind_tool_invocation, capture_tool_output, finalize_tool_invocation

        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function = MagicMock()
        tool_call.function.name = "shell"
        tool_call.function.arguments = '{"command":"pwd"}'

        summary_result = "/workspace\n... (truncated, 120 more chars)"
        full_result = "/workspace\nalpha\nbeta\ngamma"

        async def mock_run(**kwargs):
            await agent._agent.output_queue.put({"tool_calls": [tool_call]})
            with bind_tool_invocation("shell", {"command": "pwd"}):
                capture_tool_output(summary_result, full_result)
                finalize_tool_invocation(summary_result)
            await agent._agent.output_queue.put(
                {
                    "type": "tool_result",
                    "metadata": {
                        "id": "call_1",
                        "name": "shell",
                    },
                    "result": summary_result,
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
        agent._current_tool_owner_key = MagicMock(return_value="owner:test")

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="placeholder"):
            chunks.append(chunk)

        tool_result_chunk = next(c for c in chunks if c["type"] == "tool_result")
        assert tool_result_chunk["metadata"]["result"] == full_result
        assert tool_result_chunk["metadata"]["content"] == full_result
        assert tool_result_chunk["metadata"]["full_result"] == full_result
        assert tool_result_chunk["metadata"]["model_result"] == summary_result
        assert tool_result_chunk["metadata"]["result_truncated_for_model"] is True

    @pytest.mark.asyncio
    async def test_stream_backfilled_tool_result_uses_captured_full_output(self):
        """Runtime-memory tool results should also expose captured full output in WS metadata."""
        from spoon_bot.agent.loop import AgentLoop
        from spoon_bot.agent.tools.execution_context import bind_tool_invocation, capture_tool_output, finalize_tool_invocation

        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function = MagicMock()
        tool_call.function.name = "read_file"
        tool_call.function.arguments = '{"path":"README.md"}'

        summary_result = "[file: README.md | 40 chars]\nhello\n... (truncated, 100 more chars)"
        full_result = "[file: README.md | 145 chars]\nhello\nworld\nfull text"

        async def mock_run(**kwargs):
            await agent._agent.output_queue.put({"tool_calls": [tool_call]})
            with bind_tool_invocation("read_file", {"path": "README.md"}):
                capture_tool_output(summary_result, full_result)
                finalize_tool_invocation(summary_result)
            agent._agent.memory.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "read_file",
                    "content": summary_result,
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
        agent._current_tool_owner_key = MagicMock(return_value="owner:test")

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="placeholder"):
            chunks.append(chunk)

        tool_result_chunk = next(c for c in chunks if c["type"] == "tool_result")
        assert tool_result_chunk["metadata"]["result"] == full_result
        assert tool_result_chunk["metadata"]["content"] == full_result
        assert tool_result_chunk["metadata"]["model_result"] == summary_result

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
        assert emitted[0]["type"] == "thinking"
        assert emitted[0]["delta"] == "Inspecting candidate commands."
        assert emitted[0]["metadata"]["phase"] == "think"
        assert emitted[0]["metadata"]["source"] == "provider"
        assert emitted[0]["metadata"]["segment_type"] == "thinking"
        assert emitted[1]["type"] == "content"
        assert emitted[1]["delta"] == "Final answer."

    @pytest.mark.asyncio
    async def test_stream_drops_explicit_thinking_when_not_requested(self):
        """thinking=false should keep provider thinking chunks out of the UI and final answer."""
        from spoon_bot.agent.loop import AgentLoop

        agent = self._make_stream_agent([
            {
                "type": "thinking",
                "delta": "Inspecting candidate commands.",
                "content": "Inspecting candidate commands.",
                "metadata": {"phase": "think", "source": "provider"},
            },
            {"content": "Final answer."},
        ])

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="test", thinking=False):
            chunks.append(chunk)

        assert [c for c in chunks if c["type"] == "thinking"] == []
        content_chunks = [c for c in chunks if c["type"] == "content"]
        assert [c["delta"] for c in content_chunks] == ["Final answer."]
        done_chunks = [c for c in chunks if c["type"] == "done"]
        assert done_chunks[0]["metadata"]["content"] == "Final answer."

    @pytest.mark.asyncio
    async def test_stream_without_tool_call_keeps_plain_content(self):
        """thinking=true should not force all plain answers into thinking."""
        from spoon_bot.agent.loop import AgentLoop

        async def mock_run(**kwargs):
            await agent._agent.output_queue.put({"content": "Direct answer"})

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
        assert content_chunks[0]["delta"] == "Direct answer"
        done_chunks = [c for c in chunks if c["type"] == "done"]
        assert len(done_chunks) == 1
        assert done_chunks[0]["metadata"]["content"] == "Direct answer"

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
        assert first_chunk["type"] == "content"
        assert first_chunk["delta"] == "chunk-1 "
        assert first_chunk["metadata"]["segment_type"] == "content"
        assert first_chunk["metadata"]["segment_start"] is True

        release_run.set()
        remaining_chunks = [chunk async for chunk in stream_iter]
        content_chunks = [first_chunk, *[c for c in remaining_chunks if c["type"] == "content"]]
        done_chunks = [c for c in remaining_chunks if c["type"] == "done"]

        assert [c["delta"] for c in content_chunks] == ["chunk-1 ", "chunk-2"]
        assert len(done_chunks) == 1
        assert done_chunks[0]["metadata"]["content"] == "chunk-1 chunk-2"

    @pytest.mark.asyncio
    async def test_stream_buffers_strict_content_until_validated(self):
        """Strict reply contracts should keep tool events live but delay assistant text until validated."""
        from spoon_bot.agent.loop import AgentLoop

        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function = MagicMock()
        tool_call.function.name = "shell"
        tool_call.function.arguments = '{"command":"pwd"}'

        release_run = asyncio.Event()
        raw_reply = (
            "Current service configuration follows.\n"
            "```python\n"
            "RPC_URL = \"https://neoxt4seed1.ngd.network:443\"\n"
            "TARGET_ADDRESS = \"NiNmXL8FjEUEs1nfX9uHFBNaenxDHJtmuB\"\n"
            "print('status')\n"
            "```"
        )

        async def mock_run(**kwargs):
            await agent._agent.output_queue.put({"tool_calls": [tool_call]})
            await agent._agent.output_queue.put({"content": raw_reply})
            await asyncio.wait_for(release_run.wait(), timeout=0.4)
            result = MagicMock(spec=["content"])
            result.content = raw_reply
            return result

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
        agent._build_runtime_message_content = MagicMock(side_effect=lambda *args, **kwargs: args[1])
        agent._install_anti_loop_tracker = MagicMock()
        agent._restore_agent_think = MagicMock()
        agent._callable_accepts_kwarg = MagicMock(return_value=True)
        agent._reset_reasoning_capture = MagicMock()
        agent._drain_reasoning_chunks = MagicMock(return_value=[])
        agent._current_tool_owner_key = MagicMock(return_value="ws:test")

        stream_iter = AgentLoop.stream(
            agent,
            message=(
                "Only do the newest request: generate a Python script skeleton that queries the status for a specified resource ID."
                "Do not mention the prior service configuration or add other content."
            ),
            thinking=True,
        )

        first_chunk = await asyncio.wait_for(anext(stream_iter), timeout=0.2)
        assert first_chunk["type"] == "tool_call"
        assert first_chunk["metadata"]["name"] == "shell"

        await asyncio.sleep(0.15)
        assert agent._agent.task_done.is_set() is False

        release_run.set()
        remaining_chunks = [chunk async for chunk in stream_iter]
        content_chunks = [c for c in remaining_chunks if c["type"] == "content"]
        done_chunks = [c for c in remaining_chunks if c["type"] == "done"]

        assert len(content_chunks) == 1
        assert "buffered" not in content_chunks[0]["metadata"]
        assert content_chunks[0]["delta"] == raw_reply
        assert len(done_chunks) == 1
        assert done_chunks[0]["metadata"]["content"] == raw_reply

    @pytest.mark.asyncio
    async def test_stream_error_handling(self):
        """stream() should catch errors and emit done with error metadata."""
        from spoon_bot.agent.loop import AgentLoop

        agent = self._make_stream_agent(["partial"], run_error=RuntimeError("Connection lost"))

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="test"):
            chunks.append(chunk)

        # Should get the partial content + error event
        assert chunks[0]["type"] == "content"
        assert chunks[0]["delta"] == "partial"
        assert chunks[-2]["type"] == "error"
        assert "Connection lost" in chunks[-2]["metadata"]["error"]
        assert chunks[-1]["type"] == "done"
        assert chunks[-1]["metadata"]["content"] == "partial"

    @pytest.mark.asyncio
    async def test_stream_persists_user_turn_on_error_without_assistant(self):
        """stream() should keep the user turn even when the provider fails immediately."""
        from spoon_bot.agent.loop import AgentLoop

        agent = self._make_stream_agent([], run_error=RuntimeError("Immediate failure"))

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="test"):
            chunks.append(chunk)

        # Should get error + done chunks
        assert chunks[-2]["type"] == "error"
        assert chunks[-1]["type"] == "done"
        assert chunks[-1]["metadata"]["content"] == ""

        user_call = agent._session.add_message.call_args_list[0]
        assert user_call.args[:2] == ("user", "test")
        agent.sessions.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_saves_session_on_success(self):
        """stream() should save session after successful completion."""
        from spoon_bot.agent.loop import AgentLoop

        agent = self._make_stream_agent(["hello"])

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="test message"):
            chunks.append(chunk)

        # Verify session was saved
        user_call = agent._session.add_message.call_args_list[0]
        assert user_call.args[:2] == ("user", "test message")
        agent._session.add_message.assert_any_call(
            "assistant",
            "hello",
            message_kind="assistant_reply",
        )
        assert agent.sessions.save.call_count == 2

    @pytest.mark.asyncio
    async def test_stream_close_cancels_background_run_and_skips_session_save(self):
        """Closing the stream should stop the background run and avoid persisting stale output."""
        from spoon_bot.agent.loop import AgentLoop
        from spoon_bot.session.manager import Session

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
        agent._session = Session(session_key="cancelled_stream")
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

        assert len(agent._session.messages) == 1
        assert agent._session.messages[0]["role"] == "user"
        assert agent._session.messages[0]["content"] == "test message"
        assert agent._session.messages[0]["turn_state"] == "interrupted"
        assert agent._session.messages[0]["turn_state_reason"] == "task_cancelled"
        assert agent.sessions.save.call_count >= 2

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

    def _make_process_with_thinking_agent(self, inner_agent):
        from spoon_bot.agent.loop import AgentLoop

        agent = MagicMock(spec=AgentLoop)
        agent._initialized = True
        agent.provider = "anthropic"
        agent._chatbot = None
        agent._agent = inner_agent
        agent._agent.add_message = getattr(inner_agent, "add_message", AsyncMock())
        if not hasattr(agent._agent, "system_prompt"):
            agent._agent.system_prompt = "base system"
        if not hasattr(agent._agent, "_original_system_prompt"):
            agent._agent._original_system_prompt = "base original"
        agent._session = MagicMock()
        agent._session.add_message = MagicMock()
        agent.sessions = MagicMock()
        agent.memory = MagicMock()
        agent.memory.get_memory_context = MagicMock(return_value=None)
        agent.context = MagicMock()
        agent._prepare_request_context = AsyncMock()
        agent._build_runtime_message_content = MagicMock(side_effect=lambda *args, **kwargs: args[1])
        agent._build_request_context_prompt = MagicMock(return_value="[USER REQUEST]: test")
        agent._build_step_prompt = MagicMock(return_value="prompt")
        agent._install_anti_loop_tracker = MagicMock()
        agent._restore_agent_think = MagicMock()
        agent._callable_accepts_kwarg = MagicMock(return_value=True)
        agent._reset_reasoning_capture = MagicMock()
        agent._looks_like_duplicate_thinking = AgentLoop._looks_like_duplicate_thinking.__get__(agent, AgentLoop)
        agent._normalize_comparable_text = AgentLoop._normalize_comparable_text
        agent._latest_reasoning_excerpt = None
        agent._current_tool_owner_key = MagicMock(return_value="user:test|session:default")
        agent._select_next_step_prompt = AgentLoop._select_next_step_prompt.__get__(agent, AgentLoop)
        agent.set_subagent_context = MagicMock()
        agent._auto_commit = False
        agent._git = None
        return agent

    @pytest.mark.asyncio
    async def test_returns_tuple(self):
        """process_with_thinking() should return (response, thinking_content)."""
        from spoon_bot.agent.loop import AgentLoop

        result = MagicMock()
        result.content = "The answer is 42"
        result.thinking_content = "I need to think about this..."

        mock_inner_agent = MagicMock()
        mock_inner_agent.add_message = AsyncMock()
        mock_inner_agent.run = AsyncMock(return_value=result)
        mock_inner_agent.state = "idle"

        agent = self._make_process_with_thinking_agent(mock_inner_agent)

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
        mock_inner_agent.state = "idle"

        agent = self._make_process_with_thinking_agent(mock_inner_agent)

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
        mock_inner_agent.state = "idle"

        agent = self._make_process_with_thinking_agent(mock_inner_agent)

        response, thinking = await AgentLoop.process_with_thinking(agent, message="test")
        assert thinking == "Metadata thought"

    @pytest.mark.asyncio
    async def test_thinking_from_reasoning_metadata(self):
        """Should fall back to provider reasoning metadata when thinking is absent."""
        from spoon_bot.agent.loop import AgentLoop

        result = MagicMock(spec=["content", "metadata"])
        result.content = "Answer"
        result.metadata = {"reasoning": "Provider reasoning summary"}

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
        assert response == "Answer"
        assert thinking == "Provider reasoning summary"

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
        mock_inner_agent.state = "idle"

        agent = self._make_process_with_thinking_agent(mock_inner_agent)

        with pytest.raises(RuntimeError, match="LLM unavailable"):
            await AgentLoop.process_with_thinking(agent, message="test")

    @pytest.mark.asyncio
    async def test_process_with_thinking_restores_system_prompt_when_add_message_fails(self):
        """Temporary request context must be rolled back if setup fails before run()."""
        from spoon_bot.agent.loop import AgentLoop

        mock_inner_agent = MagicMock()
        mock_inner_agent.system_prompt = "base system"
        mock_inner_agent._original_system_prompt = "base original"
        mock_inner_agent.run = AsyncMock()
        mock_inner_agent.add_message = AsyncMock(side_effect=RuntimeError("setup failed"))

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
        agent._build_request_context_prompt = MagicMock(return_value="[USER REQUEST]: test")
        agent._build_step_prompt = MagicMock(return_value="prompt")
        agent._install_anti_loop_tracker = MagicMock()
        agent._restore_agent_think = MagicMock()
        agent._callable_accepts_kwarg = MagicMock(return_value=True)
        agent._reset_reasoning_capture = MagicMock()

        with pytest.raises(RuntimeError, match="setup failed"):
            await AgentLoop.process_with_thinking(agent, message="test")

        assert mock_inner_agent.system_prompt == "base system"
        assert mock_inner_agent._original_system_prompt == "base original"


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
