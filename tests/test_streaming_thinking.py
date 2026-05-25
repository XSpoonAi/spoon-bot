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


class TestStreamErrorFallback:
    def test_provider_error_replaces_pre_tool_preamble_when_tools_ran(self):
        from spoon_bot.agent.loop import AgentLoop

        assert AgentLoop._should_replace_stream_error_preamble(
            "I'll check",
            saw_tool_call=True,
            saw_content_after_tool_call=False,
        )
        assert not AgentLoop._should_replace_stream_error_preamble(
            "Completed from tool evidence.",
            saw_tool_call=True,
            saw_content_after_tool_call=True,
        )


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
        assert "[HISTORY VERIFICATION]:" in prompt
        assert "search_history(scope='current')" in prompt

    def test_build_request_context_prompt_marks_prior_action_fact_check_required(self):
        """Prior-action follow-ups should not use live tools as prior-action proof."""
        from spoon_bot.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        loop.workspace = Path("/workspace")
        loop._extract_env_for_prompt = MagicMock(return_value="")

        prompt = AgentLoop._build_request_context_prompt(
            loop,
            "不是啊，你刚刚不是已经运行过很多次，你忘记了吗？",
        )

        assert "[CURRENT SESSION FACT CHECK REQUIRED]:" in prompt
        assert "External or live-state tools" in prompt
        assert "long-term memory" in prompt
        assert "search_history(scope='current')" in prompt

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

    def test_build_request_context_prompt_does_not_add_github_skill_route(self):
        """GitHub skill install intent stays in tool contracts, not prompt routes."""
        from spoon_bot.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        loop.workspace = Path("/workspace")
        loop._extract_env_for_prompt = MagicMock(return_value="")

        message = (
            "https://github.com/example-org/example-skill\n"
            "Help me install the skill in the repo."
        )

        prompt = AgentLoop._build_request_context_prompt(loop, message)

        assert "[TURN PRIORITY]:" in prompt
        assert "[GITHUB SKILL INSTALL CONTEXT]:" not in prompt
        assert "skill_marketplace(action='install_skill'" not in prompt
        assert "Do not use web_fetch or git clone only to confirm SKILL.md" not in prompt
        assert message in prompt

    def test_system_prompt_has_generic_stop_condition_not_case_route(self):
        """The loop should prevent premature setup-only stops with a generic contract."""
        # Check the initialized system-prompt fragment without constructing
        # the full spoon-core agent stack in this focused unit test.
        source = Path("spoon_bot/agent/loop.py").read_text(encoding="utf-8")

        assert "### Stop condition" in source
        assert "Do not stop after setup" in source
        assert "safe next action that the newest request already asked" in source
        assert "public browser app with frontend plus API/WebSocket/backend" in source
        assert "smallest local preflight" in source
        assert "[GITHUB SKILL INSTALL CONTEXT]" not in source

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

    def test_finalize_response_content_strips_ill_run_scratchpad_prefix(self):
        """First-person execution preambles should also stay private."""
        from spoon_bot.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        content = (
            "I'll run the command you specified and report the result briefly."
            "Great! The wallet command executed successfully."
        )

        cleaned = AgentLoop._finalize_response_content(
            loop,
            "Run the wallet command",
            content,
            turn_memory_start_index=0,
        )

        assert cleaned == "The wallet command executed successfully."

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

    def test_finalize_response_content_omits_file_body_when_user_requested_it(self):
        """If the user says not to show file content, final text must honor it too."""
        from spoon_bot.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        content = (
            "文件的前3行显示：\n\n"
            "```text\n"
            "title=demo\n"
            "color = blue\n"
            "status=ready\n"
            "```\n\n"
            "修改已完成。"
        )

        cleaned = AgentLoop._finalize_response_content(
            loop,
            "读取文件并验证，不要展示文件正文。",
            content,
            turn_memory_start_index=0,
        )

        assert "title=demo" not in cleaned
        assert "status=ready" not in cleaned
        assert "文件正文已按要求省略" in cleaned
        assert "修改已完成" in cleaned

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

        assert "Completed from the latest tool evidence" in cleaned
        assert "raw tool transcript" not in cleaned
        assert "FINAL_STATE task=347 action=joined" in cleaned
        assert "Tool `shell` output" not in cleaned
        assert "Step 1:" not in cleaned
        assert len(cleaned) < 1600

    def test_finalize_response_content_replaces_markdown_tool_trace_leak(self):
        """Markdown pseudo tool calls should not be accepted as completed work."""
        from spoon_bot.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        loop._agent = SimpleNamespace(
            memory=SimpleNamespace(
                messages=[
                    SimpleNamespace(role="user", content="create a weather skill"),
                    SimpleNamespace(
                        role="tool",
                        content="Observed output of cmd list_dir execution: (no items)",
                        name="list_dir",
                        tool_call_id="call_list",
                    ),
                ]
            )
        )
        content = (
            "- write_file(path='skills/weather/SKILL.md'): "
            "Observed output of cmd write_file execution: Success\n"
            "- shell(command='bash skills/weather/scripts/weather.sh London'): "
            "Observed output of cmd shell execution: Light rain\n\n"
            "Weather skill created successfully."
        )

        cleaned = AgentLoop._finalize_response_content(
            loop,
            "create a weather skill",
            content,
            turn_memory_start_index=0,
        )

        assert "Internal tool details were suppressed" in cleaned
        assert "raw tool transcript" not in cleaned
        assert "Tool `list_dir` output:" not in cleaned
        assert "write_file(path=" not in cleaned
        assert "Weather skill created successfully" not in cleaned

    def test_finalize_response_content_omits_file_body_from_fallback(self):
        """Raw transcript cleanup must not expose full file bodies to users."""
        from spoon_bot.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        loop._agent = SimpleNamespace(
            memory=SimpleNamespace(
                messages=[
                    SimpleNamespace(role="user", content="inspect file"),
                    SimpleNamespace(
                        role="tool",
                        content=(
                            "Observed output of cmd read_file execution: "
                            "[file: app.py | 100 chars | lines 1-10/10]\n"
                            "SECRET_BODY_SHOULD_NOT_BE_VISIBLE\n"
                            "print('implementation detail')"
                        ),
                        name="read_file",
                        tool_call_id="call_read",
                    ),
                ]
            )
        )
        content = (
            "Step 1: Observed output of cmd read_file execution: "
            "[file: app.py | 100 chars | lines 1-10/10]\n"
            "SECRET_BODY_SHOULD_NOT_BE_VISIBLE"
        )

        cleaned = AgentLoop._finalize_response_content(
            loop,
            "inspect file",
            content,
            turn_memory_start_index=0,
        )

        assert "Internal tool details were suppressed" in cleaned
        assert "[file: app.py | 100 chars | lines 1-10/10]" not in cleaned
        assert "content body omitted from user-visible fallback" not in cleaned
        assert "SECRET_BODY_SHOULD_NOT_BE_VISIBLE" not in cleaned
        assert "raw tool transcript" not in cleaned

    def test_stream_tool_result_metadata_caps_large_payloads(self):
        from spoon_bot.agent.loop import AgentLoop

        full_output = "head evidence\n" + ("x" * 210_000) + "\ntail settlement evidence"
        merged = AgentLoop._merge_stream_tool_result_metadata(
            {},
            streamed_result="short",
            captured_output=SimpleNamespace(
                summary_output="short",
                full_output=full_output,
            ),
        )

        assert merged["stream_output_truncated"] is True
        assert merged["stream_output_original_chars"] == len(full_output)
        assert len(merged["full_output"]) < 201_000
        assert "head evidence" in merged["full_output"]
        assert "tail settlement evidence" in merged["full_output"]
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

    def test_request_execution_hints_extract_local_skill_commands_and_urls(self, tmp_path):
        """Request execution hints should derive local executable skill metadata generically."""
        from spoon_bot.agent.loop import AgentLoop

        workspace = tmp_path / "workspace"
        skill_dir = workspace / "skills" / "spot-agent-cypher"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            """---
description: Play the SPOT game through the local CLI
when_to_use: Use when the user wants to play SPOT or continue a SPOT game.
---
# Spot Agent

CLI := node skills/spot-agent-cypher/cli/index.js

```bash
$CLI wallet
$CLI join A
```

API base: http://13.251.72.206:8080/api/agent/games
""",
            encoding="utf-8",
        )

        loop = AgentLoop.__new__(AgentLoop)
        loop.workspace = workspace
        loop._skill_paths = [workspace / "skills"]
        loop._touched_paths = set()

        hints = AgentLoop._build_request_execution_hints(
            loop,
            "Continue the SPOT game with spot-agent-cypher and finish the round.",
        )

        assert hints["explicit_request_urls"] == []
        assert len(hints["local_executable_skills"]) == 1
        skill_hint = hints["local_executable_skills"][0]
        assert skill_hint["name"] == "spot-agent-cypher"
        assert "node skills/spot-agent-cypher/cli/index.js" in skill_hint["commands"][0]
        assert any("join A" in command for command in skill_hint["commands"])
        assert "http://13.251.72.206:8080/api/agent/games" in skill_hint["urls"]
        assert hints["exact_shell_commands"] == []

    def test_request_execution_hints_prefer_documented_commands_section(self, tmp_path):
        """Local skill hints should expose downstream command forms, not setup loops."""
        from spoon_bot.agent.loop import AgentLoop
        from spoon_bot.agent.request_hints import format_local_executable_skill_context

        workspace = tmp_path / "workspace"
        skill_dir = workspace / "skills" / "spot-agent-cypher"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            """---
description: Play the SPOT game through the local CLI
when_to_use: Use when the user wants to play SPOT or continue a SPOT game.
---
# Spot Agent

CLI := node skills/spot-agent-cypher/cli/index.js

## Setup

```text
RUN $CLI wallet
MATCH output: "No wallet" -> run $CLI wallet again
RUN $CLI join {gameId} {spot} again
```

## Commands

```bash
$CLI wallet
$CLI faucet [-c <code>]
$CLI faucet-answer <challengeId> <answer> [-c <code>]
$CLI invitation-code
$CLI game status <gameId>
$CLI game context <gameId>
$CLI game snapshot <gameId>
$CLI challenge-start <gameId>
$CLI challenge-answer <gameId> [challengeId] "<answer>"
$CLI join [gameId] [spot]
$CLI wait <gameId>
$CLI reveal <gameId>
$CLI settlement <gameId>
$CLI history
$CLI summary
$CLI stats
```
""",
            encoding="utf-8",
        )

        loop = AgentLoop.__new__(AgentLoop)
        loop.workspace = workspace
        loop._skill_paths = [workspace / "skills"]
        loop._touched_paths = set()

        hints = AgentLoop._build_request_execution_hints(
            loop,
            "Use spot-agent-cypher to join and finish the SPOT game.",
        )
        commands = hints["local_executable_skills"][0]["commands"]
        context = format_local_executable_skill_context(hints)

        assert any("join [gameId] [spot]" in command for command in commands)
        assert any("settlement <gameId>" in command for command in commands)
        assert not any("wallet again" in command for command in commands)
        assert "join [gameId] [spot]" in context
        assert "settlement <gameId>" in context
        assert "extracted command forms" in context

    def test_request_execution_hints_keep_github_urls_as_plain_evidence(self, tmp_path):
        """Explicit GitHub URLs stay evidence, not request-routed workflows."""
        from spoon_bot.agent.loop import AgentLoop

        workspace = tmp_path / "workspace"
        workspace.mkdir()

        loop = AgentLoop.__new__(AgentLoop)
        loop.workspace = workspace
        loop._skill_paths = [workspace / "skills"]
        loop._touched_paths = set()

        message = (
            "https://github.com/example-org/example-skill\n"
            "Please install the skill in this repo."
        )

        hints = AgentLoop._build_request_execution_hints(loop, message)

        assert hints["explicit_request_urls"] == ["https://github.com/example-org/example-skill"]
        assert hints["local_executable_skills"] == []
        assert "github_skill_install_request" not in hints
        assert "execution_workflows" not in hints
        assert "tool_call_mode" not in hints

    def test_exact_spot_prompt_keeps_original_request_data_without_route(self, tmp_path):
        """The SPOT prompt keeps explicit facts without rewriting or route policy."""
        from spoon_bot.agent.loop import AgentLoop
        from spoon_bot.agent.request_hints import format_explicit_request_values_context

        workspace = tmp_path / "workspace"
        workspace.mkdir()

        loop = AgentLoop.__new__(AgentLoop)
        loop.workspace = workspace
        loop._skill_paths = [workspace / "skills"]
        loop._touched_paths = set()

        message = (
            "https://github.com/Agent-Cypher-Lab/agent-spot-cypher\n"
            "Help me install the skill in the repo, use the faucet, register 8004 agent id, "
            "and then join the latest spot game. Invited Code: 3KK57S."
        )

        hints = AgentLoop._build_request_execution_hints(loop, message)

        assert hints["explicit_request_urls"] == [
            "https://github.com/Agent-Cypher-Lab/agent-spot-cypher"
        ]
        assert hints["explicit_request_values"] == [{
            "value": "3KK57S",
            "labels": ["code", "invited"],
            "label": "Invited Code",
        }]
        assert "github_skill_install_request" not in hints
        assert "execution_workflows" not in hints
        assert "tool_call_mode" not in hints
        assert message.endswith("Invited Code: 3KK57S.")

        context = format_explicit_request_values_context(message)
        assert "[STRUCTURED USER REQUEST FACTS]:" in context
        assert "Invited Code: 3KK57S" in context

    def test_request_execution_hints_do_not_treat_plain_repo_clone_as_skill_install(self, tmp_path):
        """A normal GitHub repo request should not be routed into workspace/skills."""
        from spoon_bot.agent.loop import AgentLoop

        workspace = tmp_path / "workspace"
        workspace.mkdir()

        loop = AgentLoop.__new__(AgentLoop)
        loop.workspace = workspace
        loop._skill_paths = [workspace / "skills"]
        loop._touched_paths = set()

        message = "Clone https://github.com/example-org/example-repo and inspect the README."

        hints = AgentLoop._build_request_execution_hints(loop, message)

        assert "github_skill_install_request" not in hints
        assert hints["explicit_request_urls"] == ["https://github.com/example-org/example-repo"]

    def test_skill_contract_check_uses_tool_evidence(self):
        """Skill workflows get a generic verifier pass from tool evidence."""
        from spoon_bot.agent.turn_verifiers import should_run_skill_contract_check

        events = [{
            "type": "tool_result",
            "metadata": {
                "name": "skill_marketplace",
                "result": "SUCCESS: Skill 'example-skill' installed (1 files).",
            },
        }]

        assert should_run_skill_contract_check(events) is True

    def test_skill_contract_check_survives_compacted_skill_setup_evidence(self):
        """A remaining skill CLI NEXT still marks the turn as a skill workflow."""
        from spoon_bot.agent.turn_verifiers import should_run_skill_contract_check

        events = [{
            "type": "tool_result",
            "metadata": {
                "name": "shell",
                "result": (
                    "Game=152\n"
                    "Phase=Finished\n"
                    "NEXT: node skills/example-skill/cli/index.js settlement 152"
                ),
            },
        }]

        assert should_run_skill_contract_check(events) is True

    def test_plain_repo_result_does_not_run_skill_contract_check(self):
        """Regular repository tool results should not use the skill verifier."""
        from spoon_bot.agent.turn_verifiers import should_run_skill_contract_check

        events = [{
            "type": "tool_result",
            "metadata": {
                "name": "shell",
                "result": "Cloned https://github.com/example-org/example-repo.git",
            },
        }]

        assert should_run_skill_contract_check(events) is False

    def test_skill_read_summary_remains_model_visible(self):
        """SKILL.md execution summaries should help recovery without leaking to clients."""
        from spoon_bot.agent.loop import AgentLoop

        summary = (
            "[file: skills/spot-agent-cypher/SKILL.md | 200 chars | skill-ref]\n"
            "[SKILL.md execution summary]\n"
            "CLI entrypoint:\n"
            "CLI := node skills/spot-agent-cypher/cli/index.js\n\n"
            "Documented commands:\n"
            "$CLI join [gameId] [spot]\n"
            "$CLI settlement <gameId>"
        )

        metadata = AgentLoop._merge_stream_tool_result_metadata(
            {"name": "read_file"},
            streamed_result=summary,
            captured_output=None,
        )

        assert metadata["result"] == (
            "[file: skills/spot-agent-cypher/SKILL.md | 200 chars | skill-ref] "
            "content body omitted from user-visible tool result"
        )
        assert "join [gameId] [spot]" in metadata["model_result"]
        assert "settlement <gameId>" in metadata["model_result"]
        assert metadata["result_body_omitted"] is True
        assert "model_result_body_omitted" not in metadata

    def test_repeated_read_recovery_prompt_includes_skill_context(self):
        """Duplicate SKILL.md reads should recover toward documented commands."""
        from spoon_bot.agent.loop import AgentLoop

        prompt = AgentLoop._build_repeated_read_recovery_prompt(
            "Use the skill to finish the workflow.",
            request_context=(
                "[LOCAL SKILL EXECUTION CONTEXT]:\n"
                "- demo-skill at skills/demo-skill/SKILL.md; commands: "
                "node skills/demo-skill/cli/index.js | "
                "node skills/demo-skill/cli/index.js run"
            ),
        )

        assert "installed SKILL.md" in prompt
        assert "node skills/demo-skill/cli/index.js run" in prompt

    def test_tool_evidence_fallback_prefers_user_summary_marker(self):
        """Fallback cleanup should use user-facing evidence, not raw tool transcript."""
        from spoon_bot.agent.turn_verifiers import build_user_facing_tool_evidence_answer

        tx_hash = "0x" + "ab" * 32
        answer = build_user_facing_tool_evidence_answer([
            (
                f"JOINED game=116 tx={tx_hash}\n"
                "Read it aloud: SETTLEMENT game=116 result=WIN rank=1/4 spot=A score=10."
            ),
        ])

        assert answer == "Completed.\n\nSETTLEMENT game=116 result=WIN rank=1/4 spot=A score=10."
        assert tx_hash not in answer
        assert "Tool `" not in answer

    def test_tool_event_summary_marker_distinguishes_completion_from_status(self):
        """Verifier acceptance should depend on explicit user-facing tool evidence."""
        from spoon_bot.agent.turn_verifiers import tool_events_have_user_summary_marker

        status_events = [{
            "type": "tool_result",
            "metadata": {
                "name": "shell",
                "result": "SUCCESS: setup complete\nGames 19 settled / 19 total",
            },
        }]
        completion_events = [{
            "type": "tool_result",
            "metadata": {
                "name": "shell",
                "result": "Read it aloud: workflow finished with result=WIN.",
            },
        }]

        assert tool_events_have_user_summary_marker(status_events) is False
        assert tool_events_have_user_summary_marker(completion_events) is True

    def test_tool_event_summary_marker_uses_full_output_tail(self):
        """Verifier should not miss terminal evidence omitted from model summaries."""
        from spoon_bot.agent.turn_verifiers import (
            build_user_facing_tool_event_answer,
            latest_tool_event_has_user_summary_marker,
            tool_events_have_user_summary_marker,
        )

        events = [{
            "type": "tool_result",
            "metadata": {
                "name": "shell",
                "model_result": "JOINED game=123\nNEXT: node tool wait 123",
                "full_result": (
                    "JOINED game=123\n"
                    "NEXT: node tool wait 123\n"
                    "Read it aloud: SETTLEMENT game=123 result=WIN rank=1/4"
                ),
            },
        }]

        assert tool_events_have_user_summary_marker(events) is True
        assert latest_tool_event_has_user_summary_marker(events) is True
        assert build_user_facing_tool_event_answer(events) == (
            "Completed.\n\nSETTLEMENT game=123 result=WIN rank=1/4"
        )

    def test_latest_summary_marker_only_accepts_latest_tool_output(self):
        """Earlier summaries should not hide a later unfinished skill step."""
        from spoon_bot.agent.turn_verifiers import latest_tool_event_has_user_summary_marker

        events = [
            {
                "type": "tool_result",
                "metadata": {
                    "name": "shell",
                    "result": "Read it aloud: setup completed.",
                },
            },
            {
                "type": "tool_result",
                "metadata": {
                    "name": "shell",
                    "result": "NEXT: node skills/example-skill/cli/index.js run",
                },
            },
        ]

        assert latest_tool_event_has_user_summary_marker(events) is False

    def test_skill_install_completion_is_detected_from_skill_manager(self):
        """Install-only requests can finish from skill manager evidence."""
        from spoon_bot.agent.turn_verifiers import (
            build_skill_install_completion_answer,
            tool_events_have_skill_install_completion,
        )

        events = [{
            "type": "tool_result",
            "metadata": {
                "name": "skill_marketplace",
                "result": "SUCCESS: Skill 'example-skill' installed (3 files).",
            },
        }]

        assert tool_events_have_skill_install_completion(events) is True
        assert build_skill_install_completion_answer(events) == (
            "Completed.\n\nSkill 'example-skill' installed."
        )

    def test_instruction_text_is_not_user_facing_completion_evidence(self):
        """Instruction-only skill text should not become a completed fallback."""
        from spoon_bot.agent.turn_verifiers import build_user_facing_tool_evidence_answer

        answer = build_user_facing_tool_evidence_answer([
            (
                "RULE settlement_reporting:\n"
                "  run $CLI settlement {gameId}\n"
                "  use backend settlement detail only after the game has finished\n"
                "  after settlement appears, report both the result and the reason\n"
            )
        ])

        assert "Completed from the latest tool evidence" not in answer
        assert "Internal tool details were suppressed" in answer

    def test_latest_tool_event_has_next_command_detects_pending_continuation(self):
        """A latest follow-up command keeps a skill workflow open until evidence completes."""
        from spoon_bot.agent.turn_verifiers import latest_tool_event_has_next_command

        events = [
            {
                "type": "tool_result",
                "metadata": {
                    "name": "shell",
                    "result": "CHALLENGE PASSED\nNEXT: node tool join 119 A",
                },
            }
        ]

        assert latest_tool_event_has_next_command(events) is True

        action_required_events = [
            {
                "type": "tool_result",
                "metadata": {
                    "name": "shell",
                    "result": (
                        "ACTION REQUIRED: choose a spot and run: "
                        "node skills/example/cli/index.js join 145 <A|B|C|D|E>"
                    ),
                },
            }
        ]

        assert latest_tool_event_has_next_command(action_required_events) is True

    def test_skill_contract_resolves_answer_placeholder_next_command(self):
        """Verifier should expose a resolved follow-up command after a scalar calculation."""
        from spoon_bot.agent.turn_verifiers import (
            extract_resolved_tool_next_commands,
            has_pending_placeholder_next_command,
        )

        events = [
            {
                "type": "tool_result",
                "metadata": {
                    "name": "shell",
                    "result": (
                        "QUESTION: What is the answer?\n"
                        "NEXT: node tool challenge-answer 123 \"<answer>\""
                    ),
                },
            },
            {
                "type": "tool_result",
                "metadata": {"name": "shell", "result": "18"},
            },
        ]

        commands = extract_resolved_tool_next_commands(events)

        assert commands[-1] == 'node tool challenge-answer 123 "18"'
        assert has_pending_placeholder_next_command(events) is True

        events.append({
            "type": "tool_result",
            "metadata": {
                "name": "shell",
                "result": "CHALLENGE PASSED\nNEXT: node tool join 123 A",
            },
        })
        assert has_pending_placeholder_next_command(events) is False

    def test_skill_contract_extracts_embedded_followup_command_without_next_prefix(self):
        """Continuation extraction should not depend on a NEXT prefix."""
        from spoon_bot.agent.turn_verifiers import (
            extract_resolved_tool_next_commands,
            has_pending_placeholder_next_command,
        )

        events = [
            {
                "type": "tool_result",
                "metadata": {
                    "name": "shell",
                    "result": (
                        "ACTION REQUIRED: decide a spot and run: "
                        "node skills/example/cli/index.js join 145 <A|B|C|D|E>"
                    ),
                },
            }
        ]

        commands = extract_resolved_tool_next_commands(events)

        assert commands[-1] == "node skills/example/cli/index.js join 145 <A|B|C|D|E>"
        assert has_pending_placeholder_next_command(events) is True

    def test_pending_followup_recovery_prompt_is_generic(self):
        """Placeholder recovery prompt should rely on evidence, not route literals."""
        from spoon_bot.agent.loop import AgentLoop

        events = [{
            "type": "tool_result",
            "metadata": {
                "name": "shell",
                "result": (
                    "Game=145\n"
                    "ACTION REQUIRED: decide a spot and run: "
                    "node skills/example/cli/index.js join 145 <A|B|C|D|E>"
                ),
            },
        }]

        prompt = AgentLoop._build_pending_followup_recovery_prompt(
            "continue the workflow",
            events,
        )

        assert "[INTERNAL FOLLOW-UP PLACEHOLDER RECOVERY]" in prompt
        assert "node skills/example/cli/index.js join 145 <A|B|C|D|E>" in prompt
        assert "Do not read the same SKILL.md again" in prompt

    @pytest.mark.asyncio
    async def test_process_structural_followup_keeps_direct_tool_event(self):
        """Non-stream structural NEXT execution should remain visible for summary checks."""
        from spoon_bot.agent.loop import AgentLoop
        from spoon_bot.agent.turn_verifiers import tool_events_have_user_summary_marker

        agent = AgentLoop.__new__(AgentLoop)
        agent.tools = MagicMock()
        agent.tools.execute = AsyncMock(
            return_value=(
                "SETTLEMENT game=152 status=finished\n"
                "Read it aloud: SETTLEMENT game=152 result=WIN rank=1/4."
            )
        )
        agent._workspace_posix_path = MagicMock(return_value="/workspace")
        events = [{
            "type": "tool_result",
            "metadata": {
                "name": "shell",
                "result": (
                    "Game=152\n"
                    "Phase=Finished\n"
                    "NEXT: node skills/example/cli/index.js settlement 152"
                ),
            },
        }]

        emitted = await AgentLoop._execute_structural_followup_commands_for_process(
            agent,
            events,
            reason="test",
        )

        assert emitted
        assert tool_events_have_user_summary_marker(events) is True
        assert "settlement 152" in agent.tools.execute.await_args.args[1]["command"]
        forbidden_env_source = "source " + ".env.local"
        assert forbidden_env_source not in agent.tools.execute.await_args.args[1]["command"]
        assert agent.tools.execute.await_args.args[1]["working_dir"] == "/workspace"

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

        # Should have content chunks + done. Initial content may be buffered
        # and emitted as a validated segment, so do not require original chunk
        # boundaries to be preserved.
        content_chunks = [c for c in chunks if c["type"] == "content"]
        done_chunks = [c for c in chunks if c["type"] == "done"]

        assert "".join(c["delta"] for c in content_chunks) == "Hello World"
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
        assert "".join(chunk["delta"] for chunk in content_chunks) == "Hello World"

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
    async def test_stream_withholds_initial_content_when_tool_follows(self):
        """Initial content before the first tool call is private tool preamble."""
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
        assert [c["type"] for c in emitted] == ["tool_call", "content"]
        assert emitted[0]["metadata"]["name"] == "shell"
        assert emitted[1]["delta"] == "Part B."
        done_chunks = [c for c in chunks if c["type"] == "done"]
        assert done_chunks[0]["metadata"]["content"] == "Part B."

    @pytest.mark.asyncio
    async def test_stream_withholds_split_initial_content_when_tool_follows(self):
        """Split initial content should not leak before the first tool call."""
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
        assert [c["type"] for c in emitted] == ["tool_call", "content"]
        assert emitted[0]["metadata"]["name"] == "read_file"
        assert emitted[1]["delta"] == "Done."

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
        assert [c["type"] for c in emitted] == ["tool_call", "content"]
        assert emitted[0]["metadata"]["segment_start"] is True
        assert emitted[1]["metadata"]["segment_start"] is True
        assert emitted[0]["metadata"]["segment_index"] != emitted[1]["metadata"]["segment_index"]
        assert emitted[0]["metadata"]["segment_type"] == "tool_call"
        assert emitted[1]["metadata"]["segment_type"] == "content"

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
        assert "Completed from the latest tool evidence" in emitted_content[0]["delta"]
        assert "raw tool transcript" not in emitted_content[0]["delta"]
        done_chunks = [c for c in chunks if c["type"] == "done"]
        assert "Completed from the latest tool evidence" in done_chunks[0]["metadata"]["content"]
        assert "raw tool transcript" not in done_chunks[0]["metadata"]["content"]
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
        assert "I skipped a repeated action that had already run." in emitted_content[0]["delta"]
        assert "STOP_TOOL_LOOP" not in emitted_content[0]["delta"]
        assert "tool guardrail" not in emitted_content[0]["delta"]

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
        assert "I stopped retrying after repeated failures." in emitted_content[0]["delta"]
        assert "STOP_TOOL_LOOP" not in emitted_content[0]["delta"]
        assert "tool guardrail" not in emitted_content[0]["delta"]

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
    async def test_stream_replaces_post_tool_markdown_tool_trace_before_emit(self):
        """Pseudo tool-call markdown should trigger a real tool-call retry."""
        from spoon_bot.agent.loop import AgentLoop

        def _tool_call(call_id: str, name: str, arguments: str):
            tool_call = MagicMock()
            tool_call.id = call_id
            tool_call.function = MagicMock()
            tool_call.function.name = name
            tool_call.function.arguments = arguments
            return tool_call

        initial_tool = _tool_call("call_1", "list_dir", '{"path":"skills"}')
        repair_tool = _tool_call(
            "call_2",
            "write_file",
            '{"path":"skills/weather/SKILL.md","content":"# Weather"}',
        )

        leaked_transcript = (
            "- write_file(path='skills/weather/SKILL.md'): "
            "Observed output of cmd write_file execution: Success\n"
            "- shell(command='bash skills/weather/scripts/weather.sh London'): "
            "Observed output of cmd shell execution: Light rain\n\n"
            "Weather skill created successfully."
        )
        agent = self._make_stream_agent([])
        attempts = 0

        async def run(**kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                await agent._agent.output_queue.put({"tool_calls": [initial_tool]})
                await agent._agent.output_queue.put({"content": leaked_transcript})
                return MagicMock(content=leaked_transcript)
            await agent._agent.output_queue.put({"tool_calls": [repair_tool]})
            await agent._agent.output_queue.put({
                "type": "tool_result",
                "name": "write_file",
                "tool_call_id": "call_2",
                "result": "Success",
            })
            await agent._agent.output_queue.put({"content": "Weather skill created."})
            return MagicMock(content="Weather skill created.")

        agent._agent.run = AsyncMock(side_effect=run)

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="placeholder", thinking=True):
            chunks.append(chunk)

        content_text = "".join(c["delta"] for c in chunks if c["type"] == "content")
        tool_call_names = [
            c["metadata"]["name"]
            for c in chunks
            if c["type"] == "tool_call"
        ]

        assert attempts == 2
        assert tool_call_names == ["list_dir", "write_file"]
        assert content_text == "Weather skill created."
        assert "raw tool transcript" not in content_text
        assert "write_file(path=" not in content_text
        done_chunks = [c for c in chunks if c["type"] == "done"]
        assert done_chunks[0]["metadata"]["content"] == "Weather skill created."
        assert "write_file(path=" not in done_chunks[0]["metadata"]["content"]

    def test_build_tool_loop_fallback_response_unwraps_exact_shell_failure(self):
        """Exact requested shell failures should surface as the blocker, not generic guardrail prose."""
        from spoon_bot.agent.loop import AgentLoop

        event = {
            "type": "tool_result",
            "metadata": {
                "name": "shell",
                "result": (
                    "STOP_TOOL_LOOP: Exact requested shell command failed. "
                    "Report this blocker directly instead of switching to additional exploratory tools.\n"
                    "STDERR:\nSPOT API GET /api/agent/games/assign failed after 5 attempts: fetch failed\n\n"
                    "Exit code: 1"
                ),
            },
        }

        cleaned = AgentLoop._build_tool_loop_fallback_response([event], reason="tool_suppression")

        assert "tool guardrail suppressed repeated work" not in cleaned
        assert "STOP_TOOL_LOOP" not in cleaned
        assert "SPOT API GET /api/agent/games/assign failed after 5 attempts: fetch failed" in cleaned

    def test_build_tool_loop_fallback_response_reports_prior_result_for_duplicate_suppression(self):
        """Duplicate suppression should not surface STOP_TOOL_LOOP internals to users."""
        from spoon_bot.agent.loop import AgentLoop

        events = [
            {
                "type": "tool_result",
                "metadata": {
                    "name": "shell",
                    "result": "Inserted comment rows for Alice and alice@example.com.",
                },
            },
            {
                "type": "tool_result",
                "metadata": {
                    "name": "shell",
                    "result": (
                        "STOP_TOOL_LOOP: Error: duplicate tool invocation suppressed. "
                        "The same tool and arguments already executed in this request."
                    ),
                },
            },
        ]

        cleaned = AgentLoop._build_tool_loop_fallback_response(events, reason="tool_suppression")

        assert "I skipped a repeated action that had already run." in cleaned
        assert "Latest available result:" in cleaned
        assert "Inserted comment rows" in cleaned
        assert "STOP_TOOL_LOOP" not in cleaned
        assert "tool guardrail" not in cleaned
        assert "duplicate tool invocation suppressed" not in cleaned

    def test_duplicate_suppression_preserves_fact_check_blocker_without_skip_prose(self):
        """Fact-check sequencing blockers should not be wrapped as duplicate-action output."""
        from spoon_bot.agent.loop import AgentLoop

        blocker = (
            "Current-session fact check required: this request asks about prior "
            "actions/results in the conversation. Call search_history(scope='current') first."
        )
        events = [
            {
                "type": "tool_result",
                "metadata": {"name": "shell", "result": blocker},
            },
            {
                "type": "tool_result",
                "metadata": {
                    "name": "shell",
                    "result": (
                        "STOP_TOOL_LOOP: Error: duplicate tool invocation suppressed. "
                        "The same tool and arguments already executed in this request."
                    ),
                },
            },
        ]

        cleaned = AgentLoop._build_tool_loop_fallback_response(events, reason="tool_suppression")

        assert cleaned.startswith("Current-session fact check required")
        assert "search_history(scope='current')" in cleaned
        assert "I skipped a repeated action" not in cleaned
        assert "STOP_TOOL_LOOP" not in cleaned

    @pytest.mark.asyncio
    async def test_stream_buffers_split_pre_tool_scratchpad_prefix(self):
        """Tiny split scratchpad chunks before the first tool call should not leak."""
        from spoon_bot.agent.loop import AgentLoop

        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function = MagicMock()
        tool_call.function.name = "shell"
        tool_call.function.arguments = '{"command":"node skills/spot-agent-cypher/cli/index.js wallet"}'

        agent = self._make_stream_agent([
            {"content": "I"},
            {"content": "'ll run the command and report briefly."},
            {"tool_calls": [tool_call]},
            {"content": "Wallet command executed successfully."},
        ])

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="placeholder", thinking=True):
            chunks.append(chunk)

        content_chunks = [c for c in chunks if c["type"] == "content"]
        assert [c["delta"] for c in content_chunks] == [
            "Wallet command executed successfully."
        ]
        done_chunks = [c for c in chunks if c["type"] == "done"]
        assert done_chunks[0]["metadata"]["content"] == "Wallet command executed successfully."

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
    async def test_stream_tool_result_delta_omits_file_body(self):
        """User-visible tool_result delta should not expose read_file bodies."""
        from spoon_bot.agent.loop import AgentLoop

        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function = MagicMock()
        tool_call.function.name = "read_file"
        tool_call.function.arguments = '{"path":"secret.txt"}'
        file_result = (
            "[file: secret.txt | 80 chars | lines 1-3/3]\n"
            "SECRET_BODY_SHOULD_STAY_INTERNAL\n"
            "implementation detail"
        )

        async def mock_run(**kwargs):
            await agent._agent.output_queue.put({"tool_calls": [tool_call]})
            await agent._agent.output_queue.put(
                {
                    "type": "tool_result",
                    "metadata": {
                        "id": "call_1",
                        "name": "read_file",
                    },
                    "result": file_result,
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

        tool_result_chunk = next(c for c in chunks if c["type"] == "tool_result")
        assert "[file: secret.txt | 80 chars | lines 1-3/3]" in tool_result_chunk["delta"]
        assert "content body omitted from user-visible tool result" in tool_result_chunk["delta"]
        assert "SECRET_BODY_SHOULD_STAY_INTERNAL" not in tool_result_chunk["delta"]
        assert "SECRET_BODY_SHOULD_STAY_INTERNAL" not in json.dumps(
            tool_result_chunk["metadata"],
            ensure_ascii=False,
        )
        assert tool_result_chunk["metadata"]["result_body_omitted"] is True
        assert tool_result_chunk["metadata"]["model_result_body_omitted"] is True

    @pytest.mark.asyncio
    async def test_stream_withholds_short_pre_tool_content(self):
        """Short content before a tool call is preamble and should not reach clients."""
        from spoon_bot.agent.loop import AgentLoop

        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function = MagicMock()
        tool_call.function.name = "shell"
        tool_call.function.arguments = '{"command":"pwd"}'

        async def mock_run(**kwargs):
            await agent._agent.output_queue.put({"content": "我将"})
            await agent._agent.output_queue.put({"content": "按照您的要求执行："})
            await agent._agent.output_queue.put({"tool_calls": [tool_call]})
            await agent._agent.output_queue.put(
                {
                    "type": "tool_result",
                    "metadata": {
                        "id": "call_1",
                        "name": "shell",
                    },
                    "result": "/workspace",
                }
            )
            await agent._agent.output_queue.put({"content": "完成。"})

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

        content_chunks = [c for c in chunks if c["type"] == "content"]
        assert [c["delta"] for c in content_chunks] == ["完成。"]
        done_chunk = next(c for c in chunks if c["type"] == "done")
        assert done_chunk["metadata"]["content"] == "完成。"
        assert "我将按照您的要求执行" not in json.dumps(chunks, ensure_ascii=False)

    @pytest.mark.asyncio
    async def test_stream_does_not_duplicate_no_tool_content_after_initial_buffer(self):
        """Once no-tool content streams through, later chunks should not be re-emitted."""
        from spoon_bot.agent.loop import AgentLoop

        first_delta = "这是一个不需要工具的长回答。" * 25
        second_delta = "后续内容。"

        async def mock_run(**kwargs):
            await agent._agent.output_queue.put({"content": first_delta})
            await agent._agent.output_queue.put({"content": second_delta})

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

        content_chunks = [c for c in chunks if c["type"] == "content"]
        assert [c["delta"] for c in content_chunks] == [first_delta, second_delta]
        done_chunk = next(c for c in chunks if c["type"] == "done")
        assert done_chunk["metadata"]["content"] == first_delta + second_delta

    def test_user_visible_text_masks_internal_guardrails(self):
        """Internal loop-control markers should be converted before display."""
        from spoon_bot.agent.loop import AgentLoop

        cleaned = AgentLoop._mask_user_visible_text(
            '第二次读取显示 "redundant file read suppressed" 错误。'
        )

        assert "redundant file read suppressed" not in cleaned
        assert "duplicate file read skipped" in cleaned

    def test_stream_metadata_preserves_guardrail_stop_without_raw_marker(self):
        """Sanitized WS metadata still has a structural stop signal."""
        from spoon_bot.agent.loop import AgentLoop

        metadata = AgentLoop._merge_stream_tool_result_metadata(
            {"name": "read_file"},
            streamed_result=(
                "STOP_TOOL_LOOP: Error: duplicate tool invocation suppressed. "
                "The same tool and arguments already ran."
            ),
            captured_output=None,
        )
        event = {"type": "tool_result", "delta": metadata["model_result"], "metadata": metadata}

        assert AgentLoop._is_tool_loop_suppression_event(event) is True
        assert AgentLoop._tool_loop_suppression_message(event) == (
            "I skipped a repeated action that had already run."
        )
        assert "STOP_TOOL_LOOP" not in json.dumps(metadata, ensure_ascii=False)
        assert "duplicate tool invocation suppressed" not in json.dumps(
            metadata,
            ensure_ascii=False,
        )

    def test_redundant_file_read_cache_hit_does_not_stop_stream_loop(self):
        """Duplicate read_file cache hits are recoverable evidence, not final fallbacks."""
        from spoon_bot.agent.loop import AgentLoop

        metadata = AgentLoop._merge_stream_tool_result_metadata(
            {"name": "read_file"},
            streamed_result=(
                "READ_FILE_CACHE_HIT: requested file range already available in this "
                "request. Treat this read as complete and continue the user's "
                "remaining instructions without calling read_file again for the "
                "same path and range."
            ),
            captured_output=None,
        )
        event = {"type": "tool_result", "delta": metadata["model_result"], "metadata": metadata}

        assert AgentLoop._is_tool_loop_suppression_event(event) is False
        assert "guardrail_stop" not in metadata
        assert "READ_FILE_CACHE_HIT" not in json.dumps(metadata, ensure_ascii=False)
        assert "STOP_TOOL_LOOP" not in json.dumps(metadata, ensure_ascii=False)

    def test_stream_tool_result_metadata_caps_long_internal_outputs(self):
        """Huge non-file tool outputs should not flood client-visible WS payloads."""
        from spoon_bot.agent.loop import AgentLoop

        metadata = AgentLoop._merge_stream_tool_result_metadata(
            {"name": "spawn"},
            streamed_result="sub-agent result\n" + ("detail\n" * 5000),
            captured_output=None,
        )
        visible_delta = AgentLoop._stream_tool_result_visible_delta("", metadata)

        assert metadata["stream_output_truncated"] is True
        assert metadata["model_output_truncated"] is True
        assert len(metadata["result"]) < 13_000
        assert len(metadata["model_result"]) < 13_000
        assert len(visible_delta) < 4_500
        assert "stream output middle truncated" in metadata["result"]

    @pytest.mark.asyncio
    async def test_stream_recovers_repeated_read_file_storm_before_tool_budget(self):
        """Repeated identical reads should switch to continuation recovery quickly."""
        from spoon_bot.agent.loop import AgentLoop

        def _tool_call(call_id: str, name: str, arguments: str) -> MagicMock:
            tool_call = MagicMock()
            tool_call.id = call_id
            tool_call.function = MagicMock()
            tool_call.function.name = name
            tool_call.function.arguments = arguments
            return tool_call

        read_arguments = '{"path":"demo.txt","offset":1,"limit":3}'
        agent = self._make_stream_agent([])
        agent.provider_total_timeout = 30.0
        agent.tool_followup_timeout = 30.0
        agent.max_stream_tool_results_without_content = 99
        run_count = 0

        async def mock_run(**kwargs):
            nonlocal run_count
            run_count += 1
            if run_count == 1:
                for index in range(20):
                    call_id = f"read_{index}"
                    await agent._agent.output_queue.put(
                        {"tool_calls": [_tool_call(call_id, "read_file", read_arguments)]}
                    )
                    await agent._agent.output_queue.put(
                        {
                            "type": "tool_result",
                            "metadata": {"id": call_id, "name": "read_file"},
                            "result": "[file: demo.txt | 12 chars | lines 1-3/3]\ncolor = red",
                        }
                    )
                return MagicMock(content="")

            edit_call = _tool_call(
                "edit_1",
                "edit_file",
                '{"path":"demo.txt","old_text":"color = red","new_text":"color = blue"}',
            )
            verify_call = _tool_call("verify_1", "read_file", read_arguments)
            await agent._agent.output_queue.put({"tool_calls": [edit_call]})
            await agent._agent.output_queue.put(
                {
                    "type": "tool_result",
                    "metadata": {"id": "edit_1", "name": "edit_file"},
                    "result": "Successfully edited demo.txt",
                }
            )
            await agent._agent.output_queue.put({"tool_calls": [verify_call]})
            await agent._agent.output_queue.put(
                {
                    "type": "tool_result",
                    "metadata": {"id": "verify_1", "name": "read_file"},
                    "result": "[file: demo.txt | 13 chars | lines 1-3/3]\ncolor = blue",
                }
            )
            await agent._agent.output_queue.put({"content": "Done."})
            return MagicMock(content="Done.")

        agent._agent.run = AsyncMock(side_effect=mock_run)

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="read twice, then edit"):
            chunks.append(chunk)

        tool_call_chunks = [chunk for chunk in chunks if chunk["type"] == "tool_call"]
        read_file_calls = [
            chunk
            for chunk in tool_call_chunks
            if chunk["metadata"].get("name") == "read_file"
        ]
        edit_file_calls = [
            chunk
            for chunk in tool_call_chunks
            if chunk["metadata"].get("name") == "edit_file"
        ]

        assert run_count == 2
        assert len(read_file_calls) < 10
        assert len(edit_file_calls) == 1
        assert edit_file_calls[0]["metadata"].get("repair") == "repeated_read_recovery"
        assert "Done." in "".join(
            chunk["delta"] for chunk in chunks if chunk["type"] == "content"
        )

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
        assert "The tool workflow stopped before a final answer" in content
        assert "Latest blocker" in content
        assert "remote endpoint did not respond" in content
        assert "stopped the tool loop" not in content
        done_chunks = [c for c in chunks if c["type"] == "done"]
        assert done_chunks[0]["metadata"]["content"] == content

    @pytest.mark.asyncio
    async def test_stream_runtime_error_after_tools_returns_latest_evidence(self):
        """Provider errors after tool progress should not collapse to a generic empty response."""
        from spoon_bot.agent.loop import AgentLoop

        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function = MagicMock()
        tool_call.function.name = "shell"
        tool_call.function.arguments = '{"command":"run workflow"}'

        agent = self._make_stream_agent(
            [
                {"tool_calls": [tool_call]},
                {
                    "type": "tool_result",
                    "metadata": {"id": "call_1", "name": "shell"},
                    "result": (
                        "CHALLENGE PASSED task=42\n"
                        "NEXT: node skills/example/cli/index.js finish 42"
                    ),
                },
            ],
            run_error=RuntimeError(
                "provider failed with secret key URL https://example.com/keys/abcdef"
            ),
        )

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="continue workflow"):
            chunks.append(chunk)

        content = "".join(c["delta"] for c in chunks if c["type"] == "content")
        errors = [c for c in chunks if c["type"] == "error"]

        assert "The tool workflow stopped before a final answer" in content
        assert "CHALLENGE PASSED task=42" in content
        assert "Pending next step" in content
        assert "node skills/example/cli/index.js finish 42" in content
        assert "secret key URL" not in "".join(str(c.get("delta", "")) for c in chunks)
        assert errors and errors[0]["delta"] == "An unexpected error occurred. Please try again."

    def test_tool_loop_fallback_summarizes_runtime_errors_without_stacktrace(self):
        """Timeout fallback should not expose raw internal transcript details."""
        from spoon_bot.agent.loop import AgentLoop

        event = {
            "type": "tool_result",
            "metadata": {"name": "shell"},
            "result": (
                "STDERR:\n"
                "node:events:502\n"
                "Error: listen EADDRINUSE: address already in use :::3000\n"
                "    at Server.setupListenHandle [as _listen2] (node:net:1902:16)\n"
                "Node.js v20.19.2"
            ),
        }

        cleaned = AgentLoop._build_tool_loop_fallback_response(
            [event],
            reason="total_timeout",
        )

        assert "The tool workflow stopped before a final answer" in cleaned
        assert "Latest blocker" in cleaned
        assert "EADDRINUSE" in cleaned
        assert "Server.setupListenHandle" not in cleaned
        assert "Recent tool evidence" not in cleaned

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
        assert "The tool workflow stopped before a final answer" in content
        assert "Internal tool details were suppressed" in content
        assert "stopped the tool loop" not in content

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
        assert "The tool workflow stopped before a final answer" in content
        assert "Internal tool details were suppressed" in content
        assert "response time budget" not in content
        assert "tokenAddress" not in content
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
    async def test_stream_total_timeout_waits_for_active_shell_budget(self):
        """An active shell tool should get its foreground budget before stream fallback."""
        from spoon_bot.agent.loop import AgentLoop

        first_tool = MagicMock()
        first_tool.id = "call_0"
        first_tool.function = MagicMock()
        first_tool.function.name = "shell"
        first_tool.function.arguments = '{"command":"setup-step"}'

        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function = MagicMock()
        tool_call.function.name = "shell"
        tool_call.function.arguments = '{"command":"long-game-command"}'

        agent = self._make_stream_agent([])
        agent.provider_silence_timeout = 0.05
        agent.provider_total_timeout = 0.05
        agent.tool_followup_timeout = 0.05
        agent.shell_timeout = 0.1
        agent.max_stream_tool_results_without_content = 99

        async def slow_shell_result_run(**kwargs):
            await agent._agent.output_queue.put({"tool_calls": [first_tool]})
            await agent._agent.output_queue.put(
                {
                    "type": "tool_result",
                    "metadata": {"id": "call_0", "name": "shell"},
                    "result": "setup step completed",
                }
            )
            await agent._agent.output_queue.put({"tool_calls": [tool_call]})
            await asyncio.sleep(0.12)
            await agent._agent.output_queue.put(
                {
                    "type": "tool_result",
                    "metadata": {"id": "call_1", "name": "shell"},
                    "result": "long game command moved forward",
                }
            )
            return "finished after active shell"

        agent._agent.run = AsyncMock(side_effect=slow_shell_result_run)

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="placeholder"):
            chunks.append(chunk)

        content = "".join(c["delta"] for c in chunks if c["type"] == "content")
        assert "finished after active shell" in content
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
        """Runtime-memory file results should backfill without exposing file bodies."""
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
        assert tool_result_chunk["delta"] == (
            "[file: README.md | 40 chars] "
            "content body omitted from user-visible tool result"
        )
        assert tool_result_chunk["metadata"]["result"] == (
            "[file: README.md | 145 chars] "
            "content body omitted from user-visible tool result"
        )
        assert tool_result_chunk["metadata"]["content"] == tool_result_chunk["metadata"]["result"]
        assert tool_result_chunk["metadata"]["model_result"] == tool_result_chunk["delta"]
        assert tool_result_chunk["metadata"]["result_body_omitted"] is True
        assert tool_result_chunk["metadata"]["model_result_body_omitted"] is True
        assert "hello" not in json.dumps(tool_result_chunk["metadata"], ensure_ascii=False)
        assert "world" not in json.dumps(tool_result_chunk["metadata"], ensure_ascii=False)

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
    async def test_stream_without_tool_call_releases_initial_content_after_run_finishes(self):
        """Initial content is released once the turn finishes without a tool call."""
        from spoon_bot.agent.loop import AgentLoop

        async def mock_run(**kwargs):
            await agent._agent.output_queue.put({"content": "chunk-1 "})
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

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="test", thinking=True):
            chunks.append(chunk)

        content_chunks = [c for c in chunks if c["type"] == "content"]
        done_chunks = [c for c in chunks if c["type"] == "done"]

        assert [c["delta"] for c in content_chunks] == ["chunk-1 chunk-2"]
        assert content_chunks[0]["metadata"]["withheld_initial_content"] is True
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

        # Depending on scheduling, the buffered partial may be released before
        # the terminal done chunk, but the error must be surfaced and the done
        # content must match what was actually emitted.
        error_chunks = [chunk for chunk in chunks if chunk["type"] == "error"]
        emitted = "".join(chunk["delta"] for chunk in chunks if chunk["type"] == "content")
        assert len(error_chunks) == 1
        assert error_chunks[0]["metadata"]["error_code"] == "RuntimeError"
        assert "unexpected error" in error_chunks[0]["metadata"]["error"].lower()
        assert chunks[-1]["type"] == "done"
        assert chunks[-1]["metadata"]["content"] == emitted

    @pytest.mark.asyncio
    async def test_stream_persists_user_turn_on_error_without_assistant(self):
        """stream() should keep the user turn even when the provider fails immediately."""
        from spoon_bot.agent.loop import AgentLoop

        agent = self._make_stream_agent([], run_error=RuntimeError("Immediate failure"))

        chunks = []
        async for chunk in AgentLoop.stream(agent, message="test"):
            chunks.append(chunk)

        # Should get an error event, then a user-facing fallback content chunk and done.
        assert any(chunk["type"] == "error" for chunk in chunks)
        assert chunks[-1]["type"] == "done"
        assert "unexpected error" in chunks[-1]["metadata"]["content"].lower()

        user_call = agent._session.add_message.call_args_list[0]
        assert user_call.args[:2] == ("user", "test")
        agent.sessions.save.assert_called()

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
        emitted_text = "hello " * 70

        async def mock_run(**kwargs):
            await agent._agent.output_queue.put({"content": emitted_text})
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
        assert first_chunk["delta"] == emitted_text

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

    def test_non_default_session_key_requires_runtime_clone_support(self):
        """Non-default sessions must not reuse a mock/default runtime unsafely."""
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

        assert response.status_code == 500
        detail = response.json()["detail"]
        assert detail["code"] == "AGENT_ERROR"
        assert "Session runtime cloning is unavailable" in detail["message"]
        mock_agent.process.assert_not_called()

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
