"""Tests for tool message ordering and orphan cleanup in AgentLoop.

Covers:
- _repair_tool_pairing: drops tool messages with missing/unmatched tool_call_id
- _reorder_tool_messages: moves tool results next to their issuing assistant turn
- _normalize_runtime_tool_context: combined repair pipeline
- _uses_strict_tool_turn_order: provider detection
"""

from __future__ import annotations

import pytest

try:
    from spoon_ai.schema import Message, ToolCall as CoreToolCall, Function
    SPOON_CORE_AVAILABLE = True
except ImportError:
    SPOON_CORE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not SPOON_CORE_AVAILABLE, reason="spoon-core SDK not installed"
)


def _msg(role: str, content: str = "", **kwargs) -> Message:
    return Message(role=role, content=content, **kwargs)


def _assistant_with_tools(tool_ids: list[str], content: str = "") -> Message:
    tool_calls = [
        CoreToolCall(
            id=tid,
            type="function",
            function=Function(name=f"tool_{tid}", arguments="{}"),
        )
        for tid in tool_ids
    ]
    return Message(role="assistant", content=content, tool_calls=tool_calls)


def _tool_result(tool_call_id: str, content: str = "result") -> Message:
    return Message(role="tool", content=content, tool_call_id=tool_call_id)


# Lazy import to avoid import-time side effects
def _get_agent_loop_class():
    from spoon_bot.agent.loop import AgentLoop
    return AgentLoop


# ---------------------------------------------------------------------------
# _repair_tool_pairing
# ---------------------------------------------------------------------------

class TestRepairToolPairing:
    def test_drops_tool_message_with_no_tool_call_id(self):
        AgentLoop = _get_agent_loop_class()
        messages = [
            _msg("user", "hello"),
            _msg("tool", "orphan result"),  # no tool_call_id
            _msg("assistant", "hi"),
        ]
        removed = AgentLoop._repair_tool_pairing(messages)
        assert removed >= 1
        roles = [m.role if isinstance(m.role, str) else m.role.value for m in messages]
        assert "tool" not in roles

    def test_drops_tool_message_with_unmatched_tool_call_id(self):
        AgentLoop = _get_agent_loop_class()
        messages = [
            _msg("user", "run"),
            _assistant_with_tools(["call_A"]),
            _tool_result("call_A"),
            _tool_result("call_GHOST"),  # no matching tool_call
        ]
        removed = AgentLoop._repair_tool_pairing(messages)
        assert removed >= 1
        tool_ids = [
            m.tool_call_id for m in messages
            if getattr(m, "tool_call_id", None)
        ]
        assert "call_GHOST" not in tool_ids
        assert "call_A" in tool_ids

    def test_keeps_valid_paired_tool_messages(self):
        AgentLoop = _get_agent_loop_class()
        messages = [
            _msg("user", "run two"),
            _assistant_with_tools(["c1", "c2"]),
            _tool_result("c1"),
            _tool_result("c2"),
            _msg("assistant", "done"),
        ]
        removed = AgentLoop._repair_tool_pairing(messages)
        assert removed == 0
        tool_msgs = [m for m in messages if getattr(m, "tool_call_id", None)]
        assert len(tool_msgs) == 2

    def test_removes_assistant_tool_calls_without_results(self):
        AgentLoop = _get_agent_loop_class()
        messages = [
            _msg("user", "run"),
            _assistant_with_tools(["c1", "c2"]),
            _tool_result("c1"),  # only c1 answered, c2 has no result
        ]
        removed = AgentLoop._repair_tool_pairing(messages)
        assert removed >= 1
        assistant = messages[1]
        remaining_ids = [tc.id for tc in (assistant.tool_calls or [])]
        assert "c2" not in remaining_ids


# ---------------------------------------------------------------------------
# _reorder_tool_messages
# ---------------------------------------------------------------------------

class TestReorderToolMessages:
    def test_moves_tool_result_adjacent_to_assistant(self):
        AgentLoop = _get_agent_loop_class()
        messages = [
            _msg("user", "do something"),
            _assistant_with_tools(["call_X"]),
            _msg("user", "focus please"),  # interleaved user message
            _tool_result("call_X"),
        ]
        moved = AgentLoop._reorder_tool_messages(messages)
        assert moved > 0
        roles = [m.role if isinstance(m.role, str) else m.role.value for m in messages]
        assert roles == ["user", "assistant", "tool", "user"]

    def test_no_change_when_already_ordered(self):
        AgentLoop = _get_agent_loop_class()
        messages = [
            _msg("user", "run"),
            _assistant_with_tools(["c1"]),
            _tool_result("c1"),
            _msg("assistant", "done"),
        ]
        moved = AgentLoop._reorder_tool_messages(messages)
        assert moved == 0


# ---------------------------------------------------------------------------
# _normalize_runtime_tool_context
# ---------------------------------------------------------------------------

class TestNormalizeRuntimeToolContext:
    def test_combined_reorder_and_repair(self):
        AgentLoop = _get_agent_loop_class()
        messages = [
            _msg("user", "do it"),
            _assistant_with_tools(["c1"]),
            _msg("user", "focus"),
            _tool_result("c1"),
            _msg("tool", "orphan"),  # no tool_call_id
        ]
        normalized = AgentLoop._normalize_runtime_tool_context(messages)
        assert normalized > 0
        roles = [m.role if isinstance(m.role, str) else m.role.value for m in messages]
        assert roles == ["user", "assistant", "tool", "user"]
        tool_msgs = [m for m in messages if (m.role if isinstance(m.role, str) else m.role.value) == "tool"]
        assert all(getattr(m, "tool_call_id", None) for m in tool_msgs)


# ---------------------------------------------------------------------------
# _uses_strict_tool_turn_order
# ---------------------------------------------------------------------------

class TestUsesStrictToolTurnOrder:
    def _make_loop_stub(self, provider: str, model: str = "", base_url: str = ""):
        """Create a minimal object with the attributes _uses_strict_tool_turn_order reads."""
        AgentLoop = _get_agent_loop_class()

        class Stub:
            pass

        stub = Stub()
        stub.provider = provider
        stub.model = model
        stub.base_url = base_url
        stub._uses_strict_tool_turn_order = AgentLoop._uses_strict_tool_turn_order.__get__(stub)
        return stub

    def test_openai_is_strict(self):
        stub = self._make_loop_stub(provider="openai")
        assert stub._uses_strict_tool_turn_order() is True

    def test_openrouter_is_strict(self):
        stub = self._make_loop_stub(provider="openrouter")
        assert stub._uses_strict_tool_turn_order() is True

    def test_gemini_is_strict(self):
        stub = self._make_loop_stub(provider="gemini")
        assert stub._uses_strict_tool_turn_order() is True

    def test_gpt_model_is_strict(self):
        stub = self._make_loop_stub(provider="custom", model="gpt-5.2")
        assert stub._uses_strict_tool_turn_order() is True

    def test_o3_model_is_strict(self):
        stub = self._make_loop_stub(provider="custom", model="o3-mini")
        assert stub._uses_strict_tool_turn_order() is True

    def test_openai_base_url_is_strict(self):
        stub = self._make_loop_stub(provider="custom", base_url="https://api.openai.com/v1")
        assert stub._uses_strict_tool_turn_order() is True

    def test_anthropic_not_strict(self):
        stub = self._make_loop_stub(provider="anthropic", model="claude-sonnet-4.5")
        assert stub._uses_strict_tool_turn_order() is False

    def test_deepseek_not_strict(self):
        stub = self._make_loop_stub(provider="deepseek", model="deepseek-chat")
        assert stub._uses_strict_tool_turn_order() is False
