"""Tests for tool message ordering and orphan cleanup in AgentLoop.

Covers:
- _repair_tool_pairing: drops tool messages with missing/unmatched tool_call_id
- _reorder_tool_messages: moves tool results next to their issuing assistant turn
- _normalize_runtime_tool_context: combined repair pipeline without pruning live pending calls
- _uses_strict_tool_turn_order: provider detection
"""

from __future__ import annotations

from types import SimpleNamespace

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

    def test_moves_tool_result_persisted_before_assistant(self):
        AgentLoop = _get_agent_loop_class()
        messages = [
            _msg("user", "run two"),
            _tool_result("c2", "second result was persisted early"),
            _assistant_with_tools(["c1", "c2"]),
            _tool_result("c1", "first result"),
            _msg("assistant", "done"),
        ]

        moved = AgentLoop._reorder_tool_messages(messages)

        assert moved > 0
        roles = [m.role if isinstance(m.role, str) else m.role.value for m in messages]
        assert roles == ["user", "assistant", "tool", "tool", "assistant"]
        assert [m.tool_call_id for m in messages[2:4]] == ["c1", "c2"]

    def test_reorders_dict_runtime_messages(self):
        AgentLoop = _get_agent_loop_class()
        messages = [
            {"role": "user", "content": "run"},
            {"role": "tool", "tool_call_id": "dict_call", "content": "result"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "dict_call",
                        "type": "function",
                        "function": {"name": "shell", "arguments": "{}"},
                    }
                ],
            },
        ]

        moved = AgentLoop._reorder_tool_messages(messages)

        assert moved > 0
        assert [m["role"] for m in messages] == ["user", "assistant", "tool"]
        assert messages[2]["tool_call_id"] == "dict_call"


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

    def test_runtime_normalization_keeps_pending_tool_calls(self):
        AgentLoop = _get_agent_loop_class()
        messages = [
            _msg("user", "run"),
            _assistant_with_tools(["c1", "c2"]),
            _tool_result("c1"),
        ]

        normalized = AgentLoop._normalize_runtime_tool_context(messages)

        assert normalized == 0
        remaining_ids = [tc.id for tc in (messages[1].tool_calls or [])]
        assert remaining_ids == ["c1", "c2"]

    def test_finalized_normalization_trims_unanswered_tool_calls(self):
        AgentLoop = _get_agent_loop_class()
        messages = [
            _msg("user", "run"),
            _assistant_with_tools(["c1", "c2"]),
            _tool_result("c1"),
        ]

        normalized = AgentLoop._normalize_runtime_tool_context(messages, finalized=True)

        assert normalized >= 1
        remaining_ids = [tc.id for tc in (messages[1].tool_calls or [])]
        assert remaining_ids == ["c1"]

    def test_compaction_boundary_moves_back_to_preserve_matching_assistant(self):
        AgentLoop = _get_agent_loop_class()
        messages = [
            _msg("system", "system"),
            _msg("user", "older request"),
            _assistant_with_tools(["c1", "c2"]),
            _tool_result("c1"),
            _msg("assistant", "done"),
        ]

        adjusted = AgentLoop._adjust_message_start_to_preserve_tool_context(
            messages,
            3,
            floor=1,
        )

        assert adjusted == 2

    def test_persist_turn_tool_trace_normalizes_before_saving(self):
        AgentLoop = _get_agent_loop_class()

        saved: list[tuple[str, str, dict]] = []

        class DummySession:
            def add_message(self, role: str, content: str, **kwargs):
                saved.append((role, content, kwargs))

        stub = SimpleNamespace(
            _agent=SimpleNamespace(
                memory=SimpleNamespace(
                    messages=[
                        _msg("user", "run two"),
                        _tool_result("c2", "second result was persisted early"),
                        _assistant_with_tools(["c1", "c2"]),
                        _tool_result("c1", "first result"),
                    ]
                )
            ),
            _session=DummySession(),
        )
        stub._should_persist_tool_trace = lambda: True
        stub._capture_turn_tool_trace = AgentLoop._capture_turn_tool_trace.__get__(stub)

        persisted = AgentLoop._persist_turn_tool_trace(stub, 0)

        assert persisted == 3
        assert [entry[0] for entry in saved] == ["assistant", "tool", "tool"]
        assert [entry[2].get("tool_call_id") for entry in saved[1:]] == ["c1", "c2"]

    def test_normalize_persisted_session_removes_unrecoverable_orphans(self):
        AgentLoop = _get_agent_loop_class()

        class DummySessions:
            def __init__(self):
                self.saved = 0

            def save(self, session):
                self.saved += 1

        session = SimpleNamespace(
            messages=[
                {"role": "user", "content": "run"},
                {"role": "tool", "tool_call_id": "ghost", "content": "orphan"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "ok",
                            "type": "function",
                            "function": {"name": "shell", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "ok", "content": "result"},
            ]
        )
        sessions = DummySessions()
        stub = SimpleNamespace(_session=session, sessions=sessions)

        repaired = AgentLoop._normalize_persisted_session_tool_context(stub)

        assert repaired > 0
        assert sessions.saved == 1
        assert [m["role"] for m in session.messages] == ["user", "assistant", "tool"]
        assert session.messages[2]["tool_call_id"] == "ok"

    def test_pre_run_normalization_repairs_after_current_user_inserted(self):
        AgentLoop = _get_agent_loop_class()
        messages = [
            _msg("user", "previous task"),
            _tool_result("c1", "persisted tool result was early"),
            _assistant_with_tools(["c1"]),
            _msg("user", "current request"),
        ]
        stub = SimpleNamespace(
            _agent=SimpleNamespace(memory=SimpleNamespace(messages=messages))
        )

        repaired = AgentLoop._normalize_runtime_memory_before_run(stub, "test")

        assert repaired > 0
        assert [
            m.role if isinstance(m.role, str) else m.role.value for m in messages
        ] == ["user", "assistant", "tool", "user"]
        assert messages[-1].content == "current request"

    @pytest.mark.asyncio
    async def test_provider_run_adds_continuation_user_after_assistant_tail(self):
        AgentLoop = _get_agent_loop_class()

        class DummyAgent:
            def __init__(self):
                self.memory = SimpleNamespace(
                    messages=[
                        _msg("user", "previous request"),
                        _msg("assistant", "previous answer"),
                    ]
                )
                self.next_step_prompt = "Continue with the latest task."
                self.run_tail_roles: list[str] = []

            async def add_message(self, role: str, content: str, **kwargs):
                self.memory.messages.append(_msg(role, content, **kwargs))

            async def run(self, **kwargs):
                self.run_tail_roles = [
                    msg.role if isinstance(msg.role, str) else msg.role.value
                    for msg in self.memory.messages
                ]
                return "ok"

        agent = DummyAgent()
        stub = SimpleNamespace(_agent=agent, session_key="tail-test", user_id=None)
        stub._current_tool_owner_key = AgentLoop._current_tool_owner_key.__get__(stub)

        result = await AgentLoop._run_agent_with_retry(stub, label="test")

        assert result == "ok"
        assert agent.run_tail_roles[-1] == "user"
        assert agent.memory.messages[-1].content == "Continue with the latest task."

    @pytest.mark.asyncio
    async def test_provider_run_does_not_duplicate_existing_user_tail(self):
        AgentLoop = _get_agent_loop_class()

        class DummyAgent:
            def __init__(self):
                self.memory = SimpleNamespace(messages=[_msg("user", "current request")])
                self.next_step_prompt = "Should not be added."

            async def add_message(self, role: str, content: str, **kwargs):
                self.memory.messages.append(_msg(role, content, **kwargs))

            async def run(self, **kwargs):
                return len(self.memory.messages)

        agent = DummyAgent()
        stub = SimpleNamespace(_agent=agent, session_key="tail-test", user_id=None)
        stub._current_tool_owner_key = AgentLoop._current_tool_owner_key.__get__(stub)

        result = await AgentLoop._run_agent_with_retry(stub, label="test")

        assert result == 1
        assert [msg.content for msg in agent.memory.messages] == ["current request"]

    def test_runtime_compaction_preserves_tool_trajectory_and_pending_calls(self):
        AgentLoop = _get_agent_loop_class()
        loop = AgentLoop.__new__(AgentLoop)
        loop.context_window = 64
        loop._agent = SimpleNamespace(
            memory=SimpleNamespace(
                messages=[
                    _msg("system", "system prompt"),
                    _msg("user", "old request 1"),
                    _msg("assistant", "old answer 1"),
                    _msg("user", "old request 2"),
                    _assistant_with_tools(["keep_1", "keep_2"]),
                    _tool_result("keep_1", "partial result"),
                    _msg("assistant", "mid answer"),
                    _msg("user", "old request 3"),
                    _msg("assistant", "old answer 3"),
                    _msg("user", "old request 4"),
                    _msg("assistant", "old answer 4"),
                    _msg("user", "old request 5"),
                    _msg("assistant", "old answer 5"),
                    _msg("user", "old request 6"),
                    _msg("assistant", "old answer 6"),
                    _msg("user", "old request 7"),
                    _msg("assistant", "latest answer"),
                ]
            )
        )

        compressed = AgentLoop._compress_runtime_context(
            loop,
            force=True,
            budget_tokens=1,
        )

        assert compressed > 0
        assistant_index = next(
            index
            for index, message in enumerate(loop._agent.memory.messages)
            if (message.role if isinstance(message.role, str) else message.role.value) == "assistant"
            and getattr(message, "tool_calls", None)
        )
        assistant = loop._agent.memory.messages[assistant_index]
        tool_result = loop._agent.memory.messages[assistant_index + 1]
        assert [tc.id for tc in (assistant.tool_calls or [])] == ["keep_1", "keep_2"]
        assert tool_result.tool_call_id == "keep_1"

    def test_incomplete_turn_marker_closes_dangling_user(self):
        AgentLoop = _get_agent_loop_class()

        class DummySession:
            def __init__(self):
                self.messages = [{"role": "user", "content": "previous failed request"}]

            def add_message(self, role: str, content: str, **kwargs):
                self.messages.append({"role": role, "content": content, **kwargs})

        class DummySessions:
            def __init__(self):
                self.saved = 0

            def save(self, session):
                self.saved += 1

        session = DummySession()
        sessions = DummySessions()
        stub = SimpleNamespace(_session=session, sessions=sessions)

        AgentLoop._persist_incomplete_turn_marker(
            stub,
            label="process",
            reason=RuntimeError("network"),
        )
        AgentLoop._persist_incomplete_turn_marker(
            stub,
            label="process",
            reason=RuntimeError("network"),
        )

        assert [m["role"] for m in session.messages] == ["user", "assistant"]
        assert session.messages[1]["incomplete"] is True
        assert session.messages[1]["incomplete_source"] == "process"
        assert "RuntimeError" in session.messages[1]["content"]
        assert sessions.saved == 1


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

    def test_deepseek_is_strict(self):
        stub = self._make_loop_stub(provider="deepseek", model="deepseek-chat")
        assert stub._uses_strict_tool_turn_order() is True
