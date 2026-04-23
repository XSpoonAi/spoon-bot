"""Unit-level regression tests for the runtime context-compaction pipeline.

These tests cover the invariants introduced for the "all tool inputs/outputs
stay in context and remain searchable after compression" requirement:

* ``_trim_context_if_needed`` is a no-op (the persisted session store is
  the authoritative transcript and must never be trimmed destructively).
* ``_snap_drop_end_to_turn_boundary`` never splits an
  ``assistant(tool_calls) -> tool(result)`` pair.
* Proactive compression in ``_compress_runtime_context`` only drops on
  safe boundaries and inserts a ``[history-compacted]`` marker.
* ``_force_compress_runtime_context`` honours the same segment-aware rule.

The tests operate on a minimal ``AgentLoop`` built via ``__new__`` plus a
``MagicMock`` memory (same pattern as ``test_runtime_message_sequence``)
so they do not require any LLM / network setup.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_loop(messages: list, context_window: int = 10_000):
    """Construct a bare ``AgentLoop`` with an in-memory messages list."""
    from spoon_bot.agent.loop import AgentLoop

    loop = AgentLoop.__new__(AgentLoop)
    loop.context_window = context_window
    loop._agent = MagicMock()
    loop._agent.memory = MagicMock()
    loop._agent.memory.messages = messages
    return loop


def _roles(messages: list) -> list[str | None]:
    from spoon_bot.agent.loop import AgentLoop

    return [AgentLoop._message_role_value(m) for m in messages]


def test_resolve_context_window_uses_400k_for_gpt54():
    from spoon_bot.config import resolve_context_window

    assert resolve_context_window("gpt-5.4") == 400_000
    assert resolve_context_window("openai/gpt-5.4") == 400_000


@pytest.mark.asyncio
async def test_prepare_request_context_proactively_compacts_near_context_limit():
    from spoon_bot.agent.loop import AgentLoop

    loop = AgentLoop.__new__(AgentLoop)
    loop.context_window = 400_000
    loop.session_key = "ctx-limit"
    loop._agent = MagicMock()
    loop._agent.memory = MagicMock()
    loop._agent.memory.messages = [MagicMock()]
    loop._trim_context_if_needed = MagicMock(return_value=0)
    loop._sync_runtime_history_from_session = AsyncMock(return_value=5)
    loop._normalize_runtime_tool_context = MagicMock(return_value=0)
    loop._estimate_runtime_tokens = MagicMock(side_effect=[390_000, 280_000])
    loop._estimate_token_count = MagicMock(return_value=390_000)
    loop._compress_runtime_context = MagicMock(return_value=3)
    loop._force_compress_runtime_context = MagicMock(return_value=0)

    await AgentLoop._prepare_request_context(loop)

    loop._compress_runtime_context.assert_called_once_with(
        force=True,
        budget_tokens=380_000,
    )
    loop._force_compress_runtime_context.assert_not_called()


@pytest.mark.asyncio
async def test_prepare_request_context_skips_proactive_compaction_until_near_limit():
    from spoon_bot.agent.loop import AgentLoop

    loop = AgentLoop.__new__(AgentLoop)
    loop.context_window = 400_000
    loop.session_key = "ctx-not-near-limit"
    loop._agent = MagicMock()
    loop._agent.memory = MagicMock()
    loop._agent.memory.messages = [MagicMock()]
    loop._trim_context_if_needed = MagicMock(return_value=0)
    loop._sync_runtime_history_from_session = AsyncMock(return_value=5)
    loop._normalize_runtime_tool_context = MagicMock(return_value=0)
    loop._estimate_runtime_tokens = MagicMock(return_value=360_000)
    loop._estimate_token_count = MagicMock(return_value=360_000)
    loop._compress_runtime_context = MagicMock(return_value=0)
    loop._force_compress_runtime_context = MagicMock(return_value=0)

    await AgentLoop._prepare_request_context(loop)

    loop._compress_runtime_context.assert_not_called()
    loop._force_compress_runtime_context.assert_not_called()


@pytest.mark.requires_spoon_core
def test_trim_context_if_needed_is_noop():
    """``_trim_context_if_needed`` must never mutate the session store."""
    from spoon_ai.schema import Message
    from spoon_bot.agent.loop import AgentLoop

    loop = _make_loop([Message(role="user", content=f"m{i}") for i in range(100)])
    # ``_session`` and other session-related attrs are intentionally missing;
    # the method must bail out cleanly regardless.
    result = AgentLoop._trim_context_if_needed(loop)
    assert result == 0


@pytest.mark.requires_spoon_core
def test_snap_drop_end_never_splits_tool_pair():
    """Snapping must land on a user boundary when a tool pair would split."""
    from spoon_ai.schema import Function, Message, ToolCall
    from spoon_bot.agent.loop import AgentLoop

    tc = ToolCall(id="call_a", function=Function(name="shell", arguments="{}"))
    messages = [
        Message(role="system", content="sys"),                      # 0 (head)
        Message(role="user", content="first user"),                 # 1
        Message(role="assistant", content="", tool_calls=[tc]),     # 2
        Message(role="tool", content="done", tool_call_id="call_a", name="shell"),  # 3
        Message(role="user", content="second user"),                # 4
        Message(role="assistant", content="ok"),                    # 5
    ]

    # Desired end = 3 would split the (assistant tool_calls, tool) pair
    # that spans 2 -> 3.  Snap must walk back to the user at index 1.
    snap = AgentLoop._snap_drop_end_to_turn_boundary(messages, keep_head=1, desired_end=3)
    assert snap == 1, f"expected snap back to user@1, got {snap}"

    # Desired end = 4 is allowed: it's exactly the next user turn, and
    # the tool pair at 2-3 is fully contained in the drop range.
    snap2 = AgentLoop._snap_drop_end_to_turn_boundary(messages, keep_head=1, desired_end=4)
    assert snap2 == 4, f"expected snap to user@4, got {snap2}"


@pytest.mark.requires_spoon_core
def test_snap_drop_end_stops_on_settled_assistant_turn():
    """An assistant without tool_calls, not followed by tool, is a safe cut."""
    from spoon_ai.schema import Message
    from spoon_bot.agent.loop import AgentLoop

    messages = [
        Message(role="system", content="sys"),     # 0
        Message(role="user", content="u1"),        # 1
        Message(role="assistant", content="a1"),   # 2  (settled)
        Message(role="user", content="u2"),        # 3
        Message(role="assistant", content="a2"),   # 4
    ]

    # Desired end = 3 lands directly on a user turn (preferred).
    snap = AgentLoop._snap_drop_end_to_turn_boundary(messages, keep_head=1, desired_end=3)
    assert snap == 3

    # Desired end = 2 lands on a settled assistant whose next message is
    # ``user``, so it is also a valid cut.
    snap2 = AgentLoop._snap_drop_end_to_turn_boundary(messages, keep_head=1, desired_end=2)
    assert snap2 == 2


@pytest.mark.requires_spoon_core
def test_insert_compaction_marker_is_runtime_only():
    """The marker lands in runtime memory with role=user and references search_history."""
    from spoon_ai.schema import Message
    from spoon_bot.agent.loop import AgentLoop

    messages = [Message(role="system", content="sys"), Message(role="user", content="u")]
    loop = _make_loop(messages)
    AgentLoop._insert_compaction_marker(loop, messages, dropped=7)

    # Marker is inserted after the head (system) message.
    assert _roles(messages)[:3] == ["system", "user", "user"]
    marker = messages[1]
    content = getattr(marker, "content", "")
    assert "[history-compacted]" in content
    assert "7" in content
    assert "search_history" in content
    assert "latest real user request" in content
    assert "assistant analysis/conclusions" in content


@pytest.mark.requires_spoon_core
def test_compress_runtime_context_neutralizes_old_assistant_conclusions_but_keeps_latest_user_request():
    from spoon_ai.schema import Message
    from spoon_bot.agent.loop import AgentLoop

    latest_request = (
        "Please inspect the repo, follow the latest instruction, and do not stop at a summary. "
        + "z" * 220
    )
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="Earlier request 1 " + "a" * 400),
        Message(
            role="assistant",
            content="Conclusion: the task is already done, so no more tools are needed. " + "b" * 320,
        ),
        Message(role="user", content="Earlier request 2 " + "c" * 400),
        Message(role="assistant", content="Older analysis " + "d" * 320),
        Message(role="user", content="Earlier request 3 " + "e" * 400),
        Message(role="assistant", content="Recent note"),
        Message(role="user", content=latest_request),
        Message(role="assistant", content="Working on it."),
    ]

    loop = _make_loop(messages, context_window=400)
    compressed = AgentLoop._compress_runtime_context(loop, force=True, budget_tokens=1)

    assert compressed > 0
    assert messages[1].content.startswith("[earlier user message compacted]")
    assert messages[2].content.startswith("[assistant reply compacted;")
    assert "Prioritize the latest user request" in messages[2].content
    assert messages[7].content == latest_request


@pytest.mark.requires_spoon_core
def test_force_compress_drops_segment_aware_and_inserts_marker():
    """Force-compression must not split tool pairs and must insert the marker."""
    from spoon_ai.schema import Function, Message, ToolCall
    from spoon_bot.agent.loop import AgentLoop

    def tool_pair(i: int) -> list:
        tc = ToolCall(
            id=f"call_{i}",
            function=Function(name="shell", arguments="{}"),
        )
        return [
            Message(role="user", content=f"u{i} " + "x" * 400),
            Message(role="assistant", content="", tool_calls=[tc]),
            Message(role="tool", content=f"r{i} " + "y" * 400, tool_call_id=f"call_{i}", name="shell"),
        ]

    messages: list = [Message(role="system", content="sys" + "z" * 400)]
    for i in range(6):  # 18 tool-turn messages + 1 system = 19 messages
        messages.extend(tool_pair(i))

    loop = _make_loop(messages, context_window=200)  # force aggressive drops
    before_len = len(messages)

    dropped_actions = AgentLoop._force_compress_runtime_context(loop)

    # Something was dropped or truncated.
    assert dropped_actions > 0

    # No orphan: every ``role=tool`` message must be immediately preceded
    # by an ``assistant`` with tool_calls containing the matching id.
    for idx, msg in enumerate(messages):
        if AgentLoop._message_role_value(msg) != "tool":
            continue
        assert idx > 0, "tool message at position 0 is invalid"
        prev = messages[idx - 1]
        prev_role = AgentLoop._message_role_value(prev)
        assert prev_role == "assistant", (
            f"tool@{idx} not preceded by assistant (got {prev_role})"
        )
        prev_tc = getattr(prev, "tool_calls", None) or []
        ids = {getattr(tc, "id", None) for tc in prev_tc}
        assert getattr(msg, "tool_call_id", None) in ids, (
            f"tool@{idx} has tool_call_id {getattr(msg, 'tool_call_id', None)} "
            f"not present in preceding assistant tool_calls {ids}"
        )

    # If we actually shrunk the list, a compaction marker must exist.
    if len(messages) < before_len:
        marker_present = any(
            getattr(m, "role", None) in ("user", "User")
            and "[history-compacted]" in (getattr(m, "content", "") or "")
            for m in messages
        )
        assert marker_present, "expected [history-compacted] marker in runtime memory"


@pytest.mark.requires_spoon_core
def test_compress_runtime_context_keeps_tool_pairs_atomic():
    """Phase-3 drops in ``_compress_runtime_context`` must not orphan tools."""
    from spoon_ai.schema import Function, Message, ToolCall
    from spoon_bot.agent.loop import AgentLoop

    def tool_pair(i: int) -> list:
        tc = ToolCall(
            id=f"call_{i}",
            function=Function(name="shell", arguments="{}"),
        )
        return [
            Message(role="user", content=f"u{i} " + "x" * 200),
            Message(role="assistant", content="", tool_calls=[tc]),
            Message(role="tool", content=f"r{i} " + "y" * 200, tool_call_id=f"call_{i}", name="shell"),
        ]

    messages: list = [Message(role="system", content="sys")]
    for i in range(8):
        messages.extend(tool_pair(i))

    loop = _make_loop(messages, context_window=400)  # tiny budget -> triggers phase 3
    AgentLoop._compress_runtime_context(loop)

    for idx, msg in enumerate(messages):
        if AgentLoop._message_role_value(msg) != "tool":
            continue
        assert idx > 0, "tool message at position 0 is invalid"
        prev = messages[idx - 1]
        prev_role = AgentLoop._message_role_value(prev)
        assert prev_role == "assistant", (
            f"tool@{idx} not preceded by assistant (got {prev_role})"
        )
        prev_tc = getattr(prev, "tool_calls", None) or []
        ids = {getattr(tc, "id", None) for tc in prev_tc}
        assert getattr(msg, "tool_call_id", None) in ids
