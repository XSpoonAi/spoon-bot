from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.mark.requires_spoon_core
def test_compress_runtime_context_reorders_tool_messages_before_next_step_prompt():
    from spoon_ai.schema import Function, Message, ToolCall
    from spoon_bot.agent.loop import AgentLoop

    assistant_tool_call = ToolCall(
        id="call_1",
        function=Function(name="shell", arguments="{}"),
    )
    runtime_messages = [
        Message(role="user", content="Check the workspace"),
        Message(role="assistant", content="", tool_calls=[assistant_tool_call]),
        Message(role="user", content="Focus on the user and continue."),
        Message(role="tool", content="done", tool_call_id="call_1", name="shell"),
    ]

    loop = AgentLoop.__new__(AgentLoop)
    loop.context_window = 10_000
    loop._agent = MagicMock()
    loop._agent.memory = MagicMock()
    loop._agent.memory.messages = runtime_messages

    actions = AgentLoop._compress_runtime_context(loop)

    assert [AgentLoop._message_role_value(msg) for msg in runtime_messages] == [
        "user",
        "assistant",
        "tool",
        "user",
    ]
    assert runtime_messages[2].tool_call_id == "call_1"
    assert actions >= 1


@pytest.mark.requires_spoon_core
@pytest.mark.asyncio
async def test_install_anti_loop_tracker_suppresses_prompt_for_strict_tool_turn_providers():
    from spoon_ai.schema import Function, Message, ToolCall
    from spoon_bot.agent.loop import AgentLoop

    seen_prompts: list[str | None] = []

    async def base_think() -> bool:
        seen_prompts.append(agent._agent.next_step_prompt)
        return True

    assistant_tool_call = ToolCall(
        id="call_1",
        function=Function(name="shell", arguments="{}"),
    )

    agent = AgentLoop.__new__(AgentLoop)
    agent.workspace = Path("/workspace")
    agent.provider = "openrouter"
    agent.model = "google/gemini-2.5-pro"
    agent.base_url = None
    agent._agent = MagicMock()
    agent._agent.think = base_think
    agent._agent._spoon_bot_base_think = base_think
    agent._agent.next_step_prompt = "prompt"
    agent._agent.tool_calls = []
    agent._agent.memory = MagicMock()
    agent._agent.memory.messages = [
        Message(role="assistant", content="", tool_calls=[assistant_tool_call]),
        Message(role="tool", content="done", tool_call_id="call_1", name="shell"),
    ]
    agent._compress_runtime_context = MagicMock(return_value=0)

    AgentLoop._install_anti_loop_tracker(agent, "prompt")
    await agent._agent.think()

    assert seen_prompts == [None]
    assert agent._agent.next_step_prompt == "prompt"


@pytest.mark.requires_spoon_core
@pytest.mark.asyncio
async def test_install_anti_loop_tracker_keeps_prompt_for_non_strict_providers():
    from spoon_ai.schema import Function, Message, ToolCall
    from spoon_bot.agent.loop import AgentLoop

    seen_prompts: list[str | None] = []

    async def base_think() -> bool:
        seen_prompts.append(agent._agent.next_step_prompt)
        return True

    assistant_tool_call = ToolCall(
        id="call_1",
        function=Function(name="shell", arguments="{}"),
    )

    agent = AgentLoop.__new__(AgentLoop)
    agent.workspace = Path("/workspace")
    agent.provider = "anthropic"
    agent.model = "claude-sonnet-4.5"
    agent.base_url = None
    agent._agent = MagicMock()
    agent._agent.think = base_think
    agent._agent._spoon_bot_base_think = base_think
    agent._agent.next_step_prompt = "prompt"
    agent._agent.tool_calls = []
    agent._agent.memory = MagicMock()
    agent._agent.memory.messages = [
        Message(role="assistant", content="", tool_calls=[assistant_tool_call]),
        Message(role="tool", content="done", tool_call_id="call_1", name="shell"),
    ]
    agent._compress_runtime_context = MagicMock(return_value=0)

    AgentLoop._install_anti_loop_tracker(agent, "prompt")
    await agent._agent.think()

    assert seen_prompts == ["prompt"]
    assert agent._agent.next_step_prompt == "prompt"
