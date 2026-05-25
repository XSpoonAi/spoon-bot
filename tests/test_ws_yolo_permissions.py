from __future__ import annotations

import asyncio

import pytest

from spoon_bot.core import SpoonBotConfig
from spoon_bot.gateway.websocket.agent import AgentState, ToolPermission, WSDialogueAgent


def test_ws_dialogue_agent_allows_shell_permissions_in_yolo_mode() -> None:
    agent = WSDialogueAgent(
        "session-yolo",
        config=SpoonBotConfig(yolo_mode=True),
    )

    assert agent.get_tool_permission("shell_execute") == ToolPermission.ALLOW
    assert agent.get_tool_permission("file_delete") == ToolPermission.ALLOW


def test_ws_dialogue_agent_keeps_confirmation_defaults_without_yolo_mode() -> None:
    agent = WSDialogueAgent(
        "session-normal",
        config=SpoonBotConfig(yolo_mode=False),
    )

    assert agent.get_tool_permission("shell_execute") == ToolPermission.CONFIRM
    assert agent.get_tool_permission("file_delete") == ToolPermission.CONFIRM


@pytest.mark.asyncio
async def test_ws_confirmation_pauses_until_user_response() -> None:
    events: list[tuple[str, dict]] = []

    async def emit(event: str, data: dict) -> None:
        events.append((event, data))

    agent = WSDialogueAgent(
        "session-confirm",
        config=SpoonBotConfig(yolo_mode=False),
        event_callback=emit,
    )

    pending = asyncio.create_task(
        agent.request_confirmation(
            action="shell_execute",
            description="Run shell command",
            tool_name="shell",
            arguments={"command": "echo ok"},
            timeout_seconds=5,
        )
    )

    for _ in range(20):
        if events:
            break
        await asyncio.sleep(0)

    assert events
    assert events[0][0] == "confirm.request"
    assert agent.state == AgentState.WAITING_CONFIRM
    assert not pending.done()

    request_id = events[0][1]["request_id"]
    response = await agent.respond_confirmation(request_id, approved=True)

    assert response["success"] is True
    assert await pending is True
    assert agent.state == AgentState.STREAMING
