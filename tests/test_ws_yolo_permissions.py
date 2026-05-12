from __future__ import annotations

from spoon_bot.core import SpoonBotConfig
from spoon_bot.gateway.websocket.agent import ToolPermission, WSDialogueAgent


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
