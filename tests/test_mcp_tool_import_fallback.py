from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


def _reload_with_missing_mcp_tool(module_name: str) -> dict[str, object]:
    target = importlib.import_module(module_name)

    with pytest.MonkeyPatch.context() as mp:
        fake_mcp_tool = types.ModuleType("spoon_ai.tools.mcp_tool")
        fake_mcp_tool.__file__ = "<fake spoon_ai.tools.mcp_tool>"
        mp.setitem(sys.modules, "spoon_ai.tools.mcp_tool", fake_mcp_tool)

        reloaded = importlib.reload(target)
        observed = {
            "module": reloaded,
            "mcp_tool_available": getattr(reloaded, "MCP_TOOL_AVAILABLE", None),
            "mcp_tool_is_none": getattr(reloaded, "MCPTool", object()) is None,
        }

    importlib.reload(target)
    return observed


def test_agent_loop_module_loads_when_mcp_tool_import_fails():
    observed = _reload_with_missing_mcp_tool("spoon_bot.agent.loop")
    assert observed["mcp_tool_available"] is False
    assert observed["mcp_tool_is_none"] is True


def test_core_module_loads_when_mcp_tool_import_fails():
    observed = _reload_with_missing_mcp_tool("spoon_bot.core")
    assert observed["mcp_tool_available"] is False
    assert observed["mcp_tool_is_none"] is True


def test_gateway_core_integration_loads_when_mcp_tool_import_fails():
    observed = _reload_with_missing_mcp_tool("spoon_bot.gateway.core_integration")
    assert observed["mcp_tool_available"] is False
    assert observed["mcp_tool_is_none"] is True


@pytest.mark.asyncio
async def test_agent_loop_skips_mcp_servers_when_mcp_tool_unavailable(tmp_path: Path):
    loop_mod = importlib.import_module("spoon_bot.agent.loop")

    class _FakeChatBot:
        def __init__(self, *args, **kwargs):
            pass

    class _FakeAgent:
        def __init__(self, *args, **kwargs):
            self.available_tools = kwargs.get("tools")

        async def initialize(self):
            return None

    with pytest.MonkeyPatch.context() as mp:
        fake_mcp_tool = types.ModuleType("spoon_ai.tools.mcp_tool")
        fake_mcp_tool.__file__ = "<fake spoon_ai.tools.mcp_tool>"
        mp.setitem(sys.modules, "spoon_ai.tools.mcp_tool", fake_mcp_tool)

        loop_mod = importlib.reload(loop_mod)
        mp.setattr(loop_mod, "ChatBot", _FakeChatBot)
        mp.setattr(loop_mod, "SpoonReactMCP", _FakeAgent)
        mp.setattr(loop_mod, "SpoonReactSkill", _FakeAgent)

        agent = loop_mod.AgentLoop(
            workspace=tmp_path / "workspace",
            model="gpt-4o-mini",
            provider="openai",
            enable_skills=False,
            auto_commit=False,
            mcp_config={"demo": {"transport": "websocket", "url": "ws://127.0.0.1:1"}},
        )
        await agent.initialize()

        assert loop_mod.MCP_TOOL_AVAILABLE is False
        assert agent._mcp_tools == []

    importlib.reload(loop_mod)
