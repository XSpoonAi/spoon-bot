from __future__ import annotations

from typing import Any

import pytest

from spoon_bot.agent.tools.base import Tool, ToolParameterSchema
from spoon_bot.agent.tools.execution_context import track_tool_invocations


class CountingTool(Tool):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def name(self) -> str:
        return "counting_tool"

    @property
    def description(self) -> str:
        return "Counts real executions."

    @property
    def parameters(self) -> ToolParameterSchema:
        return {
            "type": "object",
            "properties": {
                "value": {"type": "string", "description": "Value to echo"},
            },
            "required": ["value"],
        }

    async def execute(self, **kwargs: Any) -> str:
        self.calls += 1
        return f"executed:{kwargs['value']}:{self.calls}"


@pytest.mark.asyncio
async def test_track_tool_invocations_suppresses_exact_repeated_call() -> None:
    tool = CountingTool()

    with track_tool_invocations(max_repeats=1):
        first = await tool(value="alpha")
        duplicate = await tool(value="alpha")
        different = await tool(value="beta")

    assert first == "executed:alpha:1"
    assert "duplicate tool invocation suppressed" in duplicate
    assert different == "executed:beta:2"
    assert tool.calls == 2


@pytest.mark.asyncio
async def test_tool_calls_are_not_deduped_without_request_scope() -> None:
    tool = CountingTool()

    first = await tool(value="alpha")
    second = await tool(value="alpha")

    assert first == "executed:alpha:1"
    assert second == "executed:alpha:2"
    assert tool.calls == 2


@pytest.mark.asyncio
async def test_default_request_scope_allows_one_intentional_rerun() -> None:
    tool = CountingTool()

    with track_tool_invocations():
        first = await tool(value="alpha")
        second = await tool(value="alpha")
        third = await tool(value="alpha")

    assert first == "executed:alpha:1"
    assert second == "executed:alpha:2"
    assert "duplicate tool invocation suppressed" in third
    assert tool.calls == 2
