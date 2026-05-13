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


class SeriesTool(CountingTool):
    @property
    def name(self) -> str:
        return "series_tool"

    def tool_invocation_series_key(self, kwargs: dict[str, Any]) -> str | None:
        return kwargs.get("series")


class FailingTool(CountingTool):
    @property
    def name(self) -> str:
        return "failing_tool"

    async def execute(self, **kwargs: Any) -> str:
        self.calls += 1
        return f"provider {kwargs['value']} request failed\nExit code: 1"


@pytest.mark.asyncio
async def test_track_tool_invocations_suppresses_exact_repeated_call() -> None:
    tool = CountingTool()

    with track_tool_invocations(max_repeats=1):
        first = await tool(value="alpha")
        duplicate = await tool(value="alpha")
        different = await tool(value="beta")

    assert first == "executed:alpha:1"
    assert "STOP_TOOL_LOOP" in duplicate
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
    assert "STOP_TOOL_LOOP" in third
    assert "duplicate tool invocation suppressed" in third
    assert tool.calls == 2


@pytest.mark.asyncio
async def test_track_tool_invocations_suppresses_repeated_side_effect_series() -> None:
    tool = SeriesTool()

    with track_tool_invocations(max_series_repeats=2):
        first = await tool(value="alpha", series="external-submit")
        second = await tool(value="beta", series="external-submit")
        third = await tool(value="gamma", series="external-submit")
        different = await tool(value="delta", series="external-verify")

    assert first == "executed:alpha:1"
    assert second == "executed:beta:2"
    assert "STOP_TOOL_LOOP" in third
    assert "repeated side-effecting tool series suppressed" in third
    assert different == "executed:delta:3"
    assert tool.calls == 3


@pytest.mark.asyncio
async def test_track_tool_invocations_suppresses_consecutive_failures() -> None:
    tool = FailingTool()

    with track_tool_invocations(max_consecutive_failures=3):
        first = await tool(value="alpha")
        second = await tool(value="beta")
        third = await tool(value="gamma")
        fourth = await tool(value="delta")

    assert "provider alpha request failed" in first
    assert "provider beta request failed" in second
    assert "provider gamma request failed" in third
    assert "STOP_TOOL_LOOP" in fourth
    assert "consecutive tool failures suppressed" in fourth
    assert tool.calls == 3
