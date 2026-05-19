from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from spoon_bot.agent.tools.base import Tool, ToolParameterSchema
from spoon_bot.agent.tools.execution_context import track_tool_invocations
from spoon_bot.agent.tools.filesystem import EditFileTool, ReadFileTool


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


class PollingTool(CountingTool):
    @property
    def name(self) -> str:
        return "polling_tool"

    def tool_invocation_dedup_key(self, kwargs: dict[str, Any]) -> dict[str, Any] | None:
        if kwargs.get("action") == "status":
            return None
        return kwargs


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
async def test_tool_can_opt_out_of_exact_duplicate_dedup_for_polling() -> None:
    tool = PollingTool()

    with track_tool_invocations(max_repeats=1):
        first = await tool(value="alpha", action="status")
        second = await tool(value="alpha", action="status")
        execute = await tool(value="alpha", action="execute")
        duplicate_execute = await tool(value="alpha", action="execute")

    assert first == "executed:alpha:1"
    assert second == "executed:alpha:2"
    assert execute == "executed:alpha:3"
    assert "STOP_TOOL_LOOP" in duplicate_execute
    assert tool.calls == 3


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


@pytest.mark.asyncio
async def test_track_tool_invocations_suppresses_redundant_file_ranges(tmp_path: Path) -> None:
    file_path = tmp_path / "app.py"
    file_path.write_text("line1\nline2\nline3\nline4\n", encoding="utf-8")
    tool = ReadFileTool(workspace=tmp_path)

    with track_tool_invocations():
        first = await tool.execute(path=str(file_path))
        duplicate_range = await tool.execute(path=str(file_path), offset=2, limit=2)

    with track_tool_invocations():
        partial = await tool.execute(path=str(file_path), offset=1, limit=2)
        uncovered_range = await tool.execute(path=str(file_path), offset=4, limit=1)

    assert "line2" in first
    assert "STOP_TOOL_LOOP" in duplicate_range
    assert "redundant file read suppressed" in duplicate_range
    assert "line4" not in duplicate_range
    assert "line1" in partial
    assert "line4" in uncovered_range


@pytest.mark.asyncio
async def test_truncated_file_read_only_covers_visible_complete_lines(tmp_path: Path) -> None:
    file_path = tmp_path / "large.txt"
    file_path.write_text("line1\nline2\nline3\nline4\n", encoding="utf-8")
    tool = ReadFileTool(workspace=tmp_path, max_output=9)

    with track_tool_invocations():
        first = await tool.execute(path=str(file_path))
        hidden_range = await tool.execute(path=str(file_path), offset=3, limit=1)
        visible_range = await tool.execute(path=str(file_path), offset=1, limit=1)

    assert "... (truncated" in first
    assert "line3" in hidden_range
    assert "STOP_TOOL_LOOP" in visible_range
    assert "redundant file read suppressed" in visible_range


@pytest.mark.asyncio
async def test_file_edit_invalidates_read_dedup_for_verification(tmp_path: Path) -> None:
    file_path = tmp_path / "settings.txt"
    file_path.write_text("original=keep\ncolor = red\n", encoding="utf-8")
    read_tool = ReadFileTool(workspace=tmp_path)
    edit_tool = EditFileTool(workspace=tmp_path)

    with track_tool_invocations():
        first = await read_tool(path=str(file_path))
        duplicate_before_edit = await read_tool(path=str(file_path))
        edited = await edit_tool(
            path=str(file_path),
            old_text="color = red",
            new_text="color = blue",
        )
        verify = await read_tool(path=str(file_path))

    assert "color = red" in first
    assert "STOP_TOOL_LOOP" in duplicate_before_edit
    assert "Successfully edited" in edited
    assert "STOP_TOOL_LOOP" not in verify
    assert "color = blue" in verify


@pytest.mark.asyncio
async def test_changed_file_content_allows_same_range_read(tmp_path: Path) -> None:
    file_path = tmp_path / "settings.txt"
    file_path.write_text("color = red\n", encoding="utf-8")
    read_tool = ReadFileTool(workspace=tmp_path)

    with track_tool_invocations():
        first = await read_tool.execute(path=str(file_path))
        duplicate = await read_tool.execute(path=str(file_path))
        file_path.write_text("color = blue\n", encoding="utf-8")
        changed = await read_tool.execute(path=str(file_path))

    assert "color = red" in first
    assert "STOP_TOOL_LOOP" in duplicate
    assert "STOP_TOOL_LOOP" not in changed
    assert "color = blue" in changed
