from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_read_file_tool_records_full_output_for_stream_capture(tmp_path: Path):
    from spoon_bot.agent.tools.execution_context import (
        bind_tool_invocation,
        capture_tool_outputs,
        consume_captured_tool_output,
        finalize_tool_invocation,
    )
    from spoon_bot.agent.tools.filesystem import ReadFileTool

    file_path = tmp_path / "long.txt"
    file_path.write_text("A" * 120, encoding="utf-8")

    tool = ReadFileTool(workspace=tmp_path, max_output=40)

    with capture_tool_outputs() as scope_id:
        with bind_tool_invocation("read_file", {"path": str(file_path)}):
            result = await tool.execute(path=str(file_path))
            finalize_tool_invocation(result)

        captured = consume_captured_tool_output(
            scope_id,
            tool_name="read_file",
            arguments={"path": str(file_path)},
        )

    assert captured is not None
    assert "... (truncated," in result
    assert "... (truncated," in captured.summary_output
    assert "... (truncated," not in captured.full_output
    assert "A" * 120 in captured.full_output


@pytest.mark.asyncio
async def test_capture_defaults_to_returned_result_when_tool_does_not_publish_full_output():
    from spoon_bot.agent.tools.execution_context import (
        bind_tool_invocation,
        capture_tool_outputs,
        consume_captured_tool_output,
        finalize_tool_invocation,
    )

    with capture_tool_outputs() as scope_id:
        with bind_tool_invocation("dummy_tool", {"x": 1}):
            finalize_tool_invocation("plain result")

        captured = consume_captured_tool_output(
            scope_id,
            tool_name="dummy_tool",
            arguments={"x": 1},
        )

    assert captured is not None
    assert captured.summary_output == "plain result"
    assert captured.full_output == "plain result"


def test_skill_execution_summary_marks_optional_input_as_non_blocking():
    from spoon_bot.agent.tools.filesystem import ReadFileTool

    summary = ReadFileTool._extract_skill_cli_content(
        """
---
name: sample-skill
---

## Setup

```text
PROCEDURE setup():
  IF account is missing:
    ask the user whether they have optional configuration first
    RUN cli setup
```
""",
        budget=4000,
    )

    assert "Non-interactive optional-input precedence" in summary
    assert "default/no-extra path" in summary
    assert "Omitted `ask ...` directives" in summary
    assert "Do not emit optional-input questions as visible progress" in summary
    assert "ask the user whether they have optional configuration first" not in summary
