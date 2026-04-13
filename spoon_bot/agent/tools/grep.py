"""Grep tool: search file contents using ripgrep or grep fallback.

Uses subprocess.run in a thread to avoid Windows event-loop issues
with asyncio.create_subprocess_exec (SelectorEventLoop lacks subprocess support).
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from spoon_bot.agent.tools.base import Tool
from spoon_bot.agent.tools.execution_context import capture_tool_output


class GrepTool(Tool):
    """Search file contents using ripgrep (rg) or grep fallback."""

    def __init__(
        self,
        workspace: Path | str | None = None,
        max_output: int = 8000,
    ):
        self._workspace = Path(workspace).resolve() if workspace else None
        self._max_output = max_output

    @property
    def name(self) -> str:
        return "grep"

    @property
    def description(self) -> str:
        return (
            "Search file contents for a pattern using ripgrep or grep. "
            "Returns matching lines with file paths and line numbers. "
            "Use this INSTEAD of reading entire files when you need specific info "
            "(e.g., a contract address, function signature, config value). "
            "Supports regex patterns. Output truncated to limit matches."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (regex or literal string)",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search (default: workspace root)",
                },
                "glob": {
                    "type": "string",
                    "description": "Filter files by glob, e.g. '*.md' or '*.ts'",
                },
                "ignore_case": {
                    "type": "boolean",
                    "description": "Case-insensitive search (default: false)",
                },
                "context": {
                    "type": "integer",
                    "description": "Lines of context before/after match (default: 0)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max matches to return (default: 50)",
                },
            },
            "required": ["pattern"],
        }

    async def execute(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        ignore_case: bool = False,
        context: int = 0,
        limit: int = 50,
        **kwargs: Any,
    ) -> str:
        if not pattern:
            return "Error: pattern is required"

        search_path = path or "."
        if self._workspace and not os.path.isabs(search_path):
            search_path = str(self._workspace / search_path)

        if not os.path.exists(search_path):
            return f"Error: Path not found: {search_path}"

        rg_path = shutil.which("rg")
        grep_path = shutil.which("grep")

        if rg_path:
            args = self._build_rg_args(rg_path, pattern, search_path, glob, ignore_case, context, limit)
        elif grep_path:
            args = self._build_grep_args(grep_path, pattern, search_path, glob, ignore_case, context, limit)
        else:
            return "Error: neither rg nor grep found in PATH"

        try:
            result = await asyncio.to_thread(
                subprocess.run,
                args,
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = result.stdout or ""
        except subprocess.TimeoutExpired:
            return "Error: search timed out after 30s"
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

        if not output.strip():
            return "No matches found"

        lines = output.strip().split("\n")
        if len(lines) > limit:
            output = "\n".join(lines[:limit]) + f"\n... [{len(lines) - limit} more matches]"

        if self._workspace:
            ws_str = str(self._workspace)
            output = output.replace(ws_str + os.sep, "").replace(ws_str + "/", "")

        full_output = output
        if len(output) > self._max_output:
            output = output[: self._max_output] + f"\n... [truncated, {len(output) - self._max_output} more chars]"

        capture_tool_output(output, full_output)
        return output

    @staticmethod
    def _build_rg_args(
        rg_path: str, pattern: str, search_path: str,
        glob: str | None, ignore_case: bool, context: int, limit: int,
    ) -> list[str]:
        args = [rg_path, "--line-number", "--color=never", "--no-heading"]
        if ignore_case:
            args.append("--ignore-case")
        if glob:
            args.append(f"--glob={glob}")
        if context > 0:
            args.extend(["-C", str(context)])
        args.extend(["-m", str(limit), pattern, search_path])
        return args

    @staticmethod
    def _build_grep_args(
        grep_path: str, pattern: str, search_path: str,
        glob: str | None, ignore_case: bool, context: int, limit: int,
    ) -> list[str]:
        args = [grep_path, "-Ern", "--color=never"]
        if ignore_case:
            args.append("-i")
        if context > 0:
            args.extend(["-C", str(context)])
        if glob:
            args.append(f"--include={glob}")
        args.extend(["-e", pattern, "--", search_path])
        return args
