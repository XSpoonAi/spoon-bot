"""File system tools: read, write, edit, list.

All tools enforce workspace boundary security to prevent path traversal attacks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import aiofiles
import aiofiles.os

from spoon_bot.agent.tools.base import Tool
from spoon_bot.agent.tools.path_validator import (
    PathValidator,
    validate_read_path,
    validate_write_path,
    validate_directory_path,
    set_default_validator,
)


class ReadFileTool(Tool):
    """Tool to read file contents with encoding fallback and path traversal protection."""

    def __init__(
        self,
        workspace: Path | str | None = None,
        additional_read_paths: list[Path | str] | None = None,
        max_output: int = 6000,
    ):
        self._workspace = Path(workspace).resolve() if workspace else None
        self._additional_read_paths = additional_read_paths
        self._max_output = max_output
        self._validator = (
            PathValidator(workspace=workspace, additional_read_paths=additional_read_paths)
            if workspace else None
        )

    def set_workspace(self, workspace: Path | str) -> None:
        """Set the workspace boundary for path validation."""
        self._workspace = Path(workspace).resolve()
        self._validator = PathValidator(
            workspace=self._workspace,
            additional_read_paths=self._additional_read_paths,
        )
        set_default_validator(self._validator)

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        workspace_note = ""
        if self._workspace:
            workspace_note = f" Files must be within the workspace: {self._workspace}"
        return (
            f"Read file contents.{workspace_note} "
            "Use offset+limit for large files (line-based). "
            "For searching specific values, prefer `grep` tool instead."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to read (must be within workspace)",
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start from (1-indexed). Omit to read from start.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max lines to read. Omit for whole file (subject to truncation).",
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding (default: utf-8)",
                },
            },
            "required": ["path"],
        }

    async def execute(
        self,
        path: str,
        offset: int | None = None,
        limit: int | None = None,
        encoding: str = "utf-8",
        **kwargs: Any,
    ) -> str:
        """Read file contents with path traversal protection."""
        try:
            result = validate_read_path(path, workspace=self._workspace)
            if not result.valid:
                return f"Security Error: {result.error}"

            file_path = result.resolved_path
            assert file_path is not None

            if not file_path.exists():
                return f"Error: File not found: {path}"
            if not file_path.is_file():
                return f"Error: Not a file: {path}"

            try:
                async with aiofiles.open(file_path, "r", encoding=encoding) as f:
                    content = await f.read()
            except UnicodeDecodeError:
                async with aiofiles.open(file_path, "r", encoding="latin-1") as f:
                    content = await f.read()

            # Line-range selection (Pi-style offset+limit)
            if offset is not None or limit is not None:
                all_lines = content.split("\n")
                total_lines = len(all_lines)
                start = max(0, (offset or 1) - 1)
                if start >= total_lines:
                    return f"Error: Offset {offset} beyond end of file ({total_lines} lines)"
                end = min(total_lines, start + limit) if limit else total_lines
                content = "\n".join(all_lines[start:end])
                range_note = f" | lines {start + 1}-{end}/{total_lines}"
            else:
                range_note = ""

            total_size = len(content)
            parts = file_path.parts
            is_skill_file = "skills" in parts

            if self._max_output and total_size > self._max_output:
                content = content[:self._max_output] + f"\n... (truncated, {total_size - self._max_output} more chars)"

            # Use relative path to workspace for dedup, fallback to name
            try:
                if self._workspace:
                    rel = file_path.relative_to(self._workspace)
                    display_path = str(rel).replace("\\", "/")
                else:
                    display_path = file_path.name
            except ValueError:
                display_path = file_path.name

            actual_size = len(content)
            header = f"[file: {display_path} | {actual_size} chars{range_note}"
            if is_skill_file:
                header += " | skill-ref"
            header += "]\n"
            return header + content

        except PermissionError:
            return f"Error: Permission denied: {path}"
        except Exception as e:
            return f"Error reading file: {str(e)}"

    @staticmethod
    def _extract_skill_cli_content(content: str, budget: int = 2500) -> str:
        """Extract CLI-relevant content from SKILL.md, stripping JS code blocks.

        Keeps: frontmatter, headings, text, tables, `cast`/`curl`/`bash` code blocks.
        Strips: `js`/`javascript`/`typescript` code blocks (replaced with one-line stub).
        This steers the agent toward direct CLI usage.
        """
        import re
        lines = content.split("\n")
        out: list[str] = []
        total = 0
        in_code = False
        code_lang = ""
        skip_block = False
        js_langs = {"js", "javascript", "typescript", "ts", "mjs"}

        for line in lines:
            if total > budget:
                out.append(f"\n... [SKILL.md truncated at {budget} chars — use CLI examples above]")
                break

            if not in_code:
                m = re.match(r'^```(\w*)', line)
                if m:
                    in_code = True
                    code_lang = m.group(1).lower()
                    skip_block = code_lang in js_langs
                    if skip_block:
                        stub = f"```{code_lang}\n// [JS code block omitted — use cast/curl CLI equivalent below]\n```"
                        out.append(stub)
                        total += len(stub)
                    else:
                        out.append(line)
                        total += len(line) + 1
                else:
                    out.append(line)
                    total += len(line) + 1
            else:
                if line.strip().startswith("```"):
                    in_code = False
                    if not skip_block:
                        out.append(line)
                        total += len(line) + 1
                    skip_block = False
                elif not skip_block:
                    out.append(line)
                    total += len(line) + 1

        return "\n".join(out)


class WriteFileTool(Tool):
    """Tool to write content to a file with path traversal protection."""

    def __init__(self, workspace: Path | str | None = None):
        """
        Initialize the write file tool.

        Args:
            workspace: The allowed workspace directory. Writes outside this
                       directory will be rejected for security.
        """
        self._workspace = Path(workspace).resolve() if workspace else None
        self._validator = PathValidator(workspace=workspace) if workspace else None

    def set_workspace(self, workspace: Path | str) -> None:
        """Set the workspace boundary for path validation."""
        self._workspace = Path(workspace).resolve()
        self._validator = PathValidator(workspace=self._workspace)
        set_default_validator(self._validator)

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        workspace_note = ""
        if self._workspace:
            workspace_note = f" Files must be within the workspace: {self._workspace}"
        return f"Write content to a file at the given path. Creates parent directories if needed.{workspace_note}"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to write to (must be within workspace)",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write",
                },
            },
            "required": ["path", "content"],
        }

    async def execute(
        self,
        path: str,
        content: str,
        **kwargs: Any,
    ) -> str:
        """Write content to file with path traversal protection."""
        try:
            # Validate the path (write validation is stricter)
            result = validate_write_path(path, workspace=self._workspace)
            if not result.valid:
                return f"Security Error: {result.error}"

            file_path = result.resolved_path
            assert file_path is not None  # Guaranteed by valid=True

            # Create parent directories (already validated to be within workspace)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                await f.write(content)

            return f"Successfully wrote {len(content)} bytes to {path}"

        except PermissionError:
            return f"Error: Permission denied: {path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"


class EditFileTool(Tool):
    """Tool to edit a file by replacing text with path traversal protection."""

    def __init__(self, workspace: Path | str | None = None):
        """
        Initialize the edit file tool.

        Args:
            workspace: The allowed workspace directory. Edits outside this
                       directory will be rejected for security.
        """
        self._workspace = Path(workspace).resolve() if workspace else None
        self._validator = PathValidator(workspace=workspace) if workspace else None

    def set_workspace(self, workspace: Path | str) -> None:
        """Set the workspace boundary for path validation."""
        self._workspace = Path(workspace).resolve()
        self._validator = PathValidator(workspace=self._workspace)
        set_default_validator(self._validator)

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        workspace_note = ""
        if self._workspace:
            workspace_note = f" Files must be within the workspace: {self._workspace}"
        return (
            "Edit a file by replacing old_text with new_text. "
            f"The old_text must exist exactly once in the file.{workspace_note}"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to edit (must be within workspace)",
                },
                "old_text": {
                    "type": "string",
                    "description": "The exact text to find and replace",
                },
                "new_text": {
                    "type": "string",
                    "description": "The text to replace with",
                },
            },
            "required": ["path", "old_text", "new_text"],
        }

    async def execute(
        self,
        path: str,
        old_text: str,
        new_text: str,
        **kwargs: Any,
    ) -> str:
        """Edit file by replacing text with path traversal protection."""
        try:
            # Validate the path (use write validation since we're modifying)
            result = validate_write_path(path, workspace=self._workspace)
            if not result.valid:
                return f"Security Error: {result.error}"

            file_path = result.resolved_path
            assert file_path is not None  # Guaranteed by valid=True

            if not file_path.exists():
                return f"Error: File not found: {path}"

            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()

            if old_text not in content:
                return "Error: old_text not found in file. Make sure it matches exactly."

            # Count occurrences
            count = content.count(old_text)
            if count > 1:
                return (
                    f"Warning: old_text appears {count} times. "
                    "Please provide more context to make it unique."
                )

            new_content = content.replace(old_text, new_text, 1)

            async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                await f.write(new_content)

            return f"Successfully edited {path}"

        except PermissionError:
            return f"Error: Permission denied: {path}"
        except Exception as e:
            return f"Error editing file: {str(e)}"


class ListDirTool(Tool):
    """Tool to list directory contents with path traversal protection."""

    def __init__(
        self,
        workspace: Path | str | None = None,
        additional_read_paths: list[Path | str] | None = None,
    ):
        self._workspace = Path(workspace).resolve() if workspace else None
        self._additional_read_paths = additional_read_paths
        self._validator = (
            PathValidator(workspace=workspace, additional_read_paths=additional_read_paths)
            if workspace else None
        )

    def set_workspace(self, workspace: Path | str) -> None:
        """Set the workspace boundary for path validation."""
        self._workspace = Path(workspace).resolve()
        self._validator = PathValidator(
            workspace=self._workspace,
            additional_read_paths=self._additional_read_paths,
        )
        set_default_validator(self._validator)

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        workspace_note = ""
        if self._workspace:
            workspace_note = f" Directory must be within the workspace: {self._workspace}"
        return f"List the contents of a directory with file/folder indicators.{workspace_note}"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The directory path to list (must be within workspace)",
                },
                "show_hidden": {
                    "type": "boolean",
                    "description": "Show hidden files (default: false)",
                },
            },
            "required": ["path"],
        }

    async def execute(
        self,
        path: str,
        show_hidden: bool = False,
        **kwargs: Any,
    ) -> str:
        """List directory contents with path traversal protection."""
        try:
            # Validate the path
            result = validate_directory_path(path, workspace=self._workspace)
            if not result.valid:
                return f"Security Error: {result.error}"

            dir_path = result.resolved_path
            assert dir_path is not None  # Guaranteed by valid=True

            if not dir_path.exists():
                return f"Error: Directory not found: {path}"
            if not dir_path.is_dir():
                return f"Error: Not a directory: {path}"

            items = []
            for item in sorted(dir_path.iterdir()):
                # Skip hidden files unless requested
                if not show_hidden and item.name.startswith("."):
                    continue

                if item.is_dir():
                    items.append(f"[DIR]  {item.name}/")
                elif item.is_symlink():
                    # Check if symlink target is safe to show
                    try:
                        target = item.resolve()
                        if self._workspace and not self._is_within_workspace(target):
                            items.append(f"[LINK] {item.name} -> (outside workspace)")
                        else:
                            items.append(f"[LINK] {item.name}")
                    except OSError:
                        items.append(f"[LINK] {item.name} -> (broken)")
                else:
                    # Show file size
                    try:
                        size = item.stat().st_size
                        if size < 1024:
                            size_str = f"{size}B"
                        elif size < 1024 * 1024:
                            size_str = f"{size // 1024}KB"
                        else:
                            size_str = f"{size // (1024 * 1024)}MB"
                        items.append(f"[FILE] {item.name} ({size_str})")
                    except OSError:
                        items.append(f"[FILE] {item.name}")

            if not items:
                return f"Directory {path} is empty"

            return "\n".join(items)

        except PermissionError:
            return f"Error: Permission denied: {path}"
        except Exception as e:
            return f"Error listing directory: {str(e)}"

    def _is_within_workspace(self, path: Path) -> bool:
        """Check if a path is within the workspace."""
        if self._workspace is None:
            return True
        try:
            path.relative_to(self._workspace)
            return True
        except ValueError:
            return False
