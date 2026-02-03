"""File system tools: read, write, edit, list."""

from pathlib import Path
from typing import Any
import aiofiles
import aiofiles.os

from spoon_bot.agent.tools.base import Tool


class ReadFileTool(Tool):
    """Tool to read file contents with encoding fallback."""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the contents of a file at the given path."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to read",
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
        encoding: str = "utf-8",
        **kwargs: Any,
    ) -> str:
        """Read file contents."""
        try:
            file_path = Path(path).expanduser().resolve()

            if not file_path.exists():
                return f"Error: File not found: {path}"
            if not file_path.is_file():
                return f"Error: Not a file: {path}"

            # Try primary encoding, fallback to latin-1
            try:
                async with aiofiles.open(file_path, "r", encoding=encoding) as f:
                    content = await f.read()
            except UnicodeDecodeError:
                async with aiofiles.open(file_path, "r", encoding="latin-1") as f:
                    content = await f.read()

            return content

        except PermissionError:
            return f"Error: Permission denied: {path}"
        except Exception as e:
            return f"Error reading file: {str(e)}"


class WriteFileTool(Tool):
    """Tool to write content to a file, creating parent directories."""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write content to a file at the given path. Creates parent directories if needed."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to write to",
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
        """Write content to file."""
        try:
            file_path = Path(path).expanduser().resolve()

            # Create parent directories
            file_path.parent.mkdir(parents=True, exist_ok=True)

            async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                await f.write(content)

            return f"Successfully wrote {len(content)} bytes to {path}"

        except PermissionError:
            return f"Error: Permission denied: {path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"


class EditFileTool(Tool):
    """Tool to edit a file by replacing text."""

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return (
            "Edit a file by replacing old_text with new_text. "
            "The old_text must exist exactly once in the file."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to edit",
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
        """Edit file by replacing text."""
        try:
            file_path = Path(path).expanduser().resolve()

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
    """Tool to list directory contents with visual indicators."""

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return "List the contents of a directory with file/folder indicators."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The directory path to list",
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
        """List directory contents."""
        try:
            dir_path = Path(path).expanduser().resolve()

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
                    items.append(f"[LINK] {item.name}")
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
