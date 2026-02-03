"""Shell execution tool with safety guards."""

import asyncio
import os
import sys
from typing import Any

from spoon_bot.agent.tools.base import Tool


class ShellTool(Tool):
    """
    Tool to execute shell commands with safety guards.

    Safety features:
    - Timeout (default 60s) to prevent hanging
    - Output truncation (default 10KB) to prevent context explosion
    - Working directory isolation
    """

    def __init__(
        self,
        timeout: int = 60,
        max_output: int = 10000,
        working_dir: str | None = None,
    ):
        """
        Initialize shell tool.

        Args:
            timeout: Command timeout in seconds (default 60).
            max_output: Maximum output characters (default 10000).
            working_dir: Default working directory.
        """
        self.timeout = timeout
        self.max_output = max_output
        self.working_dir = working_dir

    @property
    def name(self) -> str:
        return "shell"

    @property
    def description(self) -> str:
        return (
            "Execute a shell command and return its output. "
            f"Commands timeout after {self.timeout}s. "
            "Use with caution for destructive operations."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                },
                "working_dir": {
                    "type": "string",
                    "description": "Optional working directory for the command",
                },
            },
            "required": ["command"],
        }

    async def execute(
        self,
        command: str,
        working_dir: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Execute a shell command.

        Args:
            command: The command to execute.
            working_dir: Optional working directory.

        Returns:
            Command output or error message.
        """
        cwd = working_dir or self.working_dir or os.getcwd()

        # Determine shell based on platform
        if sys.platform == "win32":
            # Use cmd.exe on Windows
            shell_cmd = command
        else:
            # Use bash on Unix
            shell_cmd = command

        try:
            process = await asyncio.create_subprocess_shell(
                shell_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return f"Error: Command timed out after {self.timeout} seconds"

            output_parts = []

            if stdout:
                stdout_text = stdout.decode("utf-8", errors="replace")
                output_parts.append(stdout_text)

            if stderr:
                stderr_text = stderr.decode("utf-8", errors="replace")
                if stderr_text.strip():
                    output_parts.append(f"STDERR:\n{stderr_text}")

            if process.returncode != 0:
                output_parts.append(f"\nExit code: {process.returncode}")

            result = "\n".join(output_parts) if output_parts else "(no output)"

            # Truncate very long output
            if len(result) > self.max_output:
                truncated = len(result) - self.max_output
                result = result[: self.max_output] + f"\n... (truncated, {truncated} more chars)"

            return result

        except FileNotFoundError:
            return f"Error: Working directory not found: {cwd}"
        except PermissionError:
            return f"Error: Permission denied for command or directory"
        except Exception as e:
            return f"Error executing command: {str(e)}"
