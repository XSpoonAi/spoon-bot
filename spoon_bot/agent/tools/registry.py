"""Tool registry for managing agent tools."""

from typing import Any
from loguru import logger

from spoon_bot.agent.tools.base import Tool


class ToolRegistry:
    """
    Central registry for all agent tools.

    Tool priority (loaded in order):
    1. Native OS tools (shell, filesystem) - always available
    2. Self-management tools (self_config, memory, etc.)
    3. spoon-toolkit tools (crypto, blockchain, social)
    4. MCP tools (dynamically loaded)
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """
        Register a tool.

        Args:
            tool: The tool to register.
        """
        if tool.name in self._tools:
            logger.warning(f"Tool {tool.name} already registered, overwriting")
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def unregister(self, name: str) -> bool:
        """
        Unregister a tool by name.

        Args:
            name: The tool name to unregister.

        Returns:
            True if tool was removed, False if not found.
        """
        if name in self._tools:
            del self._tools[name]
            logger.debug(f"Unregistered tool: {name}")
            return True
        return False

    def get(self, name: str) -> Tool | None:
        """
        Get a tool by name.

        Args:
            name: The tool name.

        Returns:
            The tool if found, None otherwise.
        """
        return self._tools.get(name)

    def get_definitions(self) -> list[dict[str, Any]]:
        """
        Get all tool definitions in OpenAI function format.

        Returns:
            List of tool schemas.
        """
        return [tool.to_schema() for tool in self._tools.values()]

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """
        Execute a tool by name.

        Args:
            name: The tool name.
            arguments: The tool arguments.

        Returns:
            The tool execution result.
        """
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: Unknown tool '{name}'"

        try:
            result = await tool.execute(**arguments)
            return result
        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            return f"Error executing tool {name}: {str(e)}"

    def list_tools(self) -> list[str]:
        """
        List all registered tool names.

        Returns:
            List of tool names.
        """
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
