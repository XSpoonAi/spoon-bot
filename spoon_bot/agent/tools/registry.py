"""Tool registry for managing agent tools."""

from __future__ import annotations

from typing import Any

from loguru import logger

from spoon_bot.agent.tools.base import Tool, ToolSchema


class ToolRegistry:
    """
    Central registry for all agent tools.

    Tool priority (loaded in order):
    1. Native OS tools (shell, filesystem) - always available
    2. Self-management tools (self_config, memory, etc.)
    3. spoon-toolkit tools (crypto, blockchain, social)
    4. MCP tools (dynamically loaded)

    Features:
    - Tool registration and lookup
    - Parameter validation before execution
    - Execution error handling
    """

    def __init__(self, validate_params: bool = True) -> None:
        """
        Initialize the tool registry.

        Args:
            validate_params: Whether to validate tool parameters before execution.
        """
        self._tools: dict[str, Tool] = {}
        self._validate_params = validate_params

    def register(self, tool: Tool) -> None:
        """
        Register a tool.

        Args:
            tool: The tool to register.

        Raises:
            ValueError: If tool name is empty.
        """
        if not tool.name:
            raise ValueError("Tool name cannot be empty")

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

    def get_definitions(self) -> list[ToolSchema]:
        """
        Get all tool definitions in OpenAI function format.

        Returns:
            List of tool schemas.
        """
        return [tool.to_schema() for tool in self._tools.values()]

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        validate: bool | None = None,
    ) -> str:
        """
        Execute a tool by name.

        Args:
            name: The tool name.
            arguments: The tool arguments.
            validate: Override default parameter validation setting.

        Returns:
            The tool execution result.
        """
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(sorted(self._tools.keys())[:10])
            return f"Error: Unknown tool '{name}'. Available tools: {available}..."

        # Validate parameters if enabled
        should_validate = validate if validate is not None else self._validate_params
        if should_validate:
            errors = tool.validate_parameters(**arguments)
            if errors:
                error_msg = "; ".join(errors)
                logger.warning(f"Parameter validation failed for {name}: {error_msg}")
                return f"Error: Invalid parameters for tool '{name}': {error_msg}"

        try:
            result = await tool.execute(**arguments)
            return result
        except TypeError as e:
            # Handle missing or extra arguments
            logger.error(f"Type error executing tool {name}: {e}")
            return f"Error: Invalid arguments for tool '{name}': {str(e)}"
        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            return f"Error executing tool {name}: {str(e)}"

    def list_tools(self) -> list[str]:
        """
        List all registered tool names.

        Returns:
            List of tool names sorted alphabetically.
        """
        return sorted(self._tools.keys())

    def get_tool_info(self, name: str) -> dict[str, Any] | None:
        """
        Get detailed information about a tool.

        Args:
            name: The tool name.

        Returns:
            Dictionary with tool info or None if not found.
        """
        tool = self._tools.get(name)
        if tool is None:
            return None

        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }

    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def __iter__(self):
        """Iterate over registered tools."""
        return iter(self._tools.values())
