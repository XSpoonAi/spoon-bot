"""Tool registry for managing agent tools with filtering and caching."""

from __future__ import annotations

from typing import Any

from loguru import logger

from spoon_bot.agent.tools.base import Tool, ToolSchema


# ---------------------------------------------------------------------------
# Core tools: the minimal set loaded into the agent by default.
# All other tools remain in the registry (schemas in memory) and can be
# activated dynamically via AgentLoop.add_tool().
# ---------------------------------------------------------------------------
CORE_TOOLS: frozenset[str] = frozenset({
    "shell", "read_file", "write_file", "edit_file", "list_dir",
    "web_search", "web_fetch", "activate_tool",
})

# ---------------------------------------------------------------------------
# Tool profiles: named subsets of tools for different task types.
# "core" uses CORE_TOOLS above; "all" is implicit (no filter).
# ---------------------------------------------------------------------------
TOOL_PROFILES: dict[str, frozenset[str]] = {
    "core": CORE_TOOLS,
    "coding": frozenset({
        "shell", "read_file", "write_file", "edit_file", "list_dir",
        "web_search", "web_fetch", "document_parse",
    }),
    "web3": frozenset({
        "balance_check", "transfer", "swap", "contract_call",
        "web_search", "web_fetch",
    }),
    "research": frozenset({
        "web_search", "web_fetch", "read_file", "list_dir",
        "document_parse",
    }),
    "full": frozenset({
        "shell", "read_file", "write_file", "edit_file", "list_dir",
        "self_config", "memory", "self_upgrade", "spawn",
        "web_search", "web_fetch", "document_parse",
        "balance_check", "transfer", "swap", "contract_call",
    }),
}


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
    - Profile-based / explicit tool filtering
    - Schema caching (invalidated on mutation)
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
        # Filtering
        self._enabled_tools: set[str] | None = None  # None = all tools
        # Schema cache
        self._definitions_cache: list[ToolSchema] | None = None
        self._definitions_cache_key: tuple[int, frozenset[str] | None] = (-1, None)

    # ------------------------------------------------------------------
    # Tool filtering
    # ------------------------------------------------------------------

    def set_tool_filter(
        self,
        *,
        enabled_tools: set[str] | None = None,
        tool_profile: str | None = None,
    ) -> None:
        """
        Filter which tools are active.

        Args:
            enabled_tools: Explicit set of tool names to enable. None = all.
            tool_profile: Named profile from TOOL_PROFILES.

        If both are provided, enabled_tools takes precedence.
        """
        if enabled_tools is not None:
            self._enabled_tools = set(enabled_tools)
        elif tool_profile is not None:
            profile = TOOL_PROFILES.get(tool_profile)
            if profile is None:
                available = ", ".join(sorted(TOOL_PROFILES.keys()))
                raise ValueError(
                    f"Unknown tool profile '{tool_profile}'. "
                    f"Available: {available}"
                )
            self._enabled_tools = set(profile)
        else:
            self._enabled_tools = None
        self._invalidate_cache()

    def clear_tool_filter(self) -> None:
        """Remove tool filter, enabling all registered tools."""
        self._enabled_tools = None
        self._invalidate_cache()

    def get_active_tools(self) -> dict[str, Tool]:
        """Get only the currently active (filtered) tools."""
        if self._enabled_tools is None:
            return self._tools
        return {
            name: tool for name, tool in self._tools.items()
            if name in self._enabled_tools
        }

    def get_inactive_tools(self) -> dict[str, Tool]:
        """Get tools registered but not currently active."""
        if self._enabled_tools is None:
            return {}
        return {
            name: tool for name, tool in self._tools.items()
            if name not in self._enabled_tools
        }

    def activate_tool(self, name: str) -> bool:
        """
        Activate a registered tool (add to the enabled set).

        Args:
            name: Tool name to activate.

        Returns:
            True if the tool was activated, False if not found or already active.
        """
        if name not in self._tools:
            logger.warning(f"Cannot activate unknown tool: {name}")
            return False
        if self._enabled_tools is None:
            # All tools are already active
            return False
        if name in self._enabled_tools:
            return False
        self._enabled_tools.add(name)
        self._invalidate_cache()
        logger.info(f"Activated tool: {name}")
        return True

    def deactivate_tool(self, name: str) -> bool:
        """
        Deactivate an active tool (remove from the enabled set).

        Args:
            name: Tool name to deactivate.

        Returns:
            True if the tool was deactivated, False if not active.
        """
        if self._enabled_tools is None:
            # Switch from "all" mode to explicit mode with this tool removed
            self._enabled_tools = set(self._tools.keys())
        if name not in self._enabled_tools:
            return False
        self._enabled_tools.discard(name)
        self._invalidate_cache()
        logger.info(f"Deactivated tool: {name}")
        return True

    def get_all_tool_summaries(self) -> list[dict[str, str]]:
        """
        Get name + description for ALL registered tools (active and inactive).

        Useful for including in system prompts so the LLM knows what tools
        are available for dynamic loading.

        Returns:
            List of {"name": ..., "description": ..., "active": bool} dicts.
        """
        active = self.get_active_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "active": tool.name in active,
            }
            for tool in self._tools.values()
        ]

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _invalidate_cache(self) -> None:
        self._definitions_cache = None
        self._definitions_cache_key = (-1, None)

    def _current_cache_key(self) -> tuple[int, frozenset[str] | None]:
        enabled = frozenset(self._enabled_tools) if self._enabled_tools is not None else None
        return (len(self.get_active_tools()), enabled)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

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
        self._invalidate_cache()
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
            self._invalidate_cache()
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
        Get tool definitions for active (filtered) tools.

        Returns cached result when possible.

        Returns:
            List of tool schemas in OpenAI function format.
        """
        key = self._current_cache_key()
        if self._definitions_cache is not None and self._definitions_cache_key == key:
            return self._definitions_cache
        active = self.get_active_tools()
        self._definitions_cache = [tool.to_schema() for tool in active.values()]
        self._definitions_cache_key = key
        return self._definitions_cache

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        validate: bool | None = None,
    ) -> str:
        """
        Execute a tool by name.

        Looks up in ALL registered tools (not filtered), because the LLM
        may reference a tool that was active in an earlier turn.

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
        List active tool names.

        Returns:
            List of tool names sorted alphabetically.
        """
        return sorted(self.get_active_tools().keys())

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
        """Return number of active tools."""
        return len(self.get_active_tools())

    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered (in all tools, not just active)."""
        return name in self._tools

    def __iter__(self):
        """Iterate over active tools."""
        return iter(self.get_active_tools().values())
