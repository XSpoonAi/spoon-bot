"""MCP integration using spoon-core's MCPTool and MCPClientMixin.

This adapter provides a unified interface for MCP tools, supporting:
- spoon-core's comprehensive MCPTool (npx, uvx, python, SSE, HTTP, WebSocket)
- Fallback to the local client.py implementation if spoon-core is not installed
"""
from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from spoon_bot.agent.tools.base import Tool
from spoon_bot.exceptions import (
    MCPConnectionError,
    MCPServerNotFoundError,
    MCPToolExecutionError,
)


# Check if spoon-core is available
_SPOON_CORE_AVAILABLE = False
_SPOON_CORE_IMPORT_ERROR: str | None = None
_SpoonCoreMCPTool = None
_MCPClientMixin = None

try:
    from spoon_ai.tools.mcp_tool import MCPTool as SpoonCoreMCPTool
    from spoon_ai.agents.mcp_client_mixin import MCPClientMixin
    _SpoonCoreMCPTool = SpoonCoreMCPTool
    _MCPClientMixin = MCPClientMixin
    _SPOON_CORE_AVAILABLE = True
    logger.debug("spoon-core MCP module available - using enhanced transport support")
except ImportError as e:
    _SPOON_CORE_IMPORT_ERROR = str(e)
    logger.debug(f"spoon-core not available - using fallback MCP implementation: {e}")


class SpoonCoreMCPToolWrapper(Tool):
    """
    Wrapper that adapts a spoon-core MCPTool to spoon-bot's Tool interface.

    This allows spoon-core's MCPTool (which uses Pydantic BaseModel) to work
    seamlessly with spoon-bot's Tool ABC.
    """

    def __init__(
        self,
        mcp_tool: Any,  # spoon-core MCPTool instance
        tool_name: str,
        tool_description: str,
        tool_parameters: dict[str, Any],
    ):
        self._mcp_tool = mcp_tool
        self._name = tool_name
        self._description = tool_description
        self._parameters = tool_parameters

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    async def execute(self, **kwargs: Any) -> str:
        """Execute the MCP tool using spoon-core's implementation.

        Returns:
            Tool execution result as string.
            In case of error, returns a user-friendly error message.
        """
        try:
            result = await self._mcp_tool.execute(**kwargs)
            if result is None:
                return ""
            return str(result)
        except asyncio.TimeoutError as e:
            logger.error(f"MCP tool {self._name} timed out: {e}")
            return f"Error: Tool '{self._name}' timed out. The MCP server may be unresponsive."
        except ConnectionError as e:
            logger.error(f"MCP connection error for {self._name}: {e}")
            return f"Error: Could not connect to MCP server for tool '{self._name}'."
        except Exception as e:
            error_type = type(e).__name__
            logger.error(f"Error executing MCP tool {self._name}: {error_type}: {e}")
            # Provide a user-friendly message while logging the full error
            return f"Error executing '{self._name}': {str(e)}"


class SpoonCoreMCPAdapter:
    """
    Adapter for spoon-core's MCP tools with fallback to local implementation.

    This adapter automatically detects if spoon-core is installed and uses its
    comprehensive MCPTool implementation. If spoon-core is not available, it
    falls back to the local client.py implementation.

    Supported transport types (when using spoon-core):
    - stdio: Generic subprocess communication
    - npx: Node.js npx-based MCP servers
    - uvx: Python uvx-based MCP servers
    - python: Direct Python script MCP servers
    - sse: Server-sent events over HTTP
    - http: Streamable HTTP transport
    - websocket: WebSocket transport

    Usage:
        adapter = SpoonCoreMCPAdapter()
        adapter.add_server("github", {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {"GITHUB_TOKEN": "..."}
        })
        tools = await adapter.load_tools()
        for tool in tools:
            registry.register(tool)
    """

    def __init__(self):
        self._use_spoon_core = _SPOON_CORE_AVAILABLE
        self._server_configs: dict[str, dict[str, Any]] = {}
        self._mcp_tools: dict[str, Any] = {}  # server_name -> MCPTool instance
        self._loaded_tools: list[Tool] = []
        self._fallback_adapter = None
        self._failed_servers: dict[str, str] = {}  # server_name -> error message

        if not self._use_spoon_core:
            # Import fallback adapter lazily
            try:
                from spoon_bot.mcp.tool_adapter import MCPToolAdapter
                self._fallback_adapter = MCPToolAdapter()
                logger.info("SpoonCoreMCPAdapter initialized with fallback implementation")
            except ImportError as e:
                logger.warning(f"Could not initialize MCP fallback adapter: {e}")
        else:
            logger.info("SpoonCoreMCPAdapter initialized with spoon-core MCPTool")

    @property
    def uses_spoon_core(self) -> bool:
        """Check if the adapter is using spoon-core's implementation."""
        return self._use_spoon_core

    def add_server(self, name: str, config: dict[str, Any]) -> bool:
        """
        Add an MCP server configuration.

        This method handles errors gracefully and will not raise exceptions.
        Failed servers are tracked and can be queried via get_failed_servers().

        Args:
            name: Unique name for the server.
            config: Server configuration dict with keys:
                - command: Command to run (npx, uvx, python, or custom)
                - args: Command arguments
                - url: URL for HTTP/SSE/WebSocket transports
                - transport: Transport type hint (sse, http)
                - env: Environment variables
                - timeout: Connection timeout in seconds
                - health_check_interval: Health check interval in seconds
                - max_retries: Maximum retry attempts

        Returns:
            True if the server was added successfully, False otherwise.
        """
        self._server_configs[name] = config

        # Clear any previous failure status
        self._failed_servers.pop(name, None)

        if self._use_spoon_core:
            # Create MCPTool instance from config
            try:
                mcp_tool = _SpoonCoreMCPTool(
                    name=name,
                    description=f"MCP server: {name}",
                    mcp_config=config,
                )
                self._mcp_tools[name] = mcp_tool
                logger.debug(f"Created spoon-core MCPTool for server: {name}")
                return True
            except FileNotFoundError as e:
                error_msg = f"Command not found: {config.get('command', 'unknown')}"
                self._failed_servers[name] = error_msg
                logger.warning(f"Failed to create MCPTool for {name}: {error_msg}")
                return False
            except ValueError as e:
                error_msg = f"Invalid configuration: {e}"
                self._failed_servers[name] = error_msg
                logger.warning(f"Failed to create MCPTool for {name}: {error_msg}")
                return False
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                self._failed_servers[name] = error_msg
                logger.warning(f"Failed to create MCPTool for {name}: {error_msg}")
                return False
        else:
            # Use fallback adapter
            if self._fallback_adapter is None:
                error_msg = "No MCP adapter available (spoon-core not installed)"
                self._failed_servers[name] = error_msg
                logger.warning(f"Cannot add server {name}: {error_msg}")
                return False
            try:
                self._fallback_adapter.add_server(name, config)
                logger.debug(f"Added server {name} to fallback adapter")
                return True
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                self._failed_servers[name] = error_msg
                logger.warning(f"Failed to add server {name} to fallback: {error_msg}")
                return False

    def get_failed_servers(self) -> dict[str, str]:
        """Get a dictionary of failed server names and their error messages."""
        return self._failed_servers.copy()

    def add_servers(self, servers: dict[str, dict[str, Any]]) -> None:
        """Add multiple MCP server configurations."""
        for name, config in servers.items():
            self.add_server(name, config)

    async def load_tools(self, server_name: str | None = None) -> list[Tool]:
        """
        Load tools from MCP servers.

        This method discovers available tools from the MCP server(s) and wraps
        them as spoon-bot Tool instances.

        Args:
            server_name: Specific server to load from, or None for all.

        Returns:
            List of Tool instances ready for use.
        """
        if not self._use_spoon_core:
            # Use fallback adapter
            tools = await self._fallback_adapter.load_tools(server_name)
            self._loaded_tools = tools
            return tools

        # Use spoon-core MCPTool
        servers_to_load = (
            [server_name] if server_name else list(self._mcp_tools.keys())
        )

        self._loaded_tools = []

        for srv_name in servers_to_load:
            mcp_tool = self._mcp_tools.get(srv_name)
            if not mcp_tool:
                logger.warning(f"Server {srv_name} not found")
                continue

            try:
                # Ensure parameters are loaded from the MCP server
                await mcp_tool.ensure_parameters_loaded()

                # Get list of available tools from the server
                available_tools = await mcp_tool.list_available_tools()

                if not available_tools:
                    # If no tools listed, create a single wrapper for the MCPTool
                    wrapper = SpoonCoreMCPToolWrapper(
                        mcp_tool=mcp_tool,
                        tool_name=mcp_tool.name,
                        tool_description=mcp_tool.description,
                        tool_parameters=mcp_tool.parameters,
                    )
                    self._loaded_tools.append(wrapper)
                    logger.debug(f"Loaded MCP tool: {mcp_tool.name} from {srv_name}")
                else:
                    # Create wrappers for each discovered tool
                    for tool_info in available_tools:
                        tool_name = tool_info.get("name", "")
                        if not tool_name:
                            continue

                        # Create a dedicated MCPTool for each discovered tool
                        config = self._server_configs[srv_name].copy()
                        dedicated_mcp_tool = _SpoonCoreMCPTool(
                            name=tool_name,
                            description=tool_info.get("description", f"MCP tool: {tool_name}"),
                            parameters=tool_info.get("inputSchema", {}),
                            mcp_config=config,
                        )

                        wrapper = SpoonCoreMCPToolWrapper(
                            mcp_tool=dedicated_mcp_tool,
                            tool_name=tool_name,
                            tool_description=tool_info.get("description", f"MCP tool: {tool_name}"),
                            tool_parameters=tool_info.get("inputSchema", {}),
                        )
                        self._loaded_tools.append(wrapper)
                        logger.debug(f"Loaded MCP tool: {tool_name} from {srv_name}")

            except Exception as e:
                logger.error(f"Failed to load tools from {srv_name}: {e}")
                continue

        logger.info(f"Loaded {len(self._loaded_tools)} MCP tools")
        return self._loaded_tools

    def get_loaded_tools(self) -> list[Tool]:
        """Get all loaded tools."""
        if not self._use_spoon_core and self._fallback_adapter:
            return self._fallback_adapter.get_loaded_tools()
        return self._loaded_tools

    async def connect(self, server_name: str) -> bool:
        """
        Connect to a specific MCP server.

        For spoon-core, this triggers parameter loading and health check.

        Args:
            server_name: Name of the server to connect to.

        Returns:
            True if connection successful, False otherwise.
        """
        # Check if server failed during add_server
        if server_name in self._failed_servers:
            logger.warning(
                f"Cannot connect to {server_name}: previously failed with: "
                f"{self._failed_servers[server_name]}"
            )
            return False

        if not self._use_spoon_core:
            if self._fallback_adapter is None:
                logger.error(f"Cannot connect to {server_name}: no MCP adapter available")
                return False
            try:
                return await self._fallback_adapter.connect(server_name)
            except Exception as e:
                logger.error(f"Failed to connect to {server_name} via fallback: {e}")
                return False

        mcp_tool = self._mcp_tools.get(server_name)
        if not mcp_tool:
            # Check if server was configured but failed
            if server_name in self._server_configs:
                logger.error(
                    f"Server {server_name} was configured but failed to initialize"
                )
            else:
                logger.error(f"Unknown server: {server_name}")
            return False

        try:
            await mcp_tool.ensure_parameters_loaded()
            logger.info(f"Connected to MCP server: {server_name}")
            return True
        except asyncio.TimeoutError:
            error_msg = "Connection timed out"
            self._failed_servers[server_name] = error_msg
            logger.error(f"Failed to connect to {server_name}: {error_msg}")
            return False
        except ConnectionError as e:
            error_msg = f"Connection error: {e}"
            self._failed_servers[server_name] = error_msg
            logger.error(f"Failed to connect to {server_name}: {error_msg}")
            return False
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            self._failed_servers[server_name] = error_msg
            logger.error(f"Failed to connect to {server_name}: {error_msg}")
            return False

    async def connect_all(self) -> dict[str, bool]:
        """Connect to all configured MCP servers."""
        if not self._use_spoon_core:
            return await self._fallback_adapter.connect_all()

        results = {}
        for server_name in self._mcp_tools:
            results[server_name] = await self.connect(server_name)
        return results

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """
        Call a tool on an MCP server directly.

        Args:
            server_name: Name of the MCP server.
            tool_name: Name of the tool to call.
            arguments: Arguments to pass to the tool.

        Returns:
            Tool execution result as string.
            Returns error message string if the call fails.
        """
        # Check if server previously failed
        if server_name in self._failed_servers:
            return (
                f"Error: Server '{server_name}' is not available: "
                f"{self._failed_servers[server_name]}"
            )

        if not self._use_spoon_core:
            if self._fallback_adapter is None:
                return "Error: No MCP adapter available (spoon-core not installed)"
            try:
                return await self._fallback_adapter._client.call_tool(
                    server_name, tool_name, arguments
                )
            except Exception as e:
                logger.error(f"Error calling tool {tool_name} via fallback: {e}")
                return f"Error: {str(e)}"

        mcp_tool = self._mcp_tools.get(server_name)
        if not mcp_tool:
            if server_name in self._server_configs:
                return f"Error: Server '{server_name}' was configured but failed to initialize"
            return f"Error: Unknown server '{server_name}'"

        try:
            result = await mcp_tool.execute(tool_name=tool_name, **arguments)
            return str(result) if result else ""
        except asyncio.TimeoutError:
            logger.error(f"Timeout calling tool {tool_name} on {server_name}")
            return f"Error: Tool '{tool_name}' timed out"
        except ConnectionError as e:
            logger.error(f"Connection error calling tool {tool_name} on {server_name}: {e}")
            return f"Error: Could not connect to server '{server_name}'"
        except Exception as e:
            error_type = type(e).__name__
            logger.error(f"Error calling tool {tool_name} on {server_name}: {error_type}: {e}")
            return f"Error executing '{tool_name}': {str(e)}"

    async def disconnect(self, server_name: str) -> None:
        """
        Disconnect from an MCP server.

        For spoon-core, this triggers cleanup of the MCPTool.
        """
        if not self._use_spoon_core:
            await self._fallback_adapter._client.disconnect(server_name)
            return

        mcp_tool = self._mcp_tools.get(server_name)
        if mcp_tool and hasattr(mcp_tool, 'cleanup'):
            try:
                await mcp_tool.cleanup()
            except Exception as e:
                logger.warning(f"Error during cleanup of {server_name}: {e}")

        # Remove from loaded tools
        self._loaded_tools = [
            t for t in self._loaded_tools
            if not (hasattr(t, '_mcp_tool') and
                    getattr(t._mcp_tool, 'name', '') == server_name)
        ]

        if server_name in self._mcp_tools:
            del self._mcp_tools[server_name]

        logger.info(f"Disconnected from MCP server: {server_name}")

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        if not self._use_spoon_core:
            await self._fallback_adapter.disconnect_all()
            return

        for server_name in list(self._mcp_tools.keys()):
            await self.disconnect(server_name)

        self._loaded_tools = []

    def get_server_names(self) -> list[str]:
        """Get list of configured server names."""
        return list(self._server_configs.keys())

    def get_server_config(self, server_name: str) -> dict[str, Any] | None:
        """Get configuration for a specific server."""
        return self._server_configs.get(server_name)
