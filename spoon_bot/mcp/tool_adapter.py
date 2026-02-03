"""Adapter for MCP tools to work with spoon-bot's ToolRegistry."""

from typing import Any

from loguru import logger

from spoon_bot.agent.tools.base import Tool
from spoon_bot.mcp.client import MCPClientAdapter, MCPTool


class MCPToolWrapper(Tool):
    """Wrapper that makes an MCP tool behave like a native Tool."""

    def __init__(
        self,
        mcp_tool: MCPTool,
        client: MCPClientAdapter,
    ):
        self._mcp_tool = mcp_tool
        self._client = client

    @property
    def name(self) -> str:
        return self._mcp_tool.name

    @property
    def description(self) -> str:
        return self._mcp_tool.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._mcp_tool.parameters

    async def execute(self, **kwargs: Any) -> str:
        """Execute the MCP tool."""
        return await self._client.call_tool(
            self._mcp_tool.server_name,
            self._mcp_tool.name,
            kwargs,
        )


class MCPToolAdapter:
    """
    Adapter that loads MCP tools and converts them to native Tool instances.

    Usage:
        adapter = MCPToolAdapter()
        adapter.add_server("github", {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {"GITHUB_TOKEN": "..."}
        })
        tools = await adapter.load_tools()
        for tool in tools:
            registry.register(tool)
    """

    def __init__(self):
        self._client = MCPClientAdapter()
        self._loaded_tools: list[Tool] = []

    def add_server(self, name: str, config: dict[str, Any]) -> None:
        """Add an MCP server configuration."""
        self._client.add_server_from_dict(name, config)

    def add_servers(self, servers: dict[str, dict[str, Any]]) -> None:
        """Add multiple MCP server configurations."""
        for name, config in servers.items():
            self.add_server(name, config)

    async def connect(self, server_name: str) -> bool:
        """Connect to a specific MCP server."""
        return await self._client.connect(server_name)

    async def connect_all(self) -> dict[str, bool]:
        """Connect to all configured MCP servers."""
        results = {}
        for server_name in self._client._servers:
            results[server_name] = await self.connect(server_name)
        return results

    async def load_tools(self, server_name: str | None = None) -> list[Tool]:
        """
        Load tools from MCP servers.

        Args:
            server_name: Specific server to load from, or None for all.

        Returns:
            List of Tool instances.
        """
        # Connect to servers
        if server_name:
            await self.connect(server_name)
        else:
            await self.connect_all()

        # Get MCP tools
        mcp_tools = self._client.get_tools()

        # Filter by server if specified
        if server_name:
            mcp_tools = [t for t in mcp_tools if t.server_name == server_name]

        # Wrap as native Tools
        self._loaded_tools = []
        for mcp_tool in mcp_tools:
            wrapper = MCPToolWrapper(mcp_tool, self._client)
            self._loaded_tools.append(wrapper)
            logger.debug(f"Loaded MCP tool: {mcp_tool.name} from {mcp_tool.server_name}")

        return self._loaded_tools

    def get_loaded_tools(self) -> list[Tool]:
        """Get all loaded tools."""
        return self._loaded_tools

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        await self._client.disconnect_all()
        self._loaded_tools = []
