"""MCP client adapter for connecting to MCP servers."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field, field_validator, model_validator


class MCPServerConfig(BaseModel):
    """
    Configuration for an MCP server with validation.

    Attributes:
        name: Unique server identifier.
        transport: Transport type (stdio, sse, http-stream).
        command: Command to run for stdio transport.
        args: Command arguments.
        url: URL for HTTP/SSE transports.
        env: Environment variables for the server process.
        timeout: Connection timeout in seconds.
    """

    name: str = Field(..., min_length=1, description="Unique server identifier")
    transport: str = Field(
        default="stdio",
        description="Transport type (stdio, sse, http-stream)"
    )
    command: str | None = Field(
        default=None,
        description="Command to run for stdio transport"
    )
    args: list[str] = Field(
        default_factory=list,
        description="Command arguments"
    )
    url: str | None = Field(
        default=None,
        description="URL for HTTP/SSE transports"
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables"
    )
    timeout: int = Field(
        default=30,
        ge=1,
        le=3600,
        description="Connection timeout in seconds (1-3600)"
    )

    @field_validator("transport")
    @classmethod
    def validate_transport(cls, v: str) -> str:
        """Validate transport type."""
        valid = {"stdio", "sse", "http-stream", "http", "websocket", "npx", "uvx", "python"}
        v = v.lower().strip()
        if v not in valid:
            raise ValueError(f"Invalid transport: {v}. Must be one of: {valid}")
        return v

    @model_validator(mode="after")
    def validate_transport_requirements(self) -> "MCPServerConfig":
        """Validate that required fields are present for each transport type."""
        if self.transport in ("stdio", "npx", "uvx", "python"):
            if not self.command:
                raise ValueError(
                    f"Transport '{self.transport}' requires 'command' field"
                )
        elif self.transport in ("sse", "http-stream", "http", "websocket"):
            if not self.url:
                raise ValueError(
                    f"Transport '{self.transport}' requires 'url' field"
                )
        return self

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str | None) -> str | None:
        """Validate URL format if provided."""
        if v is not None:
            v = v.strip()
            if not v.startswith(("http://", "https://", "ws://", "wss://")):
                raise ValueError(
                    "URL must start with http://, https://, ws://, or wss://"
                )
        return v


class MCPTool(BaseModel):
    """
    Represents a tool from an MCP server.

    Attributes:
        name: Tool name (unique within server).
        description: Tool description.
        parameters: JSON Schema for tool parameters.
        server_name: Name of the server this tool belongs to.
    """

    name: str = Field(..., min_length=1, description="Tool name")
    description: str = Field(default="", description="Tool description")
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}},
        description="JSON Schema for parameters"
    )
    server_name: str = Field(..., min_length=1, description="Server name")

    @property
    def full_name(self) -> str:
        """Get full tool name including server prefix."""
        return f"{self.server_name}:{self.name}"


class MCPClientAdapter:
    """
    Adapter for connecting to MCP servers and loading tools.

    Supports multiple transport types:
    - stdio: Local subprocess communication
    - sse: Server-sent events over HTTP
    - http-stream: HTTP streaming

    Note: This is a simplified implementation. For production use,
    consider using fastmcp or mcp-python-sdk directly.
    """

    def __init__(self):
        self._servers: dict[str, MCPServerConfig] = {}
        self._tools: dict[str, MCPTool] = {}
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._connected: set[str] = set()

    def add_server(self, config: MCPServerConfig) -> None:
        """Add an MCP server configuration."""
        self._servers[config.name] = config
        logger.debug(f"Added MCP server config: {config.name}")

    def add_server_from_dict(self, name: str, config: dict[str, Any]) -> None:
        """Add an MCP server from a dictionary configuration."""
        server_config = MCPServerConfig(
            name=name,
            transport=config.get("transport", "stdio"),
            command=config.get("command"),
            args=config.get("args", []),
            url=config.get("url"),
            env=config.get("env", {}),
            timeout=config.get("timeout", 30),
        )
        self.add_server(server_config)

    async def connect(self, server_name: str) -> bool:
        """
        Connect to an MCP server.

        Args:
            server_name: Name of the server to connect to.

        Returns:
            True if connection successful.
        """
        if server_name not in self._servers:
            logger.error(f"Unknown MCP server: {server_name}")
            return False

        if server_name in self._connected:
            logger.debug(f"Already connected to {server_name}")
            return True

        config = self._servers[server_name]

        try:
            if config.transport == "stdio":
                return await self._connect_stdio(config)
            elif config.transport in ("sse", "http-stream"):
                return await self._connect_http(config)
            else:
                logger.error(f"Unknown transport: {config.transport}")
                return False
        except Exception as e:
            logger.error(f"Failed to connect to {server_name}: {e}")
            return False

    async def _connect_stdio(self, config: MCPServerConfig) -> bool:
        """Connect to stdio-based MCP server."""
        if not config.command:
            logger.error(f"No command specified for stdio server {config.name}")
            return False

        try:
            import subprocess

            # Build command
            cmd = [config.command] + config.args

            # Merge environment
            import os
            env = os.environ.copy()
            env.update(config.env)

            # Start process
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            self._processes[config.name] = process
            self._connected.add(config.name)

            # Discover tools
            await self._discover_tools_stdio(config.name)

            logger.info(f"Connected to MCP server: {config.name} (stdio)")
            return True

        except Exception as e:
            logger.error(f"Failed to start stdio MCP server {config.name}: {e}")
            return False

    async def _connect_http(self, config: MCPServerConfig) -> bool:
        """Connect to HTTP-based MCP server (SSE or HTTP-stream)."""
        if not config.url:
            logger.error(f"No URL specified for HTTP server {config.name}")
            return False

        try:
            import httpx

            # Test connection
            async with httpx.AsyncClient(timeout=config.timeout) as client:
                # Try to list tools
                response = await client.get(f"{config.url}/tools")
                if response.status_code == 200:
                    tools_data = response.json()
                    self._parse_tools_response(config.name, tools_data)

            self._connected.add(config.name)
            logger.info(f"Connected to MCP server: {config.name} ({config.transport})")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to HTTP MCP server {config.name}: {e}")
            return False

    async def _discover_tools_stdio(self, server_name: str) -> None:
        """Discover tools from a stdio MCP server."""
        process = self._processes.get(server_name)
        if not process:
            return

        try:
            # Send tools/list request
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {}
            }

            request_bytes = (json.dumps(request) + "\n").encode()
            process.stdin.write(request_bytes)
            await process.stdin.drain()

            # Read response with timeout
            try:
                response_line = await asyncio.wait_for(
                    process.stdout.readline(),
                    timeout=10.0
                )
                if response_line:
                    response = json.loads(response_line.decode())
                    if "result" in response and "tools" in response["result"]:
                        self._parse_tools_response(server_name, response["result"]["tools"])
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for tools list from {server_name}")

        except Exception as e:
            logger.error(f"Error discovering tools from {server_name}: {e}")

    def _parse_tools_response(self, server_name: str, tools_data: list[dict[str, Any]]) -> None:
        """
        Parse tools response from MCP server.

        Args:
            server_name: Name of the server these tools belong to.
            tools_data: List of tool definitions from the server.
        """
        for tool_data in tools_data:
            tool_name = tool_data.get("name", "")
            if not tool_name:
                logger.warning(f"Skipping tool with empty name from {server_name}")
                continue

            try:
                tool = MCPTool(
                    name=tool_name,
                    description=tool_data.get("description", ""),
                    parameters=tool_data.get("inputSchema", {"type": "object", "properties": {}}),
                    server_name=server_name,
                )
                # Use server:tool_name as key to avoid conflicts
                key = tool.full_name
                self._tools[key] = tool
                logger.debug(f"Discovered MCP tool: {key}")
            except Exception as e:
                logger.warning(f"Failed to parse tool '{tool_name}' from {server_name}: {e}")

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """
        Call a tool on an MCP server.

        Args:
            server_name: Name of the MCP server.
            tool_name: Name of the tool to call.
            arguments: Arguments to pass to the tool.

        Returns:
            Tool execution result as string.
        """
        if server_name not in self._connected:
            # Try to connect
            if not await self.connect(server_name):
                return f"Error: Not connected to MCP server {server_name}"

        config = self._servers[server_name]

        try:
            if config.transport == "stdio":
                return await self._call_tool_stdio(server_name, tool_name, arguments)
            elif config.transport in ("sse", "http-stream"):
                return await self._call_tool_http(config, tool_name, arguments)
            else:
                return f"Error: Unknown transport {config.transport}"
        except Exception as e:
            logger.error(f"Error calling MCP tool {tool_name}: {e}")
            return f"Error: {str(e)}"

    async def _call_tool_stdio(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Call a tool via stdio transport."""
        process = self._processes.get(server_name)
        if not process:
            return f"Error: No process for server {server_name}"

        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            }
        }

        request_bytes = (json.dumps(request) + "\n").encode()
        process.stdin.write(request_bytes)
        await process.stdin.drain()

        try:
            response_line = await asyncio.wait_for(
                process.stdout.readline(),
                timeout=60.0
            )
            if response_line:
                response = json.loads(response_line.decode())
                if "result" in response:
                    result = response["result"]
                    if isinstance(result, dict) and "content" in result:
                        content = result["content"]
                        if isinstance(content, list) and len(content) > 0:
                            return content[0].get("text", str(result))
                    return str(result)
                elif "error" in response:
                    return f"Error: {response['error']}"
            return "No response from MCP server"
        except asyncio.TimeoutError:
            return "Error: Timeout waiting for response"

    async def _call_tool_http(
        self,
        config: MCPServerConfig,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Call a tool via HTTP transport."""
        import httpx

        async with httpx.AsyncClient(timeout=config.timeout) as client:
            response = await client.post(
                f"{config.url}/tools/call",
                json={
                    "name": tool_name,
                    "arguments": arguments,
                }
            )

            if response.status_code == 200:
                result = response.json()
                if isinstance(result, dict) and "content" in result:
                    content = result["content"]
                    if isinstance(content, list) and len(content) > 0:
                        return content[0].get("text", str(result))
                return str(result)
            else:
                return f"Error: HTTP {response.status_code}"

    def get_tools(self) -> list[MCPTool]:
        """Get all discovered tools."""
        return list(self._tools.values())

    def get_tool(self, server_name: str, tool_name: str) -> MCPTool | None:
        """Get a specific tool."""
        key = f"{server_name}:{tool_name}"
        return self._tools.get(key)

    async def disconnect(self, server_name: str) -> None:
        """Disconnect from an MCP server."""
        if server_name in self._processes:
            process = self._processes[server_name]
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                process.kill()
            del self._processes[server_name]

        # Remove tools from this server
        keys_to_remove = [k for k in self._tools if k.startswith(f"{server_name}:")]
        for key in keys_to_remove:
            del self._tools[key]

        self._connected.discard(server_name)
        logger.info(f"Disconnected from MCP server: {server_name}")

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        for server_name in list(self._connected):
            await self.disconnect(server_name)
