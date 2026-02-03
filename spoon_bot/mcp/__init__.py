"""MCP integration for spoon-bot.

This module provides MCP (Model Context Protocol) integration with support for:
- Local implementation (MCPClientAdapter, MCPToolAdapter)
- spoon-core integration (SpoonCoreMCPAdapter) with enhanced transport support

The SpoonCoreMCPAdapter automatically uses spoon-core's MCPTool if available,
falling back to the local implementation otherwise.
"""

from spoon_bot.mcp.client import MCPClientAdapter
from spoon_bot.mcp.tool_adapter import MCPToolAdapter
from spoon_bot.mcp.spoon_core_mcp import SpoonCoreMCPAdapter

__all__ = ["MCPClientAdapter", "MCPToolAdapter", "SpoonCoreMCPAdapter"]
