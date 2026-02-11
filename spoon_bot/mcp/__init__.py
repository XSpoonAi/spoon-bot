"""
MCP module - Uses spoon-core SDK directly.

All MCP functionality is provided by spoon-core's MCPTool and MCPClientMixin.
No local reimplementations - use spoon-core directly.
"""

try:
    from spoon_ai.tools.mcp_tool import MCPTool
    from spoon_ai.agents.mcp_client_mixin import MCPClientMixin
    from spoon_ai.graph.mcp_integration import MCPIntegrationManager, MCPConfigManager

    __all__ = [
        "MCPTool",
        "MCPClientMixin",
        "MCPIntegrationManager",
        "MCPConfigManager",
    ]

except ImportError:
    raise ImportError(
        "spoon-bot requires spoon-core SDK for MCP functionality. "
        "Install with: pip install spoon-ai"
    )
