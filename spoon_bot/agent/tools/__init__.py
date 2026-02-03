"""Agent tools: native OS tools and tool registry."""

from spoon_bot.agent.tools.base import Tool
from spoon_bot.agent.tools.registry import ToolRegistry
from spoon_bot.agent.tools.shell import ShellTool
from spoon_bot.agent.tools.filesystem import (
    ReadFileTool,
    WriteFileTool,
    EditFileTool,
    ListDirTool,
)
from spoon_bot.agent.tools.self_config import (
    SelfConfigTool,
    MemoryManagementTool,
    SelfUpgradeTool,
)
from spoon_bot.agent.tools.web3 import (
    BalanceCheckTool,
    TransferTool,
    SwapTool,
    ContractCallTool,
)
from spoon_bot.agent.tools.web import (
    WebSearchTool,
    WebFetchTool,
    WebBrowserTool,
)

__all__ = [
    # Base
    "Tool",
    "ToolRegistry",
    # Shell
    "ShellTool",
    # Filesystem
    "ReadFileTool",
    "WriteFileTool",
    "EditFileTool",
    "ListDirTool",
    # Self-Config
    "SelfConfigTool",
    "MemoryManagementTool",
    "SelfUpgradeTool",
    # Web3
    "BalanceCheckTool",
    "TransferTool",
    "SwapTool",
    "ContractCallTool",
    # Web
    "WebSearchTool",
    "WebFetchTool",
    "WebBrowserTool",
]
