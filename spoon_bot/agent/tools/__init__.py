"""Agent tools: native OS tools and tool registry."""

from spoon_bot.agent.tools.base import Tool
from spoon_bot.agent.tools.registry import ToolRegistry
from spoon_bot.agent.tools.shell import (
    ShellTool,
    SafeShellTool,
    CommandValidator,
    ShellSecurityError,
)
from spoon_bot.agent.tools.filesystem import (
    ReadFileTool,
    WriteFileTool,
    EditFileTool,
    ListDirTool,
)
from spoon_bot.agent.tools.path_validator import (
    PathValidator,
    PathValidationResult,
    validate_read_path,
    validate_write_path,
    validate_directory_path,
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
    "SafeShellTool",
    "CommandValidator",
    "ShellSecurityError",
    # Filesystem
    "ReadFileTool",
    "WriteFileTool",
    "EditFileTool",
    "ListDirTool",
    # Path Validation
    "PathValidator",
    "PathValidationResult",
    "validate_read_path",
    "validate_write_path",
    "validate_directory_path",
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
