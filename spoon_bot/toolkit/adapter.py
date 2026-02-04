"""Adapter for spoon-toolkits integration."""

from __future__ import annotations

import asyncio
from typing import Any, TYPE_CHECKING

from loguru import logger

from spoon_bot.agent.tools.base import Tool
from spoon_bot.exceptions import ToolExecutionError, ToolTimeoutError

if TYPE_CHECKING:
    pass


class ToolkitToolWrapper(Tool):
    """Wrapper that adapts spoon-toolkit tools to spoon-bot Tool interface."""

    def __init__(self, toolkit_tool: Any):
        """
        Initialize wrapper.

        Args:
            toolkit_tool: Instance from spoon-toolkits.
        """
        self._tool = toolkit_tool

    @property
    def name(self) -> str:
        return getattr(self._tool, "name", self._tool.__class__.__name__)

    @property
    def description(self) -> str:
        return getattr(self._tool, "description", "No description")

    @property
    def parameters(self) -> dict[str, Any]:
        # Try to get parameters from tool
        if hasattr(self._tool, "parameters"):
            return self._tool.parameters
        elif hasattr(self._tool, "input_schema"):
            return self._tool.input_schema
        elif hasattr(self._tool, "args_schema"):
            # Pydantic schema
            schema = self._tool.args_schema
            if hasattr(schema, "model_json_schema"):
                return schema.model_json_schema()
            elif hasattr(schema, "schema"):
                return schema.schema()
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        """Execute the toolkit tool.

        Returns:
            Tool execution result as string.
            In case of error, returns a user-friendly error message.
        """
        try:
            # Check if tool has async execute
            if hasattr(self._tool, "arun"):
                result = await self._tool.arun(**kwargs)
            elif hasattr(self._tool, "execute"):
                if asyncio.iscoroutinefunction(self._tool.execute):
                    result = await self._tool.execute(**kwargs)
                else:
                    # Run sync function in executor to avoid blocking
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: self._tool.execute(**kwargs)
                    )
            elif hasattr(self._tool, "run"):
                # Run sync function in executor to avoid blocking
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._tool.run(**kwargs)
                )
            elif callable(self._tool):
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._tool(**kwargs)
                )
            else:
                return f"Error: Tool '{self.name}' is not callable"

            return str(result) if result is not None else "Success"

        except asyncio.TimeoutError as e:
            logger.error(f"Toolkit tool {self.name} timed out: {e}")
            return f"Error: Tool '{self.name}' timed out"
        except asyncio.CancelledError:
            logger.warning(f"Toolkit tool {self.name} was cancelled")
            return f"Error: Tool '{self.name}' was cancelled"
        except ConnectionError as e:
            logger.error(f"Toolkit tool {self.name} connection error: {e}")
            return f"Error: Connection failed for tool '{self.name}'"
        except PermissionError as e:
            logger.error(f"Toolkit tool {self.name} permission error: {e}")
            return f"Error: Permission denied for tool '{self.name}'"
        except ValueError as e:
            logger.error(f"Toolkit tool {self.name} value error: {e}")
            return f"Error: Invalid arguments for tool '{self.name}': {e}"
        except Exception as e:
            error_type = type(e).__name__
            logger.error(f"Toolkit tool {self.name} error ({error_type}): {e}")
            return f"Error executing '{self.name}': {str(e)}"


class ToolkitAdapter:
    """
    Adapter for loading and using spoon-toolkits tools.

    Supports lazy loading to avoid import errors when toolkit is not installed.
    """

    def __init__(self):
        """Initialize toolkit adapter."""
        self._tools: list[Tool] = []
        self._loaded_categories: set[str] = set()

    def load_crypto_tools(self) -> list[Tool]:
        """Load crypto data tools."""
        if "crypto" in self._loaded_categories:
            return [t for t in self._tools if t.name.startswith("crypto_")]

        tools = []
        try:
            from spoon_toolkits.crypto.crypto_data_tools import (
                GetTokenPriceTool,
                Get24hStatsTool,
                GetKlineDataTool,
                PriceThresholdAlertTool,
                SuddenPriceIncreaseTool,
                LendingRateMonitorTool,
            )

            toolkit_tools = [
                GetTokenPriceTool(),
                Get24hStatsTool(),
                GetKlineDataTool(),
                PriceThresholdAlertTool(),
                SuddenPriceIncreaseTool(),
                LendingRateMonitorTool(),
            ]

            for tool in toolkit_tools:
                wrapped = ToolkitToolWrapper(tool)
                tools.append(wrapped)
                self._tools.append(wrapped)

            self._loaded_categories.add("crypto")
            logger.info(f"Loaded {len(tools)} crypto tools")

        except ImportError as e:
            logger.warning(f"spoon-toolkits crypto tools not available: {e}")

        return tools

    def load_blockchain_tools(self) -> list[Tool]:
        """Load blockchain data tools (Chainbase, ThirdWeb)."""
        if "blockchain" in self._loaded_categories:
            return [t for t in self._tools if "chain" in t.name.lower() or "thirdweb" in t.name.lower()]

        tools = []
        try:
            from spoon_toolkits.data_platforms.chainbase import (
                GetLatestBlockNumberTool,
                GetBlockByNumberTool,
                GetTransactionByHashTool,
                GetAccountTransactionsTool,
                GetAccountTokensTool,
                GetAccountBalanceTool,
            )

            toolkit_tools = [
                GetLatestBlockNumberTool(),
                GetBlockByNumberTool(),
                GetTransactionByHashTool(),
                GetAccountTransactionsTool(),
                GetAccountTokensTool(),
                GetAccountBalanceTool(),
            ]

            for tool in toolkit_tools:
                wrapped = ToolkitToolWrapper(tool)
                tools.append(wrapped)
                self._tools.append(wrapped)

            self._loaded_categories.add("blockchain")
            logger.info(f"Loaded {len(tools)} blockchain tools")

        except ImportError as e:
            logger.warning(f"spoon-toolkits blockchain tools not available: {e}")

        return tools

    def load_security_tools(self) -> list[Tool]:
        """Load security tools (GoPlusLabs)."""
        if "security" in self._loaded_categories:
            return [t for t in self._tools if "security" in t.name.lower()]

        tools = []
        try:
            from spoon_toolkits.security.gopluslabs import get_token_risk_and_security_data

            # Wrap the function as a tool
            class TokenSecurityTool(Tool):
                @property
                def name(self) -> str:
                    return "token_security_check"

                @property
                def description(self) -> str:
                    return "Check token risk and security data using GoPlusLabs"

                @property
                def parameters(self) -> dict[str, Any]:
                    return {
                        "type": "object",
                        "properties": {
                            "token_address": {"type": "string", "description": "Token contract address"},
                            "chain_id": {"type": "string", "description": "Chain ID (e.g., '1' for Ethereum)"},
                        },
                        "required": ["token_address", "chain_id"],
                    }

                async def execute(self, **kwargs: Any) -> str:
                    import asyncio
                    result = await asyncio.to_thread(
                        get_token_risk_and_security_data,
                        kwargs.get("token_address"),
                        kwargs.get("chain_id"),
                    )
                    return str(result)

            security_tool = TokenSecurityTool()
            tools.append(security_tool)
            self._tools.append(security_tool)

            self._loaded_categories.add("security")
            logger.info(f"Loaded {len(tools)} security tools")

        except ImportError as e:
            logger.warning(f"spoon-toolkits security tools not available: {e}")

        return tools

    def load_social_tools(self) -> list[Tool]:
        """Load social media tools."""
        if "social" in self._loaded_categories:
            return [t for t in self._tools if "social" in t.name.lower()]

        tools = []
        try:
            from spoon_toolkits.social_media import TwitterTool

            twitter_tool = ToolkitToolWrapper(TwitterTool())
            tools.append(twitter_tool)
            self._tools.append(twitter_tool)

            self._loaded_categories.add("social")
            logger.info(f"Loaded {len(tools)} social tools")

        except ImportError as e:
            logger.warning(f"spoon-toolkits social tools not available: {e}")

        return tools

    def load_all(self) -> list[Tool]:
        """Load all available toolkit tools."""
        all_tools = []
        all_tools.extend(self.load_crypto_tools())
        all_tools.extend(self.load_blockchain_tools())
        all_tools.extend(self.load_security_tools())
        all_tools.extend(self.load_social_tools())
        return all_tools

    def get_loaded_tools(self) -> list[Tool]:
        """Get all loaded tools."""
        return self._tools.copy()

    @property
    def loaded_categories(self) -> set[str]:
        """Get set of loaded categories."""
        return self._loaded_categories.copy()
