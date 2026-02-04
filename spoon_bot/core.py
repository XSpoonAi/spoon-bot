"""
spoon-bot core module - Direct integration with spoon-core SDK.

This module provides the primary interface for spoon-bot, using spoon-core SDK
components directly without any wrapper layers or reimplementations.

Components used from spoon-core:
- ChatBot: LLM interface with multi-provider support
- SpoonReactMCP: ReAct agent with MCP tool support
- SpoonReactSkill: Agent with skill system
- ToolManager: Tool registry and execution
- SkillManager: Skill lifecycle management
- MCPTool: MCP server integration
- ERC8004Client: Web3 identity
- X402PaymentService: Payment gating

Usage:
    from spoon_bot.core import create_agent, SpoonBot

    # Quick start with defaults
    bot = await create_agent()
    response = await bot.chat("Hello!")

    # With MCP servers
    bot = await create_agent(
        mcp_servers={
            "github": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"]}
        }
    )
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator

from loguru import logger

# Import spoon-core SDK components
try:
    from spoon_ai.chat import ChatBot
    from spoon_ai.schema import Message, ToolCall as CoreToolCall
    from spoon_ai.llm.interface import LLMResponse
    from spoon_ai.agents import SpoonReactAI
    from spoon_ai.agents.spoon_react_mcp import SpoonReactMCP
    from spoon_ai.agents.spoon_react_skill import SpoonReactSkill
    from spoon_ai.agents.skill_mixin import SkillEnabledMixin
    from spoon_ai.tools import BaseTool, ToolManager
    from spoon_ai.tools.mcp_tool import MCPTool
    from spoon_ai.skills import SkillManager
    from spoon_ai.graph import StateGraph

    # Optional modules
    try:
        from spoon_ai.identity import ERC8004Client
    except ImportError:
        ERC8004Client = None

    try:
        from spoon_ai.payments import X402PaymentService
    except ImportError:
        X402PaymentService = None

    SPOON_CORE_AVAILABLE = True
    logger.info("spoon-core SDK loaded successfully")

except ImportError as e:
    logger.error(f"spoon-core SDK is required but not installed: {e}")
    logger.error("Install with: pip install spoon-ai")
    raise ImportError(
        "spoon-bot requires spoon-core SDK. Install with: pip install spoon-ai"
    ) from e


@dataclass
class SpoonBotConfig:
    """Configuration for SpoonBot."""

    # LLM settings
    model: str = "claude-sonnet-4-20250514"
    provider: str = "anthropic"
    api_key: str | None = None
    base_url: str | None = None

    # Agent settings
    max_steps: int = 15
    system_prompt: str | None = None

    # MCP servers
    mcp_servers: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Skills
    enable_skills: bool = True
    skill_paths: list[str] = field(default_factory=list)

    # Workspace
    workspace: Path = field(default_factory=lambda: Path.home() / ".spoon-bot" / "workspace")

    # Memory
    enable_memory: bool = True

    @classmethod
    def from_env(cls) -> "SpoonBotConfig":
        """Create config from environment variables."""
        return cls(
            model=os.environ.get("SPOON_MODEL", "claude-sonnet-4-20250514"),
            provider=os.environ.get("SPOON_PROVIDER", "anthropic"),
            api_key=os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            max_steps=int(os.environ.get("SPOON_MAX_STEPS", "15")),
        )


class SpoonBot:
    """
    Main spoon-bot class using spoon-core SDK directly.

    This class provides a unified interface for agent interaction,
    delegating all functionality to spoon-core components.
    """

    def __init__(self, config: SpoonBotConfig | None = None):
        """
        Initialize SpoonBot.

        Args:
            config: Bot configuration. If None, loads from environment.
        """
        self.config = config or SpoonBotConfig.from_env()
        self._chatbot: ChatBot | None = None
        self._agent: SpoonReactMCP | SpoonReactSkill | None = None
        self._tool_manager: ToolManager | None = None
        self._skill_manager: SkillManager | None = None
        self._mcp_tools: list[MCPTool] = []
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return

        # Ensure workspace exists
        self.config.workspace.mkdir(parents=True, exist_ok=True)

        # Create ChatBot (spoon-core's LLM interface)
        self._chatbot = ChatBot(
            model_name=self.config.model,
            llm_provider=self.config.provider,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            system_prompt=self.config.system_prompt,
        )

        # Create MCP tools
        for name, mcp_config in self.config.mcp_servers.items():
            mcp_tool = MCPTool(
                name=name,
                description=f"MCP server: {name}",
                mcp_config=mcp_config,
            )
            self._mcp_tools.append(mcp_tool)

        # Create ToolManager
        if self._mcp_tools:
            self._tool_manager = ToolManager(self._mcp_tools)

        # Create SkillManager if enabled
        if self.config.enable_skills:
            skill_paths = self.config.skill_paths or [str(self.config.workspace / "skills")]
            self._skill_manager = SkillManager(
                skill_paths=skill_paths,
                llm=self._chatbot,
                auto_discover=True,
            )

        # Create appropriate agent based on configuration
        if self.config.enable_skills and self._skill_manager:
            self._agent = SpoonReactSkill(
                llm=self._chatbot,
                tools=self._tool_manager,
                skill_manager=self._skill_manager,
                max_steps=self.config.max_steps,
            )
        else:
            self._agent = SpoonReactMCP(
                llm=self._chatbot,
                tools=self._tool_manager,
                max_steps=self.config.max_steps,
            )

        # Initialize agent
        await self._agent.initialize()

        self._initialized = True
        logger.info(
            f"SpoonBot initialized: model={self.config.model}, "
            f"provider={self.config.provider}, "
            f"mcp_servers={len(self._mcp_tools)}, "
            f"skills={self.config.enable_skills}"
        )

    async def chat(self, message: str, **kwargs: Any) -> str:
        """
        Send a message and get a response.

        Args:
            message: User message.
            **kwargs: Additional arguments for the agent.

        Returns:
            Agent response.
        """
        if not self._initialized:
            await self.initialize()

        result = await self._agent.run(message, **kwargs)

        if hasattr(result, "content"):
            return result.content
        return str(result)

    async def stream(self, message: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        """
        Stream a response.

        Args:
            message: User message.
            **kwargs: Additional arguments.

        Yields:
            Response chunks.
        """
        if not self._initialized:
            await self.initialize()

        async for chunk in self._agent.stream(message, **kwargs):
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content
            elif isinstance(chunk, str):
                yield chunk

    async def ask_with_tools(
        self,
        message: str,
        tools: list[BaseTool] | None = None,
        **kwargs: Any,
    ) -> tuple[str, list[CoreToolCall]]:
        """
        Ask with explicit tool list.

        Args:
            message: User message.
            tools: Tools to make available.
            **kwargs: Additional arguments.

        Returns:
            Tuple of (response, tool_calls).
        """
        if not self._initialized:
            await self.initialize()

        response = await self._chatbot.ask_tool(
            messages=[{"role": "user", "content": message}],
            tools=tools,
            **kwargs,
        )

        content = response.content if hasattr(response, "content") else str(response)
        tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []

        return content, tool_calls

    def add_tool(self, tool: BaseTool) -> None:
        """Add a tool to the agent."""
        if self._tool_manager:
            self._tool_manager.add_tool(tool)
        else:
            self._tool_manager = ToolManager([tool])

    def add_mcp_server(self, name: str, config: dict[str, Any]) -> None:
        """Add an MCP server configuration."""
        self.config.mcp_servers[name] = config
        mcp_tool = MCPTool(name=name, description=f"MCP server: {name}", mcp_config=config)
        self._mcp_tools.append(mcp_tool)
        if self._tool_manager:
            self._tool_manager.add_tool(mcp_tool)

    async def activate_skill(self, skill_name: str, context: dict | None = None) -> bool:
        """Activate a skill."""
        if not self._skill_manager:
            return False
        try:
            await self._skill_manager.activate(skill_name, context=context)
            return True
        except Exception as e:
            logger.error(f"Failed to activate skill {skill_name}: {e}")
            return False

    async def deactivate_skill(self, skill_name: str) -> bool:
        """Deactivate a skill."""
        if not self._skill_manager:
            return False
        return await self._skill_manager.deactivate(skill_name)

    @property
    def tools(self) -> list[str]:
        """Get list of available tool names."""
        if self._tool_manager:
            return list(self._tool_manager.tools.keys())
        return []

    @property
    def skills(self) -> list[str]:
        """Get list of available skill names."""
        if self._skill_manager:
            return self._skill_manager.list()
        return []

    @property
    def chatbot(self) -> ChatBot:
        """Access the underlying ChatBot."""
        return self._chatbot

    @property
    def agent(self) -> SpoonReactMCP | SpoonReactSkill:
        """Access the underlying agent."""
        return self._agent

    @property
    def tool_manager(self) -> ToolManager | None:
        """Access the ToolManager."""
        return self._tool_manager

    @property
    def skill_manager(self) -> SkillManager | None:
        """Access the SkillManager."""
        return self._skill_manager

    def get_status(self) -> dict[str, Any]:
        """Get bot status."""
        return {
            "initialized": self._initialized,
            "model": self.config.model,
            "provider": self.config.provider,
            "tools": self.tools,
            "skills": self.skills,
            "mcp_servers": list(self.config.mcp_servers.keys()),
        }


async def create_agent(
    model: str = "claude-sonnet-4-20250514",
    provider: str = "anthropic",
    mcp_servers: dict[str, dict[str, Any]] | None = None,
    enable_skills: bool = True,
    skill_paths: list[str] | None = None,
    workspace: str | Path | None = None,
    **kwargs: Any,
) -> SpoonBot:
    """
    Create and initialize a SpoonBot agent.

    This is the recommended way to create an agent.

    Args:
        model: Model name.
        provider: LLM provider (anthropic, openai, etc.)
        mcp_servers: MCP server configurations.
        enable_skills: Whether to enable skill system.
        skill_paths: Paths to search for skills.
        workspace: Workspace directory.
        **kwargs: Additional config options.

    Returns:
        Initialized SpoonBot instance.

    Example:
        >>> bot = await create_agent()
        >>> response = await bot.chat("Hello!")

        >>> bot = await create_agent(
        ...     mcp_servers={"github": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"]}}
        ... )
    """
    config = SpoonBotConfig(
        model=model,
        provider=provider,
        mcp_servers=mcp_servers or {},
        enable_skills=enable_skills,
        skill_paths=skill_paths or [],
        workspace=Path(workspace) if workspace else Path.home() / ".spoon-bot" / "workspace",
        **kwargs,
    )

    bot = SpoonBot(config)
    await bot.initialize()
    return bot


# Re-export spoon-core types for convenience
__all__ = [
    "SpoonBot",
    "SpoonBotConfig",
    "create_agent",
    # spoon-core types
    "ChatBot",
    "Message",
    "LLMResponse",
    "BaseTool",
    "ToolManager",
    "MCPTool",
    "SkillManager",
    "SpoonReactMCP",
    "SpoonReactSkill",
    "ERC8004Client",
    "X402PaymentService",
]
