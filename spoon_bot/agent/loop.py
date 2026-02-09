"""
Agent loop: the core processing engine using spoon-core SDK.

This module provides the main agent interface, integrating spoon-core's
ChatBot, SpoonReactMCP, and SkillManager with spoon-bot's native OS tools.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncGenerator

from loguru import logger

# Import spoon-core SDK components (required)
try:
    from spoon_ai.chat import ChatBot
    from spoon_ai.schema import Message, ToolCall as CoreToolCall
    from spoon_ai.llm.interface import LLMResponse
    from spoon_ai.agents.spoon_react_mcp import SpoonReactMCP
    from spoon_ai.agents.spoon_react_skill import SpoonReactSkill
    from spoon_ai.tools import BaseTool, ToolManager
    from spoon_ai.tools.mcp_tool import MCPTool
    from spoon_ai.skills import SkillManager

    SPOON_CORE_AVAILABLE = True
except ImportError as e:
    logger.error(f"spoon-core SDK is required: {e}")
    raise ImportError(
        "spoon-bot requires spoon-core SDK. Install with: pip install spoon-ai"
    ) from e

# Import spoon-bot native tools and components
from spoon_bot.agent.context import ContextBuilder
from spoon_bot.agent.tools.registry import CORE_TOOLS, ToolRegistry
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
from spoon_bot.agent.tools.web import WebSearchTool, WebFetchTool
from spoon_bot.agent.tools.document import DocumentParseTool
from spoon_bot.config import AgentLoopConfig, validate_agent_loop_params
from spoon_bot.services.spawn import SpawnTool
from spoon_bot.session.manager import SessionManager
from spoon_bot.memory.store import MemoryStore
from spoon_bot.exceptions import (
    SpoonBotError,
    APIKeyMissingError,
    LLMError,
    LLMConnectionError,
    LLMTimeoutError,
    SkillActivationError,
    user_friendly_error,
)
from spoon_bot.services.git import GitManager

if TYPE_CHECKING:
    from spoon_bot.session.manager import Session


class AgentLoop:
    """
    The agent loop is the core processing engine using spoon-core SDK.

    It uses:
    - spoon-core's ChatBot for LLM interactions
    - spoon-core's SpoonReactMCP/SpoonReactSkill for agent orchestration
    - spoon-core's SkillManager for skill lifecycle
    - spoon-core's MCPTool for MCP server integration
    - spoon-bot's native OS tools (shell, filesystem, etc.)
    """

    # Task-agnostic next_step_prompt that replaces the crypto-centric default.
    # The original spoon-core template ("You can interact with the Neo blockchain …
    # Pick tools by matching the user's request to the tool names …") is injected
    # as a USER message every step, causing the LLM to waste iterations echoing
    # policy text instead of doing useful work.  This minimal version avoids that.
    DEFAULT_NEXT_STEP_PROMPT = (
        "Continue with the next step. "
        "When the task is fully complete, provide your final answer."
    )

    def __init__(
        self,
        workspace: Path | str | None = None,
        model: str = "claude-sonnet-4-20250514",
        provider: str = "anthropic",
        api_key: str | None = None,
        base_url: str | None = None,
        max_iterations: int = 20,
        shell_timeout: int = 60,
        max_output: int = 10000,
        session_key: str = "default",
        skill_paths: list[Path | str] | None = None,
        mcp_config: dict[str, dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        enable_skills: bool = True,
        auto_commit: bool = True,
        enabled_tools: set[str] | None = None,
        tool_profile: str | None = None,
    ) -> None:
        """
        Initialize the agent loop.

        Args:
            workspace: Path to workspace directory.
            model: Model name to use.
            provider: LLM provider (anthropic, openai, etc.)
            api_key: API key for the LLM provider.
            base_url: Custom base URL for the provider.
            max_iterations: Maximum tool call iterations.
            shell_timeout: Shell command timeout in seconds.
            max_output: Maximum output characters for shell.
            session_key: Session identifier for persistence.
            skill_paths: Additional paths to search for skills.
            mcp_config: MCP server configurations.
            system_prompt: Custom system prompt.
            enable_skills: Whether to enable skill system.
            auto_commit: Whether to auto-commit workspace changes after each message.
            enabled_tools: Explicit set of tool names to enable. None = all.
            tool_profile: Named profile ('coding', 'web3', 'research', 'full').
        """
        # Validate parameters
        try:
            self._config = validate_agent_loop_params(
                workspace=workspace,
                model=model,
                max_iterations=max_iterations,
                shell_timeout=shell_timeout,
                max_output=max_output,
                session_key=session_key,
                skill_paths=skill_paths,
                mcp_config=mcp_config,
            )
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise ValueError(f"Invalid AgentLoop configuration: {e}") from e

        # Store config
        self.workspace = self._config.workspace
        self.model = model
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        self.max_iterations = self._config.max_iterations
        self.shell_timeout = self._config.shell_timeout
        self.max_output = self._config.max_output
        self.session_key = self._config.session_key
        self._enable_skills = enable_skills
        self._mcp_config = mcp_config or {}
        self._system_prompt = system_prompt
        self._auto_commit = auto_commit

        # spoon-core components (initialized later)
        self._chatbot: ChatBot | None = None
        self._agent: SpoonReactMCP | SpoonReactSkill | None = None
        self._skill_manager: SkillManager | None = None
        self._mcp_tools: list[MCPTool] = []

        # spoon-bot components
        self.context = ContextBuilder(self.workspace)
        self.tools = ToolRegistry()
        self.sessions = SessionManager(self.workspace)
        self.memory = MemoryStore(self.workspace)
        self._git = GitManager(self.workspace) if auto_commit else None

        # Skill paths
        self._skill_paths = [self.workspace / "skills"]
        if skill_paths:
            self._skill_paths.extend([Path(p) for p in skill_paths])

        # Session
        self._session = self.sessions.get_or_create(self.session_key)

        # Inject memory context
        memory_context = self.memory.get_memory_context()
        if memory_context:
            self.context.set_memory_context(memory_context)

        # Register native tools
        self._register_native_tools()

        # Apply tool filter: explicit > profile > core default
        if enabled_tools is not None or tool_profile is not None:
            self.tools.set_tool_filter(
                enabled_tools=enabled_tools,
                tool_profile=tool_profile,
            )
        else:
            # Default: only load core tools into the agent.
            # All other tools remain in the registry and can be activated
            # dynamically via add_tool().
            self.tools.set_tool_filter(enabled_tools=set(CORE_TOOLS))

        # Track initialization state
        self._initialized = False

        active_count = len(self.tools)
        total_count = len(self.tools._tools)
        logger.info(
            f"AgentLoop created: model={model}, provider={provider}, "
            f"tools={active_count}/{total_count}, session={session_key}"
        )

    async def initialize(self) -> None:
        """Initialize spoon-core components."""
        if self._initialized:
            return

        # Build system prompt (spoon-bot context + available tool summaries)
        system_prompt = self._system_prompt or self.context.build_system_prompt()

        # Append available-tools summary so the LLM knows what can be loaded
        inactive_tools = self.tools.get_inactive_tools()
        if inactive_tools:
            tool_lines = "\n".join(
                f"- {t.name}: {t.description}"
                for t in inactive_tools.values()
            )
            system_prompt += (
                "\n\n## Dynamically Loadable Tools\n\n"
                "The following tools are available but not currently loaded. "
                "If you need any of them, tell the user which tool you need "
                "and they can be loaded on demand.\n\n"
                f"{tool_lines}"
            )

        # Create ChatBot (spoon-core LLM interface)
        self._chatbot = ChatBot(
            model_name=self.model,
            llm_provider=self.provider,
            api_key=self.api_key,
            base_url=self.base_url,
            system_prompt=system_prompt,
        )

        # Create MCP tools
        for name, config in self._mcp_config.items():
            mcp_tool = MCPTool(
                name=name,
                description=f"MCP server: {name}",
                mcp_config=config,
            )
            self._mcp_tools.append(mcp_tool)

        # Only pass active (filtered) tools + MCP tools to the agent
        active_tools = list(self.tools.get_active_tools().values()) + self._mcp_tools

        # Common agent kwargs
        agent_kwargs: dict[str, Any] = {
            "llm": self._chatbot,
            "tools": ToolManager(active_tools) if active_tools else None,
            "max_steps": self.max_iterations,
            "system_prompt": system_prompt,
            "next_step_prompt": self.DEFAULT_NEXT_STEP_PROMPT,
        }

        # Create SkillManager if enabled
        if self._enable_skills:
            self._skill_manager = SkillManager(
                skill_paths=[str(p) for p in self._skill_paths],
                llm=self._chatbot,
                auto_discover=True,
                include_default_paths=False,
            )
            agent_kwargs["skill_manager"] = self._skill_manager
            self._agent = SpoonReactSkill(**agent_kwargs)
        else:
            self._agent = SpoonReactMCP(**agent_kwargs)

        # Initialize agent
        await self._agent.initialize()

        # Increase default step timeout for proxy/custom endpoints
        if self.base_url:
            self._agent._default_timeout = 300.0

        self._initialized = True
        active_count = len(self.tools.get_active_tools())
        total_count = len(self.tools._tools)
        logger.info(
            f"AgentLoop initialized: tools={active_count}/{total_count}, "
            f"mcp_servers={len(self._mcp_tools)}, "
            f"skills_enabled={self._enable_skills}"
        )

    def _register_native_tools(self) -> None:
        """Register the default native OS tools."""
        # Shell tool
        self.tools.register(ShellTool(
            timeout=self.shell_timeout,
            max_output=self.max_output,
            working_dir=str(self.workspace),
        ))

        # Filesystem tools
        self.tools.register(ReadFileTool(workspace=self.workspace))
        self.tools.register(WriteFileTool(workspace=self.workspace))
        self.tools.register(EditFileTool(workspace=self.workspace))
        self.tools.register(ListDirTool(workspace=self.workspace))

        # Self-management tools
        self.tools.register(SelfConfigTool())
        memory_tool = MemoryManagementTool()
        memory_tool.set_memory_store(self.memory)
        self.tools.register(memory_tool)
        self.tools.register(SelfUpgradeTool(workspace=self.workspace))

        # Background task tool
        self.tools.register(SpawnTool())

        # Web tools
        self.tools.register(WebSearchTool())
        self.tools.register(WebFetchTool())

        # Document processing tools
        self.tools.register(DocumentParseTool(workspace=self.workspace))

        # Web3 tools
        self.tools.register(BalanceCheckTool())
        self.tools.register(TransferTool())
        self.tools.register(SwapTool())
        self.tools.register(ContractCallTool())

        # Toolkit tools (optional)
        try:
            from spoon_bot.toolkit.adapter import ToolkitAdapter
            toolkit = ToolkitAdapter()
            for tool in toolkit.load_all():
                self.tools.register(tool)
        except ImportError:
            logger.debug("spoon-toolkits not available, skipping toolkit tools")

        logger.debug(f"Registered native tools: {self.tools.list_tools()}")

    async def process(
        self,
        message: str,
        media: list[str] | None = None,
    ) -> str:
        """
        Process a user message and return the agent's response.

        Args:
            message: The user's message.
            media: Optional list of media file paths.

        Returns:
            The agent's response text.
        """
        # Ensure initialized
        if not self._initialized:
            await self.initialize()

        logger.info(f"Processing message: {message[:100]}...")

        # Refresh memory context
        try:
            memory_context = self.memory.get_memory_context()
            if memory_context:
                self.context.set_memory_context(memory_context)
        except Exception as e:
            logger.warning(f"Failed to load memory context: {e}")

        # Run agent
        try:
            result = await self._agent.run(message)

            # Extract content
            if hasattr(result, "content"):
                final_content = result.content
            else:
                final_content = str(result)

        except Exception as e:
            logger.error(f"Agent processing error: {e}")
            return f"I encountered an error: {user_friendly_error(e)}"

        # Save to session
        try:
            self._session.add_message("user", message)
            self._session.add_message("assistant", final_content)
            self.sessions.save(self._session)
        except Exception as e:
            logger.warning(f"Failed to save session: {e}")

        # Auto-commit workspace changes
        if self._auto_commit and self._git:
            try:
                if self._git.has_changes():
                    self._git.commit(message)
            except Exception as e:
                logger.warning(f"Failed to auto-commit: {e}")

        return final_content

    async def stream(
        self,
        message: str,
        media: list[str] | None = None,
        thinking: bool = False,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream a response with typed chunks.

        Args:
            message: User message.
            media: Optional media files.
            thinking: Whether to include thinking output.

        Yields:
            Dicts with keys:
              type:     "content" | "thinking" | "tool_call" | "tool_result" | "done"
              delta:    Incremental text (may be empty for non-text events)
              metadata: Extra context (tool name, args, step number, etc.)
        """
        if not self._initialized:
            await self.initialize()

        kwargs: dict[str, Any] = {}
        if thinking:
            kwargs["thinking"] = True
        if media:
            kwargs["media"] = media

        full_content = ""
        try:
            async for chunk in self._agent.stream(message, **kwargs):
                chunk_type = "content"
                delta = ""
                metadata: dict[str, Any] = {}

                # -- Thinking chunks --
                if hasattr(chunk, "type") and chunk.type == "thinking":
                    chunk_type = "thinking"
                    delta = chunk.content if hasattr(chunk, "content") else str(chunk)
                elif (
                    hasattr(chunk, "metadata")
                    and isinstance(chunk.metadata, dict)
                    and chunk.metadata.get("type") == "thinking"
                ):
                    chunk_type = "thinking"
                    delta = (
                        getattr(chunk, "delta", None)
                        or getattr(chunk, "content", None)
                        or str(chunk)
                    )

                # -- Dict chunks (tool_calls, structured events) --
                elif isinstance(chunk, dict):
                    if "tool_calls" in chunk and chunk["tool_calls"]:
                        for tc in chunk["tool_calls"]:
                            fn = tc.get("function", {})
                            yield {
                                "type": "tool_call",
                                "delta": "",
                                "metadata": {
                                    "id": tc.get("id", ""),
                                    "name": fn.get("name", ""),
                                    "arguments": fn.get("arguments", ""),
                                },
                            }
                        continue
                    if "content" in chunk and chunk["content"]:
                        delta = chunk["content"]
                        full_content += delta

                # -- Object chunks with content --
                elif hasattr(chunk, "content") and chunk.content:
                    delta = chunk.content
                    full_content += delta

                # -- Plain string chunks --
                elif isinstance(chunk, str):
                    delta = chunk
                    full_content += delta

                if delta:
                    yield {"type": chunk_type, "delta": delta, "metadata": metadata}

            # Emit done
            yield {"type": "done", "delta": "", "metadata": {"content": full_content}}

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield {"type": "done", "delta": "", "metadata": {"error": str(e)}}

        # Save to session only if we got actual content
        if full_content:
            try:
                self._session.add_message("user", message)
                self._session.add_message("assistant", full_content)
                self.sessions.save(self._session)
            except Exception as e:
                logger.warning(f"Failed to save session after streaming: {e}")

    async def process_with_thinking(
        self,
        message: str,
        media: list[str] | None = None,
    ) -> tuple[str, str | None]:
        """
        Process a user message and return the agent's response with thinking content.

        Args:
            message: The user's message.
            media: Optional list of media file paths.

        Returns:
            Tuple of (response_text, thinking_content). thinking_content may be None.
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"Processing message (with thinking): {message[:100]}...")

        # Refresh memory context
        try:
            memory_context = self.memory.get_memory_context()
            if memory_context:
                self.context.set_memory_context(memory_context)
        except Exception as e:
            logger.warning(f"Failed to load memory context: {e}")

        # Run agent with thinking enabled
        try:
            run_kwargs: dict[str, Any] = {"thinking": True}
            if media:
                run_kwargs["media"] = media
            result = await self._agent.run(message, **run_kwargs)

            # Extract content and thinking
            if hasattr(result, "content"):
                final_content = result.content
            else:
                final_content = str(result)

            thinking_content = None
            if hasattr(result, "thinking_content"):
                thinking_content = result.thinking_content
            elif hasattr(result, "thinking"):
                thinking_content = result.thinking
            elif hasattr(result, "metadata") and isinstance(result.metadata, dict):
                thinking_content = result.metadata.get("thinking")

        except Exception as e:
            logger.error(f"Agent processing error: {e}")
            return f"I encountered an error: {user_friendly_error(e)}", None

        # Save to session
        try:
            self._session.add_message("user", message)
            self._session.add_message("assistant", final_content)
            self.sessions.save(self._session)
        except Exception as e:
            logger.warning(f"Failed to save session: {e}")

        # Auto-commit workspace changes
        if self._auto_commit and self._git:
            try:
                if self._git.has_changes():
                    self._git.commit(message)
            except Exception as e:
                logger.warning(f"Failed to auto-commit: {e}")

        return final_content, thinking_content

    # ------------------------------------------------------------------
    # Dynamic tool management
    # ------------------------------------------------------------------

    def add_tool(self, name: str) -> bool:
        """
        Dynamically activate a tool and inject it into the running agent.

        The tool must already be registered in the ToolRegistry.  This method
        activates it in the filter, then adds it to the agent's ToolManager
        so it becomes available for the next LLM step.

        Args:
            name: Registered tool name to activate.

        Returns:
            True if the tool was activated, False otherwise.
        """
        tool = self.tools.get(name)
        if tool is None:
            logger.warning(f"add_tool: '{name}' is not registered")
            return False

        if not self.tools.activate_tool(name):
            logger.debug(f"add_tool: '{name}' is already active")
            return False

        # If the agent is initialized, inject the tool into its ToolManager
        if self._agent and hasattr(self._agent, "available_tools"):
            tm: ToolManager = self._agent.available_tools
            if name not in tm.tool_map:
                tm.add_tool(tool)
                logger.info(f"Injected tool '{name}' into running agent")

        return True

    def add_tools(self, *names: str) -> list[str]:
        """
        Activate multiple tools at once.

        Args:
            *names: Tool names to activate.

        Returns:
            List of tool names that were successfully activated.
        """
        activated = []
        for name in names:
            if self.add_tool(name):
                activated.append(name)
        return activated

    def remove_tool(self, name: str) -> bool:
        """
        Dynamically deactivate a tool and remove it from the running agent.

        Args:
            name: Tool name to deactivate.

        Returns:
            True if the tool was deactivated, False otherwise.
        """
        if not self.tools.deactivate_tool(name):
            return False

        # Remove from agent's ToolManager if running
        if self._agent and hasattr(self._agent, "available_tools"):
            tm: ToolManager = self._agent.available_tools
            if name in tm.tool_map:
                tm.remove_tool(name)
                logger.info(f"Removed tool '{name}' from running agent")

        return True

    def get_available_tools(self) -> list[dict[str, Any]]:
        """
        List all registered tools with their active/inactive status.

        Returns:
            List of dicts with name, description, and active flag.
        """
        return self.tools.get_all_tool_summaries()

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._session.clear()
        self.sessions.save(self._session)
        logger.info("Conversation history cleared")

    def get_history(self) -> list[dict[str, str]]:
        """Get current conversation history."""
        return self._session.get_history()

    def remember(self, fact: str, category: str = "Facts") -> None:
        """Add a fact to long-term memory."""
        if not fact or not fact.strip():
            raise ValueError("Fact cannot be empty")
        self.memory.add_memory(fact.strip(), category)

    def note(self, content: str) -> None:
        """Add a note to today's daily file."""
        if not content or not content.strip():
            raise ValueError("Note content cannot be empty")
        self.memory.add_daily_note(content.strip())

    @property
    def skills(self) -> list[str]:
        """Get available skill names."""
        if self._skill_manager:
            return self._skill_manager.list()
        return []

    @property
    def chatbot(self) -> ChatBot | None:
        """Access the underlying ChatBot."""
        return self._chatbot

    @property
    def agent(self) -> SpoonReactMCP | SpoonReactSkill | None:
        """Access the underlying spoon-core agent."""
        return self._agent


async def create_agent(
    model: str = "claude-sonnet-4-20250514",
    provider: str = "anthropic",
    api_key: str | None = None,
    workspace: Path | str | None = None,
    session_key: str = "default",
    base_url: str | None = None,
    mcp_config: dict[str, dict[str, Any]] | None = None,
    enable_skills: bool = True,
    auto_commit: bool = True,
    enabled_tools: set[str] | None = None,
    tool_profile: str | None = None,
    **kwargs: Any,
) -> AgentLoop:
    """
    Create and initialize an AgentLoop.

    This is the recommended way to create an agent.

    By default only core tools (shell, read_file, write_file, edit_file,
    list_dir) are loaded into the agent.  Use ``enabled_tools`` or
    ``tool_profile`` to override, or call ``agent.add_tool(name)`` after
    creation to dynamically inject additional tools.

    Args:
        model: Model name.
        provider: LLM provider (anthropic, openai, etc.)
        api_key: API key for the provider.
        workspace: Workspace directory path.
        session_key: Session identifier.
        base_url: Custom base URL for the provider.
        mcp_config: MCP server configurations.
        enable_skills: Whether to enable skill system.
        auto_commit: Whether to auto-commit workspace changes after each message.
        enabled_tools: Explicit set of tool names to enable. None = core only.
        tool_profile: Named profile ('core', 'coding', 'web3', 'research', 'full').
        **kwargs: Additional arguments for AgentLoop.

    Returns:
        Initialized AgentLoop instance.

    Example:
        >>> agent = await create_agent()
        >>> response = await agent.process("Hello!")

        >>> # Load all tools
        >>> agent = await create_agent(tool_profile="full")

        >>> # Dynamically add a tool after creation
        >>> agent = await create_agent()
        >>> agent.add_tool("web_search")
    """
    agent = AgentLoop(
        model=model,
        provider=provider,
        api_key=api_key,
        workspace=workspace,
        session_key=session_key,
        base_url=base_url,
        mcp_config=mcp_config,
        enable_skills=enable_skills,
        auto_commit=auto_commit,
        enabled_tools=enabled_tools,
        tool_profile=tool_profile,
        **kwargs,
    )

    await agent.initialize()
    return agent


# Export spoon-core types for convenience
__all__ = [
    "AgentLoop",
    "create_agent",
    # spoon-core types
    "ChatBot",
    "Message",
    "LLMResponse",
    "SpoonReactMCP",
    "SpoonReactSkill",
    "ToolManager",
    "MCPTool",
    "SkillManager",
]
