"""Agent loop: the core processing engine."""

import asyncio
import json
from pathlib import Path
from typing import Any

from loguru import logger

from spoon_bot.agent.context import ContextBuilder
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
from spoon_bot.agent.tools.web import WebSearchTool, WebFetchTool
from spoon_bot.services.spawn import SpawnTool
from spoon_bot.toolkit.adapter import ToolkitAdapter
from spoon_bot.llm.base import LLMProvider
from spoon_bot.llm.spoon_core_provider import SpoonCoreProvider, is_spoon_core_available
from spoon_bot.session.manager import SessionManager
from spoon_bot.memory.store import MemoryStore
from spoon_bot.skills.manager import SkillManager
from spoon_bot.skills.spoon_core_skills import SpoonCoreSkillManager
from spoon_bot.mcp.spoon_core_mcp import SpoonCoreMCPAdapter


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Returns responses

    Native OS tools are registered by default and always available.
    """

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path | str | None = None,
        model: str | None = None,
        max_iterations: int = 20,
        shell_timeout: int = 60,
        max_output: int = 10000,
        session_key: str = "default",
        skill_paths: list[Path | str] | None = None,
        mcp_config: dict | None = None,
        use_spoon_core_skills: bool = True,
    ):
        """
        Initialize the agent loop.

        Args:
            provider: LLM provider for chat completions.
            workspace: Path to workspace directory.
            model: Model to use (defaults to provider's default).
            max_iterations: Maximum tool call iterations (safety limit).
            shell_timeout: Shell command timeout in seconds.
            max_output: Maximum output characters for shell.
            session_key: Session identifier for persistence.
            skill_paths: Additional paths to search for skills.
            mcp_config: MCP server configurations (name -> config dict).
            use_spoon_core_skills: Use spoon-core SkillManager if available.
        """
        self.provider = provider
        self.workspace = Path(workspace or Path.home() / ".spoon-bot" / "workspace")
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.shell_timeout = shell_timeout
        self.max_output = max_output
        self.session_key = session_key

        # Initialize components
        self.context = ContextBuilder(self.workspace)
        self.tools = ToolRegistry()
        self.sessions = SessionManager(self.workspace)
        self.memory = MemoryStore(self.workspace)

        # Initialize skills (prefer spoon-core if available)
        default_skill_paths = [self.workspace / "skills"]
        if skill_paths:
            default_skill_paths.extend([Path(p) for p in skill_paths])

        if use_spoon_core_skills:
            self.skills = SpoonCoreSkillManager(
                skill_paths=default_skill_paths,
                llm=provider if is_spoon_core_available() else None,
                auto_discover=True,
            )
            if self.skills.using_spoon_core:
                logger.info("Using spoon-core SkillManager with LLM intent matching")
        else:
            self.skills = SkillManager(skill_paths=default_skill_paths, auto_discover=True)

        # Initialize MCP adapter
        self.mcp_adapter = SpoonCoreMCPAdapter()
        if mcp_config:
            for name, config in mcp_config.items():
                self.mcp_adapter.add_server(name, config)
            if self.mcp_adapter.uses_spoon_core:
                logger.info(f"Using spoon-core MCP with {len(mcp_config)} servers configured")

        # Get or create session
        self._session = self.sessions.get_or_create(session_key)

        # Inject memory context into system prompt
        memory_context = self.memory.get_memory_context()
        if memory_context:
            self.context.set_memory_context(memory_context)

        self._register_native_tools()
        skill_count = len(self.skills.list())
        logger.info(f"AgentLoop initialized with {len(self.tools)} tools, {skill_count} skills, session: {session_key}")

    def _register_native_tools(self) -> None:
        """Register the default native OS tools."""
        # Shell tool
        self.tools.register(ShellTool(
            timeout=self.shell_timeout,
            max_output=self.max_output,
            working_dir=str(self.workspace),
        ))

        # Filesystem tools
        self.tools.register(ReadFileTool())
        self.tools.register(WriteFileTool())
        self.tools.register(EditFileTool())
        self.tools.register(ListDirTool())

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

        # Web3 tools
        self.tools.register(BalanceCheckTool())
        self.tools.register(TransferTool())
        self.tools.register(SwapTool())
        self.tools.register(ContractCallTool())

        # Toolkit tools (optional, lazy-loaded)
        try:
            toolkit = ToolkitAdapter()
            for tool in toolkit.load_all():
                self.tools.register(tool)
        except ImportError:
            logger.debug("spoon-toolkits not available, skipping toolkit tools")

        logger.debug(f"Registered native tools: {self.tools.list_tools()}")

    async def load_mcp_tools(self) -> int:
        """
        Load tools from configured MCP servers.

        Returns:
            Number of MCP tools loaded.
        """
        mcp_tools = await self.mcp_adapter.load_tools()
        for tool in mcp_tools:
            self.tools.register(tool)
        if mcp_tools:
            logger.info(f"Loaded {len(mcp_tools)} MCP tools")
        return len(mcp_tools)

    def load_skill_tools(self) -> int:
        """
        Load tools from active skills.

        Returns:
            Number of skill tools loaded.
        """
        skill_tools = self.skills.get_active_tools()
        for tool in skill_tools:
            self.tools.register(tool)
        if skill_tools:
            logger.info(f"Loaded {len(skill_tools)} skill tools")
        return len(skill_tools)

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
        logger.info(f"Processing message: {message[:100]}...")

        # Refresh memory context
        memory_context = self.memory.get_memory_context()
        if memory_context:
            self.context.set_memory_context(memory_context)

        # Match and activate relevant skills
        matched_skills = self.skills.match_triggers(message)
        for skill in matched_skills[:2]:  # Activate top 2 matching skills
            try:
                await self.skills.activate(skill.name)
                logger.info(f"Activated skill: {skill.name}")
            except Exception as e:
                logger.warning(f"Failed to activate skill {skill.name}: {e}")

        # Get skill context for injection
        skill_context = self.skills.get_active_context()
        if skill_context:
            self.context.set_skill_context(skill_context)

        # Build messages with session history
        messages = self.context.build_messages(
            history=self._session.get_history(),
            current_message=message,
            media=media,
        )

        # Agent loop
        iteration = 0
        final_content = None

        while iteration < self.max_iterations:
            iteration += 1
            logger.debug(f"Agent iteration {iteration}/{self.max_iterations}")

            # Call LLM
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
            )

            # Handle tool calls
            if response.has_tool_calls:
                # Add assistant message with tool calls
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts
                )

                # Execute tools
                for tool_call in response.tool_calls:
                    logger.info(f"Executing tool: {tool_call.name}")
                    result = await self.tools.execute(
                        tool_call.name,
                        tool_call.arguments,
                    )
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                # No tool calls, we're done
                final_content = response.content
                break

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        # Save to session
        self._session.add_message("user", message)
        self._session.add_message("assistant", final_content)
        self.sessions.save(self._session)

        return final_content

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._session.clear()
        self.sessions.save(self._session)
        logger.info("Conversation history cleared")

    def get_history(self) -> list[dict[str, Any]]:
        """Get current conversation history."""
        return self._session.get_history()

    def remember(self, fact: str, category: str = "Facts") -> None:
        """Add a fact to long-term memory."""
        self.memory.add_memory(fact, category)

    def note(self, content: str) -> None:
        """Add a note to today's daily file."""
        self.memory.add_daily_note(content)


async def create_agent(
    api_key: str | None = None,
    model: str | None = None,
    workspace: Path | str | None = None,
    provider: str = "anthropic",
    session_key: str = "default",
    use_spoon_core: bool = True,
    base_url: str | None = None,
    mcp_config: dict | None = None,
    **kwargs: Any,
) -> AgentLoop:
    """
    Create an agent with the specified configuration.

    Args:
        api_key: API key for the LLM provider.
        model: Model to use.
        workspace: Workspace directory path.
        provider: LLM provider name ("anthropic", "openai", "deepseek", "ollama", "gemini", "openrouter").
        session_key: Session identifier.
        use_spoon_core: Use spoon-core providers if available.
        base_url: Custom base URL for the provider (e.g., for Ollama).
        mcp_config: MCP server configurations.
        **kwargs: Additional arguments for AgentLoop.

    Returns:
        Configured AgentLoop instance.
    """
    # Try spoon-core provider first if available and requested
    if use_spoon_core and is_spoon_core_available():
        logger.info(f"Using spoon-core provider: {provider}")
        llm_provider = SpoonCoreProvider(
            provider=provider,
            api_key=api_key,
            model=model,
            base_url=base_url,
        )
    else:
        # Fallback to local providers
        if provider == "anthropic":
            from spoon_bot.llm.anthropic import AnthropicProvider
            llm_provider = AnthropicProvider(api_key=api_key, model=model)
        elif provider == "openai":
            from spoon_bot.llm.openai import OpenAIProvider
            llm_provider = OpenAIProvider(api_key=api_key, model=model)
        else:
            raise ValueError(f"Unknown provider: {provider}. Install spoon-core for more providers.")

    agent = AgentLoop(
        provider=llm_provider,
        workspace=workspace,
        model=model,
        session_key=session_key,
        mcp_config=mcp_config,
        **kwargs,
    )

    # Load MCP tools if configured
    if mcp_config:
        await agent.load_mcp_tools()

    return agent
