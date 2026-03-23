"""
Agent loop: the core processing engine using spoon-core SDK.

This module provides the main agent interface, integrating spoon-core's
ChatBot, SpoonReactMCP, and SkillManager with spoon-bot's native OS tools.
"""

from __future__ import annotations

import asyncio
import json
import logging as stdlib_logging
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncGenerator

from loguru import logger


class _InterceptHandler(stdlib_logging.Handler):
    """Route stdlib logging into loguru so spoon-core agent logs are visible."""

    def emit(self, record: stdlib_logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = stdlib_logging.currentframe(), 2
        while frame and frame.f_code.co_filename == stdlib_logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


stdlib_logging.basicConfig(handlers=[_InterceptHandler()], level=stdlib_logging.INFO, force=True)
for _noisy in ("httpx", "httpcore", "urllib3", "hpack", "h2", "openai._base_client"):
    stdlib_logging.getLogger(_noisy).setLevel(stdlib_logging.WARNING)

# Import spoon-core SDK components (required)
try:
    from spoon_ai.chat import ChatBot
    from spoon_ai.schema import Message, ToolCall as CoreToolCall, AgentState
    from spoon_ai.llm.interface import LLMResponse
    from spoon_ai.agents.spoon_react_mcp import SpoonReactMCP
    from spoon_ai.agents.spoon_react_skill import SpoonReactSkill
    from spoon_ai.tools import BaseTool, ToolManager
    from spoon_ai.skills import SkillManager

    SPOON_CORE_AVAILABLE = True
except ImportError as e:
    logger.error(f"spoon-core SDK is required: {e}")
    raise ImportError(
        "spoon-bot requires spoon-core SDK. Install with: pip install spoon-ai-sdk"
    ) from e

try:
    from spoon_ai.tools.mcp_tool import MCPTool
    MCP_TOOL_AVAILABLE = True
    _MCP_TOOL_IMPORT_ERROR: Exception | None = None
except ImportError as e:
    MCPTool = None  # type: ignore[assignment]
    MCP_TOOL_AVAILABLE = False
    _MCP_TOOL_IMPORT_ERROR = e
    logger.warning(
        f"MCPTool import failed ({e}). MCP integrations are disabled; AgentLoop remains available."
    )

# Import spoon-bot native tools and components
from spoon_bot.agent.context import ContextBuilder
from spoon_bot.agent.tools.registry import (
    CORE_TOOLS,
    ToolRegistry,
)
from spoon_bot.agent.tools.shell import ShellTool
from spoon_bot.agent.tools.filesystem import (
    ReadFileTool,
    WriteFileTool,
    EditFileTool,
    ListDirTool,
)
from spoon_bot.agent.tools.grep import GrepTool
from spoon_bot.agent.tools.self_config import (
    ActivateToolTool,
    SelfConfigTool,
    MemoryManagementTool,
    SelfUpgradeTool,
)
from spoon_bot.agent.tools.web import WebSearchTool, WebFetchTool
from spoon_bot.config import AgentLoopConfig, MemSearchConfig, validate_agent_loop_params, resolve_context_window
from spoon_bot.services.hotreload import HotReloadService
from spoon_bot.subagent.manager import SubagentManager
from spoon_bot.subagent.tools import SubagentTool
from spoon_bot.subagent.catalog import format_roles_for_prompt
from spoon_bot.subagent.models import SubagentState
from spoon_bot.session.manager import SessionManager
from spoon_bot.session.store import create_session_store
from spoon_bot.memory.store import MemoryStore
from spoon_bot.exceptions import (
    SpoonBotError,
    APIKeyMissingError,
    ContextOverflowError,
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

    # Minimal per-step prompt. Kept short since it's injected every iteration.
    # The agent builds its own reasoning from the system prompt + memory.
    DEFAULT_NEXT_STEP_PROMPT = (
        "Continue working on the user's request. "
        "Do NOT repeat previous actions. Do NOT fabricate output. "
        "Make autonomous choices when input is needed. "
        "When the task is complete, summarize concrete results."
    )

    def __init__(
        self,
        workspace: Path | str | None = None,
        model: str | None = None,
        provider: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_iterations: int = 50,
        shell_timeout: int = 90,
        max_output: int = 10000,
        session_key: str = "default",
        skill_paths: list[Path | str] | None = None,
        mcp_config: dict[str, dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        enable_skills: bool = True,
        auto_commit: bool = True,
        enabled_tools: set[str] | None = None,
        tool_profile: str | None = None,
        session_manager: SessionManager | None = None,
        subagent_manager: SubagentManager | None = None,
        session_store_backend: str = "file",
        session_store_dsn: str | None = None,
        session_store_db_path: str | None = None,
        context_window: int | None = None,
        memsearch_config: MemSearchConfig | dict[str, Any] | None = None,
        auto_reload: bool = False,
        auto_reload_interval: float = 5.0,
        config_path: Path | str | None = None,
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
            session_manager: Existing SessionManager to reuse for persistence.
            subagent_manager: Existing SubagentManager to reuse for child agents.
            session_store_backend: Session storage backend ('file', 'sqlite', 'postgres').
            session_store_dsn: PostgreSQL DSN for 'postgres' backend.
            session_store_db_path: SQLite DB path for 'sqlite' backend.
            context_window: Override context window in tokens (auto-resolved from model if None).
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

        # Store config — callers must provide model/provider explicitly
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

        # Context window — auto-resolved from model when not explicit
        self.context_window = resolve_context_window(model, context_window)
        logger.info(f"Context window: {self.context_window:,} tokens (model={model})")

        # spoon-core components (initialized later)
        self._chatbot: ChatBot | None = None
        self._agent: SpoonReactMCP | SpoonReactSkill | None = None
        self._skill_manager: SkillManager | None = None
        self._mcp_tools: list[MCPTool] = []

        # spoon-bot components
        self.context = ContextBuilder(self.workspace)
        self.tools = ToolRegistry()

        # Session persistence — configurable backend
        if session_manager is not None:
            self.sessions = session_manager
            logger.info("Session store: inherited from parent SessionManager")
        else:
            _store_backend = session_store_backend or "file"
            _store_db_path = session_store_db_path
            if _store_backend == "sqlite" and not _store_db_path:
                _store_db_path = str(self.workspace / "sessions.db")
            try:
                _session_store = create_session_store(
                    backend=_store_backend,
                    workspace=self.workspace,
                    db_path=_store_db_path,
                    dsn=session_store_dsn,
                )
                self.sessions = SessionManager(workspace=self.workspace, store=_session_store)
                logger.info(f"Session store: {_store_backend}")
            except Exception as exc:
                logger.warning(f"Session store '{_store_backend}' init failed ({exc}), falling back to file")
                self.sessions = SessionManager(self.workspace)

        # Memory store — semantic (memsearch) or file-based
        self._memsearch_config: MemSearchConfig | None = None
        if memsearch_config is not None:
            if isinstance(memsearch_config, dict):
                self._memsearch_config = MemSearchConfig(**memsearch_config)
            else:
                self._memsearch_config = memsearch_config

        if self._memsearch_config and self._memsearch_config.enabled:
            try:
                from spoon_bot.memory.semantic_store import SemanticMemoryStore
                self.memory = SemanticMemoryStore(
                    self.workspace,
                    embedding_provider=self._memsearch_config.embedding_provider,
                    embedding_model=self._memsearch_config.get_embedding_model(),
                    embedding_api_key=self._memsearch_config.get_embedding_api_key(),
                    embedding_base_url=self._memsearch_config.get_embedding_base_url(),
                    milvus_uri=self._memsearch_config.milvus_uri,
                    collection=self._memsearch_config.collection,
                )
                logger.info("Using SemanticMemoryStore (memsearch)")
            except ImportError:
                logger.warning("memsearch not installed, falling back to file-based memory")
                self.memory = MemoryStore(self.workspace)
        else:
            self.memory = MemoryStore(self.workspace)

        self._git = GitManager(self.workspace) if auto_commit else None

        # Sub-agent manager — shares this agent's SessionManager so sub-agents
        # can reuse the same persistence backend with their own unique session keys.
        if subagent_manager is not None:
            self._subagent_manager = subagent_manager
            self._owns_subagent_manager = False
        else:
            self._subagent_manager = SubagentManager(
                session_manager=self.sessions,
                workspace=self.workspace,
                max_depth=self._config.subagent.max_depth,
                max_children_per_agent=self._config.subagent.max_children_per_agent,
                max_total_subagents=self._config.subagent.max_total_subagents,
                parent_model=model,
                parent_provider=provider,
                parent_api_key=api_key,
                parent_base_url=base_url,
                persist_runs=self._config.subagent.persist_runs,
                persist_file=self._config.subagent.persist_file,
                archive_after_minutes=self._config.subagent.archive_after_minutes,
                sweeper_interval_seconds=self._config.subagent.sweeper_interval_seconds,
                max_persistent_agents=self._config.subagent.max_persistent_agents,
            )
            self._owns_subagent_manager = True

        # Skill paths: runtime workspace + bundled skills shipped with spoon-bot
        self._skill_paths = [self.workspace / "skills"]
        _bundled_skills = Path(__file__).resolve().parent.parent.parent / "workspace" / "skills"
        if _bundled_skills.is_dir() and _bundled_skills != self._skill_paths[0]:
            self._skill_paths.append(_bundled_skills)
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

        # Hot-reload service
        self._auto_reload = auto_reload
        self._auto_reload_interval = auto_reload_interval
        self._config_path = Path(config_path) if config_path else None
        self._hot_reload: HotReloadService | None = None

        # Track initialization state
        self._initialized = False

        # Skill tool names registered via _register_skill_tools() — for cleanup on reload
        self._skill_tool_names: set[str] = set()

        # Stop flag: set by stop_current_task(), cleared on next process() call
        self._stop_requested = False
        self._last_response_source = self._build_response_source()

        active_count = len(self.tools)
        total_count = len(self.tools._tools)
        logger.info(
            f"AgentLoop created: model={model}, provider={provider}, "
            f"tools={active_count}/{total_count}, session={session_key}"
        )

    @staticmethod
    def _build_response_source(
        *,
        source_type: str = "agent",
        subagent_id: str | None = None,
        subagent_name: str | None = None,
    ) -> dict[str, Any]:
        """Build machine-readable metadata describing who produced a result."""
        return {
            "type": source_type,
            "is_subagent": source_type == "subagent",
            "subagent_id": subagent_id,
            "subagent_name": subagent_name,
        }

    def _set_last_response_source(
        self,
        source: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store and return the latest result source metadata."""
        resolved = dict(source or self._build_response_source())
        self._last_response_source = resolved
        return dict(resolved)

    def get_last_response_source(self) -> dict[str, Any]:
        """Return the latest result source metadata."""
        return dict(getattr(self, "_last_response_source", self._build_response_source()))

    async def initialize(self) -> None:
        """Initialize spoon-core components."""
        if self._initialized:
            return

        if not self.model:
            raise ValueError(
                "model is required but not set. "
                "Configure agent.model in config.yaml or set SPOON_BOT_DEFAULT_MODEL env var."
            )
        if not self.provider:
            raise ValueError(
                "provider is required but not set. "
                "Configure agent.provider in config.yaml or set SPOON_BOT_DEFAULT_PROVIDER env var."
            )

        # Initialize semantic memory if enabled
        if hasattr(self.memory, 'initialize'):
            await self.memory.initialize()

        # Build system prompt (spoon-bot context + available tool summaries)
        system_prompt = self._system_prompt or self.context.build_system_prompt()

        # Context budget hint (compact)
        system_prompt += f"\n\n[Context: {self.context_window:,} tokens — be concise.]\n"

        # Skills section (Openclaw pattern: XML metadata in system prompt)
        skills_xml = self._build_skills_for_prompt()
        if skills_xml:
            system_prompt += f"\n## Installed Skills\n{skills_xml}\n"

        if skills_xml:
            system_prompt += (
                "\n> **FIRST ACTION**: Match the user's request to a skill above "
                "by name/description, then `read_file` its `<location>` path. "
                "Do NOT call skill_marketplace, web_search, or filesystem search first.\n"
            )

        system_prompt += (
            "\n## Workflow\n"
            f"You have up to {self.max_iterations} steps. Minimize steps.\n\n"
            "1. **Read SKILL.md**: If a skill above matches, `read_file` its path. "
            "Extract config, addresses, commands.\n"
            "2. **Execute**: Run commands from SKILL.md directly via shell "
            "(`cast`, `curl`, etc.). Do NOT write script files.\n"
            "3. **Done**: Summarize result. Save key state to `soul.md`.\n\n"
            "Only use `web_search` if NO installed skill matches the task.\n\n"
            "### Rules\n"
            "- Do NOT re-read files already in context.\n"
            "- `source .env.local` before commands that need env vars.\n"
            "- If a command fails, analyze the error and retry with fixes.\n"
            "- Follow user instructions exactly — respect specific IDs, names, actions.\n"
            "\n## Agent Memory (soul.md)\n"
            "After completing significant actions, append a timestamped entry to `soul.md`.\n"
            "Format: `## YYYY-MM-DD HH:MM — <topic>` followed by bullet points.\n"
        )

        # Inject soul.md content if it exists
        soul_path = self.workspace / "soul.md"
        if soul_path.exists():
            try:
                soul_content = soul_path.read_text(encoding="utf-8").strip()
                if soul_content:
                    _max_soul = 2000
                    if len(soul_content) > _max_soul:
                        soul_content = soul_content[-_max_soul:]
                    system_prompt += f"\n## Agent Memory (from soul.md)\n{soul_content}\n"
            except Exception:
                pass

        # ----------------------------------------------------------------
        # Multi-Agent Orchestration section
        # Injected when the SubagentTool (spawn) is available.
        # Mirrors opencode task.txt's usage notes and openclaw's
        # sessions-spawn-tool description pattern.
        # ----------------------------------------------------------------
        if "spawn" in self.tools.get_active_tools():
            roles_block = format_roles_for_prompt()
            system_prompt += (
                "\n\n## Multi-Agent Orchestration\n\n"
                "You are an **Orchestrator**. For complex tasks (e.g. 'build a user "
                "management system', 'implement a REST API with frontend'), decompose "
                "the work and delegate to specialised sub-agents using the `spawn` tool.\n\n"
                "**You decide which agents to use and in what order** — there is no fixed "
                "pipeline. Use your judgement based on the task requirements.\n\n"
                "### Available specialised agent roles\n"
                f"{roles_block}\n\n"
                "### When to use sub-agents\n"
                "- Complex tasks requiring multiple specialised skills\n"
                "- Tasks that benefit from sequential specialisation "
                "(e.g. plan → implement → review)\n"
                "- Research + implementation tasks that can be parallelised\n\n"
                "### When NOT to use sub-agents\n"
                "- Simple, single-step tasks (file reads, quick questions, small edits)\n"
                "- Tasks you can complete directly in a few tool calls\n\n"
                "### Orchestration workflow (decide autonomously)\n"
                "1. For complex tasks, start with `spawn(action='spawn', role='planner', "
                "task='Analyse requirements for: <user request>')` to get a technical plan\n"
                "2. Wait for the planner: `spawn(action='wait', timeout=120)`\n"
                "3. Based on the plan, spawn implementation agents as needed:\n"
                "   - `spawn(action='spawn', role='backend', task='...<plan context>...')`\n"
                "   - `spawn(action='spawn', role='frontend', task='...<plan context>...')`\n"
                "   - Launch independent agents in parallel (multiple tool calls at once)\n"
                "4. Wait for results: `spawn(action='wait', timeout=180)`\n"
                "5. Optionally spawn a `reviewer` to check the implementation\n"
                "6. Summarise all results to the user\n\n"
                "### Key rules (from opencode task.txt)\n"
                "- Each sub-agent starts with a **fresh context** — provide all necessary "
                "context in the `task` parameter\n"
                "- The sub-agent's output is NOT visible to the user until you summarise it\n"
                "- Use `task_id` from a result to resume the same sub-agent session later\n"
                "- Use `spawn(action='list_roles')` to see all available roles at runtime\n"
            )
            logger.info("Injected Multi-Agent Orchestration section into system prompt")

        # NOTE: inactive-tools prompt is built later, after skill tools are
        # registered so that skill-provided tools are included in the list.

        # Create ChatBot (spoon-core LLM interface)
        self._chatbot = ChatBot(
            model_name=self.model,
            llm_provider=self.provider,
            api_key=self.api_key,
            base_url=self.base_url,
            system_prompt=system_prompt,
        )

        # Warn about potential provider/model mismatch that may cause fallback noise
        if self.provider and self.model:
            model_lower = self.model.lower()
            provider_lower = self.provider.lower()
            _provider_model_hints = {
                "anthropic": ["claude"],
                "openai": ["gpt", "o1", "o3", "o4"],
                "deepseek": ["deepseek"],
                "gemini": ["gemini"],
                "openrouter": [],  # openrouter supports all models
            }
            expected_prefixes = _provider_model_hints.get(provider_lower, [])
            if expected_prefixes:
                model_base = model_lower.rsplit("/", 1)[-1]
                if not any(model_base.startswith(p) for p in expected_prefixes):
                    logger.warning(
                        f"Model '{self.model}' may not be native to provider "
                        f"'{self.provider}'. If you see fallback errors in logs, "
                        f"consider using 'openrouter' as provider or matching "
                        f"the model to the provider."
                    )

        # Create MCP tools – expand each server into individual tools (#5)
        await self._init_mcp_tools()

        # Create SkillManager if enabled — BEFORE building active_tools
        # so that skill tools are included in the agent's ToolManager.
        if self._enable_skills:
            import inspect
            _sm_sig = inspect.signature(SkillManager.__init__)
            _sm_kwargs: dict[str, Any] = {
                "skill_paths": [str(p) for p in self._skill_paths],
                "llm": self._chatbot,
                "auto_discover": True,
            }
            if "include_default_paths" in _sm_sig.parameters:
                _sm_kwargs["include_default_paths"] = False
            self._skill_manager = SkillManager(**_sm_kwargs)
            # Load skill tools into the ToolRegistry (inactive by default)
            self._register_skill_tools()
            # Only auto-activate skill tools when no skills are shown in prompt
            # (prevents agent from calling skill_marketplace instead of reading SKILL.md)
            if not self._build_skills_for_prompt():
                for skill_tool_name in self._skill_tool_names:
                    self.tools.activate_tool(skill_tool_name)

        # Build active tools list AFTER skill tools are registered and activated
        active_tools = list(self.tools.get_active_tools().values()) + self._mcp_tools

        # Common agent kwargs
        agent_kwargs: dict[str, Any] = {
            "llm": self._chatbot,
            "tools": ToolManager(active_tools) if active_tools else None,
            "max_steps": self.max_iterations,
            "system_prompt": system_prompt,
            "next_step_prompt": self.DEFAULT_NEXT_STEP_PROMPT,
        }

        if self._enable_skills:
            agent_kwargs["skill_manager"] = self._skill_manager
            self._agent = SpoonReactSkill(**agent_kwargs)
        else:
            self._agent = SpoonReactMCP(**agent_kwargs)

        # Force-set the next_step_prompt so _refresh_prompts() won't overwrite
        # it with spoon-core's default template (which duplicates tool lists
        # and can cause the model to summarize instead of continue).
        self._agent.next_step_prompt = self.DEFAULT_NEXT_STEP_PROMPT
        self._agent._custom_next_step_prompt = True

        # Build dynamic-tools prompt now that all tools (native + skill) are registered
        inactive_tools = self.tools.get_inactive_tools()
        if inactive_tools:
            system_prompt += self._build_dynamic_tools_prompt(inactive_tools)
            # Update the agent's system prompt with the full list
            if hasattr(self._agent, "system_prompt"):
                self._agent.system_prompt = system_prompt

        # Initialize agent
        await self._agent.initialize()

        # Increase default step timeout — on-chain txs (cast send) can take 60s+
        self._agent._default_timeout = 300.0

        self._initialized = True
        active_count = len(self.tools.get_active_tools())
        total_count = len(self.tools._tools)
        logger.info(
            f"AgentLoop initialized: tools={active_count}/{total_count}, "
            f"mcp_servers={len(self._mcp_tools)}, "
            f"skills_enabled={self._enable_skills}"
        )

        # Start sub-agent archive sweeper (no-op if persistence is disabled)
        await self._subagent_manager.start_sweeper()

        # Start background hot-reload service if enabled
        if self._auto_reload:
            self._hot_reload = HotReloadService(
                agent=self,
                poll_interval=self._auto_reload_interval,
                watch_skills=self._enable_skills,
                watch_config=self._config_path is not None,
                config_path=self._config_path,
            )
            await self._hot_reload.start()

    def _register_native_tools(self) -> None:
        """Register the default native OS tools."""
        # Shell tool — allow_chaining + allow_substitution lets the agent
        # compose multi-step ops and use $(), ${}, and backtick expressions.
        self.tools.register(ShellTool(
            timeout=self.shell_timeout,
            max_output=self.max_output,
            working_dir=str(self.workspace),
            allow_chaining=True,
            allow_substitution=True,
        ))

        # Filesystem tools — allow reads from the user home directory so that
        # skill-managed data (e.g. ~/.agent-wallet, ~/.spoon-bot/skills) is
        # accessible.  The PathValidator blocklist still blocks truly sensitive
        # paths (.ssh, .aws, etc.).
        _extra_read = [Path.home()]
        self.tools.register(ReadFileTool(workspace=self.workspace, additional_read_paths=_extra_read, max_output=15000))
        self.tools.register(WriteFileTool(workspace=self.workspace))
        self.tools.register(EditFileTool(workspace=self.workspace))
        self.tools.register(ListDirTool(workspace=self.workspace, additional_read_paths=_extra_read))
        self.tools.register(GrepTool(workspace=self.workspace))

        # Self-management tools
        self.tools.register(SelfConfigTool())
        memory_tool = MemoryManagementTool()
        memory_tool.set_memory_store(self.memory)
        self.tools.register(memory_tool)

        # Self-upgrade tool — with agent loop reference for hot-reload
        upgrade_tool = SelfUpgradeTool(workspace=self.workspace)
        upgrade_tool.set_agent_loop(self)
        self.tools.register(upgrade_tool)

        # Dynamic tool activation — lets the LLM load tools on demand
        self.tools.register(ActivateToolTool(
            activate_fn=self.add_tool,
            list_inactive_fn=lambda: [
                {"name": t.name, "description": t.description}
                for t in self.tools.get_inactive_tools().values()
            ],
        ))

        # Sub-agent tool — replaces the old placeholder SpawnTool
        spawn_tool = SubagentTool(manager=getattr(self, "_subagent_manager", None))
        self.tools.register(spawn_tool)

        # Web tools
        self.tools.register(WebSearchTool())
        self.tools.register(WebFetchTool())

        logger.debug(f"Registered native tools: {self.tools.list_tools()}")

    def _register_skill_tools(self) -> None:
        """Load tools from discovered skill directories and register as inactive.

        Scans ``self._skill_paths`` for skill directories containing ``tools.py``,
        loads their ``BaseTool`` subclasses via the SkillManager's loader, wraps
        each in a :class:`SkillToolBridge`, and registers them in the ToolRegistry
        as **inactive** tools.  This makes them visible via ``activate_tool list``
        so the agent can dynamically load them on demand.
        """
        from spoon_bot.agent.tools.skill_bridge import SkillToolBridge

        # Unregister previously registered skill tools (handles reload)
        for name in list(self._skill_tool_names):
            self.tools.unregister(name)
        self._skill_tool_names.clear()

        if not self._skill_manager:
            return

        loader = getattr(self._skill_manager, "_loader", None)
        if loader is None:
            return

        for skill_path in self._skill_paths:
            if not skill_path.is_dir():
                continue
            for skill_dir in sorted(skill_path.iterdir()):
                if not skill_dir.is_dir():
                    continue
                if not (skill_dir / "tools.py").exists():
                    continue
                try:
                    tools = loader.load_tools(skill_dir)
                    for base_tool in tools:
                        if base_tool.name not in self.tools:
                            bridge = SkillToolBridge(base_tool)
                            self.tools.register(bridge)
                            self._skill_tool_names.add(base_tool.name)
                except Exception as exc:
                    logger.debug(f"Skill tools from {skill_dir.name}: {exc}")

        if self._skill_tool_names:
            logger.info(
                f"Registered {len(self._skill_tool_names)} skill tool(s): "
                f"{sorted(self._skill_tool_names)}"
            )

    async def _init_mcp_tools(self) -> None:
        """Discover and create MCP tools from ``self._mcp_config``.

        Populates ``self._mcp_tools``.  Safe to call multiple times — the
        caller is responsible for cleaning up old tools first.
        """
        if self._mcp_config and not MCP_TOOL_AVAILABLE:
            logger.warning(
                "MCP configuration provided but MCPTool is unavailable "
                f"({_MCP_TOOL_IMPORT_ERROR}); skipping MCP server setup."
            )
        for name, config in self._mcp_config.items():
            if not MCP_TOOL_AVAILABLE:
                break

            mcp_tool = MCPTool(
                name=name,
                description=f"MCP server: {name}",
                mcp_config=config,
            )
            # Try to discover real server tools and create one MCPTool per tool
            if hasattr(mcp_tool, "expand_server_tools"):
                try:
                    expanded = await mcp_tool.expand_server_tools()
                    if expanded:
                        self._mcp_tools.extend(expanded)
                        logger.info(f"MCP server '{name}': expanded to {len(expanded)} tools")
                    else:
                        self._mcp_tools.append(mcp_tool)
                        logger.warning(f"MCP server '{name}': no tools discovered, keeping proxy")
                except Exception as exc:
                    logger.warning(f"MCP server '{name}': expansion failed ({exc}), keeping proxy")
                    self._mcp_tools.append(mcp_tool)
            else:
                self._mcp_tools.append(mcp_tool)
                logger.info(
                    f"MCP server '{name}': expand_server_tools() unavailable in this spoon-core version; using proxy."
                )

    # ------------------------------------------------------------------
    # Hot-reload: skills / MCP / all
    # ------------------------------------------------------------------

    async def reload_skills(self) -> dict[str, Any]:
        """Re-discover skills from skill paths and swap into the running agent.

        Returns:
            Summary dict with ``before``, ``after``, ``added``, ``removed``.
        """
        old_skills: list[str] = []
        if self._skill_manager:
            try:
                old_skills = list(self._skill_manager.list())
            except Exception:
                pass

        if not self._enable_skills:
            return {"before": old_skills, "after": old_skills, "added": [], "removed": []}

        # Create a fresh SkillManager with the same paths
        import inspect
        _sm_sig = inspect.signature(SkillManager.__init__)
        _sm_kwargs: dict[str, Any] = {
            "skill_paths": [str(p) for p in self._skill_paths],
            "llm": self._chatbot,
            "auto_discover": True,
        }
        if "include_default_paths" in _sm_sig.parameters:
            _sm_kwargs["include_default_paths"] = False

        new_sm = SkillManager(**_sm_kwargs)
        self._skill_manager = new_sm

        # Remember old skill tool names before re-registering
        old_skill_tool_names = set(self._skill_tool_names)

        # Re-register skill tools from the new SkillManager
        self._register_skill_tools()
        # Auto-activate skill tools so the agent can use them immediately
        for skill_tool_name in self._skill_tool_names:
            self.tools.activate_tool(skill_tool_name)

        # Inject into the running agent
        if self._agent and isinstance(self._agent, SpoonReactSkill):
            # Try common attribute names across spoon-core versions
            for attr in ("skill_manager", "_skill_manager"):
                if hasattr(self._agent, attr):
                    setattr(self._agent, attr, new_sm)
                    break

        # Sync skill tools with the running agent's ToolManager
        # (similar to how reload_mcp() syncs MCP tools)
        if self._agent and hasattr(self._agent, "available_tools"):
            tm: ToolManager = self._agent.available_tools
            # Remove old skill tools that are no longer registered
            for old_name in old_skill_tool_names - self._skill_tool_names:
                if old_name in tm.tool_map:
                    tm.remove_tool(old_name)
            # Add newly registered skill tools
            for skill_name in self._skill_tool_names:
                tool = self.tools.get(skill_name)
                if tool and skill_name not in tm.tool_map:
                    tm.add_tool(tool)
                    logger.info(f"Added skill tool '{skill_name}' to running agent")

        new_skills: list[str] = []
        try:
            new_skills = list(new_sm.list())
        except Exception:
            pass

        added = [s for s in new_skills if s not in old_skills]
        removed = [s for s in old_skills if s not in new_skills]
        logger.info(f"Skills reloaded: {len(old_skills)} -> {len(new_skills)} (added={added}, removed={removed})")
        return {"before": old_skills, "after": new_skills, "added": added, "removed": removed}

    async def reload_mcp(self, new_config: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
        """Shutdown existing MCP tools and re-initialize from config.

        Args:
            new_config: Optional new MCP config dict.  If ``None`` the
                existing ``self._mcp_config`` is reused.

        Returns:
            Summary dict with ``before`` and ``after`` server names.
        """
        old_names = [getattr(t, "name", "?") for t in self._mcp_tools]

        # Cleanup existing MCP tools
        for mcp_tool in self._mcp_tools:
            for method in ("close", "cleanup", "shutdown"):
                fn = getattr(mcp_tool, method, None)
                if fn and callable(fn):
                    try:
                        result = fn()
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as exc:
                        logger.debug(f"MCP cleanup ({method}) for '{getattr(mcp_tool, 'name', '?')}': {exc}")
                    break

        # Remove old MCP tools from the agent's ToolManager
        if self._agent and hasattr(self._agent, "available_tools"):
            tm: ToolManager = self._agent.available_tools
            for mcp_tool in self._mcp_tools:
                name = getattr(mcp_tool, "name", None)
                if name and name in tm.tool_map:
                    tm.remove_tool(name)

        self._mcp_tools.clear()

        # Apply new config if provided
        if new_config is not None:
            self._mcp_config = new_config

        # Re-init MCP tools
        await self._init_mcp_tools()

        # Inject new tools into the running agent
        if self._agent and hasattr(self._agent, "available_tools"):
            tm = self._agent.available_tools
            for mcp_tool in self._mcp_tools:
                name = getattr(mcp_tool, "name", None)
                if name and name not in tm.tool_map:
                    tm.add_tool(mcp_tool)

        new_names = [getattr(t, "name", "?") for t in self._mcp_tools]
        logger.info(f"MCP reloaded: {old_names} -> {new_names}")
        return {"before": old_names, "after": new_names}

    async def reload_all(self) -> dict[str, Any]:
        """Reload both skills and MCP servers.

        Returns:
            Combined summary with ``skills`` and ``mcp`` keys.
        """
        skills_result = await self.reload_skills()
        mcp_result = await self.reload_mcp()
        return {"skills": skills_result, "mcp": mcp_result}

    async def cleanup(self) -> None:
        """Shut down all managed resources (MCP tools, skills, sub-agents, etc.)."""
        # Cleanup sub-agents first so they can still access tools during their own cleanup
        if hasattr(self, "_subagent_manager") and getattr(self, "_owns_subagent_manager", True):
            try:
                await self._subagent_manager.cleanup()
            except Exception as exc:
                logger.debug(f"Sub-agent manager cleanup: {exc}")

        # Cleanup MCP tools
        for mcp_tool in self._mcp_tools:
            for method in ("close", "cleanup", "shutdown"):
                fn = getattr(mcp_tool, method, None)
                if fn and callable(fn):
                    try:
                        result = fn()
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as exc:
                        logger.debug(f"MCP cleanup ({method}): {exc}")
                    break
        self._mcp_tools.clear()

        # Deactivate skills
        if self._skill_manager:
            try:
                result = self._skill_manager.deactivate_all()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                logger.debug(f"Skill deactivate_all: {exc}")

        self._initialized = False
        logger.info("AgentLoop cleanup complete")

    def _estimate_token_count(self) -> int:
        """Rough token estimate for current session messages (~4 chars per token)."""
        return sum(len(m.get("content", "")) for m in self._session.messages) // 4

    def _trim_context_if_needed(self) -> int:
        """Auto-trim oldest messages if context budget approaches the limit.

        Keeps at minimum the two most recent messages (last user + assistant pair).
        Raises ContextOverflowError if even after trimming the context is too large.
        """
        if not self._session.messages:
            return 0

        budget = int(self.context_window * 0.75)
        estimated = self._estimate_token_count()

        if estimated <= budget:
            return 0

        logger.warning(
            f"Context approaching limit: ~{estimated:,} tokens "
            f"(budget: {budget:,}, window: {self.context_window:,}). "
            f"Trimming oldest messages."
        )

        messages = self._session.messages
        trimmed_count = 0
        while len(messages) > 2 and self._estimate_token_count() > budget:
            removed = messages.pop(0)
            trimmed_count += 1
            logger.debug(
                f"Trimmed message: role={removed.get('role')}, "
                f"len={len(removed.get('content', ''))}"
            )

        # If still over the hard limit after trimming, raise
        final_estimate = self._estimate_token_count()
        if final_estimate > self.context_window:
            raise ContextOverflowError(final_estimate, self.context_window)

        return trimmed_count

    async def _sync_runtime_history_from_session(self) -> int:
        """Sync persisted session history into spoon-core runtime memory."""
        if not self._agent:
            return 0

        memory = getattr(self._agent, "memory", None)
        if memory is None or not hasattr(memory, "clear"):
            return 0

        try:
            memory.clear()
        except Exception as exc:
            logger.warning(f"Failed to clear runtime memory before history sync: {exc}")
            return 0

        injected_count = 0
        for msg in self._session.get_history():
            role = str(msg.get("role", "")).strip().lower()
            if role not in {"user", "assistant", "tool"}:
                continue

            content = msg.get("content", "")
            if not isinstance(content, str):
                try:
                    content = json.dumps(content, ensure_ascii=False)
                except Exception:
                    content = str(content)

            try:
                await self._agent.add_message(role, content)
                injected_count += 1
            except Exception as exc:
                logger.warning(
                    f"Failed to inject session history message "
                    f"(role={role}, index={injected_count}): {exc}"
                )

        return injected_count

    async def _prepare_request_context(self) -> None:
        """Prepare request context by trimming and injecting session history."""
        trimmed_count = self._trim_context_if_needed()
        injected_count = await self._sync_runtime_history_from_session()
        estimated_tokens = self._estimate_token_count()

        logger.info(
            f"Session context prepared: session={self.session_key}, "
            f"injected_messages={injected_count}, "
            f"estimated_tokens~{estimated_tokens}, "
            f"trimmed_messages={trimmed_count}"
        )

    def set_subagent_context(
        self,
        *,
        session_key: str | None = None,
        channel: str | None = None,
    ) -> None:
        """Bind the spawn tool to the current requester session/channel."""
        spawn_tool = self.tools.get("spawn")
        if spawn_tool and isinstance(spawn_tool, SubagentTool):
            spawn_tool.set_spawner_context(
                session_key=session_key or self.session_key,
                channel=channel,
            )

    def _persist_turn(self, user_message: str, assistant_message: str) -> None:
        """Save a completed user/assistant turn to session storage."""
        try:
            self._session.add_message("user", user_message)
            self._session.add_message("assistant", assistant_message)
            self.sessions.save(self._session)
        except Exception as e:
            logger.warning(f"Failed to save session: {e}")

    def _should_skip_specialist_auto_route(self, message: str) -> bool:
        """Return True when the current message should not be auto-routed."""
        stripped = message.lstrip()
        return (
            self.session_key.startswith("subagent-")
            or stripped.startswith("[Sub-agent Completed]")
        )

    def _maybe_create_persistent_subagent_from_request(
        self,
        message: str,
    ) -> str | None:
        """Create a persistent subagent from a natural-language user request."""
        if self._should_skip_specialist_auto_route(message):
            return None
        parsed = self._subagent_manager.parse_persistent_subagent_request(message)
        if parsed is None:
            return None

        profile = self._subagent_manager.create_persistent_subagent(
            description=parsed["specialization"],
        )
        keywords = ", ".join(profile.match_keywords[:8]) or "(none)"
        response = (
            f"Persistent subagent `{profile.name}` created.\n"
            f"Specialization: {profile.specialization}\n"
            f"Auto-route: {'ON' if profile.auto_route else 'OFF'}\n"
            f"Tool profile: {profile.tool_profile}\n"
            f"Keywords: {keywords}\n\n"
            "Future matching requests will be routed to this subagent automatically."
        )
        return response

    async def _maybe_route_to_persistent_specialist_result(
        self,
        message: str,
    ) -> tuple[str, str | None, dict[str, Any]] | None:
        """Attempt to route a matching top-level request to a persistent subagent."""
        if self._should_skip_specialist_auto_route(message):
            return None

        matched = self._subagent_manager.find_best_auto_route_specialist(message)
        if matched is None:
            return None

        agent_name = matched["agent_name"]
        reason_text = ", ".join(matched["reasons"]) or "high-confidence subagent match"
        logger.info(
            f"Auto-routing request to persistent subagent {agent_name!r} "
            f"(score={matched['score']}; {reason_text})"
        )

        current_channel = getattr(self._subagent_manager, "_current_spawner_channel", None)
        current_metadata = getattr(self._subagent_manager, "_current_spawner_metadata", {})
        current_reply_to = getattr(self._subagent_manager, "_current_spawner_reply_to", None)
        record = await self._subagent_manager.dispatch_persistent_subagent(
            agent_name=agent_name,
            task=message,
            spawner_session_key=self.session_key,
            spawner_channel=current_channel,
            spawner_metadata=current_metadata,
            spawner_reply_to=current_reply_to,
        )
        wait_timeout = float(record.config.timeout_seconds or 300)
        results = await self._subagent_manager.collect_results(
            timeout=wait_timeout,
            spawner_session_key=self.session_key,
            agent_id=record.agent_id,
        )
        if not results:
            response = (
                f"Auto-routed to subagent `{agent_name}`, but it did not finish "
                f"within {int(wait_timeout)}s."
            )
            note = (
                f"Auto-routed to persistent subagent {agent_name} "
                f"(score={matched['score']}; {reason_text})."
            )
            return (
                response,
                note,
                self._build_response_source(
                    source_type="subagent",
                    subagent_id=record.agent_id,
                    subagent_name=agent_name,
                ),
            )

        result = results[-1]
        source = self._build_response_source(
            source_type="subagent",
            subagent_id=result.agent_id or record.agent_id,
            subagent_name=agent_name,
        )
        if result.state == SubagentState.COMPLETED:
            result_text = self._filter_execution_steps(result.result or "(no output)")
            response = f"Subagent `{agent_name}` handled this request.\n\n{result_text}"
        elif result.state == SubagentState.FAILED:
            response = (
                f"Subagent `{agent_name}` failed while handling this request.\n\n"
                f"{result.error or '(no error message)'}"
            )
        elif result.state == SubagentState.CANCELLED:
            response = f"Subagent `{agent_name}` was cancelled before finishing."
        else:
            response = result.result or result.error or f"Subagent `{agent_name}` returned no output."

        note = (
            f"Auto-routed to persistent subagent {agent_name} "
            f"(score={matched['score']}; {reason_text})."
        )
        return response, note, source

    async def _maybe_route_to_persistent_specialist(
        self,
        message: str,
    ) -> tuple[str, str | None] | None:
        """Backward-compatible wrapper without source metadata."""
        routed = await self._maybe_route_to_persistent_specialist_result(message)
        if routed is None:
            return None
        response, note, _source = routed
        return response, note

    async def process(
        self,
        message: str,
        media: list[str] | None = None,
        attachments: list[dict[str, Any]] | None = None,
        session_key: str | None = None,
    ) -> str:
        """
        Process a user message and return the agent's response.

        Args:
            message: The user's message.
            media: Optional list of media file paths.
            session_key: Optional session key for multi-user/multi-channel isolation.
                         When provided, switches to this session before processing.

        Returns:
            The agent's response text.
        """
        AgentLoop._set_last_response_source(self)

        # Switch session if requested
        if session_key and session_key != self.session_key:
            self.session_key = session_key
            self._session = self.sessions.get_or_create(session_key)
            logger.info(f"Switched to session: {session_key}")

        # Honour stop request from previous /stop command
        if self._stop_requested:
            self._stop_requested = False
            logger.info("Task skipped due to stop request")
            return "Task stopped."

        # Ensure initialized
        if not self._initialized:
            await self.initialize()

        # Switch session if a different key is requested
        if session_key and session_key != self.session_key:
            self._session = self.sessions.get_or_create(session_key)
            self.session_key = session_key
            logger.debug(f"Switched to session: {session_key}")

        current_session_key = getattr(self, "session_key", "default")
        self.set_subagent_context(session_key=current_session_key)

        logger.info(f"Processing message: {message[:100]}...")

        created = self._maybe_create_persistent_subagent_from_request(message)
        if isinstance(created, str):
            self._persist_turn(message, created)
            return created

        routed = await self._maybe_route_to_persistent_specialist_result(message)
        if isinstance(routed, tuple) and len(routed) == 3:
            routed_content, _route_note, route_source = routed
            AgentLoop._set_last_response_source(self, route_source)
            self._persist_turn(message, routed_content)
            if self._auto_commit and self._git:
                try:
                    if self._git.has_changes():
                        self._git.commit(message)
                except Exception as e:
                    logger.warning(f"Failed to auto-commit: {e}")
            return routed_content

        # Refresh memory context
        try:
            memory_context = self.memory.get_memory_context()
            if memory_context:
                self.context.set_memory_context(memory_context)
        except Exception as e:
            logger.warning(f"Failed to load memory context: {e}")

        # Trim and inject persisted history into runtime memory
        await self._prepare_request_context()

        # Defensive: reset agent state if stuck from a previous run
        if hasattr(self._agent, 'state') and self._agent.state != AgentState.IDLE:
            logger.warning(
                f"Agent {self._agent.name} was in {self._agent.state} state, "
                f"resetting to IDLE before processing"
            )
            self._agent.state = AgentState.IDLE
            self._agent.current_step = 0

        # Clear shutdown flag so the agent loop can run
        if hasattr(self._agent, '_shutdown_event') and self._agent._shutdown_event.is_set():
            logger.info("Clearing previous shutdown signal before processing")
            self._agent._shutdown_event.clear()

        # Pre-inject matched skill content into the message
        message = self._pre_inject_matched_skill(message)

        # Build a minimal per-step prompt with the user's request for context.
        # The anti-loop tracker will dynamically append progress info.
        _base_prompt = self._build_step_prompt(message)
        self._agent.next_step_prompt = _base_prompt

        # Install anti-loop tracker to prevent repeated tool calls
        self._install_anti_loop_tracker(_base_prompt)

        # Run agent — with recovery for LLM API errors (context overflow, etc.)
        # Retry up to 2 times on ANY error, with increasingly aggressive compression.
        _max_retries = 2
        try:
            for _attempt in range(_max_retries + 1):
                try:
                    run_kwargs: dict[str, Any] = {}
                    if media:
                        run_kwargs["media"] = media
                    result = await self._agent.run(message, **run_kwargs)

                    logger.debug(f"Agent result type: {type(result)}")
                    if hasattr(result, 'content'):
                        logger.info(f"Agent result.content (first 300): {str(result.content)[:300]}")

                    # Extract content — guard against result.content being None
                    if hasattr(result, "content") and result.content is not None:
                        final_content = result.content
                    elif hasattr(result, "content"):
                        final_content = str(result) if str(result) != "None" else ""
                    else:
                        final_content = str(result)

                    # Fallback: when toolcall.run() returns "No results"
                    if final_content.strip() in ("No results", ""):
                        logger.warning(
                            "Agent returned empty/no-results — attempting to extract "
                            "content from agent memory"
                        )
                        _extracted = self._extract_last_assistant_content()
                        if _extracted:
                            final_content = _extracted

                    # Filter out technical execution steps
                    final_content = self._filter_execution_steps(final_content)
                    break

                except Exception as e:
                    # Log the FULL error (before cli.py sanitizes it)
                    import traceback
                    logger.error(
                        f"Agent run error (attempt {_attempt + 1}/{_max_retries + 1}): "
                        f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                    )
                    if _attempt < _max_retries:
                        logger.warning("Compressing context and retrying...")
                        # First retry: normal compression. Second: force-compress.
                        if _attempt == 0:
                            compressed = self._compress_runtime_context()
                            if compressed == 0:
                                self._force_compress_runtime_context()
                        else:
                            self._force_compress_runtime_context()
                        # Reset agent state so it can re-enter run()
                        if hasattr(self._agent, 'state'):
                            self._agent.state = AgentState.IDLE
                            self._agent.current_step = 0
                        if hasattr(self._agent, '_shutdown_event'):
                            self._agent._shutdown_event.clear()
                        continue
                    raise
        finally:
            # Always ensure agent is back in IDLE state after processing
            if hasattr(self._agent, 'state') and self._agent.state != AgentState.IDLE:
                logger.warning(
                    f"Post-run cleanup: resetting agent from {self._agent.state} to IDLE"
                )
                self._agent.state = AgentState.IDLE
                self._agent.current_step = 0

        # Save to session
        self._persist_turn(message, final_content)
        AgentLoop._set_last_response_source(self)

        # Auto-commit workspace changes
        if self._auto_commit and self._git:
            try:
                if self._git.has_changes():
                    self._git.commit(message)
            except Exception as e:
                logger.warning(f"Failed to auto-commit: {e}")

        return final_content

    def _extract_last_assistant_content(self) -> str:
        """Extract the last assistant message content from the agent's memory.

        Used as a fallback when toolcall.run() returns "No results" but the
        LLM actually produced meaningful content stored in memory.
        """
        try:
            if not hasattr(self._agent, "memory"):
                return ""
            messages = (
                self._agent.memory.get_messages()
                if hasattr(self._agent.memory, "get_messages")
                else []
            )
            # Walk backwards to find the last assistant message with content
            for msg in reversed(messages):
                role = getattr(msg, "role", None)
                # role may be an enum (Role.ASSISTANT) or a string
                role_str = role.value if hasattr(role, "value") else str(role)
                if role_str != "assistant":
                    continue
                # Prefer .text_content (handles multimodal), fall back to .content
                text = getattr(msg, "text_content", None) or getattr(msg, "content", None)
                if text and isinstance(text, str) and text.strip():
                    # Skip internal sentinel messages
                    if text.strip() in (
                        "Task completed",
                        "Task completed based on finish_reason signal",
                        "Thinking completed. No action needed. Task finished.",
                    ):
                        continue
                    logger.info(
                        f"Extracted fallback content from memory (len={len(text)})"
                    )
                    return text.strip()
        except Exception as exc:
            logger.warning(f"Failed to extract content from agent memory: {exc}")
        return ""

    @staticmethod
    def _msg_char_count(msg) -> int:
        """Return total character count of a message's content."""
        if isinstance(msg.content, str):
            return len(msg.content)
        if isinstance(msg.content, list):
            return sum(len(getattr(b, 'text', '')) for b in msg.content)
        return 0

    def _estimate_runtime_tokens(self) -> int:
        """Rough token estimate from the agent's runtime messages (~4 chars/token)."""
        if not self._agent or not hasattr(self._agent, 'memory'):
            return 0
        return sum(self._msg_char_count(m) for m in self._agent.memory.messages) // 4

    def _is_next_step_user_msg(self, msg) -> bool:
        """True when *msg* looks like an injected next_step_prompt (not a real user message)."""
        role = getattr(msg, 'role', None)
        if hasattr(role, 'value'):
            role = role.value
        if role != 'user':
            return False
        text = msg.content if isinstance(msg.content, str) else ''
        return text.startswith('[ORIGINAL USER REQUEST]') or text.startswith('Focus on the user')

    @staticmethod
    def _repair_tool_pairing(messages: list) -> int:
        """Remove orphaned tool results and tool calls after message deletion.

        Ensures every tool_call_id in a tool-result message has a matching
        tool_calls entry in a preceding assistant message, and vice-versa.
        Without this, the LLM API rejects the conversation.

        Returns the number of messages removed.
        """
        removed = 0

        # Collect all tool_call IDs from assistant messages
        offered_ids: set[str] = set()
        for msg in messages:
            if getattr(msg, 'tool_calls', None):
                for tc in msg.tool_calls:
                    tc_id = getattr(tc, 'id', None)
                    if tc_id:
                        offered_ids.add(tc_id)

        # Remove tool-result messages whose tool_call_id is not in offered_ids
        i = 0
        while i < len(messages):
            msg = messages[i]
            tc_id = getattr(msg, 'tool_call_id', None)
            if tc_id and tc_id not in offered_ids:
                del messages[i]
                removed += 1
                continue
            i += 1

        # Collect all answered tool_call IDs
        answered_ids: set[str] = set()
        for msg in messages:
            tc_id = getattr(msg, 'tool_call_id', None)
            if tc_id:
                answered_ids.add(tc_id)

        # Remove tool_calls from assistant messages that have no matching result
        for msg in messages:
            tc_list = getattr(msg, 'tool_calls', None)
            if not tc_list:
                continue
            original_len = len(tc_list)
            tc_list[:] = [tc for tc in tc_list if getattr(tc, 'id', None) in answered_ids]
            if len(tc_list) < original_len:
                removed += original_len - len(tc_list)
            # If all tool_calls were removed, clear the attribute
            if not tc_list:
                msg.tool_calls = None

        if removed:
            logger.info(f"Repaired tool pairing: removed {removed} orphaned messages/calls")
        return removed

    def _compress_runtime_context(self) -> int:
        """Proactively compress the agent's runtime context.

        Strategy (inspired by Openclaw's context engine):
        1. Drop redundant next_step_prompt user messages (keep only the latest).
        2. Truncate ALL older message content (tool results, assistant, user).
        3. If still over budget, drop entire old message rounds.

        Trigger: estimated tokens > 50 % of context_window.
        """
        if not self._agent or not hasattr(self._agent, 'memory'):
            return 0

        messages = self._agent.memory.messages
        if len(messages) <= 6:
            return 0

        estimated = self._estimate_runtime_tokens()
        budget = int(self.context_window * 0.50)

        if estimated <= budget:
            return 0

        logger.warning(
            f"Context compression triggered: ~{estimated:,} tokens "
            f"(budget: {budget:,}, window: {self.context_window:,}). "
            f"Messages: {len(messages)}"
        )

        compressed = 0

        # Phase 1: Remove all but the LAST next_step_prompt user message.
        last_nsp_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if self._is_next_step_user_msg(messages[i]):
                last_nsp_idx = i
                break
        indices_to_remove = []
        for i in range(len(messages)):
            if i != last_nsp_idx and self._is_next_step_user_msg(messages[i]):
                indices_to_remove.append(i)
        for idx in reversed(indices_to_remove):
            del messages[idx]
            compressed += 1
        if indices_to_remove:
            logger.info(f"Phase 1: removed {len(indices_to_remove)} old next_step_prompt messages")

        # Phase 2: Truncate content of ALL messages except the first and last 6.
        keep_tail = min(6, len(messages))
        max_content = 300
        for i in range(1, max(1, len(messages) - keep_tail)):
            msg = messages[i]
            if isinstance(msg.content, str) and len(msg.content) > max_content:
                orig = len(msg.content)
                msg.content = msg.content[:max_content] + f"\n...[truncated {orig - max_content} chars]"
                compressed += 1

        # Phase 3: If still over budget, drop oldest rounds (keep first + last 8).
        estimated = self._estimate_runtime_tokens()
        if estimated > budget and len(messages) > 12:
            keep_head = 1
            keep_tail_drop = min(8, len(messages) - 1)
            droppable = len(messages) - keep_head - keep_tail_drop
            if droppable > 4:
                drop_count = droppable // 2
                del messages[keep_head:keep_head + drop_count]
                compressed += drop_count
                logger.info(f"Phase 3: dropped {drop_count} oldest messages")

        # Phase 4: Repair tool_use/tool_result pairing broken by message deletion.
        compressed += self._repair_tool_pairing(messages)

        final_est = self._estimate_runtime_tokens()
        logger.info(
            f"Context compression done: {compressed} actions, "
            f"tokens ~{estimated:,} -> ~{final_est:,}, "
            f"messages {len(messages)}"
        )
        return compressed

    def _force_compress_runtime_context(self) -> int:
        """Emergency context compression when normal compression is insufficient.

        Aggressively truncates ALL content and drops messages to get under 40 %
        of context_window.
        """
        if not self._agent or not hasattr(self._agent, 'memory'):
            return 0

        messages = self._agent.memory.messages
        if len(messages) <= 4:
            return 0

        compressed = 0

        # Truncate ALL messages to 150 chars
        for msg in messages:
            if isinstance(msg.content, str) and len(msg.content) > 150:
                msg.content = msg.content[:150] + "\n...[force-truncated]"
                compressed += 1

        # Drop all but first + last 6 messages
        if len(messages) > 8:
            keep_head = 1
            keep_tail = min(6, len(messages) - 1)
            drop_count = len(messages) - keep_head - keep_tail
            if drop_count > 0:
                del messages[keep_head:keep_head + drop_count]
                compressed += drop_count

        # Repair tool pairing broken by message deletion
        compressed += self._repair_tool_pairing(messages)

        logger.warning(f"Force-compressed {compressed} messages/results for recovery")
        return compressed

    def _install_anti_loop_tracker(self, base_prompt: str) -> None:
        """Monkey-patch the agent's think() to inject dynamic progress tracking.

        Tracks tool calls with their key arguments so the model sees a precise
        history of what has been done and is warned about repeated actions with
        the same parameters (e.g. reading the same file twice).
        """
        agent = self._agent
        if agent is None:
            return

        agent_loop = self
        original_think = agent.think
        call_tracker: Counter = Counter()
        detail_tracker: Counter = Counter()
        read_files: set = set()

        def _extract_key_arg(tc) -> str:
            """Extract the most meaningful argument for dedup tracking."""
            fn = getattr(tc, 'function', tc)
            name = getattr(fn, 'name', '')
            raw_args = getattr(fn, 'arguments', '')
            if not raw_args:
                return ''
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except (json.JSONDecodeError, TypeError):
                return ''
            if not isinstance(args, dict):
                return ''
            if name in ('read_file', 'read_text_file'):
                return args.get('path', '')[:80]
            if name == 'shell':
                return args.get('command', '')[:60]
            if name == 'skill_marketplace':
                return f"{args.get('action', '')}:{args.get('url', args.get('skill_name', ''))}"[:80]
            if name in ('write_file', 'edit_file', 'list_dir', 'list_directory'):
                return args.get('path', '')[:80]
            if name == 'self_upgrade':
                return args.get('action', '')
            return ''

        ws_posix = str(self.workspace).replace("\\", "/")
        import re as _re
        ws_posix = _re.sub(r'^([A-Za-z]):', lambda m: f'/{m.group(1).lower()}', ws_posix)

        def _evict_duplicate_tool_results():
            """When the same file/skill/dir is read again, replace OLD content with a stub.

            This frees context budget without breaking tool message pairing.
            Handles: read_file results, skill_marketplace results, list_dir results.
            """
            if not hasattr(agent, 'memory') or not agent.memory.messages:
                return
            messages = agent.memory.messages
            # Map: dedup_key -> index of LATEST tool message with that content
            latest_by_key: dict[str, int] = {}
            evicted = 0
            for i, msg in enumerate(messages):
                if getattr(msg, 'role', '') != 'tool':
                    continue
                content = msg.content if isinstance(msg.content, str) else ''
                if len(content) < 80:
                    continue

                dedup_key = None
                # read_file results have [file: name | size] header
                if content.startswith('[file:'):
                    first_line = content.split('\n', 1)[0]
                    if '|' in first_line:
                        dedup_key = f"file:{first_line.split('|')[0].replace('[file:', '').strip()}"
                # skill_marketplace results
                elif content.startswith('Skill:') or content.startswith('SUCCESS: Skill'):
                    for line in content.split('\n')[:3]:
                        if line.startswith('Skill:') or "Skill '" in line:
                            dedup_key = f"skill:{line.strip()[:60]}"
                            break
                # list_dir results
                elif content.startswith('[DIR]') or content.startswith('[FILE]'):
                    dedup_key = f"dir:{content[:40]}"
                # Shell results containing file content (e.g. cat, read via shell)
                elif len(content) > 200:
                    _shell_prefix = 'Observed output of cmd shell execution: '
                    if content.startswith(_shell_prefix):
                        _body = content[len(_shell_prefix):]
                        # Detect env file dumps
                        if 'PRIVATE_KEY=' in _body or 'AGENT_ADDRESS=' in _body:
                            dedup_key = "shell_env:env_local"
                        # Detect file reads via shell
                        elif _body.startswith('---\nname:') or _body.startswith('# '):
                            header = _body.split('\n', 1)[0]
                            dedup_key = f"shell_doc:{header.strip()[:60]}"

                if not dedup_key:
                    continue

                if dedup_key in latest_by_key:
                    old_idx = latest_by_key[dedup_key]
                    old_msg = messages[old_idx]
                    old_content = old_msg.content if isinstance(old_msg.content, str) else ''
                    if len(old_content) > 100:
                        old_msg.content = f"[{dedup_key} — superseded by newer result]"
                        evicted += 1
                latest_by_key[dedup_key] = i

            if evicted:
                logger.info(f"Evicted {evicted} duplicate tool results from context")

        async def _tracked_think() -> bool:
            _evict_duplicate_tool_results()
            agent_loop._compress_runtime_context()

            if call_tracker:
                completed_parts = []
                for key, count in detail_tracker.most_common(8):
                    completed_parts.append(f"  - {key} (x{count})")
                completed_summary = "\n".join(completed_parts)

                anti_loop = f"\n[DONE]:\n{completed_summary}"

                if read_files:
                    recent = sorted(read_files)[-8:]
                    anti_loop += "\n[FILES ALREADY READ — do NOT read again]:\n"
                    anti_loop += "\n".join(f"  - {f}" for f in recent)

                repeated_details = [k for k, c in detail_tracker.items() if c >= 2]
                if repeated_details:
                    anti_loop += (
                        "\n⚠️ STOP REPEATING! You already did these actions. "
                        "Move to the NEXT step toward the goal."
                    )

                if len(anti_loop) > 1000:
                    anti_loop = anti_loop[:1000] + "..."

                agent.next_step_prompt = base_prompt + anti_loop

            result = await original_think()

            _log_agent_reasoning()
            _log_tool_calls()

            if hasattr(agent, 'tool_calls') and agent.tool_calls:
                for tc in agent.tool_calls:
                    fn = getattr(tc, 'function', tc)
                    name = getattr(fn, 'name', '')
                    call_tracker[name] += 1
                    key_arg = _extract_key_arg(tc)
                    detail_key = f"{name}({key_arg})" if key_arg else name
                    detail_tracker[detail_key] += 1
                    if name in ('read_file', 'read_text_file') and key_arg:
                        read_files.add(key_arg)
                    elif name == 'shell' and key_arg:
                        for _env_pat in ('.env.local', '.env', 'SKILL.md'):
                            if _env_pat in key_arg:
                                read_files.add(_env_pat)

            return result

        def _log_agent_reasoning():
            """Extract and log the agent's reasoning text from its last response."""
            from spoon_bot.utils.privacy import mask_secrets
            if not hasattr(agent, 'memory') or not agent.memory.messages:
                return
            for msg in reversed(agent.memory.messages[-3:]):
                role = getattr(msg, 'role', '')
                if role != 'assistant':
                    continue
                content = msg.content if isinstance(msg.content, str) else ''
                if not content or not content.strip():
                    break
                safe_text = mask_secrets(content.strip())
                if len(safe_text) > 500:
                    safe_text = safe_text[:500] + "…"
                logger.info(f"💭 Agent reasoning: {safe_text}")
                break

        def _log_tool_calls():
            """Log each tool call with arguments so TUI can display them."""
            from spoon_bot.utils.privacy import mask_secrets
            if not hasattr(agent, 'tool_calls') or not agent.tool_calls:
                return
            for tc in agent.tool_calls:
                fn = getattr(tc, 'function', tc)
                name = getattr(fn, 'name', '')
                raw_args = getattr(fn, 'arguments', '')
                safe_args = mask_secrets(raw_args) if raw_args else ''
                logger.info(f"Tool call: {name}({safe_args})")

        agent.think = _tracked_think

    def _filter_execution_steps(self, content: str) -> str:
        """
        Filter out technical execution steps from agent output.
        Removes lines like "Step 1: Observed output of cmd..."

        Args:
            content: Raw agent output

        Returns:
            Cleaned content without execution steps
        """
        import re

        if not content:
            return content or ""

        lines = content.split('\n')
        filtered_lines = []
        skip_until_blank = False

        for line in lines:
            # Skip lines that match execution step patterns
            if re.match(r'^Step \d+:', line):
                skip_until_blank = True
                continue

            # Skip lines that are part of step output
            if skip_until_blank:
                # If we hit a blank line or normal content, stop skipping
                if line.strip() == '' or not line.startswith((' ', '\t', 'Observed', 'Error', 'Security', 'Command:', 'Successfully')):
                    skip_until_blank = False
                    if line.strip():  # Add the line if it's not blank
                        filtered_lines.append(line)
                continue

            # Keep all other lines
            filtered_lines.append(line)

        # If everything was filtered (e.g. all lines were "Step N: ..."), fall back to
        # extracting the inline content from the last Step line so we never return "".
        if not filtered_lines and lines:
            for raw_line in reversed(lines):
                m = re.match(r'^Step \d+:\s*(.+)', raw_line)
                if m and m.group(1).strip():
                    filtered_lines = [m.group(1).strip()]
                    break
            if not filtered_lines:
                # Last-resort: return original content unchanged
                return content.strip()

        # Join and clean up excessive blank lines
        result = '\n'.join(filtered_lines)
        result = re.sub(r'\n{3,}', '\n\n', result)  # Replace 3+ newlines with 2
        return result.strip()


    async def stream(
        self,
        message: str,
        media: list[str] | None = None,
        attachments: list[dict[str, Any]] | None = None,
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
              source:   Machine-readable producer metadata for this output
        """
        if not self._initialized:
            await self.initialize()

        current_source = AgentLoop._set_last_response_source(self)
        logger.info(f"Streaming message: {message[:100]}...")

        current_session_key = getattr(self, "session_key", "default")
        self.set_subagent_context(session_key=current_session_key)

        created = self._maybe_create_persistent_subagent_from_request(message)
        if isinstance(created, str):
            if created:
                full_content = created
                yield {
                    "type": "content",
                    "delta": created,
                    "metadata": {"persistent_subagent_created": True},
                    "source": current_source,
                }
                yield {
                    "type": "done",
                    "delta": "",
                    "metadata": {"content": full_content},
                    "source": current_source,
                }
                self._persist_turn(message, full_content)
            return

        routed = await self._maybe_route_to_persistent_specialist_result(message)
        if isinstance(routed, tuple) and len(routed) == 3:
            routed_content, _route_note, current_source = routed
            AgentLoop._set_last_response_source(self, current_source)
            full_content = routed_content or ""
            if full_content:
                yield {
                    "type": "content",
                    "delta": full_content,
                    "metadata": {"auto_routed_to_subagent": True},
                    "source": current_source,
                }
            yield {
                "type": "done",
                "delta": "",
                "metadata": {"content": full_content},
                "source": current_source,
            }
            if full_content:
                self._persist_turn(message, full_content)
            return

        # Refresh memory context
        try:
            memory_context = self.memory.get_memory_context()
            if memory_context:
                self.context.set_memory_context(memory_context)
        except Exception as e:
            logger.warning(f"Failed to load memory context: {e}")

        full_content = ""

        # Trim and inject persisted history into runtime memory
        await self._prepare_request_context()

        _base_prompt = self._build_step_prompt(message)
        self._agent.next_step_prompt = _base_prompt
        self._install_anti_loop_tracker(_base_prompt)

        # ------------------------------------------------------------------
        # Streaming uses the spoon-core run+stream pattern:
        #   1. Clear task_done + drain output_queue
        #   2. Start run(message) in background — sets task_done on finish
        #   3. Read chunks from output_queue until task_done AND queue empty
        # ------------------------------------------------------------------

        try:
            # 1. Reset streaming state
            self._agent.task_done.clear()
            while not self._agent.output_queue.empty():
                try:
                    await asyncio.wait_for(self._agent.output_queue.get(), timeout=0.1)
                except (asyncio.TimeoutError, Exception):
                    break

            # 2. Start run() in background
            run_result_text = ""

            async def _run_and_signal() -> None:
                nonlocal run_result_text
                try:
                    result = await self._agent.run(request=message)
                    if hasattr(result, "content"):
                        run_result_text = result.content or ""
                    elif isinstance(result, str):
                        run_result_text = result
                    elif result is not None:
                        run_result_text = str(result)
                except Exception as exc:
                    logger.error(f"Background agent run failed: {exc}")
                    try:
                        await self._agent.output_queue.put({
                            "type": "error",
                            "delta": str(exc),
                            "metadata": {"error": str(exc), "error_code": type(exc).__name__},
                        })
                    except Exception:
                        pass
                finally:
                    self._agent.task_done.set()

            logger.debug(f"Creating bg task, agent state={self._agent.state}")
            bg_task = asyncio.create_task(_run_and_signal())

            # Force a yield to allow the background task to start
            await asyncio.sleep(0)

            # 3. Read output chunks (mirrors fixed BaseAgent.stream logic)
            oq = self._agent.output_queue
            td = self._agent.task_done
            logger.debug(f"output_queue type={type(oq).__name__}, task_done type={type(td).__name__}")
            stream_timeout = 120.0
            deadline = asyncio.get_event_loop().time() + stream_timeout
            chunk_count = 0

            logger.debug(f"Entering stream loop: td={td.is_set()}, qempty={oq.empty()}, qsize={oq.qsize()}")
            while not (td.is_set() and oq.empty()):
                if asyncio.get_event_loop().time() > deadline:
                    logger.warning("Streaming deadline reached, stopping")
                    break

                try:
                    # Use oq.get() without timeout kwarg — works for both
                    # asyncio.Queue and ThreadSafeOutputQueue. Timeout is
                    # handled by the outer asyncio.wait_for.
                    chunk = await asyncio.wait_for(oq.get(), timeout=2.0)
                    chunk_count += 1
                    logger.debug(f"Got chunk #{chunk_count}: type={type(chunk).__name__}, repr={repr(chunk)[:200]}")
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    logger.warning("Streaming cancelled")
                    break
                except Exception as e:
                    logger.warning(f"Queue get error: {type(e).__name__}: {e}")
                    continue

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
                    if chunk.get("type") == "error":
                        yield {
                            "type": "error",
                            "delta": chunk.get("delta", ""),
                            "metadata": chunk.get("metadata", {}),
                            "source": current_source,
                        }
                        continue

                    if "tool_calls" in chunk and chunk["tool_calls"]:
                        for tc in chunk["tool_calls"]:
                            # tc may be a ToolCall pydantic object or a dict
                            if isinstance(tc, dict):
                                fn = tc.get("function", {})
                                tc_id = tc.get("id", "")
                                fn_name = fn.get("name", "") if isinstance(fn, dict) else getattr(fn, "name", "")
                                fn_args = fn.get("arguments", "") if isinstance(fn, dict) else getattr(fn, "arguments", "")
                            else:
                                tc_id = getattr(tc, "id", "")
                                fn_obj = getattr(tc, "function", None)
                                fn_name = getattr(fn_obj, "name", "") if fn_obj else ""
                                fn_args = getattr(fn_obj, "arguments", "") if fn_obj else ""
                            yield {
                                "type": "tool_call",
                                "delta": "",
                                "metadata": {
                                    "id": tc_id,
                                    "name": fn_name,
                                    "arguments": fn_args,
                                },
                                "source": current_source,
                            }
                        continue
                    # Support both "content" and "delta" keys (#10)
                    text = chunk.get("content") or chunk.get("delta") or ""
                    if text:
                        delta = text
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
                    yield {
                        "type": chunk_type,
                        "delta": delta,
                        "metadata": metadata,
                        "source": current_source,
                    }

            logger.debug(f"Stream loop exited: td={td.is_set()}, qempty={oq.empty()}, chunks_received={chunk_count}, full_content_len={len(full_content)}")

            # Ensure background task completes
            try:
                await asyncio.wait_for(bg_task, timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                pass

            # Fallback: if run() completed but no stream chunks were emitted,
            # use final run result as one content chunk to avoid empty output.
            if not full_content and run_result_text:
                logger.warning(
                    "Stream produced no content chunks; "
                    "falling back to run() result text."
                )
                full_content = run_result_text
                yield {
                    "type": "content",
                    "delta": run_result_text,
                    "metadata": {"fallback": "run_result_no_chunks"},
                    "source": current_source,
                }

            # Emit done
            yield {
                "type": "done",
                "delta": "",
                "metadata": {"content": full_content},
                "source": current_source,
            }

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            current_source = AgentLoop.get_last_response_source(self)
            yield {
                "type": "error",
                "delta": str(e),
                "metadata": {"error": str(e)},
                "source": current_source,
            }
            yield {
                "type": "done",
                "delta": "",
                "metadata": {"error": str(e)},
                "source": current_source,
            }

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
        attachments: list[dict[str, Any]] | None = None,
        session_key: str | None = None,
    ) -> tuple[str, str | None]:
        """
        Process a user message and return the agent's response with thinking content.

        Args:
            message: The user's message.
            media: Optional list of media file paths.
            session_key: Optional session key for multi-user/multi-channel isolation.

        Returns:
            Tuple of (response_text, thinking_content). thinking_content may be None.
        """
        AgentLoop._set_last_response_source(self)

        # Switch session if requested
        if session_key and session_key != self.session_key:
            self.session_key = session_key
            self._session = self.sessions.get_or_create(session_key)
            logger.info(f"Switched to session: {session_key}")

        if not self._initialized:
            await self.initialize()

        # Switch session if a different key is requested
        if session_key and session_key != self.session_key:
            self._session = self.sessions.get_or_create(session_key)
            self.session_key = session_key
            logger.debug(f"Switched to session: {session_key}")

        current_session_key = getattr(self, "session_key", "default")
        self.set_subagent_context(session_key=current_session_key)

        logger.info(f"Processing message (with thinking): {message[:100]}...")

        created = self._maybe_create_persistent_subagent_from_request(message)
        if isinstance(created, str):
            self._persist_turn(message, created)
            return created, "Created persistent subagent from natural-language request."

        routed = await self._maybe_route_to_persistent_specialist_result(message)
        if isinstance(routed, tuple) and len(routed) == 3:
            routed_content, route_note, route_source = routed
            AgentLoop._set_last_response_source(self, route_source)
            self._persist_turn(message, routed_content)
            if self._auto_commit and self._git:
                try:
                    if self._git.has_changes():
                        self._git.commit(message)
                except Exception as e:
                    logger.warning(f"Failed to auto-commit: {e}")
            return routed_content, route_note

        # Refresh memory context
        try:
            memory_context = self.memory.get_memory_context()
            if memory_context:
                self.context.set_memory_context(memory_context)
        except Exception as e:
            logger.warning(f"Failed to load memory context: {e}")

        # Trim and inject persisted history into runtime memory
        await self._prepare_request_context()

        _base_prompt = self._build_step_prompt(message)
        self._agent.next_step_prompt = _base_prompt
        self._install_anti_loop_tracker(_base_prompt)

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
            raise

        # Save to session
        self._persist_turn(message, final_content)
        AgentLoop._set_last_response_source(self)

        # Auto-commit workspace changes
        if self._auto_commit and self._git:
            try:
                if self._git.has_changes():
                    self._git.commit(message)
            except Exception as e:
                logger.warning(f"Failed to auto-commit: {e}")

        return final_content, thinking_content

    def _workspace_posix_path(self) -> str:
        """Return the workspace path in POSIX form for shell commands."""
        import re as _re
        import sys
        raw = str(self.workspace).replace("\\", "/")
        if sys.platform == "win32":
            raw = _re.sub(r'^([A-Za-z]):', lambda m: f'/{m.group(1).lower()}', raw)
        return raw

    def _pre_inject_matched_skill(self, message: str) -> str:
        """Match user message to an installed skill and prepend SKILL.md content.

        Saves 2-5 agent steps by providing skill content upfront so the
        agent can start executing immediately without calling read_file.
        """
        import re as _re

        skills_dir = self.workspace / "skills"
        if not skills_dir.is_dir():
            return message

        msg_lower = message.lower()
        best_skill = None
        best_score = 0

        for child in sorted(skills_dir.iterdir()):
            if not child.is_dir():
                continue
            skill_md = child / "SKILL.md"
            if not skill_md.exists():
                continue

            name = child.name
            score = 0

            # Match skill name tokens against user message
            name_tokens = _re.split(r'[-_]', name)
            for token in name_tokens:
                if len(token) >= 3 and token.lower() in msg_lower:
                    score += 2

            # Match trigger words from frontmatter
            try:
                raw = skill_md.read_text(encoding="utf-8", errors="replace")
                fm = _re.match(r'^---\s*\n(.*?)\n---', raw, _re.DOTALL)
                if fm:
                    for line in fm.group(1).split("\n"):
                        if "triggers" in line.lower() or "trigger" in line.lower():
                            triggers = _re.findall(r'"([^"]+)"', line)
                            for t in triggers:
                                if t.lower() in msg_lower:
                                    score += 3
            except Exception:
                pass

            if score > best_score:
                best_score = score
                best_skill = (name, skill_md)

        if not best_skill or best_score < 2:
            return message

        skill_name, skill_path = best_skill
        try:
            content = skill_path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            return message

        logger.info(f"Pre-injected skill '{skill_name}' (score={best_score}) into message")
        return (
            f"{message}\n\n"
            f"---\n"
            f"[PRE-LOADED SKILL: {skill_name}] "
            f"The following SKILL.md is already loaded — execute its steps directly. "
            f"Do NOT call read_file on this skill again.\n\n"
            f"{content}\n"
            f"---"
        )

    def _build_step_prompt(self, message: str) -> str:
        """Build a minimal per-step prompt from the user's request.

        Keeps only the user's original request and workspace path.
        Injects env vars so they survive short-term memory pruning.
        The anti-loop tracker dynamically appends progress info.
        """
        _truncated = message[:300] + ("…" if len(message) > 300 else "")
        _ws = self._workspace_posix_path()
        prompt = (
            f"[USER REQUEST]: {_truncated}\n"
            f"[WORKSPACE]: {_ws}/\n\n"
            + self.DEFAULT_NEXT_STEP_PROMPT
        )
        env_section = self._extract_env_for_prompt()
        if env_section:
            prompt += env_section
        return prompt

    def _extract_env_for_prompt(self) -> str:
        """Extract env vars from .env.local for the step prompt.

        Non-sensitive values shown directly. Private keys masked —
        agent told to use ``source .env.local`` then ``$VAR``.
        Persists across short-term memory pruning.
        """
        env_file = self.workspace / ".env.local"
        if not env_file.exists():
            return ""
        try:
            raw = env_file.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            return ""

        env_vars: dict[str, str] = {}
        for line in raw.split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip()
            if not key:
                continue
            _sensitive = any(s in key.upper() for s in (
                "PRIVATE", "SECRET", "KEY", "PASSWORD", "TOKEN", "MNEMONIC",
                "CREDENTIAL", "AUTH", "PASSPHRASE",
            ))
            env_vars[key] = "<set>" if _sensitive and val else val

        if not env_vars:
            return ""

        parts = ["\n[ENV — from .env.local — do NOT re-read, use `source .env.local` for secrets]:"]
        for k, v in env_vars.items():
            parts.append(f"  {k}={v}")
        return "\n".join(parts) + "\n"

    # ------------------------------------------------------------------
    # Dynamic prompt helpers
    # ------------------------------------------------------------------

    def _build_skills_for_prompt(self) -> str:
        """Build Openclaw-style XML metadata for installed skills.

        Scans skill directories, extracts name + description from SKILL.md
        frontmatter, and returns an <available_skills> XML block for the
        system prompt. The agent reads SKILL.md lazily when needed.
        """
        import re as _re, os as _os, sys as _sys

        skills_dir = self.workspace / "skills"
        if not skills_dir.is_dir():
            return ""

        ws_str = str(self.workspace).replace("\\", "/")
        if _sys.platform == "win32":
            ws_str = _re.sub(r'^([A-Za-z]):', lambda m: f'/{m.group(1).lower()}', ws_str)

        entries = []
        for child in sorted(skills_dir.iterdir()):
            if not child.is_dir():
                continue
            skill_md = child / "SKILL.md"
            if not skill_md.exists():
                continue

            name = child.name
            description = ""
            try:
                raw = skill_md.read_text(encoding="utf-8", errors="replace")
                fm_match = _re.match(r'^---\s*\n(.*?)\n---', raw, _re.DOTALL)
                if fm_match:
                    for line in fm_match.group(1).split("\n"):
                        stripped = line.strip()
                        if stripped.startswith("description:"):
                            val = stripped.split(":", 1)[1].strip().strip("'\"")
                            if val and val != ">":
                                description = val[:200]
                                break
                    if not description:
                        in_desc = False
                        desc_lines = []
                        for line in fm_match.group(1).split("\n"):
                            if line.strip().startswith("description:"):
                                in_desc = True
                                val = line.split(":", 1)[1].strip()
                                if val and val not in (">", "|"):
                                    desc_lines.append(val)
                                continue
                            if in_desc:
                                if line.startswith("  ") or line.startswith("\t"):
                                    desc_lines.append(line.strip())
                                else:
                                    break
                        description = " ".join(desc_lines)[:200]
                if not description:
                    for line in raw.split("\n")[1:20]:
                        stripped = line.strip()
                        if stripped and not stripped.startswith(("#", "---", "```")):
                            description = stripped[:200]
                            break
            except Exception:
                description = name

            entries.append(
                f'<skill name="{name}">\n'
                f'  <description>{description}</description>\n'
                f'  <location>skills/{name}/SKILL.md</location>\n'
                f'</skill>'
            )

        if not entries:
            return ""
        return "<available_skills>\n" + "\n".join(entries) + "\n</available_skills>"

    @staticmethod
    def _build_dynamic_tools_prompt(inactive_tools: dict[str, "Tool"]) -> str:
        """Build the 'Dynamically Loadable Tools' system-prompt section.

        Lists ALL inactive tools with their descriptions so the AI Agent
        can autonomously decide which to activate. No hardcoded topic
        mapping — the LLM reads tool descriptions and decides for itself.
        """
        lines: list[str] = [
            "\n\n## Inactive Tools (activate on demand)\n\n"
            "Call `activate_tool(action='activate', tool_name='<name>')` to load any of these. "
            "Activate what you need BEFORE answering.\n"
        ]

        for tool in inactive_tools.values():
            lines.append(f"- `{tool.name}`: {tool.description}")

        return "\n".join(lines)

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

    # ------------------------------------------------------------------
    # Session management helpers (new)
    # ------------------------------------------------------------------

    def stop_current_task(self) -> bool:
        """Request the current or next agent task to stop.

        Sets a flag that is checked at the start of the next ``process()``
        call.  Cannot interrupt a task that is already mid-execution, but
        will prevent the queued task from running.

        Returns:
            True (always succeeds in setting the flag).
        """
        self._stop_requested = True
        logger.info("Stop requested — will be honoured on next process() call")
        return True

    def new_session(self, session_key: str | None = None) -> str:
        """Switch to a brand-new conversation session.

        Creates a new session in the configured backend and replaces the
        current in-memory session.  The old session is preserved in storage.

        Args:
            session_key: Optional explicit key for the new session.
                         Defaults to a UUID-based key.

        Returns:
            The new session key.
        """
        import uuid

        new_key = session_key or f"session-{uuid.uuid4().hex[:8]}"
        self._session = self.sessions.get_or_create(new_key)
        self.session_key = new_key
        logger.info(f"Switched to new session: {new_key}")
        return new_key

    def compact_session(self) -> int:
        """Compact conversation history by keeping only first 2 and last 2 messages.

        Useful for reducing context size when a session becomes very long.
        The compacted history is immediately persisted.

        Returns:
            Number of messages removed.
        """
        history = self._session.get_history()
        if len(history) <= 4:
            return 0

        removed = len(history) - 4
        keep = history[:2] + history[-2:]
        self._session.messages.clear()
        for msg in keep:
            self._session.add_message(msg["role"], msg["content"])
        self.sessions.save(self._session)
        logger.info(f"Session compacted: removed {removed} messages, kept {len(keep)}")
        return removed

    def get_usage(self) -> dict[str, Any]:
        """Return basic usage statistics for the current session.

        Returns:
            Dictionary with message count, session key, and model.
        """
        history = self._session.get_history()
        return {
            "messages": len(history),
            "session_key": self.session_key,
            "model": self.model,
        }

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

    @property
    def subagent_manager(self) -> SubagentManager | None:
        """Expose sub-agent manager for external access (e.g. slash commands)."""
        return getattr(self, "_subagent_manager", None)


async def create_agent(
    model: str | None = None,
    provider: str | None = None,
    api_key: str | None = None,
    workspace: Path | str | None = None,
    session_key: str = "default",
    base_url: str | None = None,
    mcp_config: dict[str, dict[str, Any]] | None = None,
    enable_skills: bool = True,
    auto_commit: bool = True,
    enabled_tools: set[str] | None = None,
    tool_profile: str | None = None,
    memsearch_config: MemSearchConfig | dict[str, Any] | None = None,
    auto_reload: bool = False,
    auto_reload_interval: float = 5.0,
    config_path: Path | str | None = None,
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
        memsearch_config=memsearch_config,
        auto_reload=auto_reload,
        auto_reload_interval=auto_reload_interval,
        config_path=config_path,
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
