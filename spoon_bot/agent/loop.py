"""
Agent loop: the core processing engine using spoon-core SDK.

This module provides the main agent interface, integrating spoon-core's
ChatBot, SpoonReactMCP, and SkillManager with spoon-bot's native OS tools.
"""

from __future__ import annotations

import asyncio
import json
import logging as stdlib_logging
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
        "spoon-bot requires spoon-core SDK. Install with: pip install spoon-ai"
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
from spoon_bot.agent.tools.self_config import (
    ActivateToolTool,
    SelfConfigTool,
    MemoryManagementTool,
    SelfUpgradeTool,
)
from spoon_bot.agent.tools.web import WebSearchTool, WebFetchTool
from spoon_bot.config import AgentLoopConfig, MemSearchConfig, validate_agent_loop_params, resolve_context_window
from spoon_bot.services.hotreload import HotReloadService
from spoon_bot.services.spawn import SpawnTool
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

    # Injected as a USER message at each agent step to steer tool usage.
    # Must be concise (it's repeated every iteration) but directive enough
    # to prevent hallucination and ensure the model calls tools.
    DEFAULT_NEXT_STEP_PROMPT = (
        "The user's LATEST message is the top priority — always respond to what they just said. "
        "For simple conversational messages (greetings, thanks, etc.) respond directly without calling tools or running skill initializations. "
        "For explicit task requests, pick the best tool and call it. "
        "Do NOT fabricate output or pretend to run commands. "
        "If a step fails or returns an error, STOP and report the failure — do not continue with follow-up steps. "
        "NEVER ask the user questions — make autonomous default choices when needed."
    )

    def __init__(
        self,
        workspace: Path | str | None = None,
        model: str | None = None,
        provider: str | None = None,
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
            # Load skill tools into the ToolRegistry
            self._register_skill_tools()
            # Auto-activate skill tools so the agent can use them immediately
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
        # Shell tool — allow_chaining lets the agent compose multi-step ops
        self.tools.register(ShellTool(
            timeout=self.shell_timeout,
            max_output=self.max_output,
            working_dir=str(self.workspace),
            allow_chaining=True,
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

        # Background task tool
        self.tools.register(SpawnTool())

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
        """Shut down all managed resources (MCP tools, skills, etc.)."""
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
        # Honour stop request from previous /stop command
        if self._stop_requested:
            self._stop_requested = False
            logger.info("Task skipped due to stop request")
            return "Task stopped."

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

        # Run agent
        try:
            result = await self._agent.run(message)

            logger.debug(f"Agent result type: {type(result)}, attrs: {[a for a in dir(result) if not a.startswith('_')]}")
            if hasattr(result, 'content'):
                logger.info(f"Agent result.content (first 500): {str(result.content)[:500]}")
            if hasattr(result, 'steps'):
                logger.info(f"Agent steps count: {len(result.steps) if result.steps else 0}")
                for i, step in enumerate(result.steps or []):
                    step_str = str(step)[:200]
                    logger.debug(f"  Step {i}: {step_str}")
            if hasattr(result, 'tool_calls'):
                logger.info(f"Agent tool_calls: {result.tool_calls}")

            # Extract content — guard against result.content being None
            if hasattr(result, "content") and result.content is not None:
                final_content = result.content
            elif hasattr(result, "content"):
                # result.content exists but is None; fall back to str(result)
                final_content = str(result) if str(result) != "None" else ""
            else:
                final_content = str(result)

            # Filter out technical execution steps (Step 1:, Step 2:, etc.)
            final_content = self._filter_execution_steps(final_content)

        except Exception as e:
            logger.error(f"Agent processing error: {e}")
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

        logger.info(f"Streaming message: {message[:100]}...")

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
                    yield {"type": chunk_type, "delta": delta, "metadata": metadata}

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
                }

            # Emit done
            yield {"type": "done", "delta": "", "metadata": {"content": full_content}}

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield {"type": "error", "delta": str(e), "metadata": {"error": str(e)}}
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

        # Trim and inject persisted history into runtime memory
        await self._prepare_request_context()

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
    # Dynamic prompt helpers
    # ------------------------------------------------------------------

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
