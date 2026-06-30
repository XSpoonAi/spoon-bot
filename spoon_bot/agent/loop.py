"""
Agent loop: the core processing engine using spoon-core SDK.

This module provides the main agent interface, integrating spoon-core's
ChatBot, SpoonReactMCP, and SkillManager with spoon-bot's native OS tools.
"""

from __future__ import annotations

import asyncio
import copy
import logging as stdlib_logging
import os
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Callable

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
    from spoon_ai.schema import Message, AgentState
    from spoon_ai.llm.interface import LLMResponse
    from spoon_ai.agents.spoon_react_mcp import SpoonReactMCP
    from spoon_ai.agents.spoon_react_skill import SpoonReactSkill
    from spoon_ai.tools import ToolManager
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
from spoon_bot.agent.execution_ledger import (
    ExecutionLedger,
    bind_execution_ledger,
    persist_execution_ledger,
)
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
from spoon_bot.agent.tools.history_search import SearchHistoryTool
from spoon_bot.agent.tools.execution_context import (
    bind_request_execution_hints,
    bind_tool_owner,
    bind_tool_workspace,
    clear_captured_tool_outputs,
    capture_tool_outputs,
    consume_captured_tool_output,
    track_tool_invocations,
)
from spoon_bot.agent.turn_verifiers import (
    build_user_facing_tool_event_answer,
    latest_tool_event_from_skill_continuation,
    latest_tool_event_has_user_summary_marker,
    should_run_skill_contract_check,
    skill_contract_has_progress,
    skill_contract_inspection_stalled_after_progress,
    skill_contract_needs_continuation,
    tool_events_need_more_evidence,
)
from spoon_bot.agent.tools.self_config import (
    ActivateToolTool,
    SelfConfigTool,
    MemoryManagementTool,
    SelfUpgradeTool,
)
from spoon_bot.agent.tools.cron import CronTool
from spoon_bot.agent.tools.web import WebSearchTool, WebFetchTool
from spoon_bot.agent.tools.wallet import WalletTool
from spoon_bot.agent.tools.web3 import (
    BalanceCheckTool,
    ContractCallTool,
    SwapTool,
    TransferTool,
)
from spoon_bot.config import (
    DEFAULT_MAX_STREAM_TOOL_RESULTS_WITHOUT_CONTENT,
    DEFAULT_MAX_OUTPUT,
    DEFAULT_PROVIDER_MAX_RETRIES,
    DEFAULT_PROVIDER_SILENCE_TIMEOUT,
    DEFAULT_PROVIDER_TOTAL_TIMEOUT,
    DEFAULT_SHELL_BACKGROUND_HANDOFF_TIMEOUT,
    DEFAULT_SHELL_MAX_TIMEOUT,
    DEFAULT_SHELL_TIMEOUT,
    DEFAULT_TOOL_FOLLOWUP_TIMEOUT,
    MemSearchConfig,
    resolve_context_window,
    validate_agent_loop_params,
)
from spoon_bot.skills.zip_install import InstalledSkillZip
from spoon_bot.services.hotreload import HotReloadService
from spoon_bot.subagent.manager import SubagentManager
from spoon_bot.subagent.tools import SubagentTool
from spoon_bot.subagent.catalog import format_roles_for_prompt
from spoon_bot.session.manager import SessionManager
from spoon_bot.session.store import create_session_store
from spoon_bot.memory.store import MemoryStore
from spoon_bot.wallet import ensure_wallet_runtime
from spoon_bot.exceptions import (
    user_friendly_error,
)
from spoon_bot.services.git import GitManager
from spoon_bot.utils.retry import (
    RetryConfig,
)

if TYPE_CHECKING:
    pass


from spoon_bot.agent import loop_protocol as _loop_protocol
from spoon_bot.agent import loop_skills as _loop_skills
from spoon_bot.agent import loop_state as _loop_state
from spoon_bot.agent.loop_protocol import (
    LoopProtocolMixin,
    _REPEATED_READ_RECOVERY_THRESHOLD,
)
from spoon_bot.agent.loop_skills import (
    LoopSkillsMixin,
)
from spoon_bot.agent.loop_state import (
    LoopStateMixin,
    _DEFAULT_INTERNAL_RECOVERY_TIMEOUT,
    _DEFAULT_NON_SHELL_ACTIVE_TOOL_TIMEOUT,
    _DEFAULT_POST_TOOL_RESULT_SILENCE_TIMEOUT,
    _DEFAULT_PROVIDER_ASK_TIMEOUT,
    _MISSING,
    _TURN_STATE_COMPLETED,
    _TURN_STATE_INTERRUPTED,
    _ensure_attachment_context,  # noqa: F401 - compatibility re-export
    _strip_attachment_context,  # noqa: F401 - compatibility re-export
)


_DEFAULT_STREAM_HEARTBEAT_INITIAL_DELAY = 15.0
_DEFAULT_STREAM_HEARTBEAT_INTERVAL = 30.0


class AgentLoop(LoopStateMixin, LoopProtocolMixin, LoopSkillsMixin):
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
        "Continue only if the latest user request still needs another "
        "tool-backed action. If the latest tool result already satisfies the "
        "requested workflow, reaches a terminal outcome, or shows a concrete "
        "blocker, answer the user now. Do not start a new external workflow or "
        "repeat an external side effect unless the latest user request or the "
        "active skill contract explicitly asks for that next action. Never "
        "treat your own prior question, draft, or status message as the user's "
        "approval to continue. Do not poll unchanged external waiting states "
        "indefinitely; after bounded checks with no material progress, report "
        "the blocker and how the user can resume."
    )

    def __init__(
        self,
        workspace: Path | str | None = None,
        model: str | None = None,
        provider: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_iterations: int = 50,
        shell_timeout: int = DEFAULT_SHELL_TIMEOUT,
        shell_max_timeout: int = DEFAULT_SHELL_MAX_TIMEOUT,
        max_output: int = DEFAULT_MAX_OUTPUT,
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
        yolo_mode: bool = False,
        provider_max_retries: int = DEFAULT_PROVIDER_MAX_RETRIES,
        provider_retry_base_delay: float = 1.0,
        provider_retry_max_delay: float = 60.0,
        provider_retry_backoff_factor: float = 2.0,
        reasoning_effort: str | None = None,
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
            shell_timeout: Default shell foreground timeout in seconds.
            shell_max_timeout: Maximum per-command timeout override in seconds.
            max_output: Maximum output characters for shell.
            session_key: Session identifier for persistence.
            skill_paths: Additional paths to search for skills.
            mcp_config: MCP server configurations.
            system_prompt: Custom system prompt.
            enable_skills: Whether to enable skill system.
            auto_commit: Whether to auto-commit workspace changes after each message.
            enabled_tools: Explicit set of tool names to enable. None = all.
            tool_profile: Named profile ('coding', 'research', 'full').
            session_manager: Existing SessionManager to reuse for persistence.
            subagent_manager: Existing SubagentManager to reuse for child agents.
            session_store_backend: Session storage backend ('file', 'sqlite', 'postgres').
            session_store_dsn: PostgreSQL DSN for 'postgres' backend.
            session_store_db_path: SQLite DB path for 'sqlite' backend.
            context_window: Override context window in tokens (auto-resolved from model if None).
            yolo_mode: Operate directly in user's path without sandbox isolation.
        """
        # Validate parameters
        try:
            self._config = validate_agent_loop_params(
                workspace=workspace,
                model=model,
                reasoning_effort=reasoning_effort,
                max_iterations=max_iterations,
                shell_timeout=shell_timeout,
                shell_max_timeout=shell_max_timeout,
                max_output=max_output,
                session_key=session_key,
                skill_paths=skill_paths,
                mcp_config=mcp_config,
                yolo_mode=yolo_mode,
                provider_max_retries=provider_max_retries,
                provider_retry_base_delay=provider_retry_base_delay,
                provider_retry_max_delay=provider_retry_max_delay,
                provider_retry_backoff_factor=provider_retry_backoff_factor,
            )
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise ValueError(f"Invalid AgentLoop configuration: {e}") from e

        # Store config - callers must provide model/provider explicitly
        self.workspace = self._config.workspace
        self.yolo_mode = self._config.yolo_mode
        self.model = model
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        self.reasoning_effort = self._config.reasoning_effort
        self.max_iterations = self._config.max_iterations
        self.shell_timeout = self._config.shell_timeout
        self.shell_max_timeout = self._config.shell_max_timeout
        self.max_output = self._config.max_output
        self.provider_silence_timeout = self._float_env(
            "SPOON_BOT_PROVIDER_SILENCE_TIMEOUT",
            DEFAULT_PROVIDER_SILENCE_TIMEOUT,
        )
        self.provider_total_timeout = self._float_env(
            "SPOON_BOT_PROVIDER_TOTAL_TIMEOUT",
            DEFAULT_PROVIDER_TOTAL_TIMEOUT,
            allow_zero=True,
        )
        self.tool_followup_timeout = self._float_env(
            "SPOON_BOT_TOOL_FOLLOWUP_TIMEOUT",
            DEFAULT_TOOL_FOLLOWUP_TIMEOUT,
        )
        self.internal_recovery_timeout = self._float_env(
            "SPOON_BOT_INTERNAL_RECOVERY_TIMEOUT",
            _DEFAULT_INTERNAL_RECOVERY_TIMEOUT,
            allow_zero=True,
        )
        self.provider_ask_timeout = self._float_env(
            "SPOON_BOT_PROVIDER_ASK_TIMEOUT",
            _DEFAULT_PROVIDER_ASK_TIMEOUT,
            allow_zero=True,
        )
        self.max_stream_tool_results_without_content = self._int_env(
            "SPOON_BOT_MAX_STREAM_TOOL_RESULTS_WITHOUT_CONTENT",
            DEFAULT_MAX_STREAM_TOOL_RESULTS_WITHOUT_CONTENT,
        )
        self.session_key = self._config.session_key
        self.user_id = "anonymous"
        self._enable_skills = enable_skills
        self._mcp_config = mcp_config or {}
        self._system_prompt = system_prompt
        self._auto_commit = auto_commit
        self._enabled_tools_override = set(enabled_tools) if enabled_tools is not None else None
        self._tool_profile = tool_profile
        self._session_store_backend = session_store_backend or "file"
        self._session_store_dsn = session_store_dsn
        self._session_store_db_path = session_store_db_path
        self._user_skill_paths = [Path(p) for p in skill_paths] if skill_paths else []

        # Context window - auto-resolved from model when not explicit
        self.context_window = resolve_context_window(model, context_window)
        logger.info(f"Context window: {self.context_window:,} tokens (model={model})")

        # spoon-core components (initialized later)
        self._chatbot: ChatBot | None = None
        self._agent: SpoonReactMCP | SpoonReactSkill | None = None
        self._skill_manager: SkillManager | None = None
        self._mcp_tools: list[MCPTool] = []
        self._latest_reasoning_excerpt: str | None = None
        self._pending_reasoning_chunks: list[str] = []
        self._pending_runtime_notices: list[dict[str, Any]] = []
        self._current_turn_skill_zip_installs: list[InstalledSkillZip] = []
        self._current_turn_skill_zip_failures: list[str] = []
        # Track which invalid persisted attachment/media refs we've already
        # warned about per session, so history sync doesn't spam identical
        # warnings on every user turn. Keyed by session_key -> set of refs.
        self._warned_invalid_attachment_refs: dict[str, set[str]] = {}
        self._warned_invalid_media_refs: dict[str, set[str]] = {}

        # spoon-bot components
        self.context = ContextBuilder(self.workspace, yolo_mode=self.yolo_mode)
        self.tools = ToolRegistry()
        self._cron_service = None

        if getattr(self, "yolo_mode", False):
            logger.info(f"YOLO mode enabled - operating directly in: {self.workspace}")

        # Session persistence - configurable backend
        if session_manager is not None:
            self.sessions = session_manager
            logger.info("Session store: inherited from parent SessionManager")
        else:
            _store_backend = self._session_store_backend
            _store_db_path = self._session_store_db_path
            if _store_backend == "sqlite" and not _store_db_path:
                _store_db_path = str(self.workspace / "sessions.db")
            try:
                _session_store = create_session_store(
                    backend=_store_backend,
                    workspace=self.workspace,
                    db_path=_store_db_path,
                    dsn=self._session_store_dsn,
                )
                self.sessions = SessionManager(workspace=self.workspace, store=_session_store)
                logger.info(f"Session store: {_store_backend}")
            except Exception as exc:
                logger.warning(
                    f"Session store '{_store_backend}' init failed ({exc}), falling back to file"
                )
                self.sessions = SessionManager(self.workspace)

        # Memory store - semantic (memsearch) or file-based
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

        # Sub-agent manager - shares this agent's SessionManager so sub-agents
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
                parent_enable_skills=enable_skills,
                default_model=self._config.subagent.default_model,
                default_tool_profile=self._config.subagent.default_tool_profile,
                persist_runs=self._config.subagent.persist_runs,
                persist_file=self._config.subagent.persist_file,
                archive_after_minutes=self._config.subagent.archive_after_minutes,
                sweeper_interval_seconds=self._config.subagent.sweeper_interval_seconds,
                max_persistent_agents=self._config.subagent.max_persistent_agents,
            )
            self._owns_subagent_manager = True
        # Deploy built-in skills to the runtime workspace (idempotent)
        try:
            from spoon_bot.skills.builtin import ensure_builtin_skills

            _newly_installed = ensure_builtin_skills(self.workspace)
            if _newly_installed:
                logger.info(
                    f"Installed {len(_newly_installed)} built-in skill(s): "
                    f"{[p.name for p in _newly_installed]}"
                )
        except Exception as exc:
            logger.debug(f"Built-in skill deployment skipped: {exc}")

        # Skill paths: runtime workspace + bundled skills shipped with spoon-bot
        self._skill_paths = [self.workspace / "skills"]
        _bundled_skills = Path(__file__).resolve().parent.parent.parent / "workspace" / "skills"
        if _bundled_skills.is_dir() and _bundled_skills != self._skill_paths[0]:
            self._skill_paths.append(_bundled_skills)
        if self._user_skill_paths:
            self._skill_paths.extend(self._user_skill_paths)

        # Session
        self._session = self.sessions.get_or_create(self.session_key)

        # Inject memory context
        memory_context = self.memory.get_memory_context()
        if memory_context:
            self.context.set_memory_context(memory_context)

        # Register native tools
        self._register_native_tools()

        # Apply tool filter: explicit > profile > core default
        if self._enabled_tools_override is not None or self._tool_profile is not None:
            self.tools.set_tool_filter(
                enabled_tools=self._enabled_tools_override,
                tool_profile=self._tool_profile,
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

        # Skill tool names registered via _register_skill_tools() - for cleanup on reload
        self._skill_tool_names: set[str] = set()

        # Stop flag: set by stop_current_task(), cleared on next process() call
        self._stop_requested = False
        self._last_response_source = self._build_response_source()

        self._recent_invoked_skill_contexts: list[dict[str, Any]] = []

        # Conditional activation - file paths touched during this session.
        # Skills declaring ``paths`` frontmatter are dormant until a touched
        # file matches one of their patterns (mirrors Claude Code behaviour).
        self._touched_paths: set[str] = set()

        # Provider retry config (exponential backoff for transient LLM errors)
        self._retry_config = RetryConfig(
            max_retries=self._config.provider_max_retries,
            base_delay=self._config.provider_retry_base_delay,
            max_delay=self._config.provider_retry_max_delay,
            backoff_factor=self._config.provider_retry_backoff_factor,
        )

        active_count = len(self.tools)
        total_count = len(self.tools._tools)
        logger.info(
            f"AgentLoop created: model={model}, provider={provider}, "
            f"tools={active_count}/{total_count}, session={session_key}"
        )

    @staticmethod
    def _float_env(name: str, default: float, *, allow_zero: bool = False) -> float:
        """Read a float env override for runtime stream budgets."""
        value = os.environ.get(name)
        if value is None or not value.strip():
            return default
        try:
            parsed = float(value)
        except ValueError:
            logger.warning(f"Ignoring invalid {name}={value!r}; expected a number")
            return default
        if parsed > 0:
            return parsed
        if allow_zero and parsed == 0:
            return 0.0
        return default

    @staticmethod
    def _positive_runtime_budget(value: Any, default: float) -> float:
        """Resolve numeric runtime budget values without accepting mock objects."""
        if not isinstance(value, (int, float, str)):
            return default
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default

    @staticmethod
    def _int_env(name: str, default: int) -> int:
        """Read a positive integer env override for runtime stream budgets."""
        value = os.environ.get(name)
        if value is None or not value.strip():
            return default
        try:
            parsed = int(value)
        except ValueError:
            logger.warning(f"Ignoring invalid {name}={value!r}; expected an integer")
            return default
        return parsed if parsed > 0 else default

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

    @staticmethod
    def _align_chatbot_manager_provider(chatbot: Any, provider: str | None) -> None:
        """Keep spoon-core internal LLM calls on the explicit agent provider."""
        provider_name = str(provider or "").strip().lower()
        manager = getattr(chatbot, "llm_manager", None)
        if not provider_name or manager is None:
            return

        try:
            if hasattr(manager, "default_provider"):
                manager.default_provider = provider_name

            if os.environ.get("LLM_FALLBACK_CHAIN"):
                return

            set_fallback_chain = getattr(manager, "set_fallback_chain", None)
            if callable(set_fallback_chain):
                set_fallback_chain([provider_name])
            elif hasattr(manager, "fallback_chain"):
                manager.fallback_chain = [provider_name]
        except Exception as exc:
            logger.debug(f"Could not align LLM manager provider: {exc}")

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
        if hasattr(self.memory, "initialize"):
            await self.memory.initialize()

        # Build system prompt (spoon-bot context + available tool summaries)
        system_prompt = self._system_prompt or self.context.build_system_prompt()

        # Context budget hint (compact)
        system_prompt += f"\n\n[Context: {self.context_window:,} tokens - be concise.]\n"

        # Skills section (Openclaw pattern: XML metadata in system prompt)
        skills_xml = self._build_skills_for_prompt()
        if skills_xml:
            system_prompt += f"\n## Installed Skills\n{skills_xml}\n"
            system_prompt += (
                "\nUse this catalog as available context, not as a hidden router. "
                "When a skill is directly relevant, read its SKILL.md and follow "
                "the skill's own procedures. Treat those procedures, rules, and "
                "negative instructions as the execution contract for that task. "
                "Before running skill-provided primary commands, complete any "
                "Setup, Prerequisites, Install, or dependency steps declared in "
                "the skill. "
                "If the skill asks for optional non-essential user input during an "
                "already selected workflow, skip that optional branch and continue "
                "with the default/no-extra path instead of asking the user. "
                "If the skill tells you to decide autonomously or not ask the user, "
                "continue with the skill flow instead of asking for input. Otherwise "
                "use the normal tools.\n"
            )

        system_prompt += (
            "\n## Workflow\n"
            f"You have up to {self.max_iterations} steps. Minimize steps.\n\n"
            "1. Decide the next action from the latest user request and available context.\n"
            "2. If an installed skill or dynamically loadable tool is directly relevant, "
            "prefer the most specific available tool for that workflow. If the skill has "
            "no tool, `read_file` its SKILL.md path, then execute its procedure exactly, "
            "including its rules about when to ask the user and when to decide "
            "autonomously. Complete any Setup, Prerequisites, Install, or dependency "
            "steps before primary commands. When a skill defines conditional setup "
            "checks, run the check first and perform setup actions only when the "
            "documented condition is met. Do not read optional reference files unless "
            "the SKILL.md summary lacks enough information for the next action or a "
            "tool result points to a specific reference for recovery. When installing "
            "dependencies for a skill CLI, run the package manager in the directory "
            "that contains that CLI's package manifest; if the skill root has no "
            "package manifest but `cli/package.json` exists, install from the `cli` "
            "directory.\n"
            "3. Run commands from SKILL.md directly via shell only when there is no "
            "matching specialized tool. Do NOT write script files unless requested.\n"
            "4. When done, return the user-facing result in the format the latest user requested. "
            "Only summarize if the user explicitly asked for a summary.\n\n"
            "### Stop condition\n"
            "- Do not stop after setup when the latest user request also asked for follow-on execution. "
            "Setup, installation, dependency checks, status checks, and readiness reports are intermediate "
            "states unless they fully satisfy the newest request.\n"
            "- When the newest user request lists multiple ordered goals or follow-on actions, treat "
            "those listed actions as one authorized workflow for the current turn. Do not pause between "
            "listed stages just to ask for feedback, approval, or whether to proceed; continue until the "
            "workflow reaches the requested terminal outcome or a concrete blocker.\n"
            "- In an authorized multi-stage workflow, optional user input is not a blocker. If a skill "
            "offers an optional configuration, preference, bonus, referral, naming, or enhancement path "
            "but the core requested action can proceed without it, choose the default/no-extra path and "
            "continue instead of asking the user. Do not emit the optional-input prompt as a visible "
            "question first; just proceed with the selected default path.\n"
            "- If a tool result contains a blocking readiness/validation warning, treat it as an "
            "intermediate state and resolve it before giving a final answer.\n"
            "- Do not ask the user whether to run a safe next action that the newest request already asked "
            "you to run. Continue with the tool or skill contract. Ask only when the core action cannot "
            "be executed without a missing required value, or when the next action is destructive, "
            "irreversible, materially ambiguous, or outside the user's requested scope.\n\n"
            "- Do not poll an external system indefinitely. If repeated checks show the same pending, "
            "waiting, or not-ready state without material progress, report the current blocker and "
            "the evidence needed to resume later instead of continuing to wait, poll, or ask for "
            "generic feedback.\n\n"
            "### Rules\n"
            "- Do NOT re-read files already in context.\n"
            "- Memory, recent replies, and conversation history are stale hints. "
            "For current workspace, skill, account, balance, job, or external-system state, "
            "verify with tools before answering.\n"
            "- The latest user request is already in the active context. Do not call "
            "`search_history` to rediscover the current/latest request, to look up "
            "examples/templates for a new build, or before starting a new coding/execution "
            "task. Use `search_history` only when the user explicitly asks about earlier "
            "conversation facts, prior tool results, or a session compact says exact prior "
            "facts are needed.\n"
            "- If a command fails, analyze the error and retry with fixes.\n"
            "- For exact arithmetic, parsing, checksums, challenge answers, or values that "
            "will be submitted to an external tool/system, verify the value with a "
            "deterministic tool before submitting it. Do not rely on mental arithmetic "
            "for state-changing submissions.\n"
            "- If `write_file` reports an existing file and the task requires a "
            "generated whole-file replacement, retry that same write with "
            "`overwrite=true`; use `edit_file` only for targeted changes.\n"
            "- For long-running skill commands, keep the documented command in the "
            "foreground and usually omit the shell tool `timeout`; the tool applies "
            "its configured budget and can hand long commands to a managed background "
            "job. If a command is moved to the background, monitor that job only "
            "while its completion is still necessary for the selected workflow "
            "unit's contract-defined terminal outcome. If its output already proves "
            "that terminal outcome or blocker, answer from that evidence instead "
            "of monitoring solely because the process is active.\n"
            "- A continuation-only user message may resume one bounded unit only when "
            "same-session evidence identifies a clear prior workflow unit. If that "
            "unit is unfinished, finish that unit through its contract-defined terminal "
            "outcome or blocker. If that unit is complete and the same workflow has a "
            "clear next single-unit action, a continuation-only message can start at "
            "most one new unit. It never inherits older counts, multi-run targets, "
            "indefinite polling, repeated retries, or all remaining repetitions.\n"
            "- Do not perform the same irreversible, paid, or externally visible side-effect "
            "series repeatedly in one user turn beyond the selected workflow unit/count. "
            "If one unit reaches its contract-defined terminal outcome, report the status "
            "and the next safe continuation step instead of starting another similar "
            "external action.\n"
            "- After a requested workflow reaches a terminal outcome, do not end the final "
            "answer by asking whether to start a new paid, irreversible, externally visible, "
            "or repeated side-effect workflow. State the result and any safe read-only "
            "follow-up instead; a later continuation-only message authorizes at most one "
            "bounded unit of the same clear workflow.\n"
            "- Follow user instructions exactly - respect specific IDs, names, actions.\n"
            "- Final user-facing answers must match the newest user's language and summarize "
            "tool evidence in normal prose. Short continuation messages still define the "
            "answer language; do not inherit the language of older requests or tool evidence. "
            "Do not paste raw tool JSON, command transcripts, or internal planning text unless "
            "the user explicitly asks for raw logs or JSON.\n"
            "- Minimize sensitive output. Do not include private keys, password paths, "
            "keystore paths, secret availability flags, raw credentials, tokens, or "
            "other authentication material in user-facing answers unless the newest "
            "request explicitly asks for that specific secret-bearing field. When a "
            "user asks for a public identifier such as a wallet address, answer only "
            "the requested public identifier and any directly relevant non-sensitive "
            "context.\n"
            "- When creating a public browser app with frontend plus API/WebSocket/backend, "
            "prefer serving all browser-required pieces from one local service before exposing it. "
            "If multiple services are necessary, expose and verify every browser-required endpoint; "
            "do not finalize with browser code pointing to localhost, loopback, or unexposed ports.\n"
            "- In sandbox/runtime environments, when you create or start a browser-facing local "
            "service, WebSocket service, API, frontend, backend, or preview app, use the "
            "`service_expose` tool to manage the background process and Cloudflare exposure "
            "unless the user explicitly asks for local-only access. Do not use one-off shell "
            "backgrounding as the final service manager when `service_expose` can do the same "
            "job, and do not finalize with only localhost, loopback, or 0.0.0.0 access.\n"
            "- Before finalizing a service task, use your own understanding of the newest "
            "user request to decide whether public access is required. Runtime verifiers only "
            "trust structured tool evidence; they will not infer this semantic requirement "
            "from prompt keywords or URL-shaped text for you.\n"
            "- Before starting or exposing a generated service, complete the smallest local preflight "
            "that proves it can launch: install declared runtime dependencies, run the relevant "
            "syntax/build check for the entrypoint when available, and fix failures before calling "
            "service exposure tools.\n"
            "- Use web search when the task needs live external facts or installed skills/tools are insufficient.\n"
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
        if (
            os.getenv("SPOON_BOT_ENABLE_ORCHESTRATION_PROMPT", "").strip().lower()
            in {"1", "true", "yes", "on"}
            and "spawn" in self.tools.get_active_tools()
        ):
            roles_block = format_roles_for_prompt()
            system_prompt += (
                "\n\n## Multi-Agent Orchestration\n\n"
                "You are an **Orchestrator**. For complex tasks (e.g. 'build a user "
                "management system', 'implement a REST API with frontend'), decompose "
                "the work and delegate to specialised sub-agents using the `spawn` tool.\n\n"
                "**You decide which agents to use and in what order** - there is no fixed "
                "pipeline. Use your judgement based on the task requirements.\n\n"
                "### Available specialised agent roles\n"
                f"{roles_block}\n\n"
                "### When to use sub-agents\n"
                "- Complex tasks requiring multiple specialised skills\n"
                "- Tasks that benefit from sequential specialisation "
                "(e.g. plan -> implement -> review)\n"
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
                "- Each sub-agent starts with a **fresh context** - provide all necessary "
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
            enable_short_term_memory=False,
        )
        AgentLoop._align_chatbot_manager_provider(self._chatbot, self.provider)

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

        # Create MCP tools - expand each server into individual tools (#5)
        await self._init_mcp_tools()

        # Create SkillManager if enabled - BEFORE building active_tools
        # so that skill tools are included in the agent's ToolManager.
        if self._enable_skills:
            import inspect

            _sm_sig = inspect.signature(SkillManager.__init__)
            _sm_kwargs: dict[str, Any] = {
                "skill_paths": [str(p) for p in self._skill_manager_discovery_paths()],
                "llm": self._chatbot,
                "auto_discover": True,
            }
            if "include_default_paths" in _sm_sig.parameters:
                _sm_kwargs["include_default_paths"] = False
            self._skill_manager = SkillManager(**_sm_kwargs)
            # Load skill tools into the ToolRegistry (inactive by default)
            self._register_skill_tools()
            # Only auto-activate ordinary skill tools when no skills are shown in
            # the prompt. Core skill-management tools remain governed by the
            # selected tool profile.
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
            agent_kwargs["auto_trigger_skills"] = False
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

        # Keep the agent's per-step timeout aligned with the effective shell ceiling
        # so long-running commands are not cancelled prematurely by the outer loop.
        effective_ceiling = max(
            self.shell_timeout, getattr(self, "shell_max_timeout", DEFAULT_SHELL_MAX_TIMEOUT)
        )
        self._agent._default_timeout = max(300.0, float(effective_ceiling))

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
        # Shell tool - allow_chaining + allow_substitution lets the agent
        # compose multi-step ops and use $(), ${}, and backtick expressions.
        self.tools.register(
            ShellTool(
                timeout=self.shell_timeout,
                max_timeout=getattr(self, "shell_max_timeout", DEFAULT_SHELL_MAX_TIMEOUT),
                max_output=self.max_output,
                working_dir=str(self.workspace),
                allow_chaining=True,
                allow_substitution=True,
                yolo_mode=bool(getattr(self, "yolo_mode", False)),
            )
        )

        # Filesystem tools - allow reads from the user home directory so that
        # skill-managed data (e.g. ~/.agent-wallet, ~/.spoon-bot/skills) is
        # accessible.  The PathValidator blocklist still blocks truly sensitive
        # paths (.ssh, .aws, etc.).
        #
        # In YOLO mode the workspace IS the user's directory, so we add its
        # parents as extra read paths to let the agent navigate freely.
        _extra_read: list[Path] = [Path.home()]
        if getattr(self, "yolo_mode", False):
            _extra_read.extend(p for p in self.workspace.parents if p != Path.home())

        _file_tools = [
            ReadFileTool(
                workspace=self.workspace, additional_read_paths=_extra_read, max_output=15000
            ),
            WriteFileTool(workspace=self.workspace),
            EditFileTool(workspace=self.workspace),
            ListDirTool(workspace=self.workspace, additional_read_paths=_extra_read),
            GrepTool(workspace=self.workspace),
        ]
        for ft in _file_tools:
            ft._path_touch_callback = lambda p: self.record_touched_paths(p)
            self.tools.register(ft)

        # Self-management tools
        self_config_tool = SelfConfigTool()
        self_config_tool.set_agent_loop(self)
        self.tools.register(self_config_tool)
        memory_tool = MemoryManagementTool()
        memory_tool.set_memory_store(self.memory)
        self.tools.register(memory_tool)

        # Self-upgrade tool - with agent loop reference for hot-reload
        upgrade_tool = SelfUpgradeTool(workspace=self.workspace)
        upgrade_tool.set_agent_loop(self)
        self.tools.register(upgrade_tool)

        # Dynamic tool activation - lets the LLM load tools on demand
        self.tools.register(
            ActivateToolTool(
                activate_fn=self.add_tool,
                list_inactive_fn=lambda: [
                    {"name": t.name, "description": t.description}
                    for t in self.tools.get_inactive_tools().values()
                ],
                list_active_fn=lambda: [
                    {"name": t.name, "description": t.description}
                    for t in self.tools.get_active_tools().values()
                ],
                tool_status_fn=self._tool_activation_status,
            )
        )

        cron_tool = CronTool()
        cron_tool.set_agent_loop(self)
        self.tools.register(cron_tool)

        # Sub-agent tool - replaces the old placeholder SpawnTool
        spawn_tool = SubagentTool(manager=getattr(self, "_subagent_manager", None))
        self.tools.register(spawn_tool)

        if hasattr(self, "sessions"):

            def _current_session_key() -> str | None:
                sess = getattr(self, "_session", None)
                return sess.session_key if sess is not None else None

            current_session = getattr(self, "_session", None)
            self._history_search_tool = SearchHistoryTool(
                self.sessions,
                default_session_key=(
                    current_session.session_key if current_session is not None else None
                ),
                session_key_resolver=_current_session_key,
            )
            self.tools.register(self._history_search_tool)

        # Web tools
        self.tools.register(WebSearchTool())
        self.tools.register(WebFetchTool())

        # Wallet and Web3 tools are registered but remain inactive under the
        # default core profile. The agent can load them when a wallet task needs
        # balance checks, signing, transfers, swaps, or contract calls.
        self.tools.register(WalletTool())
        self.tools.register(BalanceCheckTool())
        self.tools.register(TransferTool())
        self.tools.register(SwapTool())
        self.tools.register(ContractCallTool())

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
                            bridge = SkillToolBridge(base_tool, workspace=self.workspace)
                            self.tools.register(bridge)
                            self._skill_tool_names.add(base_tool.name)
                            if self._skill_tool_default_active(skill_dir):
                                self.tools.activate_tool(base_tool.name)
                except Exception as exc:
                    logger.debug(f"Skill tools from {skill_dir.name}: {exc}")

        if self._skill_tool_names:
            logger.info(
                f"Registered {len(self._skill_tool_names)} skill tool(s): "
                f"{sorted(self._skill_tool_names)}"
            )

    @staticmethod
    def _skill_tool_default_active(skill_dir: Path) -> bool:
        """Return whether a skill asks its tool wrapper to be active by default."""
        skill_md = skill_dir / "SKILL.md"
        try:
            lines = skill_md.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return False
        if not lines or lines[0].strip() != "---":
            return False
        for raw_line in lines[1:80]:
            line = raw_line.strip()
            if line == "---":
                return False
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            if key.strip().casefold() != "default_active":
                continue
            return value.strip().casefold() in {"1", "true", "yes", "on"}
        return False

    async def _init_mcp_tools(self) -> None:
        """Discover and create MCP tools from ``self._mcp_config``.

        Populates ``self._mcp_tools``. Safe to call multiple times - the
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
            "skill_paths": [str(p) for p in self._skill_manager_discovery_paths()],
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
        logger.info(
            f"Skills reloaded: {len(old_skills)} -> {len(new_skills)} (added={added}, removed={removed})"
        )
        return {"before": old_skills, "after": new_skills, "added": added, "removed": removed}

    async def reload_mcp(
        self, new_config: dict[str, dict[str, Any]] | None = None
    ) -> dict[str, Any]:
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
                        logger.debug(
                            f"MCP cleanup ({method}) for '{getattr(mcp_tool, 'name', '?')}': {exc}"
                        )
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

    def _runtime_compaction_trigger_budget(self) -> int:
        """Return the runtime token budget that should trigger preflight compaction."""
        safety_margin = max(8_000, int(self.context_window * 0.05))
        return max(8_000, self.context_window - safety_margin)

    async def process(
        self,
        message: str,
        media: list[str] | None = None,
        attachments: list[dict[str, Any]] | None = None,
        session_key: str | None = None,
        channel: str | None = None,
        metadata: dict[str, Any] | None = None,
        reply_to: str | None = None,
        reasoning_effort: str | None = None,
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
            if getattr(self, "_history_search_tool", None) is not None:
                try:
                    self._history_search_tool.set_default_session_key(session_key)
                except Exception:
                    pass
            logger.debug(f"Switched to session: {session_key}")

        current_session_key = getattr(self, "session_key", "default")
        self.set_subagent_context(
            session_key=current_session_key,
            channel=channel,
            metadata=metadata,
            reply_to=reply_to,
        )
        logger.info(f"Processing message: {message[:100]}...")
        AgentLoop._reset_runtime_notices(self)
        effective_reasoning_effort = reasoning_effort or getattr(self, "reasoning_effort", None)
        await self._install_skill_zip_attachments(attachments or [])

        # Refresh memory context
        try:
            memory_context = self.memory.get_memory_context()
            if memory_context:
                self.context.set_memory_context(memory_context)
        except Exception as e:
            logger.warning(f"Failed to load memory context: {e}")

        # Trim and inject persisted history into runtime memory
        await self._prepare_request_context(message)

        self._prepare_agent_for_new_turn()
        authoritative_message = message

        runtime_user_text = self._add_current_turn_skill_zip_context(
            self._build_current_turn_runtime_user_text(authoritative_message)
        )
        runtime_message = self._build_runtime_message_content(
            "user",
            runtime_user_text,
            media=media,
            attachments=attachments,
        )
        if isinstance(runtime_message, str):
            message = runtime_message

        AgentLoop._persist_user_turn_to_session(
            self,
            authoritative_message,
            media=media,
            attachments=attachments,
        )

        original_system_prompt: str | None = None
        original_base_system_prompt: object = _MISSING

        _base_prompt = self._select_next_step_prompt(
            authoritative_message,
            thinking=False,
        )
        self._agent.next_step_prompt = _base_prompt

        self._install_anti_loop_tracker(_base_prompt)
        original_system_prompt, original_base_system_prompt = (
            AgentLoop._apply_request_context_to_system_prompt(
                self,
                authoritative_message,
                thinking=False,
            )
        )
        await self._agent.add_message("user", runtime_message)
        self._normalize_runtime_memory_before_run("process")
        _pre_turn_memory_index = self._runtime_memory_snapshot_index()
        self._active_turn_memory_start_index = _pre_turn_memory_index

        retry_runner = AgentLoop._resolve_retry_runner(self)
        execution_ledger = ExecutionLedger(
            owner=self._current_tool_owner_key(),
            workspace=str(getattr(self, "workspace", "") or ""),
            session_id=str(getattr(self, "session_key", "") or ""),
            turn_id=uuid.uuid4().hex,
            user_request=authoritative_message,
        )
        self._active_execution_ledger = execution_ledger
        ledger_manager = bind_execution_ledger(execution_ledger)
        ledger_manager.__enter__()

        try:
            run_kwargs: dict[str, Any] = {}
            if effective_reasoning_effort and self._callable_accepts_kwarg(
                self._agent.run, "reasoning_effort"
            ):
                run_kwargs["reasoning_effort"] = effective_reasoning_effort
            request_execution_hints = self._build_request_execution_hints(authoritative_message)
            with (
                bind_request_execution_hints(request_execution_hints),
                track_tool_invocations(),
            ):
                result = await AgentLoop._run_agent_with_context_overflow_recovery(
                    self,
                    label="process",
                    retry_runner=retry_runner,
                    **run_kwargs,
                )

            logger.debug(f"Agent result type: {type(result)}")
            if hasattr(result, "content"):
                logger.info(f"Agent result.content (first 300): {str(result.content)[:300]}")

            final_content = AgentLoop._extract_run_result_text(result)

            if final_content.strip() in ("No results", ""):
                logger.warning(
                    "Agent returned empty/no-results attempting to extract "
                    "content from agent memory"
                )
                _extracted = self._extract_last_assistant_content()
                if _extracted:
                    final_content = _extracted

            if AgentLoop._looks_like_pseudo_tool_call_text(final_content):
                logger.warning(
                    "Agent returned tool-call-shaped Markdown instead of actual "
                    "tool calls; retrying once with an internal repair prompt."
                )
                AgentLoop._drop_pseudo_tool_call_assistant_messages(
                    self,
                    _pre_turn_memory_index,
                )
                AgentLoop._drain_agent_output_queue(self)
                self._reset_agent_state_for_retry()
                repair_prompt = AgentLoop._build_pseudo_tool_call_repair_prompt(
                    authoritative_message,
                    final_content,
                )
                await self._agent.add_message("user", repair_prompt)
                self._agent.next_step_prompt = repair_prompt
                with (
                    bind_request_execution_hints(request_execution_hints),
                    track_tool_invocations(),
                ):
                    result = await AgentLoop._run_agent_with_context_overflow_recovery(
                        self,
                        label="process_tool_call_repair",
                        retry_runner=retry_runner,
                        **run_kwargs,
                    )
                final_content = AgentLoop._extract_run_result_text(result)
                if final_content.strip() in ("No results", ""):
                    _extracted = self._extract_last_assistant_content()
                    if _extracted:
                        final_content = _extracted

            tool_result_events = self._collect_runtime_tool_result_events_from_memory(
                _pre_turn_memory_index
            )
            if AgentLoop._tool_events_have_history_search_budget(tool_result_events):
                final_content = await self._run_process_history_search_budget_recovery(
                    authoritative_message=authoritative_message,
                    request_execution_hints=request_execution_hints,
                    tool_result_events=tool_result_events,
                    retry_runner=retry_runner,
                    run_kwargs=run_kwargs,
                    label="process_history_search_budget_recovery",
                )
                tool_result_events = self._collect_runtime_tool_result_events_from_memory(
                    _pre_turn_memory_index
                )

            if should_run_skill_contract_check(
                tool_result_events
            ) and AgentLoop._tool_events_have_repeated_read_guardrail(tool_result_events):
                final_content = await self._run_process_repeated_read_recovery(
                    authoritative_message=authoritative_message,
                    request_execution_hints=request_execution_hints,
                    retry_runner=retry_runner,
                    run_kwargs=run_kwargs,
                )
                tool_result_events = self._collect_runtime_tool_result_events_from_memory(
                    _pre_turn_memory_index
                )

            if should_run_skill_contract_check(tool_result_events):
                (
                    final_content,
                    tool_result_events,
                ) = await self._continue_skill_contract_until_terminal(
                    authoritative_message=authoritative_message,
                    request_execution_hints=request_execution_hints,
                    final_content=final_content,
                    tool_result_events=tool_result_events,
                    retry_runner=retry_runner,
                    run_kwargs=run_kwargs,
                    memory_start_index=_pre_turn_memory_index,
                    label="process_skill_contract_continuation",
                )

            if tool_result_events:
                final_content, tool_result_events = await self._continue_task_until_terminal(
                    authoritative_message=authoritative_message,
                    request_execution_hints=request_execution_hints,
                    final_content=final_content,
                    tool_result_events=tool_result_events,
                    retry_runner=retry_runner,
                    run_kwargs=run_kwargs,
                    memory_start_index=_pre_turn_memory_index,
                    label="process_task_continuation",
                )

            if (
                tool_result_events
                and should_run_skill_contract_check(tool_result_events)
                and skill_contract_has_progress(tool_result_events)
                and not latest_tool_event_has_user_summary_marker(tool_result_events)
                and not latest_tool_event_from_skill_continuation(tool_result_events)
            ):
                final_content = await AgentLoop._synthesize_final_answer_from_tool_events(
                    self,
                    tool_result_events,
                    user_message=authoritative_message,
                    fallback_text=build_user_facing_tool_event_answer(
                        tool_result_events,
                        user_message=authoritative_message,
                    ),
                )

            if any(
                AgentLoop._is_tool_loop_suppression_event(event) for event in tool_result_events
            ):
                fallback_content = AgentLoop._build_tool_loop_fallback_response(
                    tool_result_events,
                    reason="tool_suppression",
                    user_message=authoritative_message,
                )
                final_content = await AgentLoop._synthesize_final_answer_from_tool_events(
                    self,
                    tool_result_events,
                    incomplete=True,
                    user_message=authoritative_message,
                    fallback_text=fallback_content,
                )
            elif should_run_skill_contract_check(
                tool_result_events
            ) and latest_tool_event_has_user_summary_marker(tool_result_events):
                final_content = await AgentLoop._synthesize_final_answer_from_tool_events(
                    self,
                    tool_result_events,
                    user_message=authoritative_message,
                    fallback_text=build_user_facing_tool_event_answer(
                        tool_result_events,
                        user_message=authoritative_message,
                    ),
                )
            elif tool_result_events and (
                AgentLoop._looks_like_raw_tool_transcript_leak(final_content)
            ):
                final_content = await AgentLoop._synthesize_final_answer_from_tool_events(
                    self,
                    tool_result_events,
                    incomplete=tool_events_need_more_evidence(tool_result_events),
                    user_message=authoritative_message,
                    fallback_text=build_user_facing_tool_event_answer(
                        tool_result_events,
                        incomplete=tool_events_need_more_evidence(tool_result_events),
                        user_message=authoritative_message,
                    ),
                )
            elif tool_result_events and AgentLoop._final_answer_script_mismatch(
                authoritative_message,
                final_content,
            ):
                final_content = await AgentLoop._synthesize_final_answer_from_tool_events(
                    self,
                    tool_result_events,
                    incomplete=tool_events_need_more_evidence(tool_result_events),
                    user_message=authoritative_message,
                    fallback_text=build_user_facing_tool_event_answer(
                        tool_result_events,
                        incomplete=tool_events_need_more_evidence(tool_result_events),
                        user_message=authoritative_message,
                    ),
                )
            elif tool_result_events and (
                bool(request_execution_hints.get("current_session_fact_check_required"))
                or bool(request_execution_hints.get("session_evidence_synthesis_preferred"))
            ):
                synthesis_events = self._session_evidence_synthesis_events(
                    tool_result_events,
                    user_message=authoritative_message,
                )
                final_content = await AgentLoop._synthesize_final_answer_from_tool_events(
                    self,
                    synthesis_events,
                    incomplete=tool_events_need_more_evidence(synthesis_events),
                    user_message=authoritative_message,
                    fallback_text=build_user_facing_tool_event_answer(
                        synthesis_events,
                        incomplete=tool_events_need_more_evidence(synthesis_events),
                        user_message=authoritative_message,
                    ),
                )
            elif tool_result_events and (
                bool(request_execution_hints.get("current_session_fact_check_required"))
                or bool(request_execution_hints.get("session_evidence_synthesis_preferred"))
            ):
                synthesis_events = self._session_evidence_synthesis_events(
                    tool_result_events,
                    user_message=authoritative_message,
                )
                final_content = await AgentLoop._synthesize_final_answer_from_tool_events(
                    self,
                    synthesis_events,
                    incomplete=tool_events_need_more_evidence(synthesis_events),
                    user_message=authoritative_message,
                    fallback_text=build_user_facing_tool_event_answer(
                        synthesis_events,
                        incomplete=tool_events_need_more_evidence(synthesis_events),
                        user_message=authoritative_message,
                    ),
                )
                thinking_content = None
            elif (
                tool_result_events
                and should_run_skill_contract_check(tool_result_events)
                and not skill_contract_has_progress(tool_result_events)
            ):
                final_content = await AgentLoop._synthesize_final_answer_from_tool_events(
                    self,
                    tool_result_events,
                    incomplete=tool_events_need_more_evidence(tool_result_events),
                    user_message=authoritative_message,
                    fallback_text=build_user_facing_tool_event_answer(
                        tool_result_events,
                        incomplete=tool_events_need_more_evidence(tool_result_events),
                        user_message=authoritative_message,
                    ),
                )
            final_content = AgentLoop._finalize_response_content(
                self,
                authoritative_message,
                final_content,
                turn_memory_start_index=_pre_turn_memory_index,
            )
        except asyncio.CancelledError:
            self._persist_cancelled_turn_context(start_index=_pre_turn_memory_index)
            raise
        except Exception as e:
            import traceback

            logger.error(f"Agent run error: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            AgentLoop._persist_failed_turn_context(
                self,
                label="process",
                reason=e,
                start_index=_pre_turn_memory_index,
            )
            raise
        finally:
            try:
                persist_execution_ledger(execution_ledger)
            except Exception as exc:
                logger.debug(f"Execution ledger persistence skipped: {exc}")
            try:
                ledger_manager.__exit__(None, None, None)
            except Exception:
                pass
            if getattr(self, "_active_execution_ledger", None) is execution_ledger:
                self._active_execution_ledger = None
            if getattr(self, "_active_turn_memory_start_index", None) == _pre_turn_memory_index:
                self._active_turn_memory_start_index = None
            # Always ensure agent is back in IDLE state after processing
            self._restore_agent_think()
            AgentLoop._restore_request_context_system_prompt(
                self,
                original_system_prompt,
                original_base_system_prompt,
            )
            if hasattr(self._agent, "state") and self._agent.state != AgentState.IDLE:
                logger.warning(
                    f"Post-run cleanup: resetting agent from {self._agent.state} to IDLE"
                )
                self._agent.state = AgentState.IDLE
                self._agent.current_step = 0

        try:
            self._merge_turn_invoked_skills_from_runtime(_pre_turn_memory_index)
            AgentLoop._mark_latest_user_turn_state(
                self,
                _TURN_STATE_COMPLETED,
            )
            self._persist_turn_tool_trace(_pre_turn_memory_index)
            self._session.add_message(
                "assistant",
                final_content,
                **AgentLoop._assistant_session_save_kwargs(final_content),
            )
            AgentLoop._persist_session_if_possible(self)
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
        attachments: list[dict[str, Any]] | None = None,
        thinking: bool = False,
        channel: str | None = None,
        metadata: dict[str, Any] | None = None,
        reply_to: str | None = None,
        reasoning_effort: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream a response with typed chunks.

        Args:
            message: User message.
            media: Optional media files.
            thinking: Whether to include thinking output.

        Yields:
            Dicts with keys:
              type:     "content" | "thinking" | "tool_call" | "tool_result" | "notice" | "done"
              delta:    Incremental text (may be empty for non-text events)
              metadata: Extra context (tool name, args, step number, etc.)
              source:   Machine-readable producer metadata for this output
        """
        if not self._initialized:
            await self.initialize()

        current_source = AgentLoop._set_last_response_source(self)
        logger.info(f"Streaming message: {message[:100]}...")
        self._reset_reasoning_capture()
        AgentLoop._reset_runtime_notices(self)
        effective_reasoning_effort = reasoning_effort or getattr(self, "reasoning_effort", None)
        authoritative_message = message
        buffer_stream_content = False
        await self._install_skill_zip_attachments(attachments or [])

        current_session_key = getattr(self, "session_key", "default")
        self.set_subagent_context(
            session_key=current_session_key,
            channel=channel,
            metadata=metadata,
            reply_to=reply_to,
        )

        # Refresh memory context
        bg_task: asyncio.Task | None = None

        try:
            memory_context = self.memory.get_memory_context()
            if memory_context:
                self.context.set_memory_context(memory_context)
        except Exception as e:
            logger.warning(f"Failed to load memory context: {e}")

        full_content = ""
        pre_tool_scratchpad_buffer = ""
        pre_tool_scratchpad_events: list[dict[str, Any]] = []
        post_tool_content_buffer = ""
        post_tool_content_events: list[dict[str, Any]] = []
        saw_tool_call = False
        saw_content_after_tool_call = False
        streamed_content_after_latest_tool_call = False
        pseudo_tool_repair_attempted = False
        stream_completed = False
        stream_cancelled = False
        bg_task: asyncio.Task[None] | None = None
        original_system_prompt: str | None = None
        original_base_system_prompt: object = _MISSING
        tool_output_capture_scope: str | None = None
        capture_manager = None
        pending_fallback_content_emit = False
        pending_fallback_reason: str | None = None
        pending_fallback_delta = ""
        tool_loop_fallback_active = False
        initial_content_passthrough_started = False
        post_tool_content_passthrough_started = False
        emitted_content_text = ""
        stream_error_reason: BaseException | str | None = None
        interrupted_assistant_reply_persisted = False
        execution_ledger: ExecutionLedger | None = None
        ledger_manager = None
        _pre_turn_memory_index: int | None = None
        leaked_tool_protocol_buffer = ""
        leaked_tool_protocol_detected = False
        leaked_tool_protocol_probe = ""

        def _persist_interrupted_stream_reply(reason: str = "task_cancelled") -> None:
            nonlocal interrupted_assistant_reply_persisted
            if interrupted_assistant_reply_persisted:
                return
            interrupted_assistant_reply_persisted = AgentLoop._persist_interrupted_assistant_reply(
                self,
                full_content,
                reason=reason,
            )

        # Trim and inject persisted history into runtime memory
        await self._prepare_request_context(message)
        self._prepare_agent_for_new_turn()
        authoritative_message = message

        runtime_user_text = self._add_current_turn_skill_zip_context(
            self._build_current_turn_runtime_user_text(authoritative_message)
        )
        runtime_message = self._build_runtime_message_content(
            "user",
            runtime_user_text,
            media=media,
            attachments=attachments,
        )
        if isinstance(runtime_message, str):
            message = runtime_message
        AgentLoop._persist_user_turn_to_session(
            self,
            authoritative_message,
            media=media,
            attachments=attachments,
        )

        try:
            capture_manager = capture_tool_outputs()
            tool_output_capture_scope = capture_manager.__enter__()
            _base_prompt = self._select_next_step_prompt(authoritative_message, thinking=thinking)
            original_system_prompt, original_base_system_prompt = (
                AgentLoop._apply_request_context_to_system_prompt(
                    self,
                    authoritative_message,
                    thinking=thinking,
                )
            )
            self._agent.next_step_prompt = _base_prompt
            self._install_anti_loop_tracker(_base_prompt)

            # ------------------------------------------------------------------
            # Streaming uses the spoon-core run+stream pattern:
            #   1. Clear task_done + drain output_queue
            #   2. Start run(message) in background - sets task_done on finish
            #   3. Read chunks from output_queue until task_done AND queue empty
            # ------------------------------------------------------------------

            # 1. Reset streaming state
            self._agent.task_done.clear()
            while not self._agent.output_queue.empty():
                try:
                    await asyncio.wait_for(self._agent.output_queue.get(), timeout=0.1)
                except (asyncio.TimeoutError, Exception):
                    break

            await self._agent.add_message("user", runtime_message)
            self._normalize_runtime_memory_before_run("stream")
            _pre_turn_memory_index = self._runtime_memory_snapshot_index()
            self._active_turn_memory_start_index = _pre_turn_memory_index
            execution_ledger = ExecutionLedger(
                owner=self._current_tool_owner_key(),
                workspace=str(getattr(self, "workspace", "") or ""),
                session_id=str(getattr(self, "session_key", "") or ""),
                turn_id=uuid.uuid4().hex,
                user_request=authoritative_message,
            )
            self._active_execution_ledger = execution_ledger
            ledger_manager = bind_execution_ledger(execution_ledger)
            ledger_manager.__enter__()
            stream_tool_result_index = len(AgentLoop._get_runtime_memory_messages(self))
            emitted_tool_result_ids: set[str] = set()
            tool_call_arguments_by_id: dict[str, str] = {}
            recent_tool_result_events: list[dict[str, Any]] = []
            all_tool_result_events: list[dict[str, Any]] = []
            stream_tool_result_count = 0
            stream_tool_call_count = 0
            read_file_result_argument_counts: dict[str, int] = {}
            read_file_result_generations: dict[str, int] = {}
            read_context_generation = 0
            active_tool_call_names_by_id: dict[str, str] = {}
            last_tool_call_names: tuple[str, ...] = ()
            non_read_tool_call_count = 0
            repeated_read_recovery_attempted = False
            repeated_read_guardrail_seen = False
            history_search_budget_recovery_attempted = False
            history_search_budget_seen = False
            skill_contract_continuation_attempted = False
            skill_contract_continuation_attempts = 0
            post_tool_result_silence_recovery_seen = False
            turn_tool_invocation_state: dict[str, Any] | None = None
            # 2. Start run() in background
            run_result_text = ""
            provider_silence_retry_count = 0
            max_provider_silence_retries = max(
                0,
                int(getattr(self, "provider_silence_retries", 1) or 0),
            )
            retry_reasoning_effort: str | None = None
            request_execution_hints = self._build_request_execution_hints(authoritative_message)

            async def _run_and_signal() -> None:
                nonlocal run_result_text, stream_error_reason, turn_tool_invocation_state
                try:
                    retry_runner = AgentLoop._resolve_retry_runner(self)
                    run_kwargs: dict[str, Any] = {}
                    if thinking and self._callable_accepts_kwarg(self._agent.run, "thinking"):
                        run_kwargs["thinking"] = True
                    run_reasoning_effort = retry_reasoning_effort or effective_reasoning_effort
                    if run_reasoning_effort and self._callable_accepts_kwarg(
                        self._agent.run, "reasoning_effort"
                    ):
                        run_kwargs["reasoning_effort"] = run_reasoning_effort

                    def _drain_queue() -> None:
                        while not self._agent.output_queue.empty():
                            self._agent.output_queue.get_nowait()

                    with (
                        bind_request_execution_hints(request_execution_hints),
                        track_tool_invocations(
                            initial_state=turn_tool_invocation_state,
                        ) as invocation_state,
                    ):
                        turn_tool_invocation_state = invocation_state
                        result = await AgentLoop._run_agent_with_context_overflow_recovery(
                            self,
                            label="stream",
                            retry_runner=retry_runner,
                            pre_overflow_retry_cleanup=_drain_queue,
                            **run_kwargs,
                        )
                    if hasattr(result, "content"):
                        run_result_text = result.content or ""
                    elif isinstance(result, str):
                        run_result_text = result
                    elif result is not None:
                        run_result_text = str(result)
                except Exception as exc:
                    stream_error_reason = exc
                    logger.error(f"Background agent run failed: {exc}")
                    friendly_error = user_friendly_error(exc)
                    try:
                        await self._agent.output_queue.put(
                            {
                                "type": "error",
                                "delta": friendly_error,
                                "metadata": {
                                    "error": friendly_error,
                                    "message": friendly_error,
                                    "code": type(exc).__name__,
                                    "error_code": type(exc).__name__,
                                    "reason": "agent_run_failed",
                                },
                            }
                        )
                    except Exception:
                        pass
                finally:
                    self._agent.task_done.set()

            logger.debug(f"Creating bg task, agent state={self._agent.state}")
            with (
                bind_tool_owner(self._current_tool_owner_key()),
                bind_tool_workspace(str(getattr(self, "workspace", "") or "")),
            ):
                bg_task = asyncio.create_task(_run_and_signal())

            # Force a yield to allow the background task to start
            await asyncio.sleep(0)

            # 3. Read output chunks (mirrors fixed BaseAgent.stream logic)
            oq = self._agent.output_queue
            td = self._agent.task_done
            logger.debug(
                f"output_queue type={type(oq).__name__}, task_done type={type(td).__name__}"
            )
            chunk_count = 0
            stream_segment_index = -1
            current_stream_segment_type: str | None = None

            def _decorate_stream_event(event: dict[str, Any]) -> dict[str, Any]:
                nonlocal stream_segment_index, current_stream_segment_type

                event_type = str(event.get("type") or "")
                if event_type not in {"thinking", "content", "tool_call", "tool_result", "notice"}:
                    return event

                metadata = dict(event.get("metadata") or {})
                starts_new_segment = (
                    event_type in {"tool_call", "tool_result", "notice"}
                    or current_stream_segment_type != event_type
                )
                if starts_new_segment:
                    stream_segment_index += 1
                metadata.setdefault("segment_index", stream_segment_index)
                metadata.setdefault("segment_start", starts_new_segment)
                metadata.setdefault("segment_type", event_type)

                if event_type in {"tool_call", "tool_result", "notice"}:
                    current_stream_segment_type = None
                else:
                    current_stream_segment_type = event_type

                if event_type == "tool_result":
                    # The matching tool_call event already carries the public
                    # input preview. Re-sending arguments on result chunks
                    # makes delta-based frontends merge nested strings as if
                    # they were incremental text, which bloats long sessions
                    # and can corrupt digest fields. Internal verifier events
                    # keep their own metadata before this presentation step.
                    metadata.pop("arguments", None)
                    metadata.pop("input", None)

                return {
                    **event,
                    "metadata": metadata,
                    "source": event.get("source", current_source),
                }

            def _drain_runtime_notice_events() -> list[dict[str, Any]]:
                events: list[dict[str, Any]] = []
                for notice in AgentLoop._drain_runtime_notices(self):
                    event = AgentLoop._runtime_notice_to_stream_event(notice)
                    if event is not None:
                        events.append(_decorate_stream_event(event))
                return events

            for event in _drain_runtime_notice_events():
                yield event

            logger.debug(
                f"Entering stream loop: td={td.is_set()}, qempty={oq.empty()}, qsize={oq.qsize()}"
            )
            stream_started_at = time.monotonic()
            last_stream_activity_at = stream_started_at
            last_stream_heartbeat_at = 0.0
            stream_heartbeat_count = 0
            last_tool_progress_at = stream_started_at
            last_tool_progress_kind: str | None = None
            provider_silence_timeout = float(
                getattr(self, "provider_silence_timeout", DEFAULT_PROVIDER_SILENCE_TIMEOUT)
                or DEFAULT_PROVIDER_SILENCE_TIMEOUT
            )
            provider_silence_enabled = provider_silence_timeout > 0
            provider_total_timeout = float(
                getattr(self, "provider_total_timeout", DEFAULT_PROVIDER_TOTAL_TIMEOUT)
            )
            tool_followup_timeout = max(
                0.1,
                float(
                    getattr(self, "tool_followup_timeout", DEFAULT_TOOL_FOLLOWUP_TIMEOUT)
                    or DEFAULT_TOOL_FOLLOWUP_TIMEOUT
                ),
            )
            shell_foreground_timeout = AgentLoop._positive_runtime_budget(
                getattr(self, "shell_timeout", None),
                DEFAULT_SHELL_TIMEOUT,
            )
            shell_max_timeout = AgentLoop._positive_runtime_budget(
                getattr(self, "shell_max_timeout", None),
                DEFAULT_SHELL_MAX_TIMEOUT,
            )
            shell_handoff_timeout = AgentLoop._float_env(
                "SPOON_BOT_SHELL_BACKGROUND_HANDOFF_TIMEOUT",
                DEFAULT_SHELL_BACKGROUND_HANDOFF_TIMEOUT,
            )
            shell_active_tool_timeout = max(
                0.1,
                max(shell_foreground_timeout, shell_max_timeout) + shell_handoff_timeout + 5.0,
            )
            non_shell_active_tool_timeout = max(
                0.1,
                AgentLoop._float_env(
                    "SPOON_BOT_NON_SHELL_TOOL_ACTIVE_TIMEOUT",
                    _DEFAULT_NON_SHELL_ACTIVE_TOOL_TIMEOUT,
                ),
            )
            post_tool_result_silence_timeout = max(
                0.1,
                AgentLoop._float_env(
                    "SPOON_BOT_POST_TOOL_RESULT_SILENCE_TIMEOUT",
                    _DEFAULT_POST_TOOL_RESULT_SILENCE_TIMEOUT,
                ),
            )
            internal_recovery_timeout = AgentLoop._float_env(
                "SPOON_BOT_INTERNAL_RECOVERY_TIMEOUT",
                AgentLoop._positive_runtime_budget(
                    getattr(self, "internal_recovery_timeout", None),
                    _DEFAULT_INTERNAL_RECOVERY_TIMEOUT,
                ),
                allow_zero=True,
            )
            max_tool_results_without_content = max(
                1,
                int(
                    getattr(
                        self,
                        "max_stream_tool_results_without_content",
                        DEFAULT_MAX_STREAM_TOOL_RESULTS_WITHOUT_CONTENT,
                    )
                    or DEFAULT_MAX_STREAM_TOOL_RESULTS_WITHOUT_CONTENT
                ),
            )
            stream_heartbeat_initial_delay = AgentLoop._float_env(
                "SPOON_BOT_STREAM_HEARTBEAT_INITIAL_DELAY",
                _DEFAULT_STREAM_HEARTBEAT_INITIAL_DELAY,
                allow_zero=True,
            )
            stream_heartbeat_interval = AgentLoop._float_env(
                "SPOON_BOT_STREAM_HEARTBEAT_INTERVAL",
                _DEFAULT_STREAM_HEARTBEAT_INTERVAL,
                allow_zero=True,
            )
            stream_heartbeat_enabled = (
                stream_heartbeat_initial_delay > 0 and stream_heartbeat_interval > 0
            )

            def _record_tool_result_events(events: list[dict[str, Any]]) -> None:
                nonlocal last_tool_progress_at, last_tool_progress_kind, stream_tool_result_count
                nonlocal read_file_result_argument_counts, read_file_result_generations
                nonlocal read_context_generation
                nonlocal saw_content_after_tool_call
                if not events:
                    return
                last_tool_progress_at = time.monotonic()
                last_tool_progress_kind = "tool_result"
                saw_content_after_tool_call = False
                stream_tool_result_count += len(events)
                recent_tool_result_events.extend(events)
                all_tool_result_events.extend(events)
                del recent_tool_result_events[:-6]
                for event in events:
                    metadata = dict(event.get("metadata") or {})
                    result_tool_name = str(metadata.get("name") or "").strip()
                    if result_tool_name and result_tool_name != "read_file":
                        read_context_generation += 1
                    tool_call_id = (
                        metadata.get("tool_call_id")
                        or metadata.get("id")
                        or event.get("tool_call_id")
                        or event.get("id")
                    )
                    if tool_call_id:
                        active_tool_call_names_by_id.pop(str(tool_call_id), None)
                    elif result_tool_name:
                        for active_id, active_name in list(active_tool_call_names_by_id.items()):
                            if active_name == result_tool_name:
                                active_tool_call_names_by_id.pop(active_id, None)
                                break
                    if result_tool_name == "read_file":
                        arguments_key = AgentLoop._tool_call_arguments_key(
                            metadata.get("arguments") or ""
                        )
                        if arguments_key:
                            read_file_result_argument_counts[arguments_key] = (
                                read_file_result_argument_counts.get(arguments_key, 0) + 1
                            )
                            read_file_result_generations[arguments_key] = read_context_generation

            def _stop_tool_loop(reason: str) -> None:
                nonlocal run_result_text, pre_tool_scratchpad_events, pre_tool_scratchpad_buffer
                nonlocal post_tool_content_events, post_tool_content_buffer
                nonlocal tool_loop_fallback_active
                evidence_events = recent_tool_result_events or all_tool_result_events
                if evidence_events:
                    run_result_text = AgentLoop._build_tool_loop_fallback_response(
                        evidence_events,
                        reason=reason,
                        user_message=authoritative_message,
                    )
                else:
                    run_result_text = "NO_TOOL_RESULT_CAPTURED_BEFORE_STOP_CONDITION"
                tool_loop_fallback_active = True
                if bg_task is not None and not bg_task.done():
                    bg_task.cancel()
                pre_tool_scratchpad_events = []
                pre_tool_scratchpad_buffer = ""
                post_tool_content_events = []
                post_tool_content_buffer = ""
                try:
                    td.set()
                except Exception:
                    pass

            def _stop_if_total_timeout() -> bool:
                if provider_total_timeout <= 0:
                    return False
                now = time.monotonic()
                if now - stream_started_at < provider_total_timeout:
                    return False
                if _active_tool_within_budget(now):
                    return False
                if saw_tool_call and now - last_tool_progress_at < tool_followup_timeout:
                    return False
                logger.warning(
                    "Provider/tool loop exceeded total stream timeout; "
                    "returning a bounded fallback response."
                )
                _stop_tool_loop("total_timeout")
                return True

            def _internal_recovery_run_kwargs() -> dict[str, Any]:
                kwargs: dict[str, Any] = {}
                if self._callable_accepts_kwarg(self._agent.run, "reasoning_effort"):
                    kwargs["reasoning_effort"] = "low"
                return kwargs

            def _repair_events_from_queued_item(
                queued: Any,
                *,
                repair_reason: str,
                queued_content_parts: list[str],
            ) -> list[dict[str, Any]]:
                events: list[dict[str, Any]] = []

                if isinstance(queued, dict) and queued.get("tool_calls"):
                    for tc in queued.get("tool_calls") or []:
                        if isinstance(tc, dict):
                            fn = tc.get("function", {})
                            tc_id = tc.get("id", "")
                            fn_name = (
                                fn.get("name", "")
                                if isinstance(fn, dict)
                                else getattr(fn, "name", "")
                            )
                            fn_args = (
                                fn.get("arguments", "")
                                if isinstance(fn, dict)
                                else getattr(fn, "arguments", "")
                            )
                        else:
                            tc_id = getattr(tc, "id", "")
                            fn_obj = getattr(tc, "function", None)
                            fn_name = getattr(fn_obj, "name", "") if fn_obj else ""
                            fn_args = getattr(fn_obj, "arguments", "") if fn_obj else ""
                        if tc_id:
                            tool_call_arguments_by_id[tc_id] = (
                                AgentLoop._tool_call_arguments_key(fn_args)
                            )
                        display_args = AgentLoop._tool_call_arguments_display(fn_name, fn_args)
                        events.append(
                            {
                                "type": "tool_call",
                                "delta": "",
                                "metadata": {
                                    "id": tc_id,
                                    "name": fn_name,
                                    "arguments": display_args,
                                    "repair": repair_reason,
                                },
                            }
                        )
                    return events

                if isinstance(queued, dict) and queued.get("type") == "tool_result":
                    metadata = dict(queued.get("metadata") or {})
                    tool_name = queued.get("name") or metadata.get("name") or ""
                    tool_result = (
                        queued.get("result")
                        or queued.get("content")
                        or queued.get("response")
                        or queued.get("output")
                        or queued.get("delta")
                    )
                    serialized_result = AgentLoop._stringify_stream_payload(tool_result)
                    tool_call_id = (
                        queued.get("tool_call_id")
                        or queued.get("id")
                        or metadata.get("tool_call_id")
                        or metadata.get("id")
                    )
                    captured_output = consume_captured_tool_output(
                        tool_output_capture_scope,
                        tool_name=tool_name,
                        arguments=tool_call_arguments_by_id.get(tool_call_id, ""),
                    )
                    if tool_name:
                        metadata.setdefault("name", tool_name)
                    if tool_call_id:
                        metadata.setdefault("tool_call_id", tool_call_id)
                        metadata.setdefault("id", tool_call_id)
                        arguments = tool_call_arguments_by_id.get(tool_call_id, "")
                        if arguments:
                            metadata.setdefault(
                                "arguments",
                                AgentLoop._tool_call_arguments_display(tool_name, arguments),
                            )
                        emitted_tool_result_ids.add(tool_call_id)
                    metadata["repair"] = repair_reason
                    metadata = AgentLoop._merge_stream_tool_result_metadata(
                        metadata,
                        streamed_result=serialized_result,
                        captured_output=captured_output,
                    )
                    self._remember_stream_tool_result_metadata(tool_call_id, metadata)
                    events.append(
                        {
                            "type": "tool_result",
                            "delta": "",
                            "metadata": metadata,
                        }
                    )
                    return events

                metadata: dict[str, Any] = {}
                event_type = "content"
                if isinstance(queued, dict):
                    metadata = dict(queued.get("metadata") or {})
                    if str(queued.get("type") or "").strip() == "thinking":
                        event_type = "thinking"
                    text = queued.get("content") or queued.get("delta") or ""
                else:
                    text = getattr(queued, "content", None) or (
                        queued if isinstance(queued, str) else ""
                    )
                if text:
                    text = str(text)
                    queued_content_parts.append(text)
                    metadata.setdefault("repair", repair_reason)
                    events.append(
                        {
                            "type": event_type,
                            "delta": text,
                            "metadata": metadata,
                        }
                    )
                return events

            async def _stream_internal_recovery_run(
                run_factory: Callable[[], Awaitable[Any]],
                *,
                label: str,
                repair_reason: str,
                result_holder: dict[str, Any],
            ) -> AsyncGenerator[dict[str, Any], None]:
                """Run internal recovery while forwarding queued tool events immediately."""
                nonlocal stream_error_reason, stream_tool_result_index

                queued_content_parts: list[str] = []
                result: Any | None = None
                timed_out = False
                deadline = (
                    time.monotonic() + internal_recovery_timeout
                    if internal_recovery_timeout > 0
                    else None
                )

                task = asyncio.create_task(run_factory())

                def _collect_repair_tool_result_events() -> list[dict[str, Any]]:
                    nonlocal stream_tool_result_index
                    tool_result_events, stream_tool_result_index = (
                        AgentLoop._collect_stream_tool_result_events(
                            self,
                            stream_tool_result_index,
                            emitted_tool_result_ids,
                            tool_output_capture_scope=tool_output_capture_scope,
                            tool_call_arguments_by_id=tool_call_arguments_by_id,
                        )
                    )
                    repaired_events: list[dict[str, Any]] = []
                    for event in tool_result_events:
                        metadata = dict(event.get("metadata") or {})
                        metadata["repair"] = repair_reason
                        repaired_events.append({**event, "metadata": metadata})
                    return repaired_events

                try:
                    while True:
                        if task.done() and self._agent.output_queue.empty():
                            break

                        timeout = 0.25
                        if deadline is not None:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0 and not task.done():
                                timed_out = True
                                task.cancel()
                                break
                            timeout = min(timeout, max(0.01, remaining))

                        try:
                            queued = await asyncio.wait_for(
                                self._agent.output_queue.get(),
                                timeout=timeout,
                            )
                        except asyncio.TimeoutError:
                            for event in _collect_repair_tool_result_events():
                                yield event
                            continue
                        except Exception as exc:
                            logger.debug(f"{label} output queue polling failed: {exc}")
                            for event in _collect_repair_tool_result_events():
                                yield event
                            continue

                        for event in _repair_events_from_queued_item(
                            queued,
                            repair_reason=repair_reason,
                            queued_content_parts=queued_content_parts,
                        ):
                            yield event
                        for event in _collect_repair_tool_result_events():
                            yield event

                    if timed_out:
                        logger.warning(
                            "{} exceeded internal recovery timeout {:.1f}s; "
                            "continuing with captured tool evidence.",
                            label,
                            internal_recovery_timeout,
                        )
                        try:
                            await asyncio.wait_for(task, timeout=2.0)
                        except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                            pass
                    else:
                        result = await task

                    while not self._agent.output_queue.empty():
                        try:
                            queued = self._agent.output_queue.get_nowait()
                        except Exception:
                            break
                        for event in _repair_events_from_queued_item(
                            queued,
                            repair_reason=repair_reason,
                            queued_content_parts=queued_content_parts,
                        ):
                            yield event
                        for event in _collect_repair_tool_result_events():
                            yield event
                except asyncio.CancelledError:
                    if not task.done():
                        task.cancel()
                    raise
                except Exception as exc:
                    stream_error_reason = exc
                    logger.error(f"{label} failed: {exc}")
                    result_holder["text"] = ""
                    return

                repair_text = ""
                if result is None:
                    if timed_out:
                        # Internal recovery timeouts are soft: the caller can
                        # still synthesize a grounded answer from any queued
                        # content/tool evidence collected above. Marking the
                        # whole user turn as a stream error here makes a
                        # partially successful long workflow look cancelled.
                        pass
                    elif stream_error_reason is None:
                        stream_error_reason = RuntimeError(f"{label} returned no result")
                else:
                    repair_text = AgentLoop._extract_run_result_text(result)
                if not repair_text.strip() and queued_content_parts:
                    repair_text = "".join(queued_content_parts)
                result_holder["text"] = repair_text

                for event in _collect_repair_tool_result_events():
                    yield event

            async def _stream_pseudo_tool_call_repair(
                pseudo_content: str,
                *,
                result_holder: dict[str, Any],
                repair_prompt_override: str | None = None,
                repair_reason: str = "pseudo_tool_call_text",
                request_hint_overrides: dict[str, Any] | None = None,
            ) -> AsyncGenerator[dict[str, Any], None]:
                """Retry once when the model wrote fake tool calls as text."""
                nonlocal stream_error_reason, turn_tool_invocation_state

                AgentLoop._drop_pseudo_tool_call_assistant_messages(
                    self,
                    _pre_turn_memory_index,
                )
                AgentLoop._drain_agent_output_queue(self)
                self._reset_agent_state_for_retry()

                repair_prompt = (
                    repair_prompt_override
                    or AgentLoop._build_pseudo_tool_call_repair_prompt(
                        authoritative_message,
                        pseudo_content,
                    )
                )
                await self._agent.add_message("user", repair_prompt)
                self._agent.next_step_prompt = repair_prompt

                repair_kwargs = _internal_recovery_run_kwargs()

                request_execution_hints = self._build_request_execution_hints(authoritative_message)
                if request_hint_overrides:
                    request_execution_hints = {
                        **request_execution_hints,
                        **request_hint_overrides,
                    }
                previous_force_serial = bool(getattr(self, "_force_serial_tool_calls", False))
                self._force_serial_tool_calls = True

                async def _run_repair() -> Any:
                    nonlocal turn_tool_invocation_state
                    with (
                        bind_request_execution_hints(request_execution_hints),
                        track_tool_invocations(
                            initial_state=turn_tool_invocation_state,
                        ) as invocation_state,
                    ):
                        turn_tool_invocation_state = invocation_state
                        return await AgentLoop._run_agent_with_context_overflow_recovery(
                            self,
                            label="stream_tool_call_repair",
                            retry_runner=AgentLoop._resolve_retry_runner(self),
                            pre_overflow_retry_cleanup=lambda: (
                                AgentLoop._drain_agent_output_queue(self)
                            ),
                            **repair_kwargs,
                        )

                try:
                    async for event in _stream_internal_recovery_run(
                        _run_repair,
                        label="stream_tool_call_repair",
                        repair_reason=repair_reason,
                        result_holder=result_holder,
                    ):
                        yield event
                finally:
                    if previous_force_serial:
                        self._force_serial_tool_calls = True
                    elif hasattr(self, "_force_serial_tool_calls"):
                        delattr(self, "_force_serial_tool_calls")

            async def _stream_repeated_read_recovery(
                *,
                result_holder: dict[str, Any],
            ) -> AsyncGenerator[dict[str, Any], None]:
                """Retry once when duplicate read_file calls consume the tool loop."""
                nonlocal turn_tool_invocation_state

                AgentLoop._drain_agent_output_queue(self)
                try:
                    AgentLoop._normalize_runtime_tool_context(
                        AgentLoop._get_runtime_memory_messages(self),
                        finalized=True,
                    )
                except Exception as exc:
                    logger.debug(f"Repeated-read recovery memory normalization skipped: {exc}")
                self._reset_agent_state_for_retry()

                request_execution_hints = self._build_request_execution_hints(authoritative_message)
                repair_prompt = AgentLoop._build_repeated_read_recovery_prompt(
                    authoritative_message,
                )
                await self._agent.add_message("user", repair_prompt)
                self._agent.next_step_prompt = repair_prompt

                repair_kwargs = _internal_recovery_run_kwargs()

                previous_force_serial = bool(getattr(self, "_force_serial_tool_calls", False))
                self._force_serial_tool_calls = True

                async def _run_recovery() -> Any:
                    nonlocal turn_tool_invocation_state
                    with (
                        bind_request_execution_hints(request_execution_hints),
                        track_tool_invocations(
                            max_repeats=1,
                            initial_state=turn_tool_invocation_state,
                        ) as invocation_state,
                    ):
                        turn_tool_invocation_state = invocation_state
                        return await AgentLoop._run_agent_with_context_overflow_recovery(
                            self,
                            label="stream_repeated_read_recovery",
                            retry_runner=AgentLoop._resolve_retry_runner(self),
                            pre_overflow_retry_cleanup=lambda: (
                                AgentLoop._drain_agent_output_queue(self)
                            ),
                            **repair_kwargs,
                        )

                try:
                    async for event in _stream_internal_recovery_run(
                        _run_recovery,
                        label="stream_repeated_read_recovery",
                        repair_reason="repeated_read_recovery",
                        result_holder=result_holder,
                    ):
                        yield event
                finally:
                    if previous_force_serial:
                        self._force_serial_tool_calls = True
                    elif hasattr(self, "_force_serial_tool_calls"):
                        delattr(self, "_force_serial_tool_calls")

            async def _emit_repair_event_stream(
                repair_stream: AsyncGenerator[dict[str, Any], None],
                *,
                visible_tool_result_delta: bool = False,
                collected_events: list[dict[str, Any]] | None = None,
            ) -> AsyncGenerator[dict[str, Any], None]:
                nonlocal emitted_content_text, full_content
                nonlocal last_tool_progress_at, last_tool_progress_kind
                nonlocal saw_tool_call, saw_content_after_tool_call
                nonlocal streamed_content_after_latest_tool_call

                async for event in repair_stream:
                    event_type = str(event.get("type") or "")
                    if event_type == "tool_call":
                        saw_tool_call = True
                        saw_content_after_tool_call = False
                        streamed_content_after_latest_tool_call = False
                        last_tool_progress_at = time.monotonic()
                        last_tool_progress_kind = "tool_call"
                    if event_type == "tool_result":
                        if visible_tool_result_delta:
                            metadata = dict(event.get("metadata") or {})
                            event = {
                                **event,
                                "delta": AgentLoop._stream_tool_result_visible_delta(
                                    event.get("delta", ""),
                                    metadata,
                                ),
                                "metadata": metadata,
                            }
                        _record_tool_result_events([event])
                    if event_type == "content":
                        delta = str(event.get("delta") or "")
                        if not delta or AgentLoop._is_internal_completion_sentinel(delta):
                            continue
                        delta = AgentLoop._trim_repeated_stream_prefix(
                            delta,
                            emitted_content_text,
                        )
                        delta = AgentLoop._mask_user_visible_text(delta)
                        if not delta:
                            continue
                        metadata = dict(event.get("metadata") or {})
                        metadata.setdefault("validated", True)
                        event = {**event, "delta": delta, "metadata": metadata}
                        full_content += delta
                        emitted_content_text += delta
                        saw_content_after_tool_call = True
                        streamed_content_after_latest_tool_call = True
                    elif event_type == "thinking" and not thinking:
                        continue
                    if collected_events is not None:
                        collected_events.append(event)
                    yield _decorate_stream_event(event)

            def _active_tool_within_budget(now: float | None = None) -> bool:
                if not saw_tool_call or last_tool_progress_kind != "tool_call":
                    return False
                now = time.monotonic() if now is None else now
                active_names = {
                    name for name in last_tool_call_names if isinstance(name, str) and name
                }
                if not active_names:
                    active_names = {
                        name
                        for name in active_tool_call_names_by_id.values()
                        if isinstance(name, str) and name
                    }
                budget = (
                    shell_active_tool_timeout
                    if "shell" in active_names
                    else non_shell_active_tool_timeout
                )
                return now - last_tool_progress_at < budget

            def _repeated_read_storm_active() -> bool:
                return not repeated_read_recovery_attempted and any(
                    count >= _REPEATED_READ_RECOVERY_THRESHOLD
                    for count in read_file_result_argument_counts.values()
                )

            def _read_file_call_repeats_current_context(arguments_key: str) -> bool:
                if not arguments_key:
                    return False
                if read_file_result_argument_counts.get(arguments_key, 0) <= 0:
                    return False
                return read_file_result_generations.get(arguments_key) == read_context_generation

            def _redundant_read_tool_result_event(
                *,
                tool_call_id: str,
                tool_name: str,
                arguments: str,
            ) -> dict[str, Any]:
                result = (
                    "READ_FILE_CACHE_HIT: requested file range was already "
                    "provided earlier in this request. Continue from the "
                    "existing file evidence and advance the task instead of "
                    "waiting for another identical read_file result."
                )
                metadata: dict[str, Any] = {
                    "name": tool_name or "read_file",
                    "arguments": AgentLoop._tool_call_arguments_display(
                        tool_name or "read_file",
                        arguments,
                    ),
                    "result": result,
                    "repair": "repeated_read_call_guard",
                }
                if tool_call_id:
                    metadata["id"] = tool_call_id
                    metadata["tool_call_id"] = tool_call_id
                metadata = AgentLoop._merge_stream_tool_result_metadata(
                    metadata,
                    streamed_result=result,
                    captured_output=None,
                )
                self._remember_stream_tool_result_metadata(tool_call_id, metadata)
                return {
                    "type": "tool_result",
                    "delta": AgentLoop._stream_tool_result_visible_delta(
                        "",
                        metadata,
                    ),
                    "metadata": metadata,
                }

            def _events_have_repeated_read_guardrail(
                events: list[dict[str, Any]],
            ) -> bool:
                for event in events:
                    metadata = dict(event.get("metadata") or {})
                    if str(metadata.get("name") or "").strip() != "read_file":
                        continue
                    payload = (
                        metadata.get("model_result")
                        or metadata.get("model_output")
                        or metadata.get("model_content")
                        or metadata.get("output")
                        or metadata.get("result")
                        or metadata.get("content")
                        or event.get("delta")
                    )
                    text = AgentLoop._stringify_stream_payload(payload).lower()
                    if (
                        "repeated redundant read_file suppressed" in text
                        or "requested file range was already provided" in text
                        or "requested file range already available" in text
                        or "already available in this request" in text
                        or "file content already available" in text
                    ):
                        return True
                return False

            def _stop_current_run_for_repeated_read_recovery() -> None:
                if bg_task is not None and not bg_task.done():
                    bg_task.cancel()
                try:
                    td.set()
                except Exception:
                    pass

            def _stop_current_run_for_history_search_budget_recovery() -> None:
                if bg_task is not None and not bg_task.done():
                    bg_task.cancel()
                try:
                    td.set()
                except Exception:
                    pass

            def _stop_current_run_for_post_tool_result_silence_recovery() -> None:
                if bg_task is not None and not bg_task.done():
                    bg_task.cancel()
                try:
                    td.set()
                except Exception:
                    pass

            def _stop_current_run_for_tool_protocol_leak() -> None:
                if bg_task is not None and not bg_task.done():
                    bg_task.cancel()
                try:
                    td.set()
                except Exception:
                    pass

            plain_continuation = AgentLoop._request_is_plain_bounded_continuation(
                authoritative_message,
                request_execution_hints,
            )

            def _can_auto_continue_for_current_request() -> bool:
                if not plain_continuation:
                    return True
                return AgentLoop._plain_continuation_can_auto_continue_same_unit(
                    all_tool_result_events,
                )

            def _can_run_skill_contract_continuation() -> bool:
                return (
                    skill_contract_continuation_attempts
                    < AgentLoop._skill_contract_continuation_attempt_limit()
                    and _can_auto_continue_for_current_request()
                )

            def _mark_skill_contract_continuation_attempt() -> None:
                nonlocal skill_contract_continuation_attempted
                nonlocal skill_contract_continuation_attempts
                skill_contract_continuation_attempted = True
                skill_contract_continuation_attempts += 1

            def _mark_stream_activity(event: dict[str, Any]) -> dict[str, Any]:
                nonlocal last_stream_activity_at
                last_stream_activity_at = time.monotonic()
                return event

            def _heartbeat_poll_timeout(base_timeout: float) -> float:
                if not stream_heartbeat_enabled:
                    return base_timeout
                heartbeat_timeout = min(
                    stream_heartbeat_initial_delay,
                    stream_heartbeat_interval,
                )
                return min(base_timeout, max(0.05, heartbeat_timeout))

            def _should_emit_stream_heartbeat(now: float) -> bool:
                if not stream_heartbeat_enabled or td.is_set():
                    return False
                if _active_tool_within_budget(now):
                    return False
                last_visible_at = max(last_stream_activity_at, last_stream_heartbeat_at)
                wait_seconds = (
                    stream_heartbeat_initial_delay
                    if stream_heartbeat_count == 0
                    else stream_heartbeat_interval
                )
                return now - last_visible_at >= wait_seconds

            def _stream_heartbeat_event(now: float) -> dict[str, Any]:
                nonlocal last_stream_heartbeat_at, stream_heartbeat_count
                last_stream_heartbeat_at = now
                stream_heartbeat_count += 1
                elapsed_seconds = max(0, int(now - stream_started_at))
                return _decorate_stream_event(
                    {
                        "type": "notice",
                        "delta": "",
                        "part": {
                            "type": "thinking",
                            "text": "Agent is still running.",
                        },
                        "metadata": {
                            "kind": "agent_progress_heartbeat",
                            "status": "running",
                            "elapsed_seconds": elapsed_seconds,
                            "synthetic": True,
                        },
                    }
                )

            while not (td.is_set() and oq.empty()):
                now = time.monotonic()
                if _stop_if_total_timeout():
                    break

                for event in _drain_runtime_notice_events():
                    yield _mark_stream_activity(event)

                tool_result_events, stream_tool_result_index = (
                    AgentLoop._collect_stream_tool_result_events(
                        self,
                        stream_tool_result_index,
                        emitted_tool_result_ids,
                        tool_output_capture_scope=tool_output_capture_scope,
                        tool_call_arguments_by_id=tool_call_arguments_by_id,
                    )
                )
                _record_tool_result_events(tool_result_events)
                if _events_have_repeated_read_guardrail(tool_result_events):
                    repeated_read_guardrail_seen = True
                    if not repeated_read_recovery_attempted:
                        logger.warning(
                            "Repeated read_file guardrail returned; switching to "
                            "internal continuation recovery."
                        )
                        _stop_current_run_for_repeated_read_recovery()
                        break
                if AgentLoop._tool_events_have_history_search_budget(tool_result_events):
                    history_search_budget_seen = True
                for event in tool_result_events:
                    yield _mark_stream_activity(_decorate_stream_event(event))
                if history_search_budget_seen and not history_search_budget_recovery_attempted:
                    logger.warning(
                        "History search budget was exhausted without task progress; "
                        "switching to internal continuation recovery."
                    )
                    _stop_current_run_for_history_search_budget_recovery()
                    break
                if any(
                    AgentLoop._is_tool_loop_suppression_event(event) for event in tool_result_events
                ):
                    logger.warning("Stopping tool loop after repeated-tool guardrail result.")
                    _stop_tool_loop("tool_suppression")
                    break
                if _repeated_read_storm_active():
                    logger.warning(
                        "Repeated read_file results are no longer making progress; "
                        "switching to internal continuation recovery."
                    )
                    _stop_current_run_for_repeated_read_recovery()
                    break
                if (
                    _can_run_skill_contract_continuation()
                    and skill_contract_inspection_stalled_after_progress(all_tool_result_events)
                ):
                    logger.warning(
                        "Skill turn returned to repeated setup/read-only inspection "
                        "after progress; switching to internal continuation recovery."
                    )
                    post_tool_result_silence_recovery_seen = True
                    _stop_current_run_for_post_tool_result_silence_recovery()
                    break
                if _stop_if_total_timeout():
                    break
                if (
                    saw_tool_call
                    and last_tool_progress_kind == "tool_call"
                    and not _active_tool_within_budget()
                ):
                    logger.warning(
                        "Stopping tool loop because an active tool call "
                        "did not produce a result within its bounded active budget."
                    )
                    _stop_tool_loop("active_tool_timeout")
                    break

                if (
                    saw_tool_call
                    and last_tool_progress_kind == "tool_result"
                    and not saw_content_after_tool_call
                    and recent_tool_result_events
                    and now - last_tool_progress_at >= post_tool_result_silence_timeout
                ):
                    if (
                        should_run_skill_contract_check(all_tool_result_events)
                        and _can_run_skill_contract_continuation()
                    ):
                        logger.warning(
                            "Provider was silent after skill tool results; "
                            "switching to internal skill-contract continuation."
                        )
                        post_tool_result_silence_recovery_seen = True
                        _stop_current_run_for_post_tool_result_silence_recovery()
                    else:
                        logger.warning(
                            "Stopping tool loop because provider produced no "
                            "content after the latest tool result within the "
                            "post-tool silence budget."
                        )
                        _stop_tool_loop("post_tool_result_silence_timeout")
                    break

                if (
                    saw_tool_call
                    and not saw_content_after_tool_call
                    and recent_tool_result_events
                    and not _active_tool_within_budget()
                    and (
                        stream_tool_result_count >= max_tool_results_without_content
                        or time.monotonic() - last_tool_progress_at >= tool_followup_timeout
                    )
                ):
                    logger.warning(
                        "Stopping tool loop without final content after "
                        f"{stream_tool_result_count} tool result(s) and "
                        f"{stream_tool_call_count} tool call(s)."
                    )
                    _stop_tool_loop("tool_followup_timeout")
                    break

                # tracked_reasoning is inferred from assistant output logs and is
                # not a reliable API thinking source. Clear it so it does not leak
                # duplicated final content into WS/REST responses.
                self._drain_reasoning_chunks()
                try:
                    # Poll without a hard stream deadline so long-running tasks
                    # only stop when the caller explicitly cancels them.
                    # Use oq.get() without timeout kwarg - works for both
                    # asyncio.Queue and ThreadSafeOutputQueue. Timeout is
                    # handled by the outer asyncio.wait_for.
                    queue_poll_timeout = (
                        min(2.0, max(0.05, provider_silence_timeout))
                        if provider_silence_enabled
                        else 2.0
                    )
                    queue_poll_timeout = _heartbeat_poll_timeout(queue_poll_timeout)
                    chunk = await asyncio.wait_for(oq.get(), timeout=queue_poll_timeout)
                    chunk_count += 1
                    logger.debug(
                        f"Got chunk #{chunk_count}: type={type(chunk).__name__}, repr={repr(chunk)[:200]}"
                    )
                except asyncio.TimeoutError:
                    if (
                        saw_tool_call
                        and last_tool_progress_kind == "tool_call"
                        and not _active_tool_within_budget()
                    ):
                        logger.warning(
                            "Stopping tool loop because an active tool call "
                            "did not produce a result within its bounded active budget."
                        )
                        _stop_tool_loop("active_tool_timeout")
                        break
                    heartbeat_now = time.monotonic()
                    if _should_emit_stream_heartbeat(heartbeat_now):
                        yield _mark_stream_activity(_stream_heartbeat_event(heartbeat_now))
                    if (
                        provider_silence_enabled
                        and not td.is_set()
                        and not saw_tool_call
                        and not full_content
                        and time.monotonic() - stream_started_at >= provider_silence_timeout
                    ):
                        if provider_silence_retry_count < max_provider_silence_retries:
                            provider_silence_retry_count += 1
                            logger.warning(
                                "Provider produced no stream output before silence "
                                "timeout; retrying the same turn once."
                            )
                            if bg_task is not None and not bg_task.done():
                                bg_task.cancel()
                                try:
                                    await asyncio.wait_for(bg_task, timeout=2.0)
                                except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                                    pass
                            AgentLoop._drain_agent_output_queue(self)
                            run_result_text = ""
                            pre_tool_scratchpad_events = []
                            pre_tool_scratchpad_buffer = ""
                            if (
                                isinstance(effective_reasoning_effort, str)
                                and effective_reasoning_effort.strip().lower() != "low"
                            ):
                                retry_reasoning_effort = "low"
                            try:
                                td.clear()
                            except Exception:
                                pass
                            with (
                                bind_tool_owner(self._current_tool_owner_key()),
                                bind_tool_workspace(str(getattr(self, "workspace", "") or "")),
                            ):
                                bg_task = asyncio.create_task(_run_and_signal())
                            await asyncio.sleep(0)
                            stream_started_at = time.monotonic()
                            continue
                        run_result_text = (
                            "The model provider did not produce a response before "
                            "the request timeout. Please retry this request."
                        )
                        logger.warning(
                            "Provider produced no stream output before silence timeout; "
                            "returning a bounded fallback response."
                        )
                        if bg_task is not None and not bg_task.done():
                            bg_task.cancel()
                        pre_tool_scratchpad_events = []
                        pre_tool_scratchpad_buffer = ""
                        try:
                            td.set()
                        except Exception:
                            pass
                        break
                    continue
                except asyncio.CancelledError:
                    logger.warning("Streaming cancelled")
                    AgentLoop._drain_agent_output_queue(self)
                    AgentLoop._truncate_runtime_memory(self, _pre_turn_memory_index)
                    raise
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
                        if stream_error_reason is None:
                            error_text = (
                                (
                                    chunk.get("metadata", {}).get("error")
                                    if isinstance(chunk.get("metadata"), dict)
                                    else None
                                )
                                or chunk.get("delta")
                                or "stream_error"
                            )
                            stream_error_reason = RuntimeError(str(error_text))
                        yield _mark_stream_activity(
                            {
                                "type": "error",
                                "delta": chunk.get("delta", ""),
                                "metadata": chunk.get("metadata", {}),
                                "source": current_source,
                            }
                        )
                        continue

                    if "tool_calls" in chunk and chunk["tool_calls"]:
                        leaked_tool_protocol_buffer = ""
                        leaked_tool_protocol_detected = False
                        leaked_tool_protocol_probe = ""
                        if self._should_disable_parallel_tool_calls() and isinstance(
                            chunk.get("tool_calls"), (list, tuple)
                        ):
                            raw_tool_calls = list(chunk.get("tool_calls") or [])
                            if len(raw_tool_calls) > 1:
                                chunk = dict(chunk)
                                chunk["tool_calls"] = raw_tool_calls[:1]
                        if full_content:
                            full_content = ""
                        saw_content_after_tool_call = False
                        streamed_content_after_latest_tool_call = False
                        post_tool_content_passthrough_started = False
                        saw_tool_call = True
                        stream_tool_call_count += len(chunk["tool_calls"])
                        last_tool_progress_at = time.monotonic()
                        last_tool_progress_kind = "tool_call"
                        current_tool_call_names: list[str] = []
                        # Initial content before a tool call is tool preamble by
                        # structure; do not classify it with language phrases.
                        if pre_tool_scratchpad_events:
                            pre_tool_scratchpad_events = []
                            pre_tool_scratchpad_buffer = ""
                        if post_tool_content_buffer:
                            post_tool_content_events = []
                            post_tool_content_buffer = ""
                        repeated_read_call_guard_events: list[dict[str, Any]] = []
                        for tc in chunk["tool_calls"]:
                            # tc may be a ToolCall pydantic object or a dict
                            if isinstance(tc, dict):
                                fn = tc.get("function", {})
                                tc_id = tc.get("id", "")
                                fn_name = (
                                    fn.get("name", "")
                                    if isinstance(fn, dict)
                                    else getattr(fn, "name", "")
                                )
                                fn_args = (
                                    fn.get("arguments", "")
                                    if isinstance(fn, dict)
                                    else getattr(fn, "arguments", "")
                                )
                            else:
                                tc_id = getattr(tc, "id", "")
                                fn_obj = getattr(tc, "function", None)
                                fn_name = getattr(fn_obj, "name", "") if fn_obj else ""
                                fn_args = getattr(fn_obj, "arguments", "") if fn_obj else ""
                            arguments_key = AgentLoop._tool_call_arguments_key(fn_args)
                            normalized_fn_name = str(fn_name or "").strip()
                            if normalized_fn_name:
                                current_tool_call_names.append(normalized_fn_name)
                            if tc_id:
                                tool_call_arguments_by_id[tc_id] = arguments_key
                                if normalized_fn_name:
                                    active_tool_call_names_by_id[str(tc_id)] = normalized_fn_name
                            if normalized_fn_name != "read_file":
                                non_read_tool_call_count += 1
                            yield _mark_stream_activity(
                                _decorate_stream_event(
                                    {
                                        "type": "tool_call",
                                        "delta": "",
                                        "metadata": {
                                            "id": tc_id,
                                            "name": fn_name,
                                            "arguments": AgentLoop._tool_call_arguments_display(
                                                fn_name,
                                                fn_args,
                                            ),
                                        },
                                    }
                                )
                            )
                            if (
                                normalized_fn_name == "read_file"
                                and _read_file_call_repeats_current_context(arguments_key)
                            ):
                                repeated_read_call_guard_events.append(
                                    _redundant_read_tool_result_event(
                                        tool_call_id=str(tc_id or ""),
                                        tool_name=normalized_fn_name,
                                        arguments=fn_args,
                                    )
                                )
                        if current_tool_call_names:
                            last_tool_call_names = tuple(current_tool_call_names)
                        if repeated_read_call_guard_events:
                            repeated_read_guardrail_seen = True
                            for event in repeated_read_call_guard_events:
                                metadata = dict(event.get("metadata") or {})
                                tool_call_id = metadata.get("tool_call_id") or metadata.get("id")
                                if tool_call_id:
                                    emitted_tool_result_ids.add(str(tool_call_id))
                                _record_tool_result_events([event])
                                yield _mark_stream_activity(_decorate_stream_event(event))
                            logger.warning(
                                "Repeated read_file call matched already completed "
                                "tool evidence; closing the tool event and switching "
                                "to internal continuation recovery."
                            )
                            _stop_current_run_for_repeated_read_recovery()
                            break
                        if _stop_if_total_timeout():
                            break
                        continue
                    chunk_metadata = chunk.get("metadata")
                    if isinstance(chunk_metadata, dict):
                        metadata = dict(chunk_metadata)
                    dict_type = chunk.get("type")
                    if dict_type == "thinking":
                        chunk_type = "thinking"
                    elif dict_type == "tool_result":
                        chunk_type = "tool_result"
                        tool_name = chunk.get("name") or metadata.get("name") or ""
                        tool_result = (
                            chunk.get("result")
                            or chunk.get("content")
                            or chunk.get("response")
                            or chunk.get("output")
                            or chunk.get("delta")
                        )
                        serialized_result = AgentLoop._stringify_stream_payload(tool_result)
                        tool_call_id = (
                            chunk.get("tool_call_id")
                            or chunk.get("id")
                            or metadata.get("tool_call_id")
                            or metadata.get("id")
                        )
                        captured_output = consume_captured_tool_output(
                            tool_output_capture_scope,
                            tool_name=tool_name,
                            arguments=tool_call_arguments_by_id.get(tool_call_id, ""),
                        )
                        if tool_name:
                            metadata.setdefault("name", tool_name)
                        if tool_call_id:
                            metadata.setdefault("tool_call_id", tool_call_id)
                            metadata.setdefault("id", tool_call_id)
                            arguments = tool_call_arguments_by_id.get(tool_call_id, "")
                            if arguments:
                                metadata.setdefault(
                                    "arguments",
                                    AgentLoop._tool_call_arguments_display(tool_name, arguments),
                                )
                            emitted_tool_result_ids.add(tool_call_id)
                        metadata = AgentLoop._merge_stream_tool_result_metadata(
                            metadata,
                            streamed_result=serialized_result,
                            captured_output=captured_output,
                        )
                        self._remember_stream_tool_result_metadata(tool_call_id, metadata)
                        _record_tool_result_events(
                            [
                                {
                                    "type": "tool_result",
                                    "delta": delta,
                                    "metadata": metadata,
                                }
                            ]
                        )
                    elif dict_type == "content":
                        chunk_type = "content"
                    # Support both "content" and "delta" keys (#10)
                    text = chunk.get("content") or chunk.get("delta") or ""
                    if text:
                        delta = text

                # -- Object chunks with content --
                elif hasattr(chunk, "content") and chunk.content:
                    delta = str(chunk.content)

                # -- Plain string chunks --
                elif isinstance(chunk, str):
                    delta = chunk

                if delta and chunk_type in {"thinking", "content"}:
                    leaked_tool_protocol_probe = (leaked_tool_protocol_probe + str(delta))[-1200:]
                    if leaked_tool_protocol_detected or AgentLoop._looks_like_tool_call_protocol_fragment(
                        leaked_tool_protocol_probe
                    ):
                        leaked_tool_protocol_detected = True
                        leaked_tool_protocol_buffer = (
                            leaked_tool_protocol_buffer + str(delta)
                        )[-12_000:]
                        logger.warning(
                            "Provider streamed tool-call protocol markup as {} text; "
                            "suppressing it and retrying through the tool-call API.",
                            chunk_type,
                        )
                        _stop_current_run_for_tool_protocol_leak()
                        continue

                if chunk_type == "tool_result":
                    delta = AgentLoop._stream_tool_result_visible_delta(delta, metadata)
                    tool_result_event = {
                        "type": chunk_type,
                        "delta": delta,
                        "metadata": metadata,
                    }
                    yield _mark_stream_activity(_decorate_stream_event(tool_result_event))
                    if _events_have_repeated_read_guardrail([tool_result_event]):
                        repeated_read_guardrail_seen = True
                        if not repeated_read_recovery_attempted:
                            logger.warning(
                                "Repeated read_file guardrail returned; switching to "
                                "internal continuation recovery."
                            )
                            _stop_current_run_for_repeated_read_recovery()
                            break
                    if AgentLoop._tool_events_have_history_search_budget([tool_result_event]):
                        history_search_budget_seen = True
                    if history_search_budget_seen and not history_search_budget_recovery_attempted:
                        logger.warning(
                            "History search budget was exhausted without task progress; "
                            "switching to internal continuation recovery."
                        )
                        _stop_current_run_for_history_search_budget_recovery()
                        break
                    if AgentLoop._is_tool_loop_suppression_event(tool_result_event):
                        logger.warning("Stopping tool loop after repeated-tool guardrail result.")
                        _stop_tool_loop("tool_suppression")
                        break
                    if _repeated_read_storm_active():
                        logger.warning(
                            "Repeated read_file results are no longer making progress; "
                            "switching to internal continuation recovery."
                        )
                        _stop_current_run_for_repeated_read_recovery()
                        break
                    if (
                        _can_run_skill_contract_continuation()
                        and skill_contract_inspection_stalled_after_progress(all_tool_result_events)
                    ):
                        logger.warning(
                            "Skill turn returned to repeated setup/read-only inspection "
                            "after progress; switching to internal continuation recovery."
                        )
                        post_tool_result_silence_recovery_seen = True
                        _stop_current_run_for_post_tool_result_silence_recovery()
                        break
                    if _stop_if_total_timeout():
                        break
                    continue

                if chunk_type == "thinking":
                    if thinking and delta:
                        yield _mark_stream_activity(
                            _decorate_stream_event(
                                {
                                    "type": "thinking",
                                    "delta": delta,
                                    "metadata": metadata,
                                }
                            )
                        )
                        if _stop_if_total_timeout():
                            break
                    continue

                if chunk_type == "content" and AgentLoop._is_internal_completion_sentinel(delta):
                    logger.debug("Suppressing internal completion sentinel from stream content")
                    continue

                if delta:
                    metadata_phase = metadata.get("phase") if isinstance(metadata, dict) else None
                    explicit_pre_tool_phase = (
                        chunk_type == "content" and thinking and metadata_phase == "think"
                    )
                    if chunk_type == "content" and explicit_pre_tool_phase:
                        yield _mark_stream_activity(
                            _decorate_stream_event(
                                {
                                    "type": "thinking",
                                    "delta": delta,
                                    "metadata": {
                                        **metadata,
                                        "source": metadata.get("source", "phase_think"),
                                    },
                                }
                            )
                        )
                        if _stop_if_total_timeout():
                            break
                    else:
                        event = {"type": chunk_type, "delta": delta, "metadata": metadata}
                        if chunk_type == "content":
                            if (
                                not saw_tool_call
                                and not initial_content_passthrough_started
                                and len(pre_tool_scratchpad_buffer + delta) <= 300
                            ):
                                pre_tool_scratchpad_buffer += delta
                                pre_tool_scratchpad_events.append(event)
                                continue
                            if pre_tool_scratchpad_buffer:
                                delta = pre_tool_scratchpad_buffer + delta
                                pre_tool_scratchpad_events = []
                                pre_tool_scratchpad_buffer = ""
                            if not saw_tool_call:
                                initial_content_passthrough_started = True
                                delta = AgentLoop._mask_user_visible_text(delta)
                                if not delta:
                                    continue
                                full_content += delta
                            else:
                                if not post_tool_content_passthrough_started:
                                    post_tool_content_buffer += delta
                                    post_tool_content_events.append(event)
                                    pending_content = AgentLoop._strip_leaked_scratchpad_prefix(
                                        post_tool_content_buffer
                                    )
                                    compact_pending = " ".join(pending_content.strip().split())
                                    if (
                                        not compact_pending
                                        or len(compact_pending) < 12
                                        or AgentLoop._looks_like_internal_scratchpad_text(
                                            pending_content
                                        )
                                        or AgentLoop._looks_like_incomplete_repeated_stream_prefix(
                                            pending_content,
                                            emitted_content_text,
                                        )
                                    ):
                                        continue
                                    delta = pending_content
                                    post_tool_content_events = []
                                    post_tool_content_buffer = ""
                                    post_tool_content_passthrough_started = True
                                delta = AgentLoop._trim_repeated_stream_prefix(
                                    delta,
                                    emitted_content_text,
                                )
                                delta = AgentLoop._mask_user_visible_text(delta)
                                if not delta:
                                    continue
                                saw_content_after_tool_call = True
                                streamed_content_after_latest_tool_call = True
                                full_content += delta
                            if buffer_stream_content:
                                continue
                            event["delta"] = delta
                            emitted_content_text += delta
                        yield _mark_stream_activity(_decorate_stream_event(event))
                        if _stop_if_total_timeout():
                            break

            logger.debug(
                f"Stream loop exited: td={td.is_set()}, qempty={oq.empty()}, chunks_received={chunk_count}, full_content_len={len(full_content)}"
            )

            # Ensure background task completes
            try:
                await asyncio.wait_for(bg_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass

            self._drain_reasoning_chunks()
            tool_result_events, stream_tool_result_index = (
                AgentLoop._collect_stream_tool_result_events(
                    self,
                    stream_tool_result_index,
                    emitted_tool_result_ids,
                    tool_output_capture_scope=tool_output_capture_scope,
                    tool_call_arguments_by_id=tool_call_arguments_by_id,
                )
            )
            _record_tool_result_events(tool_result_events)
            if AgentLoop._tool_events_have_history_search_budget(tool_result_events):
                history_search_budget_seen = True
            for event in tool_result_events:
                yield _decorate_stream_event(event)
            for event in _drain_runtime_notice_events():
                yield event

            if leaked_tool_protocol_detected and not pseudo_tool_repair_attempted:
                logger.warning(
                    "Stream contained leaked tool-call protocol text; running "
                    "internal tool-call repair."
                )
                pseudo_tool_repair_attempted = True
                repair_state = {}
                async for event in _emit_repair_event_stream(
                    _stream_pseudo_tool_call_repair(
                        leaked_tool_protocol_buffer
                        or "Leaked tool-call protocol text was suppressed.",
                        result_holder=repair_state,
                        repair_reason="tool_call_protocol_leak",
                    ),
                    visible_tool_result_delta=True,
                ):
                    yield event
                repair_text = str(repair_state.get("text") or "")
                run_result_text = repair_text
                post_tool_content_buffer = repair_text
                post_tool_content_events = []
                leaked_tool_protocol_buffer = ""
                leaked_tool_protocol_detected = False
                leaked_tool_protocol_probe = ""

            if history_search_budget_seen and not history_search_budget_recovery_attempted:
                logger.warning(
                    "History search budget consumed the tool loop; running "
                    "internal continuation recovery."
                )
                history_search_budget_recovery_attempted = True
                repair_prompt = AgentLoop._build_history_search_budget_recovery_prompt(
                    authoritative_message,
                    all_tool_result_events,
                )
                repair_state: dict[str, Any] = {}
                async for event in _emit_repair_event_stream(
                    _stream_pseudo_tool_call_repair(
                        "History search budget reached before task completion.",
                        result_holder=repair_state,
                        repair_prompt_override=repair_prompt,
                        repair_reason="history_search_budget_recovery",
                        request_hint_overrides={"history_search_budget_exhausted": True},
                    ),
                    visible_tool_result_delta=True,
                ):
                    yield event
                repair_text = str(repair_state.get("text") or "")
                run_result_text = repair_text
                post_tool_content_buffer = repair_text
                post_tool_content_events = []

            if all_tool_result_events or should_run_skill_contract_check(
                all_tool_result_events,
            ):
                buffer_stream_content = True

            repeated_read_storm = not latest_tool_event_has_user_summary_marker(
                all_tool_result_events
            ) and (
                _repeated_read_storm_active()
                or (not repeated_read_recovery_attempted and repeated_read_guardrail_seen)
            )
            if repeated_read_storm:
                logger.warning(
                    "Repeated read_file calls consumed the tool loop; "
                    "running internal continuation recovery."
                )
                repeated_read_recovery_attempted = True
                if _can_run_skill_contract_continuation() and skill_contract_needs_continuation(
                    run_result_text,
                    all_tool_result_events,
                ):
                    _mark_skill_contract_continuation_attempt()
                    repair_prompt = AgentLoop._build_skill_contract_continuation_prompt(
                        authoritative_message,
                        all_tool_result_events,
                        previous_draft=run_result_text or full_content,
                    )
                    repair_state = {}
                    repair_stream = _stream_pseudo_tool_call_repair(
                        run_result_text or "Repeated skill file reads stopped progress.",
                        result_holder=repair_state,
                        repair_prompt_override=repair_prompt,
                        repair_reason="skill_contract_continuation",
                    )
                    tool_loop_fallback_active = False
                else:
                    repair_state = {}
                    repair_stream = _stream_repeated_read_recovery(result_holder=repair_state)
                async for event in _emit_repair_event_stream(
                    repair_stream,
                    visible_tool_result_delta=True,
                ):
                    yield event
                repair_text = str(repair_state.get("text") or "")
                run_result_text = repair_text
                post_tool_content_buffer = repair_text
                post_tool_content_events = []

            if (
                post_tool_result_silence_recovery_seen
                and not latest_tool_event_has_user_summary_marker(all_tool_result_events)
                and _can_run_skill_contract_continuation()
                and (
                    skill_contract_needs_continuation(
                        run_result_text,
                        all_tool_result_events,
                    )
                    or skill_contract_inspection_stalled_after_progress(all_tool_result_events)
                )
            ):
                logger.warning(
                    "Tool result silence paused a skill turn before completion; "
                    "running internal skill-contract continuation."
                )
                _mark_skill_contract_continuation_attempt()
                repair_prompt = AgentLoop._build_skill_contract_continuation_prompt(
                    authoritative_message,
                    all_tool_result_events,
                    previous_draft=run_result_text or full_content,
                )
                repair_state = {}
                async for event in _emit_repair_event_stream(
                    _stream_pseudo_tool_call_repair(
                        run_result_text or "Provider was silent after skill tool results.",
                        result_holder=repair_state,
                        repair_prompt_override=repair_prompt,
                        repair_reason="skill_contract_continuation",
                    ),
                    visible_tool_result_delta=True,
                ):
                    yield event
                repair_text = str(repair_state.get("text") or "")
                run_result_text = repair_text
                post_tool_content_buffer = repair_text
                post_tool_content_events = []

            if pre_tool_scratchpad_buffer and not saw_tool_call:
                for buffered_event in pre_tool_scratchpad_events:
                    buffered_delta = AgentLoop._mask_user_visible_text(
                        str(buffered_event.get("delta") or "")
                    )
                    if not buffered_delta:
                        continue
                    full_content += buffered_delta
                    emitted_content_text += buffered_delta
                    yield _decorate_stream_event(
                        {
                            **buffered_event,
                            "delta": buffered_delta,
                        }
                    )
                pre_tool_scratchpad_events = []
                pre_tool_scratchpad_buffer = ""

            if post_tool_content_buffer:
                pending_content = AgentLoop._strip_leaked_scratchpad_prefix(
                    post_tool_content_buffer
                )
                post_tool_content_events = []
                post_tool_content_buffer = ""
                if AgentLoop._is_internal_completion_sentinel(pending_content):
                    pending_content = ""
                if (
                    pending_content.strip()
                    and AgentLoop._looks_like_pseudo_tool_call_text(pending_content)
                    and not pseudo_tool_repair_attempted
                ):
                    logger.warning(
                        "Stream content contained tool-call-shaped Markdown instead "
                        "of actual tool calls; retrying once with an internal repair prompt."
                    )
                    pseudo_tool_repair_attempted = True
                    repair_state = {}
                    async for event in _emit_repair_event_stream(
                        _stream_pseudo_tool_call_repair(
                            pending_content,
                            result_holder=repair_state,
                        ),
                    ):
                        yield event
                    repair_text = str(repair_state.get("text") or "")
                    run_result_text = repair_text
                    pending_content = ""
                if pending_content.strip() and not AgentLoop._looks_like_internal_scratchpad_text(
                    pending_content
                ):
                    pending_content = AgentLoop._finalize_response_content(
                        self,
                        authoritative_message,
                        pending_content,
                        turn_memory_start_index=_pre_turn_memory_index,
                    )
                    if not pending_content.strip():
                        pending_content = ""
                if pending_content.strip():
                    pending_content = AgentLoop._trim_repeated_stream_prefix(
                        pending_content,
                        emitted_content_text,
                    )
                    pending_content = AgentLoop._mask_user_visible_text(pending_content)
                    if not pending_content.strip():
                        pending_content = ""
                if pending_content.strip():
                    saw_content_after_tool_call = True
                    streamed_content_after_latest_tool_call = True
                    full_content += pending_content
                    if not buffer_stream_content:
                        yield _decorate_stream_event(
                            {
                                "type": "content",
                                "delta": pending_content,
                                "metadata": {"validated": True},
                            }
                        )
                        emitted_content_text += pending_content

            if (
                tool_loop_fallback_active
                and not latest_tool_event_has_user_summary_marker(all_tool_result_events)
                and _can_run_skill_contract_continuation()
                and skill_contract_needs_continuation(
                    run_result_text,
                    all_tool_result_events,
                )
            ):
                logger.warning(
                    "Tool-loop guard stopped a skill turn after contract setup; "
                    "running internal skill-contract continuation."
                )
                _mark_skill_contract_continuation_attempt()
                repair_prompt = AgentLoop._build_skill_contract_continuation_prompt(
                    authoritative_message,
                    all_tool_result_events,
                    previous_draft=run_result_text or full_content,
                )
                repair_state = {}
                async for event in _emit_repair_event_stream(
                    _stream_pseudo_tool_call_repair(
                        run_result_text or "Tool guard stopped before skill-contract progress.",
                        result_holder=repair_state,
                        repair_prompt_override=repair_prompt,
                        repair_reason="skill_contract_continuation",
                    ),
                ):
                    yield event
                repair_text = str(repair_state.get("text") or "")
                if repair_text.strip():
                    run_result_text = repair_text
                    full_content = repair_text
                    pending_fallback_content_emit = True
                    pending_fallback_reason = "skill_contract_continuation"
                    pending_fallback_delta = repair_text
                    tool_loop_fallback_active = False

            fallback_after_tool_only_preamble = (
                saw_tool_call
                and not saw_content_after_tool_call
                and bool(run_result_text)
                and self._normalize_comparable_text(full_content)
                != self._normalize_comparable_text(run_result_text)
            )

            # Fallback: if run() completed but no stream chunks were emitted,
            # or only a pre-tool preamble was emitted, use the final run result
            # to avoid ending the stream without the actual answer text.
            if AgentLoop._is_internal_completion_sentinel(run_result_text):
                run_result_text = ""
            if run_result_text and (not full_content or fallback_after_tool_only_preamble):
                logger.warning(
                    "Stream produced no content chunks; falling back to run() result text."
                )
                fallback_delta = run_result_text
                if full_content and fallback_after_tool_only_preamble and tool_loop_fallback_active:
                    full_content = run_result_text
                    fallback_reason = "tool_loop_guardrail"
                elif full_content and fallback_after_tool_only_preamble:
                    full_content, fallback_delta = AgentLoop._resolve_stream_fallback_delta(
                        full_content,
                        run_result_text,
                    )
                    fallback_reason = "run_result_after_tool_preamble"
                else:
                    full_content = run_result_text
                    fallback_reason = "run_result_no_chunks"
                sanitized_full_content = AgentLoop._strip_leaked_scratchpad_prefix(full_content)
                if sanitized_full_content != full_content:
                    full_content = sanitized_full_content
                    fallback_delta = sanitized_full_content
                full_content = AgentLoop._mask_user_visible_text(full_content)
                fallback_delta = AgentLoop._mask_user_visible_text(fallback_delta)
                if fallback_delta:
                    pending_fallback_content_emit = True
                    pending_fallback_reason = fallback_reason
                    pending_fallback_delta = fallback_delta

            if (
                full_content.strip()
                and AgentLoop._looks_like_pseudo_tool_call_text(full_content)
                and not pseudo_tool_repair_attempted
            ):
                logger.warning(
                    "Final stream content contained tool-call-shaped Markdown instead "
                    "of actual tool calls; retrying once with an internal repair prompt."
                )
                pseudo_tool_repair_attempted = True
                repair_state = {}
                async for event in _emit_repair_event_stream(
                    _stream_pseudo_tool_call_repair(
                        full_content,
                        result_holder=repair_state,
                    ),
                ):
                    yield event
                repair_text = str(repair_state.get("text") or "")
                full_content = repair_text
                pending_fallback_content_emit = True
                pending_fallback_reason = "pseudo_tool_call_repair"
                pending_fallback_delta = repair_text

            def _has_stateful_terminal_content(text: str | None) -> bool:
                active_ledger = getattr(self, "_active_execution_ledger", None)
                normalized = str(text or "").strip()
                return (
                    isinstance(active_ledger, ExecutionLedger)
                    and active_ledger.has_stateful_progress()
                    and bool(normalized)
                    and normalized not in {"No results", "NO_CONCISE_TOOL_EVIDENCE"}
                )

            while (
                not _has_stateful_terminal_content(full_content)
                and _can_run_skill_contract_continuation()
                and skill_contract_needs_continuation(
                    full_content,
                    all_tool_result_events,
                )
            ):
                before_fingerprint = AgentLoop._skill_contract_evidence_fingerprint(
                    all_tool_result_events,
                )
                logger.warning(
                    "Final stream content stopped after skill setup/read-only work; "
                    "continuing with an internal skill-contract continuation prompt."
                )
                _mark_skill_contract_continuation_attempt()
                repair_prompt = AgentLoop._build_skill_contract_continuation_prompt(
                    authoritative_message,
                    all_tool_result_events,
                    previous_draft=full_content,
                )
                repair_state = {}
                repair_events: list[dict[str, Any]] = []
                async for event in _emit_repair_event_stream(
                    _stream_pseudo_tool_call_repair(
                        full_content,
                        result_holder=repair_state,
                        repair_prompt_override=repair_prompt,
                        repair_reason="skill_contract_continuation",
                    ),
                    collected_events=repair_events,
                ):
                    yield event
                repair_text = str(repair_state.get("text") or "")
                if (
                    not repair_text.strip() or repair_text.strip() == "NO_CONCISE_TOOL_EVIDENCE"
                ) and any(event.get("type") == "tool_result" for event in repair_events):
                    repair_text = build_user_facing_tool_event_answer(
                        repair_events,
                        user_message=authoritative_message,
                    )
                if repair_text.strip():
                    full_content = repair_text
                    pending_fallback_content_emit = True
                    pending_fallback_reason = "skill_contract_continuation"
                    pending_fallback_delta = repair_text
                after_fingerprint = AgentLoop._skill_contract_evidence_fingerprint(
                    all_tool_result_events,
                )
                if after_fingerprint == before_fingerprint:
                    logger.warning(
                        "Skill-contract continuation produced no new tool "
                        "evidence; stopping internal continuation and summarizing state."
                    )
                    break

            task_continuation_attempts = 0
            while (
                all_tool_result_events
                and task_continuation_attempts
                < AgentLoop._task_completion_continuation_attempt_limit()
                and _can_auto_continue_for_current_request()
            ):
                verdict = await self._evaluate_task_completion_verdict(
                    authoritative_message=authoritative_message,
                    final_content=full_content,
                    tool_result_events=all_tool_result_events,
                )
                if not isinstance(verdict, dict) or verdict.get("status") != "needs_continuation":
                    break

                before_fingerprint = AgentLoop._skill_contract_evidence_fingerprint(
                    all_tool_result_events,
                )
                task_continuation_attempts += 1
                logger.warning(
                    "Final stream content did not satisfy tool-backed request; "
                    "continuing with a generic task continuation prompt."
                )
                repair_prompt = AgentLoop._build_task_continuation_prompt(
                    authoritative_message,
                    all_tool_result_events,
                    previous_draft=full_content,
                    continuation_reason=verdict.get("reason"),
                    continuation_focus=verdict.get("next_focus"),
                )
                repair_state = {}
                repair_events = []
                async for event in _emit_repair_event_stream(
                    _stream_pseudo_tool_call_repair(
                        full_content,
                        result_holder=repair_state,
                        repair_prompt_override=repair_prompt,
                        repair_reason="task_continuation",
                    ),
                    visible_tool_result_delta=True,
                    collected_events=repair_events,
                ):
                    yield event
                repair_text = str(repair_state.get("text") or "")
                if (
                    not repair_text.strip() or repair_text.strip() == "NO_CONCISE_TOOL_EVIDENCE"
                ) and any(event.get("type") == "tool_result" for event in repair_events):
                    repair_text = build_user_facing_tool_event_answer(
                        repair_events,
                        user_message=authoritative_message,
                    )
                after_fingerprint = AgentLoop._skill_contract_evidence_fingerprint(
                    all_tool_result_events,
                )
                if after_fingerprint == before_fingerprint:
                    logger.warning(
                        "Generic task continuation produced no new tool evidence; "
                        "stopping continuation."
                    )
                    existing_terminal_content = str(full_content or "").strip()
                    if (
                        existing_terminal_content
                        and existing_terminal_content
                        not in {"No results", "NO_CONCISE_TOOL_EVIDENCE"}
                        and not AgentLoop._looks_like_internal_scratchpad_text(
                            existing_terminal_content
                        )
                    ):
                        pending_fallback_content_emit = True
                        pending_fallback_reason = (
                            pending_fallback_reason or "preserved_terminal_content"
                        )
                        pending_fallback_delta = existing_terminal_content
                        break
                    active_ledger = getattr(self, "_active_execution_ledger", None)
                    if isinstance(active_ledger, ExecutionLedger):
                        if not (active_ledger.has_stateful_progress() or active_ledger.file_reads):
                            active_ledger.record_blocker(
                                tool_name="agent_loop",
                                reason="continuation_without_tool_progress",
                                summary=(
                                    "The request still needed another tool step, but "
                                    "the internal continuation produced no new verified "
                                    "tool evidence."
                                ),
                            )
                        ledger_summary = active_ledger.render_user_facing_summary(max_chars=5000)
                    else:
                        ledger_summary = ""
                    full_content = ledger_summary or build_user_facing_tool_event_answer(
                        all_tool_result_events,
                        incomplete=True,
                        user_message=authoritative_message,
                    )
                    pending_fallback_content_emit = True
                    pending_fallback_reason = pending_fallback_reason or "continuation_no_progress"
                    pending_fallback_delta = full_content
                    break
                if repair_text.strip():
                    full_content = repair_text
                    pending_fallback_content_emit = True
                    pending_fallback_reason = pending_fallback_reason or "task_continuation"
                    pending_fallback_delta = repair_text

            if (
                should_run_skill_contract_check(all_tool_result_events)
                and skill_contract_has_progress(all_tool_result_events)
                and not latest_tool_event_has_user_summary_marker(all_tool_result_events)
                and not latest_tool_event_from_skill_continuation(all_tool_result_events)
                and not streamed_content_after_latest_tool_call
            ):
                full_content = await AgentLoop._synthesize_final_answer_from_tool_events(
                    self,
                    all_tool_result_events,
                    user_message=authoritative_message,
                    fallback_text=build_user_facing_tool_event_answer(
                        all_tool_result_events,
                        user_message=authoritative_message,
                    ),
                )
                pending_fallback_content_emit = True
                pending_fallback_reason = pending_fallback_reason or "skill_progress_summary"
                pending_fallback_delta = full_content

            if (
                full_content.strip()
                and should_run_skill_contract_check(all_tool_result_events)
                and latest_tool_event_has_user_summary_marker(all_tool_result_events)
                and not streamed_content_after_latest_tool_call
            ):
                full_content = await AgentLoop._synthesize_final_answer_from_tool_events(
                    self,
                    all_tool_result_events,
                    user_message=authoritative_message,
                    fallback_text=build_user_facing_tool_event_answer(
                        all_tool_result_events,
                        user_message=authoritative_message,
                    ),
                )
                pending_fallback_content_emit = True
                pending_fallback_reason = pending_fallback_reason or "skill_summary"
                pending_fallback_delta = full_content

            if stream_error_reason is not None:
                if all_tool_result_events and (
                    AgentLoop._should_replace_stream_error_preamble(
                        full_content,
                        saw_tool_call=saw_tool_call,
                        saw_content_after_tool_call=saw_content_after_tool_call,
                    )
                ):
                    full_content = await AgentLoop._synthesize_final_answer_from_tool_events(
                        self,
                        all_tool_result_events,
                        incomplete=True,
                        user_message=authoritative_message,
                        fallback_text=build_user_facing_tool_event_answer(
                            all_tool_result_events,
                            incomplete=True,
                            user_message=authoritative_message,
                        ),
                    )
                    pending_fallback_content_emit = True
                    pending_fallback_reason = "runtime_error_tool_evidence"
                    pending_fallback_delta = full_content
                elif not full_content.strip():
                    if isinstance(stream_error_reason, Exception):
                        friendly_error = user_friendly_error(stream_error_reason)
                    else:
                        friendly_error = "An unexpected error occurred. Please try again."
                    full_content = friendly_error
                    pending_fallback_content_emit = True
                    pending_fallback_reason = "runtime_error"
                    pending_fallback_delta = full_content

            if tool_loop_fallback_active and all_tool_result_events and full_content.strip():
                full_content = await AgentLoop._synthesize_final_answer_from_tool_events(
                    self,
                    all_tool_result_events,
                    incomplete=True,
                    user_message=authoritative_message,
                    fallback_text=full_content,
                )
                pending_fallback_content_emit = True
                pending_fallback_reason = pending_fallback_reason or "tool_loop_guardrail"
                pending_fallback_delta = full_content

            if full_content.strip() and AgentLoop._looks_like_raw_tool_transcript_leak(
                full_content
            ):
                synthesis_events = (
                    all_tool_result_events
                    or self._collect_runtime_tool_result_events_from_memory(_pre_turn_memory_index)
                )
                if synthesis_events:
                    all_tool_result_events = all_tool_result_events or synthesis_events
                    full_content = await AgentLoop._synthesize_final_answer_from_tool_events(
                        self,
                        synthesis_events,
                        user_message=authoritative_message,
                        fallback_text=build_user_facing_tool_event_answer(
                            synthesis_events,
                            user_message=authoritative_message,
                        ),
                    )
                    pending_fallback_content_emit = True
                    pending_fallback_reason = pending_fallback_reason or "raw_tool_transcript"
                    pending_fallback_delta = full_content

            if (
                all_tool_result_events
                and full_content.strip()
                and AgentLoop._looks_like_raw_tool_transcript_leak(full_content)
            ):
                full_content = await AgentLoop._synthesize_final_answer_from_tool_events(
                    self,
                    all_tool_result_events,
                    user_message=authoritative_message,
                    fallback_text=build_user_facing_tool_event_answer(
                        all_tool_result_events,
                        user_message=authoritative_message,
                    ),
                )
                pending_fallback_content_emit = True
                pending_fallback_reason = pending_fallback_reason or "raw_tool_transcript"
                pending_fallback_delta = full_content

            if (
                all_tool_result_events
                and full_content.strip()
                and AgentLoop._final_answer_script_mismatch(
                    authoritative_message,
                    full_content,
                )
            ):
                full_content = await AgentLoop._synthesize_final_answer_from_tool_events(
                    self,
                    all_tool_result_events,
                    incomplete=tool_events_need_more_evidence(all_tool_result_events),
                    user_message=authoritative_message,
                    fallback_text=build_user_facing_tool_event_answer(
                        all_tool_result_events,
                        incomplete=tool_events_need_more_evidence(all_tool_result_events),
                        user_message=authoritative_message,
                    ),
                )
                pending_fallback_content_emit = True
                pending_fallback_reason = pending_fallback_reason or "language_mismatch"
                pending_fallback_delta = full_content
            elif (
                all_tool_result_events
                and (
                    bool(request_execution_hints.get("current_session_fact_check_required"))
                    or bool(request_execution_hints.get("session_evidence_synthesis_preferred"))
                )
                and not streamed_content_after_latest_tool_call
            ):
                synthesis_events = self._session_evidence_synthesis_events(
                    all_tool_result_events,
                    user_message=authoritative_message,
                )
                full_content = await AgentLoop._synthesize_final_answer_from_tool_events(
                    self,
                    synthesis_events,
                    incomplete=tool_events_need_more_evidence(synthesis_events),
                    user_message=authoritative_message,
                    fallback_text=build_user_facing_tool_event_answer(
                        synthesis_events,
                        incomplete=tool_events_need_more_evidence(synthesis_events),
                        user_message=authoritative_message,
                    ),
                )
                pending_fallback_content_emit = True
                pending_fallback_reason = pending_fallback_reason or "session_evidence"
                pending_fallback_delta = full_content
            elif (
                all_tool_result_events
                and should_run_skill_contract_check(all_tool_result_events)
                and not skill_contract_has_progress(all_tool_result_events)
                and not streamed_content_after_latest_tool_call
            ):
                full_content = await AgentLoop._synthesize_final_answer_from_tool_events(
                    self,
                    all_tool_result_events,
                    incomplete=tool_events_need_more_evidence(all_tool_result_events),
                    user_message=authoritative_message,
                    fallback_text=build_user_facing_tool_event_answer(
                        all_tool_result_events,
                        incomplete=tool_events_need_more_evidence(all_tool_result_events),
                        user_message=authoritative_message,
                    ),
                )
                pending_fallback_content_emit = True
                pending_fallback_reason = pending_fallback_reason or "skill_evidence_summary"
                pending_fallback_delta = full_content

            if (
                full_content.strip() == "NO_CONCISE_TOOL_EVIDENCE"
                and all_tool_result_events
                and latest_tool_event_from_skill_continuation(all_tool_result_events)
            ):
                full_content = build_user_facing_tool_event_answer(
                    all_tool_result_events,
                    user_message=authoritative_message,
                )
                pending_fallback_content_emit = True
                pending_fallback_reason = pending_fallback_reason or "skill_continuation_evidence"
                pending_fallback_delta = full_content

            active_ledger = getattr(self, "_active_execution_ledger", None)
            ledger_final = AgentLoop._deterministic_ledger_final(
                active_ledger,
                max_chars=5000,
            )
            if (
                all_tool_result_events
                and ledger_final
                and AgentLoop._should_use_deterministic_ledger_final(
                    full_content,
                    user_message=authoritative_message,
                )
            ):
                full_content = await AgentLoop._synthesize_final_answer_from_tool_events(
                    self,
                    all_tool_result_events,
                    incomplete=tool_events_need_more_evidence(all_tool_result_events),
                    user_message=authoritative_message,
                    fallback_text=ledger_final,
                )
                pending_fallback_content_emit = True
                pending_fallback_reason = (
                    pending_fallback_reason or "execution_ledger_deterministic"
                )
                pending_fallback_delta = full_content
            elif (
                all_tool_result_events
                and isinstance(active_ledger, ExecutionLedger)
                and active_ledger.has_stateful_progress()
                and (active_ledger.file_writes or active_ledger.services)
                and AgentLoop._needs_tool_evidence_synthesis(
                    full_content,
                    user_message=authoritative_message,
                )
            ):
                ledger_fallback = active_ledger.render_user_facing_summary(max_chars=5000)
                if ledger_fallback:
                    full_content = await AgentLoop._synthesize_final_answer_from_tool_events(
                        self,
                        all_tool_result_events,
                        user_message=authoritative_message,
                        fallback_text=ledger_fallback,
                    )
                    pending_fallback_content_emit = True
                    pending_fallback_reason = pending_fallback_reason or "execution_ledger"
                    pending_fallback_delta = full_content

            pre_finalize_full_content = full_content
            final_content = AgentLoop._finalize_response_content(
                self,
                authoritative_message,
                full_content,
                turn_memory_start_index=_pre_turn_memory_index,
            )
            final_content_changed = self._normalize_comparable_text(
                final_content
            ) != self._normalize_comparable_text(pre_finalize_full_content)
            if (
                (
                    buffer_stream_content
                    and (not emitted_content_text.strip() or final_content_changed)
                )
                or pending_fallback_content_emit
            ) and final_content:
                emit_delta = final_content
                if pending_fallback_content_emit and self._normalize_comparable_text(
                    final_content
                ) == self._normalize_comparable_text(pre_finalize_full_content):
                    emit_delta = pending_fallback_delta or final_content
                trimmed_emit_delta = AgentLoop._trim_repeated_stream_prefix(
                    emit_delta,
                    emitted_content_text,
                )
                trimmed_emit_delta = AgentLoop._mask_user_visible_text(trimmed_emit_delta)
                emit_delta = trimmed_emit_delta
                emit_metadata = {
                    "buffered": bool(buffer_stream_content),
                    "validated": True,
                }
                if pending_fallback_reason:
                    emit_metadata["fallback"] = pending_fallback_reason
                if emit_delta.strip():
                    yield _decorate_stream_event(
                        {
                            "type": "content",
                            "delta": emit_delta,
                            "metadata": emit_metadata,
                        }
                    )
            full_content = final_content

            # Emit done
            stream_completed = True
            yield {
                "type": "done",
                "delta": "",
                "metadata": {"content": full_content},
                "source": current_source,
            }

        except asyncio.CancelledError:
            stream_cancelled = True
            logger.warning("Streaming cancelled")
            AgentLoop._drain_agent_output_queue(self)
            AgentLoop._truncate_runtime_memory(self, _pre_turn_memory_index)
            _persist_interrupted_stream_reply("task_cancelled")
            AgentLoop._mark_latest_user_turn_state(
                self,
                _TURN_STATE_INTERRUPTED,
                reason="task_cancelled",
            )
            current_source = AgentLoop.get_last_response_source(self)
            yield {
                "type": "cancelled",
                "delta": "Task cancelled.",
                "metadata": {
                    "cancelled": True,
                    "code": "CANCELLED",
                    "error_code": "CANCELLED",
                    "message": "Task cancelled.",
                    "reason": "task_cancelled",
                },
                "source": current_source,
            }
            raise
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            stream_error_reason = e
            stream_completed = True
            current_source = AgentLoop.get_last_response_source(self)
            error_metadata = {
                "error": str(e),
                "message": str(e),
                "code": type(e).__name__,
                "error_code": type(e).__name__,
                "reason": "stream_failed",
            }
            yield {
                "type": "error",
                "delta": str(e),
                "metadata": error_metadata,
                "source": current_source,
            }
            yield {
                "type": "done",
                "delta": "",
                "metadata": error_metadata,
                "source": current_source,
            }
        finally:
            if execution_ledger is not None:
                try:
                    persist_execution_ledger(execution_ledger)
                except Exception as exc:
                    logger.debug(f"Execution ledger persistence skipped: {exc}")
            if ledger_manager is not None:
                try:
                    ledger_manager.__exit__(None, None, None)
                except Exception:
                    pass
            if getattr(self, "_active_execution_ledger", None) is execution_ledger:
                self._active_execution_ledger = None
            if getattr(self, "_active_turn_memory_start_index", None) == _pre_turn_memory_index:
                self._active_turn_memory_start_index = None
            if capture_manager is not None:
                try:
                    capture_manager.__exit__(None, None, None)
                except ValueError as exc:
                    logger.debug(f"Tool output capture context cleanup skipped: {exc}")
                    clear_captured_tool_outputs(tool_output_capture_scope)
            self._restore_agent_think()
            AgentLoop._restore_request_context_system_prompt(
                self,
                original_system_prompt,
                original_base_system_prompt,
            )
            if bg_task is not None and not bg_task.done():
                bg_task.cancel()
                try:
                    cleanup_timeout = 2.0 if stream_cancelled else 5.0
                    await asyncio.wait_for(bg_task, timeout=cleanup_timeout)
                except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                    pass
            if not stream_completed and not stream_cancelled:
                _persist_interrupted_stream_reply("task_cancelled")
                AgentLoop._mark_latest_user_turn_state(
                    self,
                    _TURN_STATE_INTERRUPTED,
                    reason="task_cancelled",
                )

        if stream_error_reason is not None and not stream_cancelled:
            AgentLoop._persist_failed_turn_context(
                self,
                label="stream",
                reason=stream_error_reason,
                start_index=_pre_turn_memory_index,
            )

        if (
            full_content
            and stream_completed
            and not stream_cancelled
            and stream_error_reason is None
        ):
            try:
                self._merge_turn_invoked_skills_from_runtime(_pre_turn_memory_index)
                AgentLoop._mark_latest_user_turn_state(
                    self,
                    _TURN_STATE_COMPLETED,
                )
                # Persist intermediate tool-call artifacts generated while
                # streaming so they remain searchable even after runtime
                # context compaction.
                self._persist_turn_tool_trace(_pre_turn_memory_index)
                self._session.add_message(
                    "assistant",
                    full_content,
                    **AgentLoop._assistant_session_save_kwargs(full_content),
                )
                AgentLoop._persist_session_if_possible(self)
            except Exception as e:
                logger.warning(f"Failed to save session after streaming: {e}")

    async def process_with_thinking(
        self,
        message: str,
        media: list[str] | None = None,
        attachments: list[dict[str, Any]] | None = None,
        session_key: str | None = None,
        thinking_level: str | None = None,
        channel: str | None = None,
        metadata: dict[str, Any] | None = None,
        reply_to: str | None = None,
        reasoning_effort: str | None = None,
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

        if not self._initialized:
            await self.initialize()

        # Switch session if a different key is requested
        if session_key and session_key != self.session_key:
            self._session = self.sessions.get_or_create(session_key)
            self.session_key = session_key
            if getattr(self, "_history_search_tool", None) is not None:
                try:
                    self._history_search_tool.set_default_session_key(session_key)
                except Exception:
                    pass
            logger.debug(f"Switched to session: {session_key}")

        current_session_key = getattr(self, "session_key", "default")
        self.set_subagent_context(
            session_key=current_session_key,
            channel=channel,
            metadata=metadata,
            reply_to=reply_to,
        )
        logger.info(f"Processing message (with thinking): {message[:100]}...")
        self._reset_reasoning_capture()
        AgentLoop._reset_runtime_notices(self)
        effective_reasoning_effort = reasoning_effort or getattr(self, "reasoning_effort", None)
        original_system_prompt: str | None = None
        original_base_system_prompt: object = _MISSING
        await self._install_skill_zip_attachments(attachments or [])

        # Refresh memory context
        try:
            memory_context = self.memory.get_memory_context()
            if memory_context:
                self.context.set_memory_context(memory_context)
        except Exception as e:
            logger.warning(f"Failed to load memory context: {e}")

        # Trim and inject persisted history into runtime memory
        await self._prepare_request_context(message)
        self._prepare_agent_for_new_turn()
        authoritative_message = message

        runtime_user_text = self._add_current_turn_skill_zip_context(
            self._build_current_turn_runtime_user_text(authoritative_message)
        )
        runtime_message = self._build_runtime_message_content(
            "user",
            runtime_user_text,
            media=media,
            attachments=attachments,
        )
        if isinstance(runtime_message, str):
            message = runtime_message
        AgentLoop._persist_user_turn_to_session(
            self,
            authoritative_message,
            media=media,
            attachments=attachments,
        )

        execution_ledger: ExecutionLedger | None = None
        ledger_manager = None
        _pre_turn_memory_index: int | None = None

        try:
            retry_runner = AgentLoop._resolve_retry_runner(self)
            _base_prompt = self._select_next_step_prompt(authoritative_message, thinking=True)
            original_system_prompt, original_base_system_prompt = (
                AgentLoop._apply_request_context_to_system_prompt(
                    self,
                    authoritative_message,
                    thinking=True,
                )
            )
            self._agent.next_step_prompt = _base_prompt
            self._install_anti_loop_tracker(_base_prompt)
            await self._agent.add_message("user", runtime_message)
            self._normalize_runtime_memory_before_run("process_with_thinking")
            _pre_turn_memory_index = self._runtime_memory_snapshot_index()
            self._active_turn_memory_start_index = _pre_turn_memory_index
            execution_ledger = ExecutionLedger(
                owner=self._current_tool_owner_key(),
                workspace=str(getattr(self, "workspace", "") or ""),
                session_id=str(getattr(self, "session_key", "") or ""),
                turn_id=uuid.uuid4().hex,
                user_request=authoritative_message,
            )
            self._active_execution_ledger = execution_ledger
            ledger_manager = bind_execution_ledger(execution_ledger)
            ledger_manager.__enter__()

            # Run agent with thinking enabled
            run_kwargs: dict[str, Any] = {}
            requested_thinking = thinking_level or True
            if self._callable_accepts_kwarg(self._agent.run, "thinking"):
                run_kwargs["thinking"] = requested_thinking
            if effective_reasoning_effort and self._callable_accepts_kwarg(
                self._agent.run, "reasoning_effort"
            ):
                run_kwargs["reasoning_effort"] = effective_reasoning_effort
            request_execution_hints = self._build_request_execution_hints(authoritative_message)
            with (
                bind_request_execution_hints(request_execution_hints),
                track_tool_invocations(),
            ):
                result = await AgentLoop._run_agent_with_context_overflow_recovery(
                    self,
                    label="process_with_thinking",
                    retry_runner=retry_runner,
                    **run_kwargs,
                )

            # Extract content and thinking
            final_content = AgentLoop._extract_run_result_text(result)

            thinking_content = None
            if hasattr(result, "thinking_content"):
                thinking_content = result.thinking_content
            elif hasattr(result, "thinking"):
                thinking_content = result.thinking
            elif hasattr(result, "metadata") and isinstance(result.metadata, dict):
                thinking_content = result.metadata.get("thinking") or result.metadata.get(
                    "reasoning"
                )
            if AgentLoop._looks_like_pseudo_tool_call_text(final_content):
                logger.warning(
                    "Agent returned tool-call-shaped Markdown instead of actual "
                    "tool calls during thinking run; retrying once with an internal "
                    "repair prompt."
                )
                AgentLoop._drop_pseudo_tool_call_assistant_messages(
                    self,
                    _pre_turn_memory_index,
                )
                AgentLoop._drain_agent_output_queue(self)
                self._reset_agent_state_for_retry()
                repair_prompt = AgentLoop._build_pseudo_tool_call_repair_prompt(
                    authoritative_message,
                    final_content,
                )
                await self._agent.add_message("user", repair_prompt)
                self._agent.next_step_prompt = repair_prompt
                with (
                    bind_request_execution_hints(request_execution_hints),
                    track_tool_invocations(),
                ):
                    result = await AgentLoop._run_agent_with_context_overflow_recovery(
                        self,
                        label="process_with_thinking_tool_call_repair",
                        retry_runner=retry_runner,
                        **run_kwargs,
                    )
                final_content = AgentLoop._extract_run_result_text(result)
                thinking_content = None
                if hasattr(result, "thinking_content"):
                    thinking_content = result.thinking_content
                elif hasattr(result, "thinking"):
                    thinking_content = result.thinking
                elif hasattr(result, "metadata") and isinstance(result.metadata, dict):
                    thinking_content = result.metadata.get("thinking") or result.metadata.get(
                        "reasoning"
                    )

            tool_result_events = self._collect_runtime_tool_result_events_from_memory(
                _pre_turn_memory_index
            )
            if AgentLoop._tool_events_have_history_search_budget(tool_result_events):
                final_content = await self._run_process_history_search_budget_recovery(
                    authoritative_message=authoritative_message,
                    request_execution_hints=request_execution_hints,
                    tool_result_events=tool_result_events,
                    retry_runner=retry_runner,
                    run_kwargs=run_kwargs,
                    label="process_with_thinking_history_search_budget_recovery",
                )
                tool_result_events = self._collect_runtime_tool_result_events_from_memory(
                    _pre_turn_memory_index
                )
                thinking_content = None

            if should_run_skill_contract_check(
                tool_result_events
            ) and AgentLoop._tool_events_have_repeated_read_guardrail(tool_result_events):
                final_content = await self._run_process_repeated_read_recovery(
                    authoritative_message=authoritative_message,
                    request_execution_hints=request_execution_hints,
                    retry_runner=retry_runner,
                    run_kwargs=run_kwargs,
                )
                tool_result_events = self._collect_runtime_tool_result_events_from_memory(
                    _pre_turn_memory_index
                )
                thinking_content = None

            if should_run_skill_contract_check(tool_result_events):
                (
                    final_content,
                    tool_result_events,
                ) = await self._continue_skill_contract_until_terminal(
                    authoritative_message=authoritative_message,
                    request_execution_hints=request_execution_hints,
                    final_content=final_content,
                    tool_result_events=tool_result_events,
                    retry_runner=retry_runner,
                    run_kwargs=run_kwargs,
                    memory_start_index=_pre_turn_memory_index,
                    label="process_with_thinking_skill_contract_continuation",
                )
                thinking_content = None

            if tool_result_events:
                final_content, tool_result_events = await self._continue_task_until_terminal(
                    authoritative_message=authoritative_message,
                    request_execution_hints=request_execution_hints,
                    final_content=final_content,
                    tool_result_events=tool_result_events,
                    retry_runner=retry_runner,
                    run_kwargs=run_kwargs,
                    memory_start_index=_pre_turn_memory_index,
                    label="process_with_thinking_task_continuation",
                )
                thinking_content = None

            if (
                tool_result_events
                and should_run_skill_contract_check(tool_result_events)
                and skill_contract_has_progress(tool_result_events)
                and not latest_tool_event_has_user_summary_marker(tool_result_events)
                and not latest_tool_event_from_skill_continuation(tool_result_events)
            ):
                final_content = await AgentLoop._synthesize_final_answer_from_tool_events(
                    self,
                    tool_result_events,
                    user_message=authoritative_message,
                    fallback_text=build_user_facing_tool_event_answer(
                        tool_result_events,
                        user_message=authoritative_message,
                    ),
                )
                thinking_content = None

            if any(
                AgentLoop._is_tool_loop_suppression_event(event) for event in tool_result_events
            ):
                fallback_content = AgentLoop._build_tool_loop_fallback_response(
                    tool_result_events,
                    reason="tool_suppression",
                    user_message=authoritative_message,
                )
                final_content = await AgentLoop._synthesize_final_answer_from_tool_events(
                    self,
                    tool_result_events,
                    incomplete=True,
                    user_message=authoritative_message,
                    fallback_text=fallback_content,
                )
            elif should_run_skill_contract_check(
                tool_result_events
            ) and latest_tool_event_has_user_summary_marker(tool_result_events):
                final_content = await AgentLoop._synthesize_final_answer_from_tool_events(
                    self,
                    tool_result_events,
                    user_message=authoritative_message,
                    fallback_text=build_user_facing_tool_event_answer(
                        tool_result_events,
                        user_message=authoritative_message,
                    ),
                )
            elif tool_result_events and (
                AgentLoop._looks_like_raw_tool_transcript_leak(final_content)
            ):
                final_content = await AgentLoop._synthesize_final_answer_from_tool_events(
                    self,
                    tool_result_events,
                    incomplete=tool_events_need_more_evidence(tool_result_events),
                    user_message=authoritative_message,
                    fallback_text=build_user_facing_tool_event_answer(
                        tool_result_events,
                        incomplete=tool_events_need_more_evidence(tool_result_events),
                        user_message=authoritative_message,
                    ),
                )
            elif tool_result_events and AgentLoop._final_answer_script_mismatch(
                authoritative_message,
                final_content,
            ):
                final_content = await AgentLoop._synthesize_final_answer_from_tool_events(
                    self,
                    tool_result_events,
                    incomplete=tool_events_need_more_evidence(tool_result_events),
                    user_message=authoritative_message,
                    fallback_text=build_user_facing_tool_event_answer(
                        tool_result_events,
                        incomplete=tool_events_need_more_evidence(tool_result_events),
                        user_message=authoritative_message,
                    ),
                )
            elif (
                tool_result_events
                and should_run_skill_contract_check(tool_result_events)
                and not skill_contract_has_progress(tool_result_events)
            ):
                final_content = await AgentLoop._synthesize_final_answer_from_tool_events(
                    self,
                    tool_result_events,
                    incomplete=tool_events_need_more_evidence(tool_result_events),
                    user_message=authoritative_message,
                    fallback_text=build_user_facing_tool_event_answer(
                        tool_result_events,
                        incomplete=tool_events_need_more_evidence(tool_result_events),
                        user_message=authoritative_message,
                    ),
                )
            active_ledger = getattr(self, "_active_execution_ledger", None)
            ledger_final = AgentLoop._deterministic_ledger_final(
                active_ledger,
                max_chars=5000,
            )
            if (
                tool_result_events
                and ledger_final
                and AgentLoop._should_use_deterministic_ledger_final(
                    final_content,
                    user_message=authoritative_message,
                )
            ):
                final_content = await AgentLoop._synthesize_final_answer_from_tool_events(
                    self,
                    tool_result_events,
                    incomplete=tool_events_need_more_evidence(tool_result_events),
                    user_message=authoritative_message,
                    fallback_text=ledger_final,
                )
            elif (
                tool_result_events
                and isinstance(active_ledger, ExecutionLedger)
                and active_ledger.has_stateful_progress()
                and (active_ledger.file_writes or active_ledger.services)
                and AgentLoop._needs_tool_evidence_synthesis(
                    final_content,
                    user_message=authoritative_message,
                )
            ):
                ledger_fallback = active_ledger.render_user_facing_summary(max_chars=5000)
                if ledger_fallback:
                    final_content = await AgentLoop._synthesize_final_answer_from_tool_events(
                        self,
                        tool_result_events,
                        user_message=authoritative_message,
                        fallback_text=ledger_fallback,
                    )
            if self._looks_like_duplicate_thinking(thinking_content, final_content):
                thinking_content = None
            final_content = AgentLoop._finalize_response_content(
                self,
                authoritative_message,
                final_content,
                turn_memory_start_index=_pre_turn_memory_index,
            )

        except asyncio.CancelledError:
            self._persist_cancelled_turn_context(start_index=_pre_turn_memory_index)
            raise
        except Exception as e:
            logger.error(f"Agent processing error: {e}")
            AgentLoop._persist_failed_turn_context(
                self,
                label="process_with_thinking",
                reason=e,
                start_index=_pre_turn_memory_index,
            )
            raise
        finally:
            if execution_ledger is not None:
                try:
                    persist_execution_ledger(execution_ledger)
                except Exception as exc:
                    logger.debug(f"Execution ledger persistence skipped: {exc}")
            if ledger_manager is not None:
                try:
                    ledger_manager.__exit__(None, None, None)
                except Exception:
                    pass
            if getattr(self, "_active_execution_ledger", None) is execution_ledger:
                self._active_execution_ledger = None
            if getattr(self, "_active_turn_memory_start_index", None) == _pre_turn_memory_index:
                self._active_turn_memory_start_index = None
            self._restore_agent_think()
            AgentLoop._restore_request_context_system_prompt(
                self,
                original_system_prompt,
                original_base_system_prompt,
            )

        try:
            self._merge_turn_invoked_skills_from_runtime(_pre_turn_memory_index)
            AgentLoop._mark_latest_user_turn_state(
                self,
                _TURN_STATE_COMPLETED,
            )
            # Persist intermediate tool-call artifacts (tool results and the
            # assistant messages that called them) so they remain searchable
            # even after runtime context compaction.
            self._persist_turn_tool_trace(_pre_turn_memory_index)
            self._session.add_message(
                "assistant",
                final_content,
                **AgentLoop._assistant_session_save_kwargs(final_content),
            )
            AgentLoop._persist_session_if_possible(self)
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

    def attach_cron_service(self, cron_service: Any | None) -> None:
        """Attach the active cron service so the cron tool can call it directly."""
        self._cron_service = cron_service
        tool = self.tools.get("cron")
        if tool is not None and hasattr(tool, "set_cron_service"):
            tool.set_cron_service(cron_service)

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.clear_session_history()
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

    def clear_session_history(self, session_key: str | None = None) -> str:
        """Clear a persisted session and keep the in-memory pointer in sync."""
        target_key = session_key or self.session_key
        session = self.sessions.get_or_create(target_key)
        session.clear()
        self.sessions.save(session)
        if target_key == self.session_key:
            self._session = session
        logger.info(f"Cleared session history: {target_key}")
        return target_key

    def build_creation_kwargs(self, **overrides: Any) -> dict[str, Any]:
        """Return kwargs that recreate this agent's runtime configuration."""
        kwargs: dict[str, Any] = {
            "workspace": self.workspace,
            "model": self.model,
            "provider": self.provider,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "max_iterations": self.max_iterations,
            "shell_timeout": self.shell_timeout,
            "shell_max_timeout": self.shell_max_timeout,
            "max_output": self.max_output,
            "session_key": self.session_key,
            "skill_paths": list(self._user_skill_paths),
            "mcp_config": copy.deepcopy(self._mcp_config),
            "system_prompt": self._system_prompt,
            "enable_skills": self._enable_skills,
            "auto_commit": self._auto_commit,
            "enabled_tools": (
                set(self._enabled_tools_override)
                if self._enabled_tools_override is not None
                else None
            ),
            "tool_profile": self._tool_profile,
            "session_store_backend": self._session_store_backend,
            "session_store_dsn": self._session_store_dsn,
            "session_store_db_path": self._session_store_db_path,
            "context_window": self.context_window,
            "memsearch_config": (
                self._memsearch_config.model_dump(mode="python")
                if isinstance(self._memsearch_config, MemSearchConfig)
                else None
            ),
            "auto_reload": self._auto_reload,
            "auto_reload_interval": self._auto_reload_interval,
            "config_path": self._config_path,
            "yolo_mode": self.yolo_mode,
            "provider_max_retries": self._config.provider_max_retries,
            "provider_retry_base_delay": self._config.provider_retry_base_delay,
            "provider_retry_max_delay": self._config.provider_retry_max_delay,
            "provider_retry_backoff_factor": self._config.provider_retry_backoff_factor,
        }
        kwargs.update(overrides)
        return kwargs

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
        logger.info("Stop requested - will be honoured on next process() call")
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


_loop_state.AgentLoop = AgentLoop
_loop_protocol.AgentLoop = AgentLoop
_loop_skills.AgentLoop = AgentLoop


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
    yolo_mode: bool = False,
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
        tool_profile: Named profile ('core', 'coding', 'research', 'full').
        yolo_mode: Operate directly in user's path without sandbox isolation.
        **kwargs: Additional arguments for AgentLoop.

    Returns:
        Initialized AgentLoop instance.

    Example:
        >>> agent = await create_agent()
        >>> response = await agent.process("Hello!")

        >>> # Load all tools
        >>> agent = await create_agent(tool_profile="full")

        >>> # YOLO mode - work in /home/user/project directly
        >>> agent = await create_agent(yolo_mode=True, workspace="/home/user/project")
    """
    try:
        ensure_wallet_runtime(workspace)
    except Exception:
        logger.warning("Wallet runtime bootstrap failed; continuing without built-in wallet env")
        import os

        os.environ["SPOON_BOT_WALLET_AUTO_CREATED"] = "0"

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
        yolo_mode=yolo_mode,
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
