"""
Agent loop: the core processing engine using spoon-core SDK.

This module provides the main agent interface, integrating spoon-core's
ChatBot, SpoonReactMCP, and SkillManager with spoon-bot's native OS tools.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import logging as stdlib_logging
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable

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
from spoon_bot.agent.context import ContextBuilder, format_current_datetime_context
from spoon_bot.agent.session_compact import build_session_compact_context
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
    build_tool_owner_key,
    clear_captured_tool_outputs,
    capture_tool_outputs,
    consume_captured_tool_output,
    normalize_tool_arguments,
    track_tool_invocations,
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
    AgentLoopConfig,
    DEFAULT_MAX_STREAM_TOOL_RESULTS_WITHOUT_CONTENT,
    DEFAULT_MAX_OUTPUT,
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
from spoon_bot.skills.zip_install import InstalledSkillZip, install_skill_zip_archive
from spoon_bot.services.hotreload import HotReloadService
from spoon_bot.subagent.manager import SubagentManager
from spoon_bot.subagent.tools import SubagentTool
from spoon_bot.subagent.catalog import format_roles_for_prompt
from spoon_bot.subagent.models import SubagentState
from spoon_bot.session.manager import SessionManager
from spoon_bot.session.store import create_session_store
from spoon_bot.memory.store import MemoryStore
from spoon_bot.wallet import ensure_wallet_runtime
from spoon_bot.exceptions import (
    SpoonBotError,
    APIKeyMissingError,
    ContextOverflowError,
    LLMError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMTimeoutError,
    SkillActivationError,
    user_friendly_error,
)
from spoon_bot.services.git import GitManager
from spoon_bot.utils.retry import (
    DEFAULT_RETRY_CONFIG,
    RetryConfig,
    is_context_overflow_error,
    with_provider_retry,
)

if TYPE_CHECKING:
    from spoon_bot.session.manager import Session


_ATTACHMENT_CONTEXT_HEADER = "Attached workspace files (source of truth for this request):"
_ATTACHMENT_ONLY_PLACEHOLDER = (
    "The user attached files without extra text. Inspect the files and answer based on their contents."
)
_SANDBOX_WORKSPACE_ROOT = "/workspace"
_MISSING = object()
_EXTERNAL_SIDE_EFFECT_BOUNDARY = (
    "[EXTERNAL SIDE-EFFECT BOUNDARY]: For external systems, account/wallet "
    "state, remote jobs, approvals, registrations, entries, submissions, "
    "trades, or any action that spends credits/tokens/funds or changes remote "
    "state, do at most one unit of side effect per user request unless the "
    "newest request gives an explicit count or range. After one unit, report "
    "the result or concrete blocker; do not loop into additional units just "
    "because tools remain available. Do not batch multiple alternative tool "
    "attempts in one assistant step; run one attempt, inspect its result, and "
    "then either answer with the blocker or proceed only when the newest "
    "request explicitly allows another attempt. If the user gives an exact command for "
    "a replay, simulation, dry-run, or no-op check, run that command exactly "
    "as written; never remove protective wrappers such as echo/printf or "
    "dry-run/no-op flags, and never convert a simulated command into a live "
    "side-effecting command.\n"
)
_TURN_STATE_PENDING = "pending"
_TURN_STATE_COMPLETED = "completed"
_TURN_STATE_INTERRUPTED = "interrupted"
_TURN_STATE_SUPERSEDED = "superseded"
def _workspace_root_path(workspace: Path | str | None) -> Path:
    """Resolve the workspace root used to validate persisted file references."""
    return Path(workspace or Path.home() / ".spoon-bot" / "workspace").expanduser().resolve()


def _resolve_workspace_file(path_str: str, workspace: Path | str | None) -> Path | None:
    """Resolve a file path and ensure it stays within the configured workspace."""
    candidate = str(path_str or "").strip()
    if not candidate:
        return None

    workspace_root = _workspace_root_path(workspace)
    sandbox_root = _SANDBOX_WORKSPACE_ROOT.rstrip("/")
    try:
        if candidate.startswith("/"):
            normalized = Path(candidate).as_posix()
            workspace_root_str = workspace_root.as_posix().rstrip("/")
            if normalized == sandbox_root or normalized.startswith(sandbox_root + "/"):
                relative = normalized[len(sandbox_root):].lstrip("/")
                resolved = (workspace_root / relative).resolve(strict=True)
            elif normalized == workspace_root_str or normalized.startswith(workspace_root_str + "/"):
                relative = normalized[len(workspace_root_str):].lstrip("/")
                resolved = (workspace_root / relative).resolve(strict=True)
            else:
                resolved = Path(candidate).expanduser().resolve(strict=True)
        else:
            resolved = (workspace_root / candidate).resolve(strict=True)
    except (FileNotFoundError, OSError):
        return None

    if resolved != workspace_root and workspace_root not in resolved.parents:
        return None
    if not resolved.is_file():
        return None
    return resolved


def _normalize_media_list(raw: Any) -> list[str]:
    """Normalize stored media payloads to a list of non-empty file paths."""
    if not isinstance(raw, list):
        return []

    items: list[str] = []
    for item in raw:
        if isinstance(item, str):
            value = item.strip()
        else:
            value = str(item).strip() if item is not None else ""
        if value:
            items.append(value)
    return items


def _sanitize_media_list(raw: Any, workspace: Path | str | None) -> list[str]:
    """Keep only workspace-backed media paths from persisted session metadata."""
    return [
        path
        for path in _normalize_media_list(raw)
        if _resolve_workspace_file(path, workspace) is not None
    ]


def _normalize_attachment_refs(raw: Any) -> list[dict[str, Any]]:
    """Normalize stored attachment references from session metadata."""
    if not isinstance(raw, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        uri = str(item.get("workspace_path") or item.get("uri") or "").strip()
        if not uri:
            continue
        attachment = dict(item)
        attachment["uri"] = uri
        attachment.setdefault("workspace_path", uri)
        normalized.append(attachment)
    return normalized


def _sanitize_attachment_refs(raw: Any, workspace: Path | str | None) -> list[dict[str, Any]]:
    """Keep only workspace-backed attachment references from persisted metadata."""
    sanitized: list[dict[str, Any]] = []
    for item in _normalize_attachment_refs(raw):
        uri = str(item.get("workspace_path") or item.get("uri") or "").strip()
        if _resolve_workspace_file(uri, workspace) is None:
            continue
        sanitized.append(item)
    return sanitized


def _attachment_context_entries(attachments: list[dict[str, Any]]) -> list[tuple[dict[str, Any], str]]:
    """Return normalized attachment entries with resolved display paths."""
    normalized_items: list[tuple[dict[str, Any], str]] = []
    for item in attachments:
        if not isinstance(item, dict):
            continue
        uri = str(item.get("workspace_path") or item.get("uri") or "").strip()
        if uri:
            normalized_items.append((item, uri))
    return normalized_items


def _build_attachment_context_lines(attachments: list[dict[str, Any]]) -> list[str]:
    """Build the synthetic attachment context block appended to prompts."""
    normalized_items = _attachment_context_entries(attachments)
    if not normalized_items:
        return []

    lines = [_ATTACHMENT_CONTEXT_HEADER]
    for item, uri in normalized_items:
        name = str(item.get("name") or item.get("file_name") or "").strip()
        mime_type = str(item.get("mime_type") or item.get("file_type") or "").strip()
        size = item.get("size") if "size" in item else item.get("file_size")
        suffix = []
        if name:
            suffix.append(f"name: {name}")
        if mime_type:
            suffix.append(f"mime: {mime_type}")
        if isinstance(size, int) and size > 0:
            suffix.append(f"size: {size} bytes")
        line = f"- {uri}"
        if suffix:
            line += f" ({', '.join(suffix)})"
        lines.append(line)
    lines.append("Use these attached workspace files as the primary source of truth for this request.")
    return lines


def _ensure_attachment_context(content: str, attachments: list[dict[str, Any]]) -> str:
    """Append attachment path context unless it is already present in content."""
    if not attachments:
        return content

    text = content if isinstance(content, str) else str(content)
    normalized_items = _attachment_context_entries(attachments)
    uris = [uri for _, uri in normalized_items]
    if not uris:
        return text
    if _ATTACHMENT_CONTEXT_HEADER in text and all(uri in text for uri in uris):
        return text

    lines = [text.strip()] if text.strip() else [_ATTACHMENT_ONLY_PLACEHOLDER]
    lines.extend(["", *_build_attachment_context_lines(attachments)])
    return "\n".join(lines)


def _strip_attachment_context(content: str, attachments: list[dict[str, Any]]) -> str:
    """Remove the exact synthetic attachment block so sessions keep user-authored text only."""
    if not attachments:
        return content

    text = content if isinstance(content, str) else str(content)
    if _ATTACHMENT_CONTEXT_HEADER not in text:
        return text

    context_lines = _build_attachment_context_lines(attachments)
    if not context_lines:
        return text

    delimiter = f"\n\n{context_lines[0]}\n"
    if delimiter not in text:
        return text

    prefix, suffix = text.split(delimiter, 1)
    expected_suffix = "\n".join(context_lines[1:])
    if suffix != expected_suffix:
        return text
    if prefix == _ATTACHMENT_ONLY_PLACEHOLDER:
        return ""
    return prefix


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
        "Continue working on the user's latest request. "
        "Treat the latest real user request as authoritative, even if earlier history was compacted. "
        "Prior conversation is reference only; do not continue, finish, or repair an earlier task unless the latest request explicitly asks for that. "
        "Do NOT repeat previous actions. Do NOT fabricate output. "
        "Make autonomous choices when input is needed. "
        "Do NOT summarize prior work unless the user explicitly asked for a summary. "
        "If the task can be completed now, return the final user-facing answer in the format the user requested and stop. "
        "If an authoritative request-ending block is present, it is copied verbatim from the newest user message and your final answer must satisfy it exactly."
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
        provider_max_retries: int = 5,
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
        )
        self.tool_followup_timeout = self._float_env(
            "SPOON_BOT_TOOL_FOLLOWUP_TIMEOUT",
            DEFAULT_TOOL_FOLLOWUP_TIMEOUT,
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
                logger.warning(f"Session store '{_store_backend}' init failed ({exc}), falling back to file")
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
    def _float_env(name: str, default: float) -> float:
        """Read a positive float env override for runtime stream budgets."""
        value = os.environ.get(name)
        if value is None or not value.strip():
            return default
        try:
            parsed = float(value)
        except ValueError:
            logger.warning(f"Ignoring invalid {name}={value!r}; expected a number")
            return default
        return parsed if parsed > 0 else default

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
        system_prompt += f"\n\n[Context: {self.context_window:,} tokens - be concise.]\n"

        # Skills section (Openclaw pattern: XML metadata in system prompt)
        skills_xml = self._build_skills_for_prompt()
        if skills_xml:
            system_prompt += f"\n## Installed Skills\n{skills_xml}\n"
            system_prompt += (
                "\nUse this catalog as available context, not as a hidden router. "
                "When a skill is directly relevant, read its SKILL.md and follow "
                "the skill's own procedures. Otherwise use the normal tools.\n"
            )

        system_prompt += (
            "\n## Workflow\n"
            f"You have up to {self.max_iterations} steps. Minimize steps.\n\n"
            "1. Decide the next action from the latest user request and available context.\n"
            "2. If an installed skill is directly relevant, `read_file` its SKILL.md path, "
            "then execute its procedure.\n"
            "3. Run commands from SKILL.md directly via shell. Do NOT write script files unless requested.\n"
            "4. When done, return the user-facing result in the format the latest user requested. "
            "Only summarize if the user explicitly asked for a summary.\n\n"
            "### Rules\n"
            "- Do NOT re-read files already in context.\n"
            "- Memory, recent replies, and conversation history are stale hints. "
            "For current workspace, skill, account, balance, job, or external-system state, "
            "verify with tools before answering.\n"
            "- `source .env.local` before commands that need env vars.\n"
            "- If a command fails, analyze the error and retry with fixes.\n"
            "- Follow user instructions exactly - respect specific IDs, names, actions.\n"
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
        if "spawn" in self.tools.get_active_tools():
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

        # Create MCP tools - expand each server into individual tools (#5)
        await self._init_mcp_tools()

        # Create SkillManager if enabled - BEFORE building active_tools
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

        # Keep the agent's per-step timeout aligned with the effective shell ceiling
        # so long-running commands are not cancelled prematurely by the outer loop.
        effective_ceiling = max(self.shell_timeout, getattr(self, "shell_max_timeout", DEFAULT_SHELL_MAX_TIMEOUT))
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
        self.tools.register(ShellTool(
            timeout=self.shell_timeout,
            max_timeout=getattr(self, "shell_max_timeout", DEFAULT_SHELL_MAX_TIMEOUT),
            max_output=self.max_output,
            working_dir=str(self.workspace),
            allow_chaining=True,
            allow_substitution=True,
        ))

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
            ReadFileTool(workspace=self.workspace, additional_read_paths=_extra_read, max_output=15000),
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
        self.tools.register(ActivateToolTool(
            activate_fn=self.add_tool,
            list_inactive_fn=lambda: [
                {"name": t.name, "description": t.description}
                for t in self.tools.get_inactive_tools().values()
            ],
        ))

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
        """No-op placeholder.

        History compaction is deliberately runtime-only now. Persisted session
        history remains the source of truth so search/recovery keeps working.
        """
        return 0

    def _warn_dropped_refs(self, *, kind: str, dropped: list[str]) -> None:
        """Log dropped persisted refs once per (session, ref) pair."""
        unique_refs = [ref for ref in dict.fromkeys(filter(None, dropped))]
        if not unique_refs:
            return

        cache_map = (
            self._warned_invalid_attachment_refs
            if kind == "attachment"
            else self._warned_invalid_media_refs
        )
        seen = cache_map.setdefault(self.session_key, set())

        new_refs = [ref for ref in unique_refs if ref not in seen]
        if new_refs:
            seen.update(new_refs)
            preview = ", ".join(new_refs[:5])
            suffix = f" (+{len(new_refs) - 5} more)" if len(new_refs) > 5 else ""
            logger.warning(
                f"Dropped invalid persisted {kind} refs outside workspace during "
                f"history sync (session={self.session_key}): {preview}{suffix}"
            )
        else:
            logger.debug(
                f"Dropped invalid persisted {kind} refs outside workspace during "
                f"history sync (session={self.session_key}): "
                f"{len(unique_refs)} already-known ref(s)"
            )

    @staticmethod
    def _message_token_count(message: str) -> int:
        """Approximate token count for lightweight history-scope decisions."""
        return len(re.findall(r"[0-9A-Za-z\u4e00-\u9fff]+", message or ""))

    @classmethod
    def _history_rehydrate_scope(
        cls,
        upcoming_message: str | None,
        *,
        history_messages: list[dict[str, Any]] | None = None,
    ) -> str:
        """Choose how much prior session history should enter the runtime context."""
        normalized = (upcoming_message or "").strip()
        if not normalized:
            return "full"

        # Runtime history is intentionally isolated by default. Persisted session
        # history remains searchable/observable, but the active agent loop should
        # not inherit unfinished tool chains or prior-task intent for a new user
        # turn. This mirrors a Claude Code-style active-request boundary without
        # deriving control flow from prompt text.
        return "minimal"

    @staticmethod
    def _filter_rehydratable_history(
        history_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Drop interrupted/superseded turn fragments from runtime rehydration."""
        filtered: list[dict[str, Any]] = []
        skipping_aborted_turn = False

        for message in history_messages:
            if not isinstance(message, dict):
                continue

            role = str(message.get("role") or "").strip().lower()
            if role == "user":
                turn_state = AgentLoop._turn_state_of_message(message)
                if turn_state in {_TURN_STATE_INTERRUPTED, _TURN_STATE_SUPERSEDED}:
                    skipping_aborted_turn = True
                    continue
                skipping_aborted_turn = False
                filtered.append(message)
                continue

            if skipping_aborted_turn and role in {"assistant", "tool"}:
                continue

            filtered.append(message)

        return filtered

    async def _sync_runtime_history_from_session(
        self,
        *,
        upcoming_message: str | None = None,
    ) -> int:
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
        history_messages = (
            self._session.get_messages()
            if hasattr(self._session, "get_messages")
            else self._session.get_history()
        )
        if not isinstance(history_messages, list):
            history_messages = []
        history_messages = AgentLoop._filter_rehydratable_history(history_messages)

        rehydrate_scope = AgentLoop._history_rehydrate_scope(
            upcoming_message,
            history_messages=history_messages,
        )
        if rehydrate_scope == "minimal":
            history_messages = []

        try:
            from spoon_ai.schema import Function as _CoreFunction  # noqa: F401
        except Exception:
            _CoreFunction = None  # type: ignore[assignment]

        def _rehydrate_tool_calls(raw: Any) -> list | None:
            if not raw or _CoreFunction is None:
                return None
            if not isinstance(raw, list):
                return None
            out: list = []
            for item in raw:
                if isinstance(item, CoreToolCall):
                    out.append(item)
                    continue
                if not isinstance(item, dict):
                    continue
                tc_id = item.get("id")
                if not tc_id:
                    continue
                fn = item.get("function") or {}
                if isinstance(fn, dict):
                    name = fn.get("name") or ""
                    arguments = fn.get("arguments") or ""
                else:
                    name = getattr(fn, "name", "") or ""
                    arguments = getattr(fn, "arguments", "") or ""
                if arguments is not None and not isinstance(arguments, str):
                    try:
                        arguments = json.dumps(arguments, ensure_ascii=False)
                    except Exception:
                        arguments = str(arguments)
                try:
                    out.append(
                        CoreToolCall(
                            id=str(tc_id),
                            type=str(item.get("type") or "function"),
                            function=_CoreFunction(
                                name=str(name), arguments=str(arguments or "")
                            ),
                        )
                    )
                except Exception:
                    continue
            return out or None

        for msg in history_messages:
            role = str(msg.get("role", "")).strip().lower()
            if role not in {"user", "assistant", "tool"}:
                continue
            if role == "user" and AgentLoop._turn_state_of_message(msg) in {
                _TURN_STATE_INTERRUPTED,
                _TURN_STATE_SUPERSEDED,
            }:
                continue

            assistant_tool_calls = None
            if role == "assistant":
                assistant_tool_calls = _rehydrate_tool_calls(msg.get("tool_calls"))
                if assistant_tool_calls is None:
                    continue
            content = msg.get("content", "")

            if not isinstance(content, str):
                try:
                    content = json.dumps(content, ensure_ascii=False)
                except Exception:
                    content = str(content)

            raw_media = _normalize_media_list(msg.get("media"))
            media = _sanitize_media_list(raw_media, self.workspace)
            if raw_media and len(media) != len(raw_media):
                self._warn_dropped_refs(
                    kind="media",
                    dropped=[ref for ref in raw_media if ref not in media],
                )

            raw_attachments = _normalize_attachment_refs(msg.get("attachments"))
            attachments = _sanitize_attachment_refs(raw_attachments, self.workspace)
            if raw_attachments and len(attachments) != len(raw_attachments):
                kept = {
                    str(item.get("workspace_path") or item.get("uri") or "").strip()
                    for item in attachments
                }
                dropped = [
                    str(item.get("workspace_path") or item.get("uri") or "").strip()
                    for item in raw_attachments
                    if str(item.get("workspace_path") or item.get("uri") or "").strip() not in kept
                ]
                self._warn_dropped_refs(kind="attachment", dropped=dropped)

            content = self._build_runtime_message_content(
                role,
                content,
                media=media,
                attachments=attachments,
            )

            extra_kwargs: dict[str, Any] = {}
            if role == "tool":
                tc_id = msg.get("tool_call_id")
                if tc_id:
                    extra_kwargs["tool_call_id"] = str(tc_id)
                tool_name = msg.get("name") or msg.get("tool_name")
                if tool_name:
                    extra_kwargs["tool_name"] = str(tool_name)
            elif role == "assistant":
                if assistant_tool_calls:
                    extra_kwargs["tool_calls"] = assistant_tool_calls

            try:
                await self._agent.add_message(role, content, **extra_kwargs)
                injected_count += 1
            except Exception as exc:
                logger.warning(
                    f"Failed to inject session history message "
                    f"(role={role}, index={injected_count}): {exc}"
                )

        return injected_count

    async def _prepare_request_context(self, upcoming_message: str | None = None) -> None:
        """Prepare request context by injecting persisted history into runtime memory.

        Persisted history (``self._session.messages``) is the authoritative
        store and is never trimmed here — see :meth:`_trim_context_if_needed`.
        Runtime memory is rebuilt from the persisted transcript for the next
        provider request. If that rebuilt runtime context is already close to
        the configured ``context_window``, older runtime-only history is
        compacted before the provider call starts.
        """
        self._refresh_recent_turn_notice()
        self._refresh_recent_invoked_skill_contexts()
        trimmed_count = self._trim_context_if_needed()
        persisted_repaired = self._normalize_persisted_session_tool_context()
        session = getattr(self, "_session", None)
        if session is None:
            session_history = []
        else:
            session_history = (
                session.get_messages()
                if hasattr(session, "get_messages")
                else session.get_history()
            )
        if not isinstance(session_history, list):
            session_history = []
        session_history = AgentLoop._filter_rehydratable_history(session_history)
        rehydrate_scope = AgentLoop._history_rehydrate_scope(
            upcoming_message,
            history_messages=session_history,
        )
        injected_count = await self._sync_runtime_history_from_session(
            upcoming_message=upcoming_message
        )

        # Repair tool-call ordering after history injection - session storage
        # may not preserve tool_call_id metadata, producing orphaned tool
        # messages that providers (OpenAI, Gemini, etc.) reject.
        repaired = 0
        messages_ref: list | None = None
        if self._agent and hasattr(self._agent, "memory"):
            messages_ref = getattr(self._agent.memory, "messages", None)
            if isinstance(messages_ref, list):
                repaired = self._normalize_runtime_tool_context(messages_ref)

        compressed = 0
        runtime_tokens_before = self._estimate_runtime_tokens()
        runtime_tokens = runtime_tokens_before
        trigger_budget = self._runtime_compaction_trigger_budget()
        if isinstance(messages_ref, list) and messages_ref and runtime_tokens > trigger_budget:
            try:
                compressed = self._compress_runtime_context(
                    force=True,
                    budget_tokens=trigger_budget,
                )
            except Exception as exc:
                logger.debug(f"Proactive runtime compaction skipped: {exc}")
            if runtime_tokens > trigger_budget and compressed == 0:
                compressed = self._force_compress_runtime_context()
            runtime_tokens = self._estimate_runtime_tokens()
            if compressed:
                AgentLoop._queue_runtime_notice(
                    self,
                    kind="runtime_compaction",
                    stage="preflight",
                    compressed_actions=compressed,
                    estimated_tokens_before=runtime_tokens_before,
                    estimated_tokens_after=runtime_tokens,
                    trigger_budget=trigger_budget,
                )
        session_tokens = self._estimate_token_count()

        logger.info(
            f"Session context prepared: session={self.session_key}, "
            f"injected_messages={injected_count}, "
            f"runtime_tokens~{runtime_tokens}, session_tokens~{session_tokens}, "
            f"trimmed_messages={trimmed_count}, rehydrate_scope={rehydrate_scope}"
            + (f", repaired_session_tool_order={persisted_repaired}" if persisted_repaired else "")
            + (f", repaired_tool_order={repaired}" if repaired else "")
            + (f", compressed_actions={compressed}" if compressed else "")
        )

    def _runtime_memory_snapshot_index(self) -> int:
        """Return the current length of runtime memory."""
        try:
            return len(AgentLoop._get_runtime_memory_messages(self))
        except Exception:
            return 0

    def _normalize_persisted_session_tool_context(self) -> int:
        """Repair persisted session tool traces before injecting them into runtime memory."""
        messages = getattr(getattr(self, "_session", None), "messages", None)
        if not isinstance(messages, list) or not messages:
            return 0

        try:
            before = json.dumps(messages, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            before = repr(messages)

        repaired = AgentLoop._normalize_runtime_tool_context(messages, finalized=True)

        try:
            after = json.dumps(messages, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            after = repr(messages)

        if before != after:
            try:
                self.sessions.save(self._session)
            except Exception as exc:
                logger.debug(f"Failed to save normalized session tool context: {exc}")
            return max(repaired, 1)
        return repaired

    def _normalize_runtime_memory_before_run(self, label: str) -> int:
        """Repair runtime tool-call context immediately before provider execution."""
        try:
            messages = AgentLoop._get_runtime_memory_messages(self)
            if not isinstance(messages, list) or not messages:
                return 0
            repaired = AgentLoop._normalize_runtime_tool_context(messages)
            if repaired:
                logger.info(
                    f"[{label}] Normalized {repaired} runtime tool-context "
                    "entry/entries before model run"
                )
            return repaired
        except Exception as exc:
            logger.debug(f"[{label}] Runtime tool-context normalization skipped: {exc}")
            return 0

    @staticmethod
    def _truncate_runtime_memory(self, start_index: int) -> None:
        """Remove runtime-only messages appended by an aborted turn."""
        if not isinstance(start_index, int) or start_index < 0:
            return
        try:
            messages = AgentLoop._get_runtime_memory_messages(self)
        except Exception:
            return
        if isinstance(messages, list) and start_index < len(messages):
            del messages[start_index:]

    @staticmethod
    def _drain_agent_output_queue(self) -> None:
        """Discard queued stream chunks from an interrupted run."""
        agent = getattr(self, "_agent", None)
        output_queue = getattr(agent, "output_queue", None)
        if output_queue is None or not hasattr(output_queue, "empty"):
            return
        while True:
            try:
                if output_queue.empty():
                    break
                if hasattr(output_queue, "get_nowait"):
                    output_queue.get_nowait()
                else:
                    break
            except Exception:
                break

    def _capture_turn_tool_trace(self, start_index: int) -> list[dict[str, Any]]:
        """Capture tool-call artifacts added since ``start_index``."""
        if not isinstance(start_index, int) or start_index < 0:
            return []

        messages = AgentLoop._get_runtime_memory_messages(self)
        if not messages or start_index >= len(messages):
            return []

        captured: list[dict[str, Any]] = []
        for msg in messages[start_index:]:
            role = AgentLoop._stream_message_role(msg).lower()
            if role not in ("tool", "assistant"):
                continue

            content = AgentLoop._stream_message_attr(msg, "text_content", None)
            if not isinstance(content, str) or not content:
                content = AgentLoop._stream_message_attr(msg, "content", "") or ""
            if not isinstance(content, str):
                try:
                    content = json.dumps(content, ensure_ascii=False)
                except Exception:
                    content = str(content)

            extras: dict[str, Any] = {}

            if role == "tool":
                tool_call_id = AgentLoop._stream_message_attr(msg, "tool_call_id", None) \
                    or AgentLoop._stream_message_attr(msg, "id", None)
                if tool_call_id:
                    extras["tool_call_id"] = str(tool_call_id)
                name = AgentLoop._stream_message_attr(msg, "name", None)
                if name:
                    extras["name"] = str(name)
            else:
                tool_calls = AgentLoop._stream_message_attr(msg, "tool_calls", None) or []
                serialized: list[dict[str, Any]] = []
                for tc in tool_calls:
                    tc_id = getattr(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else None)
                    tc_type = (
                        getattr(tc, "type", None)
                        or (tc.get("type") if isinstance(tc, dict) else None)
                        or "function"
                    )
                    fn = getattr(tc, "function", None) or (
                        tc.get("function") if isinstance(tc, dict) else None
                    )
                    if fn is not None:
                        tc_name = getattr(fn, "name", None) or (
                            fn.get("name") if isinstance(fn, dict) else None
                        )
                        tc_args = getattr(fn, "arguments", None) or (
                            fn.get("arguments") if isinstance(fn, dict) else None
                        )
                    else:
                        tc_name = getattr(tc, "name", None)
                        tc_args = getattr(tc, "arguments", None)

                    if tc_args is not None and not isinstance(tc_args, str):
                        try:
                            tc_args = json.dumps(tc_args, ensure_ascii=False)
                        except Exception:
                            tc_args = str(tc_args)

                    if tc_name or tc_id:
                        serialized.append(
                            {
                                "id": tc_id,
                                "type": tc_type,
                                "function": {
                                    "name": tc_name,
                                    "arguments": tc_args or "",
                                },
                            }
                        )
                if not serialized:
                    continue
                extras["tool_calls"] = serialized

            captured.append({"role": role, "content": content, "extras": extras})
        return captured

    def _persist_turn_tool_trace(self, start_index: int) -> int:
        """Append captured tool-trace messages to ``self._session``."""
        if not self._should_persist_tool_trace():
            return 0

        try:
            messages = AgentLoop._get_runtime_memory_messages(self)
            if isinstance(messages, list) and messages:
                AgentLoop._normalize_runtime_tool_context(messages)
            trace = self._capture_turn_tool_trace(start_index)
        except Exception as exc:
            logger.debug(f"Tool-trace capture failed: {exc}")
            return 0

        if not trace:
            return 0

        for entry in trace:
            try:
                self._session.add_message(
                    entry["role"], entry["content"], **entry.get("extras", {})
                )
            except Exception as exc:
                logger.debug(f"Tool-trace persist skipped a message: {exc}")
        return len(trace)

    def _persist_tool_trace_entries(self, trace: list[dict[str, Any]]) -> int:
        """Persist already-captured tool trace entries after generic pairing repair."""
        if not self._should_persist_tool_trace() or not trace:
            return 0

        messages: list[dict[str, Any]] = []
        for entry in trace:
            if not isinstance(entry, dict):
                continue
            role = str(entry.get("role") or "").strip().lower()
            if role not in {"assistant", "tool"}:
                continue
            content = entry.get("content", "")
            if not isinstance(content, str):
                try:
                    content = json.dumps(content, ensure_ascii=False)
                except Exception:
                    content = str(content)
            extras = entry.get("extras") if isinstance(entry.get("extras"), dict) else {}
            messages.append({"role": role, "content": content, **extras})

        if not messages:
            return 0

        AgentLoop._normalize_runtime_tool_context(messages, finalized=True)

        persisted = 0
        for msg in messages:
            role = str(msg.get("role") or "").strip().lower()
            if role not in {"assistant", "tool"}:
                continue
            if role == "assistant" and not msg.get("tool_calls"):
                continue
            if role == "tool" and not msg.get("tool_call_id"):
                continue
            content = msg.get("content", "")
            extras = {
                key: value
                for key, value in msg.items()
                if key not in {"role", "content", "timestamp"}
            }
            try:
                self._session.add_message(role, content, **extras)
                persisted += 1
            except Exception as exc:
                logger.debug(f"Tool-trace persist skipped a streamed message: {exc}")
        return persisted

    @staticmethod
    def _tool_call_name_and_arguments(tool_call: Any) -> tuple[str, Any]:
        """Extract a function/tool name and raw arguments from common tool-call shapes."""
        fn = getattr(tool_call, "function", None) or (
            tool_call.get("function") if isinstance(tool_call, dict) else None
        )
        if fn is not None:
            name = getattr(fn, "name", None) or (
                fn.get("name") if isinstance(fn, dict) else None
            )
            arguments = getattr(fn, "arguments", None) or (
                fn.get("arguments") if isinstance(fn, dict) else None
            )
            return str(name or ""), arguments

        name = getattr(tool_call, "name", None) or (
            tool_call.get("name") if isinstance(tool_call, dict) else None
        )
        arguments = getattr(tool_call, "arguments", None) or (
            tool_call.get("arguments") if isinstance(tool_call, dict) else None
        )
        return str(name or ""), arguments

    @staticmethod
    def _parse_tool_arguments(arguments: Any) -> dict[str, Any]:
        """Parse structured tool-call arguments without inspecting user prompt text."""
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str) and arguments.strip():
            try:
                parsed = json.loads(arguments)
            except Exception:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    @staticmethod
    def _skill_name_from_workspace_path(path_value: Any) -> str | None:
        """Return the skill name for structured workspace skill paths."""
        if not isinstance(path_value, str) or not path_value.strip():
            return None
        normalized = path_value.strip().replace("\\", "/")
        parts = [part for part in normalized.split("/") if part and part != "."]
        for index, part in enumerate(parts[:-1]):
            if part == "skills" and index + 1 < len(parts):
                name = parts[index + 1].strip()
                return name or None
        return None

    @staticmethod
    def _extract_skill_names_from_tool_call(tool_name: str, arguments: Any) -> list[str]:
        """Extract skill usage from explicit structured tool calls only."""
        name = str(tool_name or "").strip()
        parsed = AgentLoop._parse_tool_arguments(arguments)
        if not parsed:
            return []

        names: list[str] = []
        for key in ("path", "file_path"):
            skill_name = AgentLoop._skill_name_from_workspace_path(parsed.get(key))
            if skill_name and skill_name not in names:
                names.append(skill_name)

        if name == "skill_marketplace":
            for key in ("skill_name", "name"):
                value = str(parsed.get(key) or "").strip()
                if value and value not in names:
                    names.append(value)
        return names

    def _discover_invoked_skill_contexts_from_runtime(
        self,
        start_index: int,
    ) -> list[dict[str, Any]]:
        """Infer skill usage from actual tool calls/results in the current turn."""
        if not isinstance(start_index, int) or start_index < 0:
            return []

        messages = AgentLoop._get_runtime_memory_messages(self)
        if not messages or start_index >= len(messages):
            return []

        discovered_by_name: dict[str, tuple[int, dict[str, Any]]] = {}
        order = 0

        def _add_name(name: str) -> None:
            nonlocal order
            skill_name = str(name or "").strip()
            if not skill_name:
                return
            context = self._resolve_skill_context_by_name(skill_name)
            if context is None:
                return
            order += 1
            context = {**context, "source": "tool_usage"}
            discovered_by_name[skill_name] = (order, context)

        for msg in messages[start_index:]:
            role = AgentLoop._stream_message_role(msg).lower()
            if role == "assistant":
                tool_calls = AgentLoop._stream_message_attr(msg, "tool_calls", None) or []
                for tool_call in tool_calls:
                    tool_name, arguments = AgentLoop._tool_call_name_and_arguments(tool_call)
                    for skill_name in AgentLoop._extract_skill_names_from_tool_call(
                        tool_name,
                        arguments,
                    ):
                        _add_name(skill_name)
                continue

        return [
            context
            for _order, context in sorted(
                discovered_by_name.values(),
                key=lambda item: item[0],
                reverse=True,
            )
        ]

    def _merge_latest_user_invoked_skills(self, skill_contexts: list[dict[str, Any]]) -> int:
        """Merge discovered skill metadata into the latest persisted user turn."""
        if not skill_contexts:
            return 0

        session = getattr(self, "_session", None)
        messages = getattr(session, "messages", None)
        if not isinstance(messages, list):
            return 0

        target: dict[str, Any] | None = None
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").strip().lower()
            if role == "user":
                state = AgentLoop._turn_state_of_message(message)
                if state not in {_TURN_STATE_INTERRUPTED, _TURN_STATE_SUPERSEDED}:
                    target = message
                break

        if target is None:
            return 0

        existing = AgentLoop._iter_message_invoked_skill_refs(target)
        merged: list[dict[str, Any]] = []
        seen_names: set[str] = set()

        for skill in [*existing, *skill_contexts]:
            skill_meta = AgentLoop._session_skill_metadata(skill)
            if not skill_meta:
                continue
            name = skill_meta["name"]
            if name in seen_names:
                continue
            seen_names.add(name)
            merged.append(skill_meta)

        if not merged:
            return 0

        if target.get("invoked_skills") == merged:
            return 0

        target["invoked_skills"] = merged
        return len(merged)

    def _merge_turn_invoked_skills_from_runtime(self, start_index: int) -> int:
        """Persist current-turn skill usage discovered from actual tool execution."""
        try:
            contexts = self._discover_invoked_skill_contexts_from_runtime(start_index)
            return self._merge_latest_user_invoked_skills(contexts)
        except Exception as exc:
            logger.debug(f"Runtime skill usage merge skipped: {exc}")
            return 0

    @staticmethod
    def _should_persist_tool_trace() -> bool:
        """Whether to persist tool traces alongside user/assistant turns."""
        import os as _os

        raw = _os.getenv("SPOON_BOT_PERSIST_TOOL_TRACE")
        if raw is None:
            return True
        return raw.strip().lower() not in {"0", "false", "no", "off"}

    @staticmethod
    def _attachment_looks_like_skill_zip(item: dict[str, Any]) -> bool:
        name = str(item.get("name") or item.get("file_name") or "").strip().lower()
        mime_type = str(item.get("mime_type") or item.get("file_type") or "").strip().lower()
        uri = str(item.get("workspace_path") or item.get("uri") or "").strip().lower()
        return (
            name.endswith(".zip")
            or uri.endswith(".zip")
            or mime_type in {"application/zip", "application/x-zip-compressed"}
        )

    async def _install_skill_zip_attachments(
        self,
        attachments: list[dict[str, Any]] | None,
    ) -> list[InstalledSkillZip]:
        """Install attached skill zip archives before the LLM plans commands."""
        self._current_turn_skill_zip_installs = []
        self._current_turn_skill_zip_failures = []
        if not attachments:
            return []

        installed: list[InstalledSkillZip] = []
        for item in attachments:
            if not isinstance(item, dict) or not self._attachment_looks_like_skill_zip(item):
                continue
            uri = str(item.get("workspace_path") or item.get("uri") or "").strip()
            path = _resolve_workspace_file(uri, self.workspace)
            if path is None:
                failure = f"{uri or '<missing>'}: attachment is not available in workspace"
                self._current_turn_skill_zip_failures.append(failure)
                logger.warning(f"Skill zip attachment skipped: {failure}")
                continue
            try:
                result = install_skill_zip_archive(
                    path,
                    self.workspace,
                    name_hint=str(item.get("name") or path.name),
                    reinstall=True,
                )
            except Exception as exc:
                failure = f"{path.name}: {exc}"
                self._current_turn_skill_zip_failures.append(failure)
                logger.warning(f"Skill zip attachment install failed: {failure}")
                continue
            if result is None:
                continue
            installed.append(result)
            try:
                self.record_touched_paths(result.skill_md)
            except Exception:
                pass
            AgentLoop._queue_runtime_notice(
                self,
                kind="skill_zip_install",
                stage="prepare",
                name=result.name,
                path=self._workspace_relative_display_path(result.skill_md),
                reinstalled=result.reinstalled,
                file_count=result.file_count,
            )

        if installed:
            try:
                await self.reload_skills()
            except Exception as exc:
                failure = f"reload_skills failed after zip install: {exc}"
                self._current_turn_skill_zip_failures.append(failure)
                logger.warning(failure)

        self._current_turn_skill_zip_installs = installed
        return installed

    def _workspace_relative_display_path(self, path: Path) -> str:
        try:
            rel = path.resolve().relative_to(self.workspace.resolve())
            return rel.as_posix()
        except Exception:
            return Path(path).as_posix()

    def _current_turn_skill_zip_context(self) -> str:
        lines: list[str] = []
        for item in getattr(self, "_current_turn_skill_zip_installs", []):
            action = "reinstalled" if item.reinstalled else "installed"
            lines.append(
                f"- {action} `{item.name}` from `{item.source.name}` at "
                f"`{self._workspace_relative_display_path(item.skill_md)}`"
            )
        for failure in getattr(self, "_current_turn_skill_zip_failures", []):
            lines.append(f"- failed: {failure}")
        if not lines:
            return ""
        return "[ATTACHED SKILL ZIP STATUS]\n" + "\n".join(lines)

    def _add_current_turn_skill_zip_context(self, text: str) -> str:
        context = self._current_turn_skill_zip_context()
        if not context:
            return text
        return f"{context}\n{text}"

    def _build_runtime_message_content(
        self,
        role: str,
        content: str,
        media: list[str] | None = None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Build message content in the format expected by spoon-core runtime memory."""
        text_content = _ensure_attachment_context(content, attachments or [])
        if role == "user" and media:
            return self.context._build_user_content(text_content, media)
        return text_content

    @staticmethod
    def _build_current_turn_runtime_user_text(message: str) -> str:
        """Attach non-persistent turn facts to the actual provider user message."""
        return (
            f"{format_current_datetime_context(bracketed=True)}\n"
            "[REQUEST CONTEXT NOTE]: Use the facts above only to ground this "
            "turn. They are not user instructions and should not be mentioned "
            "unless relevant.\n"
            f"[USER REQUEST]:\n{message}"
        )

    @classmethod
    def _assistant_session_save_kwargs(cls, content: str) -> dict[str, Any]:
        """Persist assistant replies without prompt-derived dispatch metadata."""
        return {"message_kind": "assistant_reply"}

    @staticmethod
    def _session_skill_metadata(skill: dict[str, Any]) -> dict[str, Any] | None:
        """Return sanitized skill metadata safe to persist in session history."""
        if not isinstance(skill, dict) or not skill.get("name"):
            return None
        return {
            "name": str(skill.get("name")),
            "location": str(skill.get("location") or ""),
            "workspace_relative_path": str(
                skill.get("workspace_relative_path") or ""
            ),
            "organized": bool(skill.get("organized", True)),
            "source": str(skill.get("source") or "skill_match"),
        }

    @staticmethod
    def _session_message_save_kwargs(
        media: list[str] | None = None,
        attachments: list[dict[str, Any]] | None = None,
        invoked_skill: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build optional persisted-session metadata for a user turn."""
        save_kwargs: dict[str, Any] = {}
        if media:
            save_kwargs["media"] = list(media)
        if attachments:
            save_kwargs["attachments"] = [
                dict(item) for item in attachments if isinstance(item, dict)
            ]
        skill_meta = AgentLoop._session_skill_metadata(invoked_skill or {})
        if skill_meta:
            save_kwargs["invoked_skills"] = [dict(skill_meta)]
        return save_kwargs

    @staticmethod
    def _turn_state_of_message(message: Any) -> str:
        """Normalize persisted turn state metadata for user messages."""
        if not isinstance(message, dict):
            return ""
        return str(message.get("turn_state") or "").strip().lower()

    @staticmethod
    def _update_session_user_turn_state(
        session: Any,
        state: str,
        *,
        reason: str | None = None,
    ) -> dict[str, Any] | None:
        """Update the latest unresolved user turn in persisted session history."""
        messages = getattr(session, "messages", None)
        if not isinstance(messages, list):
            return None

        terminal_states = {_TURN_STATE_COMPLETED, _TURN_STATE_INTERRUPTED, _TURN_STATE_SUPERSEDED}
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if str(message.get("role") or "").strip().lower() != "user":
                continue

            current_state = AgentLoop._turn_state_of_message(message)
            if current_state in terminal_states:
                continue

            message["turn_state"] = state
            message["turn_state_updated_at"] = datetime.now().isoformat()
            if reason:
                message["turn_state_reason"] = reason
            elif "turn_state_reason" in message:
                message.pop("turn_state_reason", None)
            return message
        return None

    @staticmethod
    def _persist_session_if_possible(self) -> None:
        """Persist the current session when the manager is available."""
        try:
            self.sessions.save(self._session)
        except Exception as exc:
            logger.warning(f"Failed to save session: {exc}")

    @staticmethod
    def _mark_latest_user_turn_state(
        self,
        state: str,
        *,
        reason: str | None = None,
    ) -> bool:
        """Mark the latest unresolved user turn as completed/interrupted/superseded."""
        message = AgentLoop._update_session_user_turn_state(
            getattr(self, "_session", None),
            state,
            reason=reason,
        )
        if message is None:
            return False
        AgentLoop._persist_session_if_possible(self)
        return True

    def _refresh_recent_turn_notice(self) -> None:
        """Expose immediate prior interruption/supersede state to the next prompt."""
        notice: str | None = None
        session = getattr(self, "_session", None)
        history_messages = (
            session.get_messages()
            if session is not None and hasattr(session, "get_messages")
            else getattr(session, "messages", [])
        )
        if isinstance(history_messages, list):
            for message in reversed(history_messages):
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role") or "").strip().lower()
                if role != "user":
                    break

                state = AgentLoop._turn_state_of_message(message)
                if state == _TURN_STATE_INTERRUPTED:
                    interrupted_request = self._truncate_request_for_prompt(
                        str(message.get("content") or "").strip()
                    )
                    notice = (
                        "The immediately previous user request was interrupted before completion.\n"
                        f"[INTERRUPTED PREVIOUS REQUEST]: {interrupted_request}\n"
                        "Resolve it against the newest user message as follows:\n"
                        "- If the newest user message is itself a standalone actionable request, "
                        "treat it as replacing the interrupted request.\n"
                        "- If the newest user message only adds constraints or details to the "
                        "interrupted request, continue the interrupted request with the new "
                        "constraints applied.\n"
                        "- Do not execute both as separate tasks unless the newest user message "
                        "explicitly asks for both."
                    )
                elif state == _TURN_STATE_SUPERSEDED:
                    notice = (
                        "A previous unfinished user request was superseded by a newer request "
                        "and is no longer pending. Execute only the newest user request unless "
                        "it explicitly asks to resume the earlier one."
                    )
                break

        self._recent_turn_notice = notice

    def _refresh_recent_invoked_skill_contexts(self) -> None:
        """Expose recent completed skill-backed turns as bounded continuity context."""
        self._recent_invoked_skill_contexts = self._find_recent_invoked_skill_contexts()

    def _recent_completed_assistant_contexts(
        self,
        *,
        max_turns: int = 6,
    ) -> list[dict[str, Any]]:
        """Return bounded prior assistant results without replaying old user tasks."""
        session = getattr(self, "_session", None)
        history_messages = (
            session.get_messages()
            if session is not None and hasattr(session, "get_messages")
            else getattr(session, "messages", [])
        )
        if not isinstance(history_messages, list):
            return []

        turns: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None

        for message in history_messages:
            if not isinstance(message, dict):
                continue

            role = str(message.get("role") or "").strip().lower()
            if role == "user":
                state = AgentLoop._turn_state_of_message(message)
                if state in {
                    "",
                    _TURN_STATE_PENDING,
                    _TURN_STATE_INTERRUPTED,
                    _TURN_STATE_SUPERSEDED,
                }:
                    current = None
                    continue

                skills = [
                    meta
                    for meta in (
                        AgentLoop._session_skill_metadata(item)
                        for item in AgentLoop._iter_message_invoked_skill_refs(message)
                    )
                    if meta
                ]
                current = {"assistant": "", "skills": skills}
                turns.append(current)
                continue

            if current is None or role != "assistant":
                continue
            message_kind = str(message.get("message_kind") or "")
            if message_kind and message_kind != "assistant_reply":
                continue
            if message.get("tool_calls"):
                continue
            if current.get("assistant"):
                continue

            content = message.get("content", "")
            if not isinstance(content, str):
                try:
                    content = json.dumps(content, ensure_ascii=False)
                except Exception:
                    content = str(content)
            if content.strip():
                current["assistant"] = content

        useful_turns = [
            turn
            for turn in turns
            if str(turn.get("assistant") or "").strip() or turn.get("skills")
        ]
        return list(reversed(useful_turns[-max_turns:]))

    @staticmethod
    def _iter_message_invoked_skill_refs(message: dict[str, Any]) -> list[dict[str, Any]]:
        """Return invoked skill references persisted on a session message."""
        invoked = message.get("invoked_skills")
        if isinstance(invoked, list):
            return [item for item in invoked if isinstance(item, dict)]
        return []

    def _find_recent_invoked_skill_contexts(
        self,
        *,
        max_user_turns: int = 12,
        max_skills: int = 4,
    ) -> list[dict[str, Any]]:
        """Return recent skill metadata without rehydrating old task history.

        The session transcript can be long and noisy.  For follow-up requests we
        keep a small newest-first set of skill identities, then resolve each
        against the current skill catalog so imported or stale session data
        cannot inject an arbitrary path/prompt.
        """
        session = getattr(self, "_session", None)
        history_messages = (
            session.get_messages()
            if session is not None and hasattr(session, "get_messages")
            else getattr(session, "messages", [])
        )
        if not isinstance(history_messages, list):
            return []

        contexts: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        user_turns_seen = 0
        for message in reversed(history_messages):
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").strip().lower()
            if role != "user":
                continue

            user_turns_seen += 1
            if user_turns_seen > max_user_turns:
                break

            state = AgentLoop._turn_state_of_message(message)
            if state in {_TURN_STATE_INTERRUPTED, _TURN_STATE_SUPERSEDED}:
                continue

            for skill in AgentLoop._iter_message_invoked_skill_refs(message):
                skill_name = str(skill.get("name") or "").strip()
                if not skill_name or skill_name in seen_names:
                    continue

                context = self._resolve_skill_context_by_name(skill_name)
                if context is None:
                    continue

                timestamp = str(message.get("timestamp") or "").strip()
                if timestamp:
                    context = {**context, "last_used_at": timestamp}
                contexts.append(context)
                seen_names.add(skill_name)
                if len(contexts) >= max_skills:
                    return contexts

        return contexts

    def _persist_user_turn_to_session(
        self,
        message: str,
        media: list[str] | None = None,
        attachments: list[dict[str, Any]] | None = None,
        invoked_skill: dict[str, Any] | None = None,
    ) -> None:
        """Persist the current user turn before the model run starts."""
        try:
            self._session.add_message(
                "user",
                _strip_attachment_context(message, attachments or []),
                turn_id=uuid.uuid4().hex,
                turn_state=_TURN_STATE_PENDING,
                turn_state_updated_at=datetime.now().isoformat(),
                **AgentLoop._session_message_save_kwargs(
                    media,
                    attachments,
                    invoked_skill=invoked_skill,
                ),
            )
            AgentLoop._persist_session_if_possible(self)
        except Exception as exc:
            logger.warning(f"Failed to persist user turn: {exc}")

    def set_subagent_context(
        self,
        *,
        session_key: str | None = None,
        channel: str | None = None,
        metadata: dict[str, Any] | None = None,
        reply_to: str | None = None,
    ) -> None:
        """Bind the spawn tool to the current requester session/channel."""
        resolved_session_key = session_key or getattr(self, "session_key", None)
        manager = getattr(self, "_subagent_manager", None)
        if manager is not None:
            manager.set_spawner_context(
                session_key=resolved_session_key,
                channel=channel,
                metadata=metadata,
                reply_to=reply_to,
            )

        registry = getattr(self, "tools", None)
        if registry is None:
            return
        spawn_tool = registry.get("spawn")
        if spawn_tool and isinstance(spawn_tool, SubagentTool):
            spawn_tool.set_spawner_context(
                session_key=resolved_session_key,
                channel=channel,
                metadata=metadata,
                reply_to=reply_to,
            )

    def _persist_turn(
        self,
        user_message: str,
        assistant_message: str,
        *,
        media: list[str] | None = None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> None:
        """Save a completed user/assistant turn to session storage."""
        try:
            save_kwargs: dict[str, Any] = {}
            if media:
                save_kwargs["media"] = list(media)
            if attachments:
                save_kwargs["attachments"] = [dict(item) for item in attachments if isinstance(item, dict)]
            self._session.add_message(
                "user",
                _strip_attachment_context(user_message, attachments or []),
                **save_kwargs,
            )
            self._session.add_message("assistant", assistant_message)
            self.sessions.save(self._session)
        except Exception as e:
            logger.warning(f"Failed to save session: {e}")

    def _persist_incomplete_turn_marker(
        self,
        *,
        label: str,
        reason: BaseException | str,
    ) -> None:
        """Close a persisted user turn when execution fails before an answer."""
        try:
            if not getattr(self, "_session", None):
                return
            reason_text = (
                type(reason).__name__
                if isinstance(reason, BaseException)
                else str(reason or "Incomplete")
            )
            messages = getattr(self._session, "messages", None)
            if isinstance(messages, list) and messages:
                last = messages[-1]
                if (
                    isinstance(last, dict)
                    and last.get("role") == "assistant"
                    and last.get("incomplete") is True
                ):
                    return
            marker = (
                "[Previous request did not complete: "
                f"{reason_text}. Treat it as historical context only; do not "
                "continue or answer it unless a later user request explicitly "
                "asks to resume or refer to it.]"
            )
            self._session.add_message(
                "assistant",
                marker,
                incomplete=True,
                incomplete_reason=reason_text,
                incomplete_source=label,
            )
            self.sessions.save(self._session)
        except Exception as exc:
            logger.debug(f"Failed to persist incomplete turn marker: {exc}")

    @staticmethod
    def _turn_failure_state_reason(label: str, reason: BaseException | str) -> str:
        """Return a bounded turn_state_reason for failed provider/runtime turns."""
        safe_label = re.sub(r"[^0-9A-Za-z_-]+", "_", str(label or "runtime")).strip("_")
        if not safe_label:
            safe_label = "runtime"
        reason_name = (
            type(reason).__name__
            if isinstance(reason, BaseException)
            else "error"
        )
        return f"{safe_label}_error:{reason_name}"

    def _persist_failed_turn_context(
        self,
        *,
        label: str,
        reason: BaseException | str,
        start_index: int | None = None,
    ) -> None:
        """Close and persist an errored turn so the next request keeps context.

        User turns are saved before model execution starts. If the model/provider
        fails before a final assistant answer, leaving the turn as ``pending``
        makes the next request look contextless and drops any tool evidence that
        only existed in runtime memory. Persist the runtime tool trace first,
        then close the user turn as interrupted and add a small marker.
        """
        if not getattr(self, "_session", None):
            return

        try:
            if isinstance(start_index, int) and start_index >= 0:
                try:
                    self._merge_turn_invoked_skills_from_runtime(start_index)
                except Exception as exc:
                    logger.debug(f"Failed-turn skill merge skipped: {exc}")
                try:
                    persisted = self._persist_turn_tool_trace(start_index)
                    if persisted:
                        AgentLoop._persist_session_if_possible(self)
                except Exception as exc:
                    logger.debug(f"Failed-turn tool trace persist skipped: {exc}")

            AgentLoop._mark_latest_user_turn_state(
                self,
                _TURN_STATE_INTERRUPTED,
                reason=AgentLoop._turn_failure_state_reason(label, reason),
            )
            AgentLoop._persist_incomplete_turn_marker(
                self,
                label=label,
                reason=reason,
            )
        except Exception as exc:
            logger.debug(f"Failed-turn persistence skipped: {exc}")

    def _current_tool_owner_key(self, session_key: str | None = None) -> str:
        """Resolve a user-scoped ownership key for background tool jobs."""
        return build_tool_owner_key(
            getattr(self, "user_id", None),
            session_key if session_key is not None else getattr(self, "session_key", None),
        )

    async def _run_agent_with_retry(
        self,
        label: str = "agent",
        pre_retry_cleanup: Callable[[], Any] | None = None,
        **run_kwargs: Any,
    ) -> Any:
        """Run ``self._agent.run()`` wrapped with provider-level retry.

        Centralises the retry pattern used by both ``process()`` and
        ``process_with_thinking()`` so the logic isn't duplicated.

        Args:
            label: Descriptive label used in log messages (e.g. "stream").
            pre_retry_cleanup: Optional sync callable invoked before each retry
                (e.g. to drain the output queue for streaming).
            **run_kwargs: Forwarded to ``self._agent.run()``.
        """
        retry_config = getattr(self, "_retry_config", None)
        if not isinstance(retry_config, RetryConfig):
            retry_config = DEFAULT_RETRY_CONFIG

        async def _do_run() -> Any:
            with bind_tool_owner(self._current_tool_owner_key()):
                return await self._agent.run(**run_kwargs)

        def _on_retry(attempt: int, exc: Exception, delay: float) -> None:
            logger.warning(
                f"[{label}] Provider transient error (attempt {attempt + 1}/"
                f"{retry_config.max_retries + 1}), "
                f"retrying in {delay:.1f}s: {type(exc).__name__}: {exc}"
            )
            if pre_retry_cleanup:
                try:
                    pre_retry_cleanup()
                except Exception:
                    pass

        return await with_provider_retry(
            _do_run,
            config=retry_config,
            on_retry=_on_retry,
        )

    @staticmethod
    def _resolve_retry_runner(self) -> Callable[..., Awaitable[Any]]:
        """Pick the configured retry runner, falling back for MagicMock-based tests."""
        retry_runner = getattr(self, "_run_agent_with_retry", None)
        self_type_module = getattr(type(self), "__module__", "")
        if not self_type_module.startswith("unittest.mock") and isinstance(self, AgentLoop):
            return retry_runner

        if callable(retry_runner):
            runner_module = getattr(type(retry_runner), "__module__", "")
            if not runner_module.startswith("unittest.mock"):
                return retry_runner
            if getattr(retry_runner, "side_effect", None) is not None:
                return retry_runner
            if getattr(retry_runner, "_mock_wraps", None) is not None:
                return retry_runner

        async def _fallback(**kwargs: Any) -> Any:
            return await AgentLoop._run_agent_with_retry(self, **kwargs)

        return _fallback

    def _reset_agent_state_for_retry(self) -> None:
        """Reset transient runtime state before retrying the same turn."""
        if hasattr(self._agent, "state") and self._agent.state != AgentState.IDLE:
            self._agent.state = AgentState.IDLE
            self._agent.current_step = 0
        if hasattr(self._agent, "_shutdown_event") and self._agent._shutdown_event.is_set():
            self._agent._shutdown_event.clear()

    def _prepare_agent_for_new_turn(self) -> None:
        """Clear transient execution state so the newest prompt owns the next run."""
        if hasattr(self._agent, "state") and self._agent.state != AgentState.IDLE:
            logger.warning(
                f"Agent {getattr(self._agent, 'name', 'runtime')} was in "
                f"{self._agent.state} state before a new turn; resetting to IDLE"
            )
            self._agent.state = AgentState.IDLE

        if hasattr(self._agent, "current_step"):
            try:
                self._agent.current_step = 0
            except Exception:
                pass

        if hasattr(self._agent, "_shutdown_event") and self._agent._shutdown_event.is_set():
            logger.info("Clearing previous shutdown signal before processing a new turn")
            self._agent._shutdown_event.clear()

        if hasattr(self._agent, "tool_calls"):
            try:
                tool_calls = getattr(self._agent, "tool_calls")
                if hasattr(tool_calls, "clear"):
                    tool_calls.clear()
                elif tool_calls:
                    self._agent.tool_calls = []
            except Exception:
                pass

    def _compress_runtime_context_for_overflow_retry(self) -> int:
        """Compress runtime context only after an explicit overflow signal."""
        compressed = 0
        try:
            compressed = self._compress_runtime_context(
                force=True,
                budget_tokens=self._runtime_compaction_trigger_budget(),
            )
        except Exception as exc:
            logger.debug(f"Soft runtime compaction failed during overflow recovery: {exc}")
        if compressed == 0:
            compressed = self._force_compress_runtime_context()
        return compressed

    async def _run_agent_with_context_overflow_recovery(
        self,
        *,
        label: str,
        retry_runner: Callable[..., Awaitable[Any]],
        pre_overflow_retry_cleanup: Callable[[], Any] | None = None,
        **run_kwargs: Any,
    ) -> Any:
        """Retry once after deliberate runtime compaction on context overflow."""
        try:
            result = await retry_runner(label=label, **run_kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result
        except Exception as exc:
            if not is_context_overflow_error(exc):
                raise

            logger.warning(
                f"[{label}] Context overflow detected: {type(exc).__name__}: {exc}. "
                "Compacting older runtime history and retrying once."
            )
            if pre_overflow_retry_cleanup:
                try:
                    pre_overflow_retry_cleanup()
                except Exception:
                    pass

            compressed = self._compress_runtime_context_for_overflow_retry()
            if compressed <= 0:
                raise

            AgentLoop._queue_runtime_notice(
                self,
                kind="runtime_compaction",
                stage="overflow_retry",
                compressed_actions=compressed,
                estimated_tokens_before=getattr(exc, "estimated_tokens", None),
                trigger_budget=getattr(exc, "max_tokens", None),
                error_type=type(exc).__name__,
            )

            self._reset_agent_state_for_retry()
            result = await retry_runner(label=f"{label}:overflow_retry", **run_kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result

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
            AgentLoop._build_current_turn_runtime_user_text(authoritative_message)
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
        _pre_turn_memory_index = self._runtime_memory_snapshot_index()
        original_system_prompt: str | None = None
        original_base_system_prompt: object = _MISSING

        _base_prompt = self._build_step_prompt(authoritative_message)
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

        retry_runner = AgentLoop._resolve_retry_runner(self)

        try:
            run_kwargs: dict[str, Any] = {}
            if (
                effective_reasoning_effort
                and self._callable_accepts_kwarg(self._agent.run, "reasoning_effort")
            ):
                run_kwargs["reasoning_effort"] = effective_reasoning_effort
            request_execution_hints = self._build_request_execution_hints(authoritative_message)
            with bind_request_execution_hints(request_execution_hints), track_tool_invocations():
                result = await AgentLoop._run_agent_with_context_overflow_recovery(
                    self,
                    label="process",
                    retry_runner=retry_runner,
                    **run_kwargs,
                )

            logger.debug(f"Agent result type: {type(result)}")
            if hasattr(result, 'content'):
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
                with bind_request_execution_hints(request_execution_hints), track_tool_invocations():
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

            final_content = AgentLoop._finalize_response_content(
                self,
                authoritative_message,
                final_content,
                turn_memory_start_index=_pre_turn_memory_index,
            )
        except asyncio.CancelledError:
            AgentLoop._mark_latest_user_turn_state(
                self,
                _TURN_STATE_INTERRUPTED,
                reason="task_cancelled",
            )
            raise
        except Exception as e:
            import traceback
            logger.error(
                f"Agent run error: {type(e).__name__}: {e}\n{traceback.format_exc()}"
            )
            AgentLoop._persist_failed_turn_context(
                self,
                label="process",
                reason=e,
                start_index=_pre_turn_memory_index,
            )
            raise
        finally:
            # Always ensure agent is back in IDLE state after processing
            self._restore_agent_think()
            AgentLoop._restore_request_context_system_prompt(
                self,
                original_system_prompt,
                original_base_system_prompt,
            )
            if hasattr(self._agent, 'state') and self._agent.state != AgentState.IDLE:
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

    @classmethod
    def _serialize_message_content(cls, content: Any) -> Any:
        """Convert message content into JSON-serializable Python objects."""
        if content is None or isinstance(content, (str, int, float, bool)):
            return content
        if isinstance(content, list):
            return [cls._serialize_message_content(item) for item in content]
        if isinstance(content, dict):
            return {
                str(key): cls._serialize_message_content(value)
                for key, value in content.items()
            }
        if hasattr(content, "model_dump"):
            try:
                dumped = content.model_dump(exclude_none=True)
            except TypeError:
                dumped = content.model_dump()
            return cls._serialize_message_content(dumped)

        extracted: dict[str, Any] = {}
        for attr in (
            "type",
            "text",
            "image_url",
            "source",
            "file_path",
            "media_type",
            "filename",
            "name",
            "url",
            "detail",
            "data",
        ):
            if hasattr(content, attr):
                extracted[attr] = cls._serialize_message_content(getattr(content, attr))
        if extracted:
            return extracted
        return str(content)

    @classmethod
    def _multimodal_content_summary(cls, content: list[Any], max_text_chars: int) -> str:
        """Summarize multimodal content while dropping heavy binary payloads."""
        serialized = cls._serialize_message_content(content)
        if not isinstance(serialized, list):
            text = str(serialized or "")
            return text[:max_text_chars] if max_text_chars and len(text) > max_text_chars else text

        text_parts: list[str] = []
        image_count = 0
        document_count = 0
        file_refs: list[str] = []

        for block in serialized:
            if isinstance(block, dict):
                block_type = str(block.get("type") or "")
                if block_type == "text":
                    text = str(block.get("text") or "")
                    if text:
                        text_parts.append(text)
                    continue
                if block_type in {"image", "image_url"}:
                    image_count += 1
                    continue
                if block_type == "document":
                    document_count += 1
                    continue
                if block_type == "file":
                    file_path = str(block.get("file_path") or "")
                    if file_path:
                        file_refs.append(Path(file_path).name or file_path)
                    continue
            elif block:
                text_parts.append(str(block))

        text = "\n".join(part for part in text_parts if part).strip()
        if max_text_chars and len(text) > max_text_chars:
            text = text[:max_text_chars] + f"\n...[truncated {len(text) - max_text_chars} chars]"

        summary_parts = [text] if text else []
        if image_count:
            summary_parts.append(f"[{image_count} image attachment(s) omitted during context compression]")
        if document_count:
            summary_parts.append(f"[{document_count} document attachment(s) omitted during context compression]")
        if file_refs:
            shown = ", ".join(file_refs[:3])
            more = "" if len(file_refs) <= 3 else f" (+{len(file_refs) - 3} more)"
            summary_parts.append(f"[File attachment reference(s): {shown}{more}]")
        if not summary_parts:
            summary_parts.append("[Multimodal content omitted during context compression]")
        return "\n".join(summary_parts)

    @classmethod
    def _compress_message_content(cls, content: Any, max_chars: int) -> Any:
        """Truncate text content and summarize multimodal content for recovery."""
        if isinstance(content, str):
            if len(content) <= max_chars:
                return content
            return content[:max_chars] + f"\n...[truncated {len(content) - max_chars} chars]"
        if isinstance(content, list):
            return cls._multimodal_content_summary(content, max_chars)
        return content

    @classmethod
    def _compact_runtime_message_content(cls, message: Any, max_chars: int) -> Any:
        """Compact older runtime messages without preserving stale assistant authority."""
        role = cls._message_role_value(message)
        content = getattr(message, "content", None)

        if role == "assistant":
            tool_calls = getattr(message, "tool_calls", None) or []
            if tool_calls:
                if not content:
                    return content
                return (
                    "[assistant tool-call turn compacted; exact tool arguments/results "
                    "are recoverable via search_history]"
                )
            if content in (None, ""):
                return content
            if isinstance(content, list):
                return (
                    "[assistant multimodal reply compacted; earlier assistant analysis may be stale. "
                    "Prioritize the latest user request and current tool evidence. "
                    "Use search_history to recover exact earlier user/tool facts; search assistant replies "
                    "only when their literal wording is explicitly needed.]"
                )
            return (
                "[assistant reply compacted; earlier assistant analysis/conclusions may be stale. "
                "Prioritize the latest user request and current tool evidence. "
                "Use search_history to recover exact earlier user/tool facts; search assistant replies "
                "only when their literal wording is explicitly needed.]"
            )

        if role == "tool":
            compressed = cls._compress_message_content(content, max_chars)
            if compressed == content:
                return content
            if isinstance(compressed, str):
                tool_name = getattr(message, "name", None) or "tool"
                return f"[{tool_name} result compacted]\n{compressed}"
            return compressed

        if role == "user":
            compressed = cls._compress_message_content(content, max_chars)
            if compressed == content:
                return content
            if isinstance(compressed, str):
                return f"[earlier user message compacted]\n{compressed}"
            return compressed

        return cls._compress_message_content(content, max_chars)

    @classmethod
    def _message_content_char_count(cls, content: Any) -> int:
        """Return an approximate character count for text and multimodal payloads."""
        if content is None:
            return 0
        if isinstance(content, str):
            return len(content)
        if not isinstance(content, list):
            return len(str(content))

        total = 0
        serialized = cls._serialize_message_content(content)
        for block in serialized if isinstance(serialized, list) else [serialized]:
            if isinstance(block, dict):
                block_type = str(block.get("type") or "")
                if block_type == "text":
                    total += len(str(block.get("text") or ""))
                    continue
                if block_type == "image_url":
                    total += len(str((block.get("image_url") or {}).get("url") or ""))
                    continue
                if block_type == "image":
                    source = block.get("source") or {}
                    total += len(str(source.get("media_type") or ""))
                    total += len(str(source.get("data") or ""))
                    continue
                if block_type == "document":
                    source = block.get("source") or {}
                    total += len(str(source.get("media_type") or ""))
                    total += len(str(source.get("data") or ""))
                    total += len(str(block.get("filename") or ""))
                    continue
                if block_type == "file":
                    total += len(str(block.get("file_path") or ""))
                    total += len(str(block.get("media_type") or ""))
                    continue
                total += len(json.dumps(block, sort_keys=True, ensure_ascii=True, default=str))
                continue
            total += len(str(block))
        return total

    @classmethod
    def _msg_char_count(cls, msg) -> int:
        """Return total character count of a message's content."""
        return cls._message_content_char_count(getattr(msg, "content", None))

    def _estimate_runtime_tokens(self) -> int:
        """Rough token estimate from the agent's runtime messages (~4 chars/token)."""
        if not self._agent or not hasattr(self._agent, 'memory'):
            return 0
        return sum(self._msg_char_count(m) for m in self._agent.memory.messages) // 4

    def _build_session_recall_context(self, current_message: str) -> str:
        """Build a compact same-session context block for follow-up questions."""
        return build_session_compact_context(
            getattr(self, "_session", None),
            current_message,
        )

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
    def _callable_accepts_kwarg(func: Any, kwarg: str) -> bool:
        """Return True when *func* can accept *kwarg* as a keyword argument."""
        import inspect

        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            return False

        if any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        ):
            return True
        param = signature.parameters.get(kwarg)
        return bool(
            param
            and param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        )

    @staticmethod
    def _message_role_value(msg: Any) -> str | None:
        """Return the normalized string role value for a runtime message."""
        if isinstance(msg, dict):
            role = msg.get("role")
        else:
            role = getattr(msg, "role", None)
        return role.value if hasattr(role, "value") else role

    @staticmethod
    def _message_tool_calls_value(msg: Any) -> list:
        """Return tool_calls from object or dict payloads."""
        if isinstance(msg, dict):
            raw = msg.get("tool_calls")
        else:
            raw = getattr(msg, "tool_calls", None)
        return raw if isinstance(raw, list) else []

    @staticmethod
    def _tool_call_id_value(tool_call: Any) -> str | None:
        """Return a tool_call id from object or dict payloads."""
        if isinstance(tool_call, dict):
            value = tool_call.get("id")
        else:
            value = getattr(tool_call, "id", None)
        return str(value) if value else None

    @staticmethod
    def _message_tool_call_id_value(msg: Any) -> str | None:
        """Return a tool result's tool_call_id from object or dict payloads."""
        if isinstance(msg, dict):
            value = msg.get("tool_call_id") or msg.get("id")
        else:
            value = getattr(msg, "tool_call_id", None) or getattr(msg, "id", None)
        return str(value) if value else None

    @staticmethod
    def _set_message_tool_calls_value(msg: Any, value: list | None) -> None:
        """Set an assistant message's tool calls on object or dict payloads."""
        if isinstance(msg, dict):
            msg["tool_calls"] = value
        else:
            msg.tool_calls = value

    @classmethod
    def _collect_offered_tool_call_ids(cls, messages: list) -> set[str]:
        """Collect tool_call ids offered by assistant messages."""
        offered_ids: set[str] = set()
        for msg in messages:
            for tool_call in cls._message_tool_calls_value(msg):
                tool_call_id = cls._tool_call_id_value(tool_call)
                if tool_call_id:
                    offered_ids.add(tool_call_id)
        return offered_ids

    @classmethod
    def _collect_answered_tool_call_ids(cls, messages: list) -> set[str]:
        """Collect tool_call ids answered by tool-result messages."""
        answered_ids: set[str] = set()
        for msg in messages:
            if cls._message_role_value(msg) != "tool":
                continue
            tool_call_id = cls._message_tool_call_id_value(msg)
            if tool_call_id:
                answered_ids.add(tool_call_id)
        return answered_ids

    @classmethod
    def _adjust_message_start_to_preserve_tool_context(
        cls,
        messages: list,
        start_index: int,
        *,
        floor: int = 0,
    ) -> int:
        """Expand a kept-range start backward so tool trajectories stay intact."""
        if not messages:
            return max(0, floor)

        adjusted_index = max(floor, min(start_index, len(messages)))
        if adjusted_index <= floor or adjusted_index >= len(messages):
            return adjusted_index

        kept_answered_ids = cls._collect_answered_tool_call_ids(messages[adjusted_index:])
        if not kept_answered_ids:
            return adjusted_index

        kept_offered_ids = cls._collect_offered_tool_call_ids(messages[adjusted_index:])
        needed_ids = {tool_call_id for tool_call_id in kept_answered_ids if tool_call_id not in kept_offered_ids}
        if not needed_ids:
            return adjusted_index

        for index in range(adjusted_index - 1, floor - 1, -1):
            message = messages[index]
            if cls._message_role_value(message) != "assistant":
                continue

            matched_here = False
            for tool_call in cls._message_tool_calls_value(message):
                tool_call_id = cls._tool_call_id_value(tool_call)
                if tool_call_id and tool_call_id in needed_ids:
                    needed_ids.discard(tool_call_id)
                    matched_here = True

            if matched_here:
                adjusted_index = index
            if not needed_ids:
                break

        return adjusted_index

    @classmethod
    def _reorder_tool_messages(cls, messages: list) -> int:
        """Move tool results to immediately follow the issuing assistant turn."""
        if not messages:
            return 0

        claimed_tool_indices: set[int] = set()
        tool_messages_by_assistant_index: dict[int, list] = {}
        assistant_index_by_tool_call_id: dict[str, int] = {}

        for index, message in enumerate(messages):
            if cls._message_role_value(message) != "assistant":
                continue
            for tool_call in cls._message_tool_calls_value(message):
                tool_call_id = cls._tool_call_id_value(tool_call)
                if tool_call_id and tool_call_id not in assistant_index_by_tool_call_id:
                    assistant_index_by_tool_call_id[tool_call_id] = index

        for index, message in enumerate(messages):
            if cls._message_role_value(message) != "tool":
                continue

            tool_call_id = cls._message_tool_call_id_value(message)
            if not tool_call_id:
                continue

            assistant_index = assistant_index_by_tool_call_id.get(tool_call_id)
            if assistant_index is None:
                continue
            tool_messages_by_assistant_index.setdefault(assistant_index, []).append(message)
            claimed_tool_indices.add(index)

        if not claimed_tool_indices:
            return 0

        original_ids = [id(message) for message in messages]
        reordered_messages: list = []
        for index, message in enumerate(messages):
            if index in claimed_tool_indices:
                continue

            reordered_messages.append(message)
            if cls._message_role_value(message) != "assistant":
                continue

            tool_calls = cls._message_tool_calls_value(message)
            if not tool_calls:
                continue

            tool_order = {
                cls._tool_call_id_value(tool_call): position
                for position, tool_call in enumerate(tool_calls)
            }
            matched_tool_messages = tool_messages_by_assistant_index.get(index, [])
            matched_tool_messages.sort(
                key=lambda item: tool_order.get(
                    cls._message_tool_call_id_value(item),
                    len(tool_order),
                )
            )
            reordered_messages.extend(matched_tool_messages)

        reordered_ids = [id(message) for message in reordered_messages]
        if reordered_ids == original_ids:
            return 0

        moved = sum(
            1
            for original_id, reordered_id in zip(original_ids, reordered_ids)
            if original_id != reordered_id
        )
        messages[:] = reordered_messages
        logger.info(f"Reordered {moved} runtime message positions to restore tool adjacency")
        return moved

    @classmethod
    def _repair_tool_pairing(
        cls,
        messages: list,
        *,
        drop_unanswered_tool_calls: bool = True,
    ) -> int:
        """Remove orphaned tool results and tool calls after message deletion.

        Ensures every tool_call_id in a tool-result message has a matching
        tool_calls entry in a preceding assistant message, and vice-versa.
        Also removes tool-role messages with no tool_call_id at all (e.g.
        injected from session history without metadata).
        Without this, the LLM API rejects the conversation.

        When ``drop_unanswered_tool_calls`` is false, this keeps assistant
        tool_calls that are still in flight so live runtime normalization does
        not corrupt the next provider request.

        Returns the number of messages/calls removed.
        """
        removed = 0

        offered_ids = cls._collect_offered_tool_call_ids(messages)
        i = 0
        while i < len(messages):
            msg = messages[i]
            role = cls._message_role_value(msg)
            tool_call_id = cls._message_tool_call_id_value(msg)

            if role == "tool" and not tool_call_id:
                del messages[i]
                removed += 1
                continue
            if tool_call_id and tool_call_id not in offered_ids:
                del messages[i]
                removed += 1
                continue
            i += 1

        if drop_unanswered_tool_calls:
            answered_ids = cls._collect_answered_tool_call_ids(messages)
            for msg in messages:
                tc_list = cls._message_tool_calls_value(msg)
                if not tc_list:
                    continue
                original_len = len(tc_list)
                tc_list[:] = [tc for tc in tc_list if cls._tool_call_id_value(tc) in answered_ids]
                if len(tc_list) < original_len:
                    removed += original_len - len(tc_list)
                if not tc_list:
                    cls._set_message_tool_calls_value(msg, None)

        if removed:
            logger.info(f"Repaired tool pairing: removed {removed} orphaned messages/calls")
        return removed

    @classmethod
    def _normalize_runtime_tool_context(
        cls,
        messages: list,
        *,
        finalized: bool = False,
    ) -> int:
        """Repair runtime tool sequencing without pruning live tool trajectories."""
        normalized = cls._reorder_tool_messages(messages)
        normalized += cls._repair_tool_pairing(
            messages,
            drop_unanswered_tool_calls=finalized,
        )
        return normalized

    def _uses_strict_tool_turn_order(self) -> bool:
        """True for providers/models that reject non-adjacent function call turns.

        OpenAI-compatible providers and Gemini require tool-result messages to immediately
        follow the assistant message that issued the tool_calls.
        """
        provider_raw = getattr(self, "provider", None)
        model_raw = getattr(self, "model", None)
        base_url_raw = getattr(self, "base_url", None)

        provider = provider_raw.strip().lower() if isinstance(provider_raw, str) else ""
        model = model_raw.strip().lower() if isinstance(model_raw, str) else ""
        base_url = base_url_raw.strip().lower() if isinstance(base_url_raw, str) else ""

        native_non_openai_compatible = {
            "anthropic",
            "gemini",
            "google",
            "google_ai_studio",
            "google-ai-studio",
        }
        openai_compatible = provider in {"openai", "openrouter", "deepseek"} or (
            provider not in native_non_openai_compatible and bool(base_url and "/v1" in base_url)
        )
        if openai_compatible:
            return True
        if provider in {"gemini", "google", "google_ai_studio", "google-ai-studio"}:
            return True
        if any(prefix in model for prefix in ("gpt-", "o3", "o4", "openai/")):
            return True
        if "gemini" in model or model.startswith("google/"):
            return True
        if "api.openai.com" in base_url:
            return True
        return "generativelanguage.googleapis.com" in base_url

    def _should_skip_runtime_next_step_prompt(self) -> bool:
        """Avoid synthetic user turns after tool responses for strict function-turn providers."""
        if not self._uses_strict_tool_turn_order():
            return False
        if not self._agent or not hasattr(self._agent, "memory"):
            return False

        messages = getattr(self._agent.memory, "messages", None)
        if not isinstance(messages, list):
            return False

        for message in reversed(messages):
            role = self._message_role_value(message)
            if role == "system":
                continue
            return role == "tool"
        return False

    @classmethod
    def _snap_drop_end_to_turn_boundary(
        cls, messages: list, keep_head: int, desired_end: int
    ) -> int:
        """Return an index <= desired_end that sits on a safe boundary.

        A *safe* drop boundary is a position ``k`` where:
          * ``messages[k]`` is ``role="user"`` (start of the next turn), OR
          * ``messages[k]`` is ``role="assistant"`` with no ``tool_calls`` and
            not followed immediately by a ``role="tool"`` message (i.e. the
            tool chain that the previous assistant opened is fully settled
            at the end of index ``k - 1``).

        Snapping ensures we never split an assistant(tool_calls) → tool(result)
        pair, which would leak orphans the provider rejects.  If no boundary
        is found, we fall back to ``keep_head`` (i.e. drop nothing) so the
        safety invariant wins over the token-budget target.
        """
        if desired_end <= keep_head:
            return keep_head
        end = min(desired_end, len(messages))

        def _role(i: int) -> str | None:
            if i < 0 or i >= len(messages):
                return None
            return cls._message_role_value(messages[i])

        def _has_tool_calls(i: int) -> bool:
            if i < 0 or i >= len(messages):
                return False
            tc = getattr(messages[i], "tool_calls", None)
            return bool(tc)

        # Walk backwards from desired_end looking for a clean cut.  The cut
        # index k means "delete messages[keep_head:k]", so messages[k]
        # becomes the new first non-head message.
        for k in range(end, keep_head, -1):
            role = _role(k)
            if role == "user":
                return k
            if role == "assistant" and not _has_tool_calls(k):
                if _role(k + 1) != "tool":
                    return k
        return keep_head

    def _insert_compaction_marker(self, messages: list, dropped: int) -> None:
        """Insert a runtime-only marker telling the model history was trimmed.

        The marker is a ``role="user"`` system-style message that documents
        how many prior messages were removed from runtime memory and points
        the model at the :class:`SearchHistoryTool` as the recovery path.
        It is **not** persisted to the session store — the store still holds
        the full transcript — so adding the marker is idempotent across
        turns.
        """
        if not self._agent or not hasattr(self._agent, "memory"):
            return
        try:
            from spoon_ai.schema import Message, Role  # lazy import
        except Exception:  # pragma: no cover - defensive
            return

        marker_text = (
            "[history-compacted] "
            f"{dropped} older message(s) (including any tool-call/result pairs) "
            "were removed from this turn's in-memory context to fit the token "
            "budget. The persisted session transcript was NOT cleared. The "
            "latest real user request remains authoritative. Earlier assistant "
            "analysis/conclusions in compacted history may be tentative or "
            "stale, so do not treat them as current instructions. If you need "
            "an exact earlier tool result, tool argument, image description, "
            "or user statement, call the `search_history` tool (scope='current'). "
            "Plain earlier assistant replies are omitted there by default; "
            "search them explicitly only when their literal wording matters."
        )
        try:
            marker = Message(role=Role.USER, content=marker_text)
        except Exception:
            return

        insert_at = 1 if messages else 0
        messages.insert(insert_at, marker)

    @staticmethod
    def _is_history_compaction_marker(msg: Any) -> bool:
        """True when *msg* is the runtime-only history-compacted marker."""
        role = AgentLoop._message_role_value(msg)
        if role != "user":
            return False
        content = getattr(msg, "content", None)
        return isinstance(content, str) and content.startswith("[history-compacted]")

    def _latest_real_user_message_index(self, messages: list[Any]) -> int | None:
        """Return the latest user-authored message that must survive compaction intact."""
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if self._message_role_value(message) != "user":
                continue
            if self._is_next_step_user_msg(message):
                continue
            if self._is_history_compaction_marker(message):
                continue
            return index
        return None

    def _compress_runtime_context(
        self,
        *,
        force: bool = False,
        budget_tokens: int | None = None,
    ) -> int:
        """Proactively compress the agent's runtime context.

        Strategy (inspired by Openclaw's context engine):
        1. Drop redundant next_step_prompt user messages (keep only the latest).
        2. Truncate ALL older message content (tool results, assistant, user).
        3. If still over budget, drop entire old message rounds.

        Trigger: estimated tokens > the active runtime compaction budget.
        """
        if not self._agent or not hasattr(self._agent, 'memory'):
            return 0

        messages = self._agent.memory.messages
        normalized = self._normalize_runtime_tool_context(messages)
        if len(messages) <= 6:
            return normalized

        estimated = self._estimate_runtime_tokens()
        budget = budget_tokens if budget_tokens is not None else self._runtime_compaction_trigger_budget()

        if not force and estimated <= budget:
            return normalized

        logger.warning(
            f"Context compression triggered: ~{estimated:,} tokens "
            f"(budget: {budget:,}, window: {self.context_window:,}). "
            f"Messages: {len(messages)}"
        )

        compressed = normalized

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

        protected_user_index = self._latest_real_user_message_index(messages)

        # Phase 2: Compact older messages except the first and last 6.
        keep_tail = min(6, len(messages))
        max_content = 300
        for i in range(1, max(1, len(messages) - keep_tail)):
            if protected_user_index is not None and i == protected_user_index:
                continue
            msg = messages[i]
            original_content = getattr(msg, "content", None)
            compressed_content = self._compact_runtime_message_content(msg, max_content)
            if compressed_content != original_content:
                msg.content = compressed_content
                compressed += 1

        # Phase 3: If still over budget, drop oldest rounds (keep first + last 8).
        #
        # *Segment-aware* drop: we only cut along user-turn boundaries so a
        # drop never splits an ``assistant(tool_calls) -> tool(result)``
        # chain. Partial splits would leak through as orphaned tool
        # messages and the LLM provider would reject the whole request.
        # The persisted session store is untouched, so anything dropped
        # here remains reachable via ``search_history``.
        estimated = self._estimate_runtime_tokens()
        dropped_in_phase3 = 0
        if estimated > budget and len(messages) > 12:
            keep_head = 1
            keep_tail_drop = min(8, len(messages) - 1)
            tail_start = len(messages) - keep_tail_drop
            if protected_user_index is not None:
                tail_start = min(tail_start, protected_user_index)
            droppable = max(0, tail_start - keep_head)
            if droppable > 4:
                desired = droppable // 2
                snap_end = self._snap_drop_end_to_turn_boundary(
                    messages, keep_head, keep_head + desired
                )
                keep_start = self._adjust_message_start_to_preserve_tool_context(
                    messages,
                    snap_end,
                    floor=keep_head,
                )
                actual_drop = max(0, keep_start - keep_head)
                if actual_drop > 0:
                    del messages[keep_head:keep_head + actual_drop]
                    compressed += actual_drop
                    dropped_in_phase3 = actual_drop
                    logger.info(
                        f"Phase 3: dropped {actual_drop} oldest messages "
                        f"(segment-snapped from desired {desired})"
                    )

        # Phase 3b: Insert a compaction marker so the model knows prior
        # turns are recoverable via ``search_history``. The marker is
        # runtime-only; it never touches the persisted session.
        if dropped_in_phase3 > 0:
            self._insert_compaction_marker(messages, dropped_in_phase3)

        # Phase 4: Repair tool_use/tool_result pairing broken by message deletion.
        compressed += self._normalize_runtime_tool_context(messages)

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
        compressed = self._normalize_runtime_tool_context(messages)
        if len(messages) <= 4:
            return compressed

        protected_user_index = self._latest_real_user_message_index(messages)

        # Compact older messages aggressively while preserving the latest real user turn.
        for index, msg in enumerate(messages):
            if protected_user_index is not None and index == protected_user_index:
                continue
            original_content = getattr(msg, "content", None)
            compressed_content = self._compact_runtime_message_content(msg, 150)
            if compressed_content != original_content:
                msg.content = compressed_content
                compressed += 1

        # Drop all but first + last 6 messages (segment-aware)
        dropped = 0
        if len(messages) > 8:
            keep_head = 1
            keep_tail = min(6, len(messages) - 1)
            desired_end = len(messages) - keep_tail
            if protected_user_index is not None:
                desired_end = min(desired_end, protected_user_index)
            snap_end = self._snap_drop_end_to_turn_boundary(
                messages, keep_head, desired_end
            )
            keep_start = self._adjust_message_start_to_preserve_tool_context(
                messages,
                snap_end,
                floor=keep_head,
            )
            actual_drop = max(0, keep_start - keep_head)
            if actual_drop > 0:
                del messages[keep_head:keep_head + actual_drop]
                compressed += actual_drop
                dropped = actual_drop

        if dropped > 0:
            self._insert_compaction_marker(messages, dropped)

        # Repair tool pairing broken by message deletion
        compressed += self._normalize_runtime_tool_context(messages)

        logger.warning(f"Force-compressed {compressed} messages/results for recovery")
        return compressed

    def _install_anti_loop_tracker(self, base_prompt: str) -> None:
        """Wrap think() for compaction/logging without prompt-based routing."""
        agent = self._agent
        if agent is None:
            return
        self._install_tool_call_protocol_guards()

        agent_loop = self
        original_think = getattr(agent, "_spoon_bot_base_think", None)
        if original_think is None:
            original_think = agent.think
            setattr(agent, "_spoon_bot_base_think", original_think)
        else:
            agent.think = original_think

        async def _tracked_think(*args: Any, **kwargs: Any) -> bool:
            agent_loop._compress_runtime_context()

            desired_next_step_prompt = agent.next_step_prompt or base_prompt
            suppress_runtime_prompt = AgentLoop._should_skip_runtime_next_step_prompt(agent_loop)
            if suppress_runtime_prompt:
                logger.info(
                    "Suppressing synthetic next_step_prompt user turn for strict tool-call provider"
                )
                agent.next_step_prompt = None
            else:
                agent.next_step_prompt = desired_next_step_prompt

            try:
                result = await original_think(*args, **kwargs)
            finally:
                agent.next_step_prompt = desired_next_step_prompt

            _log_agent_reasoning()
            _log_tool_calls()

            return result

        def _log_agent_reasoning():
            """Extract and log the agent's reasoning text from its last response."""
            from spoon_bot.utils.privacy import mask_secrets
            summary = getattr(agent, "last_reasoning_summary", None)
            if isinstance(summary, str) and summary.strip():
                captured_summary = agent_loop._capture_reasoning_text(mask_secrets(summary.strip()))
                if captured_summary:
                    logger.info(f"💭 Agent reasoning: {captured_summary}")
                return
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
                captured = agent_loop._capture_reasoning_text(safe_text)
                if captured:
                    logger.info(f"💭 Agent reasoning: {captured}")
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

    def _install_tool_call_protocol_guards(self) -> None:
        """Install transport-level guards for provider tool-call invariants."""
        agent = getattr(self, "_agent", None)
        llm = getattr(agent, "llm", None)
        ask_tool = getattr(llm, "ask_tool", None)
        if llm is None or not callable(ask_tool):
            return

        original_ask_tool = getattr(llm, "_spoon_bot_base_ask_tool", None)
        if original_ask_tool is None:
            original_ask_tool = ask_tool
            try:
                setattr(llm, "_spoon_bot_base_ask_tool", original_ask_tool)
            except Exception:
                return

        agent_loop = self

        async def _guarded_ask_tool(*args: Any, **kwargs: Any):
            should_disable_parallel = agent_loop._should_disable_parallel_tool_calls()
            should_textualize_tool_history = (
                agent_loop._should_textualize_tool_history_for_provider()
            )
            call_args = list(args)
            call_kwargs = dict(kwargs)
            if should_textualize_tool_history:
                if "messages" in call_kwargs:
                    call_kwargs["messages"] = AgentLoop._textualize_tool_history(
                        call_kwargs["messages"]
                    )
                elif call_args:
                    call_args[0] = AgentLoop._textualize_tool_history(call_args[0])

            injected_parallel_flag = False
            if should_disable_parallel and "parallel_tool_calls" not in call_kwargs:
                call_kwargs["parallel_tool_calls"] = False
                injected_parallel_flag = True

            if "max_tokens" not in call_kwargs and "max_completion_tokens" not in call_kwargs:
                call_kwargs["max_tokens"] = AgentLoop._tool_call_output_token_budget()

            async def _ask_with_parallel_fallback(request_kwargs: dict[str, Any]):
                try:
                    return await original_ask_tool(*call_args, **request_kwargs)
                except Exception as exc:
                    if injected_parallel_flag and "parallel_tool_calls" in str(exc):
                        retry_kwargs = dict(request_kwargs)
                        retry_kwargs.pop("parallel_tool_calls", None)
                        return await original_ask_tool(*call_args, **retry_kwargs)
                    raise

            try:
                response = await _ask_with_parallel_fallback(call_kwargs)
            except Exception:
                raise

            retry_reason = AgentLoop._tool_response_needs_retry(response)
            if retry_reason:
                retry_kwargs = dict(call_kwargs)
                current_budget = retry_kwargs.get("max_tokens") or retry_kwargs.get("max_completion_tokens")
                try:
                    current_budget_int = int(current_budget) if current_budget is not None else None
                except (TypeError, ValueError):
                    current_budget_int = None
                retry_kwargs["max_tokens"] = AgentLoop._tool_call_retry_token_budget(current_budget_int)
                retry_kwargs.pop("max_completion_tokens", None)
                logger.warning(
                    "Retrying provider tool turn because tool-call arguments were incomplete: "
                    f"{retry_reason}"
                )
                retry_response = await _ask_with_parallel_fallback(retry_kwargs)
                retry_blocker = AgentLoop._tool_response_needs_retry(retry_response)
                if retry_blocker:
                    response = AgentLoop._block_incomplete_tool_calls(retry_response, retry_blocker)
                else:
                    response = retry_response

            if should_disable_parallel:
                AgentLoop._coerce_response_to_single_tool_call(response)
            return response

        try:
            setattr(llm, "ask_tool", _guarded_ask_tool)
        except Exception:
            return

    def _uses_openai_compatible_tool_api(self) -> bool:
        """Return true when the active provider accepts OpenAI-style tool kwargs."""
        provider_raw = getattr(self, "provider", None)
        base_url_raw = getattr(self, "base_url", None)
        provider = provider_raw.strip().lower() if isinstance(provider_raw, str) else ""
        base_url = base_url_raw.strip().lower() if isinstance(base_url_raw, str) else ""

        if provider in {"openai", "openrouter", "deepseek"}:
            return True
        if provider in {"anthropic", "gemini", "google", "google_ai_studio", "google-ai-studio"}:
            return False
        return bool(base_url and "/v1" in base_url)

    def _should_disable_parallel_tool_calls(self) -> bool:
        """Prefer one tool result turn at a time for strict OpenAI-compatible APIs."""
        import os as _os

        raw = _os.getenv("SPOON_BOT_PARALLEL_TOOL_CALLS")
        if raw is not None:
            return raw.strip().lower() in {"0", "false", "no", "off"}
        return self._uses_strict_tool_turn_order() and self._uses_openai_compatible_tool_api()

    def _should_textualize_tool_history_for_provider(self) -> bool:
        """Avoid replaying native completed tool-result blocks to strict providers."""
        import os as _os

        raw = _os.getenv("SPOON_BOT_TEXTUALIZE_TOOL_HISTORY")
        if raw is not None:
            return raw.strip().lower() in {"1", "true", "yes", "on", "always"}

        return self._uses_strict_tool_turn_order()

    @classmethod
    def _textualize_tool_history(cls, messages: Any) -> Any:
        """Return provider-safe messages with completed tool turns summarized as text."""
        if not isinstance(messages, list) or not messages:
            return messages

        converted: list[Message] = []
        index = 0
        changed = False
        while index < len(messages):
            msg = messages[index]
            role = cls._message_role_value(msg)
            if role == "assistant" and cls._message_tool_calls_value(msg):
                tool_calls = cls._message_tool_calls_value(msg)
                expected_ids = {
                    cls._tool_call_id_value(tool_call)
                    for tool_call in tool_calls
                    if cls._tool_call_id_value(tool_call)
                }
                results: list[Any] = []
                cursor = index + 1
                while cursor < len(messages):
                    next_msg = messages[cursor]
                    if cls._message_role_value(next_msg) != "tool":
                        break
                    tool_call_id = cls._message_tool_call_id_value(next_msg)
                    if expected_ids and tool_call_id not in expected_ids:
                        break
                    results.append(next_msg)
                    cursor += 1

                converted.append(
                    cls._build_tool_history_summary_message(msg, tool_calls, results)
                )
                changed = True
                index = cursor
                continue

            if role == "tool":
                converted.append(cls._build_standalone_tool_summary_message(msg))
                changed = True
                index += 1
                continue

            converted.append(cls._copy_message_without_provider_tool_fields(msg))
            index += 1

        return converted if changed else messages

    @classmethod
    def _copy_message_without_provider_tool_fields(cls, msg: Any) -> Message:
        role = cls._message_role_value(msg) or "user"
        if role not in {"system", "user", "assistant"}:
            role = "assistant"
        content = cls._stream_message_attr(msg, "content", None)
        if content is None:
            content = cls._stream_message_attr(msg, "text_content", "") or ""
        return Message(role=role, content=content)

    @classmethod
    def _build_tool_history_summary_message(
        cls,
        assistant_msg: Any,
        tool_calls: list[Any],
        tool_results: list[Any],
    ) -> Message:
        from spoon_bot.utils.privacy import mask_secrets

        intro = cls._stream_message_attr(assistant_msg, "text_content", None)
        if not isinstance(intro, str) or not intro:
            intro = cls._stream_message_attr(assistant_msg, "content", "") or ""
        parts: list[str] = []
        if isinstance(intro, str) and intro.strip():
            parts.append(mask_secrets(intro.strip()))
        parts.append("[Tool execution summary]")

        result_by_id = {
            cls._message_tool_call_id_value(result): result
            for result in tool_results
            if cls._message_tool_call_id_value(result)
        }
        for tool_call in tool_calls:
            tool_name, arguments = cls._tool_call_name_and_arguments(tool_call)
            tool_call_id = cls._tool_call_id_value(tool_call)
            result_msg = result_by_id.get(tool_call_id)
            result_text = ""
            if result_msg is not None:
                result_text = cls._stream_message_attr(result_msg, "text_content", None)
                if not isinstance(result_text, str) or not result_text:
                    result_text = cls._stream_message_attr(result_msg, "content", "") or ""
            if len(result_text) > 4000:
                result_text = result_text[:4000] + "\n[truncated]"
            parts.append(
                "- "
                + mask_secrets(str(tool_name or "tool"))
                + "("
                + mask_secrets(str(arguments or ""))
                + "): "
                + mask_secrets(str(result_text or "No tool result was captured."))
            )

        return Message(role="assistant", content="\n".join(parts))

    @classmethod
    def _build_standalone_tool_summary_message(cls, msg: Any) -> Message:
        from spoon_bot.utils.privacy import mask_secrets

        name = cls._stream_message_attr(msg, "name", "") or "tool"
        content = cls._stream_message_attr(msg, "text_content", None)
        if not isinstance(content, str) or not content:
            content = cls._stream_message_attr(msg, "content", "") or ""
        if len(content) > 4000:
            content = content[:4000] + "\n[truncated]"
        return Message(
            role="assistant",
            content=f"[Tool execution summary]\n- {mask_secrets(str(name))}: {mask_secrets(str(content))}",
        )

    @staticmethod
    def _coerce_response_to_single_tool_call(response: Any) -> int:
        """Keep one tool call when the transport requires serial tool-result turns."""
        tool_calls = getattr(response, "tool_calls", None)
        if not isinstance(tool_calls, list) or len(tool_calls) <= 1:
            return 0

        dropped = len(tool_calls) - 1
        try:
            response.tool_calls = tool_calls[:1]
        except Exception:
            return 0

        metadata = getattr(response, "metadata", None)
        if isinstance(metadata, dict):
            metadata["serial_tool_calls_enforced"] = True
            metadata["dropped_parallel_tool_calls"] = dropped
        logger.info(
            "Serial tool-call guard kept 1 tool call and deferred "
            f"{dropped} parallel tool call(s)"
        )
        return dropped

    @staticmethod
    def _tool_call_output_token_budget() -> int:
        """Return the default completion budget for tool-producing turns."""
        raw = os.getenv("SPOON_BOT_TOOL_CALL_MAX_TOKENS")
        if raw is None:
            return 16_384
        try:
            return max(1_024, min(200_000, int(raw.strip())))
        except (TypeError, ValueError):
            return 16_384

    @staticmethod
    def _tool_call_retry_token_budget(current: int | None = None) -> int:
        """Return the retry completion budget for truncated tool-producing turns."""
        raw = os.getenv("SPOON_BOT_TOOL_CALL_RETRY_MAX_TOKENS")
        if raw is not None:
            try:
                return max(1_024, min(200_000, int(raw.strip())))
            except (TypeError, ValueError):
                pass
        base = current or AgentLoop._tool_call_output_token_budget()
        return max(base * 2, 32_768)

    @staticmethod
    def _tool_call_arguments_json_error(arguments: Any) -> str | None:
        """Return an error string when tool-call arguments are not complete JSON."""
        if arguments is None:
            return None
        if isinstance(arguments, dict):
            return None
        if not isinstance(arguments, str):
            return f"unsupported argument type {type(arguments).__name__}"
        text = arguments.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return f"{exc.msg} at char {exc.pos}"
        if not isinstance(parsed, dict):
            return "tool arguments must decode to a JSON object"
        return None

    @staticmethod
    def _tool_call_name_and_raw_arguments(tool_call: Any) -> tuple[str, Any]:
        """Return a tool-call name and argument payload from object or dict shapes."""
        fn = getattr(tool_call, "function", None) or (
            tool_call.get("function") if isinstance(tool_call, dict) else None
        )
        if fn is not None:
            name = getattr(fn, "name", None) or (
                fn.get("name") if isinstance(fn, dict) else None
            )
            arguments = getattr(fn, "arguments", None) or (
                fn.get("arguments") if isinstance(fn, dict) else None
            )
            return str(name or ""), arguments
        name = getattr(tool_call, "name", None) or (
            tool_call.get("name") if isinstance(tool_call, dict) else None
        )
        arguments = getattr(tool_call, "arguments", None) or (
            tool_call.get("arguments") if isinstance(tool_call, dict) else None
        )
        return str(name or ""), arguments

    @staticmethod
    def _tool_response_needs_retry(response: Any) -> str | None:
        """Return why a tool response should be retried before executing tools."""
        tool_calls = getattr(response, "tool_calls", None)
        if not isinstance(tool_calls, list) or not tool_calls:
            return None

        finish_reason = str(
            getattr(response, "finish_reason", None)
            or getattr(response, "native_finish_reason", None)
            or ""
        ).strip().lower()
        if finish_reason in {"length", "max_tokens", "max_output_tokens"}:
            return f"finish_reason={finish_reason}"

        for tool_call in tool_calls:
            tool_name, arguments = AgentLoop._tool_call_name_and_raw_arguments(tool_call)
            error = AgentLoop._tool_call_arguments_json_error(arguments)
            if error:
                return f"{tool_name or 'tool'} arguments are incomplete JSON: {error}"
        return None

    @staticmethod
    def _block_incomplete_tool_calls(response: Any, reason: str) -> Any:
        """Prevent execution of tool calls that the provider returned incomplete."""
        message = (
            "Tool call generation was truncated before valid JSON arguments were complete. "
            f"Reason: {reason}. Retry the request with smaller tool payloads or shorter "
            "file writes instead of executing partial arguments."
        )
        try:
            response.tool_calls = []
        except Exception:
            pass
        try:
            response.content = message
        except Exception:
            pass
        metadata = getattr(response, "metadata", None)
        if isinstance(metadata, dict):
            metadata["incomplete_tool_calls_blocked"] = True
            metadata["incomplete_tool_call_reason"] = reason
        return response

    def _restore_agent_think(self) -> None:
        """Restore the agent's base think() implementation after a request."""
        agent = self._agent
        if agent is None:
            return
        original_think = getattr(agent, "_spoon_bot_base_think", None)
        if original_think is not None:
            agent.think = original_think

    def _reset_reasoning_capture(self) -> None:
        """Reset request-scoped reasoning captured from tracked think logs."""
        self._latest_reasoning_excerpt = None
        self._pending_reasoning_chunks = []

    def _reset_runtime_notices(self) -> None:
        """Reset request-scoped runtime notices surfaced to clients."""
        self._pending_runtime_notices = []

    def _queue_runtime_notice(self, **notice: Any) -> None:
        """Queue a request-scoped runtime notice for later streaming."""
        pending = getattr(self, "_pending_runtime_notices", None)
        if not isinstance(pending, list):
            pending = []
            self._pending_runtime_notices = pending
        payload = {
            key: value
            for key, value in notice.items()
            if value is not None
        }
        if payload:
            pending.append(payload)

    def _drain_runtime_notices(self) -> list[dict[str, Any]]:
        """Return queued runtime notices and clear the request buffer."""
        pending = getattr(self, "_pending_runtime_notices", None)
        if not isinstance(pending, list):
            self._pending_runtime_notices = []
            return []
        notices = [item for item in pending if isinstance(item, dict)]
        self._pending_runtime_notices = []
        return notices

    @staticmethod
    def _runtime_notice_to_stream_event(
        notice: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Convert an internal runtime notice into a stream event."""
        if not isinstance(notice, dict):
            return None

        kind = str(notice.get("kind") or "").strip().lower()
        if kind != "runtime_compaction":
            return None

        stage = str(notice.get("stage") or "").strip().lower()
        if stage == "overflow_retry":
            delta = (
                "Context window exceeded. Earlier history was compacted and the "
                "agent resumed the latest request."
            )
        else:
            delta = (
                "Context window near limit. Earlier history was compacted before "
                "continuing the latest request."
            )

        metadata = dict(notice)
        metadata.setdefault("visible", True)
        return {
            "type": "notice",
            "delta": delta,
            "metadata": metadata,
        }

    def _capture_reasoning_text(self, text: str | None) -> str | None:
        """Store a reasoning excerpt so gateway transports can reuse it."""
        normalized = str(text or "").strip()
        if not normalized:
            return None
        if normalized == self._latest_reasoning_excerpt:
            return normalized
        self._latest_reasoning_excerpt = normalized
        self._pending_reasoning_chunks.append(normalized)
        return normalized

    def _drain_reasoning_chunks(self) -> list[str]:
        """Return pending reasoning excerpts and clear the queue."""
        pending = [
            text
            for text in self._pending_reasoning_chunks
            if isinstance(text, str) and text.strip()
        ]
        self._pending_reasoning_chunks = []
        return pending

    @staticmethod
    def _normalize_comparable_text(text: str | None) -> str:
        """Collapse whitespace so duplicate text can be compared reliably."""
        return " ".join(str(text or "").split())

    @staticmethod
    def _resolve_stream_fallback_delta(
        streamed_text: str | None,
        final_text: str | None,
    ) -> tuple[str, str]:
        """Return merged final content plus the missing delta to emit."""
        streamed = str(streamed_text or "")
        final = str(final_text or "")
        if not streamed or not final:
            return final, final

        prefix_candidates = [streamed]
        trimmed_streamed = streamed.rstrip()
        if trimmed_streamed and trimmed_streamed != streamed:
            prefix_candidates.append(trimmed_streamed)

        for prefix in prefix_candidates:
            if prefix and final.startswith(prefix):
                return final, final[len(prefix):]

        return streamed + final, final

    def _looks_like_duplicate_thinking(
        self,
        thinking_text: str | None,
        content_text: str | None,
    ) -> bool:
        """Return True when a thinking payload is effectively the final answer."""
        normalized_thinking = self._normalize_comparable_text(thinking_text)
        normalized_content = self._normalize_comparable_text(content_text)
        if not normalized_thinking or not normalized_content:
            return False
        if normalized_thinking == normalized_content:
            return True
        shorter, longer = sorted(
            (normalized_thinking, normalized_content),
            key=len,
        )
        return len(shorter) >= 64 and shorter in longer

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

    @staticmethod
    def _looks_like_pseudo_tool_call_text(content: str) -> bool:
        """Return True when plain text is pretending that tools were called."""
        text = str(content or "")
        if "Observed output of cmd" not in text or "execution:" not in text:
            return False
        return bool(re.search(
            r"(?im)^\s*(?:[-*]\s*)?`?[a-z_][a-z0-9_]*\s*"
            r"\([^)\n]{0,700}\)`?\s*:\s*Observed output of cmd\b.*\bexecution:\s*",
            text,
        ))

    @staticmethod
    def _build_pseudo_tool_call_repair_prompt(
        user_request: str,
        pseudo_content: str,
    ) -> str:
        """Build an internal retry prompt after the model emitted fake tool text."""
        excerpt = AgentLoop._compact_tool_evidence_text(pseudo_content, limit=900)
        return (
            "[INTERNAL TOOL-CALL REPAIR]\n"
            "Your previous assistant output wrote tool-call-shaped Markdown as plain text. "
            "Those actions were NOT executed by the runtime, and any claimed outputs in that "
            "text are invalid.\n\n"
            "Discard the invalid text below and continue the latest user request by calling "
            "the real tools through the tool-call API. Do not describe a tool call, do not "
            "write `tool_name(...)` in Markdown, and do not claim success until actual tool "
            "results have been returned. If the needed tool is unavailable or fails, report "
            "that concrete blocker.\n\n"
            f"Latest user request:\n{user_request}\n\n"
            f"Invalid plain-text pseudo tool output:\n{excerpt}"
        )

    @staticmethod
    def _drop_pseudo_tool_call_assistant_messages(self, start_index: int) -> int:
        """Remove runtime assistant messages that contain fake tool-call transcripts."""
        if not isinstance(start_index, int) or start_index < 0:
            return 0
        try:
            messages = AgentLoop._get_runtime_memory_messages(self)
        except Exception:
            return 0
        if not isinstance(messages, list) or start_index >= len(messages):
            return 0

        removed = 0
        for index in range(len(messages) - 1, start_index - 1, -1):
            msg = messages[index]
            role = AgentLoop._stream_message_role(msg).lower()
            if role != "assistant":
                continue
            if AgentLoop._message_tool_calls_value(msg):
                continue
            content = AgentLoop._stream_message_attr(msg, "text_content", None)
            if content in (None, ""):
                content = AgentLoop._stream_message_attr(msg, "content", "")
            if AgentLoop._looks_like_pseudo_tool_call_text(str(content or "")):
                del messages[index]
                removed += 1
        return removed

    @staticmethod
    def _extract_run_result_text(result: Any) -> str:
        """Normalize a spoon-core run result to plain text."""
        if hasattr(result, "content") and result.content is not None:
            return str(result.content or "")
        if hasattr(result, "content"):
            return str(result) if str(result) != "None" else ""
        if result is None:
            return ""
        return str(result)

    @staticmethod
    def _looks_like_raw_tool_transcript_leak(content: str) -> bool:
        """Return True when provider fallback text is dominated by tool transcript artifacts."""
        text = str(content or "")
        if "Observed output of cmd" not in text:
            return False
        if len(text) >= 4_000:
            return True
        if re.match(r"(?is)^\s*Observed output of cmd\b.*\bexecution:\s*", text):
            return True
        if AgentLoop._looks_like_pseudo_tool_call_text(text):
            return True
        return bool(re.search(
            r"(?is)(?:^|.)\s*Step\s+\d+:\s*Observed output of cmd\b.*\bexecution:\s*",
            text,
        ))

    @staticmethod
    def _compact_tool_evidence_text(value: Any, *, limit: int = 700) -> str:
        text = AgentLoop._stringify_stream_payload(value)
        text = re.sub(r"^Observed output of cmd [^\n]* execution:\s*", "", text.strip())
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        if lines:
            text = "\n".join(lines)
        text = AgentLoop._mask_user_visible_text(text)
        if len(text) <= limit:
            return text
        head = text[: max(120, limit // 2)].rstrip()
        tail = text[-max(120, limit // 3):].lstrip()
        return f"{head}\n... (tool output truncated)\n{tail}"

    def _build_raw_tool_transcript_leak_response(self, start_index: int) -> str:
        """Build a bounded fallback when raw tool transcript text leaks as final content."""
        parts = [
            "The model returned raw tool transcript text instead of a clean final answer.",
            "Recent tool evidence:",
        ]

        try:
            messages = AgentLoop._get_runtime_memory_messages(self)
        except Exception:
            messages = []
        if not isinstance(messages, list):
            messages = []
        if isinstance(start_index, int) and start_index >= 0:
            messages = messages[start_index:]

        tool_lines: list[str] = []
        for msg in messages:
            if AgentLoop._stream_message_role(msg).lower() != "tool":
                continue
            content = AgentLoop._stream_message_attr(msg, "text_content", None)
            if content in (None, ""):
                content = AgentLoop._stream_message_attr(msg, "content", "")
            summary = AgentLoop._compact_tool_evidence_text(content)
            if not summary:
                continue
            name = str(
                AgentLoop._stream_message_attr(msg, "name", None)
                or AgentLoop._stream_message_attr(msg, "tool_name", None)
                or "tool"
            )
            tool_lines.append(f"Tool `{name}` output:\n{summary}")

        if tool_lines:
            parts.extend(tool_lines[-3:])
        else:
            parts.append("No bounded tool evidence was captured before cleanup.")
        parts.append("Please continue or retry if you need the agent to finish the remaining work.")
        return "\n\n".join(parts)

    @staticmethod
    def _looks_like_internal_scratchpad_text(text: str) -> bool:
        """Return True for short legacy ASCII provider-surfaced planning notes."""
        compact = " ".join(str(text or "").strip().split())
        if len(compact) < 12 or len(compact) > 500:
            return False

        ascii_chars = sum(1 for ch in compact if ord(ch) < 128)
        if ascii_chars / max(len(compact), 1) < 0.70:
            return False
        return bool(re.search(
            r"(?ix)"
            r"\b("
            r"need\s+(?:to|answer|respond|reply|summarize|mention|follow|continue|inspect|check|use|run|find|resolve|handle|tool)|"
            r"i\s+(?:need|should|have\s+to)|"
            r"i(?:'|’)ll\s+(?:run|check|inspect|fetch|use|look|read|verify|confirm|execute|search|open|review|try|handle|continue|see|start)|"
            r"we\s+(?:need|should|may|can)|"
            r"user\s+(?:asks|asked|wants|requested|said)|"
            r"let\s+me\s+(?:start|check|inspect|fetch|run|use|look|read|verify|confirm|execute|search|open|review|try|handle|continue|see)|"
            r"let(?:'|’)s|likely|maybe"
            r")\b",
            compact,
        )) or (
            bool(re.search(r"(?i)\bneed\s+\S+", compact))
            and bool(re.search(
                r"(?i)\b(?:check|inspect|run|use|call|try|maybe|likely|let(?:'|’)s|tool|command)\b",
                compact,
            ))
        )

    @staticmethod
    def _mask_user_visible_text(text: str) -> str:
        """Mask secrets before content is streamed, persisted, or finalized."""
        try:
            from spoon_bot.utils.privacy import mask_secrets

            return mask_secrets(str(text or ""))
        except Exception:
            return str(text or "")

    @staticmethod
    def _strip_leaked_scratchpad_prefix(content: str) -> str:
        """Remove a short internal planning preamble from a final answer.

        Some OpenAI-compatible providers can surface scratchpad-style text as a
        normal content segment. This is output hygiene, not task routing: only
        an initial, mostly-ASCII planning note is removed, and only when a
        substantive answer remains.
        """
        text = str(content or "")
        if not text.strip():
            return text

        def _strip_once(value: str) -> str:
            for match in re.finditer(r"[\u4e00-\u9fff]", value[:800]):
                if match.start() <= 0:
                    continue
                prefix = value[:match.start()]
                if re.search(r"[\u4e00-\u9fff]", prefix):
                    continue
                suffix = value[match.start():]
                if suffix.strip() and AgentLoop._looks_like_internal_scratchpad_text(prefix):
                    return suffix.lstrip()

            sentence_match = re.match(
                r"^((?:[^\n.!?。！？]{1,240}[.!?。！？]\s*){1,3})(\S[\s\S]*)$",
                value,
            )
            if sentence_match:
                prefix, suffix = sentence_match.group(1), sentence_match.group(2)
                if suffix.strip() and AgentLoop._looks_like_internal_scratchpad_text(prefix):
                    return suffix.lstrip()
            return value

        for _ in range(4):
            stripped = _strip_once(text)
            if stripped == text:
                return text
            text = stripped
        return text

    def _finalize_response_content(
        self,
        message: str,
        content: str,
        *,
        turn_memory_start_index: int,
    ) -> str:
        """Apply generic execution-step filtering without prompt-derived dispatch."""
        if AgentLoop._looks_like_raw_tool_transcript_leak(content):
            return AgentLoop._mask_user_visible_text(
                AgentLoop._build_raw_tool_transcript_leak_response(
                    self,
                    turn_memory_start_index,
                )
            )
        filtered = AgentLoop._filter_execution_steps(self, content)
        cleaned = AgentLoop._strip_leaked_scratchpad_prefix(filtered)
        return AgentLoop._mask_user_visible_text(cleaned)
    @staticmethod
    def _stream_message_attr(message: Any, key: str, default: Any = None) -> Any:
        """Read a message field from either dict or object runtime payloads."""
        if isinstance(message, dict):
            return message.get(key, default)
        return getattr(message, key, default)

    @staticmethod
    def _stream_message_role(message: Any) -> str:
        """Normalize runtime message roles to plain strings."""
        role = AgentLoop._stream_message_attr(message, "role", "")
        return role.value if hasattr(role, "value") else str(role or "")

    @staticmethod
    def _stringify_stream_payload(payload: Any) -> str:
        """Serialize structured tool payloads for websocket metadata."""
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload
        if isinstance(payload, (dict, list)):
            try:
                return json.dumps(payload, ensure_ascii=False)
            except Exception:
                return str(payload)
        return str(payload)

    @staticmethod
    def _tool_call_arguments_key(arguments: Any) -> str:
        """Normalize tool-call arguments for captured-output matching."""
        return normalize_tool_arguments(arguments)

    @staticmethod
    def _cap_stream_metadata_text(value: Any, *, limit: int = 200_000) -> tuple[str, bool, int]:
        """Return a websocket-safe text payload plus truncation metadata."""
        text = AgentLoop._stringify_stream_payload(value)
        original_len = len(text)
        if original_len <= limit:
            return text, False, original_len
        suffix = f"\n... (stream output truncated, {original_len - limit} more chars)"
        return text[:limit] + suffix, True, original_len

    @staticmethod
    def _merge_stream_tool_result_metadata(
        metadata: dict[str, Any],
        *,
        streamed_result: str,
        captured_output: Any | None,
    ) -> dict[str, Any]:
        """Prefer captured full tool output while retaining model-visible summary."""
        merged = dict(metadata)
        summary_result = streamed_result
        full_result = streamed_result

        if captured_output is not None:
            captured_summary = getattr(captured_output, "summary_output", "") or summary_result
            captured_full = getattr(captured_output, "full_output", "") or captured_summary
            summary_result = captured_summary
            full_result = captured_full

        if full_result:
            stream_full_result, stream_truncated, stream_original_len = (
                AgentLoop._cap_stream_metadata_text(full_result)
            )
            merged["result"] = stream_full_result
            merged["content"] = stream_full_result
            merged["output"] = stream_full_result
            merged["full_result"] = stream_full_result
            merged["full_content"] = stream_full_result
            merged["full_output"] = stream_full_result
            if stream_truncated:
                merged["stream_output_truncated"] = True
                merged["stream_output_original_chars"] = stream_original_len
        if summary_result:
            merged.setdefault("model_result", summary_result)
            merged.setdefault("model_content", summary_result)
            merged.setdefault("model_output", summary_result)
        if full_result and summary_result and full_result != summary_result:
            merged["result_truncated_for_model"] = True
        return merged

    @staticmethod
    def _stream_tool_result_event_summary(event: dict[str, Any], *, limit: int = 2400) -> str:
        """Build a compact, user-visible summary from a structural tool result event."""
        metadata = dict(event.get("metadata") or {})
        tool_name = str(metadata.get("name") or metadata.get("tool") or "tool").strip() or "tool"
        payload = (
            metadata.get("model_result")
            or metadata.get("model_output")
            or metadata.get("model_content")
            or metadata.get("output")
            or metadata.get("result")
            or metadata.get("content")
            or metadata.get("full_output")
            or metadata.get("full_result")
            or event.get("delta")
        )
        text = AgentLoop._stringify_stream_payload(payload).strip()
        text = AgentLoop._mask_user_visible_text(text)
        if len(text) > limit:
            text = text[:limit].rstrip() + "\n... (tool output truncated)"
        if not text:
            text = "completed without text output"
        return f"Tool `{tool_name}` output:\n{text}"

    @staticmethod
    def _is_tool_loop_suppression_event(event: dict[str, Any]) -> bool:
        """Return True for tool guardrail results that should end the current loop."""
        metadata = dict(event.get("metadata") or {})
        payload = (
            metadata.get("model_result")
            or metadata.get("model_output")
            or metadata.get("model_content")
            or metadata.get("output")
            or metadata.get("result")
            or metadata.get("content")
            or metadata.get("full_output")
            or metadata.get("full_result")
            or event.get("delta")
        )
        text = AgentLoop._stringify_stream_payload(payload).lower()
        return "stop_tool_loop" in text

    @staticmethod
    def _extract_exact_command_failure_blocker(event: dict[str, Any]) -> str | None:
        """Extract a user-facing blocker from an exact-command STOP_TOOL_LOOP shell result."""
        metadata = dict(event.get("metadata") or {})
        tool_name = str(metadata.get("name") or metadata.get("tool") or "").strip().lower()
        if tool_name != "shell":
            return None

        payload = (
            metadata.get("model_result")
            or metadata.get("model_output")
            or metadata.get("model_content")
            or metadata.get("output")
            or metadata.get("result")
            or metadata.get("content")
            or metadata.get("full_output")
            or metadata.get("full_result")
            or event.get("delta")
        )
        text = AgentLoop._stringify_stream_payload(payload).strip()
        if "STOP_TOOL_LOOP: Exact requested shell command failed." not in text:
            return None

        cleaned = re.sub(
            r"^STOP_TOOL_LOOP: Exact requested shell command failed\.[^\n]*\n?",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        if cleaned:
            return AgentLoop._mask_user_visible_text(cleaned)
        return "The exact requested shell command failed before the task could continue."

    @staticmethod
    def _tool_loop_suppression_message(event: dict[str, Any]) -> str | None:
        """Return a user-facing message for an internal STOP_TOOL_LOOP guardrail."""
        metadata = dict(event.get("metadata") or {})
        payload = (
            metadata.get("model_result")
            or metadata.get("model_output")
            or metadata.get("model_content")
            or metadata.get("output")
            or metadata.get("result")
            or metadata.get("content")
            or metadata.get("full_output")
            or metadata.get("full_result")
            or event.get("delta")
        )
        text = AgentLoop._stringify_stream_payload(payload).lower()
        if "stop_tool_loop" not in text:
            return None
        if "exact requested shell command failed" in text:
            return None
        if "consecutive tool failures suppressed" in text:
            return "I stopped retrying after repeated failures."
        if "repeated side-effecting tool series suppressed" in text:
            return "I stopped before repeating the same external action."
        if "duplicate tool invocation suppressed" in text:
            return "I skipped a repeated action that had already run."
        return "I stopped before repeating the same action."

    @staticmethod
    def _extract_tool_suppression_user_response(
        tool_result_events: list[dict[str, Any]],
    ) -> str | None:
        """Convert STOP_TOOL_LOOP events into a clean final answer."""
        suppression_message: str | None = None
        for event in reversed(tool_result_events):
            message = AgentLoop._tool_loop_suppression_message(event)
            if message:
                suppression_message = message
                break
        if not suppression_message:
            return None

        for event in reversed(tool_result_events):
            if AgentLoop._is_tool_loop_suppression_event(event):
                continue
            summary = AgentLoop._stream_tool_result_event_summary(event)
            if summary:
                return (
                    f"{suppression_message}\n\n"
                    "Latest available result:\n\n"
                    f"{summary}\n\n"
                    "No additional duplicate action was run."
                )

        return f"{suppression_message} No new result was produced."

    @staticmethod
    def _build_tool_loop_fallback_response(
        tool_result_events: list[dict[str, Any]],
        *,
        reason: str,
    ) -> str:
        """Return a bounded final answer when a tool loop never yields final content."""
        if reason == "tool_suppression":
            for event in reversed(tool_result_events):
                exact_blocker = AgentLoop._extract_exact_command_failure_blocker(event)
                if exact_blocker:
                    return exact_blocker
            suppression_response = AgentLoop._extract_tool_suppression_user_response(
                tool_result_events
            )
            if suppression_response:
                return suppression_response

        if reason == "tool_followup_timeout":
            reason_text = "the agent kept running tools without producing a final answer"
        elif reason == "tool_suppression":
            reason_text = "a tool guardrail suppressed repeated work for the same action"
        else:
            reason_text = "the request reached the response time budget while tools were still active"
        parts = [
            f"The agent stopped the tool loop because {reason_text}.",
            "Recent tool evidence:",
        ]
        summaries = [
            AgentLoop._stream_tool_result_event_summary(event)
            for event in tool_result_events[-3:]
        ]
        if summaries:
            parts.extend(summaries)
        else:
            parts.append("No tool result text was captured before the stop condition.")
        if reason == "tool_suppression":
            parts.append("Stop here and report the latest result or blocker; do not retry the same tool action.")
        else:
            parts.append("Continue the task to let the agent proceed from this tool evidence.")
        return "\n\n".join(parts)

    def _get_runtime_memory_messages(self) -> list[Any]:
        """Return runtime memory messages exposed by the active inner agent."""
        if not hasattr(self._agent, "memory") or self._agent.memory is None:
            return []

        memory = self._agent.memory
        if hasattr(memory, "get_messages"):
            try:
                messages = memory.get_messages()
                if isinstance(messages, list):
                    return messages
            except Exception as exc:
                logger.debug(f"Failed to read runtime memory via get_messages(): {exc}")

        messages = getattr(memory, "messages", None)
        return messages if isinstance(messages, list) else []

    def _collect_stream_tool_result_events(
        self,
        start_index: int,
        emitted_tool_result_ids: set[str],
        *,
        tool_output_capture_scope: str | None,
        tool_call_arguments_by_id: dict[str, str],
    ) -> tuple[list[dict[str, Any]], int]:
        """Collect newly-added tool result messages from runtime memory."""
        messages = AgentLoop._get_runtime_memory_messages(self)
        if start_index < 0 or start_index > len(messages):
            start_index = 0

        events: list[dict[str, Any]] = []
        next_index = start_index
        for index, msg in enumerate(messages[start_index:], start=start_index):
            if AgentLoop._stream_message_role(msg) != "tool":
                next_index = index + 1
                continue

            tool_call_id = (
                AgentLoop._stream_message_attr(msg, "tool_call_id", "")
                or AgentLoop._stream_message_attr(msg, "id", "")
            )
            if tool_call_id and tool_call_id not in tool_call_arguments_by_id:
                break
            if tool_call_id and tool_call_id in emitted_tool_result_ids:
                next_index = index + 1
                continue

            result_payload = AgentLoop._stream_message_attr(msg, "text_content", None)
            if result_payload in (None, ""):
                result_payload = AgentLoop._stream_message_attr(msg, "content", "")
            serialized_result = AgentLoop._stringify_stream_payload(result_payload)
            tool_name = AgentLoop._stream_message_attr(msg, "name", "")
            captured_output = consume_captured_tool_output(
                tool_output_capture_scope,
                tool_name=tool_name,
                arguments=tool_call_arguments_by_id.get(tool_call_id, ""),
            )

            metadata: dict[str, Any] = {"name": tool_name}
            if tool_call_id:
                metadata["id"] = tool_call_id
                metadata["tool_call_id"] = tool_call_id
                emitted_tool_result_ids.add(tool_call_id)
            metadata = AgentLoop._merge_stream_tool_result_metadata(
                metadata,
                streamed_result=serialized_result,
                captured_output=captured_output,
            )

            events.append({
                "type": "tool_result",
                "delta": "",
                "metadata": metadata,
            })
            next_index = index + 1

        return events, next_index


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
        pseudo_tool_repair_attempted = False
        stream_completed = False
        stream_cancelled = False
        bg_task: asyncio.Task[None] | None = None
        original_system_prompt: str | None = None
        original_base_system_prompt: object = _MISSING
        tool_output_capture_scope: str | None = None
        capture_manager = None
        withheld_initial_content = False
        pending_fallback_content_emit = False
        pending_fallback_reason: str | None = None
        pending_fallback_delta = ""
        stream_error_reason: BaseException | str | None = None

        # Trim and inject persisted history into runtime memory
        await self._prepare_request_context(message)
        self._prepare_agent_for_new_turn()
        authoritative_message = message

        runtime_user_text = self._add_current_turn_skill_zip_context(
            AgentLoop._build_current_turn_runtime_user_text(authoritative_message)
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
        _pre_turn_memory_index = self._runtime_memory_snapshot_index()

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
            stream_tool_result_index = len(AgentLoop._get_runtime_memory_messages(self))
            emitted_tool_result_ids: set[str] = set()
            tool_call_arguments_by_id: dict[str, str] = {}
            recent_tool_result_events: list[dict[str, Any]] = []
            stream_tool_result_count = 0
            stream_tool_call_count = 0
            # 2. Start run() in background
            run_result_text = ""
            provider_silence_retry_count = 0
            max_provider_silence_retries = max(
                0,
                int(getattr(self, "provider_silence_retries", 1) or 0),
            )
            retry_reasoning_effort: str | None = None

            async def _run_and_signal() -> None:
                nonlocal run_result_text, stream_error_reason
                try:
                    retry_runner = AgentLoop._resolve_retry_runner(self)
                    run_kwargs: dict[str, Any] = {}
                    if thinking and self._callable_accepts_kwarg(self._agent.run, "thinking"):
                        run_kwargs["thinking"] = True
                    run_reasoning_effort = retry_reasoning_effort or effective_reasoning_effort
                    if (
                        run_reasoning_effort
                        and self._callable_accepts_kwarg(self._agent.run, "reasoning_effort")
                    ):
                        run_kwargs["reasoning_effort"] = run_reasoning_effort

                    def _drain_queue() -> None:
                        while not self._agent.output_queue.empty():
                            self._agent.output_queue.get_nowait()

                    request_execution_hints = self._build_request_execution_hints(authoritative_message)
                    with bind_request_execution_hints(request_execution_hints), track_tool_invocations():
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
            with bind_tool_owner(self._current_tool_owner_key()):
                bg_task = asyncio.create_task(_run_and_signal())

            # Force a yield to allow the background task to start
            await asyncio.sleep(0)

            # 3. Read output chunks (mirrors fixed BaseAgent.stream logic)
            oq = self._agent.output_queue
            td = self._agent.task_done
            logger.debug(f"output_queue type={type(oq).__name__}, task_done type={type(td).__name__}")
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

            logger.debug(f"Entering stream loop: td={td.is_set()}, qempty={oq.empty()}, qsize={oq.qsize()}")
            stream_started_at = time.monotonic()
            last_tool_progress_at = stream_started_at
            last_tool_progress_kind: str | None = None
            provider_silence_timeout = float(
                getattr(self, "provider_silence_timeout", DEFAULT_PROVIDER_SILENCE_TIMEOUT)
                or DEFAULT_PROVIDER_SILENCE_TIMEOUT
            )
            provider_total_timeout = float(
                getattr(self, "provider_total_timeout", DEFAULT_PROVIDER_TOTAL_TIMEOUT)
                or DEFAULT_PROVIDER_TOTAL_TIMEOUT
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
            shell_handoff_timeout = AgentLoop._float_env(
                "SPOON_BOT_SHELL_BACKGROUND_HANDOFF_TIMEOUT",
                DEFAULT_SHELL_BACKGROUND_HANDOFF_TIMEOUT,
            )
            active_tool_timeout = max(
                tool_followup_timeout,
                shell_foreground_timeout + shell_handoff_timeout + 5.0,
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

            def _record_tool_result_events(events: list[dict[str, Any]]) -> None:
                nonlocal last_tool_progress_at, last_tool_progress_kind, stream_tool_result_count
                if not events:
                    return
                last_tool_progress_at = time.monotonic()
                last_tool_progress_kind = "tool_result"
                stream_tool_result_count += len(events)
                recent_tool_result_events.extend(events)
                del recent_tool_result_events[:-6]

            def _stop_tool_loop(reason: str) -> None:
                nonlocal run_result_text, pre_tool_scratchpad_events, pre_tool_scratchpad_buffer
                nonlocal post_tool_content_events, post_tool_content_buffer
                if recent_tool_result_events:
                    run_result_text = AgentLoop._build_tool_loop_fallback_response(
                        recent_tool_result_events,
                        reason=reason,
                    )
                else:
                    run_result_text = (
                        "The task did not reach a final answer before the request "
                        "timeout. Please retry or continue this task."
                    )
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

            async def _run_pseudo_tool_call_repair(pseudo_content: str) -> tuple[str, list[dict[str, Any]]]:
                """Retry once when the model wrote fake tool calls as text."""
                nonlocal stream_tool_result_index

                AgentLoop._drop_pseudo_tool_call_assistant_messages(
                    self,
                    _pre_turn_memory_index,
                )
                AgentLoop._drain_agent_output_queue(self)
                self._reset_agent_state_for_retry()

                repair_prompt = AgentLoop._build_pseudo_tool_call_repair_prompt(
                    authoritative_message,
                    pseudo_content,
                )
                await self._agent.add_message("user", repair_prompt)
                self._agent.next_step_prompt = repair_prompt

                repair_kwargs: dict[str, Any] = {}
                if thinking and self._callable_accepts_kwarg(self._agent.run, "thinking"):
                    repair_kwargs["thinking"] = True
                repair_reasoning_effort = retry_reasoning_effort or effective_reasoning_effort
                if (
                    repair_reasoning_effort
                    and self._callable_accepts_kwarg(self._agent.run, "reasoning_effort")
                ):
                    repair_kwargs["reasoning_effort"] = repair_reasoning_effort

                request_execution_hints = self._build_request_execution_hints(authoritative_message)
                with bind_request_execution_hints(request_execution_hints), track_tool_invocations():
                    result = await AgentLoop._run_agent_with_context_overflow_recovery(
                        self,
                        label="stream_tool_call_repair",
                        retry_runner=AgentLoop._resolve_retry_runner(self),
                        pre_overflow_retry_cleanup=lambda: AgentLoop._drain_agent_output_queue(self),
                        **repair_kwargs,
                    )

                repair_text = AgentLoop._extract_run_result_text(result)
                repair_events: list[dict[str, Any]] = []
                queued_content_parts: list[str] = []

                while not self._agent.output_queue.empty():
                    try:
                        queued = self._agent.output_queue.get_nowait()
                    except Exception:
                        break

                    if isinstance(queued, dict) and queued.get("tool_calls"):
                        for tc in queued.get("tool_calls") or []:
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
                            if tc_id:
                                tool_call_arguments_by_id[tc_id] = AgentLoop._tool_call_arguments_key(fn_args)
                            repair_events.append({
                                "type": "tool_call",
                                "delta": "",
                                "metadata": {
                                    "id": tc_id,
                                    "name": fn_name,
                                    "arguments": fn_args,
                                    "repair": "pseudo_tool_call_text",
                                },
                            })
                        continue

                    if isinstance(queued, dict) and queued.get("type") == "tool_result":
                        metadata = dict(queued.get("metadata") or {})
                        tool_name = queued.get("name") or metadata.get("name") or ""
                        tool_result = (
                            queued.get("result")
                            or queued.get("content")
                            or queued.get("response")
                            or queued.get("output")
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
                            emitted_tool_result_ids.add(tool_call_id)
                        metadata["repair"] = "pseudo_tool_call_text"
                        metadata = AgentLoop._merge_stream_tool_result_metadata(
                            metadata,
                            streamed_result=serialized_result,
                            captured_output=captured_output,
                        )
                        repair_events.append({
                            "type": "tool_result",
                            "delta": "",
                            "metadata": metadata,
                        })
                        continue

                    if isinstance(queued, dict):
                        text = queued.get("content") or queued.get("delta") or ""
                    else:
                        text = getattr(queued, "content", None) or (
                            queued if isinstance(queued, str) else ""
                        )
                    if text:
                        queued_content_parts.append(str(text))

                if not repair_text.strip() and queued_content_parts:
                    repair_text = "".join(queued_content_parts)

                tool_result_events, stream_tool_result_index = (
                    AgentLoop._collect_stream_tool_result_events(
                        self,
                        stream_tool_result_index,
                        emitted_tool_result_ids,
                        tool_output_capture_scope=tool_output_capture_scope,
                        tool_call_arguments_by_id=tool_call_arguments_by_id,
                    )
                )
                for event in tool_result_events:
                    metadata = dict(event.get("metadata") or {})
                    metadata["repair"] = "pseudo_tool_call_text"
                    repair_events.append({**event, "metadata": metadata})

                return repair_text, repair_events

            def _active_tool_within_budget(now: float | None = None) -> bool:
                if not saw_tool_call or last_tool_progress_kind != "tool_call":
                    return False
                now = time.monotonic() if now is None else now
                return now - last_tool_progress_at < active_tool_timeout

            while not (td.is_set() and oq.empty()):
                if _stop_if_total_timeout():
                    break

                for event in _drain_runtime_notice_events():
                    yield event

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
                for event in tool_result_events:
                    yield _decorate_stream_event(event)
                if any(AgentLoop._is_tool_loop_suppression_event(event) for event in tool_result_events):
                    logger.warning("Stopping tool loop after repeated-tool guardrail result.")
                    _stop_tool_loop("tool_suppression")
                    break
                if _stop_if_total_timeout():
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
                    queue_poll_timeout = min(
                        2.0,
                        max(0.05, provider_silence_timeout),
                    )
                    chunk = await asyncio.wait_for(oq.get(), timeout=queue_poll_timeout)
                    chunk_count += 1
                    logger.debug(f"Got chunk #{chunk_count}: type={type(chunk).__name__}, repr={repr(chunk)[:200]}")
                except asyncio.TimeoutError:
                    if (
                        not td.is_set()
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
                            with bind_tool_owner(self._current_tool_owner_key()):
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
                                chunk.get("metadata", {}).get("error")
                                if isinstance(chunk.get("metadata"), dict)
                                else None
                            ) or chunk.get("delta") or "stream_error"
                            stream_error_reason = RuntimeError(str(error_text))
                        yield {
                            "type": "error",
                            "delta": chunk.get("delta", ""),
                            "metadata": chunk.get("metadata", {}),
                            "source": current_source,
                        }
                        continue

                    if "tool_calls" in chunk and chunk["tool_calls"]:
                        saw_tool_call = True
                        stream_tool_call_count += len(chunk["tool_calls"])
                        last_tool_progress_at = time.monotonic()
                        last_tool_progress_kind = "tool_call"
                        # Initial content before a tool call is tool preamble by
                        # structure; do not classify it with language phrases.
                        if pre_tool_scratchpad_events:
                            pre_tool_scratchpad_events = []
                            pre_tool_scratchpad_buffer = ""
                        if post_tool_content_buffer:
                            post_tool_content_events = []
                            post_tool_content_buffer = ""
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
                            if tc_id:
                                tool_call_arguments_by_id[tc_id] = AgentLoop._tool_call_arguments_key(fn_args)
                            yield _decorate_stream_event({
                                "type": "tool_call",
                                "delta": "",
                                "metadata": {
                                    "id": tc_id,
                                    "name": fn_name,
                                    "arguments": fn_args,
                                },
                            })
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
                        tool_name = (
                            chunk.get("name")
                            or metadata.get("name")
                            or ""
                        )
                        tool_result = (
                            chunk.get("result")
                            or chunk.get("content")
                            or chunk.get("response")
                            or chunk.get("output")
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
                            emitted_tool_result_ids.add(tool_call_id)
                        metadata = AgentLoop._merge_stream_tool_result_metadata(
                            metadata,
                            streamed_result=serialized_result,
                            captured_output=captured_output,
                        )
                        _record_tool_result_events([{
                            "type": "tool_result",
                            "delta": delta,
                            "metadata": metadata,
                        }])
                    elif dict_type == "content":
                        chunk_type = "content"
                    # Support both "content" and "delta" keys (#10)
                    text = chunk.get("content") or chunk.get("delta") or ""
                    if text:
                        delta = text

                # -- Object chunks with content --
                elif hasattr(chunk, "content") and chunk.content:
                    delta = chunk.content

                # -- Plain string chunks --
                elif isinstance(chunk, str):
                    delta = chunk

                if chunk_type == "tool_result":
                    tool_result_event = {
                        "type": chunk_type,
                        "delta": delta,
                        "metadata": metadata,
                    }
                    yield _decorate_stream_event(tool_result_event)
                    if AgentLoop._is_tool_loop_suppression_event(tool_result_event):
                        logger.warning("Stopping tool loop after repeated-tool guardrail result.")
                        _stop_tool_loop("tool_suppression")
                        break
                    if _stop_if_total_timeout():
                        break
                    continue

                if chunk_type == "thinking":
                    if thinking and delta:
                        yield _decorate_stream_event({
                            "type": "thinking",
                            "delta": delta,
                            "metadata": metadata,
                        })
                        if _stop_if_total_timeout():
                            break
                    continue

                if delta:
                    metadata_phase = metadata.get("phase") if isinstance(metadata, dict) else None
                    explicit_pre_tool_phase = (
                        chunk_type == "content"
                        and thinking
                        and metadata_phase == "think"
                    )
                    if chunk_type == "content" and explicit_pre_tool_phase:
                        yield _decorate_stream_event({
                            "type": "thinking",
                            "delta": delta,
                            "metadata": {
                                **metadata,
                                "source": metadata.get("source", "phase_think"),
                            },
                        })
                        if _stop_if_total_timeout():
                            break
                    else:
                        event = {"type": chunk_type, "delta": delta, "metadata": metadata}
                        if chunk_type == "content":
                            if thinking and not saw_tool_call:
                                pre_tool_scratchpad_buffer += delta
                                pre_tool_scratchpad_events.append(event)
                                continue
                            if not saw_tool_call:
                                delta = AgentLoop._mask_user_visible_text(delta)
                                if not delta:
                                    continue
                                full_content += delta
                            else:
                                post_tool_content_buffer += delta
                                post_tool_content_events.append(event)
                                continue
                            if buffer_stream_content:
                                continue
                            event["delta"] = delta
                        yield _decorate_stream_event(event)
                        if _stop_if_total_timeout():
                            break

            logger.debug(f"Stream loop exited: td={td.is_set()}, qempty={oq.empty()}, chunks_received={chunk_count}, full_content_len={len(full_content)}")

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
            for event in tool_result_events:
                yield _decorate_stream_event(event)
            for event in _drain_runtime_notice_events():
                yield event

            if pre_tool_scratchpad_buffer and not saw_tool_call:
                full_content += AgentLoop._mask_user_visible_text(pre_tool_scratchpad_buffer)
                withheld_initial_content = True
                pre_tool_scratchpad_events = []
                pre_tool_scratchpad_buffer = ""

            if post_tool_content_buffer:
                pending_content = AgentLoop._strip_leaked_scratchpad_prefix(
                    post_tool_content_buffer
                )
                post_tool_content_events = []
                post_tool_content_buffer = ""
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
                    repair_text, repair_events = await _run_pseudo_tool_call_repair(
                        pending_content
                    )
                    for event in repair_events:
                        if event.get("type") == "tool_result":
                            _record_tool_result_events([event])
                        yield _decorate_stream_event(event)
                    run_result_text = repair_text
                    pending_content = ""
                if (
                    pending_content.strip()
                    and not AgentLoop._looks_like_internal_scratchpad_text(pending_content)
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
                    pending_content = AgentLoop._mask_user_visible_text(pending_content)
                    saw_content_after_tool_call = True
                    full_content += pending_content
                    if not buffer_stream_content:
                        yield _decorate_stream_event({
                            "type": "content",
                            "delta": pending_content,
                            "metadata": {"validated": True},
                        })

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
            if run_result_text and (not full_content or fallback_after_tool_only_preamble):
                logger.warning(
                    "Stream produced no content chunks; "
                    "falling back to run() result text."
                )
                fallback_delta = run_result_text
                if full_content and fallback_after_tool_only_preamble:
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
                repair_text, repair_events = await _run_pseudo_tool_call_repair(full_content)
                for event in repair_events:
                    if event.get("type") == "tool_result":
                        _record_tool_result_events([event])
                    yield _decorate_stream_event(event)
                full_content = repair_text
                pending_fallback_content_emit = True
                pending_fallback_reason = "pseudo_tool_call_repair"
                pending_fallback_delta = repair_text

            pre_finalize_full_content = full_content
            final_content = AgentLoop._finalize_response_content(
                self,
                authoritative_message,
                full_content,
                turn_memory_start_index=_pre_turn_memory_index,
            )
            if (
                (
                    buffer_stream_content
                    or withheld_initial_content
                    or pending_fallback_content_emit
                )
                and final_content
            ):
                emit_delta = final_content
                if (
                    pending_fallback_content_emit
                    and self._normalize_comparable_text(final_content)
                    == self._normalize_comparable_text(pre_finalize_full_content)
                ):
                    emit_delta = pending_fallback_delta or final_content
                emit_metadata = {
                    "buffered": bool(buffer_stream_content),
                    "withheld_initial_content": bool(withheld_initial_content),
                    "validated": True,
                }
                if pending_fallback_reason:
                    emit_metadata["fallback"] = pending_fallback_reason
                yield _decorate_stream_event({
                    "type": "content",
                    "delta": emit_delta,
                    "metadata": emit_metadata,
                })
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
            AgentLoop._mark_latest_user_turn_state(
                self,
                _TURN_STATE_INTERRUPTED,
                reason="task_cancelled",
            )
            raise
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            stream_error_reason = e
            stream_completed = True
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
        finally:
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
                AgentLoop._mark_latest_user_turn_state(
                    self,
                    _TURN_STATE_INTERRUPTED,
                    reason="task_cancelled",
                )

        if stream_error_reason is not None and not full_content and not stream_cancelled:
            AgentLoop._persist_failed_turn_context(
                self,
                label="stream",
                reason=stream_error_reason,
                start_index=_pre_turn_memory_index,
            )

        if full_content and stream_completed and not stream_cancelled:
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
            AgentLoop._build_current_turn_runtime_user_text(authoritative_message)
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
        _pre_turn_memory_index = self._runtime_memory_snapshot_index()

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

            # Run agent with thinking enabled
            run_kwargs: dict[str, Any] = {}
            requested_thinking = thinking_level or True
            if self._callable_accepts_kwarg(self._agent.run, "thinking"):
                run_kwargs["thinking"] = requested_thinking
            if (
                effective_reasoning_effort
                and self._callable_accepts_kwarg(self._agent.run, "reasoning_effort")
            ):
                run_kwargs["reasoning_effort"] = effective_reasoning_effort
            request_execution_hints = self._build_request_execution_hints(authoritative_message)
            with bind_request_execution_hints(request_execution_hints), track_tool_invocations():
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
                thinking_content = (
                    result.metadata.get("thinking")
                    or result.metadata.get("reasoning")
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
                with bind_request_execution_hints(request_execution_hints), track_tool_invocations():
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
                    thinking_content = (
                        result.metadata.get("thinking")
                        or result.metadata.get("reasoning")
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
            AgentLoop._mark_latest_user_turn_state(
                self,
                _TURN_STATE_INTERRUPTED,
                reason="task_cancelled",
            )
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

    def _workspace_posix_path(self) -> str:
        """Return the workspace path in POSIX form for shell commands."""
        import re as _re
        import sys
        raw = str(self.workspace).replace("\\", "/")
        if sys.platform == "win32":
            raw = _re.sub(r'^([A-Za-z]):', lambda m: f'/{m.group(1).lower()}', raw)
        return raw

    # Directories in the workspace root that are never skills
    _WORKSPACE_INFRA_DIRS = frozenset({
        "skills", "logs", "sessions", "memory",
        "channels", "wallet", ".git",
    })

    # ------------------------------------------------------------------
    # Conditional activation helpers
    # ------------------------------------------------------------------

    def record_touched_paths(self, *paths: str | Path) -> None:
        """Register file paths the agent has interacted with.

        Called by file-oriented tools (read_file, write_file, edit_file, etc.)
        so that path-conditional skills can be activated dynamically.

        Also performs **dynamic discovery**: walks up from the touched path
        looking for directories containing ``SKILL.md`` and adds them to
        ``_skill_paths`` if not already known (inspired by Claude Code's
        ``discoverSkillDirsForPaths``).
        """
        for p in paths:
            resolved = Path(p)
            if not resolved.is_absolute():
                resolved = self.workspace / resolved
            try:
                resolved = resolved.resolve()
            except OSError:
                pass

            try:
                rel = str(resolved.relative_to(self.workspace))
            except (ValueError, OSError):
                rel = str(p)
            self._touched_paths.add(rel.replace("\\", "/"))

            self._discover_skills_near(resolved)

    def _discover_skills_near(self, file_path: Path) -> None:
        """Walk up from *file_path* looking for ``skills/`` directories.

        When a directory containing ``SKILL.md`` files is found and is not
        already in ``_skill_paths``, it is added dynamically.  This mirrors
        Claude Code's ``discoverSkillDirsForPaths`` which walks up from every
        file operation to find project-level ``.claude/skills/`` directories.

        Only walks up to ``self.workspace`` (never above).
        """
        known = {str(Path(sp).resolve()) for sp in self._skill_paths}
        current = file_path.parent if file_path.is_file() else file_path

        try:
            ws_resolved = self.workspace.resolve()
        except OSError:
            return

        while True:
            try:
                if not current.is_relative_to(ws_resolved):
                    break
            except (TypeError, ValueError):
                break

            skills_dir = current / "skills"
            if skills_dir.is_dir() and str(skills_dir.resolve()) not in known:
                has_skill = any(
                    (child / "SKILL.md").exists()
                    for child in skills_dir.iterdir()
                    if child.is_dir()
                )
                if has_skill:
                    self._skill_paths.append(skills_dir)
                    known.add(str(skills_dir.resolve()))
                    logger.info(f"Dynamic skill discovery: found {skills_dir}")

            if current == ws_resolved:
                break
            current = current.parent

    @staticmethod
    def _skill_paths_match(
        patterns: list[str], touched: set[str], workspace: Path | None = None,
    ) -> bool:
        """Return True if any *touched* file matches at least one *pattern*.

        Patterns use gitignore-style globs (``fnmatch``).  A leading ``!``
        negates (exclude) the pattern - same semantics as ``.gitignore``.
        An empty *patterns* list means "always active" (unconditional).
        """
        from fnmatch import fnmatch

        if not patterns:
            return True
        if not touched:
            return False

        for fp in touched:
            included = False
            for pat in patterns:
                negate = pat.startswith("!")
                glob = pat.lstrip("!")
                if fnmatch(fp, glob) or fnmatch(fp, f"**/{glob}"):
                    included = not negate
            if included:
                return True
        return False

    @staticmethod
    def _parse_skill_frontmatter(skill_md: Path) -> dict[str, str | list[str]]:
        """Extract description, when_to_use, triggers, and paths from SKILL.md YAML frontmatter.

        The ``paths`` field (list of gitignore-style glob patterns) enables
        conditional activation: skills declaring ``paths`` are only active when
        recently-touched files match at least one pattern.  Skills without
        ``paths`` are unconditionally active.
        """
        import re as _re

        result: dict[str, str | list[str]] = {
            "description": "", "when_to_use": "", "triggers": "", "paths": [],
        }
        try:
            raw = skill_md.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return result

        fm = _re.match(r'^---\s*\n(.*?)\n---', raw, _re.DOTALL)
        if not fm:
            for line in raw.split("\n")[1:20]:
                stripped = line.strip()
                if stripped and not stripped.startswith(("#", "---", "```")):
                    result["description"] = stripped[:300]
                    break
            return result

        fm_text = fm.group(1)
        in_multiline = ""
        in_paths_list = False
        trigger_fragments: list[str] = []
        paths_list: list[str] = []

        for line in fm_text.split("\n"):
            stripped = line.strip()

            if in_paths_list:
                if stripped.startswith("- "):
                    paths_list.append(stripped[2:].strip().strip("'\""))
                    continue
                elif line.startswith("  ") or line.startswith("\t"):
                    if stripped.startswith("- "):
                        paths_list.append(stripped[2:].strip().strip("'\""))
                    continue
                else:
                    in_paths_list = False

            if in_multiline:
                if line.startswith("  ") or line.startswith("\t"):
                    if in_multiline == "description":
                        result["description"] += " " + stripped
                    elif in_multiline == "when_to_use":
                        result["when_to_use"] += " " + stripped
                    continue
                else:
                    in_multiline = ""

            if stripped.startswith("description:"):
                val = stripped.split(":", 1)[1].strip().strip("'\"")
                if val and val not in (">", "|"):
                    result["description"] = val
                elif val in (">", "|"):
                    in_multiline = "description"

            elif stripped.startswith(("when_to_use:", "whenToUse:")):
                val = stripped.split(":", 1)[1].strip().strip("'\"")
                if val and val not in (">", "|"):
                    result["when_to_use"] = val
                elif val in (">", "|"):
                    in_multiline = "when_to_use"

            elif stripped.startswith("paths:"):
                inline = stripped.split(":", 1)[1].strip()
                if inline.startswith("[") and inline.endswith("]"):
                    for p in inline[1:-1].split(","):
                        p = p.strip().strip("'\"")
                        if p:
                            paths_list.append(p)
                else:
                    in_paths_list = True

            elif "triggers" in stripped.lower() or "trigger" in stripped.lower():
                trigger_fragments.extend(_re.findall(r'"([^"]+)"', stripped))

        if not result["description"]:
            for line in raw.split("\n")[1:20]:
                stripped = line.strip()
                if stripped and not stripped.startswith(("#", "---", "```")):
                    result["description"] = stripped[:300]
                    break

        result["description"] = str(result["description"]).strip()[:300]
        result["when_to_use"] = str(result["when_to_use"]).strip()[:200]
        result["triggers"] = "|".join(trigger_fragments)
        result["paths"] = paths_list
        return result

    def _iter_skill_candidates(
        self, *, include_dormant: bool = False,
    ) -> list[tuple[str, Path, Path, bool]]:
        """Return (name, dir, skill_md_path, is_organized) for all discoverable skills.

        Scans in priority order:
        1. ``workspace/skills/`` - primary organized location
        2. All entries in ``_skill_paths`` (includes bundled/dev skills)
        3. Workspace root - unorganized skills the user may have dropped in

        Deduplicates by both name AND resolved (realpath) canonical path so
        symlinks pointing to the same physical directory are counted once.
        Earlier entries take priority (matches Claude Code's first-wins logic).

        **Conditional activation (``paths`` frontmatter)**:
        Skills declaring ``paths`` patterns are dormant until a touched file
        matches.  Set *include_dormant* to ``True`` to include them anyway
        (useful for listing all skills in the prompt with a "dormant" tag).
        """
        skills_dir = self.workspace / "skills"

        # (parent_dir, is_organized)
        scan_dirs: list[tuple[Path, bool]] = []
        if skills_dir.is_dir():
            scan_dirs.append((skills_dir, True))
        for sp in getattr(self, "_skill_paths", []):
            resolved = Path(sp).resolve()
            if resolved.is_dir() and resolved != skills_dir.resolve():
                scan_dirs.append((resolved, True))
        if self.workspace.is_dir():
            scan_dirs.append((self.workspace, False))

        candidates: list[tuple[str, Path, Path, bool]] = []
        seen_names: set[str] = set()
        seen_realpaths: set[str] = set()

        for parent_dir, is_organized in scan_dirs:
            for child in sorted(parent_dir.iterdir()):
                if not (child.is_dir() or child.is_symlink()):
                    continue
                if not is_organized and child.name in self._WORKSPACE_INFRA_DIRS:
                    continue
                skill_md = child / "SKILL.md"
                if not skill_md.exists():
                    continue
                name = child.name
                if name in seen_names:
                    continue

                try:
                    canonical = str(skill_md.resolve())
                except OSError:
                    canonical = str(skill_md)
                if canonical in seen_realpaths:
                    logger.debug(
                        f"Skipping duplicate skill '{name}' "
                        f"(same file already loaded via another path)"
                    )
                    continue

                if not include_dormant:
                    fm = self._parse_skill_frontmatter(skill_md)
                    skill_paths = fm.get("paths", [])
                    if isinstance(skill_paths, list) and skill_paths:
                        if not self._skill_paths_match(
                            skill_paths, self._touched_paths, self.workspace,
                        ):
                            logger.debug(
                                f"Skill '{name}' dormant (paths not matched)"
                            )
                            continue

                seen_names.add(name)
                seen_realpaths.add(canonical)
                candidates.append((name, child, skill_md, is_organized))

        return candidates

    def _build_skill_context(
        self,
        skill_name: str,
        skill_dir: Path,
        *,
        is_organized: bool,
    ) -> dict[str, Any]:
        """Build sanitized skill metadata suitable for session persistence."""
        import re as _re
        import sys as _sys

        base_dir = str(skill_dir).replace("\\", "/")
        if _sys.platform == "win32":
            base_dir = _re.sub(r"^([A-Za-z]):", lambda m: f"/{m.group(1).lower()}", base_dir)

        skill_rel = f"skills/{skill_name}" if is_organized else skill_name
        return {
            "name": skill_name,
            "base_dir": base_dir,
            "workspace_relative_path": f"{skill_rel}/",
            "location": f"{skill_rel}/SKILL.md",
            "organized": bool(is_organized),
        }

    def _resolve_skill_context_by_name(self, skill_name: str) -> dict[str, Any] | None:
        """Resolve a persisted skill name against the current skill catalog."""
        try:
            candidates = self._iter_skill_candidates(include_dormant=True)
        except Exception:
            return None

        for name, skill_dir, _skill_md, is_organized in candidates:
            if name == skill_name:
                return self._build_skill_context(
                    name,
                    skill_dir,
                    is_organized=is_organized,
                )
        return None

    def _build_step_prompt(self, message: str) -> str:
        """Build a minimal per-step prompt from the user's request.

        Keeps only the user's original request and workspace path.
        Injects env vars so they survive short-term memory pruning.
        """
        _truncated = self._truncate_request_for_prompt(message)
        _ws = self._workspace_posix_path()
        prompt = (
            "[TURN PRIORITY]: Execute only the newest user request. "
            "Any unfinished plan, stale tool sequence, or previous task assumption is superseded "
            "unless the newest user message explicitly says to continue it.\n"
            "[HISTORY BOUNDARY]: Prior conversation is reference only. Do not run prior tasks, "
            "do not append prior-task work, and stop as soon as the newest request is satisfied.\n"
            f"{_EXTERNAL_SIDE_EFFECT_BOUNDARY}"
            f"{format_current_datetime_context(bracketed=True)}\n"
            f"[USER REQUEST]: {_truncated}\n"
            f"[WORKSPACE]: {_ws}/\n\n"
            + self.DEFAULT_NEXT_STEP_PROMPT
        )
        skill_zip_context = self._current_turn_skill_zip_context()
        if skill_zip_context:
            prompt = f"{skill_zip_context}\n{prompt}"
        recent_turn_notice = getattr(self, "_recent_turn_notice", None)
        if isinstance(recent_turn_notice, str) and recent_turn_notice.strip():
            prompt = (
                f"[PREVIOUS TURN STATUS]: {recent_turn_notice.strip()}\n"
                + prompt
            )
        env_section = self._extract_env_for_prompt()
        if env_section:
            prompt += env_section
        return prompt

    def _build_request_context_prompt(self, message: str) -> str:
        """Build the compact request context block used for thinking runs."""
        _truncated = self._truncate_request_for_prompt(message)
        _ws = self._workspace_posix_path()
        prompt = (
            "[TURN PRIORITY]: Execute only the newest user request. "
            "Any unfinished plan, stale tool sequence, or previous task assumption is superseded "
            "unless the newest user message explicitly says to continue it.\n"
            "[HISTORY BOUNDARY]: Prior conversation is reference only. Do not run prior tasks, "
            "do not append prior-task work, and stop as soon as the newest request is satisfied.\n"
            f"{_EXTERNAL_SIDE_EFFECT_BOUNDARY}"
            f"{format_current_datetime_context(bracketed=True)}\n"
            f"[USER REQUEST]: {_truncated}\n"
            f"[WORKSPACE]: {_ws}/\n"
        )
        skill_zip_context = self._current_turn_skill_zip_context()
        if skill_zip_context:
            prompt = f"{skill_zip_context}\n{prompt}"
        recent_turn_notice = getattr(self, "_recent_turn_notice", None)
        if isinstance(recent_turn_notice, str) and recent_turn_notice.strip():
            prompt = (
                f"[PREVIOUS TURN STATUS]: {recent_turn_notice.strip()}\n"
                + prompt
            )
        session_recall = self._build_session_recall_context(message)
        if isinstance(session_recall, str) and session_recall.strip():
            prompt = f"{session_recall.strip()}\n" + prompt
        env_section = self._extract_env_for_prompt()
        if env_section:
            prompt += env_section
        return prompt

    @staticmethod
    def _tokenize_request_matching_text(text: str) -> set[str]:
        """Return normalized keyword tokens for lightweight skill matching."""
        return {
            token
            for token in re.findall(r"[a-z0-9][a-z0-9_-]{1,}", str(text or "").lower())
            if len(token) >= 2
        }

    @staticmethod
    def _request_explicitly_needs_remote_lookup(message: str) -> bool:
        """Return True when the latest request clearly asks for web/API lookup."""
        text = str(message or "").lower()
        if "http://" in text or "https://" in text:
            return True
        return bool(re.search(
            r"(?i)\b(web_fetch|web_search|curl|http|https|api|docs?|documentation|"
            r"search|browse|lookup|look up|fetch|官网|文档|接口|网页|搜索)\b",
            text,
        ))

    @staticmethod
    def _extract_exact_shell_commands_from_request(message: str) -> list[str]:
        """Extract explicit shell commands quoted in the latest user request."""
        commands: list[str] = []
        seen: set[str] = set()
        for match in re.findall(r"`([^`\n]{3,300})`", str(message or "")):
            candidate = " ".join(match.strip().split())
            if not candidate:
                continue
            if not re.match(r"(?i)^(node|python|uv|bash|sh|curl|cast|git|npm|pnpm|yarn)\b", candidate):
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            commands.append(candidate)
        return commands[:4]

    @staticmethod
    def _extract_skill_command_hints(skill_md: Path) -> tuple[list[str], list[str]]:
        """Extract runnable command hints and referenced URLs from a SKILL.md file."""
        try:
            content = skill_md.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return [], []

        commands: list[str] = []
        seen_commands: set[str] = set()

        cli_match = re.search(r"CLI\s*:?=\s*(.+)", content)
        cli_value = cli_match.group(1).strip() if cli_match else ""

        def _push_command(command: str) -> None:
            normalized = " ".join(str(command or "").strip().split())
            if not normalized or normalized in seen_commands:
                return
            seen_commands.add(normalized)
            commands.append(normalized)

        if cli_value:
            _push_command(cli_value)

        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("$CLI "):
                expanded = stripped.replace("$CLI", cli_value or "$CLI", 1)
                _push_command(expanded)
                continue
            if stripped.startswith("node ") or stripped.startswith("python ") or stripped.startswith("uv "):
                _push_command(stripped)

        urls = list(dict.fromkeys(
            match.rstrip(").,")
            for match in re.findall(r"https?://[^\s`<>\"]+", content)
        ))
        return commands[:8], urls[:12]

    def _build_request_execution_hints(self, message: str) -> dict[str, Any]:
        """Build request-scoped generic execution hints for tool-side guardrails."""
        message_text = str(message or "")
        request_tokens = AgentLoop._tokenize_request_matching_text(message_text)
        local_executable_skills: list[dict[str, Any]] = []

        for name, _skill_dir, skill_md, is_organized in self._iter_skill_candidates(include_dormant=True):
            fm = self._parse_skill_frontmatter(skill_md)
            name_tokens = AgentLoop._tokenize_request_matching_text(name.replace("-", " "))
            desc_tokens = AgentLoop._tokenize_request_matching_text(
                " ".join(
                    part for part in (
                        fm.get("description") or "",
                        fm.get("when_to_use") or "",
                    ) if part
                )
            )
            overlap = len(request_tokens & (name_tokens | desc_tokens))
            direct_name_match = name.lower() in message_text.lower()
            if not direct_name_match and overlap < 2:
                continue

            commands, urls = AgentLoop._extract_skill_command_hints(skill_md)
            if not commands:
                continue

            skill_rel = f"skills/{name}/SKILL.md" if is_organized else f"{name}/SKILL.md"
            local_executable_skills.append({
                "name": name,
                "location": skill_rel,
                "commands": commands,
                "urls": urls,
                "score": (100 if direct_name_match else 0) + overlap,
            })

        local_executable_skills.sort(key=lambda item: int(item.get("score") or 0), reverse=True)
        return {
            "allow_remote_probe": AgentLoop._request_explicitly_needs_remote_lookup(message_text),
            "exact_shell_commands": AgentLoop._extract_exact_shell_commands_from_request(message_text),
            "local_executable_skills": local_executable_skills[:3],
        }

    @staticmethod
    def _truncate_request_for_prompt(
        message: str,
        *,
        head_chars: int = 220,
        tail_chars: int = 180,
    ) -> str:
        """Keep both the head and tail of the latest request for prompt scaffolding."""
        normalized = (message or "").strip()
        if len(normalized) <= head_chars + tail_chars + 24:
            return normalized
        head = normalized[:head_chars].rstrip()
        tail = normalized[-tail_chars:].lstrip()
        return (
            f"{head}\n"
            "[... middle omitted to save tokens; preserve latest tail instructions ...]\n"
            f"{tail}"
        )

    def _apply_request_context_to_system_prompt(
        self,
        message: str,
        *,
        thinking: bool,
    ) -> tuple[str | None, object]:
        """Temporarily append active request context to the agent system prompt."""
        if not getattr(self, "_agent", None):
            return None, _MISSING

        current_prompt = getattr(self._agent, "system_prompt", None)
        if not isinstance(current_prompt, str) or not current_prompt:
            return None, _MISSING

        request_context = self._build_request_context_prompt(message)
        augmented_prompt = (
            f"{current_prompt}\n\n"
            f"## Active Request Context\n"
            f"{request_context}"
        )
        self._agent.system_prompt = augmented_prompt

        original_base_prompt = _MISSING
        if hasattr(self._agent, "_original_system_prompt"):
            original_base_prompt = getattr(self._agent, "_original_system_prompt")
            if isinstance(original_base_prompt, str) and original_base_prompt:
                self._agent._original_system_prompt = (
                    f"{original_base_prompt}\n\n"
                    f"## Active Request Context\n"
                    f"{request_context}"
                )

        return current_prompt, original_base_prompt

    def _restore_request_context_system_prompt(
        self,
        original_prompt: str | None,
        original_base_prompt: object,
    ) -> None:
        """Restore the agent system prompt after a thinking run completes."""
        if not getattr(self, "_agent", None):
            return
        if original_prompt is not None:
            self._agent.system_prompt = original_prompt
        if original_base_prompt is not _MISSING and hasattr(self._agent, "_original_system_prompt"):
            self._agent._original_system_prompt = original_base_prompt

    def _select_next_step_prompt(self, message: str, *, thinking: bool) -> str:
        """Choose the per-step prompt shape for the current request.

        Thinking runs keep a lightweight per-step prompt across all providers.
        The active request context is injected into the system prompt instead,
        so runtime pruning still preserves user/workspace/env guidance without
        reintroducing the heavier synthetic user-turn prompt that suppresses
        streamed reasoning on some providers.
        """
        if thinking:
            return self.DEFAULT_NEXT_STEP_PROMPT
        return self._build_step_prompt(message)

    def _extract_env_for_prompt(self) -> str:
        """Extract env vars from .env.local for the step prompt.

        Non-sensitive values shown directly. Private keys masked -
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

        parts = ["\n[ENV - from .env.local - do NOT re-read, use `source .env.local` for secrets]:"]
        for k, v in env_vars.items():
            parts.append(f"  {k}={v}")
        return "\n".join(parts) + "\n"

    # ------------------------------------------------------------------
    # Dynamic prompt helpers
    # ------------------------------------------------------------------

    def _build_skills_for_prompt(self) -> str:
        """Build Openclaw-style XML metadata for installed skills.

        Uses ``_iter_skill_candidates`` and ``_parse_skill_frontmatter`` to
        build an ``<available_skills>`` XML block.  Unorganized skills (in the
        workspace root) are flagged so the agent knows to move them first.

        Path-conditional skills are included with ``include_dormant=True``
        so the agent is *aware* of them, but they carry a ``<status>dormant``
        tag indicating they will activate when matching files are touched.
        """
        candidates = self._iter_skill_candidates(include_dormant=True)
        if not candidates:
            return ""

        entries: list[str] = []
        for name, _dir, skill_md, is_organized in candidates:
            fm = self._parse_skill_frontmatter(skill_md)
            description = fm["description"] or name
            when_to_use = fm["when_to_use"]
            skill_paths = fm.get("paths", [])

            location = f"skills/{name}/SKILL.md" if is_organized else f"{name}/SKILL.md"

            parts = [
                f'<skill name="{name}">',
                f'  <description>{description}</description>',
            ]
            if when_to_use:
                parts.append(f'  <when_to_use>{when_to_use}</when_to_use>')
            parts.append(f'  <location>{location}</location>')
            if not is_organized:
                parts.append(
                    f'  <status>unorganized - move to skills/{name}/ before use</status>'
                )
            elif isinstance(skill_paths, list) and skill_paths:
                is_active = self._skill_paths_match(
                    skill_paths, self._touched_paths, self.workspace,
                )
                if not is_active:
                    parts.append(
                        f'  <status>dormant - activates when files matching '
                        f'{", ".join(skill_paths[:3])} are touched</status>'
                    )
            parts.append('</skill>')
            entries.append("\n".join(parts))

        return "<available_skills>\n" + "\n".join(entries) + "\n</available_skills>"

    @staticmethod
    def _build_dynamic_tools_prompt(inactive_tools: dict[str, "Tool"]) -> str:
        """Build the 'Dynamically Loadable Tools' system-prompt section.

        Lists ALL inactive tools with their descriptions so the AI Agent
        can autonomously decide which to activate. No hardcoded topic
        mapping - the LLM reads tool descriptions and decides for itself.
        """
        lines: list[str] = [
            "\n\n## Dynamically Loadable Tools\n\n"
            "Prefer specialized tools over broad search when a matching tool exists.\n\n"
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


    def get_skill_catalog(self) -> list[dict[str, Any]]:
        """Return structured metadata for skills visible to this agent.

        This is an observability/catalog surface only. It does not route user
        prompts or decide when a skill should execute.
        """
        catalog: list[dict[str, Any]] = []
        seen: set[str] = set()
        try:
            candidates = self._iter_skill_candidates(include_dormant=True)
        except Exception as exc:
            return [{"error": str(exc), "status": "failed"}]

        active: set[str] = set()
        if self._skill_manager is not None:
            try:
                active = set(self._skill_manager.list())
            except Exception:
                active = set()

        workspace_skills = (self.workspace / "skills").resolve()
        user_skill_roots = {Path(p).expanduser().resolve() for p in self._user_skill_paths}
        bundled_root = (Path(__file__).resolve().parent.parent.parent / "workspace" / "skills").resolve()

        for name, skill_dir, skill_md, is_organized in candidates:
            if name in seen:
                continue
            seen.add(name)
            try:
                resolved_dir = skill_dir.resolve()
            except Exception:
                resolved_dir = skill_dir
            source = "workspace"
            if resolved_dir == bundled_root or bundled_root in resolved_dir.parents:
                source = "bundled"
            elif any(resolved_dir == root or root in resolved_dir.parents for root in user_skill_roots):
                source = "configured"
            elif not (resolved_dir == workspace_skills or workspace_skills in resolved_dir.parents):
                source = "workspace-root"

            fm = self._parse_skill_frontmatter(skill_md)
            skill_paths = fm.get("paths", [])
            status = "available"
            if not is_organized:
                status = "unorganized"
            elif isinstance(skill_paths, list) and skill_paths and not self._skill_paths_match(
                skill_paths, self._touched_paths, self.workspace,
            ):
                status = "dormant"

            catalog.append({
                "name": name,
                "description": fm.get("description") or name,
                "when_to_use": fm.get("when_to_use") or "",
                "paths": skill_paths if isinstance(skill_paths, list) else [],
                "source": source,
                "status": status,
                "active": name in active,
                "base_dir": str(skill_dir),
                "skill_md": str(skill_md),
                "organized": bool(is_organized),
            })
        return catalog

    def get_mcp_catalog(self) -> list[dict[str, Any]]:
        """Return structured metadata for configured MCP servers and loaded tools."""
        loaded_by_server: dict[str, list[str]] = {}
        for tool in self._mcp_tools:
            config = getattr(tool, "mcp_config", {}) or {}
            server_name = getattr(tool, "server_name", None) or getattr(tool, "mcp_server_name", None)
            if not server_name:
                server_name = str(getattr(tool, "name", "unknown")).split("__", 1)[0]
            loaded_by_server.setdefault(str(server_name), []).append(str(getattr(tool, "name", "unknown")))

        catalog: list[dict[str, Any]] = []
        for name, config in self._mcp_config.items():
            transport = config.get("transport") or ("stdio" if config.get("command") else "unknown")
            loaded_tools = loaded_by_server.get(name, [])
            catalog.append({
                "name": name,
                "transport": transport,
                "command": config.get("command"),
                "url": config.get("url"),
                "status": "loaded" if loaded_tools else "configured",
                "tool_count": len(loaded_tools),
                "tools": loaded_tools,
            })
        for server_name, tools in loaded_by_server.items():
            if server_name not in self._mcp_config:
                catalog.append({
                    "name": server_name,
                    "transport": "unknown",
                    "status": "loaded",
                    "tool_count": len(tools),
                    "tools": tools,
                })
        return catalog

    def get_available_tools(self) -> list[dict[str, Any]]:
        """
        List all registered tools with their active/inactive status.

        Returns:
            List of dicts with name, description, and active flag.
        """
        return self.tools.get_all_tool_summaries()

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
