"""
Agent loop: the core processing engine using spoon-core SDK.

This module provides the main agent interface, integrating spoon-core's
ChatBot, SpoonReactMCP, and SkillManager with spoon-bot's native OS tools.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging as stdlib_logging
import re
from collections import Counter
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
from spoon_bot.agent.tools.execution_context import (
    bind_tool_owner,
    build_tool_owner_key,
    capture_tool_outputs,
    consume_captured_tool_output,
    normalize_tool_arguments,
)
from spoon_bot.agent.tools.self_config import (
    ActivateToolTool,
    SelfConfigTool,
    MemoryManagementTool,
    SelfUpgradeTool,
)
from spoon_bot.agent.tools.web import WebSearchTool, WebFetchTool
from spoon_bot.config import (
    AgentLoopConfig,
    DEFAULT_MAX_OUTPUT,
    DEFAULT_SHELL_MAX_TIMEOUT,
    DEFAULT_SHELL_TIMEOUT,
    MemSearchConfig,
    resolve_context_window,
    validate_agent_loop_params,
)
from spoon_bot.services.hotreload import HotReloadService
from spoon_bot.services.spawn import SpawnTool
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
from spoon_bot.utils.retry import RetryConfig, with_provider_retry, is_retryable

if TYPE_CHECKING:
    from spoon_bot.session.manager import Session


_ATTACHMENT_CONTEXT_HEADER = "Attached workspace files (source of truth for this request):"
_ATTACHMENT_ONLY_PLACEHOLDER = (
    "The user attached files without extra text. Inspect the files and answer based on their contents."
)
_SANDBOX_WORKSPACE_ROOT = "/workspace"
_MISSING = object()
_WALLET_REQUIRED_TOOLS: frozenset[str] = frozenset({
    "balance_check",
    "transfer",
    "swap",
    "contract_call",
})


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
            tool_profile: Named profile ('coding', 'web3', 'research', 'full').
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

        # Store config — callers must provide model/provider explicitly
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
        self.session_key = self._config.session_key
        self.user_id = "anonymous"
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
        self._latest_reasoning_excerpt: str | None = None
        self._pending_reasoning_chunks: list[str] = []

        # spoon-bot components
        self.context = ContextBuilder(self.workspace, yolo_mode=self.yolo_mode)
        self.tools = ToolRegistry()

        if self.yolo_mode:
            logger.info(f"YOLO mode enabled — operating directly in: {self.workspace}")

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

        # Tracks the skill name injected by _pre_inject_matched_skill()
        self._pre_injected_skill_name: str | None = None

        # Conditional activation — file paths touched during this session.
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
            system_prompt += (
                "\nWhen a user request matches a skill by name, description, or "
                "`when_to_use`, read its SKILL.md first (unless already pre-loaded "
                "in the message). Follow the skill's own procedures directly.\n"
            )

        system_prompt += (
            "\n## Workflow\n"
            f"You have up to {self.max_iterations} steps. Minimize steps.\n\n"
            "1. If a [PRE-LOADED SKILL] block is present in the user message, "
            "execute its instructions immediately — do NOT re-read it.\n"
            "2. Otherwise, if a skill matches, `read_file` its SKILL.md path, "
            "then execute.\n"
            "3. Run commands from SKILL.md directly via shell. Do NOT write script files.\n"
            "4. Summarize result when done.\n\n"
            "### Rules\n"
            "- Do NOT re-read files already in context.\n"
            "- `source .env.local` before commands that need env vars.\n"
            "- If a command fails, analyze the error and retry with fixes.\n"
            "- Follow user instructions exactly — respect specific IDs, names, actions.\n"
            "- Only use `web_search` if NO installed skill matches the task.\n"
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
        self._agent._spoon_bot_base_think = self._agent.think

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
        effective_ceiling = max(self.shell_timeout, self.shell_max_timeout)
        self._agent._default_timeout = max(300.0, float(effective_ceiling))

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
        # Shell tool — allow_chaining + allow_substitution lets the agent
        # compose multi-step ops and use $(), ${}, and backtick expressions.
        self.tools.register(ShellTool(
            timeout=self.shell_timeout,
            max_timeout=self.shell_max_timeout,
            max_output=self.max_output,
            working_dir=str(self.workspace),
            allow_chaining=True,
            allow_substitution=True,
        ))

        # Filesystem tools — allow reads from the user home directory so that
        # skill-managed data (e.g. ~/.agent-wallet, ~/.spoon-bot/skills) is
        # accessible.  The PathValidator blocklist still blocks truly sensitive
        # paths (.ssh, .aws, etc.).
        #
        # In YOLO mode the workspace IS the user's directory, so we add its
        # parents as extra read paths to let the agent navigate freely.
        _extra_read: list[Path] = [Path.home()]
        if self.yolo_mode:
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
        history_messages = (
            self._session.get_messages()
            if hasattr(self._session, "get_messages")
            else self._session.get_history()
        )
        for msg in history_messages:
            role = str(msg.get("role", "")).strip().lower()
            if role not in {"user", "assistant", "tool"}:
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
                logger.warning("Dropped invalid persisted media refs outside workspace during history sync")

            raw_attachments = _normalize_attachment_refs(msg.get("attachments"))
            attachments = _sanitize_attachment_refs(raw_attachments, self.workspace)
            if raw_attachments and len(attachments) != len(raw_attachments):
                logger.warning("Dropped invalid persisted attachment refs outside workspace during history sync")
            content = self._build_runtime_message_content(
                role,
                content,
                media=media,
                attachments=attachments,
            )

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

        # Repair tool-call ordering after history injection — session storage
        # may not preserve tool_call_id metadata, producing orphaned tool
        # messages that providers (OpenAI, Gemini, etc.) reject.
        repaired = 0
        if self._agent and hasattr(self._agent, "memory"):
            messages = getattr(self._agent.memory, "messages", None)
            if isinstance(messages, list):
                repaired = self._normalize_runtime_tool_context(messages)

        estimated_tokens = self._estimate_token_count()

        logger.info(
            f"Session context prepared: session={self.session_key}, "
            f"injected_messages={injected_count}, "
            f"estimated_tokens~{estimated_tokens}, "
            f"trimmed_messages={trimmed_count}"
            + (f", repaired_tool_order={repaired}" if repaired else "")
        )

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
        async def _do_run() -> Any:
            with bind_tool_owner(self._current_tool_owner_key()):
                return await self._agent.run(**run_kwargs)

        def _on_retry(attempt: int, exc: Exception, delay: float) -> None:
            logger.warning(
                f"[{label}] Provider transient error (attempt {attempt + 1}/"
                f"{self._retry_config.max_retries + 1}), "
                f"retrying in {delay:.1f}s: {type(exc).__name__}: {exc}"
            )
            if pre_retry_cleanup:
                try:
                    pre_retry_cleanup()
                except Exception:
                    pass

        return await with_provider_retry(
            _do_run,
            config=self._retry_config,
            on_retry=_on_retry,
        )

    async def process(
        self,
        message: str,
        media: list[str] | None = None,
        session_key: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
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

        logger.info(f"Processing message: {message[:100]}...")
        effective_reasoning_effort = reasoning_effort or getattr(self, "reasoning_effort", None)

        # Reset per-request state
        self._pre_injected_skill_name = None

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
        runtime_message = self._build_runtime_message_content(
            "user",
            message,
            media=media,
            attachments=attachments,
        )
        if isinstance(runtime_message, str):
            message = runtime_message

        # Build a minimal per-step prompt with the user's request for context.
        # The anti-loop tracker will dynamically append progress info.
        _base_prompt = self._build_step_prompt(message)
        self._agent.next_step_prompt = _base_prompt

        # Install anti-loop tracker to prevent repeated tool calls
        self._install_anti_loop_tracker(_base_prompt)
        await self._agent.add_message("user", runtime_message)

        # Run agent — two layers of retry:
        #   1. Provider retry (inner): exponential backoff for transient LLM errors
        #      (rate limits, timeouts, 5xx) — up to provider_max_retries (default 5).
        #   2. Context-compression retry (outer): up to 2 attempts with increasingly
        #      aggressive context trimming for non-transient errors.
        _max_context_retries = 2
        retry_requires_runtime_message_check = False

        try:
            for _attempt in range(_max_context_retries + 1):
                try:
                    if (
                        retry_requires_runtime_message_check
                        and not self._recent_runtime_has_user_message_content(runtime_message)
                    ):
                        await self._agent.add_message("user", runtime_message)
                    retry_requires_runtime_message_check = False
                    run_kwargs: dict[str, Any] = {}
                    if (
                        effective_reasoning_effort
                        and self._callable_accepts_kwarg(self._agent.run, "reasoning_effort")
                    ):
                        run_kwargs["reasoning_effort"] = effective_reasoning_effort
                    result = await self._run_agent_with_retry(label="process", **run_kwargs)

                    logger.debug(f"Agent result type: {type(result)}")
                    if hasattr(result, 'content'):
                        logger.info(f"Agent result.content (first 300): {str(result.content)[:300]}")

                    if hasattr(result, "content") and result.content is not None:
                        final_content = result.content
                    elif hasattr(result, "content"):
                        final_content = str(result) if str(result) != "None" else ""
                    else:
                        final_content = str(result)

                    if final_content.strip() in ("No results", ""):
                        logger.warning(
                            "Agent returned empty/no-results — attempting to extract "
                            "content from agent memory"
                        )
                        _extracted = self._extract_last_assistant_content()
                        if _extracted:
                            final_content = _extracted

                    final_content = self._filter_execution_steps(final_content)
                    break

                except Exception as e:
                    import traceback
                    logger.error(
                        f"Agent run error (context-retry {_attempt + 1}/{_max_context_retries + 1}): "
                        f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                    )

                    if is_retryable(e):
                        logger.error("Transient provider error exhausted retry budget — not compressing context")
                        raise

                    if _attempt < _max_context_retries:
                        logger.warning("Compressing context and retrying...")
                        compression_actions = 0
                        if _attempt == 0:
                            compression_actions = self._compress_runtime_context()
                            if compression_actions == 0:
                                compression_actions += self._force_compress_runtime_context()
                        else:
                            compression_actions = self._force_compress_runtime_context()
                        retry_requires_runtime_message_check = True
                        if hasattr(self._agent, 'state'):
                            self._agent.state = AgentState.IDLE
                            self._agent.current_step = 0
                        if hasattr(self._agent, '_shutdown_event'):
                            self._agent._shutdown_event.clear()
                        continue
                    raise
        finally:
            # Always ensure agent is back in IDLE state after processing
            self._restore_agent_think()
            if hasattr(self._agent, 'state') and self._agent.state != AgentState.IDLE:
                logger.warning(
                    f"Post-run cleanup: resetting agent from {self._agent.state} to IDLE"
                )
                self._agent.state = AgentState.IDLE
                self._agent.current_step = 0

        # Save to session
        try:
            save_kwargs: dict[str, Any] = {}
            if media:
                save_kwargs["media"] = list(media)
            if attachments:
                save_kwargs["attachments"] = [dict(item) for item in attachments if isinstance(item, dict)]
            self._session.add_message(
                "user",
                _strip_attachment_context(message, attachments or []),
                **save_kwargs,
            )
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
    def _message_content_fingerprint(cls, content: Any) -> str:
        """Create a stable string fingerprint for runtime content comparisons."""
        serialized = cls._serialize_message_content(content)
        if isinstance(serialized, str):
            return f"str:{serialized}"
        return json.dumps(serialized, sort_keys=True, ensure_ascii=True, default=str)

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

    def _is_next_step_user_msg(self, msg) -> bool:
        """True when *msg* looks like an injected next_step_prompt (not a real user message)."""
        role = getattr(msg, 'role', None)
        if hasattr(role, 'value'):
            role = role.value
        if role != 'user':
            return False
        text = msg.content if isinstance(msg.content, str) else ''
        return text.startswith('[ORIGINAL USER REQUEST]') or text.startswith('Focus on the user')

    def _recent_runtime_has_user_message_content(
        self,
        content: Any,
        recent_messages: int = 10,
    ) -> bool:
        """Check whether recent runtime memory still contains the active user request."""
        if not self._agent or not hasattr(self._agent, "memory"):
            return False
        messages = getattr(self._agent.memory, "messages", None)
        if not isinstance(messages, list) or not messages:
            return False

        target = self._message_content_fingerprint(content)
        for msg in reversed(messages[-recent_messages:]):
            role = getattr(msg, "role", None)
            role = role.value if hasattr(role, "value") else role
            if role != "user":
                continue
            if self._is_next_step_user_msg(msg):
                continue
            if self._message_content_fingerprint(getattr(msg, "content", None)) == target:
                return True
        return False

    @staticmethod
    def _callable_accepts_kwarg(func: Any, kwarg: str) -> bool:
        """Return True when *func* can accept *kwarg* as a keyword argument."""
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
        role = getattr(msg, "role", None)
        return role.value if hasattr(role, "value") else role

    @classmethod
    def _reorder_tool_messages(cls, messages: list) -> int:
        """Move tool results to immediately follow the issuing assistant turn."""
        if not messages:
            return 0

        claimed_tool_indices: set[int] = set()
        tool_messages_by_assistant_index: dict[int, list] = {}

        for index, message in enumerate(messages):
            if cls._message_role_value(message) != "tool":
                continue

            tool_call_id = getattr(message, "tool_call_id", None)
            if not tool_call_id:
                continue

            for candidate_index in range(index - 1, -1, -1):
                candidate = messages[candidate_index]
                if cls._message_role_value(candidate) != "assistant":
                    continue
                tool_calls = getattr(candidate, "tool_calls", None) or []
                if any(getattr(tool_call, "id", None) == tool_call_id for tool_call in tool_calls):
                    tool_messages_by_assistant_index.setdefault(candidate_index, []).append(message)
                    claimed_tool_indices.add(index)
                    break

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

            tool_calls = getattr(message, "tool_calls", None) or []
            if not tool_calls:
                continue

            tool_order = {
                getattr(tool_call, "id", None): position
                for position, tool_call in enumerate(tool_calls)
            }
            matched_tool_messages = tool_messages_by_assistant_index.get(index, [])
            matched_tool_messages.sort(
                key=lambda item: tool_order.get(getattr(item, "tool_call_id", None), len(tool_order))
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
    def _repair_tool_pairing(cls, messages: list) -> int:
        """Remove orphaned tool results and tool calls after message deletion.

        Ensures every tool_call_id in a tool-result message has a matching
        tool_calls entry in a preceding assistant message, and vice-versa.
        Also removes tool-role messages with no tool_call_id at all (e.g.
        injected from session history without metadata).
        Without this, the LLM API rejects the conversation.

        Returns the number of messages removed.
        """
        removed = 0

        # Phase 1: remove orphaned tool messages — delegate to spoon-core's
        # shared utility when available so detection logic isn't duplicated.
        try:
            from spoon_ai.llm.message_utils import drop_orphaned_tool_messages

            before = len(messages)
            cleaned = drop_orphaned_tool_messages(messages)
            messages[:] = cleaned
            removed += before - len(messages)
        except ImportError:
            offered_ids: set[str] = set()
            for msg in messages:
                if getattr(msg, 'tool_calls', None):
                    for tc in msg.tool_calls:
                        tc_id = getattr(tc, 'id', None)
                        if tc_id:
                            offered_ids.add(tc_id)

            i = 0
            while i < len(messages):
                msg = messages[i]
                role = cls._message_role_value(msg)
                tc_id = getattr(msg, 'tool_call_id', None)

                if role == "tool" and not tc_id:
                    del messages[i]
                    removed += 1
                    continue
                if tc_id and tc_id not in offered_ids:
                    del messages[i]
                    removed += 1
                    continue
                i += 1

        # Phase 2: collect all answered tool_call IDs
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

    @classmethod
    def _normalize_runtime_tool_context(cls, messages: list) -> int:
        """Repair runtime tool-call sequencing before sending history back to the LLM."""
        normalized = cls._reorder_tool_messages(messages)
        normalized += cls._repair_tool_pairing(messages)
        return normalized

    def _uses_strict_tool_turn_order(self) -> bool:
        """True for providers/models that reject non-adjacent function call turns.

        Both OpenAI and Gemini require tool-result messages to immediately
        follow the assistant message that issued the tool_calls.
        """
        provider_raw = getattr(self, "provider", None)
        model_raw = getattr(self, "model", None)
        base_url_raw = getattr(self, "base_url", None)

        provider = provider_raw.strip().lower() if isinstance(provider_raw, str) else ""
        model = model_raw.strip().lower() if isinstance(model_raw, str) else ""
        base_url = base_url_raw.strip().lower() if isinstance(base_url_raw, str) else ""

        if provider in {"openai", "openrouter"}:
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
        normalized = self._normalize_runtime_tool_context(messages)
        if len(messages) <= 6:
            return normalized

        estimated = self._estimate_runtime_tokens()
        budget = int(self.context_window * 0.50)

        if estimated <= budget:
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

        # Phase 2: Truncate content of ALL messages except the first and last 6.
        keep_tail = min(6, len(messages))
        max_content = 300
        for i in range(1, max(1, len(messages) - keep_tail)):
            msg = messages[i]
            original_content = getattr(msg, "content", None)
            compressed_content = self._compress_message_content(original_content, max_content)
            if compressed_content != original_content:
                msg.content = compressed_content
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
        compressed = self._normalize_runtime_tool_context(messages)
        if len(messages) <= 4:
            return compressed

        # Truncate ALL messages to 150 chars
        for msg in messages:
            original_content = getattr(msg, "content", None)
            compressed_content = self._compress_message_content(original_content, 150)
            if compressed_content != original_content:
                msg.content = compressed_content
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
        original_think = getattr(agent, "_spoon_bot_base_think", None)
        if original_think is None:
            original_think = agent.think
            setattr(agent, "_spoon_bot_base_think", original_think)
        else:
            agent.think = original_think
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

        async def _tracked_think(*args: Any, **kwargs: Any) -> bool:
            _evict_duplicate_tool_results()
            agent_loop._compress_runtime_context()

            desired_next_step_prompt = agent.next_step_prompt or base_prompt
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

                desired_next_step_prompt = base_prompt + anti_loop

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
            provider_reasoning = getattr(agent, "last_reasoning_summary", None)
            if isinstance(provider_reasoning, str) and provider_reasoning.strip():
                safe_reasoning = mask_secrets(provider_reasoning.strip())
                captured = agent_loop._capture_reasoning_text(safe_reasoning)
                if captured:
                    logger.info(f"💭 Agent reasoning: {captured}")
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
            merged["result"] = full_result
            merged["content"] = full_result
            merged["full_result"] = full_result
            merged["full_content"] = full_result
        if summary_result:
            merged.setdefault("model_result", summary_result)
            merged.setdefault("model_content", summary_result)
        if full_result and summary_result and full_result != summary_result:
            merged["result_truncated_for_model"] = True
        return merged

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
              type:     "content" | "thinking" | "tool_call" | "tool_result" | "done"
              delta:    Incremental text (may be empty for non-text events)
              metadata: Extra context (tool name, args, step number, etc.)
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"Streaming message: {message[:100]}...")
        self._reset_reasoning_capture()
        effective_reasoning_effort = reasoning_effort or getattr(self, "reasoning_effort", None)

        # Refresh memory context
        try:
            memory_context = self.memory.get_memory_context()
            if memory_context:
                self.context.set_memory_context(memory_context)
        except Exception as e:
            logger.warning(f"Failed to load memory context: {e}")

        full_content = ""
        saw_tool_call = False
        saw_content_after_tool_call = False
        stream_completed = False
        stream_cancelled = False
        bg_task: asyncio.Task[None] | None = None
        original_system_prompt: str | None = None
        original_base_system_prompt: object = _MISSING
        tool_output_capture_scope: str | None = None
        capture_manager = None

        # Trim and inject persisted history into runtime memory
        await self._prepare_request_context()
        runtime_message = self._build_runtime_message_content(
            "user",
            message,
            media=media,
            attachments=attachments,
        )
        if isinstance(runtime_message, str):
            message = runtime_message

        try:
            capture_manager = capture_tool_outputs()
            tool_output_capture_scope = capture_manager.__enter__()
            _base_prompt = self._select_next_step_prompt(message, thinking=thinking)
            original_system_prompt, original_base_system_prompt = (
                AgentLoop._apply_request_context_to_system_prompt(self, message, thinking=thinking)
            )
            self._agent.next_step_prompt = _base_prompt
            self._install_anti_loop_tracker(_base_prompt)

            # ------------------------------------------------------------------
            # Streaming uses the spoon-core run+stream pattern:
            #   1. Clear task_done + drain output_queue
            #   2. Start run(message) in background — sets task_done on finish
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
            stream_tool_result_index = len(AgentLoop._get_runtime_memory_messages(self))
            emitted_tool_result_ids: set[str] = set()
            tool_call_arguments_by_id: dict[str, str] = {}

            # 2. Start run() in background
            run_result_text = ""

            async def _run_and_signal() -> None:
                nonlocal run_result_text
                try:
                    run_kwargs: dict[str, Any] = {}
                    if thinking and self._callable_accepts_kwarg(self._agent.run, "thinking"):
                        run_kwargs["thinking"] = True
                    if (
                        effective_reasoning_effort
                        and self._callable_accepts_kwarg(self._agent.run, "reasoning_effort")
                    ):
                        run_kwargs["reasoning_effort"] = effective_reasoning_effort

                    def _drain_queue() -> None:
                        while not self._agent.output_queue.empty():
                            self._agent.output_queue.get_nowait()

                    result = await self._run_agent_with_retry(
                        label="stream",
                        pre_retry_cleanup=_drain_queue,
                        **run_kwargs,
                    )
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
                if event_type not in {"thinking", "content", "tool_call", "tool_result"}:
                    return event

                metadata = dict(event.get("metadata") or {})
                starts_new_segment = (
                    event_type in {"tool_call", "tool_result"}
                    or current_stream_segment_type != event_type
                )
                if starts_new_segment:
                    stream_segment_index += 1
                metadata.setdefault("segment_index", stream_segment_index)
                metadata.setdefault("segment_start", starts_new_segment)
                metadata.setdefault("segment_type", event_type)

                if event_type in {"tool_call", "tool_result"}:
                    current_stream_segment_type = None
                else:
                    current_stream_segment_type = event_type

                return {
                    **event,
                    "metadata": metadata,
                }

            logger.debug(f"Entering stream loop: td={td.is_set()}, qempty={oq.empty()}, qsize={oq.qsize()}")
            while not (td.is_set() and oq.empty()):
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

                # tracked_reasoning is inferred from assistant output logs and is
                # not a reliable API thinking source. Clear it so it does not leak
                # duplicated final content into WS/REST responses.
                self._drain_reasoning_chunks()
                try:
                    # Poll without a hard stream deadline so long-running tasks
                    # only stop when the caller explicitly cancels them.
                    # Use oq.get() without timeout kwarg — works for both
                    # asyncio.Queue and ThreadSafeOutputQueue. Timeout is
                    # handled by the outer asyncio.wait_for.
                    chunk = await asyncio.wait_for(oq.get(), timeout=2.0)
                    chunk_count += 1
                    logger.debug(f"Got chunk #{chunk_count}: type={type(chunk).__name__}, repr={repr(chunk)[:200]}")
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    stream_cancelled = True
                    logger.warning("Streaming cancelled while waiting for output")
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
                        yield {
                            "type": "error",
                            "delta": chunk.get("delta", ""),
                            "metadata": chunk.get("metadata", {}),
                        }
                        continue

                    if "tool_calls" in chunk and chunk["tool_calls"]:
                        saw_tool_call = True
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
                    yield _decorate_stream_event({
                        "type": chunk_type,
                        "delta": delta,
                        "metadata": metadata,
                    })
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
                    else:
                        event = {"type": chunk_type, "delta": delta, "metadata": metadata}
                        if chunk_type == "content":
                            if saw_tool_call:
                                saw_content_after_tool_call = True
                            full_content += delta
                        yield _decorate_stream_event(event)

            logger.debug(f"Stream loop exited: td={td.is_set()}, qempty={oq.empty()}, chunks_received={chunk_count}, full_content_len={len(full_content)}")

            # Ensure background task completes
            try:
                await asyncio.wait_for(bg_task, timeout=5.0)
            except (asyncio.TimeoutError, Exception):
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
                if fallback_delta:
                    yield _decorate_stream_event({
                        "type": "content",
                        "delta": fallback_delta,
                        "metadata": {"fallback": fallback_reason},
                    })

            # Emit done
            stream_completed = True
            yield {"type": "done", "delta": "", "metadata": {"content": full_content}}

        except asyncio.CancelledError:
            stream_cancelled = True
            logger.warning("Streaming cancelled")
            raise
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            stream_completed = True
            yield {"type": "error", "delta": str(e), "metadata": {"error": str(e)}}
            yield {"type": "done", "delta": "", "metadata": {"error": str(e)}}
        finally:
            if capture_manager is not None:
                capture_manager.__exit__(None, None, None)
            self._restore_agent_think()
            AgentLoop._restore_request_context_system_prompt(
                self,
                original_system_prompt,
                original_base_system_prompt,
            )
            if bg_task is not None and not bg_task.done():
                bg_task.cancel()
                try:
                    await asyncio.wait_for(bg_task, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                    pass

        # Save to session only if we got actual content
        if full_content and stream_completed and not stream_cancelled:
            try:
                save_kwargs: dict[str, Any] = {}
                if media:
                    save_kwargs["media"] = list(media)
                if attachments:
                    save_kwargs["attachments"] = [dict(item) for item in attachments if isinstance(item, dict)]
                self._session.add_message(
                    "user",
                    _strip_attachment_context(message, attachments or []),
                    **save_kwargs,
                )
                self._session.add_message("assistant", full_content)
                self.sessions.save(self._session)
            except Exception as e:
                logger.warning(f"Failed to save session after streaming: {e}")

    async def process_with_thinking(
        self,
        message: str,
        media: list[str] | None = None,
        session_key: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
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
        if not self._initialized:
            await self.initialize()

        # Switch session if a different key is requested
        if session_key and session_key != self.session_key:
            self._session = self.sessions.get_or_create(session_key)
            self.session_key = session_key
            logger.debug(f"Switched to session: {session_key}")

        logger.info(f"Processing message (with thinking): {message[:100]}...")
        self._reset_reasoning_capture()
        effective_reasoning_effort = reasoning_effort or getattr(self, "reasoning_effort", None)
        original_system_prompt: str | None = None
        original_base_system_prompt: object = _MISSING

        # Refresh memory context
        try:
            memory_context = self.memory.get_memory_context()
            if memory_context:
                self.context.set_memory_context(memory_context)
        except Exception as e:
            logger.warning(f"Failed to load memory context: {e}")

        # Trim and inject persisted history into runtime memory
        await self._prepare_request_context()
        runtime_message = self._build_runtime_message_content(
            "user",
            message,
            media=media,
            attachments=attachments,
        )
        if isinstance(runtime_message, str):
            message = runtime_message

        try:
            _base_prompt = self._select_next_step_prompt(message, thinking=True)
            original_system_prompt, original_base_system_prompt = (
                AgentLoop._apply_request_context_to_system_prompt(self, message, thinking=True)
            )
            self._agent.next_step_prompt = _base_prompt
            self._install_anti_loop_tracker(_base_prompt)
            await self._agent.add_message("user", runtime_message)

            # Run agent with thinking enabled
            run_kwargs: dict[str, Any] = {}
            if self._callable_accepts_kwarg(self._agent.run, "thinking"):
                run_kwargs["thinking"] = True
            if (
                effective_reasoning_effort
                and self._callable_accepts_kwarg(self._agent.run, "reasoning_effort")
            ):
                run_kwargs["reasoning_effort"] = effective_reasoning_effort
            with bind_tool_owner(self._current_tool_owner_key()):
                result = await self._agent.run(**run_kwargs)

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
                thinking_content = (
                    result.metadata.get("thinking")
                    or result.metadata.get("reasoning")
                )
            if self._looks_like_duplicate_thinking(thinking_content, final_content):
                thinking_content = None

        except Exception as e:
            logger.error(f"Agent processing error: {e}")
            raise
        finally:
            self._restore_agent_think()
            AgentLoop._restore_request_context_system_prompt(
                self,
                original_system_prompt,
                original_base_system_prompt,
            )

        # Save to session
        try:
            save_kwargs: dict[str, Any] = {}
            if media:
                save_kwargs["media"] = list(media)
            if attachments:
                save_kwargs["attachments"] = [dict(item) for item in attachments if isinstance(item, dict)]
            self._session.add_message(
                "user",
                _strip_attachment_context(message, attachments or []),
                **save_kwargs,
            )
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

    # Common stop words filtered out during skill-matching token comparison
    _SKILL_MATCH_STOP_WORDS = frozenset({
        "when", "use", "this", "the", "for", "and", "with",
        "that", "from", "needs", "need", "should", "also",
        "through", "using", "codex", "claude", "agent",
        "local", "based", "first", "only", "needed", "prefer", "logic",
    })

    # Scoring weights for _pre_inject_matched_skill (tune here, not inline)
    _SCORE_NAME_TOKEN = 2
    _SCORE_TRIGGER = 3
    _SCORE_WHEN_TO_USE = 4
    _SCORE_DESCRIPTION = 1
    _SCORE_THRESHOLD = 2

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
        negates (exclude) the pattern — same semantics as ``.gitignore``.
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
        1. ``workspace/skills/`` — primary organized location
        2. All entries in ``_skill_paths`` (includes bundled/dev skills)
        3. Workspace root — unorganized skills the user may have dropped in

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

    def _pre_inject_matched_skill(self, message: str) -> str:
        """Match user message to an installed skill and prepend SKILL.md content.

        Scoring uses multiple signals inspired by Claude Code's skill matching:
        - Skill name tokens (split on ``-`` / ``_``)
        - YAML frontmatter ``description`` keywords
        - ``when_to_use`` / ``whenToUse`` field (highest weight)
        - ``triggers`` keyword list

        Scans both ``workspace/skills/`` (organized) and the workspace root
        (unorganized).  When an unorganized skill matches, the injection
        includes a note asking the agent to move it into ``skills/`` first.
        """
        import re as _re, sys as _sys

        candidates = self._iter_skill_candidates()
        if not candidates:
            return message

        msg_lower = message.lower()
        best_skill = None
        best_score = 0

        for name, skill_dir, skill_md, is_organized in candidates:
            score = 0

            name_tokens = _re.split(r'[-_]', name)
            for token in name_tokens:
                if len(token) >= 3 and token.lower() in msg_lower:
                    score += self._SCORE_NAME_TOKEN

            fm = self._parse_skill_frontmatter(skill_md)

            for t in fm["triggers"].split("|"):
                if t and t.lower() in msg_lower:
                    score += self._SCORE_TRIGGER

            if fm["when_to_use"]:
                wtu_tokens = _re.findall(r'[A-Za-z\u4e00-\u9fff]{2,}', fm["when_to_use"].lower())
                for token in wtu_tokens:
                    if token not in self._SKILL_MATCH_STOP_WORDS and token in msg_lower:
                        score += self._SCORE_WHEN_TO_USE

            if fm["description"]:
                desc_tokens = _re.findall(r'[A-Za-z\u4e00-\u9fff]{3,}', fm["description"].lower())
                for token in desc_tokens:
                    if token not in self._SKILL_MATCH_STOP_WORDS and token in msg_lower:
                        score += self._SCORE_DESCRIPTION

            if score > best_score:
                best_score = score
                best_skill = (name, skill_dir, skill_md, is_organized)

        if not best_skill or best_score < self._SCORE_THRESHOLD:
            return message

        skill_name, skill_dir, skill_path, is_organized = best_skill
        try:
            content = skill_path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            return message

        base_dir = str(skill_dir).replace("\\", "/")
        if _sys.platform == "win32":
            base_dir = _re.sub(r'^([A-Za-z]):', lambda m: f'/{m.group(1).lower()}', base_dir)

        ws_posix = self._workspace_posix_path()

        if is_organized:
            skill_rel = f"skills/{skill_name}"
        else:
            skill_rel = skill_name

        content = content.replace("${SKILL_DIR}", f"{ws_posix}/{skill_rel}")
        content = content.replace("$SKILL_DIR", f"{ws_posix}/{skill_rel}")

        self._pre_injected_skill_name = skill_name

        organize_note = ""
        if not is_organized:
            organize_note = (
                f"\n⚠️ ORGANIZE FIRST: This skill is at the workspace root, not in skills/.\n"
                f"Before executing, move it:\n"
                f"  mv {skill_name} skills/{skill_name}\n"
                f"Then update your working paths to use skills/{skill_name}/ as the base.\n"
            )

        logger.info(
            f"Pre-injected skill '{skill_name}' (score={best_score}, "
            f"organized={is_organized}) into message"
        )
        return (
            f"{message}\n\n"
            f"---\n"
            f"[PRE-LOADED SKILL: {skill_name}]\n"
            f"Base directory: {base_dir}\n"
            f"Workspace-relative path: {skill_rel}/\n"
            f"{organize_note}\n"
            f"The SKILL.md below is already loaded — follow its procedures directly. "
            f"Do NOT call read_file on this skill. Do NOT search for alternatives. "
            f"Start executing the skill's instructions immediately.\n\n"
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

    def _build_request_context_prompt(self, message: str) -> str:
        """Build the compact request context block used for thinking runs."""
        _truncated = message[:300] + ("…" if len(message) > 300 else "")
        _ws = self._workspace_posix_path()
        prompt = (
            f"[USER REQUEST]: {_truncated}\n"
            f"[WORKSPACE]: {_ws}/\n"
        )
        env_section = self._extract_env_for_prompt()
        if env_section:
            prompt += env_section
        return prompt

    def _apply_request_context_to_system_prompt(
        self,
        message: str,
        *,
        thinking: bool,
    ) -> tuple[str | None, object]:
        """Temporarily append active request context to the agent system prompt."""
        if not thinking or not getattr(self, "_agent", None):
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
                    f'  <status>unorganized — move to skills/{name}/ before use</status>'
                )
            elif isinstance(skill_paths, list) and skill_paths:
                is_active = self._skill_paths_match(
                    skill_paths, self._touched_paths, self.workspace,
                )
                if not is_active:
                    parts.append(
                        f'  <status>dormant — activates when files matching '
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
    yolo_mode: bool = False,
    reasoning_effort: str | None = None,
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
        yolo_mode: Operate directly in user's path without sandbox isolation.
        **kwargs: Additional arguments for AgentLoop.

    Returns:
        Initialized AgentLoop instance.

    Example:
        >>> agent = await create_agent()
        >>> response = await agent.process("Hello!")

        >>> # Load all tools
        >>> agent = await create_agent(tool_profile="full")

        >>> # YOLO mode — work in /home/user/project directly
        >>> agent = await create_agent(yolo_mode=True, workspace="/home/user/project")
    """
    wallet_required = bool(enabled_tools and _WALLET_REQUIRED_TOOLS.intersection(enabled_tools))
    try:
        ensure_wallet_runtime(workspace)
    except Exception:
        if wallet_required:
            raise
        logger.warning("Wallet runtime bootstrap failed; continuing because no wallet-required tools are enabled")
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
        reasoning_effort=reasoning_effort,
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
