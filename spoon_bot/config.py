"""Configuration validation models for spoon-bot.

This module provides Pydantic-based configuration validation for all
components of spoon-bot, ensuring type safety and early error detection.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Shared default constants (single source of truth)
# ---------------------------------------------------------------------------

DEFAULT_SHELL_TIMEOUT: int = 600       # 10 min foreground budget
DEFAULT_SHELL_MAX_TIMEOUT: int = 7200  # 2 hour per-command cap
DEFAULT_MAX_OUTPUT: int = 10_000
DEFAULT_MAX_ITERATIONS: int = 20

# ---------------------------------------------------------------------------
# Model context-window lookup (tokens)
# Used to auto-configure context budget when the user doesn't specify one.
# ---------------------------------------------------------------------------

MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # ---- Anthropic ----
    "claude-opus-4.6":       1_000_000,
    "claude-sonnet-4.5":     1_000_000,
    "claude-sonnet-4":       1_000_000,
    "claude-opus-4.5":         200_000,
    "claude-opus-4.1":         200_000,
    "claude-opus-4":           200_000,
    "claude-haiku-4.5":        200_000,
    # ---- OpenAI ----
    "gpt-5.2":                 400_000,
    "gpt-5.2-codex":           400_000,
    "gpt-5.2-chat":            128_000,
    "gpt-5.2-pro":             400_000,
    "gpt-5.1":                 400_000,
    "gpt-5.1-codex":           400_000,
    "gpt-5.1-codex-mini":      400_000,
    "gpt-5":                   400_000,
    "gpt-5-mini":              400_000,
    "gpt-5-nano":              400_000,
    "gpt-5-chat":              128_000,
    "gpt-4o":                  128_000,
    "gpt-4o-mini":             128_000,
    "o4-mini":                 200_000,
    "o3":                      200_000,
    "o3-mini":                 200_000,
    # ---- DeepSeek ----
    "deepseek-v3.2":           163_840,
    "deepseek-chat-v3.1":      32_768,
    "deepseek-chat":           163_840,
    "deepseek-r1":              64_000,
    # ---- Google Gemini ----
    "gemini-3-pro-preview":  1_048_576,
    "gemini-3-flash-preview":1_048_576,
    "gemini-2.5-pro":        1_048_576,
    "gemini-2.5-flash":      1_048_576,
    "gemini-2.5-flash-lite": 1_048_576,
    "gemini-2.0-flash":      1_048_576,
    # ---- Qwen (via OpenRouter) ----
    "qwen3-max-thinking":      262_144,
    "qwen3-coder-next":        262_144,
    "qwen3-coder-plus":      1_000_000,
    "qwen3-coder-flash":     1_000_000,
    "qwen3-max":               262_144,
    # ---- Moonshot ----
    "kimi-k2.5":               262_144,
    # ---- MiniMax ----
    "minimax-m2.5":            204_800,
}

DEFAULT_CONTEXT_WINDOW = 128_000


def resolve_context_window(model: str | None, explicit: int | None = None) -> int:
    """Return the effective context window for a given model.

    Priority:
      1. ``explicit`` value (user override)
      2. Lookup in ``MODEL_CONTEXT_WINDOWS`` (exact match, then suffix match)
      3. ``DEFAULT_CONTEXT_WINDOW`` (128 000)
    """
    if explicit is not None:
        return explicit

    if model is None:
        return DEFAULT_CONTEXT_WINDOW

    # Exact match
    if model in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[model]

    # Strip provider prefix (e.g. "anthropic/claude-sonnet-4.5" -> "claude-sonnet-4.5")
    short = model.rsplit("/", 1)[-1] if "/" in model else model

    if short in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[short]

    # Suffix / substring match (e.g. "claude-sonnet-4.5-20260101" -> "claude-sonnet-4.5")
    for key, ctx in MODEL_CONTEXT_WINDOWS.items():
        if short.startswith(key):
            return ctx

    return DEFAULT_CONTEXT_WINDOW


class TransportType(str, Enum):
    """Supported MCP transport types."""
    STDIO = "stdio"
    SSE = "sse"
    HTTP = "http"
    HTTP_STREAM = "http-stream"
    WEBSOCKET = "websocket"
    NPX = "npx"
    UVX = "uvx"
    PYTHON = "python"


class LLMProviderType(str, Enum):
    """Supported LLM provider types."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    OLLAMA = "ollama"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server with validation."""

    model_config = ConfigDict(extra="allow")  # Allow additional transport-specific fields

    name: str = Field(..., min_length=1, description="Unique server identifier")
    transport: TransportType = Field(
        default=TransportType.STDIO,
        description="Transport type for MCP communication"
    )
    command: str | None = Field(
        default=None,
        description="Command to run for stdio transport"
    )
    args: list[str] = Field(
        default_factory=list,
        description="Command arguments"
    )
    url: str | None = Field(
        default=None,
        description="URL for HTTP/SSE/WebSocket transports"
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables for the server process"
    )
    timeout: int = Field(
        default=30,
        ge=1,
        le=3600,
        description="Connection timeout in seconds (1-3600)"
    )
    health_check_interval: int = Field(
        default=60,
        ge=0,
        le=3600,
        description="Health check interval in seconds (0 to disable)"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts (0-10)"
    )

    @model_validator(mode="after")
    def validate_transport_requirements(self) -> "MCPServerConfig":
        """Validate that required fields are present for each transport type."""
        if self.transport in (TransportType.STDIO, TransportType.NPX, TransportType.UVX, TransportType.PYTHON):
            if not self.command:
                raise ValueError(
                    f"Transport '{self.transport.value}' requires 'command' field"
                )
        elif self.transport in (TransportType.SSE, TransportType.HTTP, TransportType.HTTP_STREAM, TransportType.WEBSOCKET):
            if not self.url:
                raise ValueError(
                    f"Transport '{self.transport.value}' requires 'url' field"
                )
        return self

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str | None) -> str | None:
        """Validate URL format if provided."""
        if v is not None:
            v = v.strip()
            if not v.startswith(("http://", "https://", "ws://", "wss://")):
                raise ValueError("URL must start with http://, https://, ws://, or wss://")
        return v


class SkillTriggerConfig(BaseModel):
    """Configuration for a skill trigger."""

    type: Literal["keyword", "pattern", "intent"] = Field(
        default="keyword",
        description="Trigger type"
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Keywords that activate this trigger"
    )
    patterns: list[str] = Field(
        default_factory=list,
        description="Regex patterns that activate this trigger"
    )
    intent_category: str | None = Field(
        default=None,
        description="Intent category for LLM-based matching"
    )
    priority: int = Field(
        default=50,
        ge=0,
        le=100,
        description="Trigger priority (0-100, higher is more important)"
    )

    @model_validator(mode="after")
    def validate_trigger_content(self) -> "SkillTriggerConfig":
        """Ensure at least one trigger mechanism is defined."""
        if self.type == "keyword" and not self.keywords:
            raise ValueError("Keyword trigger requires at least one keyword")
        elif self.type == "pattern" and not self.patterns:
            raise ValueError("Pattern trigger requires at least one pattern")
        elif self.type == "intent" and not self.intent_category:
            raise ValueError("Intent trigger requires intent_category")
        return self


class SkillConfig(BaseModel):
    """Configuration for a skill."""

    name: str = Field(..., min_length=1, description="Unique skill name")
    description: str = Field(..., min_length=1, description="Skill description")
    version: str = Field(default="1.0.0", description="Skill version")
    author: str | None = Field(default=None, description="Skill author")
    tags: list[str] = Field(default_factory=list, description="Skill tags")
    triggers: list[SkillTriggerConfig] = Field(
        default_factory=list,
        description="Skill activation triggers"
    )
    prerequisites: list[str] = Field(
        default_factory=list,
        description="Required skills that must be active first"
    )
    composable: bool = Field(
        default=False,
        description="Whether this skill can be composed into others"
    )
    composes: list[str] = Field(
        default_factory=list,
        description="Skills this skill composes"
    )
    persist_state: bool = Field(
        default=False,
        description="Whether to persist skill state across sessions"
    )


class ToolParameterSchema(BaseModel):
    """JSON Schema for tool parameters with validation."""

    type: Literal["object"] = Field(default="object")
    properties: dict[str, dict[str, Any]] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)

    @field_validator("required")
    @classmethod
    def validate_required_in_properties(
        cls, v: list[str], info: Any
    ) -> list[str]:
        """Ensure all required fields are defined in properties."""
        # Note: This validation runs before properties may be set,
        # so we do a model-level validation instead
        return v

    @model_validator(mode="after")
    def validate_required_fields_exist(self) -> "ToolParameterSchema":
        """Validate that all required fields exist in properties."""
        missing = set(self.required) - set(self.properties.keys())
        if missing:
            raise ValueError(
                f"Required fields not defined in properties: {missing}"
            )
        return self


class MemSearchConfig(BaseModel):
    """Configuration for memsearch-based semantic memory."""

    enabled: bool = Field(
        default=False,
        description="Enable semantic memory search via memsearch"
    )
    embedding_provider: str = Field(
        default="openai",
        description="Embedding provider: 'openai' (OpenAI-compatible), 'local', 'ollama', etc."
    )
    embedding_model: str | None = Field(
        default=None,
        description="Embedding model name (provider default if None)"
    )
    embedding_api_key: str | None = Field(
        default=None,
        description="API key for the embedding provider (falls back to OPENAI_API_KEY env)"
    )
    embedding_base_url: str | None = Field(
        default=None,
        description="Base URL for the embedding API (falls back to OPENAI_BASE_URL env)"
    )
    milvus_uri: str | None = Field(
        default=None,
        description="Milvus connection URI (defaults to workspace/memsearch/milvus.db)"
    )
    collection: str = Field(
        default="spoon_bot_memory",
        description="Milvus collection name"
    )

    def get_embedding_api_key(self) -> str | None:
        """Get API key: config > OPENAI_EMBEDDING_API_KEY > OPENAI_API_KEY."""
        return (
            self.embedding_api_key
            or os.environ.get("OPENAI_EMBEDDING_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )

    def get_embedding_base_url(self) -> str | None:
        """Get base URL: config > OPENAI_EMBEDDING_BASE_URL > OPENAI_BASE_URL."""
        return (
            self.embedding_base_url
            or os.environ.get("OPENAI_EMBEDDING_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
        )

    def get_embedding_model(self) -> str | None:
        """Get model name: config > OPENAI_EMBEDDING_MODEL."""
        return self.embedding_model or os.environ.get("OPENAI_EMBEDDING_MODEL")


class AgentLoopConfig(BaseModel):
    """Configuration for AgentLoop with validation."""

    model_config = ConfigDict(validate_default=True)

    workspace: Path = Field(
        default_factory=lambda: Path.home() / ".spoon-bot" / "workspace",
        description="Workspace directory path"
    )

    yolo_mode: bool = Field(
        default=False,
        description=(
            "YOLO mode: operate directly in the user's filesystem path "
            "instead of the sandboxed workspace. When enabled, the agent "
            "reads/writes files and runs commands in 'workspace' as a plain "
            "directory without sandbox isolation."
        ),
    )
    model: str | None = Field(
        default=None,
        description="LLM model name (uses provider default if not specified)"
    )
    max_iterations: int = Field(
        default=DEFAULT_MAX_ITERATIONS,
        ge=1,
        le=100,
        description="Maximum tool call iterations (1-100)"
    )
    shell_timeout: int = Field(
        default=DEFAULT_SHELL_TIMEOUT,
        ge=1,
        le=7200,
        description="Default shell command foreground timeout in seconds (1-7200). "
                    "Commands exceeding this are moved to background."
    )
    shell_max_timeout: int = Field(
        default=DEFAULT_SHELL_MAX_TIMEOUT,
        ge=60,
        le=7200,
        description="Maximum allowed per-command timeout override in seconds (60-7200)"
    )
    max_output: int = Field(
        default=DEFAULT_MAX_OUTPUT,
        ge=100,
        le=1000000,
        description="Maximum output characters for shell (100-1000000)"
    )
    session_key: str = Field(
        default="default",
        min_length=1,
        max_length=255,
        description="Session identifier"
    )
    skill_paths: list[Path] = Field(
        default_factory=list,
        description="Additional paths to search for skills"
    )
    mcp_servers: dict[str, MCPServerConfig] = Field(
        default_factory=dict,
        description="MCP server configurations"
    )
    use_spoon_core_skills: bool = Field(
        default=True,
        description="Use spoon-core SkillManager if available"
    )
    context_window: int | None = Field(
        default=None,
        description=(
            "Context window size in tokens. "
            "If None, auto-resolved from model name (default 128K)."
        ),
    )

    # Session persistence
    session_store_backend: str = Field(
        default="file",
        description="Session storage backend: 'file' (JSONL), 'sqlite', or 'postgres'"
    )
    session_store_dsn: str | None = Field(
        default=None,
        description="Database connection string (required for 'postgres' backend)"
    )
    session_store_db_path: str | None = Field(
        default=None,
        description="SQLite database path (for 'sqlite' backend, default: workspace/sessions.db)"
    )

    # Agent pool
    agent_pool_size: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of pre-initialized agent instances in the pool (1-32)"
    )

    # Semantic memory (memsearch)
    memsearch: MemSearchConfig = Field(
        default_factory=MemSearchConfig,
        description="Semantic memory search configuration"
    )

    # Hot-reload
    auto_reload: bool = Field(
        default=False,
        description="Enable background file watching for auto-reload of skills and MCP"
    )
    auto_reload_interval: float = Field(
        default=5.0,
        ge=1.0,
        le=300.0,
        description="Polling interval in seconds for auto-reload file watcher (1-300)"
    )

    @field_validator("workspace", mode="before")
    @classmethod
    def coerce_workspace_path(cls, v: Path | str | None) -> Path:
        """Coerce workspace to Path."""
        if v is None:
            return Path.home() / ".spoon-bot" / "workspace"
        return Path(v) if isinstance(v, str) else v

    @field_validator("skill_paths", mode="before")
    @classmethod
    def coerce_skill_paths(cls, v: list[Path | str] | None) -> list[Path]:
        """Coerce skill paths to list of Path."""
        if v is None:
            return []
        return [Path(p) if isinstance(p, str) else p for p in v]

    @model_validator(mode="after")
    def validate_workspace_accessible(self) -> "AgentLoopConfig":
        """Validate that workspace path can be created and is writable.

        In YOLO mode the directory must already exist; we skip the
        mkdir + write-test since the user is responsible for the path.
        """
        if self.yolo_mode:
            if not self.workspace.is_dir():
                raise ValueError(
                    f"YOLO mode workspace path does not exist: {self.workspace}"
                )
            return self

        try:
            self.workspace.mkdir(parents=True, exist_ok=True)

            test_file = self.workspace / ".write_test"
            try:
                test_file.touch()
                test_file.unlink()
            except (OSError, PermissionError) as e:
                raise ValueError(
                    f"Workspace directory is not writable: {self.workspace} - {e}"
                )
        except (OSError, PermissionError) as e:
            raise ValueError(
                f"Cannot create workspace directory: {self.workspace} - {e}"
            )
        return self


class LLMProviderConfig(BaseModel):
    """Configuration for LLM provider."""

    provider: LLMProviderType = Field(
        default=LLMProviderType.ANTHROPIC,
        description="LLM provider type"
    )
    api_key: str | None = Field(
        default=None,
        description="API key (falls back to environment variable)"
    )
    model: str | None = Field(
        default=None,
        description="Model name"
    )
    base_url: str | None = Field(
        default=None,
        description="Custom API base URL"
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=200000,
        description="Maximum tokens in response"
    )

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: str | None) -> str | None:
        """Validate base URL format if provided."""
        if v is not None:
            v = v.strip().rstrip("/")
            if not v.startswith(("http://", "https://")):
                raise ValueError("Base URL must start with http:// or https://")
        return v

    def get_api_key(self) -> str | None:
        """Get API key from config or environment variable."""
        if self.api_key:
            return self.api_key

        env_var_map = {
            LLMProviderType.ANTHROPIC: "ANTHROPIC_API_KEY",
            LLMProviderType.OPENAI: "OPENAI_API_KEY",
            LLMProviderType.DEEPSEEK: "DEEPSEEK_API_KEY",
            LLMProviderType.GEMINI: "GEMINI_API_KEY",
            LLMProviderType.OPENROUTER: "OPENROUTER_API_KEY",
        }

        env_var = env_var_map.get(self.provider)
        if env_var:
            return os.environ.get(env_var)
        return None


class SpoonBotSettings(BaseSettings):
    """Application-level settings loaded from environment."""

    model_config = ConfigDict(
        env_prefix="SPOON_BOT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Workspace settings
    workspace_path: Path = Field(
        default_factory=lambda: Path.home() / ".spoon-bot" / "workspace",
        description="Default workspace directory"
    )
    yolo_mode: bool = Field(
        default=False,
        description="YOLO mode: operate in user's path instead of sandbox"
    )

    # LLM settings
    default_provider: LLMProviderType = Field(
        default=LLMProviderType.ANTHROPIC,
        description="Default LLM provider"
    )
    default_model: str | None = Field(
        default=None,
        description="Default model name"
    )

    # Agent settings
    max_iterations: int = Field(
        default=DEFAULT_MAX_ITERATIONS,
        ge=1,
        le=100,
        description="Default max iterations"
    )
    shell_timeout: int = Field(
        default=DEFAULT_SHELL_TIMEOUT,
        ge=1,
        le=7200,
        description="Default shell foreground timeout in seconds"
    )
    shell_max_timeout: int = Field(
        default=DEFAULT_SHELL_MAX_TIMEOUT,
        ge=60,
        le=7200,
        description="Maximum allowed per-command timeout override in seconds"
    )
    max_output: int = Field(
        default=DEFAULT_MAX_OUTPUT,
        ge=100,
        le=1000000,
        description="Default max output"
    )

    # Memory settings
    enable_memory: bool = Field(
        default=True,
        description="Enable memory persistence"
    )

    # Logging settings
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )

    @field_validator("workspace_path", mode="before")
    @classmethod
    def coerce_workspace_path(cls, v: Path | str | None) -> Path:
        """Coerce workspace path."""
        if v is None:
            return Path.home() / ".spoon-bot" / "workspace"
        return Path(v) if isinstance(v, str) else v


def validate_mcp_config(config: dict[str, Any]) -> MCPServerConfig:
    """
    Validate an MCP server configuration dictionary.

    Args:
        config: Raw configuration dictionary.

    Returns:
        Validated MCPServerConfig.

    Raises:
        ValueError: If validation fails.
    """
    # Ensure name is present
    if "name" not in config:
        raise ValueError("MCP server config must include 'name' field")
    return MCPServerConfig.model_validate(config)


def validate_mcp_configs(configs: dict[str, dict[str, Any]]) -> dict[str, MCPServerConfig]:
    """
    Validate a dictionary of MCP server configurations.

    Args:
        configs: Dictionary mapping server names to config dicts.

    Returns:
        Dictionary mapping server names to validated MCPServerConfig objects.

    Raises:
        ValueError: If any validation fails.
    """
    validated = {}
    errors = []

    for name, config in configs.items():
        try:
            # Add name to config if not present
            config_with_name = {"name": name, **config}
            validated[name] = MCPServerConfig.model_validate(config_with_name)
        except Exception as e:
            errors.append(f"Server '{name}': {e}")

    if errors:
        raise ValueError(
            "MCP configuration validation failed:\n" + "\n".join(errors)
        )

    return validated


def validate_agent_loop_params(
    workspace: Path | str | None = None,
    model: str | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    shell_timeout: int = DEFAULT_SHELL_TIMEOUT,
    max_output: int = DEFAULT_MAX_OUTPUT,
    session_key: str = "default",
    skill_paths: list[Path | str] | None = None,
    mcp_config: dict[str, dict[str, Any]] | None = None,
    yolo_mode: bool = False,
    shell_max_timeout: int = DEFAULT_SHELL_MAX_TIMEOUT,
) -> AgentLoopConfig:
    """
    Validate AgentLoop initialization parameters.

    Args:
        workspace: Workspace directory path.
        model: LLM model name.
        max_iterations: Maximum tool call iterations.
        shell_timeout: Default shell foreground timeout in seconds.
        max_output: Maximum output characters.
        session_key: Session identifier.
        skill_paths: Additional skill search paths.
        mcp_config: MCP server configurations.
        yolo_mode: If True, operate directly in the user path without sandbox.
        shell_max_timeout: Maximum per-command timeout override in seconds.

    Returns:
        Validated AgentLoopConfig.

    Raises:
        ValueError: If validation fails.
    """
    # Convert mcp_config dict to MCPServerConfig objects
    mcp_servers = {}
    if mcp_config:
        mcp_servers = validate_mcp_configs(mcp_config)

    # In YOLO mode, default workspace to CWD instead of ~/.spoon-bot/workspace
    if yolo_mode and workspace is None:
        workspace = Path.cwd()

    return AgentLoopConfig(
        workspace=workspace,  # type: ignore
        model=model,
        max_iterations=max_iterations,
        shell_timeout=shell_timeout,
        shell_max_timeout=shell_max_timeout,
        max_output=max_output,
        session_key=session_key,
        skill_paths=skill_paths,  # type: ignore
        mcp_servers=mcp_servers,
        yolo_mode=yolo_mode,
    )
