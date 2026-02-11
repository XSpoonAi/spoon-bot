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


class AgentLoopConfig(BaseModel):
    """Configuration for AgentLoop with validation."""

    model_config = ConfigDict(validate_default=True)

    workspace: Path = Field(
        default_factory=lambda: Path.home() / ".spoon-bot" / "workspace",
        description="Workspace directory path"
    )
    model: str | None = Field(
        default=None,
        description="LLM model name (uses provider default if not specified)"
    )
    max_iterations: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum tool call iterations (1-100)"
    )
    shell_timeout: int = Field(
        default=60,
        ge=1,
        le=3600,
        description="Shell command timeout in seconds (1-3600)"
    )
    max_output: int = Field(
        default=10000,
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
        """Validate that workspace path can be created and is writable."""
        try:
            # Try to create the workspace directory
            self.workspace.mkdir(parents=True, exist_ok=True)

            # Check if it's writable by creating a test file
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
        default=20,
        ge=1,
        le=100,
        description="Default max iterations"
    )
    shell_timeout: int = Field(
        default=60,
        ge=1,
        le=3600,
        description="Default shell timeout"
    )
    max_output: int = Field(
        default=10000,
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
    max_iterations: int = 20,
    shell_timeout: int = 60,
    max_output: int = 10000,
    session_key: str = "default",
    skill_paths: list[Path | str] | None = None,
    mcp_config: dict[str, dict[str, Any]] | None = None,
) -> AgentLoopConfig:
    """
    Validate AgentLoop initialization parameters.

    Args:
        workspace: Workspace directory path.
        model: LLM model name.
        max_iterations: Maximum tool call iterations.
        shell_timeout: Shell command timeout.
        max_output: Maximum output characters.
        session_key: Session identifier.
        skill_paths: Additional skill search paths.
        mcp_config: MCP server configurations.

    Returns:
        Validated AgentLoopConfig.

    Raises:
        ValueError: If validation fails.
    """
    # Convert mcp_config dict to MCPServerConfig objects
    mcp_servers = {}
    if mcp_config:
        mcp_servers = validate_mcp_configs(mcp_config)

    return AgentLoopConfig(
        workspace=workspace,  # type: ignore
        model=model,
        max_iterations=max_iterations,
        shell_timeout=shell_timeout,
        max_output=max_output,
        session_key=session_key,
        skill_paths=skill_paths,  # type: ignore
        mcp_servers=mcp_servers,
    )
