"""Request models for the gateway API."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ChatOptions(BaseModel):
    """Options for chat requests."""

    max_iterations: int = Field(default=20, ge=1, le=100)
    stream: bool = False
    model: str | None = None


class ChatRequest(BaseModel):
    """Chat request model."""

    message: str = Field(..., min_length=1, max_length=100000)
    session_key: str = Field(default="default", pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    media: list[str] = Field(default_factory=list, max_length=10)
    options: ChatOptions | None = None

    @field_validator("message")
    @classmethod
    def sanitize_message(cls, v: str) -> str:
        """Remove control characters except newlines."""
        return "".join(c for c in v if c.isprintable() or c in "\n\r\t")


class LoginRequest(BaseModel):
    """Login request model."""

    username: str | None = None
    password: str | None = None
    api_key: str | None = None

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str | None) -> str | None:
        """Validate API key format."""
        if v and not re.match(r"^sk_(live|test|dev)_[A-Za-z0-9_-]{20,}$", v):
            raise ValueError("Invalid API key format")
        return v


class RefreshRequest(BaseModel):
    """Token refresh request model."""

    refresh_token: str = Field(..., min_length=10)


class SessionCreateRequest(BaseModel):
    """Session creation request model."""

    key: str = Field(..., pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    config: dict[str, Any] | None = None


class ToolExecuteRequest(BaseModel):
    """Tool execution request model."""

    arguments: dict[str, Any] = Field(default_factory=dict)


class SkillActivateRequest(BaseModel):
    """Skill activation request model."""

    context: dict[str, Any] | None = None


class MemoryAddRequest(BaseModel):
    """Memory addition request model."""

    content: str = Field(..., min_length=1, max_length=10000)
    tags: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: list[str]) -> list[str]:
        """Validate tag format."""
        for tag in v:
            if not re.match(r"^[a-zA-Z0-9_-]{1,32}$", tag):
                raise ValueError(f"Invalid tag format: {tag}")
        return v


class ConfigUpdateRequest(BaseModel):
    """Configuration update request model."""

    model: str | None = None
    max_iterations: int | None = Field(default=None, ge=1, le=100)
    shell_timeout: int | None = Field(default=None, ge=1, le=3600)
    max_output: int | None = Field(default=None, ge=100, le=1000000)
