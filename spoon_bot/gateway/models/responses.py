"""Response models for the gateway API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class MetaInfo(BaseModel):
    """Response metadata."""

    request_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    duration_ms: int | None = None


class ErrorDetail(BaseModel):
    """Error detail model."""

    code: str
    message: str
    details: dict[str, Any] | None = None
    help_url: str | None = None


class APIResponse(BaseModel, Generic[T]):
    """Standard API response wrapper."""

    success: bool = True
    data: T | None = None
    meta: MetaInfo


class ErrorResponse(BaseModel):
    """Error response model."""

    success: bool = False
    error: ErrorDetail
    meta: MetaInfo


class TokenResponse(BaseModel):
    """Token response for authentication."""

    access_token: str
    refresh_token: str | None = None
    token_type: str = "bearer"
    expires_in: int  # seconds


class UsageInfo(BaseModel):
    """Token usage information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ToolCallInfo(BaseModel):
    """Tool call information."""

    id: str
    name: str
    arguments: dict[str, Any]
    result: str | None = None


class ChatResponse(BaseModel):
    """Chat response model."""

    response: str
    tool_calls: list[ToolCallInfo] = Field(default_factory=list)
    usage: UsageInfo | None = None


class SessionInfo(BaseModel):
    """Session information."""

    key: str
    created_at: datetime
    message_count: int = 0
    config: dict[str, Any] = Field(default_factory=dict)


class SessionResponse(BaseModel):
    """Session response model."""

    session: SessionInfo


class SessionListResponse(BaseModel):
    """Session list response model."""

    sessions: list[SessionInfo]


class ToolInfo(BaseModel):
    """Tool information."""

    name: str
    description: str
    parameters: dict[str, Any]


class ToolResponse(BaseModel):
    """Tool execution response."""

    result: str
    success: bool = True


class ToolListResponse(BaseModel):
    """Tool list response."""

    tools: list[ToolInfo]


class SkillInfo(BaseModel):
    """Skill information."""

    name: str
    description: str
    active: bool = False
    triggers: list[str] = Field(default_factory=list)


class SkillResponse(BaseModel):
    """Skill operation response."""

    activated: bool | None = None
    deactivated: bool | None = None
    skill: SkillInfo | None = None


class SkillListResponse(BaseModel):
    """Skill list response."""

    skills: list[SkillInfo]


class MemoryResult(BaseModel):
    """Memory search result."""

    id: str
    content: str
    tags: list[str] = Field(default_factory=list)
    score: float | None = None
    created_at: datetime | None = None


class MemoryResponse(BaseModel):
    """Memory operation response."""

    id: str | None = None
    created: bool | None = None
    context: str | None = None
    results: list[MemoryResult] | None = None


class AgentStats(BaseModel):
    """Agent statistics."""

    total_requests: int = 0
    active_sessions: int = 0
    tools_available: int = 0
    skills_loaded: int = 0


class StatusResponse(BaseModel):
    """Agent status response."""

    status: str  # "ready", "busy", "error"
    current_task: str | None = None
    uptime: int  # seconds
    stats: AgentStats


class HealthCheck(BaseModel):
    """Health check item."""

    name: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str  # "healthy", "degraded", "unhealthy"
    version: str
    uptime: int
    checks: list[HealthCheck] = Field(default_factory=list)
