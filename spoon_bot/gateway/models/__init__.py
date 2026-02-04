"""Gateway request/response models."""

from spoon_bot.gateway.models.requests import (
    ChatRequest,
    ChatOptions,
    LoginRequest,
    RefreshRequest,
    SessionCreateRequest,
    ToolExecuteRequest,
    SkillActivateRequest,
    MemoryAddRequest,
    ConfigUpdateRequest,
)
from spoon_bot.gateway.models.responses import (
    APIResponse,
    ErrorResponse,
    ChatResponse,
    TokenResponse,
    SessionResponse,
    ToolResponse,
    SkillResponse,
    MemoryResponse,
    StatusResponse,
    HealthResponse,
)

__all__ = [
    # Requests
    "ChatRequest",
    "ChatOptions",
    "LoginRequest",
    "RefreshRequest",
    "SessionCreateRequest",
    "ToolExecuteRequest",
    "SkillActivateRequest",
    "MemoryAddRequest",
    "ConfigUpdateRequest",
    # Responses
    "APIResponse",
    "ErrorResponse",
    "ChatResponse",
    "TokenResponse",
    "SessionResponse",
    "ToolResponse",
    "SkillResponse",
    "MemoryResponse",
    "StatusResponse",
    "HealthResponse",
]
