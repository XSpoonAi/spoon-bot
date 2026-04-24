"""Gateway request/response models."""

from spoon_bot.gateway.models.cron import (
    CronJobCreateRequest,
    CronJobListResponse,
    CronJobPatchRequest,
    CronJobResponse,
    CronRunLogResponse,
    CronRunResponse,
    CronStatusResponse,
)
from spoon_bot.gateway.models.requests import (
    ChatOptions,
    ChatRequest,
    ConfigUpdateRequest,
    LoginRequest,
    MemoryAddRequest,
    RefreshRequest,
    SessionCreateRequest,
    SkillActivateRequest,
    ToolExecuteRequest,
)
from spoon_bot.gateway.models.responses import (
    APIResponse,
    ChatResponse,
    ResponseSource,
    ErrorResponse,
    HealthResponse,
    MemoryResponse,
    SessionResponse,
    SkillResponse,
    StatusResponse,
    StreamChunk,
    TokenResponse,
    ToolResponse,
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
    "CronJobCreateRequest",
    "CronJobPatchRequest",
    # Responses
    "APIResponse",
    "ErrorResponse",
    "ChatResponse",
    "ResponseSource",
    "StreamChunk",
    "TokenResponse",
    "SessionResponse",
    "ToolResponse",
    "SkillResponse",
    "MemoryResponse",
    "StatusResponse",
    "HealthResponse",
    "CronJobResponse",
    "CronJobListResponse",
    "CronStatusResponse",
    "CronRunResponse",
    "CronRunLogResponse",
]
