"""Typed models for scheduled tasks."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""
    return datetime.now(timezone.utc)


def ensure_utc_datetime(value: datetime) -> datetime:
    """Normalize datetimes to aware UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class CronDeliveryTarget(BaseModel):
    """Explicit outbound delivery target."""

    model_config = ConfigDict(extra="forbid")

    channel: str = Field(..., min_length=1)
    account_id: str | None = None
    target: dict[str, str] = Field(default_factory=dict)
    session_key: str | None = None

    @field_validator("target", mode="before")
    @classmethod
    def normalize_target(cls, value: Any) -> dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("target must be an object")
        normalized: dict[str, str] = {}
        for key, raw in value.items():
            if raw is None:
                continue
            text = str(raw).strip()
            if text:
                normalized[str(key)] = text
        return normalized


class CronConversationScope(BaseModel):
    """Stable ownership scope for chat-created jobs."""

    model_config = ConfigDict(extra="forbid")

    channel: str = Field(..., min_length=1)
    account_id: str | None = None
    conversation_id: str = Field(..., min_length=1)
    thread_id: str | None = None
    session_key: str | None = None

    @field_validator("channel", "conversation_id", "thread_id", "session_key", mode="before")
    @classmethod
    def normalize_scope_strings(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("account_id", mode="before")
    @classmethod
    def normalize_account_id(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


class AtSchedule(BaseModel):
    """Run once at an absolute time."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["at"] = "at"
    run_at: datetime

    @field_validator("run_at", mode="after")
    @classmethod
    def normalize_run_at(cls, value: datetime) -> datetime:
        return ensure_utc_datetime(value)


class EverySchedule(BaseModel):
    """Run repeatedly at a fixed interval."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["every"] = "every"
    seconds: int = Field(..., ge=1)
    anchor_at: datetime | None = None

    @field_validator("anchor_at", mode="after")
    @classmethod
    def normalize_anchor(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return ensure_utc_datetime(value)


class CronExpressionSchedule(BaseModel):
    """Run on a cron expression."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["cron"] = "cron"
    expression: str = Field(..., min_length=1)
    timezone: str | None = None

    @field_validator("expression")
    @classmethod
    def normalize_expression(cls, value: str) -> str:
        return " ".join(part for part in value.split() if part)


CronSchedule = Annotated[
    AtSchedule | EverySchedule | CronExpressionSchedule,
    Field(discriminator="kind"),
]


class CronJobState(BaseModel):
    """Mutable execution state for a job."""

    model_config = ConfigDict(extra="forbid")

    next_run_at: datetime | None = None
    running: bool = False
    running_at: datetime | None = None
    last_run_at: datetime | None = None
    last_status: Literal["never", "running", "success", "error"] = "never"
    last_error: str | None = None
    last_result: str | None = None

    @field_validator("next_run_at", "running_at", "last_run_at", mode="after")
    @classmethod
    def normalize_dt(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return ensure_utc_datetime(value)


class CronJobCreate(BaseModel):
    """Input model for creating a job."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=120)
    prompt: str = Field(..., min_length=1, max_length=20000)
    schedule: CronSchedule
    target_mode: Literal["session", "current", "main", "isolated"] = "session"
    session_key: str | None = Field(default=None, max_length=255)
    delivery: CronDeliveryTarget | None = None
    conversation_scope: CronConversationScope | None = None
    delivery_mode: Literal["announce", "none"] = "announce"
    allowed_tools: list[str] | None = None
    max_attempts: int = Field(default=1, ge=1, le=10)
    backoff_seconds: int = Field(default=0, ge=0, le=3600)
    enabled: bool = True

    @field_validator("allowed_tools", mode="before")
    @classmethod
    def normalize_allowed_tools(cls, value: Any) -> list[str] | None:
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError("allowed_tools must be a list")
        normalized: list[str] = []
        seen: set[str] = set()
        for raw in value:
            text = str(raw).strip()
            if not text or text in seen:
                continue
            normalized.append(text)
            seen.add(text)
        return normalized or None

    @model_validator(mode="after")
    def validate_session_mode(self) -> "CronJobCreate":
        if self.target_mode in {"session", "current"} and not self.session_key:
            raise ValueError(
                "session_key is required when target_mode is 'session' or 'current'"
            )
        if self.target_mode in {"main", "isolated"} and self.session_key:
            self.session_key = None
        if self.delivery_mode == "none":
            self.delivery = None
        return self


class CronJobPatch(BaseModel):
    """Partial update model for a job."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=120)
    prompt: str | None = Field(default=None, min_length=1, max_length=20000)
    schedule: CronSchedule | None = None
    target_mode: Literal["session", "current", "main", "isolated"] | None = None
    session_key: str | None = Field(default=None, max_length=255)
    delivery: CronDeliveryTarget | None = None
    conversation_scope: CronConversationScope | None = None
    delivery_mode: Literal["announce", "none"] | None = None
    allowed_tools: list[str] | None = None
    max_attempts: int | None = Field(default=None, ge=1, le=10)
    backoff_seconds: int | None = Field(default=None, ge=0, le=3600)
    enabled: bool | None = None

    @field_validator("allowed_tools", mode="before")
    @classmethod
    def normalize_patch_allowed_tools(cls, value: Any) -> list[str] | None:
        return CronJobCreate.normalize_allowed_tools(value)


class CronJob(CronJobCreate):
    """Persisted scheduled job."""

    id: str = Field(default_factory=lambda: f"cron_{uuid4().hex[:12]}")
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    state: CronJobState = Field(default_factory=CronJobState)

    @field_validator("created_at", "updated_at", mode="after")
    @classmethod
    def normalize_created_updated(cls, value: datetime) -> datetime:
        return ensure_utc_datetime(value)


class CronExecutionResult(BaseModel):
    """Outcome of a single cron execution."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "error"]
    session_key: str
    output: str | None = None
    error: str | None = None
    delivered: bool = False
    delivery_status: str | None = None
    attempts: int = 1


class CronRunLogEntry(BaseModel):
    """Persisted execution log record."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(default_factory=lambda: f"run_{uuid4().hex[:12]}")
    job_id: str
    status: Literal["success", "error"]
    session_key: str
    started_at: datetime = Field(default_factory=utc_now)
    ended_at: datetime = Field(default_factory=utc_now)
    error: str | None = None
    delivery_status: str | None = None
    output_excerpt: str | None = None
    attempts: int = 1

    @field_validator("started_at", "ended_at", mode="after")
    @classmethod
    def normalize_log_dt(cls, value: datetime) -> datetime:
        return ensure_utc_datetime(value)


class CronServiceStatus(BaseModel):
    """Aggregate cron service health and queue information."""

    model_config = ConfigDict(extra="forbid")

    running: bool
    total_jobs: int
    enabled_jobs: int
    active_runs: int
    next_run_at: datetime | None = None
    store_path: str

    @field_validator("next_run_at", mode="after")
    @classmethod
    def normalize_status_dt(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return ensure_utc_datetime(value)
