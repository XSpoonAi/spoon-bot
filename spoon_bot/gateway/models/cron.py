"""Gateway models for cron endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field

from spoon_bot.cron.models import (
    CronExecutionResult,
    CronJob,
    CronJobCreate,
    CronJobPatch,
    CronRunLogEntry,
    CronServiceStatus,
)


class CronJobCreateRequest(CronJobCreate):
    """Gateway request model for job creation."""


class CronJobPatchRequest(CronJobPatch):
    """Gateway request model for job updates."""


class CronJobResponse(BaseModel):
    """Single job response payload."""

    job: CronJob


class CronJobListResponse(BaseModel):
    """Multiple jobs response payload."""

    jobs: list[CronJob] = Field(default_factory=list)


class CronStatusResponse(BaseModel):
    """Cron status response payload."""

    status: CronServiceStatus


class CronRunResponse(BaseModel):
    """Immediate run response payload."""

    result: CronExecutionResult


class CronRunLogResponse(BaseModel):
    """Run log tail response payload."""

    runs: list[CronRunLogEntry] = Field(default_factory=list)
