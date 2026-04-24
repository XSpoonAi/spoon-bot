"""Cron management endpoints."""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, HTTPException, status

from spoon_bot.gateway.app import get_cron_service
from spoon_bot.gateway.auth.dependencies import CurrentUser
from spoon_bot.gateway.models.cron import (
    CronJobCreateRequest,
    CronJobListResponse,
    CronJobPatchRequest,
    CronJobResponse,
    CronRunLogResponse,
    CronRunResponse,
    CronStatusResponse,
)
from spoon_bot.gateway.models.responses import APIResponse, MetaInfo

router = APIRouter()


def _meta() -> MetaInfo:
    return MetaInfo(request_id=f"req_{uuid4().hex[:12]}")


def _service():
    try:
        return get_cron_service()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "CRON_NOT_READY", "message": str(exc)},
        ) from exc


@router.get("/status", response_model=APIResponse[CronStatusResponse])
async def cron_status(user: CurrentUser) -> APIResponse[CronStatusResponse]:
    service = _service()
    status_payload = await service.status()
    return APIResponse(
        success=True,
        data=CronStatusResponse(status=status_payload),
        meta=_meta(),
    )


@router.get("/jobs", response_model=APIResponse[CronJobListResponse])
async def list_jobs(user: CurrentUser) -> APIResponse[CronJobListResponse]:
    service = _service()
    jobs = await service.list_jobs()
    return APIResponse(
        success=True,
        data=CronJobListResponse(jobs=jobs),
        meta=_meta(),
    )


@router.post("/jobs", response_model=APIResponse[CronJobResponse])
async def create_job(
    request: CronJobCreateRequest,
    user: CurrentUser,
) -> APIResponse[CronJobResponse]:
    service = _service()
    try:
        job = await service.create_job(request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "INVALID_CRON_JOB", "message": str(exc)},
        ) from exc
    return APIResponse(success=True, data=CronJobResponse(job=job), meta=_meta())


@router.get("/jobs/{job_id}", response_model=APIResponse[CronJobResponse])
async def get_job(job_id: str, user: CurrentUser) -> APIResponse[CronJobResponse]:
    service = _service()
    try:
        job = await service.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "CRON_JOB_NOT_FOUND", "message": f"Job '{job_id}' not found"},
        ) from exc
    return APIResponse(success=True, data=CronJobResponse(job=job), meta=_meta())


@router.patch("/jobs/{job_id}", response_model=APIResponse[CronJobResponse])
async def update_job(
    job_id: str,
    request: CronJobPatchRequest,
    user: CurrentUser,
) -> APIResponse[CronJobResponse]:
    service = _service()
    try:
        job = await service.update_job(job_id, request)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "CRON_JOB_NOT_FOUND", "message": f"Job '{job_id}' not found"},
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "INVALID_CRON_JOB", "message": str(exc)},
        ) from exc
    return APIResponse(success=True, data=CronJobResponse(job=job), meta=_meta())


@router.delete("/jobs/{job_id}", response_model=APIResponse[dict])
async def delete_job(job_id: str, user: CurrentUser) -> APIResponse[dict]:
    service = _service()
    deleted = await service.delete_job(job_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "CRON_JOB_NOT_FOUND", "message": f"Job '{job_id}' not found"},
        )
    return APIResponse(success=True, data={"deleted": True, "job_id": job_id}, meta=_meta())


@router.post("/jobs/{job_id}/run", response_model=APIResponse[CronRunResponse])
async def run_job(job_id: str, user: CurrentUser) -> APIResponse[CronRunResponse]:
    service = _service()
    try:
        result = await service.run_now(job_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "CRON_JOB_NOT_FOUND", "message": f"Job '{job_id}' not found"},
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "CRON_JOB_RUNNING", "message": str(exc)},
        ) from exc
    return APIResponse(success=True, data=CronRunResponse(result=result), meta=_meta())


@router.get("/jobs/{job_id}/runs", response_model=APIResponse[CronRunLogResponse])
async def get_runs(
    job_id: str,
    user: CurrentUser,
    limit: int = 20,
) -> APIResponse[CronRunLogResponse]:
    service = _service()
    try:
        runs = await service.get_runs(job_id, limit=max(1, limit))
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "CRON_JOB_NOT_FOUND", "message": f"Job '{job_id}' not found"},
        ) from exc
    return APIResponse(success=True, data=CronRunLogResponse(runs=runs), meta=_meta())
