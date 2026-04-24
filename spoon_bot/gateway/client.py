"""Lightweight HTTP client for spoon-bot gateway APIs."""

from __future__ import annotations

from typing import Any

import httpx

from spoon_bot.cron.models import (
    CronExecutionResult,
    CronJob,
    CronJobCreate,
    CronJobPatch,
    CronRunLogEntry,
    CronServiceStatus,
)


class GatewayClientError(RuntimeError):
    """Raised when gateway API calls fail."""


class CronGatewayClient:
    """Minimal client for cron-related gateway endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 15.0,
    ) -> None:
        normalized = base_url.rstrip("/")
        if not normalized.endswith("/v1"):
            normalized = f"{normalized}/v1"
        self._base_url = normalized
        self._api_key = api_key
        self._timeout = timeout

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers: dict[str, str] = {}
        if self._api_key:
            headers["X-API-Key"] = self._api_key

        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                headers=headers,
                timeout=self._timeout,
            ) as client:
                response = await client.request(method, path, json=json_payload)
        except httpx.HTTPError as exc:
            raise GatewayClientError(f"Unable to reach gateway at {self._base_url}: {exc}") from exc

        payload: dict[str, Any]
        try:
            payload = response.json()
        except ValueError:
            payload = {}

        if response.is_success:
            data = payload.get("data")
            if not isinstance(data, dict):
                raise GatewayClientError("Gateway returned an unexpected response body")
            return data

        detail = payload.get("detail")
        if isinstance(detail, dict):
            message = detail.get("message") or detail.get("code") or response.text
        else:
            message = str(detail or response.text or response.reason_phrase)
        raise GatewayClientError(f"Gateway request failed ({response.status_code}): {message}")

    async def status(self) -> CronServiceStatus:
        data = await self._request("GET", "/cron/status")
        return CronServiceStatus.model_validate(data["status"])

    async def list_jobs(self) -> list[CronJob]:
        data = await self._request("GET", "/cron/jobs")
        jobs = data.get("jobs", [])
        return [CronJob.model_validate(job) for job in jobs]

    async def get_job(self, job_id: str) -> CronJob:
        data = await self._request("GET", f"/cron/jobs/{job_id}")
        return CronJob.model_validate(data["job"])

    async def create_job(self, payload: CronJobCreate) -> CronJob:
        data = await self._request(
            "POST",
            "/cron/jobs",
            json_payload=payload.model_dump(mode="json"),
        )
        return CronJob.model_validate(data["job"])

    async def update_job(self, job_id: str, payload: CronJobPatch) -> CronJob:
        data = await self._request(
            "PATCH",
            f"/cron/jobs/{job_id}",
            json_payload=payload.model_dump(mode="json", exclude_none=True),
        )
        return CronJob.model_validate(data["job"])

    async def delete_job(self, job_id: str) -> bool:
        data = await self._request("DELETE", f"/cron/jobs/{job_id}")
        return bool(data.get("deleted"))

    async def run_job(self, job_id: str) -> CronExecutionResult:
        data = await self._request("POST", f"/cron/jobs/{job_id}/run")
        return CronExecutionResult.model_validate(data["result"])

    async def list_runs(self, job_id: str, limit: int = 20) -> list[CronRunLogEntry]:
        data = await self._request("GET", f"/cron/jobs/{job_id}/runs?limit={max(1, limit)}")
        runs = data.get("runs", [])
        return [CronRunLogEntry.model_validate(run) for run in runs]
