from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import spoon_bot.gateway.app as app_module
from spoon_bot.cron.models import CronExecutionResult
from spoon_bot.cron.run_log import CronRunLog
from spoon_bot.cron.service import CronService
from spoon_bot.cron.store import CronStore
from spoon_bot.gateway.app import create_app, set_cron_service
from spoon_bot.gateway.config import GatewayConfig


class FakeExecutor:
    async def execute(self, job):
        return CronExecutionResult(
            status="success",
            session_key=job.session_key or f"cron_{job.id}",
            output=f"ran {job.name}",
            delivered=False,
            delivery_status="skipped",
        )

    async def close(self) -> None:
        return None


@pytest.fixture
def client(tmp_path):
    app_module._auth_required = False
    app = create_app(GatewayConfig.from_env())
    service = CronService(
        CronStore(tmp_path / "jobs.json"),
        run_log=CronRunLog(tmp_path / "runs"),
        executor=FakeExecutor(),
    )
    set_cron_service(service)

    with TestClient(app) as test_client:
        yield test_client

    set_cron_service(None)


def test_cron_gateway_create_run_and_list_runs(client):
    create_response = client.post(
        "/v1/cron/jobs",
        json={
            "name": "gateway-job",
            "prompt": "say hi",
            "schedule": {"kind": "every", "seconds": 60},
            "target_mode": "isolated",
            "enabled": True,
        },
    )
    assert create_response.status_code == 200
    job = create_response.json()["data"]["job"]

    status_response = client.get("/v1/cron/status")
    assert status_response.status_code == 200
    assert status_response.json()["data"]["status"]["total_jobs"] == 1

    list_response = client.get("/v1/cron/jobs")
    assert list_response.status_code == 200
    assert len(list_response.json()["data"]["jobs"]) == 1

    run_response = client.post(f"/v1/cron/jobs/{job['id']}/run")
    assert run_response.status_code == 200
    assert run_response.json()["data"]["result"]["status"] == "success"

    runs_response = client.get(f"/v1/cron/jobs/{job['id']}/runs?limit=5")
    assert runs_response.status_code == 200
    runs = runs_response.json()["data"]["runs"]
    assert len(runs) == 1
    assert runs[0]["job_id"] == job["id"]
    assert runs[0]["output_excerpt"] == "ran gateway-job"


def test_cron_gateway_runs_returns_404_for_missing_job(client):
    response = client.get("/v1/cron/jobs/cron_missing/runs")

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "CRON_JOB_NOT_FOUND"
