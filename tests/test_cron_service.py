from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from spoon_bot.cron.models import (
    CronExecutionResult,
    CronJob,
    CronJobCreate,
    CronJobPatch,
    CronRunLogEntry,
    EverySchedule,
)
from spoon_bot.cron.run_log import CronRunLog
from spoon_bot.cron.service import CronService
from spoon_bot.cron.store import CronStore


class FakeExecutor:
    def __init__(self) -> None:
        self.jobs = []
        self.closed = False

    async def execute(self, job):
        self.jobs.append(job)
        return CronExecutionResult(
            status="success",
            session_key=job.session_key or f"cron_{job.id}",
            output=f"completed {job.name}",
            delivered=False,
            delivery_status="skipped",
        )

    async def close(self) -> None:
        self.closed = True


class BlockingExecutor(FakeExecutor):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def execute(self, job):
        self.jobs.append(job)
        self.started.set()
        await self.release.wait()
        return CronExecutionResult(
            status="success",
            session_key=job.session_key or f"cron_{job.id}",
            output="released",
            delivered=False,
            delivery_status="skipped",
        )


class FlakyExecutor(FakeExecutor):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def execute(self, job):
        self.jobs.append(job)
        self.calls += 1
        if self.calls == 1:
            return CronExecutionResult(
                status="error",
                session_key=job.session_key or f"cron_{job.id}",
                error="Temporary network timeout",
                delivered=False,
                delivery_status="skipped",
            )
        return CronExecutionResult(
            status="success",
            session_key=job.session_key or f"cron_{job.id}",
            output="completed after retry",
            delivered=False,
            delivery_status="skipped",
        )


def _service(tmp_path, executor: FakeExecutor | None = None) -> CronService:
    return CronService(
        CronStore(tmp_path / "jobs.json"),
        run_log=CronRunLog(tmp_path / "runs"),
        executor=executor or FakeExecutor(),
    )


@pytest.mark.asyncio
async def test_cron_service_create_status_run_and_get_runs(tmp_path):
    service = _service(tmp_path)
    job = await service.create_job(
        CronJobCreate(
            name="heartbeat",
            prompt="say hello",
            schedule=EverySchedule(seconds=60),
            target_mode="isolated",
        )
    )

    assert isinstance(job.schedule, EverySchedule)
    assert job.schedule.anchor_at is not None

    status = await service.status()
    assert status.total_jobs == 1
    assert status.enabled_jobs == 1
    assert status.next_run_at is not None

    result = await service.run_now(job.id)
    assert result.status == "success"
    assert result.output == "completed heartbeat"

    persisted = await service.get_job(job.id)
    assert persisted.state.last_status == "success"
    assert persisted.state.last_result == "completed heartbeat"
    assert persisted.state.last_run_at is not None

    runs = await service.get_runs(job.id)
    assert len(runs) == 1
    assert runs[0].job_id == job.id
    assert runs[0].status == "success"
    assert runs[0].output_excerpt == "completed heartbeat"


@pytest.mark.asyncio
async def test_cron_service_update_and_delete_job(tmp_path):
    service = _service(tmp_path)
    job = await service.create_job(
        CronJobCreate(
            name="report",
            prompt="draft report",
            schedule=EverySchedule(seconds=300),
            target_mode="isolated",
        )
    )

    updated = await service.update_job(
        job.id,
        CronJobPatch(
            prompt="draft daily report",
            enabled=False,
        ),
    )
    assert updated.prompt == "draft daily report"
    assert updated.enabled is False
    assert updated.state.next_run_at is None

    assert await service.delete_job(job.id) is True
    assert await service.delete_job(job.id) is False


@pytest.mark.asyncio
async def test_cron_service_run_now_conflict(tmp_path):
    executor = BlockingExecutor()
    service = _service(tmp_path, executor=executor)
    job = await service.create_job(
        CronJobCreate(
            name="conflict",
            prompt="wait",
            schedule=EverySchedule(seconds=120),
            target_mode="isolated",
        )
    )

    first_run = asyncio.create_task(service.run_now(job.id))
    await asyncio.wait_for(executor.started.wait(), timeout=1.0)

    with pytest.raises(RuntimeError, match="already running"):
        await service.run_now(job.id)

    executor.release.set()
    await asyncio.wait_for(first_run, timeout=1.0)


@pytest.mark.asyncio
async def test_cron_service_get_runs_returns_newest_first(tmp_path):
    service = _service(tmp_path)
    job = await service.create_job(
        CronJobCreate(
            name="history",
            prompt="snapshot",
            schedule=EverySchedule(seconds=60),
            target_mode="isolated",
        )
    )

    # Seed two explicit log entries with increasing timestamps.
    assert service._run_log is not None
    service._run_log.append(
        CronRunLogEntry(
            job_id=job.id,
            status="success",
            session_key="session-1",
            started_at=datetime(2026, 3, 28, 8, 0, tzinfo=timezone.utc),
            ended_at=datetime(2026, 3, 28, 8, 1, tzinfo=timezone.utc),
            output_excerpt="first",
        )
    )
    service._run_log.append(
        CronRunLogEntry(
            job_id=job.id,
            status="success",
            session_key="session-2",
            started_at=datetime(2026, 3, 28, 9, 0, tzinfo=timezone.utc),
            ended_at=datetime(2026, 3, 28, 9, 1, tzinfo=timezone.utc),
            output_excerpt="second",
        )
    )

    runs = await service.get_runs(job.id, limit=2)

    assert [run.output_excerpt for run in runs] == ["second", "first"]


@pytest.mark.asyncio
async def test_cron_service_get_runs_missing_job_raises_keyerror(tmp_path):
    service = _service(tmp_path)

    with pytest.raises(KeyError):
        await service.get_runs("cron_missing")


@pytest.mark.asyncio
async def test_cron_service_retries_transient_failures(tmp_path):
    executor = FlakyExecutor()
    service = _service(tmp_path, executor=executor)
    job = await service.create_job(
        CronJobCreate(
            name="weather",
            prompt="Check the weather and report back",
            schedule=EverySchedule(seconds=60),
            target_mode="isolated",
            max_attempts=3,
            backoff_seconds=0,
        )
    )

    result = await service.run_now(job.id)

    assert result.status == "success"
    assert result.output == "completed after retry"
    assert result.attempts == 2
    assert executor.calls == 2

    runs = await service.get_runs(job.id)
    assert len(runs) == 1
    assert runs[0].attempts == 2


@pytest.mark.asyncio
async def test_cron_service_backfills_legacy_conversation_scope_and_persists_it(tmp_path):
    store = CronStore(tmp_path / "jobs.json")
    legacy_job = CronJob(
        name="legacy",
        prompt="summarize updates",
        schedule=EverySchedule(seconds=300),
        target_mode="isolated",
        delivery={
            "channel": "telegram",
            "account_id": "spoon_bot",
            "target": {"chat_id": "123"},
            "session_key": "telegram_spoon_bot_123",
        },
    )
    store.save_jobs([legacy_job])

    service = CronService(
        store,
        run_log=CronRunLog(tmp_path / "runs"),
        executor=FakeExecutor(),
    )
    jobs = await service.list_jobs()

    assert len(jobs) == 1
    assert jobs[0].delivery is not None
    assert jobs[0].delivery.channel == "telegram:spoon_bot"
    assert jobs[0].conversation_scope is not None
    assert jobs[0].conversation_scope.channel == "telegram"
    assert jobs[0].conversation_scope.account_id == "spoon_bot"
    assert jobs[0].conversation_scope.conversation_id == "123"

    reloaded = store.load_jobs()
    assert len(reloaded) == 1
    assert reloaded[0].delivery is not None
    assert reloaded[0].delivery.channel == "telegram:spoon_bot"
    assert reloaded[0].conversation_scope is not None
    assert reloaded[0].conversation_scope.conversation_id == "123"
