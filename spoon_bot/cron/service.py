"""Cron service lifecycle and CRUD operations."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

from loguru import logger

from spoon_bot.agent.loop import AgentLoop
from spoon_bot.channels.delivery import (
    ChannelDeliveryService,
    binding_from_session_key,
    conversation_scope_from_parts,
    conversation_scope_from_session_key,
    normalize_channel_name,
)
from spoon_bot.config import CronConfig
from spoon_bot.cron.executor import CronExecutor
from spoon_bot.cron.models import (
    CronExecutionResult,
    CronJob,
    CronJobCreate,
    CronJobPatch,
    CronRunLogEntry,
    CronServiceStatus,
    EverySchedule,
    utc_now,
)
from spoon_bot.cron.run_log import CronRunLog
from spoon_bot.cron.schedule import compute_next_run, is_due, validate_schedule
from spoon_bot.cron.store import CronStore
from spoon_bot.runtime.execution import ExecutionCoordinator


class CronService:
    """Persistent scheduled task manager."""

    def __init__(
        self,
        store: CronStore,
        *,
        run_log: CronRunLog | None = None,
        executor: CronExecutor | None = None,
        catch_up_on_start: bool = True,
        poll_interval: float = 1.0,
        max_concurrent_runs: int = 1,
    ) -> None:
        self._store = store
        self._run_log = run_log
        self._executor = executor
        self._catch_up_on_start = catch_up_on_start
        self._poll_interval = max(0.2, float(poll_interval))
        self._semaphore = asyncio.Semaphore(max(1, int(max_concurrent_runs)))
        self._jobs: dict[str, CronJob] = {}
        self._loaded = False
        self._running = False
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._active_runs: dict[str, asyncio.Task] = {}

    async def start(self) -> None:
        """Load jobs and start the polling loop."""
        async with self._lock:
            await self._ensure_loaded_locked()
            if self._running:
                return

            now = utc_now()
            changed = False
            for job in self._jobs.values():
                if job.state.running:
                    job.state.running = False
                    job.state.running_at = None
                    job.state.last_status = "error"
                    job.state.last_error = "Recovered stale running job on startup"
                    changed = True
                if job.enabled and job.state.next_run_at is None:
                    job.state.next_run_at = self._compute_next_run(job, now=now)
                    changed = True
                if job.enabled and not self._catch_up_on_start and is_due(job.state.next_run_at, now=now):
                    job.state.next_run_at = self._compute_next_run(job, now=now, after=now)
                    changed = True

            if changed:
                self._persist_jobs_locked()

            self._running = True
            self._task = asyncio.create_task(self._run_loop(), name="cron-service")

        logger.info("Cron service started")

    async def stop(self) -> None:
        """Stop polling and clean up background resources."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        if self._active_runs:
            await asyncio.gather(*self._active_runs.values(), return_exceptions=True)
            self._active_runs.clear()

        if self._executor is not None:
            await self._executor.close()

        logger.info("Cron service stopped")

    async def status(self) -> CronServiceStatus:
        """Return current service health information."""
        async with self._lock:
            await self._ensure_loaded_locked()
            next_run = min(
                (
                    job.state.next_run_at
                    for job in self._jobs.values()
                    if job.enabled and job.state.next_run_at is not None
                ),
                default=None,
            )
            return CronServiceStatus(
                running=self._running,
                total_jobs=len(self._jobs),
                enabled_jobs=sum(1 for job in self._jobs.values() if job.enabled),
                active_runs=len(self._active_runs),
                next_run_at=next_run,
                store_path=str(self._store.path),
            )

    async def list_jobs(self) -> list[CronJob]:
        """Return all jobs sorted by creation time."""
        async with self._lock:
            await self._ensure_loaded_locked()
            return sorted(self._jobs.values(), key=lambda job: (job.created_at, job.id))

    async def get_job(self, job_id: str) -> CronJob:
        """Return one job or raise KeyError."""
        async with self._lock:
            await self._ensure_loaded_locked()
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return job

    async def get_runs(self, job_id: str, limit: int = 20) -> list[CronRunLogEntry]:
        """Return recent run log entries for one job, newest first."""
        async with self._lock:
            await self._ensure_loaded_locked()
            if job_id not in self._jobs:
                raise KeyError(job_id)

        if self._run_log is None:
            return []

        runs = self._run_log.tail(job_id, limit=max(1, limit))
        runs.reverse()
        return runs

    async def create_job(self, payload: CronJobCreate) -> CronJob:
        """Create and persist a new job."""
        validate_schedule(payload.schedule)
        now = utc_now()
        job = CronJob(**payload.model_dump(mode="python"))
        self._normalize_job_delivery(job)
        self._ensure_job_conversation_scope(job)
        if isinstance(job.schedule, EverySchedule) and job.schedule.anchor_at is None:
            job.schedule.anchor_at = now
        job.created_at = now
        job.updated_at = now
        job.state.next_run_at = self._compute_next_run(job, now=now)
        if job.enabled and job.state.next_run_at is None:
            job.enabled = False

        async with self._lock:
            await self._ensure_loaded_locked()
            self._jobs[job.id] = job
            self._persist_jobs_locked()
            return job

    async def update_job(self, job_id: str, patch: CronJobPatch) -> CronJob:
        """Update a persisted job."""
        async with self._lock:
            await self._ensure_loaded_locked()
            current = self._jobs.get(job_id)
            if current is None:
                raise KeyError(job_id)

            merged = current.model_dump(mode="python")
            updates = patch.model_dump(exclude_unset=True, mode="python")
            merged.update(updates)
            merged["id"] = current.id
            merged["created_at"] = current.created_at
            merged["state"] = current.state

            if merged["target_mode"] in {"session", "current"} and not merged.get("session_key"):
                raise ValueError(
                    "session_key is required when target_mode is 'session' or 'current'"
                )
            if merged["target_mode"] in {"main", "isolated"}:
                merged["session_key"] = None

            updated = CronJob(**merged)
            self._normalize_job_delivery(updated)
            self._ensure_job_conversation_scope(updated)
            validate_schedule(updated.schedule)
            if isinstance(updated.schedule, EverySchedule) and updated.schedule.anchor_at is None:
                updated.schedule.anchor_at = utc_now()
            updated.updated_at = utc_now()

            if updated.enabled:
                updated.state.next_run_at = self._compute_next_run(updated, now=updated.updated_at)
                if updated.state.next_run_at is None:
                    updated.enabled = False
            else:
                updated.state.next_run_at = None

            self._jobs[job_id] = updated
            self._persist_jobs_locked()
            return updated

    async def delete_job(self, job_id: str) -> bool:
        """Delete a job."""
        async with self._lock:
            await self._ensure_loaded_locked()
            removed = self._jobs.pop(job_id, None)
            if removed is None:
                return False
            self._persist_jobs_locked()
            return True

    async def run_now(self, job_id: str) -> CronExecutionResult:
        """Execute a job immediately and wait for the result."""
        async with self._lock:
            await self._ensure_loaded_locked()
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            existing = self._active_runs.get(job_id)
            if existing is not None and not existing.done():
                raise RuntimeError(f"Cron job is already running: {job_id}")

        task = asyncio.create_task(self._execute_job(job.id), name=f"cron-run-manual-{job.id}")
        self._active_runs[job.id] = task
        try:
            return await task
        finally:
            self._active_runs.pop(job.id, None)

    async def _run_loop(self) -> None:
        while self._running:
            try:
                due_ids = await self._collect_due_job_ids()
                for job_id in due_ids:
                    if job_id in self._active_runs:
                        continue
                    task = asyncio.create_task(self._execute_job(job_id), name=f"cron-run-{job_id}")
                    self._active_runs[job_id] = task
                    task.add_done_callback(
                        lambda _, current_job_id=job_id: self._active_runs.pop(current_job_id, None)
                    )
                await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Cron loop error: {exc}")
                await asyncio.sleep(self._poll_interval)

    async def _collect_due_job_ids(self) -> list[str]:
        async with self._lock:
            await self._ensure_loaded_locked()
            now = utc_now()
            return [
                job.id
                for job in self._jobs.values()
                if job.enabled
                and not job.state.running
                and is_due(job.state.next_run_at, now=now)
            ]

    async def _execute_job(self, job_id: str) -> CronExecutionResult:
        if self._executor is None:
            raise RuntimeError("Cron executor is not configured")

        async with self._semaphore:
            started_at = utc_now()

            async with self._lock:
                await self._ensure_loaded_locked()
                job = self._jobs.get(job_id)
                if job is None:
                    raise KeyError(job_id)
                job.state.running = True
                job.state.running_at = started_at
                job.state.last_status = "running"
                job.state.last_error = None
                job.updated_at = started_at
                self._persist_jobs_locked()

            result = await self._execute_with_retries(job)
            finished_at = utc_now()

            async with self._lock:
                await self._ensure_loaded_locked()
                current = self._jobs.get(job_id)
                if current is None:
                    raise KeyError(job_id)

                current.state.running = False
                current.state.running_at = None
                current.state.last_run_at = finished_at
                current.state.last_status = result.status
                current.state.last_error = result.error
                current.state.last_result = (result.output or "")[:4000] or None
                current.updated_at = finished_at

                if current.enabled:
                    current.state.next_run_at = self._compute_next_run(
                        current,
                        now=finished_at,
                        after=current.state.next_run_at or finished_at,
                    )
                    if current.state.next_run_at is None:
                        current.enabled = False
                self._persist_jobs_locked()

            if self._run_log is not None:
                self._run_log.append(
                    CronRunLogEntry(
                        job_id=job_id,
                        status=result.status,
                        session_key=result.session_key,
                        started_at=started_at,
                        ended_at=finished_at,
                        error=result.error,
                        delivery_status=result.delivery_status,
                        output_excerpt=(result.output or "")[:1000] or None,
                        attempts=result.attempts,
                    )
                )

            return result

    async def _execute_with_retries(self, job: CronJob) -> CronExecutionResult:
        max_attempts = max(1, int(job.max_attempts))
        backoff_seconds = max(0, int(job.backoff_seconds))
        result: CronExecutionResult | None = None

        for attempt in range(1, max_attempts + 1):
            result = await self._executor.execute(job)
            result.attempts = attempt
            if result.status == "success":
                return result
            if attempt >= max_attempts or not self._is_retryable_error(result.error):
                return result
            if backoff_seconds > 0:
                logger.info(
                    f"Retrying cron job {job.id} after attempt {attempt}/{max_attempts} "
                    f"in {backoff_seconds}s"
                )
                await asyncio.sleep(backoff_seconds)

        assert result is not None  # pragma: no cover - defensive guard
        return result

    @staticmethod
    def _is_retryable_error(error: str | None) -> bool:
        if not error:
            return False
        text = error.lower()
        transient_markers = (
            "timeout",
            "timed out",
            "temporary",
            "temporarily",
            "connection",
            "network",
            "rate limit",
            "429",
            "503",
            "server disconnected",
            "connection reset",
        )
        return any(marker in text for marker in transient_markers)

    async def _ensure_loaded_locked(self) -> None:
        if self._loaded:
            return
        self._jobs = {job.id: job for job in self._store.load_jobs()}
        changed = False
        for job in self._jobs.values():
            if self._normalize_job_delivery(job):
                changed = True
            if self._ensure_job_conversation_scope(job):
                changed = True
        if changed:
            self._persist_jobs_locked()
        self._loaded = True

    def _persist_jobs_locked(self) -> None:
        self._store.save_jobs(self._jobs.values())

    @staticmethod
    def _ensure_job_conversation_scope(job: CronJob) -> bool:
        if job.conversation_scope is not None:
            return False
        scope = None
        if job.delivery is not None:
            scope = conversation_scope_from_parts(
                channel=job.delivery.channel,
                account_id=job.delivery.account_id,
                target=job.delivery.target,
                session_key=job.delivery.session_key or job.session_key,
            )
        if scope is None:
            scope = conversation_scope_from_session_key(job.session_key)
        if scope is None:
            return False
        job.conversation_scope = scope
        return True

    @staticmethod
    def _normalize_job_delivery(job: CronJob) -> bool:
        if job.delivery is None:
            return False
        account_id = job.delivery.account_id
        if not account_id:
            fallback_binding = binding_from_session_key(
                job.delivery.session_key or job.session_key,
            )
            if fallback_binding is not None:
                account_id = fallback_binding.account_id
        normalized_channel, effective_account = normalize_channel_name(
            job.delivery.channel,
            account_id,
        )
        changed = False
        if normalized_channel and normalized_channel != job.delivery.channel:
            job.delivery.channel = normalized_channel
            changed = True
        if effective_account != job.delivery.account_id:
            job.delivery.account_id = effective_account
            changed = True
        return changed

    @staticmethod
    def _compute_next_run(
        job: CronJob,
        *,
        now: datetime,
        after: datetime | None = None,
    ) -> datetime | None:
        return compute_next_run(
            job.schedule,
            now=now,
            created_at=job.created_at,
            after=after,
        )


def create_cron_service(
    *,
    cron_config: CronConfig,
    agent: AgentLoop | None = None,
    execution_coordinator: ExecutionCoordinator | None = None,
    delivery_service: ChannelDeliveryService | None = None,
) -> CronService:
    """Build a cron service from validated config."""
    store = CronStore(cron_config.store.path)
    run_log = CronRunLog(
        Path(cron_config.store.path).expanduser().parent / "runs",
        keep_lines=cron_config.run_log.keep_lines,
    )
    executor = None
    if agent is not None:
        if execution_coordinator is None:
            execution_coordinator = ExecutionCoordinator()
        if delivery_service is None:
            delivery_service = ChannelDeliveryService()
        executor = CronExecutor(
            agent,
            execution_coordinator,
            delivery_service,
            isolated_clear_before_run=cron_config.execution.isolated_clear_before_run,
        )
    return CronService(
        store,
        run_log=run_log,
        executor=executor,
        catch_up_on_start=cron_config.catch_up_on_start,
        poll_interval=cron_config.poll_interval,
        max_concurrent_runs=cron_config.execution.max_concurrent_runs,
    )
