"""Execution runner for scheduled tasks."""

from __future__ import annotations

import asyncio
from copy import deepcopy

from loguru import logger

from spoon_bot.agent.loop import AgentLoop, create_agent
from spoon_bot.channels.delivery import ChannelDeliveryService
from spoon_bot.cron.models import CronExecutionResult, CronJob
from spoon_bot.runtime.execution import ExecutionCoordinator

_CRON_RUNNER_TOOL_PROFILE = "automation"


class CronExecutor:
    """Executes jobs with a dedicated agent runner."""

    def __init__(
        self,
        source_agent: AgentLoop,
        execution_coordinator: ExecutionCoordinator,
        delivery_service: ChannelDeliveryService,
        *,
        isolated_clear_before_run: bool = True,
    ) -> None:
        self._source_agent = source_agent
        self._execution_coordinator = execution_coordinator
        self._delivery_service = delivery_service
        self._isolated_clear_before_run = isolated_clear_before_run
        self._runner: AgentLoop | None = None
        self._runner_init_lock = asyncio.Lock()

    async def close(self) -> None:
        """Release the dedicated runner."""
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def execute(self, job: CronJob) -> CronExecutionResult:
        """Run one job and optionally deliver its response."""
        runner = await self._ensure_runner()
        session_key = self._resolve_session_key(job)
        self._apply_job_tool_filter(runner, job)

        if job.target_mode == "isolated" and self._isolated_clear_before_run:
            runner.clear_session_history(session_key)

        session_lock = self._execution_coordinator.get_session_lock(session_key)
        runner_lock = self._execution_coordinator.get_runner_lock("cron-runner")
        try:
            async with session_lock:
                async with runner_lock:
                    try:
                        output = await runner.process(job.prompt, session_key=session_key)
                    except Exception as exc:
                        logger.error(f"Cron job {job.id} failed: {exc}")
                        return CronExecutionResult(
                            status="error",
                            session_key=session_key,
                            error=str(exc),
                            delivered=False,
                            delivery_status="skipped",
                            attempts=1,
                        )

                    delivered = False
                    delivery_status = "skipped"
                    if job.delivery_mode == "none":
                        delivery_status = "skipped:none"
                    else:
                        try:
                            session = runner.sessions.get(session_key)
                            binding = self._delivery_service.resolve_binding(
                                explicit=job.delivery.model_dump(mode="json") if job.delivery else None,
                                session=session,
                                session_key=job.session_key or session_key,
                            )
                            if binding is None:
                                delivery_status = "unresolved"
                            else:
                                await self._delivery_service.deliver(output, binding)
                                delivered = True
                                delivery_status = "sent"
                        except Exception as exc:
                            logger.warning(f"Cron job {job.id} delivery failed: {exc}")
                            delivery_status = f"error:{type(exc).__name__}"

                    return CronExecutionResult(
                        status="success",
                        session_key=session_key,
                        output=output,
                        delivered=delivered,
                        delivery_status=delivery_status,
                        attempts=1,
                    )
        finally:
            self._restore_default_tool_filter(runner)

    async def _ensure_runner(self) -> AgentLoop:
        if self._runner is not None:
            return self._runner

        async with self._runner_init_lock:
            if self._runner is not None:
                return self._runner

            kwargs = self._source_agent.build_creation_kwargs(
                session_key="cron_bootstrap",
                auto_commit=False,
                auto_reload=False,
                enabled_tools=None,
                tool_profile=_CRON_RUNNER_TOOL_PROFILE,
            )
            kwargs["mcp_config"] = deepcopy(kwargs.get("mcp_config") or {})
            self._runner = await create_agent(**kwargs)
            logger.info("Cron executor runner initialized")
            return self._runner

    @staticmethod
    def _resolve_session_key(job: CronJob) -> str:
        if job.target_mode == "isolated":
            return f"cron_{job.id}"
        if job.target_mode == "main":
            return "default"
        return str(job.session_key or "default")

    @staticmethod
    def _apply_job_tool_filter(runner: AgentLoop, job: CronJob) -> None:
        if not hasattr(runner, "tools"):
            return
        if job.allowed_tools:
            runner.tools.set_tool_filter(enabled_tools=set(job.allowed_tools))
            return
        runner.tools.set_tool_filter(tool_profile=_CRON_RUNNER_TOOL_PROFILE)

    @staticmethod
    def _restore_default_tool_filter(runner: AgentLoop) -> None:
        if not hasattr(runner, "tools"):
            return
        runner.tools.set_tool_filter(tool_profile=_CRON_RUNNER_TOOL_PROFILE)
