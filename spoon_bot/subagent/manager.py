"""Sub-agent manager — spawn, run, collect results, cleanup."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Callable, Awaitable, Optional, TYPE_CHECKING

from loguru import logger

from spoon_bot.subagent.models import (
    SubagentConfig,
    SubagentRecord,
    SubagentResult,
    SubagentState,
    TokenUsage,
)
from spoon_bot.subagent.registry import SubagentRegistry
from spoon_bot.subagent.persistence import SubagentRunsFile, SubagentSweeper
from spoon_bot.bus.events import SubagentEvent

if TYPE_CHECKING:
    from spoon_bot.session.manager import SessionManager
    from spoon_bot.bus.queue import MessageBus

# Announce retry settings
_ANNOUNCE_MAX_RETRIES = 3
_ANNOUNCE_BASE_DELAY = 1.0  # seconds
# Steer rate limit
_STEER_MIN_INTERVAL = 2.0  # seconds between steers to the same agent
# Result truncation for wake messages
_WAKE_RESULT_TRUNCATE = 2000


class SubagentManager:
    """Manages sub-agent lifecycle.

    One SubagentManager instance is created per root AgentLoop. It:
    - Maintains a SubagentRegistry for lifecycle tracking
    - Runs each sub-agent as an independent asyncio.Task
    - Delivers results via asyncio.Queue (pull) AND bus injection (push/wake)
    - Shares the parent's SessionManager (sub-agents get unique session keys)
    - Enforces depth, children-per-agent, and total concurrency limits
    - Emits SubagentEvents for channel integrations (Discord, Telegram)
    """

    def __init__(
        self,
        *,
        session_manager: SessionManager,
        workspace: Path,
        max_depth: int = 2,
        max_children_per_agent: int = 5,
        max_total_subagents: int = 20,
        # Parent agent config for inheritance
        parent_model: str | None = None,
        parent_provider: str | None = None,
        parent_api_key: str | None = None,
        parent_base_url: str | None = None,
        # Persistence settings
        persist_runs: bool = True,
        persist_file: str = "subagents/runs.json",
        archive_after_minutes: int = 60,
        sweeper_interval_seconds: int = 60,
    ) -> None:
        # Build persistence layer
        runs_file: SubagentRunsFile | None = None
        if persist_runs:
            persist_path = Path(persist_file)
            if not persist_path.is_absolute():
                persist_path = workspace / persist_path
            runs_file = SubagentRunsFile(persist_path)

        self.registry = SubagentRegistry(runs_file=runs_file)

        # Background sweeper (started lazily in start_sweeper())
        self._sweeper: SubagentSweeper | None = (
            SubagentSweeper(
                registry=self.registry,
                archive_after_minutes=archive_after_minutes,
                interval_seconds=sweeper_interval_seconds,
            )
            if persist_runs
            else None
        )
        self.session_manager = session_manager
        self.workspace = workspace
        self.max_depth = max_depth
        self.max_children_per_agent = max_children_per_agent
        self.max_total_subagents = max_total_subagents

        # Parent config used for inheritance
        self._parent_model = parent_model
        self._parent_provider = parent_provider
        self._parent_api_key = parent_api_key
        self._parent_base_url = parent_base_url

        # Pull-based result delivery
        self._results: asyncio.Queue[SubagentResult] = asyncio.Queue()

        # Push-based delivery — reference to the message bus
        self._bus: MessageBus | None = None

        # Current spawner context (updated per-process call from AgentLoop)
        self._current_spawner_session: str | None = None
        self._current_spawner_channel: str | None = None

        # Live asyncio.Task handles — agent_id → Task
        self._tasks: dict[str, asyncio.Task[None]] = {}

        # Steer support — pending steer messages and timestamps
        self._steer_requests: dict[str, str] = {}
        self._steer_timestamps: dict[str, float] = {}

        # Lifecycle event listeners
        self._event_listeners: list[Callable[[SubagentEvent], Awaitable[None] | None]] = []

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_bus(self, bus: MessageBus) -> None:
        """Attach the MessageBus for push-based wake delivery."""
        self._bus = bus

    def set_spawner_context(
        self,
        *,
        session_key: str | None,
        channel: str | None,
    ) -> None:
        """Update the current spawner context.

        Called by AgentLoop at the start of each process() / process_with_thinking()
        so that any sub-agents spawned during that call know who to deliver results to.
        """
        self._current_spawner_session = session_key
        self._current_spawner_channel = channel

    def add_event_listener(
        self,
        listener: Callable[[SubagentEvent], Awaitable[None] | None],
    ) -> None:
        """Register a lifecycle event listener (e.g. for Discord notifications)."""
        self._event_listeners.append(listener)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def spawn(
        self,
        task: str,
        *,
        label: str = "",
        parent_id: str | None = None,
        config: SubagentConfig | None = None,
    ) -> SubagentRecord:
        """Spawn a new sub-agent to work on *task*.

        Validates all limits, registers the sub-agent, then starts it as
        a background asyncio.Task.

        Args:
            task:      The message/instruction for the sub-agent.
            label:     Short human-readable label (truncated from task if empty).
            parent_id: ID of the parent sub-agent (None = spawned by root).
            config:    Optional config overrides.

        Returns:
            The newly created SubagentRecord.

        Raises:
            ValueError: If any limit is exceeded or config is invalid.
        """
        cfg = config or SubagentConfig()

        # --- Depth check ---
        parent_depth = 0
        if parent_id:
            parent_record = self.registry.get(parent_id)
            if parent_record:
                parent_depth = parent_record.depth
        child_depth = parent_depth + 1

        if child_depth > self.max_depth:
            raise ValueError(
                f"Max spawn depth ({self.max_depth}) exceeded. "
                f"Parent depth={parent_depth}, requested child depth={child_depth}."
            )

        # --- Children-per-agent limit ---
        if parent_id:
            active_children = self.registry.count_active_children(parent_id)
            if active_children >= self.max_children_per_agent:
                raise ValueError(
                    f"Max children per agent ({self.max_children_per_agent}) "
                    f"already reached for parent {parent_id}."
                )

        # --- Total sub-agent limit ---
        total_active = self.registry.count_active_total()
        if total_active >= self.max_total_subagents:
            raise ValueError(
                f"Max total active sub-agents ({self.max_total_subagents}) reached."
            )

        # --- Resolve effective model name ---
        effective_model = cfg.model or self._parent_model

        # --- Create and register record ---
        record = SubagentRecord(
            parent_id=parent_id,
            depth=child_depth,
            label=label or task[:60],
            task=task,
            config=cfg,
            model_name=effective_model,
            spawner_session_key=self._current_spawner_session,
            spawner_channel=self._current_spawner_channel,
        )

        self.registry.register(record)
        logger.info(
            f"Sub-agent {record.agent_id!r} registered: "
            f"depth={record.depth}, label={record.label!r}, model={effective_model!r}"
        )

        # Emit lifecycle event
        await self._emit_event(SubagentEvent(
            event_type="spawning",
            agent_id=record.agent_id,
            label=record.label,
            parent_id=record.parent_id,
            depth=record.depth,
            model_name=effective_model,
            spawner_session_key=record.spawner_session_key,
            spawner_channel=record.spawner_channel,
        ))

        # --- Start background execution ---
        bg_task = asyncio.create_task(
            self._run_subagent(record),
            name=f"subagent-{record.agent_id}",
        )
        self._tasks[record.agent_id] = bg_task

        return record

    async def collect_results(
        self,
        timeout: float = 0.0,
    ) -> list[SubagentResult]:
        """Drain all available results from the results queue (pull-based).

        Args:
            timeout: If > 0, wait up to this many seconds for at least one
                     result before returning. If 0, return immediately.

        Returns:
            List of SubagentResult (may be empty).
        """
        results: list[SubagentResult] = []
        try:
            if timeout > 0 and not results:
                result = await asyncio.wait_for(
                    self._results.get(), timeout=timeout
                )
                results.append(result)
            while True:
                result = self._results.get_nowait()
                results.append(result)
        except (asyncio.QueueEmpty, asyncio.TimeoutError):
            pass
        return results

    async def cancel(self, agent_id: str, *, cascade: bool = True) -> bool:
        """Cancel a running or pending sub-agent.

        Args:
            agent_id: The agent to cancel.
            cascade:  If True (default), also cancel all descendants.

        Returns:
            True if the task was found and cancellation was requested.
        """
        found = False
        if cascade:
            # Cancel descendants first (deepest first)
            descendants = self.registry.get_descendants(agent_id)
            for desc in reversed(descendants):
                task = self._tasks.get(desc.agent_id)
                if task and not task.done():
                    task.cancel()
                    logger.info(f"Cascade cancel: sub-agent {desc.agent_id!r}")
                    found = True

        task = self._tasks.get(agent_id)
        if task and not task.done():
            task.cancel()
            logger.info(f"Cancellation requested for sub-agent {agent_id}")
            found = True
        return found

    async def cancel_all(self, parent_id: str | None = None) -> int:
        """Cancel all active sub-agents, optionally filtered by parent.

        Always uses cascade kill.

        Returns:
            Number of top-level sub-agents whose cancellation was requested.
        """
        count = 0
        for record in self.registry.list_all():
            if parent_id and record.parent_id != parent_id:
                continue
            if record.state in (SubagentState.PENDING, SubagentState.RUNNING):
                if await self.cancel(record.agent_id, cascade=True):
                    count += 1
        return count

    async def steer(
        self,
        agent_id: str,
        new_message: str,
    ) -> dict[str, Any]:
        """Redirect a running sub-agent with a new message.

        The current run is aborted and the sub-agent is restarted with
        *new_message* appended to the existing session history.

        Rate limited: at most one steer per _STEER_MIN_INTERVAL seconds.

        Returns:
            Dict with keys: status, agent_id, label, message.
        """
        record = self.registry.get(agent_id)
        if record is None:
            return {"status": "not_found", "agent_id": agent_id, "message": "Agent not found."}

        if record.state not in (SubagentState.PENDING, SubagentState.RUNNING):
            return {
                "status": "done",
                "agent_id": agent_id,
                "label": record.label,
                "message": f"Agent is already in terminal state: {record.state.value}",
            }

        # Rate limiting
        last_steer = self._steer_timestamps.get(agent_id, 0.0)
        elapsed_since = time.time() - last_steer
        if elapsed_since < _STEER_MIN_INTERVAL:
            wait = round(_STEER_MIN_INTERVAL - elapsed_since, 1)
            return {
                "status": "rate_limited",
                "agent_id": agent_id,
                "label": record.label,
                "message": f"Rate limited. Try again in {wait}s.",
            }

        # Queue the steer request (processed in _run_subagent's CancelledError handler)
        self._steer_requests[agent_id] = new_message
        self._steer_timestamps[agent_id] = time.time()

        # Cancel current task (triggers the steer restart)
        task = self._tasks.get(agent_id)
        if task and not task.done():
            task.cancel()
            logger.info(
                f"Steer requested for sub-agent {agent_id!r}: "
                f"{new_message[:60]!r}"
            )
            return {
                "status": "accepted",
                "agent_id": agent_id,
                "label": record.label,
                "message": f"Steer accepted. Sub-agent {agent_id} will be redirected.",
            }
        else:
            # Task is done — start a fresh run
            self._steer_requests.pop(agent_id, None)
            return {
                "status": "done",
                "agent_id": agent_id,
                "label": record.label,
                "message": "Agent already finished. Use spawn to create a new one.",
            }

    async def get_info(self, agent_id: str) -> dict[str, Any] | None:
        """Return detailed metadata for a sub-agent.

        Returns None if the agent is not found.
        """
        record = self.registry.get(agent_id)
        if record is None:
            return None

        now = time.time()
        elapsed: float | None = None
        if record.started_at:
            end = record.completed_at or now
            elapsed = round(end - record.started_at, 2)

        pending_desc = self.registry.count_pending_descendants(agent_id)

        return {
            "agent_id": record.agent_id,
            "label": record.label,
            "state": record.state.value,
            "task": record.task,
            "depth": record.depth,
            "parent_id": record.parent_id,
            "session_key": record.session_key,
            "model": record.model_name,
            "tool_profile": record.config.tool_profile,
            "max_iterations": record.config.max_iterations,
            "thinking_level": record.config.thinking_level,
            "timeout_seconds": record.config.timeout_seconds,
            "children": record.children,
            "pending_descendants": pending_desc,
            "created_at": record.created_at,
            "started_at": record.started_at,
            "completed_at": record.completed_at,
            "elapsed_seconds": elapsed,
            "result_preview": (record.result or "")[:300] if record.result else None,
            "error": record.error,
            "token_usage": (
                record.token_usage.model_dump() if record.token_usage else None
            ),
        }

    async def start_sweeper(self) -> None:
        """Start the background archive sweeper (call after the event loop is running)."""
        if self._sweeper is not None:
            await self._sweeper.start()

    async def stop_sweeper(self) -> None:
        """Stop the background archive sweeper."""
        if self._sweeper is not None:
            await self._sweeper.stop()

    async def cleanup(self) -> None:
        """Shut down all sub-agents, stop the sweeper, and clean up sessions."""
        # Stop sweeper first so it doesn't race with cleanup
        await self.stop_sweeper()

        cancelled = await self.cancel_all()
        if cancelled:
            logger.info(f"Cancelled {cancelled} sub-agent(s) during cleanup")

        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)

        for record in self.registry.list_all():
            try:
                self.session_manager.delete(record.session_key)
            except Exception:
                pass

        self._tasks.clear()
        logger.info("SubagentManager cleanup complete")

    def get_status_summary(
        self,
        parent_id: str | None = None,
    ) -> dict[str, Any]:
        """Return a rich summary dict of sub-agent statuses.

        Separates active (pending/running) from recent (terminal) records.
        Includes model name and pending descendant counts.
        """
        records = (
            self.registry.list_by_parent(parent_id)
            if parent_id
            else self.registry.list_all()
        )

        active: list[dict[str, Any]] = []
        recent: list[dict[str, Any]] = []

        now = time.time()
        for r in records:
            elapsed: float | None = None
            if r.started_at:
                end = r.completed_at or now
                elapsed = round(end - r.started_at, 1)

            pending_desc = self.registry.count_pending_descendants(r.agent_id)

            entry = {
                "agent_id": r.agent_id,
                "label": r.label,
                "state": r.state.value,
                "depth": r.depth,
                "model": r.model_name,
                "elapsed_seconds": elapsed,
                "pending_descendants": pending_desc,
            }

            if r.state in (SubagentState.PENDING, SubagentState.RUNNING):
                active.append(entry)
            else:
                recent.append(entry)

        return {
            "total": len(records),
            "active": active,
            "recent": recent,
            # Legacy field for backward compat with existing status handler
            "by_state": {
                e["state"]: [e2 for e2 in records
                              if e2.state.value == e["state"]]
                for e in (active + recent)
            },
        }

    # ------------------------------------------------------------------
    # Internal: background execution
    # ------------------------------------------------------------------

    async def _run_subagent(
        self,
        record: SubagentRecord,
        task_override: str | None = None,
        is_restart: bool = False,
    ) -> None:
        """Background coroutine: create AgentLoop → run task → deliver result."""
        from spoon_bot.agent.loop import AgentLoop

        task_text = task_override or record.task

        if not is_restart:
            self.registry.transition(
                record.agent_id,
                SubagentState.RUNNING,
                started_at=time.time(),
            )
            await self._emit_event(SubagentEvent(
                event_type="started",
                agent_id=record.agent_id,
                label=record.label,
                parent_id=record.parent_id,
                depth=record.depth,
                model_name=record.model_name,
                spawner_session_key=record.spawner_session_key,
                spawner_channel=record.spawner_channel,
            ))

        logger.info(f"Sub-agent {record.agent_id!r} starting task: {task_text[:80]!r}")

        child_agent: AgentLoop | None = None
        _steered = False

        try:
            cfg = record.config
            effective_model = cfg.model or self._parent_model

            child_agent = AgentLoop(
                workspace=self.workspace,
                model=effective_model,
                provider=cfg.provider or self._parent_provider,
                api_key=cfg.api_key or self._parent_api_key,
                base_url=cfg.base_url or self._parent_base_url,
                max_iterations=cfg.max_iterations,
                session_key=record.session_key,
                system_prompt=cfg.system_prompt or self._build_system_prompt(record),
                enable_skills=cfg.enable_skills,
                enabled_tools=cfg.enabled_tools,
                tool_profile=cfg.tool_profile,
                context_window=cfg.context_window,
                auto_commit=False,
                session_store_backend="file",
            )

            await child_agent.initialize()

            # Inject SubagentTool with reference back to this manager
            from spoon_bot.subagent.tools import SubagentTool
            spawn_tool = child_agent.tools.get("spawn")
            if spawn_tool and isinstance(spawn_tool, SubagentTool):
                spawn_tool.set_manager(self)
                spawn_tool._parent_agent_id = record.agent_id

            # Run the task — with optional timeout and thinking
            if cfg.timeout_seconds:
                run_coro = self._run_process(child_agent, task_text, cfg)
                result_text = await asyncio.wait_for(
                    run_coro, timeout=float(cfg.timeout_seconds)
                )
            else:
                result_text = await self._run_process(child_agent, task_text, cfg)

            # Extract token usage if available
            token_usage = self._extract_token_usage(child_agent)

            await child_agent.cleanup()

            now = time.time()
            elapsed = round(now - (record.started_at or record.created_at), 2)

            self.registry.transition(
                record.agent_id,
                SubagentState.COMPLETED,
                result=result_text,
                completed_at=now,
                frozen_result_text=result_text,
                token_usage=token_usage,
            )

            result_obj = SubagentResult(
                agent_id=record.agent_id,
                label=record.label,
                state=SubagentState.COMPLETED,
                result=result_text,
                elapsed_seconds=elapsed,
                spawner_session_key=record.spawner_session_key,
                spawner_channel=record.spawner_channel,
                model_name=record.model_name,
            )
            await self._results.put(result_obj)

            logger.info(f"Sub-agent {record.agent_id!r} completed in {elapsed}s")

            # Emit lifecycle event
            await self._emit_event(SubagentEvent(
                event_type="completed",
                agent_id=record.agent_id,
                label=record.label,
                parent_id=record.parent_id,
                depth=record.depth,
                model_name=record.model_name,
                result=result_text,
                elapsed_seconds=elapsed,
                spawner_session_key=record.spawner_session_key,
                spawner_channel=record.spawner_channel,
            ))

            # Push-based delivery: announce result to spawner via bus
            await self._announce_result(result_obj)

        except asyncio.TimeoutError:
            if child_agent:
                try:
                    await child_agent.cleanup()
                except Exception:
                    pass
            error_msg = f"Sub-agent timed out after {cfg.timeout_seconds}s"
            logger.warning(f"Sub-agent {record.agent_id!r}: {error_msg}")
            self.registry.transition(
                record.agent_id,
                SubagentState.FAILED,
                error=error_msg,
                completed_at=time.time(),
            )
            result_obj = SubagentResult(
                agent_id=record.agent_id,
                label=record.label,
                state=SubagentState.FAILED,
                error=error_msg,
                spawner_session_key=record.spawner_session_key,
                spawner_channel=record.spawner_channel,
            )
            await self._results.put(result_obj)
            await self._announce_result(result_obj)
            await self._emit_event(SubagentEvent(
                event_type="failed",
                agent_id=record.agent_id,
                label=record.label,
                parent_id=record.parent_id,
                depth=record.depth,
                error=error_msg,
                spawner_session_key=record.spawner_session_key,
                spawner_channel=record.spawner_channel,
            ))

        except asyncio.CancelledError:
            if child_agent:
                try:
                    await child_agent.cleanup()
                except Exception:
                    pass

            # Check if this is a steer request (not a user cancellation)
            steer_message = self._steer_requests.pop(record.agent_id, None)
            if steer_message is not None:
                # Steer: restart with the new message, preserve session history
                _steered = True
                logger.info(f"Sub-agent {record.agent_id!r} steered with new message")
                bg_task = asyncio.create_task(
                    self._run_subagent(record, task_override=steer_message, is_restart=True),
                    name=f"subagent-{record.agent_id}",
                )
                self._tasks[record.agent_id] = bg_task
            else:
                # Regular cancellation
                self.registry.transition(
                    record.agent_id,
                    SubagentState.CANCELLED,
                    completed_at=time.time(),
                )
                result_obj = SubagentResult(
                    agent_id=record.agent_id,
                    label=record.label,
                    state=SubagentState.CANCELLED,
                    spawner_session_key=record.spawner_session_key,
                    spawner_channel=record.spawner_channel,
                )
                await self._results.put(result_obj)
                logger.info(f"Sub-agent {record.agent_id!r} was cancelled")
                await self._emit_event(SubagentEvent(
                    event_type="cancelled",
                    agent_id=record.agent_id,
                    label=record.label,
                    parent_id=record.parent_id,
                    depth=record.depth,
                    spawner_session_key=record.spawner_session_key,
                    spawner_channel=record.spawner_channel,
                ))

        except Exception as exc:
            if child_agent:
                try:
                    await child_agent.cleanup()
                except Exception:
                    pass
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.error(f"Sub-agent {record.agent_id!r} failed: {error_msg}")
            self.registry.transition(
                record.agent_id,
                SubagentState.FAILED,
                error=error_msg,
                completed_at=time.time(),
            )
            result_obj = SubagentResult(
                agent_id=record.agent_id,
                label=record.label,
                state=SubagentState.FAILED,
                error=error_msg,
                spawner_session_key=record.spawner_session_key,
                spawner_channel=record.spawner_channel,
            )
            await self._results.put(result_obj)
            await self._announce_result(result_obj)
            await self._emit_event(SubagentEvent(
                event_type="failed",
                agent_id=record.agent_id,
                label=record.label,
                parent_id=record.parent_id,
                depth=record.depth,
                error=error_msg,
                spawner_session_key=record.spawner_session_key,
                spawner_channel=record.spawner_channel,
            ))

        finally:
            if not _steered:
                self._tasks.pop(record.agent_id, None)

    @staticmethod
    async def _run_process(
        child_agent: Any,
        task_text: str,
        cfg: SubagentConfig,
    ) -> str:
        """Run the child agent process with optional extended thinking."""
        if cfg.thinking_level and cfg.thinking_level != "off":
            result_text, _ = await child_agent.process_with_thinking(task_text)
        else:
            result_text = await child_agent.process(task_text)
        return result_text or ""

    @staticmethod
    def _extract_token_usage(child_agent: Any) -> Optional[TokenUsage]:
        """Extract token usage from a child AgentLoop if available."""
        try:
            usage = child_agent.get_usage()
            if not usage:
                return None
            return TokenUsage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                cache_read_tokens=usage.get("cache_read_tokens", 0),
                cache_write_tokens=usage.get("cache_write_tokens", 0),
            )
        except Exception:
            return None

    async def _announce_result(self, result: SubagentResult) -> None:
        """Push result to spawner channel via bus injection (with retry/backoff).

        Falls back silently if no bus is set or channel cannot be reached.
        """
        if not self._bus:
            return
        if not result.spawner_session_key or not result.spawner_channel:
            return

        # Build wake message
        state = result.state.value
        content_raw = result.result or result.error or "(no output)"
        content = content_raw[:_WAKE_RESULT_TRUNCATE]
        if len(content_raw) > _WAKE_RESULT_TRUNCATE:
            content += f"\n... [{len(content_raw) - _WAKE_RESULT_TRUNCATE} chars omitted]"

        elapsed_str = (
            f" in {result.elapsed_seconds}s" if result.elapsed_seconds else ""
        )
        wake_content = (
            f"[Sub-agent Completed] '{result.label}' ({result.agent_id}) "
            f"has {state}{elapsed_str}.\n\n"
            f"{content}"
        )

        from spoon_bot.bus.events import InboundMessage
        wake_msg = InboundMessage(
            content=wake_content,
            channel=result.spawner_channel,
            session_key=result.spawner_session_key,
            sender_id="subagent_system",
            metadata={"is_subagent_wake": True, "subagent_id": result.agent_id},
        )

        for attempt in range(_ANNOUNCE_MAX_RETRIES):
            try:
                published = self._bus.publish(wake_msg)
                if published:
                    logger.info(
                        f"Wake announced for sub-agent {result.agent_id!r} "
                        f"→ {result.spawner_channel} / {result.spawner_session_key}"
                    )
                    return
                # Queue full — retry after backoff
                delay = _ANNOUNCE_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    f"Bus queue full, retrying announce for {result.agent_id!r} "
                    f"in {delay}s (attempt {attempt + 1}/{_ANNOUNCE_MAX_RETRIES})"
                )
                await asyncio.sleep(delay)
            except Exception as exc:
                logger.error(
                    f"Announce failed for {result.agent_id!r} "
                    f"(attempt {attempt + 1}): {exc}"
                )
                if attempt < _ANNOUNCE_MAX_RETRIES - 1:
                    await asyncio.sleep(_ANNOUNCE_BASE_DELAY * (2 ** attempt))

        logger.error(
            f"Failed to announce sub-agent {result.agent_id!r} result "
            f"after {_ANNOUNCE_MAX_RETRIES} attempts."
        )

    async def _emit_event(self, event: SubagentEvent) -> None:
        """Notify all registered lifecycle event listeners."""
        for listener in self._event_listeners:
            try:
                result = listener(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                logger.debug(f"Subagent event listener error: {exc}")

    def _build_system_prompt(self, record: SubagentRecord) -> str:
        """Build a focused system prompt for the sub-agent."""
        return (
            f"You are a sub-agent (ID: {record.agent_id}) working on a specific subtask.\n"
            f"Your task: {record.task}\n\n"
            f"Guidelines:\n"
            f"- Focus exclusively on this task.\n"
            f"- Be concise and deliver concrete, actionable results.\n"
            f"- Do not spawn further sub-agents unless strictly necessary.\n"
            f"- When done, provide a clear summary of results."
        )
