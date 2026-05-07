"""Session-scoped agent runtime registry.

The persisted ``SessionManager`` stores conversation history.  This module
tracks the in-memory agent runners that own mutable execution state for each
logical session.
"""

from __future__ import annotations

import asyncio
import inspect
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from spoon_bot.runtime.execution import normalize_session_key

if TYPE_CHECKING:
    from spoon_bot.agent.loop import AgentLoop


AgentFactory = Callable[..., Awaitable["AgentLoop"]]


@dataclass
class SessionRuntimeInfo:
    """Serializable runtime status for one logical session."""

    session_key: str
    created_at: datetime
    last_used_at: datetime
    active_task_id: str | None = None
    closed: bool = False
    status: str = "idle"

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_key": self.session_key,
            "created_at": self.created_at.isoformat(),
            "last_used_at": self.last_used_at.isoformat(),
            "active_task_id": self.active_task_id,
            "closed": self.closed,
            "status": self.status,
        }


@dataclass
class SessionRuntime:
    """A session-owned agent runner and its execution lock."""

    session_key: str
    agent: "AgentLoop"
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_used_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    active_task_id: str | None = None
    closed: bool = False

    def touch(self) -> None:
        self.last_used_at = datetime.now(UTC)

    def info(self) -> SessionRuntimeInfo:
        return SessionRuntimeInfo(
            session_key=self.session_key,
            created_at=self.created_at,
            last_used_at=self.last_used_at,
            active_task_id=self.active_task_id,
            closed=self.closed,
            status=self.status,
        )

    @property
    def status(self) -> str:
        if self.closed:
            return "closed"
        if self.active_task_id is not None or self.lock.locked():
            return "running"
        return "idle"


class SessionRuntimeRegistry:
    """Create, cache, and close session-scoped agent runtimes."""

    def __init__(
        self,
        default_agent: "AgentLoop | None" = None,
        *,
        agent_factory: AgentFactory | None = None,
        creation_kwargs: dict[str, Any] | None = None,
        idle_seconds: float | None = None,
        max_active: int | None = None,
    ) -> None:
        self._runtimes: dict[str, SessionRuntime] = {}
        self._lock = asyncio.Lock()
        self._agent_factory = agent_factory
        self._creation_kwargs = dict(creation_kwargs or {})
        self._default_session_key = "default"
        self._idle_seconds = self._resolve_idle_seconds(idle_seconds)
        self._max_active = self._resolve_max_active(max_active)
        self._created_total = 0
        self._closed_total = 0
        self._idle_closed_total = 0
        self._evicted_total = 0
        self._explicit_closed_total = 0

        if default_agent is not None:
            default_key = normalize_session_key(getattr(default_agent, "session_key", "default"))
            self._default_session_key = default_key
            self._runtimes[default_key] = SessionRuntime(
                session_key=default_key,
                agent=default_agent,
            )
            self._created_total = 1
            self._ensure_clone_factory(default_agent)

    @property
    def default_session_key(self) -> str:
        return self._default_session_key

    def default_agent(self) -> "AgentLoop | None":
        runtime = self._runtimes.get(self._default_session_key)
        return runtime.agent if runtime is not None else None

    def _ensure_clone_factory(self, default_agent: "AgentLoop | None" = None) -> None:
        if self._agent_factory is not None:
            return
        if default_agent is None:
            default_agent = self.default_agent()
        if default_agent is None:
            return

        build_kwargs = getattr(default_agent, "build_creation_kwargs", None)
        if not self._creation_kwargs and callable(build_kwargs):
            try:
                maybe_kwargs = build_kwargs()
            except Exception as exc:
                logger.warning(f"Failed to derive session runtime creation kwargs: {exc}")
                maybe_kwargs = None
            if inspect.isawaitable(maybe_kwargs):
                close = getattr(maybe_kwargs, "close", None)
                if callable(close):
                    close()
                maybe_kwargs = None
            if isinstance(maybe_kwargs, dict):
                self._creation_kwargs = maybe_kwargs

        if not self._creation_kwargs:
            return

        module_name = type(default_agent).__module__
        if module_name.startswith("spoon_bot.core"):
            from spoon_bot.core import create_agent
        else:
            from spoon_bot.agent.loop import create_agent

        self._agent_factory = create_agent

    async def get_or_create(self, session_key: str | None) -> SessionRuntime:
        key = normalize_session_key(session_key)
        runtime = self._runtimes.get(key)
        if runtime is not None and not runtime.closed:
            runtime.touch()
            return runtime

        async with self._lock:
            runtime = self._runtimes.get(key)
            if runtime is not None and not runtime.closed:
                runtime.touch()
                return runtime

            await self._collect_idle_unlocked(datetime.now(UTC))
            await self._enforce_max_active_unlocked(reserve=1)

            agent = await self._create_agent_for_session(key)
            runtime = SessionRuntime(session_key=key, agent=agent)
            self._runtimes[key] = runtime
            self._created_total += 1
            logger.info(f"Created session runtime: {key}")
            return runtime

    async def get(self, session_key: str | None) -> SessionRuntime | None:
        key = normalize_session_key(session_key)
        runtime = self._runtimes.get(key)
        if runtime is not None and not runtime.closed:
            runtime.touch()
            return runtime
        return None

    async def list(self) -> list[SessionRuntimeInfo]:
        return [runtime.info() for runtime in self._runtimes.values()]

    async def close(self, session_key: str | None) -> bool:
        key = normalize_session_key(session_key)
        async with self._lock:
            if key == self._default_session_key:
                return False
            runtime = self._runtimes.get(key)
            if runtime is None:
                return False
            if runtime.active_task_id is not None or runtime.lock.locked():
                return False
            self._runtimes.pop(key, None)
            self._explicit_closed_total += 1
            await self._close_runtime(runtime, cleaned_agent_ids=set())
            return True

    async def close_all(self) -> None:
        runtimes = list(self._runtimes.values())
        self._runtimes.clear()
        cleaned_agent_ids: set[int] = set()
        for runtime in runtimes:
            await self._close_runtime(runtime, cleaned_agent_ids=cleaned_agent_ids)

    async def collect_idle(self) -> list[SessionRuntimeInfo]:
        async with self._lock:
            return await self._collect_idle_unlocked(datetime.now(UTC))

    @property
    def idle_seconds(self) -> float | None:
        return self._idle_seconds

    @property
    def max_active(self) -> int | None:
        return self._max_active

    def metrics(self) -> dict[str, Any]:
        active_runtimes = [runtime for runtime in self._runtimes.values() if not runtime.closed]
        return {
            "active": len(active_runtimes),
            "running": len([runtime for runtime in active_runtimes if runtime.status == "running"]),
            "idle": len([runtime for runtime in active_runtimes if runtime.status == "idle"]),
            "created_total": self._created_total,
            "closed_total": self._closed_total,
            "idle_closed_total": self._idle_closed_total,
            "evicted_total": self._evicted_total,
            "explicit_closed_total": self._explicit_closed_total,
            "idle_seconds": self._idle_seconds,
            "max_active": self._max_active,
        }

    async def _create_agent_for_session(self, session_key: str) -> "AgentLoop":
        default_agent = self.default_agent()
        if session_key == self._default_session_key and default_agent is not None:
            return default_agent

        self._ensure_clone_factory(default_agent)

        if self._agent_factory is None or not self._creation_kwargs:
            raise RuntimeError(
                "Session runtime cloning is unavailable for this agent. "
                "Refusing to reuse the default agent for a non-default session because it would serialize streams and share mutable state."
            )

        kwargs = dict(self._creation_kwargs)
        kwargs["session_key"] = session_key

        if default_agent is not None and hasattr(default_agent, "sessions"):
            kwargs["session_manager"] = default_agent.sessions

        return await self._agent_factory(**kwargs)

    @staticmethod
    def _bind_agent_to_session(agent: "AgentLoop", session_key: str) -> None:
        sessions = getattr(agent, "sessions", None)
        if sessions is not None and hasattr(sessions, "get_or_create"):
            try:
                agent._session = sessions.get_or_create(session_key)
            except Exception:
                pass
        try:
            agent.session_key = session_key
        except Exception:
            pass

    async def _close_runtime(self, runtime: SessionRuntime, *, cleaned_agent_ids: set[int]) -> None:
        if not runtime.closed:
            self._closed_total += 1
        runtime.closed = True
        try:
            session = getattr(runtime.agent, "_session", None)
            sessions = getattr(runtime.agent, "sessions", None)
            if session is not None and sessions is not None:
                sessions.save(session)
        except Exception as exc:
            logger.debug(f"Failed to save session on runtime close: {exc}")

        agent_id = id(runtime.agent)
        if agent_id in cleaned_agent_ids:
            return
        cleaned_agent_ids.add(agent_id)

        cleanup = getattr(runtime.agent, "cleanup", None)
        if callable(cleanup):
            try:
                result = cleanup()
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                logger.debug(f"Failed to cleanup session runtime agent: {exc}")

    async def _collect_idle_unlocked(self, now: datetime) -> list[SessionRuntimeInfo]:
        if self._idle_seconds is None:
            return []

        closed: list[SessionRuntimeInfo] = []
        for key, runtime in list(self._runtimes.items()):
            if key == self._default_session_key:
                continue
            if runtime.active_task_id is not None or runtime.lock.locked():
                continue
            idle_for = (now - runtime.last_used_at).total_seconds()
            if idle_for < self._idle_seconds:
                continue
            self._runtimes.pop(key, None)
            info = runtime.info()
            await self._close_runtime(runtime, cleaned_agent_ids=set())
            self._idle_closed_total += 1
            closed.append(info)
            logger.info(f"Closed idle session runtime: {key}")
        return closed

    async def _enforce_max_active_unlocked(self, *, reserve: int = 0) -> None:
        if self._max_active is None:
            return

        while len(self._runtimes) + reserve > self._max_active:
            candidate = self._oldest_idle_runtime_key()
            if candidate is None:
                raise RuntimeError("Session runtime capacity exhausted")
            runtime = self._runtimes.pop(candidate)
            await self._close_runtime(runtime, cleaned_agent_ids=set())
            self._evicted_total += 1
            logger.info(f"Evicted session runtime due to capacity: {candidate}")

    def _oldest_idle_runtime_key(self) -> str | None:
        candidates = [
            runtime
            for key, runtime in self._runtimes.items()
            if key != self._default_session_key
            and runtime.active_task_id is None
            and not runtime.lock.locked()
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda runtime: runtime.last_used_at).session_key

    @staticmethod
    def _resolve_idle_seconds(value: float | None) -> float | None:
        if value is None:
            raw = os.environ.get("SPOON_BOT_SESSION_RUNTIME_IDLE_SECONDS", "1800")
            try:
                value = float(raw)
            except ValueError:
                value = 1800
        return value if value > 0 else None

    @staticmethod
    def _resolve_max_active(value: int | None) -> int | None:
        if value is None:
            raw = os.environ.get("SPOON_BOT_SESSION_RUNTIME_MAX_ACTIVE", "64")
            try:
                value = int(raw)
            except ValueError:
                value = 64
        return value if value > 0 else None
