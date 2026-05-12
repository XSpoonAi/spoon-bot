from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from spoon_bot.runtime.session_registry import SessionRuntimeRegistry


@dataclass
class FakeSessionManager:
    saved: list[Any] = field(default_factory=list)

    def save(self, session: Any) -> None:
        self.saved.append(session)


@dataclass
class FakeAgent:
    session_key: str = "default"
    sessions: FakeSessionManager = field(default_factory=FakeSessionManager)
    _session: dict[str, str] = field(default_factory=lambda: {"session_key": "default"})
    cleanup_calls: int = 0

    def build_creation_kwargs(self, **overrides: Any) -> dict[str, Any]:
        kwargs = {"session_key": self.session_key}
        kwargs.update(overrides)
        return kwargs

    async def cleanup(self) -> None:
        self.cleanup_calls += 1


@dataclass
class NonCloneableFakeAgent:
    session_key: str = "default"
    sessions: FakeSessionManager = field(default_factory=FakeSessionManager)
    _session: dict[str, str] = field(default_factory=lambda: {"session_key": "default"})
    cleanup_calls: int = 0

    async def cleanup(self) -> None:
        self.cleanup_calls += 1


@pytest.mark.asyncio
async def test_get_or_create_reuses_runtime_for_same_session() -> None:
    created: list[str] = []

    async def factory(**kwargs: Any) -> FakeAgent:
        created.append(kwargs["session_key"])
        return FakeAgent(session_key=kwargs["session_key"])

    registry = SessionRuntimeRegistry(
        FakeAgent(),
        agent_factory=factory,
        creation_kwargs={"session_key": "default"},
    )

    first = await registry.get_or_create("alpha")
    second = await registry.get_or_create("alpha")

    assert first is second
    assert first.session_key == "alpha"
    assert created == ["alpha"]


@pytest.mark.asyncio
async def test_get_or_create_serializes_concurrent_creation_per_key() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    created: list[str] = []

    async def factory(**kwargs: Any) -> FakeAgent:
        created.append(kwargs["session_key"])
        started.set()
        await release.wait()
        return FakeAgent(session_key=kwargs["session_key"])

    registry = SessionRuntimeRegistry(
        FakeAgent(),
        agent_factory=factory,
        creation_kwargs={"session_key": "default"},
    )

    task_one = asyncio.create_task(registry.get_or_create("beta"))
    await started.wait()
    task_two = asyncio.create_task(registry.get_or_create("beta"))
    release.set()

    runtime_one, runtime_two = await asyncio.gather(task_one, task_two)

    assert runtime_one is runtime_two
    assert created == ["beta"]


@pytest.mark.asyncio
async def test_close_all_saves_runtime_session() -> None:
    agent = FakeAgent()
    registry = SessionRuntimeRegistry(agent)

    await registry.close_all()

    assert agent.sessions.saved == [agent._session]
    assert agent.cleanup_calls == 1
    assert registry.metrics()["closed_total"] == 1


@pytest.mark.asyncio
async def test_non_cloneable_agent_rejects_non_default_session() -> None:
    agent = NonCloneableFakeAgent()
    registry = SessionRuntimeRegistry(agent)

    with pytest.raises(RuntimeError, match="Session runtime cloning is unavailable"):
        await registry.get_or_create("alpha")

    await registry.close_all()

    assert agent.cleanup_calls == 1


@pytest.mark.asyncio
async def test_collect_idle_closes_only_idle_non_default_runtime() -> None:
    created: list[FakeAgent] = []

    async def factory(**kwargs: Any) -> FakeAgent:
        agent = FakeAgent(session_key=kwargs["session_key"])
        created.append(agent)
        return agent

    registry = SessionRuntimeRegistry(
        FakeAgent(),
        agent_factory=factory,
        creation_kwargs={"session_key": "default"},
        idle_seconds=1,
    )

    idle_runtime = await registry.get_or_create("idle")
    running_runtime = await registry.get_or_create("running")
    idle_runtime.last_used_at = datetime.now(UTC) - timedelta(seconds=5)
    running_runtime.last_used_at = datetime.now(UTC) - timedelta(seconds=5)
    running_runtime.active_task_id = "task-1"

    closed = await registry.collect_idle()

    assert [info.session_key for info in closed] == ["idle"]
    assert await registry.get("idle") is None
    assert await registry.get("running") is running_runtime
    assert idle_runtime.agent.cleanup_calls == 1
    assert registry.metrics()["idle_closed_total"] == 1


@pytest.mark.asyncio
async def test_get_or_create_evicts_lru_idle_runtime_when_at_capacity() -> None:
    created: list[str] = []

    async def factory(**kwargs: Any) -> FakeAgent:
        created.append(kwargs["session_key"])
        return FakeAgent(session_key=kwargs["session_key"])

    registry = SessionRuntimeRegistry(
        FakeAgent(),
        agent_factory=factory,
        creation_kwargs={"session_key": "default"},
        max_active=3,
    )

    first = await registry.get_or_create("first")
    second = await registry.get_or_create("second")
    first.last_used_at = datetime.now(UTC) - timedelta(seconds=10)
    second.last_used_at = datetime.now(UTC)

    third = await registry.get_or_create("third")

    assert third.session_key == "third"
    assert await registry.get("first") is None
    assert await registry.get("second") is second
    assert first.agent.cleanup_calls == 1
    assert created == ["first", "second", "third"]
    assert registry.metrics()["evicted_total"] == 1
