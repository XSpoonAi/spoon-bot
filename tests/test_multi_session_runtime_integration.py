from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from spoon_bot.gateway import app as app_module
from spoon_bot.gateway.app import create_app, set_agent
from spoon_bot.gateway.config import GatewayConfig
from spoon_bot.gateway.websocket.manager import ConnectionManager
from spoon_bot.runtime.session_registry import SessionRuntimeRegistry


@dataclass
class FakeSession:
    session_key: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    messages: list[dict[str, Any]] = field(default_factory=list)


class FakeSessions:
    def __init__(self) -> None:
        self._sessions: dict[str, FakeSession] = {}
        self.saved: list[FakeSession] = []

    def get_or_create(self, key: str) -> FakeSession:
        return self._sessions.setdefault(key, FakeSession(session_key=key))

    def get(self, key: str) -> FakeSession | None:
        return self._sessions.get(key)

    def list_sessions(self) -> list[str]:
        return list(self._sessions)

    def save(self, session: FakeSession) -> None:
        self._sessions[session.session_key] = session
        self.saved.append(session)

    def delete(self, key: str) -> bool:
        return self._sessions.pop(key, None) is not None


class RuntimeAgent:
    model = "test-model"
    provider = "test"
    tools: list[str] = []
    skills: list[str] = []

    def __init__(
        self,
        session_key: str = "default",
        *,
        sessions: FakeSessions | None = None,
        on_process: Any = None,
        on_stream: Any = None,
    ) -> None:
        self.session_key = session_key
        self.sessions = sessions or FakeSessions()
        self._session = self.sessions.get_or_create(session_key)
        self.on_process = on_process
        self.on_stream = on_stream
        self.cleanup_calls = 0

    def build_creation_kwargs(self, **overrides: Any) -> dict[str, Any]:
        kwargs = {
            "session_key": self.session_key,
            "session_manager": self.sessions,
            "on_process": self.on_process,
            "on_stream": self.on_stream,
        }
        kwargs.update(overrides)
        return kwargs

    async def process(self, *, message: str, **kwargs: Any) -> str:
        self._session.messages.append({"role": "user", "content": message})
        if self.on_process is not None:
            await self.on_process(self.session_key)
        return f"{self.session_key}:{message}"

    async def process_with_thinking(self, **kwargs: Any) -> tuple[str, str]:
        return await self.process(**kwargs), "thinking"

    async def stream(self, *, message: str, **kwargs: Any):
        self._session.messages.append({"role": "user", "content": message})
        if self.on_stream is not None:
            async for chunk in self.on_stream(self.session_key, message):
                yield chunk
            return
        yield {"type": "content", "delta": f"{self.session_key}:{message}"}
        yield {
            "type": "done",
            "metadata": {"content": f"{self.session_key}:{message}"},
        }

    async def cleanup(self) -> None:
        self.cleanup_calls += 1


def _make_app_with_registry(
    *,
    on_process: Any = None,
    on_stream: Any = None,
    max_active: int = 16,
) -> tuple[Any, FakeSessions, list[RuntimeAgent]]:
    app_module._auth_required = False
    application = create_app(GatewayConfig(host="127.0.0.1", port=8080, debug=True))
    sessions = FakeSessions()
    default_agent = RuntimeAgent(
        "default",
        sessions=sessions,
        on_process=on_process,
        on_stream=on_stream,
    )
    created: list[RuntimeAgent] = [default_agent]

    async def factory(**kwargs: Any) -> RuntimeAgent:
        agent = RuntimeAgent(
            kwargs["session_key"],
            sessions=kwargs.get("session_manager") or sessions,
            on_process=kwargs.get("on_process"),
            on_stream=kwargs.get("on_stream"),
        )
        created.append(agent)
        return agent

    set_agent(default_agent)
    app_module._session_runtime_registry = SessionRuntimeRegistry(
        default_agent,
        agent_factory=factory,
        creation_kwargs=default_agent.build_creation_kwargs(),
        idle_seconds=0,
        max_active=max_active,
    )
    app_module._connection_manager = ConnectionManager()
    return application, sessions, created


@pytest.mark.asyncio
async def test_rest_different_sessions_run_on_distinct_runtimes_concurrently() -> None:
    started: list[str] = []
    both_started = asyncio.Event()

    async def on_process(session_key: str) -> None:
        started.append(session_key)
        if len(started) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=1)

    application, _sessions, created = _make_app_with_registry(on_process=on_process)
    transport = httpx.ASGITransport(app=application)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        first, second = await asyncio.gather(
            client.post("/v1/agent/chat", json={"message": "one", "session_key": "alpha"}),
            client.post("/v1/agent/chat", json={"message": "two", "session_key": "beta"}),
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert {agent.session_key for agent in created} >= {"default", "alpha", "beta"}
    assert set(started) == {"alpha", "beta"}


def test_websocket_different_sessions_stream_on_same_connection_concurrently() -> None:
    started: list[str] = []
    both_started = asyncio.Event()

    async def on_stream(session_key: str, message: str):
        started.append(session_key)
        if len(started) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=1)
        yield {"type": "content", "delta": f"{session_key}:{message}:chunk"}
        yield {
            "type": "done",
            "metadata": {"content": f"{session_key}:{message}:chunk"},
        }

    application, _sessions, created = _make_app_with_registry(on_stream=on_stream)
    client = TestClient(application)

    with client.websocket_connect("/v1/ws") as ws:
        ws.receive_json()
        ws.send_json(
            {
                "type": "request",
                "id": "alpha-req",
                "method": "chat.send",
                "params": {"message": "one", "stream": True, "session_key": "alpha"},
            }
        )
        ws.send_json(
            {
                "type": "request",
                "id": "beta-req",
                "method": "chat.send",
                "params": {"message": "two", "stream": True, "session_key": "beta"},
            }
        )

        frames: list[dict[str, Any]] = []
        for _ in range(12):
            frame = ws.receive_json()
            frames.append(frame)
            chunk_sessions = {
                item.get("data", {}).get("session_key")
                for item in frames
                if item.get("type") == "event"
                and item.get("event") == "agent.stream.chunk"
            }
            response_ids = {
                item.get("id")
                for item in frames
                if item.get("type") == "response"
            }
            if {"alpha", "beta"}.issubset(chunk_sessions) and {
                "alpha-req",
                "beta-req",
            }.issubset(response_ids):
                break

    chunk_sessions = {
        frame.get("data", {}).get("session_key")
        for frame in frames
        if frame.get("type") == "event" and frame.get("event") == "agent.stream.chunk"
    }
    assert {"alpha", "beta"}.issubset(chunk_sessions)
    assert {agent.session_key for agent in created} >= {"default", "alpha", "beta"}
    assert set(started) == {"alpha", "beta"}


def test_rest_session_close_keeps_persisted_history_and_updates_metrics() -> None:
    application, sessions, _created = _make_app_with_registry()
    client = TestClient(application)

    chat = client.post("/v1/agent/chat", json={"message": "keep me", "session_key": "alpha"})
    assert chat.status_code == 200
    assert sessions.get("alpha") is not None

    close = client.post("/v1/sessions/alpha/close")
    assert close.status_code == 200
    assert close.json()["closed"] is True

    status = client.get("/v1/agent/status")
    assert status.status_code == 200
    metrics = status.json()["data"]["runtime_metrics"]
    assert metrics["explicit_closed_total"] == 1
    assert metrics["closed_total"] == 1
    assert sessions.get("alpha") is not None


def test_websocket_session_close_and_status_metrics() -> None:
    application, _sessions, _created = _make_app_with_registry()
    client = TestClient(application)

    with client.websocket_connect("/v1/ws") as ws:
        ws.receive_json()
        ws.send_json(
            {
                "type": "request",
                "id": "switch",
                "method": "session.switch",
                "params": {"session_key": "alpha"},
            }
        )
        assert ws.receive_json()["result"]["switched"] is True

        ws.send_json(
            {
                "type": "request",
                "id": "close",
                "method": "session.close",
                "params": {"session_key": "alpha"},
            }
        )
        assert ws.receive_json()["result"]["closed"] is True

        ws.send_json(
            {
                "type": "request",
                "id": "status",
                "method": "agent.status",
                "params": {},
            }
        )
        metrics = ws.receive_json()["result"]["runtime_metrics"]
        assert metrics["explicit_closed_total"] == 1
        assert metrics["closed_total"] == 1
