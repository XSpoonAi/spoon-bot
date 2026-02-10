from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from spoon_bot.agent.context import ContextBuilder
from spoon_bot.gateway.api.v1.agent import router as agent_router
from spoon_bot.gateway.server import _resolve_provider_model


class DummyUser:
    user_id = "u1"
    session_key = "qa_session"
    scopes = ["admin", "agent:read", "agent:write"]


class DummyAgentForFallback:
    async def process(self, message: str, session_key: str | None = None, media=None):
        return "I encountered an error: An unexpected error occurred. Please try again."

    async def stream(self, message: str, session_key: str | None = None, media=None):
        # only emits generic error text; API layer should smart-fallback it
        yield "I encountered an error: An unexpected error occurred. Please try again."


class DummyAgentIntroTemplate:
    async def process(self, message: str, session_key: str | None = None, media=None):
        return "Welcome! 👋 I'm **Spoon AI**, your all-capable AI agent for the **Neo blockchain** and broader crypto ecosystem."


@pytest.fixture
def app_with_fallback_agent(monkeypatch):
    app = FastAPI()
    app.include_router(agent_router, prefix="/v1/agent")

    dummy = DummyAgentForFallback()

    def fake_get_agent():
        return dummy

    from spoon_bot.gateway import app as gateway_app
    from spoon_bot.gateway.api.v1 import agent as agent_mod

    monkeypatch.setattr(gateway_app, "get_agent", fake_get_agent)
    monkeypatch.setattr(agent_mod, "get_agent", fake_get_agent)

    async def fake_user():
        return DummyUser()

    from spoon_bot.gateway.auth import dependencies as deps
    app.dependency_overrides[deps.get_current_user] = fake_user

    return app


@pytest.fixture
def app_with_intro_template_agent(monkeypatch):
    app = FastAPI()
    app.include_router(agent_router, prefix="/v1/agent")

    dummy = DummyAgentIntroTemplate()

    def fake_get_agent():
        return dummy

    from spoon_bot.gateway import app as gateway_app
    from spoon_bot.gateway.api.v1 import agent as agent_mod

    monkeypatch.setattr(gateway_app, "get_agent", fake_get_agent)
    monkeypatch.setattr(agent_mod, "get_agent", fake_get_agent)

    async def fake_user():
        return DummyUser()

    from spoon_bot.gateway.auth import dependencies as deps
    app.dependency_overrides[deps.get_current_user] = fake_user

    return app


def test_context_includes_default_system_prompt(tmp_path):
    builder = ContextBuilder(tmp_path)
    prompt = builder.build_system_prompt()

    assert "practical and general-purpose local AI assistant" in prompt
    assert "Tool Usage and Degradation" in prompt
    assert "clarifying questions" in prompt


def test_resolve_provider_model_default_anthropic_opus(monkeypatch):
    monkeypatch.setenv("SPOON_BOT_DEFAULT_PROVIDER", "anthropic")
    monkeypatch.delenv("SPOON_BOT_DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)

    provider, model = _resolve_provider_model()
    assert provider == "anthropic"
    assert model == "claude-opus-4-20250514"


def test_chat_replaces_generic_error_with_smart_fallback(app_with_fallback_agent):
    client = TestClient(app_with_fallback_agent)

    r = client.post("/v1/agent/chat", json={"message": "help me plan a backend refactor"})
    assert r.status_code == 200
    msg = r.json()["data"]["response"]

    assert "无法稳定调用底层模型或工具" in msg
    assert "I encountered an error" not in msg


def test_chat_sse_replaces_generic_error_and_keeps_done(app_with_fallback_agent):
    client = TestClient(app_with_fallback_agent)

    with client.stream("POST", "/v1/agent/chat", json={"message": "stream this", "options": {"stream": True}}) as resp:
        body = "".join([
            chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
            for chunk in resp.iter_text()
        ])

    assert resp.status_code == 200
    assert "data: [DONE]" in body
    assert "无法稳定调用底层模型或工具" in body
    assert "I encountered an error" not in body


def test_chat_replaces_intro_template_with_weather_fallback(app_with_intro_template_agent):
    client = TestClient(app_with_intro_template_agent)

    r = client.post("/v1/agent/chat", json={"message": "查询上海实时天气"})
    assert r.status_code == 200
    msg = r.json()["data"]["response"]

    assert "实时天气数据源" in msg
    assert "You are spoon-bot" not in msg
