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


class DummyAgentGenericError:
    async def process(self, message: str, session_key: str | None = None, media=None):
        return "I encountered an error: An unexpected error occurred. Please try again."

    async def stream(self, message: str, session_key: str | None = None, media=None):
        yield "I encountered an error: An unexpected error occurred. Please try again."


@pytest.fixture
def app_with_error_agent(monkeypatch):
    app = FastAPI()
    app.include_router(agent_router, prefix="/v1/agent")

    dummy = DummyAgentGenericError()

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
    monkeypatch.delenv("SPOON_PROVIDER", raising=False)
    monkeypatch.delenv("SPOON_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)

    provider, model = _resolve_provider_model()
    assert provider == "anthropic"
    assert model == "claude-opus-4-20250514"


def test_resolve_provider_model_spoon_legacy_env(monkeypatch):
    """SPOON_PROVIDER/SPOON_MODEL are used when SPOON_BOT_* are not set."""
    monkeypatch.delenv("SPOON_BOT_DEFAULT_PROVIDER", raising=False)
    monkeypatch.delenv("SPOON_BOT_DEFAULT_MODEL", raising=False)
    monkeypatch.setenv("SPOON_PROVIDER", "gemini")
    monkeypatch.setenv("SPOON_MODEL", "gemini-3-flash-preview")

    provider, model = _resolve_provider_model()
    assert provider == "gemini"
    assert model == "gemini-3-flash-preview"


def test_chat_strict_passthrough_keeps_generic_error_text(app_with_error_agent):
    client = TestClient(app_with_error_agent)

    r = client.post("/v1/agent/chat", json={"message": "help me plan a backend refactor"})
    assert r.status_code == 200
    msg = r.json()["data"]["response"]

    assert msg == "I encountered an error: An unexpected error occurred. Please try again."


def test_chat_sse_strict_passthrough_keeps_stream_text_and_done(app_with_error_agent):
    client = TestClient(app_with_error_agent)

    with client.stream("POST", "/v1/agent/chat", json={"message": "stream this", "options": {"stream": True}}) as resp:
        body = "".join([
            chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
            for chunk in resp.iter_text()
        ])

    assert resp.status_code == 200
    assert "data: [DONE]" in body
    assert "I encountered an error: An unexpected error occurred. Please try again." in body
