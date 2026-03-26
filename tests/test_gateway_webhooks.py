from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from spoon_bot.channels.base import ChannelConfig, ChannelMode
from spoon_bot.gateway.app import create_app
from spoon_bot.gateway.config import GatewayConfig


class _DummyManager:
    def __init__(self, channels: dict[str, object]):
        self._channels = channels

    @property
    def channel_names(self) -> list[str]:
        return list(self._channels)

    def get_channel(self, name: str) -> object | None:
        return self._channels.get(name)


def _make_webhook_channel(name: str, webhook_path: str, payload: dict, *, status_code: int = 200):
    channel = SimpleNamespace()
    channel.name = name
    channel.full_name = f"{name}:default"
    channel.config = ChannelConfig(name=name, mode=ChannelMode.WEBHOOK, webhook_path=webhook_path)
    channel.handle_webhook = AsyncMock(return_value={**payload, "status_code": status_code})
    return channel


@pytest.fixture(autouse=True)
def _clear_gateway_state(monkeypatch):
    import spoon_bot.gateway.app as gateway_app

    monkeypatch.setattr(gateway_app, "_channel_manager", None)
    yield
    gateway_app._channel_manager = None


def test_webhook_router_dispatches_full_url_path(monkeypatch):
    import spoon_bot.gateway.app as gateway_app

    channel = _make_webhook_channel(
        "feishu",
        "https://example.com/feishu/webhook",
        {"challenge": "abc"},
    )
    monkeypatch.setattr(gateway_app, "_channel_manager", _DummyManager({channel.full_name: channel}))

    app = create_app(GatewayConfig())
    with TestClient(app) as client:
        response = client.post("/feishu/webhook", json={"type": "url_verification"})

    assert response.status_code == 200
    assert response.json() == {"challenge": "abc"}
    channel.handle_webhook.assert_awaited_once()


def test_webhook_router_preserves_status_code(monkeypatch):
    import spoon_bot.gateway.app as gateway_app

    channel = _make_webhook_channel(
        "telegram",
        "/telegram/webhook",
        {"ok": False, "error": "bad signature"},
        status_code=401,
    )
    monkeypatch.setattr(gateway_app, "_channel_manager", _DummyManager({channel.full_name: channel}))

    app = create_app(GatewayConfig())
    with TestClient(app) as client:
        response = client.post("/telegram/webhook", content=b"{}")

    assert response.status_code == 401
    assert response.json() == {"ok": False, "error": "bad signature"}


def test_webhook_router_returns_404_for_unknown_path(monkeypatch):
    import spoon_bot.gateway.app as gateway_app

    monkeypatch.setattr(gateway_app, "_channel_manager", _DummyManager({}))

    app = create_app(GatewayConfig())
    with TestClient(app) as client:
        response = client.post("/does-not-exist", json={})

    assert response.status_code == 404
