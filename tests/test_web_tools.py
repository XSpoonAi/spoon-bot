from __future__ import annotations

import pytest

from spoon_bot.agent.tools.web import (
    WebSearchTool,
    describe_web_search_capability,
    get_configured_web_search_provider,
)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.status_code = 200
        self.text = "ok"

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, dict]] = []

    async def get(self, url: str, *, params: dict, headers: dict):
        self.calls.append((url, params, headers))
        return _FakeResponse(
            {
                "results": [
                    {
                        "title": "Breaking update",
                        "url": "https://example.com/news",
                        "description": "Latest summary",
                        "age": "1h",
                    }
                ]
            }
        )


def test_describe_web_search_capability_reports_missing_provider(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)

    available, message = describe_web_search_capability()

    assert available is False
    assert "No web_search provider is configured" in message


def test_get_configured_web_search_provider_prefers_available_provider(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("BRAVE_API_KEY", "brave-test-key")

    assert get_configured_web_search_provider("tavily") == "brave"


@pytest.mark.asyncio
async def test_web_search_falls_back_to_brave_when_tavily_missing(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("BRAVE_API_KEY", "brave-test-key")
    fake_client = _FakeClient()
    monkeypatch.setattr("spoon_bot.agent.tools.web._get_http_client", lambda: fake_client)

    tool = WebSearchTool()
    result = await tool.execute("latest ai news", topic="news")

    assert "Breaking update" in result
    assert fake_client.calls[0][0].endswith("/news/search")
    assert fake_client.calls[0][2]["X-Subscription-Token"] == "brave-test-key"
