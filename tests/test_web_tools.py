from __future__ import annotations

import pytest

from spoon_bot.agent.tools.web import (
    WebSearchTool,
    describe_web_search_capability,
    get_configured_web_search_provider,
)


class _FakeResponse:
    def __init__(self, payload: dict | None = None, text: str = "ok") -> None:
        self._payload = payload or {}
        self.status_code = 200
        self.text = text

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, dict]] = []

    async def get(self, url: str, *, params: dict, headers: dict, **kwargs):
        self.calls.append((url, params, headers))
        if "duckduckgo.com" in url:
            return _FakeResponse(text='\n                <html><body>\n                <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fweather">Weather result</a>\n                <a class="result__snippet">Today weather summary</a>\n                </body></html>\n            ')
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


def test_describe_web_search_capability_defaults_to_duckduckgo(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)

    available, message = describe_web_search_capability()

    assert available is True
    assert "duckduckgo" in message


def test_get_configured_web_search_provider_uses_duckduckgo_without_keys(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)

    assert get_configured_web_search_provider("tavily") == "duckduckgo"


def test_get_configured_web_search_provider_honors_explicit_keyed_preference(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-test-key")
    monkeypatch.setenv("BRAVE_API_KEY", "brave-test-key")

    assert get_configured_web_search_provider("tavily") == "tavily"


@pytest.mark.asyncio
async def test_web_search_defaults_to_duckduckgo_without_keys(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    fake_client = _FakeClient()
    monkeypatch.setattr("spoon_bot.agent.tools.web._get_http_client", lambda: fake_client)

    tool = WebSearchTool()
    result = await tool.execute("today weather")

    assert "Weather result" in result
    assert "https://example.com/weather" in result
    assert "duckduckgo.com/html/" in fake_client.calls[0][0]


@pytest.mark.asyncio
async def test_web_search_uses_explicit_brave_provider(monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "brave-test-key")
    fake_client = _FakeClient()
    monkeypatch.setattr("spoon_bot.agent.tools.web._get_http_client", lambda: fake_client)

    tool = WebSearchTool()
    result = await tool.execute("latest ai news", topic="news", provider="brave")

    assert "Breaking update" in result
    assert fake_client.calls[0][0].endswith("/news/search")
    assert fake_client.calls[0][2]["X-Subscription-Token"] == "brave-test-key"
