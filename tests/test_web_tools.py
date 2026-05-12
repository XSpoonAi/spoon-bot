from __future__ import annotations

import httpx
import pytest

from spoon_bot.agent.tools.web import (
    WebSearchTool,
    describe_web_search_capability,
    get_configured_web_search_provider,
)


class _FakeResponse:
    def __init__(self, payload: dict | None = None, text: str = "ok", status_code: int = 200) -> None:
        self._payload = payload or {}
        self.status_code = status_code
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://example.com")
            response = httpx.Response(self.status_code, request=request, text=self.text)
            raise httpx.HTTPStatusError("fake error", request=request, response=response)
        return None

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    def __init__(self, *, tavily_responses: list[_FakeResponse] | None = None) -> None:
        self.calls: list[tuple[str, dict, dict]] = []
        self.tavily_responses = list(tavily_responses or [])

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

    async def post(self, url: str, *, json: dict, headers: dict, **kwargs):
        self.calls.append((url, json, headers))
        if self.tavily_responses:
            return self.tavily_responses.pop(0)
        return _FakeResponse(
            {
                "answer": "Tavily answer",
                "results": [
                    {
                        "title": "Tavily result",
                        "url": "https://example.com/tavily",
                        "content": "Tavily content",
                    }
                ],
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
async def test_web_search_prefers_tavily_when_key_is_configured(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-test-key")
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    fake_client = _FakeClient()
    monkeypatch.setattr("spoon_bot.agent.tools.web._get_http_client", lambda: fake_client)

    tool = WebSearchTool()
    result = await tool.execute("latest ai news")

    assert "Tavily answer" in result
    assert fake_client.calls[0][0] == "https://api.tavily.com/search"
    assert fake_client.calls[0][2]["Authorization"] == "Bearer tavily-test-key"


@pytest.mark.asyncio
async def test_web_search_rotates_comma_separated_tavily_keys(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "bad-key, good-key")
    tavily_responses = [
        _FakeResponse(text="quota exceeded", status_code=429),
        _FakeResponse(
            {
                "answer": "Rotated Tavily answer",
                "results": [
                    {
                        "title": "Rotated result",
                        "url": "https://example.com/rotated",
                        "content": "Recovered with second key",
                    }
                ],
            }
        ),
    ]
    fake_client = _FakeClient(tavily_responses=tavily_responses)
    monkeypatch.setattr("spoon_bot.agent.tools.web._get_http_client", lambda: fake_client)

    tool = WebSearchTool()
    result = await tool.execute("provider rotation test")

    assert "Rotated Tavily answer" in result
    assert fake_client.calls[0][2]["Authorization"] == "Bearer bad-key"
    assert fake_client.calls[1][2]["Authorization"] == "Bearer good-key"


@pytest.mark.asyncio
async def test_web_search_falls_back_to_duckduckgo_after_tavily_failures(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "bad-key,also-bad")
    tavily_responses = [
        _FakeResponse(text="quota exceeded", status_code=429),
        _FakeResponse(text="upstream error", status_code=500),
    ]
    fake_client = _FakeClient(tavily_responses=tavily_responses)
    monkeypatch.setattr("spoon_bot.agent.tools.web._get_http_client", lambda: fake_client)

    tool = WebSearchTool()
    result = await tool.execute("fallback test")

    assert "Weather result" in result
    assert fake_client.calls[0][0] == "https://api.tavily.com/search"
    assert fake_client.calls[1][0] == "https://api.tavily.com/search"
    assert fake_client.calls[2][0] == "https://duckduckgo.com/html/"


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
