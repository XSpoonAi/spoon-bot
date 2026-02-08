"""Tests for image generation tool."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from spoon_bot.agent.tools.image_gen import ImageGenerateTool


class MockResponse:
    def __init__(self, content: bytes = b"img-bytes", status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://image.pollinations.ai")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("status error", request=request, response=response)


@pytest.mark.asyncio
async def test_url_construction_and_params(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tool = ImageGenerateTool(workspace=tmp_path)
    captured: dict[str, object] = {}

    async def mock_get(self, url: str, params: dict[str, object] | None = None):
        captured["url"] = url
        captured["params"] = params or {}
        return MockResponse()

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

    result = await tool.execute(
        prompt="a cute cat",
        width=512,
        height=512,
        model="flux",
        seed=42,
        enhance=True,
    )

    assert "a%20cute%20cat" in str(captured["url"])
    params = captured["params"]
    assert isinstance(params, dict)
    assert params["width"] == 512
    assert params["height"] == 512
    assert params["model"] == "flux"
    assert params["seed"] == 42
    assert params["nologo"] == "true"
    assert params["enhance"] == "true"
    assert "Image generated successfully" in result


@pytest.mark.asyncio
async def test_parameter_validation(tmp_path: Path) -> None:
    tool = ImageGenerateTool(workspace=tmp_path)

    empty_prompt = await tool.execute(prompt="   ")
    assert "prompt cannot be empty" in empty_prompt

    invalid_size = await tool.execute(prompt="cat", width=3000, height=512)
    assert "<= 2048" in invalid_size


@pytest.mark.asyncio
async def test_error_handling_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tool = ImageGenerateTool(workspace=tmp_path)

    async def mock_get(self, url: str, params: dict[str, object] | None = None):
        raise httpx.TimeoutException("timeout")

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
    result = await tool.execute(prompt="sunset")
    assert "timed out" in result


@pytest.mark.asyncio
async def test_error_handling_network(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tool = ImageGenerateTool(workspace=tmp_path)

    async def mock_get(self, url: str, params: dict[str, object] | None = None):
        raise httpx.ConnectError("connection failed")

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
    result = await tool.execute(prompt="mountain")
    assert "network error" in result


@pytest.mark.asyncio
async def test_filename_generation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tool = ImageGenerateTool(workspace=tmp_path)

    async def mock_get(self, url: str, params: dict[str, object] | None = None):
        return MockResponse(content=b"jpg")

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

    result = await tool.execute(prompt="A very cute cat with hat")

    generated_dir = tmp_path / "generated_images"
    files = list(generated_dir.glob("img_*_A_very_cute_cat_with.jpg"))
    assert len(files) == 1
    assert files[0].read_bytes() == b"jpg"
    assert str(files[0]) in result
