from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_semantic_memory_store_uses_temporary_embedding_env(monkeypatch, tmp_path: Path):
    constructor_env: dict[str, str | None] = {}
    index_env: dict[str, str | None] = {}
    search_env: dict[str, str | None] = {}

    class _FakeStore:
        @staticmethod
        def count() -> int:
            return 0

        @staticmethod
        def indexed_sources() -> list[str]:
            return []

    class FakeMemSearch:
        def __init__(self, *args, **kwargs):
            constructor_env["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
            constructor_env["OPENAI_BASE_URL"] = os.environ.get("OPENAI_BASE_URL")
            self.store = _FakeStore()

        async def index(self, *args, **kwargs):
            index_env["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
            index_env["OPENAI_BASE_URL"] = os.environ.get("OPENAI_BASE_URL")
            return 1

        async def search(self, query: str, top_k: int = 10):
            search_env["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
            search_env["OPENAI_BASE_URL"] = os.environ.get("OPENAI_BASE_URL")
            return [{"content": query, "source": "MEMORY.md", "heading": "", "score": 1.0}]

        async def index_file(self, file_path):
            return 1

        def watch(self):
            return types.SimpleNamespace(stop=lambda: None)

        def close(self):
            return None

    fake_memsearch = types.ModuleType("memsearch")
    fake_memsearch.MemSearch = FakeMemSearch
    monkeypatch.setitem(sys.modules, "memsearch", fake_memsearch)

    semantic_mod = importlib.import_module("spoon_bot.memory.semantic_store")
    semantic_mod = importlib.reload(semantic_mod)
    SemanticMemoryStore = semantic_mod.SemanticMemoryStore

    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://llm.example/v1")

    store = SemanticMemoryStore(
        workspace=tmp_path / "workspace",
        embedding_provider="openai",
        embedding_model="dummy-embedding-model",
        embedding_api_key="embed-key",
        embedding_base_url="https://embed.example/v1",
        collection="test-env-isolation",
    )

    assert constructor_env["OPENAI_API_KEY"] == "embed-key"
    assert constructor_env["OPENAI_BASE_URL"] == "https://embed.example/v1"
    assert os.environ.get("OPENAI_API_KEY") == "llm-key"
    assert os.environ.get("OPENAI_BASE_URL") == "https://llm.example/v1"

    await store.initialize()
    assert index_env["OPENAI_API_KEY"] == "embed-key"
    assert index_env["OPENAI_BASE_URL"] == "https://embed.example/v1"
    assert os.environ.get("OPENAI_API_KEY") == "llm-key"
    assert os.environ.get("OPENAI_BASE_URL") == "https://llm.example/v1"

    results = await store.async_search("test query")
    assert results
    assert search_env["OPENAI_API_KEY"] == "embed-key"
    assert search_env["OPENAI_BASE_URL"] == "https://embed.example/v1"
    assert os.environ.get("OPENAI_API_KEY") == "llm-key"
    assert os.environ.get("OPENAI_BASE_URL") == "https://llm.example/v1"
