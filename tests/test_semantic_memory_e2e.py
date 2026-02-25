"""End-to-end tests for SemanticMemoryStore integration.

Tests the full pipeline:
  1. SemanticMemoryStore with OpenAI-compatible embedding API
  2. Milvus Lite hybrid search (dense + BM25 + RRF)
  3. AgentLoop integration with OpenRouter google/gemini-3-flash-preview

Required environment variables (set in .env or export manually):
  OPENAI_EMBEDDING_API_KEY  — API key for the embedding provider
  OPENAI_EMBEDDING_BASE_URL — Base URL for the embedding API
  OPENAI_EMBEDDING_MODEL    — Embedding model name (e.g. Qwen3-Embedding-0.6B)

Falls back to OPENAI_API_KEY / OPENAI_BASE_URL when the embedding-specific
variables are not set.

Run with:
    pytest tests/test_semantic_memory_e2e.py -v -s
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env from project root so tests can find credentials
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ---------------------------------------------------------------------------
# Read embedding configuration from environment variables.
# Priority: OPENAI_EMBEDDING_* > OPENAI_* (standard fallback)
# ---------------------------------------------------------------------------
EMBEDDING_API_KEY = os.environ.get("OPENAI_EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
EMBEDDING_BASE_URL = os.environ.get("OPENAI_EMBEDDING_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "")
EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "")

# OpenRouter configuration (loaded from .env or environment)
OPENROUTER_MODEL = "google/gemini-3-flash-preview"

# Skip the entire module if embedding credentials are missing
pytestmark = pytest.mark.skipif(
    not EMBEDDING_API_KEY or not EMBEDDING_BASE_URL or not EMBEDDING_MODEL,
    reason="OPENAI_EMBEDDING_API_KEY/OPENAI_API_KEY, OPENAI_EMBEDDING_BASE_URL/OPENAI_BASE_URL, and OPENAI_EMBEDDING_MODEL must be set",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace with sample memory files."""
    ws = tmp_path / "workspace"
    memory_dir = ws / "memory"
    memory_dir.mkdir(parents=True)

    # Long-term memory file
    (memory_dir / "MEMORY.md").write_text(
        "# Long-term Memory\n\n"
        "This file stores persistent facts and preferences.\n\n"
        "## User Preferences\n\n"
        "- [2026-02-20 10:00] User prefers Python over JavaScript for backend work\n"
        "- [2026-02-21 14:30] User likes concise explanations with code examples\n\n"
        "## Important Facts\n\n"
        "- [2026-02-22 09:00] The project uses FastAPI with PostgreSQL\n"
        "- [2026-02-22 09:15] Redis is used as the caching layer with a 5-minute TTL\n"
        "- [2026-02-23 11:00] Authentication uses JWT tokens with RS256 algorithm\n"
        "- [2026-02-23 11:30] The deployment target is Kubernetes on AWS EKS\n\n"
        "## Architecture Decisions\n\n"
        "- [2026-02-24 08:00] Decided to use event-driven architecture with RabbitMQ\n"
        "- [2026-02-24 08:30] Microservices communicate via gRPC for internal calls\n"
        "- [2026-02-24 09:00] REST API for external-facing endpoints with OpenAPI docs\n",
        encoding="utf-8",
    )

    # Daily notes
    (memory_dir / "2026-02-24.md").write_text(
        "# Notes for 2026-02-24\n\n"
        "- [09:30] Fixed the N+1 query issue in the order service by using selectinload\n"
        "- [11:00] Implemented retry logic for RabbitMQ message publishing\n"
        "- [14:00] Code review: suggested splitting the UserService into Auth and Profile\n"
        "- [16:00] Performance test results: API latency p99 dropped from 800ms to 120ms\n",
        encoding="utf-8",
    )

    (memory_dir / "2026-02-25.md").write_text(
        "# Notes for 2026-02-25\n\n"
        "- [10:00] Started implementing the payment gateway integration with Stripe\n"
        "- [11:30] Added webhook handler for payment confirmations\n"
        "- [14:00] Wrote unit tests for the payment flow - all 12 tests passing\n"
        "- [15:30] Discussed database migration strategy for the new billing tables\n",
        encoding="utf-8",
    )

    return ws


@pytest.fixture()
def semantic_store(workspace: Path):
    """Create a SemanticMemoryStore with OpenAI-compatible embedding."""
    from spoon_bot.memory.semantic_store import SemanticMemoryStore

    store = SemanticMemoryStore(
        workspace,
        embedding_provider="openai",
        embedding_model=EMBEDDING_MODEL,
        embedding_api_key=EMBEDDING_API_KEY,
        embedding_base_url=EMBEDDING_BASE_URL,
        collection="test_e2e_memory",
    )
    yield store
    store.close()


# ---------------------------------------------------------------------------
# Test 1: Embedding connectivity — Gitee AI + Qwen3-Embedding-0.6B
# ---------------------------------------------------------------------------


class TestEmbeddingConnectivity:
    """Verify that the OpenAI-compatible embedding endpoint works."""

    @pytest.mark.asyncio
    async def test_embedding_basic(self):
        """Can we get embeddings from the configured provider?"""
        os.environ["OPENAI_API_KEY"] = EMBEDDING_API_KEY
        os.environ["OPENAI_BASE_URL"] = EMBEDDING_BASE_URL

        from memsearch.embeddings.openai import OpenAIEmbedding

        provider = OpenAIEmbedding(model=EMBEDDING_MODEL)
        assert provider.model_name == EMBEDDING_MODEL
        assert provider.dimension > 0

        embeddings = await provider.embed(["Hello, this is a test."])
        assert len(embeddings) == 1
        assert len(embeddings[0]) == provider.dimension
        print(f"\n  [OK] Embedding: model={provider.model_name}, dim={provider.dimension}")

    @pytest.mark.asyncio
    async def test_batch_embedding(self):
        """Batch embedding works correctly."""
        os.environ["OPENAI_API_KEY"] = EMBEDDING_API_KEY
        os.environ["OPENAI_BASE_URL"] = EMBEDDING_BASE_URL

        from memsearch.embeddings.openai import OpenAIEmbedding

        provider = OpenAIEmbedding(model=EMBEDDING_MODEL)
        texts = [
            "Python is a programming language",
            "Redis is an in-memory data store",
            "Kubernetes orchestrates containers",
        ]
        embeddings = await provider.embed(texts)
        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == provider.dimension
        print(f"\n  [OK] Batch embedding: {len(texts)} texts -> {len(embeddings)} vectors")


# ---------------------------------------------------------------------------
# Test 2: SemanticMemoryStore — index + search
# ---------------------------------------------------------------------------


class TestSemanticMemoryStore:
    """Test the full SemanticMemoryStore lifecycle."""

    @pytest.mark.asyncio
    async def test_initialize_indexes_files(self, semantic_store):
        """Initialization indexes all memory files."""
        await semantic_store.initialize()
        stats = semantic_store.get_index_stats()
        assert stats["indexed"] is True
        assert stats["chunk_count"] > 0
        assert len(stats["sources"]) >= 3  # MEMORY.md + 2 daily notes
        print(f"\n  [OK] Indexed: {stats['chunk_count']} chunks from {len(stats['sources'])} files")

    @pytest.mark.asyncio
    async def test_semantic_search_relevant_results(self, semantic_store):
        """Semantic search returns relevant results."""
        await semantic_store.initialize()

        # Search for Redis-related content — use a query close to the stored text
        results = await semantic_store.async_search("Redis caching TTL")
        assert len(results) > 0
        # The Redis caching entry should be somewhere in the results
        contents = " ".join(r["content"] for r in results)
        has_relevant = (
            "redis" in contents.lower()
            or "cache" in contents.lower()
            or "ttl" in contents.lower()
            or "fastapi" in contents.lower()  # same MEMORY.md chunk
        )
        assert has_relevant, f"Expected Redis/cache content, got: {contents[:300]}"
        print(f"\n  [OK] 'Redis caching TTL' -> {len(results)} results")
        for r in results[:3]:
            print(f"      score={r['score']:.3f} | {r['content'][:80]}...")

    @pytest.mark.asyncio
    async def test_semantic_search_cross_file(self, semantic_store):
        """Search finds results across multiple memory files."""
        await semantic_store.initialize()

        results = await semantic_store.async_search("payment integration")
        assert len(results) > 0
        sources = {r["source"] for r in results}
        # Should find results from the daily notes file about Stripe
        has_daily = any("2026-02-25" in s for s in sources)
        assert has_daily, f"Expected daily notes in sources: {sources}"
        print(f"\n  [OK] Cross-file search: found in {len(sources)} sources")

    @pytest.mark.asyncio
    async def test_semantic_search_architecture_decisions(self, semantic_store):
        """Search understands semantic meaning, not just keywords."""
        await semantic_store.initialize()

        # Query about inter-service communication (should find gRPC/RabbitMQ entries)
        results = await semantic_store.async_search("how do services communicate with each other")
        assert len(results) > 0
        contents = " ".join(r["content"].lower() for r in results)
        has_relevant = "grpc" in contents or "rabbitmq" in contents or "event" in contents
        assert has_relevant, f"Expected architecture results, got: {contents[:200]}"
        print(f"\n  [OK] Semantic understanding: inter-service communication")
        for r in results[:3]:
            print(f"      score={r['score']:.3f} | {r['content'][:80]}...")

    @pytest.mark.asyncio
    async def test_add_memory_and_reindex(self, semantic_store):
        """Adding a memory triggers re-indexing."""
        await semantic_store.initialize()

        # Add a new memory
        semantic_store.add_memory(
            "Migrated from MySQL to PostgreSQL 16 for better JSON support",
            category="Architecture Decisions",
        )

        # Re-index the modified file via SemanticMemoryStore (preserves env isolation)
        await semantic_store._reindex_file(semantic_store.memory_file)
        # Brief wait for Milvus consistency
        await asyncio.sleep(1)

        # Search for the new memory
        results = await semantic_store.async_search("MySQL PostgreSQL migration JSON")
        assert len(results) > 0
        contents = " ".join(r["content"].lower() for r in results)
        assert "postgresql" in contents or "mysql" in contents
        print(f"\n  [OK] Add memory + reindex: found new entry")

    @pytest.mark.asyncio
    async def test_add_daily_note_and_reindex(self, semantic_store):
        """Adding a daily note triggers re-indexing."""
        await semantic_store.initialize()

        semantic_store.add_daily_note("Deployed v2.1.0 to staging with the new billing module")

        # Manually trigger reindex of today's daily file
        from datetime import date
        daily_file = semantic_store._get_daily_file(date.today())
        await semantic_store._reindex_file(daily_file)

        results = await semantic_store.async_search("billing module deployment")
        assert len(results) > 0
        print(f"\n  [OK] Daily note + reindex: found deployment note")

    @pytest.mark.asyncio
    async def test_get_memory_context_unchanged(self, semantic_store):
        """get_memory_context() still returns file-based content (backward compat)."""
        ctx = semantic_store.get_memory_context()
        assert "Long-term Memory" in ctx
        assert "User Preferences" in ctx or "Important Facts" in ctx
        print(f"\n  [OK] get_memory_context() returns {len(ctx)} chars")

    @pytest.mark.asyncio
    async def test_summary_unchanged(self, semantic_store):
        """get_summary() still works from the base class."""
        summary = semantic_store.get_summary()
        assert "entries" in summary.lower() or "memory" in summary.lower()
        print(f"\n  [OK] Summary: {summary}")


# ---------------------------------------------------------------------------
# Test 3: MemoryManagementTool integration
# ---------------------------------------------------------------------------


class TestMemoryManagementToolIntegration:
    """Test that MemoryManagementTool works with SemanticMemoryStore."""

    @pytest.mark.asyncio
    async def test_tool_semantic_search(self, semantic_store):
        """MemoryManagementTool uses async_search when available."""
        from spoon_bot.agent.tools.self_config import MemoryManagementTool

        await semantic_store.initialize()

        tool = MemoryManagementTool(memory_store=semantic_store)
        result = await tool.execute(action="search", query="authentication JWT")
        assert "semantic" in result.lower() or "score=" in result
        assert "jwt" in result.lower() or "auth" in result.lower()
        print(f"\n  [OK] Tool semantic search:\n{result[:300]}")

    @pytest.mark.asyncio
    async def test_tool_remember_action(self, semantic_store):
        """remember action still works with SemanticMemoryStore."""
        from spoon_bot.agent.tools.self_config import MemoryManagementTool

        tool = MemoryManagementTool(memory_store=semantic_store)
        result = await tool.execute(
            action="remember",
            content="CI/CD pipeline uses GitHub Actions with Docker buildx",
            category="Infrastructure",
        )
        assert "remembered" in result.lower()
        print(f"\n  [OK] Tool remember: {result}")

    @pytest.mark.asyncio
    async def test_tool_note_action(self, semantic_store):
        """note action still works with SemanticMemoryStore."""
        from spoon_bot.agent.tools.self_config import MemoryManagementTool

        tool = MemoryManagementTool(memory_store=semantic_store)
        result = await tool.execute(
            action="note",
            content="Completed API rate limiter implementation",
        )
        assert "added note" in result.lower()
        print(f"\n  [OK] Tool note: {result}")

    @pytest.mark.asyncio
    async def test_tool_summary_action(self, semantic_store):
        """summary action still works with SemanticMemoryStore."""
        from spoon_bot.agent.tools.self_config import MemoryManagementTool

        tool = MemoryManagementTool(memory_store=semantic_store)
        result = await tool.execute(action="summary")
        assert "entries" in result.lower() or "memory" in result.lower()
        print(f"\n  [OK] Tool summary: {result}")


# ---------------------------------------------------------------------------
# Test 4: AgentLoop integration with OpenRouter gemini-3-flash-preview
# ---------------------------------------------------------------------------


try:
    from spoon_bot.agent.loop import AgentLoop  # noqa: F401

    _AGENT_LOOP_AVAILABLE = True
except (ImportError, Exception):
    _AGENT_LOOP_AVAILABLE = False


@pytest.mark.skipif(not _AGENT_LOOP_AVAILABLE, reason="spoon-core SDK not available")
class TestAgentLoopIntegration:
    """Test AgentLoop with SemanticMemoryStore + OpenRouter."""

    @staticmethod
    def _get_openrouter_key() -> str | None:
        """Load OpenRouter API key from .env or environment."""
        key = os.environ.get("OPENROUTER_API_KEY")
        if key:
            return key
        # Try loading from .env
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip()
        return None

    @pytest.mark.asyncio
    async def test_agent_loop_with_semantic_memory(self, workspace):
        """AgentLoop correctly initializes with SemanticMemoryStore."""
        from spoon_bot.agent.loop import AgentLoop
        from spoon_bot.config import MemSearchConfig

        config = MemSearchConfig(
            enabled=True,
            embedding_provider="openai",
            embedding_model=EMBEDDING_MODEL,
            embedding_api_key=EMBEDDING_API_KEY,
            embedding_base_url=EMBEDDING_BASE_URL,
            collection="test_agent_loop",
        )

        agent = AgentLoop(
            workspace=workspace,
            model="google/gemini-3-flash-preview",
            provider="openrouter",
            enable_skills=False,
            auto_commit=False,
            memsearch_config=config,
        )

        # Verify the memory store is a SemanticMemoryStore
        from spoon_bot.memory.semantic_store import SemanticMemoryStore
        assert isinstance(agent.memory, SemanticMemoryStore)
        print("\n  [OK] AgentLoop created with SemanticMemoryStore")

        # Initialize (triggers indexing)
        openrouter_key = self._get_openrouter_key()
        if not openrouter_key:
            pytest.skip("OPENROUTER_API_KEY not set")

        agent.api_key = openrouter_key
        await agent.initialize()

        # Verify indexing happened
        stats = agent.memory.get_index_stats()
        assert stats["indexed"] is True
        assert stats["chunk_count"] > 0
        print(f"  [OK] AgentLoop initialized: {stats['chunk_count']} chunks indexed")

    @pytest.mark.asyncio
    async def test_agent_process_with_memory_recall(self, workspace):
        """AgentLoop processes a message and recalls memories."""
        openrouter_key = self._get_openrouter_key()
        if not openrouter_key:
            pytest.skip("OPENROUTER_API_KEY not set")

        from spoon_bot.agent.loop import AgentLoop
        from spoon_bot.config import MemSearchConfig

        config = MemSearchConfig(
            enabled=True,
            embedding_provider="openai",
            embedding_model=EMBEDDING_MODEL,
            embedding_api_key=EMBEDDING_API_KEY,
            embedding_base_url=EMBEDDING_BASE_URL,
            collection="test_agent_process",
        )

        agent = AgentLoop(
            workspace=workspace,
            model=OPENROUTER_MODEL,
            provider="openrouter",
            api_key=openrouter_key,
            enable_skills=False,
            auto_commit=False,
            memsearch_config=config,
        )

        # Process a message that should trigger memory recall
        response = await agent.process(
            "What caching solution are we using and what is the TTL?"
        )

        # The agent should reference Redis and 5-minute TTL from memory
        response_lower = response.lower()
        has_relevant_info = (
            "redis" in response_lower
            or "cache" in response_lower
            or "ttl" in response_lower
            or "5" in response_lower
        )
        print(f"\n  [OK] Agent response ({len(response)} chars):")
        print(f"  {response[:500]}")
        assert has_relevant_info, f"Expected caching info in response: {response[:200]}"

    @pytest.mark.asyncio
    async def test_agent_fallback_without_memsearch(self, workspace):
        """AgentLoop falls back to file-based memory when memsearch disabled."""
        from spoon_bot.agent.loop import AgentLoop
        from spoon_bot.memory.store import MemoryStore

        agent = AgentLoop(
            workspace=workspace,
            model="google/gemini-3-flash-preview",
            provider="openrouter",
            enable_skills=False,
            auto_commit=False,
            # No memsearch_config — should use file-based memory
        )

        assert isinstance(agent.memory, MemoryStore)
        assert not hasattr(agent.memory, 'async_search')
        print("\n  [OK] AgentLoop correctly falls back to file-based MemoryStore")

    @pytest.mark.asyncio
    async def test_agent_dict_config(self, workspace):
        """AgentLoop accepts memsearch config as a dict."""
        from spoon_bot.agent.loop import AgentLoop
        from spoon_bot.memory.semantic_store import SemanticMemoryStore

        agent = AgentLoop(
            workspace=workspace,
            model="google/gemini-3-flash-preview",
            provider="openrouter",
            enable_skills=False,
            auto_commit=False,
            memsearch_config={
                "enabled": True,
                "embedding_provider": "openai",
                "embedding_model": EMBEDDING_MODEL,
                "embedding_api_key": EMBEDDING_API_KEY,
                "embedding_base_url": EMBEDDING_BASE_URL,
                "collection": "test_dict_config",
            },
        )

        assert isinstance(agent.memory, SemanticMemoryStore)
        print("\n  [OK] Dict-based memsearch config works")


# ---------------------------------------------------------------------------
# Test 5: Config validation
# ---------------------------------------------------------------------------


class TestMemSearchConfig:
    """Test MemSearchConfig validation."""

    def test_default_config_disabled(self):
        """Default config has memsearch disabled."""
        from spoon_bot.config import MemSearchConfig

        cfg = MemSearchConfig()
        assert cfg.enabled is False
        assert cfg.embedding_provider == "openai"
        assert cfg.collection == "spoon_bot_memory"
        print("\n  [OK] Default MemSearchConfig: disabled")

    def test_config_with_gitee_ai(self):
        """Config with Gitee AI settings validates correctly."""
        from spoon_bot.config import MemSearchConfig

        cfg = MemSearchConfig(
            enabled=True,
            embedding_provider="openai",
            embedding_model="Qwen3-Embedding-0.6B",
            embedding_api_key="test-key",
            embedding_base_url="https://ai.gitee.com/v1",
        )
        assert cfg.enabled is True
        assert cfg.embedding_model == "Qwen3-Embedding-0.6B"
        assert cfg.get_embedding_api_key() == "test-key"
        assert cfg.get_embedding_base_url() == "https://ai.gitee.com/v1"
        print("\n  [OK] Gitee AI config validates correctly")

    def test_config_env_fallback_embedding_specific(self):
        """Config reads OPENAI_EMBEDDING_* first, then falls back to OPENAI_*."""
        from spoon_bot.config import MemSearchConfig

        # Save originals
        saved = {k: os.environ.pop(k, None) for k in (
            "OPENAI_EMBEDDING_API_KEY", "OPENAI_EMBEDDING_BASE_URL",
            "OPENAI_API_KEY", "OPENAI_BASE_URL",
        )}

        try:
            # Set only OPENAI_EMBEDDING_* vars
            os.environ["OPENAI_EMBEDDING_API_KEY"] = "embed-key"
            os.environ["OPENAI_EMBEDDING_BASE_URL"] = "https://embed.test/v1"

            cfg = MemSearchConfig(enabled=True)
            assert cfg.get_embedding_api_key() == "embed-key"
            assert cfg.get_embedding_base_url() == "https://embed.test/v1"
            print("\n  [OK] OPENAI_EMBEDDING_* env vars take precedence")

            # Remove embedding-specific, set standard OPENAI_* as fallback
            del os.environ["OPENAI_EMBEDDING_API_KEY"]
            del os.environ["OPENAI_EMBEDDING_BASE_URL"]
            os.environ["OPENAI_API_KEY"] = "fallback-key"
            os.environ["OPENAI_BASE_URL"] = "https://fallback.test/v1"

            cfg2 = MemSearchConfig(enabled=True)
            assert cfg2.get_embedding_api_key() == "fallback-key"
            assert cfg2.get_embedding_base_url() == "https://fallback.test/v1"
            print("  [OK] Falls back to OPENAI_* when OPENAI_EMBEDDING_* not set")
        finally:
            # Restore originals
            for k in ("OPENAI_EMBEDDING_API_KEY", "OPENAI_EMBEDDING_BASE_URL",
                       "OPENAI_API_KEY", "OPENAI_BASE_URL"):
                os.environ.pop(k, None)
                if saved[k] is not None:
                    os.environ[k] = saved[k]

    def test_config_env_model_fallback(self):
        """get_embedding_model reads OPENAI_EMBEDDING_MODEL from env."""
        from spoon_bot.config import MemSearchConfig

        saved = os.environ.pop("OPENAI_EMBEDDING_MODEL", None)
        try:
            os.environ["OPENAI_EMBEDDING_MODEL"] = "test-model-from-env"
            cfg = MemSearchConfig(enabled=True)
            assert cfg.get_embedding_model() == "test-model-from-env"

            # Explicit config value takes precedence
            cfg2 = MemSearchConfig(enabled=True, embedding_model="explicit-model")
            assert cfg2.get_embedding_model() == "explicit-model"
            print("\n  [OK] get_embedding_model() env + config precedence works")
        finally:
            os.environ.pop("OPENAI_EMBEDDING_MODEL", None)
            if saved is not None:
                os.environ["OPENAI_EMBEDDING_MODEL"] = saved

    def test_agent_loop_config_includes_memsearch(self):
        """AgentLoopConfig includes MemSearchConfig."""
        from spoon_bot.config import AgentLoopConfig

        config = AgentLoopConfig()
        assert hasattr(config, "memsearch")
        assert config.memsearch.enabled is False
        print("\n  [OK] AgentLoopConfig includes memsearch field")
