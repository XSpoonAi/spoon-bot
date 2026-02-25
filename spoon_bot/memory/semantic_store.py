"""Semantic memory store powered by memsearch (Milvus hybrid search).

This module extends the file-based MemoryStore with vector-based semantic
search via the memsearch library.  Markdown files remain the source of truth;
Milvus is a derived index that can be rebuilt at any time.

The embedding provider defaults to OpenAI-compatible endpoints, making it
easy to use alternative services (e.g. Gitee AI with Qwen3-Embedding) by
setting environment variables.  The lookup order is:

  1. Constructor parameters (``embedding_api_key``, ``embedding_base_url``)
  2. ``OPENAI_EMBEDDING_API_KEY`` / ``OPENAI_EMBEDDING_BASE_URL``
  3. ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` (standard OpenAI fallback)

This ensures embedding settings never clash with a direct OpenAI key used
for other purposes (e.g. Whisper STT/TTS).
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from spoon_bot.memory.store import MemoryStore

logger = logging.getLogger(__name__)


class SemanticMemoryStore(MemoryStore):
    """MemoryStore with semantic search via memsearch.

    Inherits all file-based functionality from :class:`MemoryStore` and adds:
    - Vector-based semantic search (``async_search``)
    - Automatic indexing of memory files
    - Optional file watcher for live re-indexing

    Parameters
    ----------
    workspace:
        Path to workspace directory (same as MemoryStore).
    embedding_provider:
        memsearch embedding provider name.  ``"openai"`` works with any
        OpenAI-compatible endpoint (set ``OPENAI_BASE_URL``).
    embedding_model:
        Model name for the embedding provider.
    embedding_api_key:
        API key for the embedding provider.  If given, sets the
        ``OPENAI_API_KEY`` environment variable for the memsearch session.
    embedding_base_url:
        Base URL for the embedding API.  If given, sets the
        ``OPENAI_BASE_URL`` environment variable for the memsearch session.
    milvus_uri:
        Milvus connection URI.  Defaults to a local Milvus Lite database
        inside the workspace.
    collection:
        Milvus collection name for isolation.
    """

    def __init__(
        self,
        workspace: Path,
        *,
        embedding_provider: str = "openai",
        embedding_model: str | None = None,
        embedding_api_key: str | None = None,
        embedding_base_url: str | None = None,
        milvus_uri: str | None = None,
        collection: str = "spoon_bot_memory",
    ) -> None:
        # Initialize file-based memory first
        super().__init__(workspace)

        self._embedding_api_key = embedding_api_key
        self._embedding_base_url = embedding_base_url

        # Default Milvus URI: local file inside workspace
        if milvus_uri is None:
            milvus_uri = str(self.workspace / "memsearch" / "milvus.db")

        # Lazy import to keep memsearch optional
        from memsearch import MemSearch

        with self._embedding_env():
            self._memsearch = MemSearch(
                paths=[str(self.memory_dir)],
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                milvus_uri=milvus_uri,
                collection=collection,
            )
        self._indexed = False
        self._watcher = None

    @contextmanager
    def _embedding_env(self):
        """Temporarily apply embedding credentials for memsearch operations."""
        prev_api_key = os.environ.get("OPENAI_API_KEY")
        prev_base_url = os.environ.get("OPENAI_BASE_URL")
        try:
            if self._embedding_api_key is not None:
                os.environ["OPENAI_API_KEY"] = self._embedding_api_key
            if self._embedding_base_url is not None:
                os.environ["OPENAI_BASE_URL"] = self._embedding_base_url
            yield
        finally:
            if prev_api_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = prev_api_key

            if prev_base_url is None:
                os.environ.pop("OPENAI_BASE_URL", None)
            else:
                os.environ["OPENAI_BASE_URL"] = prev_base_url

    async def initialize(self) -> None:
        """Index all existing memory files.

        Call this once after construction (typically in AgentLoop.initialize).
        """
        if self._indexed:
            return
        try:
            with self._embedding_env():
                count = await self._memsearch.index()
            self._indexed = True
            logger.info("SemanticMemoryStore indexed %d chunks", count)
        except Exception as exc:
            logger.error("SemanticMemoryStore indexing failed: %s", exc)

    async def async_search(
        self,
        query: str,
        *,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Semantic search across indexed memory chunks.

        Parameters
        ----------
        query:
            Natural-language search query.
        top_k:
            Maximum number of results.

        Returns
        -------
        list[dict]
            Each dict contains ``content``, ``source``, ``heading``,
            ``score``, and positional metadata.
        """
        if not self._indexed:
            await self.initialize()
        with self._embedding_env():
            return await self._memsearch.search(query, top_k=top_k)

    def search(self, query: str) -> list[str]:
        """Sync search — tries semantic first, falls back to text search.

        This override keeps backward compatibility with code that calls
        ``search()`` synchronously.  In an async context, prefer
        ``async_search()`` for full semantic results.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # We're inside an async context — can't use asyncio.run().
            # Fall back to the parent's text-based search.
            return super().search(query)

        # Not in an async context — safe to run the coroutine
        try:
            results = asyncio.run(self.async_search(query, top_k=20))
            return self._format_results(results)
        except Exception:
            return super().search(query)

    def add_memory(self, content: str, category: str = "Facts") -> None:
        """Add a fact and schedule re-indexing of the memory file."""
        super().add_memory(content, category)
        self._schedule_reindex(self.memory_file)

    def add_daily_note(self, content: str) -> None:
        """Add a daily note and schedule re-indexing."""
        super().add_daily_note(content)
        from datetime import date
        daily_file = self._get_daily_file(date.today())
        self._schedule_reindex(daily_file)

    def start_watcher(self) -> None:
        """Start background file watcher for auto-indexing."""
        if self._watcher is not None:
            return
        try:
            with self._embedding_env():
                self._watcher = self._memsearch.watch()
            logger.info("SemanticMemoryStore file watcher started")
        except Exception as exc:
            logger.warning("Failed to start file watcher: %s", exc)

    def stop_watcher(self) -> None:
        """Stop the background file watcher."""
        if self._watcher is not None:
            self._watcher.stop()
            self._watcher = None

    def close(self) -> None:
        """Release all resources."""
        self.stop_watcher()
        self._memsearch.close()

    def get_index_stats(self) -> dict[str, Any]:
        """Return indexing statistics."""
        return {
            "indexed": self._indexed,
            "chunk_count": self._memsearch.store.count(),
            "sources": list(self._memsearch.store.indexed_sources()),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _schedule_reindex(self, file_path: Path) -> None:
        """Fire-and-forget reindex of a single file.

        If a watcher is running it will handle the event automatically;
        otherwise we try to index the file immediately.
        """
        if self._watcher is not None:
            return  # watcher will pick up the file change

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._reindex_file(file_path))
        except RuntimeError:
            # Not in an async context
            pass

    async def _reindex_file(self, file_path: Path) -> None:
        """Reindex a single file."""
        try:
            with self._embedding_env():
                count = await self._memsearch.index_file(file_path)
            logger.debug("Re-indexed %s: %d chunks", file_path.name, count)
        except Exception as exc:
            logger.warning("Re-index failed for %s: %s", file_path.name, exc)

    @staticmethod
    def _format_results(results: list[dict[str, Any]]) -> list[str]:
        """Format semantic search results as plain-text lines.

        Matches the return format of :meth:`MemoryStore.search` for
        backward compatibility.
        """
        lines: list[str] = []
        for r in results:
            source = Path(r.get("source", "")).name
            heading = r.get("heading", "")
            score = r.get("score", 0)
            content = r.get("content", "").strip()
            # Truncate long content
            if len(content) > 200:
                content = content[:200] + "..."
            prefix = f"[{source}"
            if heading:
                prefix += f" > {heading}"
            prefix += f" | score={score:.3f}]"
            lines.append(f"{prefix} {content}")
        return lines
