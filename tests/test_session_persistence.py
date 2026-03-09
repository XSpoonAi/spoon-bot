"""
Tests for pluggable session persistence backends.

Covers:
  - Session data-class operations
  - FileSessionStore (JSONL)
  - SQLiteSessionStore
  - PostgresSessionStore (skipped unless PG is available)
  - SessionManager with injected store
  - create_session_store factory
  - Round-trip serialization for all backends
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Generator
from unittest.mock import AsyncMock, MagicMock

import pytest

from spoon_bot.session.manager import Session, SessionManager
from spoon_bot.session.store import (
    FileSessionStore,
    SQLiteSessionStore,
    SessionStore,
    create_session_store,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tmp_dir() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def file_store(tmp_dir: Path) -> FileSessionStore:
    return FileSessionStore(tmp_dir / "sessions")


@pytest.fixture
def sqlite_store(tmp_dir: Path) -> Generator[SQLiteSessionStore, None, None]:
    store = SQLiteSessionStore(str(tmp_dir / "test_sessions.db"))
    yield store
    store.close()


def _sample_session(key: str = "test-session") -> Session:
    s = Session(session_key=key)
    s.add_message("user", "Hello!")
    s.add_message("assistant", "Hi there, how can I help you?")
    s.metadata = {"language": "en", "topic": "greeting"}
    return s


# ============================================================================
# §1  Session data-class
# ============================================================================


class TestSessionDataClass:
    def test_add_message(self):
        s = Session(session_key="s1")
        s.add_message("user", "hello")
        assert len(s.messages) == 1
        assert s.messages[0]["role"] == "user"
        assert s.messages[0]["content"] == "hello"
        assert "timestamp" in s.messages[0]

    def test_get_history(self):
        s = _sample_session()
        h = s.get_history()
        assert len(h) == 2
        assert h[0] == {"role": "user", "content": "Hello!"}
        assert "timestamp" not in h[0]

    def test_clear(self):
        s = _sample_session()
        s.clear()
        assert len(s.messages) == 0

    def test_to_dict_from_dict_roundtrip(self):
        s1 = _sample_session()
        d = s1.to_dict()
        s2 = Session.from_dict(d)
        assert s2.session_key == s1.session_key
        assert len(s2.messages) == len(s1.messages)
        assert s2.metadata == s1.metadata

    def test_add_message_extra_kwargs(self):
        s = Session(session_key="s1")
        s.add_message("user", "hi", tool_calls=[{"name": "shell"}])
        assert s.messages[0]["tool_calls"] == [{"name": "shell"}]

    def test_updated_at_changes(self):
        s = Session(session_key="s1")
        t1 = s.updated_at
        s.add_message("user", "x")
        assert s.updated_at >= t1


# ============================================================================
# §2  FileSessionStore
# ============================================================================


class TestFileSessionStore:
    def test_save_and_load(self, file_store: FileSessionStore):
        s = _sample_session()
        file_store.save_session(s)
        loaded = file_store.load_session("test-session")
        assert loaded is not None
        assert loaded.session_key == "test-session"
        assert len(loaded.messages) == 2
        assert loaded.metadata["language"] == "en"

    def test_load_nonexistent(self, file_store: FileSessionStore):
        assert file_store.load_session("nonexistent") is None

    def test_delete(self, file_store: FileSessionStore):
        s = _sample_session()
        file_store.save_session(s)
        assert file_store.delete_session("test-session") is True
        assert file_store.load_session("test-session") is None

    def test_delete_nonexistent(self, file_store: FileSessionStore):
        assert file_store.delete_session("nope") is False

    def test_list_keys(self, file_store: FileSessionStore):
        file_store.save_session(_sample_session("a"))
        file_store.save_session(_sample_session("b"))
        keys = file_store.list_session_keys()
        assert "a" in keys and "b" in keys

    def test_overwrite(self, file_store: FileSessionStore):
        s = _sample_session()
        file_store.save_session(s)
        s.add_message("user", "Follow-up question")
        file_store.save_session(s)
        loaded = file_store.load_session("test-session")
        assert len(loaded.messages) == 3

    def test_special_chars_in_key(self, file_store: FileSessionStore):
        s = _sample_session("user@example.com:session/1")
        file_store.save_session(s)
        loaded = file_store.load_session("user@example.com:session/1")
        assert loaded is not None
        assert loaded.session_key == "user@example.com:session/1"


# ============================================================================
# §3  SQLiteSessionStore
# ============================================================================


class TestSQLiteSessionStore:
    def test_save_and_load(self, sqlite_store: SQLiteSessionStore):
        s = _sample_session()
        sqlite_store.save_session(s)
        loaded = sqlite_store.load_session("test-session")
        assert loaded is not None
        assert loaded.session_key == "test-session"
        assert len(loaded.messages) == 2
        assert loaded.messages[0]["role"] == "user"
        assert loaded.messages[0]["content"] == "Hello!"
        assert loaded.metadata["language"] == "en"

    def test_load_nonexistent(self, sqlite_store: SQLiteSessionStore):
        assert sqlite_store.load_session("nonexistent") is None

    def test_delete(self, sqlite_store: SQLiteSessionStore):
        sqlite_store.save_session(_sample_session())
        assert sqlite_store.delete_session("test-session") is True
        assert sqlite_store.load_session("test-session") is None

    def test_delete_cascade(self, sqlite_store: SQLiteSessionStore):
        """Deleting a session should cascade to messages."""
        sqlite_store.save_session(_sample_session())
        sqlite_store.delete_session("test-session")
        # Verify messages are gone too (use _exec to ensure conn is closed)
        import sqlite3
        conn = sqlite3.connect(sqlite_store._db_path)
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_key=?", ("test-session",)
            ).fetchone()[0]
        finally:
            conn.close()
        assert count == 0

    def test_delete_nonexistent(self, sqlite_store: SQLiteSessionStore):
        assert sqlite_store.delete_session("nope") is False

    def test_list_keys(self, sqlite_store: SQLiteSessionStore):
        sqlite_store.save_session(_sample_session("a"))
        sqlite_store.save_session(_sample_session("b"))
        keys = sqlite_store.list_session_keys()
        assert "a" in keys and "b" in keys

    def test_overwrite_upsert(self, sqlite_store: SQLiteSessionStore):
        s = _sample_session()
        sqlite_store.save_session(s)
        s.add_message("user", "more")
        sqlite_store.save_session(s)
        loaded = sqlite_store.load_session("test-session")
        assert len(loaded.messages) == 3

    def test_message_ordering(self, sqlite_store: SQLiteSessionStore):
        s = Session(session_key="order-test")
        for i in range(10):
            s.add_message("user" if i % 2 == 0 else "assistant", f"msg-{i}")
        sqlite_store.save_session(s)
        loaded = sqlite_store.load_session("order-test")
        for i, msg in enumerate(loaded.messages):
            assert msg["content"] == f"msg-{i}"

    def test_extra_fields_preserved(self, sqlite_store: SQLiteSessionStore):
        s = Session(session_key="extra-test")
        s.add_message("assistant", "result", tool_calls=[{"name": "shell", "args": {"cmd": "ls"}}])
        sqlite_store.save_session(s)
        loaded = sqlite_store.load_session("extra-test")
        assert loaded.messages[0]["tool_calls"] == [{"name": "shell", "args": {"cmd": "ls"}}]

    def test_timestamp_preserved(self, sqlite_store: SQLiteSessionStore):
        s = _sample_session()
        sqlite_store.save_session(s)
        loaded = sqlite_store.load_session("test-session")
        assert "timestamp" in loaded.messages[0]

    def test_metadata_preserved(self, sqlite_store: SQLiteSessionStore):
        s = _sample_session()
        s.metadata = {"complex": {"nested": True}, "count": 42}
        sqlite_store.save_session(s)
        loaded = sqlite_store.load_session("test-session")
        assert loaded.metadata["complex"]["nested"] is True
        assert loaded.metadata["count"] == 42

    def test_empty_session(self, sqlite_store: SQLiteSessionStore):
        s = Session(session_key="empty")
        sqlite_store.save_session(s)
        loaded = sqlite_store.load_session("empty")
        assert loaded is not None
        assert len(loaded.messages) == 0

    def test_unicode_content(self, sqlite_store: SQLiteSessionStore):
        s = Session(session_key="unicode")
        s.add_message("user", "你好世界 🌍 مرحبا")
        sqlite_store.save_session(s)
        loaded = sqlite_store.load_session("unicode")
        assert loaded.messages[0]["content"] == "你好世界 🌍 مرحبا"


# ============================================================================
# §4  SessionManager with injected store
# ============================================================================


class TestSessionManagerWithStore:
    def test_file_backend(self, tmp_dir: Path):
        sm = SessionManager(workspace=tmp_dir)
        s = sm.get_or_create("s1")
        s.add_message("user", "hello")
        sm.save(s)
        # New manager instance loads from disk
        sm2 = SessionManager(workspace=tmp_dir)
        loaded = sm2.get_or_create("s1")
        assert len(loaded.messages) == 1

    def test_sqlite_backend(self, tmp_dir: Path):
        store = SQLiteSessionStore(str(tmp_dir / "mgr.db"))
        sm = SessionManager(store=store)
        s = sm.get_or_create("s1")
        s.add_message("user", "hello")
        sm.save(s)
        # New manager same store
        sm2 = SessionManager(store=SQLiteSessionStore(str(tmp_dir / "mgr.db")))
        loaded = sm2.get("s1")
        assert loaded is not None
        assert len(loaded.messages) == 1

    def test_get_returns_none_when_missing(self, tmp_dir: Path):
        sm = SessionManager(workspace=tmp_dir)
        assert sm.get("nonexistent") is None

    def test_delete(self, tmp_dir: Path):
        store = SQLiteSessionStore(str(tmp_dir / "del.db"))
        sm = SessionManager(store=store)
        s = sm.get_or_create("to-delete")
        s.add_message("user", "bye")
        sm.save(s)
        assert sm.delete("to-delete") is True
        assert sm.get("to-delete") is None

    def test_list_sessions(self, tmp_dir: Path):
        store = SQLiteSessionStore(str(tmp_dir / "list.db"))
        sm = SessionManager(store=store)
        sm.get_or_create("a")
        sm.get_or_create("b")
        sm.save(sm.get("a"))
        sm.save(sm.get("b"))
        keys = sm.list_sessions()
        assert "a" in keys and "b" in keys

    def test_cache_hit(self, tmp_dir: Path):
        store = SQLiteSessionStore(str(tmp_dir / "cache.db"))
        sm = SessionManager(store=store)
        s1 = sm.get_or_create("cached")
        s2 = sm.get_or_create("cached")
        assert s1 is s2  # same object from cache

    def test_no_workspace_no_store_raises(self):
        with pytest.raises(ValueError, match="Either 'workspace' or 'store'"):
            SessionManager(workspace=None)

    def test_cache_capacity_min_boundary(self, tmp_dir: Path):
        store = SQLiteSessionStore(str(tmp_dir / "cap-min.db"))
        sm = SessionManager(store=store, max_cached_sessions=0)
        sm.get_or_create("s1")
        sm.get_or_create("s2")
        assert len(sm._sessions) == 1

    def test_cache_capacity_enforced(self, tmp_dir: Path):
        store = SQLiteSessionStore(str(tmp_dir / "cap.db"))
        sm = SessionManager(store=store, max_cached_sessions=2)
        sm.get_or_create("s1")
        sm.get_or_create("s2")
        sm.get_or_create("s3")
        assert len(sm._sessions) == 2


class _FakeRuntimeMemory:
    def __init__(self) -> None:
        self.cleared = False

    def clear(self) -> None:
        self.cleared = True


class _FakeRuntimeAgent:
    def __init__(self) -> None:
        self.memory = _FakeRuntimeMemory()
        self.calls: list[tuple[str, str]] = []

    async def add_message(self, role: str, content: str, **kwargs) -> None:
        self.calls.append((role, content))


class TestAgentLoopSessionHydration:
    @pytest.mark.asyncio
    async def test_runtime_history_injected_from_persisted_session(self):
        from spoon_bot.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        loop._agent = _FakeRuntimeAgent()
        loop._session = Session(session_key="persisted")
        loop._session.add_message("user", "我叫Alice，请记住")
        loop._session.add_message("assistant", "好的，我会记住你叫Alice。")

        injected = await AgentLoop._sync_runtime_history_from_session(loop)

        assert loop._agent.memory.cleared is True
        assert injected == 2
        assert loop._agent.calls == [
            ("user", "我叫Alice，请记住"),
            ("assistant", "好的，我会记住你叫Alice。"),
        ]


class _NoChunkRuntimeAgent:
    """Runtime agent that finishes run() but never emits queue chunks."""

    def __init__(self, final_content: str) -> None:
        self.task_done = asyncio.Event()
        self.output_queue: asyncio.Queue = asyncio.Queue()
        self.state = "IDLE"
        self._final_content = final_content

    async def run(self, **kwargs):
        return type("RunResult", (), {"content": self._final_content})()


class _ChunkedRuntimeAgent:
    """Runtime agent that emits real incremental chunks through output_queue."""

    def __init__(self, chunks: list[str]) -> None:
        self.task_done = asyncio.Event()
        self.output_queue: asyncio.Queue = asyncio.Queue()
        self.state = "IDLE"
        self._chunks = chunks

    async def run(self, **kwargs):
        for chunk in self._chunks:
            await self.output_queue.put({"content": chunk})
            await asyncio.sleep(0)
        return type("RunResult", (), {"content": "".join(self._chunks)})()


class TestAgentLoopStreamFallback:
    @pytest.mark.asyncio
    async def test_stream_falls_back_to_run_result_when_no_chunks(self):
        from spoon_bot.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        loop._initialized = True
        loop._agent = _NoChunkRuntimeAgent("fallback from run result")
        loop._session = Session(session_key="stream_fallback")
        loop.sessions = MagicMock()
        loop.sessions.save = MagicMock()
        loop.memory = MagicMock()
        loop.memory.get_memory_context = MagicMock(return_value=None)
        loop.context = MagicMock()
        loop._prepare_request_context = AsyncMock(return_value=None)

        chunks = []
        async for chunk in AgentLoop.stream(loop, message="hello"):
            chunks.append(chunk)

        content_chunks = [c for c in chunks if c["type"] == "content"]
        done_chunks = [c for c in chunks if c["type"] == "done"]

        assert len(content_chunks) >= 1
        assert content_chunks[-1]["delta"] == "fallback from run result"
        assert len(done_chunks) == 1
        assert done_chunks[0]["metadata"]["content"] == "fallback from run result"

        assert loop._session.messages[-1]["role"] == "assistant"
        assert loop._session.messages[-1]["content"] == "fallback from run result"
        loop.sessions.save.assert_called_once_with(loop._session)

    @pytest.mark.asyncio
    async def test_stream_preserves_incremental_queue_chunks(self):
        from spoon_bot.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        loop._initialized = True
        loop._agent = _ChunkedRuntimeAgent(["Hel", "lo"])
        loop._session = Session(session_key="stream_incremental")
        loop.sessions = MagicMock()
        loop.sessions.save = MagicMock()
        loop.memory = MagicMock()
        loop.memory.get_memory_context = MagicMock(return_value=None)
        loop.context = MagicMock()
        loop._prepare_request_context = AsyncMock(return_value=None)

        chunks = []
        async for chunk in AgentLoop.stream(loop, message="hello"):
            chunks.append(chunk)

        content_chunks = [c for c in chunks if c["type"] == "content"]
        done_chunks = [c for c in chunks if c["type"] == "done"]
        deltas = [c["delta"] for c in content_chunks]

        assert deltas == ["Hel", "lo"]
        assert len(done_chunks) == 1
        assert done_chunks[0]["metadata"]["content"] == "Hello"
        assert loop._session.messages[-1]["content"] == "Hello"


# ============================================================================
# §5  create_session_store factory
# ============================================================================


class TestCreateSessionStoreFactory:
    def test_file_backend(self, tmp_dir: Path):
        store = create_session_store("file", workspace=tmp_dir)
        assert isinstance(store, FileSessionStore)

    def test_sqlite_backend(self, tmp_dir: Path):
        store = create_session_store("sqlite", db_path=str(tmp_dir / "f.db"))
        assert isinstance(store, SQLiteSessionStore)

    def test_file_requires_workspace(self):
        with pytest.raises(ValueError, match="workspace"):
            create_session_store("file")

    def test_postgres_requires_dsn(self):
        with pytest.raises(ValueError, match="dsn"):
            create_session_store("postgres")

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            create_session_store("redis")

    def test_case_insensitive(self, tmp_dir: Path):
        store = create_session_store("SQLITE", db_path=str(tmp_dir / "ci.db"))
        assert isinstance(store, SQLiteSessionStore)

    def test_sqlite3_alias(self, tmp_dir: Path):
        store = create_session_store("sqlite3", db_path=str(tmp_dir / "sqlite3.db"))
        assert isinstance(store, SQLiteSessionStore)

    def test_pg_alias_requires_dsn(self):
        with pytest.raises(ValueError, match="dsn"):
            create_session_store("pg")

    def test_postgresql_alias_requires_dsn(self):
        with pytest.raises(ValueError, match="dsn"):
            create_session_store("postgresql")


# ============================================================================
# §6  Round-trip: save -> load -> compare for all in-process backends
# ============================================================================


class TestRoundTrip:
    """Verify that save -> load produces identical Session for all backends."""

    @pytest.fixture(params=["file", "sqlite"])
    def store(self, request, tmp_dir: Path) -> SessionStore:
        if request.param == "file":
            return FileSessionStore(tmp_dir / "rt-sessions")
        else:
            return SQLiteSessionStore(str(tmp_dir / "rt.db"))

    def test_full_roundtrip(self, store: SessionStore):
        original = _sample_session("roundtrip-test")
        # Add a message with extra fields
        original.add_message("assistant", "Here is the result", tool_calls=[
            {"name": "web_search", "args": {"query": "weather"}}
        ])
        original.metadata = {"user_id": "u123", "tags": ["test", "roundtrip"]}

        store.save_session(original)
        loaded = store.load_session("roundtrip-test")

        assert loaded is not None
        assert loaded.session_key == original.session_key
        assert len(loaded.messages) == len(original.messages)
        assert loaded.metadata == original.metadata

        for orig_msg, loaded_msg in zip(original.messages, loaded.messages):
            assert orig_msg["role"] == loaded_msg["role"]
            assert orig_msg["content"] == loaded_msg["content"]
            # Extra fields
            if "tool_calls" in orig_msg:
                assert loaded_msg["tool_calls"] == orig_msg["tool_calls"]

    def test_empty_session_roundtrip(self, store: SessionStore):
        s = Session(session_key="empty-rt")
        store.save_session(s)
        loaded = store.load_session("empty-rt")
        assert loaded is not None
        assert len(loaded.messages) == 0

    def test_large_conversation_roundtrip(self, store: SessionStore):
        s = Session(session_key="large-rt")
        for i in range(100):
            s.add_message("user" if i % 2 == 0 else "assistant", f"Message number {i}")
        store.save_session(s)
        loaded = store.load_session("large-rt")
        assert len(loaded.messages) == 100
        for i, msg in enumerate(loaded.messages):
            assert msg["content"] == f"Message number {i}"


# ============================================================================
# §7  PostgresSessionStore (requires live PG — skip by default)
# ============================================================================


@pytest.mark.skipif(
    not os.environ.get("TEST_PG_DSN"),
    reason="Set TEST_PG_DSN to run PostgreSQL session store tests",
)
class TestPostgresSessionStore:
    @pytest.fixture
    def pg_store(self):
        from spoon_bot.session.store import PostgresSessionStore
        store = PostgresSessionStore(os.environ["TEST_PG_DSN"])
        yield store
        # Cleanup
        import psycopg2
        conn = psycopg2.connect(os.environ["TEST_PG_DSN"])
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM messages")
                cur.execute("DELETE FROM sessions")
            conn.commit()
        finally:
            conn.close()

    def test_save_and_load(self, pg_store):
        s = _sample_session()
        pg_store.save_session(s)
        loaded = pg_store.load_session("test-session")
        assert loaded is not None
        assert len(loaded.messages) == 2

    def test_delete(self, pg_store):
        pg_store.save_session(_sample_session())
        assert pg_store.delete_session("test-session") is True
        assert pg_store.load_session("test-session") is None

    def test_list_keys(self, pg_store):
        pg_store.save_session(_sample_session("pg-a"))
        pg_store.save_session(_sample_session("pg-b"))
        keys = pg_store.list_session_keys()
        assert "pg-a" in keys and "pg-b" in keys
