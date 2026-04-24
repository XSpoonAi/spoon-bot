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
from types import SimpleNamespace
from typing import Any, Generator
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

    def test_get_messages_preserves_metadata(self):
        s = Session(session_key="s1")
        s.add_message(
            "user",
            "see attachment",
            media=["/workspace/uploads/demo.png"],
            attachments=[{"uri": "/workspace/uploads/demo.png", "name": "demo.png"}],
        )
        messages = s.get_messages()
        assert messages[0]["media"] == ["/workspace/uploads/demo.png"]
        assert messages[0]["attachments"] == [{"uri": "/workspace/uploads/demo.png", "name": "demo.png"}]

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
        s.add_message("user", "Hello world 🌍 مرحبا")
        sqlite_store.save_session(s)
        loaded = sqlite_store.load_session("unicode")
        assert loaded.messages[0]["content"] == "Hello world 🌍 مرحبا"


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
        self.calls: list[tuple[str, Any, dict]] = []

    async def add_message(self, role: str, content: Any, **kwargs) -> None:
        self.calls.append((role, content, kwargs))


def _assert_multimodal_user_call(
    call: tuple[str, Any, dict],
    *,
    expected_text: str,
    expected_data_url_suffix: str = "cG5n",
) -> None:
    role, content, kwargs = call
    assert role == "user"
    assert kwargs == {}
    assert isinstance(content, list)
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].endswith(expected_data_url_suffix)
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[-1] == {"type": "text", "text": expected_text}


class TestAgentLoopSessionHydration:
    @pytest.mark.asyncio
    async def test_runtime_history_injected_from_persisted_session(self, tmp_dir: Path):
        from spoon_bot.agent.loop import AgentLoop
        from spoon_bot.agent.context import ContextBuilder

        workspace = tmp_dir / "workspace"
        uploads = workspace / "uploads"
        uploads.mkdir(parents=True)
        attachment_path = uploads / "alice.png"
        attachment_path.write_bytes(b"png")

        loop = AgentLoop.__new__(AgentLoop)
        loop._agent = _FakeRuntimeAgent()
        loop.workspace = workspace
        loop.context = ContextBuilder(workspace)
        loop._session = Session(session_key="persisted")
        loop._session.add_message(
            "user",
            "My name is Alice. Please remember it.",
            media=[str(attachment_path)],
            attachments=[{"uri": str(attachment_path), "name": "alice.png"}],
        )
        loop._session.add_message("assistant", "Okay, I will remember your name is Alice.")

        injected = await AgentLoop._sync_runtime_history_from_session(loop)

        assert loop._agent.memory.cleared is True
        assert injected == 2
        _assert_multimodal_user_call(
            loop._agent.calls[0],
            expected_text=(
                f"My name is Alice. Please remember it.\n\n"
                f"Attached workspace files (source of truth for this request):\n"
                f"- {attachment_path} (name: alice.png)\n"
                f"Use these attached workspace files as the primary source of truth for this request."
            ),
        )
        assert loop._agent.calls[1] == ("assistant", "Okay, I will remember your name is Alice.", {})

    @pytest.mark.asyncio
    async def test_runtime_history_accepts_sandbox_alias_and_relative_refs(self, tmp_dir: Path):
        from spoon_bot.agent.loop import AgentLoop
        from spoon_bot.agent.context import ContextBuilder

        workspace = tmp_dir / "workspace"
        uploads = workspace / "uploads"
        uploads.mkdir(parents=True)
        attachment_path = uploads / "alice.png"
        attachment_path.write_bytes(b"png")

        loop = AgentLoop.__new__(AgentLoop)
        loop._agent = _FakeRuntimeAgent()
        loop.workspace = workspace
        loop.context = ContextBuilder(workspace)
        loop._session = Session(session_key="persisted")
        loop._session.add_message(
            "user",
            "Please look at this attachment.",
            media=["uploads/alice.png"],
            attachments=[{"uri": "/workspace/uploads/alice.png", "name": "alice.png"}],
        )

        injected = await AgentLoop._sync_runtime_history_from_session(loop)

        assert injected == 1
        _assert_multimodal_user_call(
            loop._agent.calls[0],
            expected_text=(
                "Please look at this attachment.\n\n"
                "Attached workspace files (source of truth for this request):\n"
                "- /workspace/uploads/alice.png (name: alice.png)\n"
                "Use these attached workspace files as the primary source of truth for this request."
            ),
        )

    @pytest.mark.asyncio
    async def test_runtime_history_skips_interrupted_user_turns(self, tmp_dir: Path):
        from spoon_bot.agent.loop import AgentLoop
        from spoon_bot.agent.context import ContextBuilder

        workspace = tmp_dir / "workspace"
        workspace.mkdir(parents=True)

        loop = AgentLoop.__new__(AgentLoop)
        loop._agent = _FakeRuntimeAgent()
        loop.workspace = workspace
        loop.context = ContextBuilder(workspace)
        loop._session = Session(session_key="persisted")
        loop._session.add_message("user", "Completed request", turn_state="completed")
        loop._session.add_message("assistant", "Completed answer")
        loop._session.add_message(
            "user",
            "Cancelled request that must not be replayed",
            turn_state="interrupted",
        )

        injected = await AgentLoop._sync_runtime_history_from_session(loop)

        assert injected == 2
        assert loop._agent.calls == [
            ("user", "Completed request", {}),
            ("assistant", "Completed answer", {}),
        ]

    def test_strip_attachment_context_recovers_original_user_text(self):
        from spoon_bot.agent.loop import _ensure_attachment_context, _strip_attachment_context

        attachments = [{"uri": "/workspace/uploads/alice.png", "name": "alice.png"}]

        injected = _ensure_attachment_context("Original question", attachments)
        assert _strip_attachment_context(injected, attachments) == "Original question"

        attachment_only = _ensure_attachment_context("", attachments)
        assert _strip_attachment_context(attachment_only, attachments) == ""


class _NoChunkRuntimeAgent:
    """Runtime agent that finishes run() but never emits queue chunks."""

    def __init__(self, final_content: str) -> None:
        self.task_done = asyncio.Event()
        self.output_queue: asyncio.Queue = asyncio.Queue()
        self.state = "IDLE"
        self._final_content = final_content
        self.add_message_calls: list[tuple[str, Any, dict]] = []
        self.run_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def add_message(self, role: str, content: Any, **kwargs) -> None:
        self.add_message_calls.append((role, content, kwargs))

    async def run(self, *args, **kwargs):
        self.run_calls.append((args, kwargs))
        return type("RunResult", (), {"content": self._final_content})()


class _FailingStreamRuntimeAgent(_NoChunkRuntimeAgent):
    """Runtime agent whose stream run fails before yielding assistant content."""

    async def run(self, *args, **kwargs):
        self.run_calls.append((args, kwargs))
        raise RuntimeError("upstream exploded")


class _ChunkedRuntimeAgent:
    """Runtime agent that emits real incremental chunks through output_queue."""

    def __init__(self, chunks: list[str]) -> None:
        self.task_done = asyncio.Event()
        self.output_queue: asyncio.Queue = asyncio.Queue()
        self.state = "IDLE"
        self._chunks = chunks
        self.add_message_calls: list[tuple[str, Any, dict]] = []
        self.run_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def add_message(self, role: str, content: Any, **kwargs) -> None:
        self.add_message_calls.append((role, content, kwargs))

    async def run(self, *args, **kwargs):
        self.run_calls.append((args, kwargs))
        for chunk in self._chunks:
            await self.output_queue.put({"content": chunk})
            await asyncio.sleep(0)
        return type("RunResult", (), {"content": "".join(self._chunks)})()


class _RetryRuntimeAgent:
    """Runtime agent that fails once, then succeeds on retry."""

    def __init__(self, final_content: str, failures: int = 1) -> None:
        self.state = "IDLE"
        self._final_content = final_content
        self._failures_remaining = failures
        self.memory = SimpleNamespace(messages=[])
        self.add_message_calls: list[tuple[str, Any, dict]] = []
        self.run_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def add_message(self, role: str, content: Any, **kwargs) -> None:
        self.add_message_calls.append((role, content, kwargs))
        self.memory.messages.append(
            SimpleNamespace(
                role=role,
                content=content,
                tool_calls=kwargs.get("tool_calls"),
                tool_call_id=kwargs.get("tool_call_id"),
            )
        )

    async def run(self, *args, **kwargs):
        self.run_calls.append((args, kwargs))
        if self._failures_remaining > 0:
            self._failures_remaining -= 1
            raise RuntimeError("transient runtime failure")
        return type("RunResult", (), {"content": self._final_content})()


class _OverflowThenSuccessRuntimeAgent(_RetryRuntimeAgent):
    """Runtime agent that overflows once, then succeeds after compaction."""

    async def run(self, *args, **kwargs):
        self.run_calls.append((args, kwargs))
        if self._failures_remaining > 0:
            self._failures_remaining -= 1
            raise RuntimeError("context length exceeded for this model")
        return type("RunResult", (), {"content": self._final_content})()


class _ThinkingRuntimeAgent:
    """Runtime agent that supports a thinking kwarg."""

    def __init__(self, final_content: str, thinking_content: str) -> None:
        self.state = "IDLE"
        self.memory = SimpleNamespace(messages=[])
        self._final_content = final_content
        self._thinking_content = thinking_content
        self.add_message_calls: list[tuple[str, Any, dict]] = []
        self.run_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def add_message(self, role: str, content: Any, **kwargs) -> None:
        self.add_message_calls.append((role, content, kwargs))
        self.memory.messages.append(
            SimpleNamespace(
                role=role,
                content=content,
                tool_calls=kwargs.get("tool_calls"),
                tool_call_id=kwargs.get("tool_call_id"),
            )
        )

    async def run(self, *args, **kwargs):
        self.run_calls.append((args, kwargs))
        return type(
            "RunResult",
            (),
            {"content": self._final_content, "thinking_content": self._thinking_content},
        )()


class _OpenAITransientAPIError(Exception):
    __module__ = "openai"


class _RetryThinkingRuntimeAgent(_ThinkingRuntimeAgent):
    """Thinking-capable runtime agent that fails once with a retryable API error."""

    def __init__(self, final_content: str, thinking_content: str, failures: int = 1) -> None:
        super().__init__(final_content, thinking_content)
        self._failures_remaining = failures

    async def run(self, *args, **kwargs):
        self.run_calls.append((args, kwargs))
        if self._failures_remaining > 0:
            self._failures_remaining -= 1
            raise _OpenAITransientAPIError(
                "An error occurred while processing your request. "
                "You can retry your request, or contact us through our help center."
            )
        return type(
            "RunResult",
            (),
            {"content": self._final_content, "thinking_content": self._thinking_content},
        )()


class TestAgentLoopCurrentRequestMultimodal:
    @pytest.mark.asyncio
    async def test_process_injects_multimodal_request_before_run(self, tmp_dir: Path):
        from spoon_bot.agent.context import ContextBuilder
        from spoon_bot.agent.loop import AgentLoop

        workspace = tmp_dir / "workspace"
        uploads = workspace / "uploads"
        uploads.mkdir(parents=True)
        image_path = uploads / "current.png"
        image_path.write_bytes(b"png")

        loop = AgentLoop.__new__(AgentLoop)
        loop._stop_requested = False
        loop._initialized = True
        loop.workspace = workspace
        loop.context = ContextBuilder(workspace)
        loop.memory = MagicMock()
        loop.memory.get_memory_context = MagicMock(return_value=None)
        loop._agent = _NoChunkRuntimeAgent("image reply")
        loop._session = Session(session_key="current_multimodal")
        loop.sessions = MagicMock()
        loop.sessions.save = MagicMock()
        loop._auto_commit = False
        loop._git = None
        loop._prepare_request_context = AsyncMock(return_value=None)
        loop._pre_inject_matched_skill = lambda message: message
        loop._build_step_prompt = lambda message: f"prompt::{message}"
        loop._install_anti_loop_tracker = lambda prompt: None
        loop._restore_agent_think = lambda: None
        loop._filter_execution_steps = lambda content: content

        response = await AgentLoop.process(
            loop,
            message="What text appears in the image?",
            media=[str(image_path)],
        )

        assert response == "image reply"
        _assert_multimodal_user_call(
            loop._agent.add_message_calls[0],
            expected_text="What text appears in the image?",
        )
        assert loop._agent.run_calls == [((), {})]
        assert loop._session.messages[0]["media"] == [str(image_path)]

    @pytest.mark.asyncio
    async def test_process_does_not_auto_compress_context_on_runtime_failure(self, tmp_dir: Path):
        from spoon_bot.agent.context import ContextBuilder
        from spoon_bot.agent.loop import AgentLoop

        workspace = tmp_dir / "workspace"
        uploads = workspace / "uploads"
        uploads.mkdir(parents=True)
        image_path = uploads / "retry.png"
        image_path.write_bytes(b"png")

        loop = AgentLoop.__new__(AgentLoop)
        loop._stop_requested = False
        loop._initialized = True
        loop.workspace = workspace
        loop.context = ContextBuilder(workspace)
        loop.memory = MagicMock()
        loop.memory.get_memory_context = MagicMock(return_value=None)
        loop._agent = _RetryRuntimeAgent("recovered reply")
        loop._session = Session(session_key="retry_multimodal")
        loop.sessions = MagicMock()
        loop.sessions.save = MagicMock()
        loop._auto_commit = False
        loop._git = None
        loop._prepare_request_context = AsyncMock(return_value=None)
        loop._pre_inject_matched_skill = lambda message: message
        loop._build_step_prompt = lambda message: f"prompt::{message}"
        loop._install_anti_loop_tracker = lambda prompt: None
        loop._restore_agent_think = lambda: None
        loop._filter_execution_steps = lambda content: content
        def _compress_runtime_context() -> int:
            loop._agent.memory.messages[-1].content = "[compressed away]"
            return 1

        loop._compress_runtime_context = MagicMock(side_effect=_compress_runtime_context)
        loop._force_compress_runtime_context = MagicMock(return_value=0)

        with pytest.raises(RuntimeError, match="transient runtime failure"):
            await AgentLoop.process(
                loop,
                message="What text appears in the image?",
                media=[str(image_path)],
            )

        assert len(loop._agent.add_message_calls) == 1
        _assert_multimodal_user_call(
            loop._agent.add_message_calls[0],
            expected_text="What text appears in the image?",
        )
        assert loop._agent.run_calls == [((), {})]
        loop._compress_runtime_context.assert_not_called()
        loop._force_compress_runtime_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_persists_user_turn_before_runtime_failure(self, tmp_dir: Path):
        from spoon_bot.agent.context import ContextBuilder
        from spoon_bot.agent.loop import AgentLoop

        workspace = tmp_dir / "workspace"
        uploads = workspace / "uploads"
        uploads.mkdir(parents=True)
        image_path = uploads / "retry-noop.png"
        image_path.write_bytes(b"png")

        loop = AgentLoop.__new__(AgentLoop)
        loop._stop_requested = False
        loop._initialized = True
        loop.workspace = workspace
        loop.context = ContextBuilder(workspace)
        loop.memory = MagicMock()
        loop.memory.get_memory_context = MagicMock(return_value=None)
        loop._agent = _RetryRuntimeAgent("recovered reply")
        loop._session = Session(session_key="retry_multimodal_noop")
        loop.sessions = MagicMock()
        loop.sessions.save = MagicMock()
        loop._auto_commit = False
        loop._git = None
        loop._prepare_request_context = AsyncMock(return_value=None)
        loop._pre_inject_matched_skill = lambda message: message
        loop._build_step_prompt = lambda message: f"prompt::{message}"
        loop._install_anti_loop_tracker = lambda prompt: None
        loop._restore_agent_think = lambda: None
        loop._filter_execution_steps = lambda content: content
        loop._compress_runtime_context = MagicMock(return_value=0)
        loop._force_compress_runtime_context = MagicMock(return_value=0)

        with pytest.raises(RuntimeError, match="transient runtime failure"):
            await AgentLoop.process(
                loop,
                message="What text appears in the image?",
                media=[str(image_path)],
            )

        assert len(loop._agent.add_message_calls) == 1
        _assert_multimodal_user_call(
            loop._agent.add_message_calls[0],
            expected_text="What text appears in the image?",
        )
        assert loop._agent.run_calls == [((), {})]
        assert len(loop._session.messages) == 1
        assert loop._session.messages[0]["role"] == "user"
        assert loop._session.messages[0]["content"] == "What text appears in the image?"
        assert loop._session.messages[0]["media"] == [str(image_path)]
        assert loop._session.messages[0]["turn_state"] == "pending"
        loop.sessions.save.assert_called_once_with(loop._session)
        loop._compress_runtime_context.assert_not_called()
        loop._force_compress_runtime_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_retries_once_after_context_overflow_compaction(self, tmp_dir: Path):
        from spoon_bot.agent.context import ContextBuilder
        from spoon_bot.agent.loop import AgentLoop

        workspace = tmp_dir / "workspace"
        uploads = workspace / "uploads"
        uploads.mkdir(parents=True)
        image_path = uploads / "overflow.png"
        image_path.write_bytes(b"png")

        loop = AgentLoop.__new__(AgentLoop)
        loop._stop_requested = False
        loop._initialized = True
        loop.workspace = workspace
        loop.context = ContextBuilder(workspace)
        loop.memory = MagicMock()
        loop.memory.get_memory_context = MagicMock(return_value=None)
        loop._agent = _OverflowThenSuccessRuntimeAgent("recovered after compaction")
        loop.context_window = 400_000
        loop._session = Session(session_key="overflow_retry")
        loop.sessions = MagicMock()
        loop.sessions.save = MagicMock()
        loop._auto_commit = False
        loop._git = None
        loop._prepare_request_context = AsyncMock(return_value=None)
        loop._pre_inject_matched_skill = lambda message: message
        loop._build_step_prompt = lambda message: f"prompt::{message}"
        loop._install_anti_loop_tracker = lambda prompt: None
        loop._restore_agent_think = lambda: None
        loop._filter_execution_steps = lambda content: content
        loop._compress_runtime_context = MagicMock(return_value=2)
        loop._force_compress_runtime_context = MagicMock(return_value=0)

        response = await AgentLoop.process(
            loop,
            message="Keep the image request and continue.",
            media=[str(image_path)],
        )

        assert response == "recovered after compaction"
        assert loop._agent.run_calls == [((), {}), ((), {})]
        loop._compress_runtime_context.assert_called_once_with(
            force=True,
            budget_tokens=380_000,
        )
        loop._force_compress_runtime_context.assert_not_called()


class TestAgentLoopStreamFallback:
    @pytest.mark.asyncio
    async def test_stream_falls_back_to_run_result_when_no_chunks(self, tmp_dir: Path):
        from spoon_bot.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        loop._initialized = True
        loop._agent = _NoChunkRuntimeAgent("fallback from run result")
        loop.workspace = tmp_dir
        loop._session = Session(session_key="stream_fallback")
        loop.sessions = MagicMock()
        loop.sessions.save = MagicMock()
        loop.memory = MagicMock()
        loop.memory.get_memory_context = MagicMock(return_value=None)
        loop.context = MagicMock()
        loop._prepare_request_context = AsyncMock(return_value=None)
        loop._build_step_prompt = lambda message: f"prompt::{message}"
        loop._install_anti_loop_tracker = lambda prompt: None

        chunks = []
        async for chunk in AgentLoop.stream(loop, message="hello"):
            chunks.append(chunk)

        content_chunks = [c for c in chunks if c["type"] == "content"]
        done_chunks = [c for c in chunks if c["type"] == "done"]

        assert len(content_chunks) >= 1
        assert content_chunks[-1]["delta"] == "fallback from run result"
        assert len(done_chunks) == 1
        assert done_chunks[0]["metadata"]["content"] == "fallback from run result"

        assert loop._session.messages[0]["turn_state"] == "completed"
        assert loop._session.messages[-1]["role"] == "assistant"
        assert loop._session.messages[-1]["content"] == "fallback from run result"
        assert loop.sessions.save.call_count == 3
        loop.sessions.save.assert_called_with(loop._session)

    @pytest.mark.asyncio
    async def test_stream_preserves_incremental_queue_chunks(self, tmp_dir: Path):
        from spoon_bot.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        loop._initialized = True
        loop._agent = _ChunkedRuntimeAgent(["Hel", "lo"])
        loop.workspace = tmp_dir
        loop._session = Session(session_key="stream_incremental")
        loop.sessions = MagicMock()
        loop.sessions.save = MagicMock()
        loop.memory = MagicMock()
        loop.memory.get_memory_context = MagicMock(return_value=None)
        loop.context = MagicMock()
        loop._prepare_request_context = AsyncMock(return_value=None)
        loop._build_step_prompt = lambda message: f"prompt::{message}"
        loop._install_anti_loop_tracker = lambda prompt: None

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

    @pytest.mark.asyncio
    async def test_stream_persists_original_user_text_instead_of_attachment_prose(self, tmp_dir: Path):
        from spoon_bot.agent.loop import AgentLoop, _ensure_attachment_context

        loop = AgentLoop.__new__(AgentLoop)
        loop._initialized = True
        loop._agent = _NoChunkRuntimeAgent("attachment reply")
        loop.workspace = tmp_dir
        loop._session = Session(session_key="stream_attachment_text")
        loop.sessions = MagicMock()
        loop.sessions.save = MagicMock()
        loop.memory = MagicMock()
        loop.memory.get_memory_context = MagicMock(return_value=None)
        loop.context = MagicMock()
        loop._prepare_request_context = AsyncMock(return_value=None)
        loop._build_step_prompt = lambda message: f"prompt::{message}"
        loop._install_anti_loop_tracker = lambda prompt: None

        attachments = [{"uri": "/workspace/uploads/demo.pdf", "name": "demo.pdf"}]
        injected_message = _ensure_attachment_context("Please summarize the attachment.", attachments)

        chunks = []
        async for chunk in AgentLoop.stream(loop, message=injected_message, attachments=attachments):
            chunks.append(chunk)

        assert any(chunk["type"] == "done" for chunk in chunks)
        assert loop._session.messages[0]["role"] == "user"
        assert loop._session.messages[0]["content"] == "Please summarize the attachment."
        assert loop._session.messages[0]["attachments"] == attachments

    @pytest.mark.asyncio
    async def test_stream_persists_user_turn_when_upstream_fails(self, tmp_dir: Path):
        from spoon_bot.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        loop._initialized = True
        loop._agent = _FailingStreamRuntimeAgent("unused")
        loop.workspace = tmp_dir
        loop._session = Session(session_key="stream_error_context")
        loop.sessions = MagicMock()
        loop.sessions.save = MagicMock()
        loop.memory = MagicMock()
        loop.memory.get_memory_context = MagicMock(return_value=None)
        loop.context = MagicMock()
        loop._prepare_request_context = AsyncMock(return_value=None)
        loop._build_step_prompt = lambda message: f"prompt::{message}"
        loop._install_anti_loop_tracker = lambda prompt: None

        chunks = []
        async for chunk in AgentLoop.stream(loop, message="keep this request in history"):
            chunks.append(chunk)

        assert any(chunk["type"] == "error" for chunk in chunks)
        assert len(loop._session.messages) == 1
        assert loop._session.messages[0]["role"] == "user"
        assert loop._session.messages[0]["content"] == "keep this request in history"
        assert loop._session.messages[0]["turn_state"] == "pending"
        loop.sessions.save.assert_called_once_with(loop._session)


class TestContextBuilderMediaPaths:
    def test_build_user_content_resolves_workspace_alias_and_relative_media(self, tmp_dir: Path):
        from spoon_bot.agent.context import ContextBuilder

        workspace = tmp_dir / "workspace"
        uploads = workspace / "uploads"
        uploads.mkdir(parents=True)
        image_path = uploads / "demo.png"
        image_path.write_bytes(b"png")

        builder = ContextBuilder(workspace)

        alias_content = builder._build_user_content("look", ["/workspace/uploads/demo.png"])
        relative_content = builder._build_user_content("look", ["uploads/demo.png"])

        assert isinstance(alias_content, list)
        assert isinstance(relative_content, list)
        assert alias_content[0]["type"] == "image_url"
        assert relative_content[0]["type"] == "image_url"
        assert alias_content[-1] == {"type": "text", "text": "look"}
        assert relative_content[-1] == {"type": "text", "text": "look"}


class TestAgentLoopRuntimeCompression:
    def test_compress_runtime_context_counts_and_summarizes_inline_image_payloads(self):
        from spoon_bot.agent.loop import AgentLoop

        data_url = "data:image/png;base64," + ("A" * 800)
        multimodal_message = [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": "Please inspect this image carefully."},
        ]
        messages = [
            SimpleNamespace(role="system", content="system prompt", tool_calls=None, tool_call_id=None),
            SimpleNamespace(role="user", content=multimodal_message, tool_calls=None, tool_call_id=None),
            SimpleNamespace(role="assistant", content="assistant 1", tool_calls=None, tool_call_id=None),
            SimpleNamespace(role="user", content="user 2", tool_calls=None, tool_call_id=None),
            SimpleNamespace(role="assistant", content="assistant 2", tool_calls=None, tool_call_id=None),
            SimpleNamespace(role="user", content="user 3", tool_calls=None, tool_call_id=None),
            SimpleNamespace(role="assistant", content="assistant 3", tool_calls=None, tool_call_id=None),
            SimpleNamespace(role="user", content="latest request", tool_calls=None, tool_call_id=None),
        ]

        loop = AgentLoop.__new__(AgentLoop)
        loop.context_window = 200
        loop._agent = SimpleNamespace(memory=SimpleNamespace(messages=messages))

        compressed = AgentLoop._compress_runtime_context(loop, force=True, budget_tokens=100)

        assert compressed >= 1
        assert isinstance(messages[1].content, str)
        assert "image attachment(s) omitted during context compression" in messages[1].content
        assert "Please inspect this image carefully." in messages[1].content
        assert "data:image/png;base64," not in messages[1].content

    def test_force_compaction_preserves_latest_real_multimodal_user_request(self):
        from spoon_bot.agent.loop import AgentLoop

        data_url = "data:image/png;base64," + ("B" * 800)
        older_multimodal_message = [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": "Older screenshot."},
        ]
        latest_multimodal_message = [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": "Current request with image must stay intact."},
        ]
        messages = [
            SimpleNamespace(role="system", content="system prompt", tool_calls=None, tool_call_id=None),
            SimpleNamespace(role="user", content=older_multimodal_message, tool_calls=None, tool_call_id=None),
            SimpleNamespace(role="assistant", content="assistant 1", tool_calls=None, tool_call_id=None),
            SimpleNamespace(role="user", content="another request", tool_calls=None, tool_call_id=None),
            SimpleNamespace(role="assistant", content="assistant 2", tool_calls=None, tool_call_id=None),
            SimpleNamespace(role="user", content=latest_multimodal_message, tool_calls=None, tool_call_id=None),
            SimpleNamespace(role="assistant", content="tool summary " * 40, tool_calls=None, tool_call_id=None),
            SimpleNamespace(role="assistant", content="another summary " * 40, tool_calls=None, tool_call_id=None),
            SimpleNamespace(role="assistant", content="tail summary " * 40, tool_calls=None, tool_call_id=None),
        ]

        loop = AgentLoop.__new__(AgentLoop)
        loop.context_window = 200
        loop._agent = SimpleNamespace(memory=SimpleNamespace(messages=messages))

        compressed = AgentLoop._force_compress_runtime_context(loop)

        assert compressed >= 1
        assert any(message.content == latest_multimodal_message for message in messages)
        assert any(message.content != older_multimodal_message for message in messages if getattr(message, "role", None) == "user")


class TestAgentLoopThinkingMode:
    @pytest.mark.asyncio
    async def test_process_with_thinking_passes_flag_when_runtime_accepts_it(self, tmp_dir: Path):
        from spoon_bot.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        loop._initialized = True
        loop.workspace = tmp_dir
        loop.memory = MagicMock()
        loop.memory.get_memory_context = MagicMock(return_value=None)
        loop.context = MagicMock()
        loop._agent = _ThinkingRuntimeAgent("answer", "reasoning trace")
        loop._session = Session(session_key="thinking_mode")
        loop.sessions = MagicMock()
        loop.sessions.save = MagicMock()
        loop._prepare_request_context = AsyncMock(return_value=None)
        loop._build_step_prompt = lambda message: f"prompt::{message}"
        loop._install_anti_loop_tracker = lambda prompt: None
        loop._restore_agent_think = lambda: None
        loop._auto_commit = False
        loop._git = None

        response, thinking = await AgentLoop.process_with_thinking(loop, message="Explain the answer.")

        assert response == "answer"
        assert thinking == "reasoning trace"
        assert loop._agent.run_calls == [((), {"thinking": True})]

    @pytest.mark.asyncio
    async def test_process_with_thinking_passes_reasoning_effort_when_runtime_accepts_it(self, tmp_dir: Path):
        from spoon_bot.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        loop._initialized = True
        loop.workspace = tmp_dir
        loop.memory = MagicMock()
        loop.memory.get_memory_context = MagicMock(return_value=None)
        loop.context = MagicMock()
        loop._agent = _ThinkingRuntimeAgent("answer", "reasoning trace")
        loop._session = Session(session_key="thinking_mode")
        loop.sessions = MagicMock()
        loop.sessions.save = MagicMock()
        loop._prepare_request_context = AsyncMock(return_value=None)
        loop._build_step_prompt = lambda message: f"prompt::{message}"
        loop._install_anti_loop_tracker = lambda prompt: None
        loop._restore_agent_think = lambda: None
        loop._auto_commit = False
        loop._git = None

        response, thinking = await AgentLoop.process_with_thinking(
            loop,
            message="Explain the answer.",
            reasoning_effort="high",
        )

        assert response == "answer"
        assert thinking == "reasoning trace"

    @pytest.mark.asyncio
    async def test_process_with_thinking_marks_current_user_turn_interrupted_on_cancel(self, tmp_dir: Path):
        from spoon_bot.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)
        loop._initialized = True
        loop.workspace = tmp_dir
        loop.memory = MagicMock()
        loop.memory.get_memory_context = MagicMock(return_value=None)
        loop.context = MagicMock()
        loop._agent = MagicMock()
        loop._agent.run = AsyncMock(side_effect=asyncio.CancelledError())
        loop._agent.add_message = AsyncMock()
        loop._session = Session(session_key="thinking_cancelled")
        loop.sessions = MagicMock()
        loop.sessions.save = MagicMock()
        loop._prepare_request_context = AsyncMock(return_value=None)
        loop._install_anti_loop_tracker = lambda prompt: None
        loop._restore_agent_think = lambda: None
        loop._auto_commit = False
        loop._git = None

        with pytest.raises(asyncio.CancelledError):
            await AgentLoop.process_with_thinking(loop, message="Choose C")

        assert len(loop._session.messages) == 1
        assert loop._session.messages[0]["role"] == "user"
        assert loop._session.messages[0]["content"] == "Choose C"
        assert loop._session.messages[0]["turn_state"] == "interrupted"
        assert loop._session.messages[0]["turn_state_reason"] == "task_cancelled"
        assert loop.sessions.save.call_count >= 2

    @pytest.mark.asyncio
    async def test_process_with_thinking_retries_openai_apierror(self, tmp_dir: Path):
        from spoon_bot.agent.loop import AgentLoop
        from spoon_bot.utils.retry import RetryConfig

        loop = AgentLoop.__new__(AgentLoop)
        loop._initialized = True
        loop.workspace = tmp_dir
        loop.memory = MagicMock()
        loop.memory.get_memory_context = MagicMock(return_value=None)
        loop.context = MagicMock()
        loop._agent = _RetryThinkingRuntimeAgent("answer", "reasoning trace")
        loop._session = Session(session_key="thinking_retry")
        loop.sessions = MagicMock()
        loop.sessions.save = MagicMock()
        loop._prepare_request_context = AsyncMock(return_value=None)
        loop._build_step_prompt = lambda message: f"prompt::{message}"
        loop._install_anti_loop_tracker = lambda prompt: None
        loop._restore_agent_think = lambda: None
        loop._auto_commit = False
        loop._git = None
        loop._retry_config = RetryConfig(max_retries=1, base_delay=0.01, max_delay=0.01, jitter=0.0)

        response, thinking = await AgentLoop.process_with_thinking(loop, message="Explain the answer.")

        assert response == "answer"
        assert thinking == "reasoning trace"
        assert loop._agent.run_calls == [
            ((), {"thinking": True}),
            ((), {"thinking": True}),
        ]


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
