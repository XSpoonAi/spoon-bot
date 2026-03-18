"""Session manager for persistent conversation history.

The ``SessionManager`` is the public API consumed by ``AgentLoop``.
It delegates all I/O to a pluggable ``SessionStore`` backend (file, sqlite,
postgres) while keeping an in-memory LRU cache for hot sessions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from threading import RLock
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from spoon_bot.session.store import SessionStore


# ============================================================================
# Session data-class  (unchanged public API)
# ============================================================================


@dataclass
class Session:
    """Represents a conversation session."""

    session_key: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    messages: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the session."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }
        self.messages.append(message)
        self.updated_at = datetime.now()

    def get_history(self) -> list[dict[str, Any]]:
        """Get message history in LLM format (role + content only)."""
        return [
            {"role": m["role"], "content": m["content"]}
            for m in self.messages
        ]

    def get_messages(self) -> list[dict[str, Any]]:
        """Get full persisted messages including optional metadata."""
        return [dict(message) for message in self.messages if isinstance(message, dict)]

    def clear(self) -> None:
        """Clear all messages."""
        self.messages.clear()
        self.updated_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "session_key": self.session_key,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "messages": self.messages,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        """Create session from dictionary."""
        return cls(
            session_key=data["session_key"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            messages=data.get("messages", []),
            metadata=data.get("metadata", {}),
        )


# ============================================================================
# SessionManager  (public API — unchanged signatures)
# ============================================================================


class SessionManager:
    """
    Manages conversation sessions with pluggable persistence.

    By default uses the original JSONL-file store. Pass a ``SessionStore``
    to switch to SQLite, PostgreSQL, or any custom backend.

    Notes:
        - Keeps an in-memory cache for hot sessions.
        - Cache size is bounded by ``max_cached_sessions`` to avoid unbounded growth.
        - Access is guarded with a reentrant lock for thread-safe callers.

    Usage (file — default, backward-compatible)::

        sm = SessionManager(workspace=Path("./workspace"))

    Usage (sqlite)::

        from spoon_bot.session.store import SQLiteSessionStore
        sm = SessionManager(store=SQLiteSessionStore("sessions.db"))

    Usage (factory helper)::

        from spoon_bot.session.store import create_session_store
        store = create_session_store("sqlite", db_path="sessions.db")
        sm = SessionManager(store=store)
    """

    DEFAULT_MAX_CACHED_SESSIONS = 128

    def __init__(
        self,
        workspace: Path | str | None = None,
        *,
        store: Optional["SessionStore"] = None,
        max_cached_sessions: int = DEFAULT_MAX_CACHED_SESSIONS,
    ) -> None:
        """
        Initialize session manager.

        Args:
            workspace: Workspace directory (creates FileSessionStore if *store* is None).
            store: Explicit session-store backend.  Takes precedence over *workspace*.
        """
        if store is not None:
            self._store = store
        else:
            if workspace is None:
                raise ValueError("Either 'workspace' or 'store' must be provided")
            from spoon_bot.session.store import FileSessionStore
            ws = Path(workspace).expanduser().resolve()
            sessions_dir = ws / "sessions"
            sessions_dir.mkdir(parents=True, exist_ok=True)
            self._store = FileSessionStore(sessions_dir)

        # In-memory cache for fast repeated access
        self._sessions: dict[str, Session] = {}
        self._lock = RLock()
        self._max_cached_sessions = max(1, int(max_cached_sessions))

        # Keep workspace for backward-compat (used by some callers)
        self.workspace = Path(workspace).expanduser().resolve() if workspace else None
        self.sessions_dir = (self.workspace / "sessions") if self.workspace else None

    # ------------------------------------------------------------------
    # Public API  (signatures unchanged)
    # ------------------------------------------------------------------

    def get_or_create(self, session_key: str) -> Session:
        """Get an existing session or create a new one."""
        with self._lock:
            if session_key in self._sessions:
                return self._sessions[session_key]

            session = self._store.load_session(session_key)
            if session is None:
                session = Session(session_key=session_key)
                logger.debug(f"Created new session: {session_key}")
            else:
                logger.debug(f"Loaded existing session: {session_key}")

            self._sessions[session_key] = session
            self._evict_if_needed()
            return session

    def get(self, session_key: str) -> Session | None:
        """Get an existing session without creating a new one."""
        with self._lock:
            if session_key in self._sessions:
                return self._sessions[session_key]

            session = self._store.load_session(session_key)
            if session is not None:
                self._sessions[session_key] = session
                self._evict_if_needed()
            return session

    def save(self, session: Session) -> None:
        """Persist a session to the configured backend."""
        with self._lock:
            self._sessions[session.session_key] = session
            self._evict_if_needed()
            self._store.save_session(session)

    def delete(self, session_key: str) -> bool:
        """Delete a session from cache and backend."""
        with self._lock:
            self._sessions.pop(session_key, None)
            return self._store.delete_session(session_key)

    def list_sessions(self) -> list[str]:
        """List all session keys (from cache + backend)."""
        with self._lock:
            keys = set(self._sessions.keys())
            keys.update(self._store.list_session_keys())
            return sorted(keys)

    def close(self) -> None:
        """Release backend resources."""
        with self._lock:
            self._store.close()

    def _evict_if_needed(self) -> None:
        """Bound in-memory cache size by evicting oldest inserted sessions."""
        while len(self._sessions) > self._max_cached_sessions:
            oldest_key = next(iter(self._sessions))
            self._sessions.pop(oldest_key, None)
