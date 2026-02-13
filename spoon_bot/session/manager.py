"""Session manager for persistent conversation history."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


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


class SessionManager:
    """
    Manages conversation sessions with JSONL persistence.

    Sessions are stored as JSONL files (one JSON object per line per message).
    """

    def __init__(self, workspace: Path):
        """
        Initialize session manager.

        Args:
            workspace: Path to workspace directory.
        """
        self.workspace = Path(workspace).expanduser().resolve()
        self.sessions_dir = self.workspace / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, Session] = {}

    def get_or_create(self, session_key: str) -> Session:
        """
        Get an existing session or create a new one.

        Args:
            session_key: Unique session identifier.

        Returns:
            Session instance.
        """
        if session_key in self._sessions:
            return self._sessions[session_key]

        # Try loading from disk
        session = self._load_session(session_key)
        if session is None:
            session = Session(session_key=session_key)
            logger.debug(f"Created new session: {session_key}")
        else:
            logger.debug(f"Loaded existing session: {session_key}")

        self._sessions[session_key] = session
        return session

    def save(self, session: Session) -> None:
        """
        Save a session to disk.

        Args:
            session: Session to save.
        """
        session_file = self._get_session_path(session.session_key)

        try:
            with open(session_file, "w", encoding="utf-8") as f:
                # Write each message as a separate line
                for message in session.messages:
                    f.write(json.dumps(message, ensure_ascii=False) + "\n")

            # Also save metadata
            meta_file = session_file.with_suffix(".meta.json")
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump({
                    "session_key": session.session_key,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                    "metadata": session.metadata,
                    "message_count": len(session.messages),
                }, f, indent=2)

            logger.debug(f"Saved session: {session.session_key}")

        except Exception as e:
            logger.error(f"Error saving session {session.session_key}: {e}")

    def delete(self, session_key: str) -> bool:
        """
        Delete a session.

        Args:
            session_key: Session to delete.

        Returns:
            True if deleted, False if not found.
        """
        if session_key in self._sessions:
            del self._sessions[session_key]

        session_file = self._get_session_path(session_key)
        meta_file = session_file.with_suffix(".meta.json")

        deleted = False
        if session_file.exists():
            session_file.unlink()
            deleted = True
        if meta_file.exists():
            meta_file.unlink()
            deleted = True

        return deleted

    def list_sessions(self) -> list[str]:
        """List all session keys."""
        sessions = set(self._sessions.keys())

        # Also check disk
        for file in self.sessions_dir.glob("*.jsonl"):
            sessions.add(file.stem)

        return sorted(sessions)

    def _get_session_path(self, session_key: str) -> Path:
        """Get the file path for a session."""
        # Sanitize session key for filename
        safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_key)
        return self.sessions_dir / f"{safe_key}.jsonl"

    def _load_session(self, session_key: str) -> Session | None:
        """Load a session from disk."""
        session_file = self._get_session_path(session_key)
        meta_file = session_file.with_suffix(".meta.json")

        if not session_file.exists():
            return None

        try:
            # Load messages from JSONL
            messages = []
            with open(session_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        messages.append(json.loads(line))

            # Load metadata
            metadata = {}
            created_at = datetime.now()
            updated_at = datetime.now()

            if meta_file.exists():
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    metadata = meta.get("metadata", {})
                    created_at = datetime.fromisoformat(meta.get("created_at", datetime.now().isoformat()))
                    updated_at = datetime.fromisoformat(meta.get("updated_at", datetime.now().isoformat()))

            return Session(
                session_key=session_key,
                created_at=created_at,
                updated_at=updated_at,
                messages=messages,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Error loading session {session_key}: {e}")
            return None
