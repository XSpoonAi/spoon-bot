"""
Pluggable session storage backends.

Backends:
  - FileSessionStore   — JSONL + .meta.json files (default, original behaviour)
  - SQLiteSessionStore — SQLite3 database (zero extra dependencies)
  - PostgresSessionStore — PostgreSQL via psycopg2 / asyncpg

All backends implement the ``SessionStore`` protocol.
``SessionManager`` accepts *any* ``SessionStore`` and uses it transparently.
"""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

# ---------------------------------------------------------------------------
# Session data-class is imported from the sibling module so that both
# store.py and manager.py share the same type.
# ---------------------------------------------------------------------------
from spoon_bot.session.manager import Session


# ============================================================================
# Abstract base
# ============================================================================


class SessionStore(ABC):
    """Abstract session storage backend."""

    # -- CRUD ---------------------------------------------------------------

    @abstractmethod
    def save_session(self, session: Session) -> None:
        """Persist a session (upsert)."""

    @abstractmethod
    def load_session(self, session_key: str) -> Optional[Session]:
        """Load a session by key.  Returns ``None`` if not found."""

    @abstractmethod
    def delete_session(self, session_key: str) -> bool:
        """Delete a session.  Returns ``True`` if it existed."""

    @abstractmethod
    def list_session_keys(self) -> List[str]:
        """Return all known session keys (sorted)."""

    # -- optional convenience (default impl) --------------------------------

    def close(self) -> None:  # noqa: B027  (intentionally empty)
        """Release resources held by the store (connections, file handles)."""


# ============================================================================
# FileSessionStore — JSONL + .meta.json  (original behaviour)
# ============================================================================


class FileSessionStore(SessionStore):
    """JSONL-file based session store (the original implementation)."""

    def __init__(self, sessions_dir: Path) -> None:
        self._dir = Path(sessions_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    # -- helpers ------------------------------------------------------------

    def _path(self, key: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
        return self._dir / f"{safe}.jsonl"

    # -- CRUD ---------------------------------------------------------------

    def save_session(self, session: Session) -> None:
        path = self._path(session.session_key)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                for msg in session.messages:
                    fh.write(json.dumps(msg, ensure_ascii=False) + "\n")
            meta = path.with_suffix(".meta.json")
            with open(meta, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "session_key": session.session_key,
                        "created_at": session.created_at.isoformat(),
                        "updated_at": session.updated_at.isoformat(),
                        "metadata": session.metadata,
                        "message_count": len(session.messages),
                    },
                    fh,
                    indent=2,
                )
            logger.debug(f"FileStore: saved session {session.session_key}")
        except Exception as exc:
            logger.error(f"FileStore: save failed for {session.session_key}: {exc}")

    def load_session(self, session_key: str) -> Optional[Session]:
        path = self._path(session_key)
        if not path.exists():
            return None
        try:
            messages: list[dict] = []
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        messages.append(json.loads(line))
            metadata: dict = {}
            created_at = datetime.now()
            updated_at = datetime.now()
            meta_path = path.with_suffix(".meta.json")
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                    metadata = meta.get("metadata", {})
                    created_at = datetime.fromisoformat(meta.get("created_at", created_at.isoformat()))
                    updated_at = datetime.fromisoformat(meta.get("updated_at", updated_at.isoformat()))
            return Session(
                session_key=session_key,
                created_at=created_at,
                updated_at=updated_at,
                messages=messages,
                metadata=metadata,
            )
        except Exception as exc:
            logger.error(f"FileStore: load failed for {session_key}: {exc}")
            return None

    def delete_session(self, session_key: str) -> bool:
        path = self._path(session_key)
        meta = path.with_suffix(".meta.json")
        deleted = False
        if path.exists():
            path.unlink()
            deleted = True
        if meta.exists():
            meta.unlink()
            deleted = True
        return deleted

    def archive_session(self, session_key: str) -> bool:
        """Archive a session by renaming its files with a deleted marker."""
        path = self._path(session_key)
        meta = path.with_suffix(".meta.json")
        marker = f".deleted.{int(datetime.now().timestamp())}"
        archived = False

        if path.exists():
            archived_path = path.with_name(f"{path.stem}{marker}{path.suffix}")
            path.rename(archived_path)
            archived = True
        if meta.exists():
            archived_meta = meta.with_name(f"{meta.stem}{marker}{meta.suffix}")
            meta.rename(archived_meta)
            archived = True
        return archived

    def list_session_keys(self) -> List[str]:
        keys = {f.stem for f in self._dir.glob("*.jsonl")}
        return sorted(keys)


# ============================================================================
# SQLiteSessionStore
# ============================================================================


class SQLiteSessionStore(SessionStore):
    """SQLite-based session store.

    Schema:
        sessions — one row per session
        messages — one row per message, ordered by ``seq``
    """

    def __init__(self, db_path: str = "sessions.db") -> None:
        self._db_path = str(Path(db_path).expanduser().resolve())
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        logger.info(f"SQLiteSessionStore: opened {self._db_path}")

    # -- schema -------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=DELETE")  # avoid WAL lock files on Windows
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _exec(self, fn):
        """Execute *fn(conn)* then close the connection (Windows-safe)."""
        conn = self._conn()
        try:
            result = fn(conn)
            conn.commit()
            return result
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        conn = self._conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_key  TEXT PRIMARY KEY,
                    created_at   TEXT NOT NULL,
                    updated_at   TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_key  TEXT NOT NULL REFERENCES sessions(session_key) ON DELETE CASCADE,
                    seq          INTEGER NOT NULL,
                    role         TEXT NOT NULL,
                    content      TEXT NOT NULL DEFAULT '',
                    timestamp    TEXT,
                    extra_json   TEXT NOT NULL DEFAULT '{}',
                    UNIQUE(session_key, seq)
                );
                CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_key, seq);
            """)
        finally:
            conn.close()

    # -- CRUD ---------------------------------------------------------------

    def save_session(self, session: Session) -> None:
        def _do(conn):
            conn.execute(
                """INSERT INTO sessions (session_key, created_at, updated_at, metadata_json)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(session_key)
                   DO UPDATE SET updated_at=excluded.updated_at,
                                 metadata_json=excluded.metadata_json""",
                (
                    session.session_key,
                    session.created_at.isoformat(),
                    session.updated_at.isoformat(),
                    json.dumps(session.metadata, ensure_ascii=False),
                ),
            )
            conn.execute("DELETE FROM messages WHERE session_key=?", (session.session_key,))
            for seq, msg in enumerate(session.messages):
                role = msg.get("role", "")
                content = msg.get("content", "")
                ts = msg.get("timestamp")
                extra = {k: v for k, v in msg.items() if k not in ("role", "content", "timestamp")}
                conn.execute(
                    """INSERT INTO messages (session_key, seq, role, content, timestamp, extra_json)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (session.session_key, seq, role, content, ts, json.dumps(extra, ensure_ascii=False)),
                )
        self._exec(_do)
        logger.debug(f"SQLiteStore: saved session {session.session_key} ({len(session.messages)} msgs)")

    def load_session(self, session_key: str) -> Optional[Session]:
        def _do(conn):
            row = conn.execute("SELECT * FROM sessions WHERE session_key=?", (session_key,)).fetchone()
            if not row:
                return None
            msg_rows = conn.execute(
                "SELECT * FROM messages WHERE session_key=? ORDER BY seq", (session_key,)
            ).fetchall()
            messages: list[dict] = []
            for mr in msg_rows:
                msg: Dict[str, Any] = {"role": mr["role"], "content": mr["content"]}
                if mr["timestamp"]:
                    msg["timestamp"] = mr["timestamp"]
                extra = json.loads(mr["extra_json"]) if mr["extra_json"] else {}
                msg.update(extra)
                messages.append(msg)
            return Session(
                session_key=session_key,
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                messages=messages,
                metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
            )
        return self._exec(_do)

    def delete_session(self, session_key: str) -> bool:
        def _do(conn):
            cur = conn.execute("DELETE FROM sessions WHERE session_key=?", (session_key,))
            return cur.rowcount > 0
        return self._exec(_do)

    def list_session_keys(self) -> List[str]:
        def _do(conn):
            rows = conn.execute("SELECT session_key FROM sessions ORDER BY session_key").fetchall()
            return [r["session_key"] for r in rows]
        return self._exec(_do)

    def close(self) -> None:
        """No-op — connections are opened/closed per call."""
        pass


# ============================================================================
# PostgresSessionStore
# ============================================================================


class PostgresSessionStore(SessionStore):
    """PostgreSQL-based session store using ``psycopg2``.

    Requires:
        pip install psycopg2-binary   (or psycopg2)

    Connection string example:
        postgresql://user:password@host:5432/dbname
    """

    def __init__(self, dsn: str) -> None:
        try:
            import psycopg2  # noqa: F401
        except ImportError:
            raise ImportError(
                "psycopg2 is required for PostgresSessionStore. "
                "Install it: pip install psycopg2-binary"
            )
        self._dsn = dsn
        self._init_schema()
        logger.info("PostgresSessionStore: connected")

    def _conn(self):
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(self._dsn)
        conn.autocommit = False
        return conn

    def _init_schema(self) -> None:
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_key  TEXT PRIMARY KEY,
                        created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        metadata_json JSONB NOT NULL DEFAULT '{}'
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id           SERIAL PRIMARY KEY,
                        session_key  TEXT NOT NULL REFERENCES sessions(session_key) ON DELETE CASCADE,
                        seq          INTEGER NOT NULL,
                        role         TEXT NOT NULL,
                        content      TEXT NOT NULL DEFAULT '',
                        timestamp    TIMESTAMPTZ,
                        extra_json   JSONB NOT NULL DEFAULT '{}',
                        UNIQUE(session_key, seq)
                    )
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_pg_msg_session
                    ON messages(session_key, seq)
                """)
            conn.commit()
        finally:
            conn.close()

    # -- CRUD ---------------------------------------------------------------

    def save_session(self, session: Session) -> None:
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO sessions (session_key, created_at, updated_at, metadata_json)
                       VALUES (%s, %s, %s, %s)
                       ON CONFLICT (session_key)
                       DO UPDATE SET updated_at=EXCLUDED.updated_at,
                                     metadata_json=EXCLUDED.metadata_json""",
                    (
                        session.session_key,
                        session.created_at.isoformat(),
                        session.updated_at.isoformat(),
                        json.dumps(session.metadata, ensure_ascii=False),
                    ),
                )
                cur.execute("DELETE FROM messages WHERE session_key=%s", (session.session_key,))
                for seq, msg in enumerate(session.messages):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    ts = msg.get("timestamp")
                    extra = {k: v for k, v in msg.items() if k not in ("role", "content", "timestamp")}
                    cur.execute(
                        """INSERT INTO messages (session_key, seq, role, content, timestamp, extra_json)
                           VALUES (%s, %s, %s, %s, %s, %s)""",
                        (session.session_key, seq, role, content, ts, json.dumps(extra, ensure_ascii=False)),
                    )
            conn.commit()
            logger.debug(f"PgStore: saved session {session.session_key} ({len(session.messages)} msgs)")
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def load_session(self, session_key: str) -> Optional[Session]:
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM sessions WHERE session_key=%s", (session_key,))
                row = cur.fetchone()
                if not row:
                    return None
                # row: (session_key, created_at, updated_at, metadata_json)
                s_key, created_at, updated_at, metadata_json = row[0], row[1], row[2], row[3]

                cur.execute(
                    "SELECT role, content, timestamp, extra_json FROM messages WHERE session_key=%s ORDER BY seq",
                    (session_key,),
                )
                msg_rows = cur.fetchall()
            messages: list[dict] = []
            for mr in msg_rows:
                msg: Dict[str, Any] = {"role": mr[0], "content": mr[1]}
                if mr[2]:
                    msg["timestamp"] = mr[2].isoformat() if hasattr(mr[2], "isoformat") else str(mr[2])
                extra = json.loads(mr[3]) if isinstance(mr[3], str) else (mr[3] or {})
                msg.update(extra)
                messages.append(msg)

            ca = created_at if isinstance(created_at, datetime) else datetime.fromisoformat(str(created_at))
            ua = updated_at if isinstance(updated_at, datetime) else datetime.fromisoformat(str(updated_at))
            md = json.loads(metadata_json) if isinstance(metadata_json, str) else (metadata_json or {})

            return Session(
                session_key=s_key,
                created_at=ca,
                updated_at=ua,
                messages=messages,
                metadata=md,
            )
        finally:
            conn.close()

    def delete_session(self, session_key: str) -> bool:
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM sessions WHERE session_key=%s", (session_key,))
                deleted = cur.rowcount > 0
            conn.commit()
            return deleted
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def list_session_keys(self) -> List[str]:
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT session_key FROM sessions ORDER BY session_key")
                return [r[0] for r in cur.fetchall()]
        finally:
            conn.close()

    def close(self) -> None:
        pass  # connections are per-call


# ============================================================================
# Factory
# ============================================================================


def create_session_store(
    backend: str = "file",
    *,
    workspace: Path | str | None = None,
    db_path: str | None = None,
    dsn: str | None = None,
) -> SessionStore:
    """Create a session store from configuration.

    Args:
        backend: One of ``"file"``, ``"sqlite"``, ``"postgres"``.
        workspace: Required for ``"file"`` backend (sessions dir parent).
        db_path: Path for ``"sqlite"`` backend (default ``"sessions.db"``).
        dsn: PostgreSQL connection string for ``"postgres"`` backend.

    Returns:
        A ``SessionStore`` instance.
    """
    backend = backend.lower().strip()
    if backend == "file":
        if workspace is None:
            raise ValueError("workspace is required for file session store")
        return FileSessionStore(Path(workspace) / "sessions")
    elif backend in ("sqlite", "sqlite3"):
        return SQLiteSessionStore(db_path or "sessions.db")
    elif backend in ("postgres", "postgresql", "pg"):
        if dsn is None:
            raise ValueError("dsn is required for postgres session store")
        return PostgresSessionStore(dsn)
    else:
        raise ValueError(f"Unknown session store backend: {backend!r}")
