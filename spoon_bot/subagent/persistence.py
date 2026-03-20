"""File-based JSON persistence for sub-agent records.

Single-file approach following the OpenClaw pattern:
  {workspace}/subagents/runs.json

On-disk format:
  {"version": 1, "runs": {"<agent_id>": {...SubagentRecord dump...}, ...}}

Three components:
  SubagentRunsFile  — Synchronous JSON file I/O (atomic writes, corrupt-file recovery)
  restore_subagent_runs — Orphan reconciliation on startup
  SubagentSweeper   — Async background task that archives old terminal records
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from loguru import logger

from spoon_bot.subagent.models import (
    PersistentSubagentProfile,
    SubagentRecord,
    SubagentState,
    SpawnMode,
)

if TYPE_CHECKING:
    from spoon_bot.subagent.registry import SubagentRegistry
    from spoon_bot.session.manager import SessionManager

# Increment this when the on-disk schema changes in a backwards-incompatible way.
_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# SubagentRunsFile
# ---------------------------------------------------------------------------


class SubagentRunsFile:
    """Read / write the runs.json persistence file.

    All I/O is synchronous — callers (SubagentRegistry) already hold the
    threading.RLock when calling these methods, so no additional locking
    is required here.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> dict[str, SubagentRecord]:
        """Load sub-agent records from disk.

        Returns an empty dict if the file does not exist or is corrupt.
        Silently handles forward-compatible schema versions (with a warning).
        """
        if not self._path.exists():
            return {}

        try:
            raw = self._path.read_text(encoding="utf-8")
            data: dict[str, Any] = json.loads(raw)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            logger.error(f"SubagentRunsFile: failed to read {self._path}: {exc}")
            self._quarantine_corrupt_file()
            return {}

        version = data.get("version", 0)
        if not isinstance(version, int) or version < 1:
            logger.warning(
                f"SubagentRunsFile: unknown version {version!r}; "
                "treating file as corrupt"
            )
            self._quarantine_corrupt_file()
            return {}

        if version > _SCHEMA_VERSION:
            logger.warning(
                f"SubagentRunsFile: file version {version} is newer than "
                f"code version {_SCHEMA_VERSION}; will attempt to load anyway"
            )

        # Future migration hook:
        # if version < _SCHEMA_VERSION:
        #     data = self._migrate(data, from_version=version)

        runs_raw: dict[str, Any] = data.get("runs", {})
        records: dict[str, SubagentRecord] = {}
        skipped = 0
        for agent_id, record_data in runs_raw.items():
            try:
                records[agent_id] = SubagentRecord.model_validate(record_data)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"SubagentRunsFile: skipping malformed record {agent_id!r}: {exc}"
                )
                skipped += 1

        if skipped:
            logger.warning(f"SubagentRunsFile: skipped {skipped} malformed record(s)")

        logger.debug(
            f"SubagentRunsFile: loaded {len(records)} record(s) from {self._path}"
        )
        return records

    def save(self, records: dict[str, SubagentRecord]) -> None:
        """Persist all records to disk atomically.

        Uses write-to-temp + rename so that a crash during write never
        leaves a half-written runs.json.

        Errors are logged but never raised — persistence failures must
        not break sub-agent logic.
        """
        payload: dict[str, Any] = {
            "version": _SCHEMA_VERSION,
            "runs": {
                agent_id: record.model_dump(mode="json")
                for agent_id, record in records.items()
            },
        }

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._path.with_suffix(".tmp")
            tmp_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            # Atomic replace (POSIX-atomic; on Windows best-effort)
            tmp_path.replace(self._path)
            logger.debug(
                f"SubagentRunsFile: persisted {len(records)} record(s) "
                f"to {self._path}"
            )
        except OSError as exc:
            logger.error(
                f"SubagentRunsFile: failed to persist to {self._path}: {exc}"
            )

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _quarantine_corrupt_file(self) -> None:
        """Rename the corrupt file so it is not lost but won't block startup."""
        backup = self._path.with_suffix(".corrupt.json")
        try:
            self._path.rename(backup)
            logger.warning(
                f"SubagentRunsFile: corrupt file moved to {backup} "
                "(will start with empty registry)"
            )
        except OSError as exc:
            logger.warning(
                f"SubagentRunsFile: could not quarantine corrupt file: {exc}"
            )


# ---------------------------------------------------------------------------
# Persistent agent directory manager
# ---------------------------------------------------------------------------


class AgentDirectory:
    """Manages the on-disk directory for a persistent (session-mode) agent.

    Directory layout:
        {workspace}/agents/{name}/
        ├── agent.json       Agent metadata (config, stats, timestamps)
        └── sessions/        Session transcript files for this agent
    """

    def __init__(self, workspace: Path, agent_name: str) -> None:
        self.root = Path(workspace) / "agents" / agent_name
        self.agent_name = agent_name

    def ensure(self) -> Path:
        """Create the directory structure and return the root path."""
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "sessions").mkdir(exist_ok=True)
        return self.root

    def save_agent_json(self, record: SubagentRecord) -> None:
        """Persist agent metadata to agent.json."""
        profile = PersistentSubagentProfile.from_subagent_config(
            name=self.agent_name,
            config=record.config,
            created_at=record.created_at,
            last_active_at=record.completed_at or record.started_at or record.created_at,
            last_run_agent_id=record.agent_id,
            last_run_state=record.state.value,
        )
        self.save_profile_json(profile)

    def save_profile_json(self, profile: PersistentSubagentProfile) -> None:
        """Persist a persistent subagent profile to agent.json."""
        agent_json_path = self.root / "agent.json"
        try:
            agent_json_path.write_text(
                json.dumps(profile.model_dump(mode="json"), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.debug(f"AgentDirectory: saved agent.json for {profile.name!r}")
        except OSError as exc:
            logger.error(f"AgentDirectory: failed to save agent.json for {profile.name!r}: {exc}")

    def load_agent_json(self) -> Optional[dict[str, Any]]:
        """Load raw agent.json, returning None if not found or corrupt."""
        agent_json_path = self.root / "agent.json"
        if not agent_json_path.exists():
            return None
        try:
            return json.loads(agent_json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error(f"AgentDirectory: failed to load agent.json for {self.agent_name!r}: {exc}")
            return None

    def load_profile_json(self) -> Optional[PersistentSubagentProfile]:
        """Load agent.json as a PersistentSubagentProfile."""
        data = self.load_agent_json()
        if data is None:
            return None
        try:
            # Backward-compat: old files used agent_id/model/state fields only.
            profile_data = dict(data)
            profile_data.pop("agent_id", None)
            profile_data.pop("spawn_mode", None)
            profile_data.pop("state", None)
            profile_data.pop("total_tokens", None)
            if "name" not in profile_data:
                profile_data["name"] = self.agent_name
            return PersistentSubagentProfile.model_validate(profile_data)
        except Exception as exc:
            logger.error(f"AgentDirectory: failed to parse profile for {self.agent_name!r}: {exc}")
            return None

    def exists(self) -> bool:
        """Return True if the agent directory exists."""
        return self.root.exists()

    @staticmethod
    def list_agents(workspace: Path) -> list[str]:
        """List all persistent agent directories under {workspace}/agents/."""
        agents_dir = Path(workspace) / "agents"
        if not agents_dir.exists():
            return []
        return sorted(d.name for d in agents_dir.iterdir() if d.is_dir())

    @staticmethod
    def list_profiles(workspace: Path) -> list[PersistentSubagentProfile]:
        """Load all persistent subagent profiles from disk."""
        profiles: list[PersistentSubagentProfile] = []
        for name in AgentDirectory.list_agents(workspace):
            profile = AgentDirectory(workspace, name).load_profile_json()
            if profile is not None:
                profiles.append(profile)
        return profiles


# ---------------------------------------------------------------------------
# Startup restoration
# ---------------------------------------------------------------------------


def restore_subagent_runs(
    records: dict[str, SubagentRecord],
) -> dict[str, SubagentRecord]:
    """Reconcile loaded records after a process restart.

    Any record still in PENDING or RUNNING state was orphaned by a process
    crash or SIGKILL.  Transition those to FAILED with a diagnostic message
    so the user sees a clear explanation rather than a stale "running" entry.

    The dict is mutated in-place and also returned for convenience.
    """
    now = time.time()
    orphaned = 0

    for record in records.values():
        if record.state in (SubagentState.PENDING, SubagentState.RUNNING):
            record.state = SubagentState.FAILED
            record.error = (
                "Orphaned: the process that was running this sub-agent restarted"
            )
            if record.completed_at is None:
                record.completed_at = now
            orphaned += 1

    if orphaned:
        logger.warning(
            f"restore_subagent_runs: marked {orphaned} orphaned "
            "record(s) as FAILED (process restarted)"
        )
    else:
        logger.debug("restore_subagent_runs: no orphaned records found")

    return records


# ---------------------------------------------------------------------------
# Background sweeper
# ---------------------------------------------------------------------------


class SubagentSweeper:
    """Periodic asyncio task that removes stale terminal records.

    Modeled on HeartbeatService: asyncio loop with graceful start/stop,
    configurable interval, and silent error recovery.
    """

    _TERMINAL_STATES = frozenset(
        [SubagentState.COMPLETED, SubagentState.FAILED, SubagentState.CANCELLED]
    )

    def __init__(
        self,
        registry: SubagentRegistry,
        session_manager: Optional[SessionManager] = None,
        archive_after_minutes: int = 60,
        interval_seconds: int = 60,
    ) -> None:
        self._registry = registry
        self._session_manager = session_manager
        self._archive_threshold = archive_after_minutes * 60.0
        self._interval = interval_seconds
        self._running = False
        self._task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background sweep loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="subagent-sweeper")
        logger.info(
            f"SubagentSweeper: started "
            f"(interval={self._interval}s, "
            f"archive_after={self._archive_threshold / 60:.0f}m)"
        )

    async def stop(self) -> None:
        """Stop the sweep loop gracefully."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("SubagentSweeper: stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self._interval)
                if not self._running:
                    break
                self._sweep()
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                logger.error(f"SubagentSweeper: unexpected error: {exc}")

    def _sweep(self) -> None:
        """Remove terminal run-mode records older than the archive threshold.

        Session-mode agents are skipped — they are never auto-archived.
        Before removing a record, its session data is archived (renamed with
        a .deleted.{timestamp} suffix rather than hard-deleted) for audit.
        """
        now = time.time()
        to_remove: list[str] = []
        session_keys_to_archive: list[str] = []

        for record in self._registry.list_all():
            # Never archive persistent session-mode agents
            if record.spawn_mode == SpawnMode.SESSION:
                continue
            if record.state not in self._TERMINAL_STATES:
                continue
            age = now - (record.completed_at or record.created_at)
            if age >= self._archive_threshold:
                to_remove.append(record.agent_id)
                session_keys_to_archive.append(record.session_key)

        # Archive session files before removing registry records
        for session_key in session_keys_to_archive:
            self._archive_session(session_key)

        for agent_id in to_remove:
            self._registry.remove(agent_id)

        if to_remove:
            logger.info(
                f"SubagentSweeper: archived {len(to_remove)} stale record(s)"
            )
        else:
            logger.debug("SubagentSweeper: sweep complete, no stale records")

    def _archive_session(self, session_key: str) -> None:
        """Archive session data via the session manager.

        For FileSessionStore this renames the session files to
        .deleted.{timestamp} rather than deleting them.
        For other backends the default delete behaviour is used.
        """
        if self._session_manager is None:
            return
        try:
            self._session_manager.archive(session_key)
        except Exception as exc:
            logger.debug(
                f"SubagentSweeper: error archiving session {session_key!r}: {exc}"
            )
