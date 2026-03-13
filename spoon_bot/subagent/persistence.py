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
from typing import TYPE_CHECKING, Any

from loguru import logger

from spoon_bot.subagent.models import SubagentRecord, SubagentState

if TYPE_CHECKING:
    from spoon_bot.subagent.registry import SubagentRegistry

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
        archive_after_minutes: int = 60,
        interval_seconds: int = 60,
    ) -> None:
        self._registry = registry
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
        """Remove terminal records that are older than the archive threshold.

        Uses registry.remove() which already holds the lock and triggers
        persistence, so we don't need additional locking here.
        """
        now = time.time()
        to_remove: list[str] = []

        for record in self._registry.list_all():
            if record.state not in self._TERMINAL_STATES:
                continue
            age = now - (record.completed_at or record.created_at)
            if age >= self._archive_threshold:
                to_remove.append(record.agent_id)

        for agent_id in to_remove:
            self._registry.remove(agent_id)

        if to_remove:
            logger.info(
                f"SubagentSweeper: archived {len(to_remove)} stale record(s)"
            )
        else:
            logger.debug("SubagentSweeper: sweep complete, no stale records")
