"""Sub-agent registry — lifecycle state tracking with optional JSON persistence."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from loguru import logger

from spoon_bot.subagent.models import SubagentRecord, SubagentState

if TYPE_CHECKING:
    from spoon_bot.subagent.persistence import SubagentRunsFile


# Valid state transitions
_VALID_TRANSITIONS: dict[SubagentState, set[SubagentState]] = {
    SubagentState.PENDING: {SubagentState.RUNNING, SubagentState.CANCELLED},
    SubagentState.RUNNING: {
        SubagentState.COMPLETED,
        SubagentState.FAILED,
        SubagentState.CANCELLED,
    },
    # Terminal states — no further transitions
    SubagentState.COMPLETED: set(),
    SubagentState.FAILED: set(),
    SubagentState.CANCELLED: set(),
}

_UNSET = object()


class SubagentRegistry:
    """In-memory registry of sub-agent records with optional JSON persistence.

    Thread-safe via RLock (matching SessionManager pattern).
    Tracks the full lifecycle of each sub-agent and maintains
    parent-child relationships.

    When *runs_file* is provided the registry loads previous records from
    disk on construction (orphan records are reconciled to FAILED) and
    persists the full records dict to disk after every mutation.
    Persistence errors are logged but never raised.
    """

    def __init__(
        self,
        runs_file: SubagentRunsFile | None = None,
    ) -> None:
        self._records: dict[str, SubagentRecord] = {}
        self._lock = threading.RLock()
        self._runs_file = runs_file

        if self._runs_file is not None:
            self._load_and_restore()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load_and_restore(self) -> None:
        """Load records from disk and reconcile orphans.

        Called once during __init__ while no other threads are active yet,
        so we access _records directly without the lock.
        """
        from spoon_bot.subagent.persistence import restore_subagent_runs

        loaded = self._runs_file.load()  # type: ignore[union-attr]
        restored = restore_subagent_runs(loaded)
        self._records = restored
        # Write reconciled state back immediately
        try:
            self._runs_file.save(self._records)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            logger.error(f"SubagentRegistry: initial persist failed: {exc}")

        if restored:
            logger.info(
                f"SubagentRegistry: restored {len(restored)} record(s) from "
                f"{self._runs_file.path}"  # type: ignore[union-attr]
            )

    def _persist(self) -> None:
        """Persist current records to disk.

        MUST be called while self._lock is already held.
        Errors are logged but never raised so that persistence failures
        cannot break sub-agent logic.
        """
        if self._runs_file is None:
            return
        try:
            self._runs_file.save(self._records)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"SubagentRegistry: persist failed: {exc}")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def register(self, record: SubagentRecord) -> None:
        """Register a new sub-agent record and link it to its parent."""
        with self._lock:
            self._records[record.agent_id] = record
            # Establish parent-child relationship
            if record.parent_id and record.parent_id in self._records:
                parent = self._records[record.parent_id]
                if record.agent_id not in parent.children:
                    parent.children.append(record.agent_id)
                    logger.debug(
                        f"Registered sub-agent {record.agent_id} "
                        f"as child of {record.parent_id}"
                    )
            self._persist()

    def get(self, agent_id: str) -> SubagentRecord | None:
        """Return the record for *agent_id*, or None if not found."""
        with self._lock:
            return self._records.get(agent_id)

    def remove(self, agent_id: str) -> bool:
        """Remove a record, re-link its children to its parent, and unlink from parent."""
        with self._lock:
            record = self._records.pop(agent_id, None)
            if record is None:
                return False

            grandparent_id = record.parent_id
            grandparent = (
                self._records.get(grandparent_id)
                if grandparent_id
                else None
            )

            # Unlink from parent's children list
            if grandparent is not None:
                try:
                    grandparent.children.remove(agent_id)
                except ValueError:
                    pass

            # Re-link children to grandparent so they don't become orphaned
            for child_id in record.children:
                child = self._records.get(child_id)
                if child is None:
                    continue
                child.parent_id = grandparent_id
                if grandparent is not None:
                    if child_id not in grandparent.children:
                        grandparent.children.append(child_id)
                # Adjust depth for the re-linked child and all its descendants
                depth_delta = -1 if record.depth > 0 else 0
                if depth_delta:
                    child.depth = max(0, child.depth + depth_delta)
                    stack = list(child.children)
                    while stack:
                        desc_id = stack.pop()
                        desc = self._records.get(desc_id)
                        if desc is None:
                            continue
                        desc.depth = max(0, desc.depth + depth_delta)
                        stack.extend(desc.children)

            self._persist()
            return True

    def update_fields(self, agent_id: str, **kwargs: Any) -> bool:
        """Update record fields without changing lifecycle state."""
        with self._lock:
            record = self._records.get(agent_id)
            if record is None:
                return False
            for key, value in kwargs.items():
                if hasattr(record, key):
                    setattr(record, key, value)
                else:
                    logger.debug(f"Unknown field '{key}' in update_fields kwargs")
            self._persist()
            return True

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def transition(
        self,
        agent_id: str,
        new_state: SubagentState,
        **kwargs: Any,
    ) -> bool:
        """Transition *agent_id* to *new_state* with optional field updates.

        Keyword arguments are applied to the record as attribute updates
        (e.g. ``result="..."`` or ``started_at=time.time()``).

        Returns True on success, False if the agent is not found or the
        transition is invalid.
        """
        with self._lock:
            record = self._records.get(agent_id)
            if record is None:
                logger.warning(f"Transition failed: agent {agent_id} not found")
                return False

            allowed = _VALID_TRANSITIONS.get(record.state, set())
            if new_state not in allowed:
                logger.warning(
                    f"Invalid transition {record.state!r} → {new_state!r} "
                    f"for sub-agent {agent_id}"
                )
                return False

            record.state = new_state
            for key, value in kwargs.items():
                if hasattr(record, key):
                    setattr(record, key, value)
                else:
                    logger.debug(f"Unknown field '{key}' in transition kwargs")

            logger.debug(
                f"Sub-agent {agent_id}: {record.state!r} → {new_state!r}"
            )
            self._persist()
            return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_all(self) -> list[SubagentRecord]:
        """Return a snapshot of all records."""
        with self._lock:
            return list(self._records.values())

    def list_by_parent(self, parent_id: str) -> list[SubagentRecord]:
        """Return all direct children of *parent_id*."""
        with self._lock:
            return [
                r for r in self._records.values()
                if r.parent_id == parent_id
            ]

    def count_active_children(self, parent_id: str) -> int:
        """Count children of *parent_id* that are PENDING or RUNNING."""
        with self._lock:
            return sum(
                1
                for r in self._records.values()
                if r.parent_id == parent_id
                and r.state in (SubagentState.PENDING, SubagentState.RUNNING)
            )

    def count_active_total(self) -> int:
        """Count all PENDING or RUNNING sub-agents across the registry."""
        with self._lock:
            return sum(
                1
                for r in self._records.values()
                if r.state in (SubagentState.PENDING, SubagentState.RUNNING)
            )

    def count_pending_descendants(self, agent_id: str) -> int:
        """Count PENDING or RUNNING descendants of *agent_id* (recursive BFS)."""
        with self._lock:
            root = self._records.get(agent_id)
            if root is None:
                return 0
            count = 0
            stack = list(root.children)
            while stack:
                child_id = stack.pop()
                child = self._records.get(child_id)
                if child:
                    if child.state in (SubagentState.PENDING, SubagentState.RUNNING):
                        count += 1
                    stack.extend(child.children)
            return count

    def get_descendants(self, agent_id: str) -> list[SubagentRecord]:
        """Return all descendants of *agent_id* via BFS traversal."""
        with self._lock:
            root = self._records.get(agent_id)
            if root is None:
                return []

            result: list[SubagentRecord] = []
            stack = list(root.children)
            while stack:
                child_id = stack.pop()
                child = self._records.get(child_id)
                if child:
                    result.append(child)
                    stack.extend(child.children)
            return result

    def prepare_for_resume(
        self,
        agent_id: str,
        *,
        task: str,
        label: str,
        run_id: str | object = _UNSET,
        parent_id: str | None | object = _UNSET,
        spawner_session_key: str | None | object = _UNSET,
        spawner_channel: str | None | object = _UNSET,
        spawner_metadata: Any = _UNSET,
        spawner_reply_to: str | None | object = _UNSET,
        model_name: str | None | object = _UNSET,
        config: Any = _UNSET,
    ) -> bool:
        """Reset a terminal session-mode agent for re-invocation.

        Bypasses the normal state machine (which forbids terminal → PENDING)
        because session-mode agents are designed to be re-invoked.

        Returns True on success, False if the agent is not found.
        """
        with self._lock:
            record = self._records.get(agent_id)
            if record is None:
                return False
            old_parent_id = record.parent_id
            old_depth = record.depth
            if run_id is not _UNSET:
                record.run_id = run_id
            record.task = task
            record.label = label
            if parent_id is not _UNSET:
                record.parent_id = parent_id
                if old_parent_id != parent_id:
                    if old_parent_id and old_parent_id in self._records:
                        old_parent = self._records[old_parent_id]
                        try:
                            old_parent.children.remove(agent_id)
                        except ValueError:
                            pass
                    if parent_id and parent_id in self._records:
                        new_parent = self._records[parent_id]
                        if agent_id not in new_parent.children:
                            new_parent.children.append(agent_id)
                        record.depth = new_parent.depth + 1
                    else:
                        record.depth = 1
                    depth_delta = record.depth - old_depth
                    if depth_delta != 0:
                        stack = list(record.children)
                        while stack:
                            child_id = stack.pop()
                            child = self._records.get(child_id)
                            if child is None:
                                continue
                            child.depth = max(1, child.depth + depth_delta)
                            stack.extend(child.children)
            if spawner_session_key is not _UNSET:
                record.spawner_session_key = spawner_session_key
            if spawner_channel is not _UNSET:
                record.spawner_channel = spawner_channel
            if spawner_metadata is not _UNSET:
                record.spawner_metadata = dict(spawner_metadata or {})
            if spawner_reply_to is not _UNSET:
                record.spawner_reply_to = spawner_reply_to
            if model_name is not _UNSET:
                record.model_name = model_name
            if config is not _UNSET:
                record.config = config
            record.state = SubagentState.PENDING
            record.result = None
            record.error = None
            record.started_at = None
            record.completed_at = None
            record.token_usage = None
            record.frozen_result_text = None
            self._persist()
            return True

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def __repr__(self) -> str:
        with self._lock:
            return f"SubagentRegistry(records={len(self._records)})"
