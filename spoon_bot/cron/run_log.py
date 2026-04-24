"""Append-only execution log for cron runs."""

from __future__ import annotations

import json
from pathlib import Path

from spoon_bot.cron.models import CronRunLogEntry


class CronRunLog:
    """Persists execution records as JSONL per job."""

    def __init__(self, root: Path | str, keep_lines: int = 2000) -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep_lines = max(1, int(keep_lines))

    def append(self, entry: CronRunLogEntry) -> None:
        """Append a single log entry and trim if needed."""
        path = self.root / f"{entry.job_id}.jsonl"
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(entry.model_dump_json() + "\n")
        self._trim(path)

    def tail(self, job_id: str, limit: int = 20) -> list[CronRunLogEntry]:
        """Read the most recent run log entries for a job."""
        path = self.root / f"{job_id}.jsonl"
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()[-max(1, limit):]
        return [CronRunLogEntry.model_validate(json.loads(line)) for line in lines if line.strip()]

    def _trim(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
        if len(lines) <= self.keep_lines:
            return
        with open(path, "w", encoding="utf-8") as handle:
            handle.writelines(lines[-self.keep_lines :])
