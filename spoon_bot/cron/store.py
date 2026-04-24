"""Persistent job store for scheduled tasks."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

from loguru import logger

from spoon_bot.cron.models import CronJob


class _FileLock:
    """Cross-platform advisory lock using a sidecar file."""

    def __init__(self, path: Path) -> None:
        self._path = Path(f"{path}.lock")
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def acquired(self):
        handle = open(self._path, "a+b")
        try:
            handle.seek(0)
            handle.write(b"0")
            handle.flush()
            handle.seek(0)

            if os.name == "nt":
                import msvcrt

                while True:
                    try:
                        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                        break
                    except OSError:
                        time.sleep(0.05)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)

            yield
        finally:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()


class CronStore:
    """JSON-backed store with atomic replacement and advisory locking."""

    VERSION = 1

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._backup_path = self.path.with_suffix(f"{self.path.suffix}.bak")
        self._lock = _FileLock(self.path)

    def load_jobs(self) -> list[CronJob]:
        """Load all persisted jobs."""
        with self._lock.acquired():
            return self._load_jobs_unlocked()

    def save_jobs(self, jobs: Iterable[CronJob]) -> None:
        """Persist the full job list atomically."""
        job_list = list(jobs)
        with self._lock.acquired():
            self._save_jobs_unlocked(job_list)

    def _load_jobs_unlocked(self) -> list[CronJob]:
        payload = self._read_payload(self.path)
        if payload is None and self._backup_path.exists():
            logger.warning(f"Cron store unreadable, recovering from backup: {self._backup_path}")
            payload = self._read_payload(self._backup_path)
        if payload is None:
            return []
        raw_jobs = payload.get("jobs", [])
        if not isinstance(raw_jobs, list):
            return []
        return [CronJob.model_validate(item) for item in raw_jobs]

    def _save_jobs_unlocked(self, jobs: list[CronJob]) -> None:
        payload = {
            "version": self.VERSION,
            "jobs": [job.model_dump(mode="json") for job in jobs],
        }
        if self.path.exists():
            try:
                self._backup_path.write_text(
                    self.path.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
            except Exception as exc:
                logger.debug(f"Unable to refresh cron backup: {exc}")

        temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(self.path)

    @staticmethod
    def _read_payload(path: Path) -> dict | None:
        if not path.exists():
            return None
        try:
            raw = path.read_text(encoding="utf-8")
            payload = json.loads(raw) if raw.strip() else {}
        except Exception as exc:
            logger.warning(f"Failed to read cron store {path}: {exc}")
            return None
        if not isinstance(payload, dict):
            logger.warning(f"Ignoring malformed cron store payload: {path}")
            return None
        return payload
