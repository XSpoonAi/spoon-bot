"""Schedule validation and next-run calculations."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from croniter import croniter

from spoon_bot.cron.models import (
    AtSchedule,
    CronExpressionSchedule,
    CronSchedule,
    EverySchedule,
    ensure_utc_datetime,
    utc_now,
)


def resolve_timezone(name: str | None) -> ZoneInfo:
    """Resolve a configured timezone name."""
    return ZoneInfo(name or "UTC")


def validate_schedule(schedule: CronSchedule) -> None:
    """Raise ValueError when a schedule is invalid."""
    if isinstance(schedule, CronExpressionSchedule):
        tz = resolve_timezone(schedule.timezone)
        try:
            croniter(schedule.expression, datetime.now(tz))
        except Exception as exc:
            raise ValueError(f"Invalid cron expression: {exc}") from exc


def compute_next_run(
    schedule: CronSchedule,
    *,
    now: datetime | None = None,
    created_at: datetime | None = None,
    after: datetime | None = None,
) -> datetime | None:
    """Compute the next run in UTC."""
    reference = ensure_utc_datetime(after or now or utc_now())

    if isinstance(schedule, AtSchedule):
        run_at = ensure_utc_datetime(schedule.run_at)
        if run_at <= reference:
            return None
        return run_at

    if isinstance(schedule, EverySchedule):
        interval = timedelta(seconds=schedule.seconds)
        anchor = ensure_utc_datetime(schedule.anchor_at or created_at or reference)
        if reference < anchor:
            return anchor

        elapsed = max(0.0, (reference - anchor).total_seconds())
        steps = math.floor(elapsed / schedule.seconds) + 1
        return anchor + (interval * steps)

    if isinstance(schedule, CronExpressionSchedule):
        tz = resolve_timezone(schedule.timezone)
        base = reference.astimezone(tz)
        iterator = croniter(schedule.expression, base)
        next_local = iterator.get_next(datetime)
        if next_local.tzinfo is None:
            next_local = next_local.replace(tzinfo=tz)
        return next_local.astimezone(timezone.utc)

    raise TypeError(f"Unsupported schedule type: {type(schedule).__name__}")


def is_due(next_run_at: datetime | None, *, now: datetime | None = None) -> bool:
    """Return True when a run is due at or before *now*."""
    if next_run_at is None:
        return False
    return ensure_utc_datetime(next_run_at) <= ensure_utc_datetime(now or utc_now())
