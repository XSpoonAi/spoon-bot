from datetime import datetime, timezone

from spoon_bot.cron.models import AtSchedule, CronExpressionSchedule, EverySchedule
from spoon_bot.cron.schedule import compute_next_run


def test_at_schedule_returns_none_for_past_times():
    schedule = AtSchedule(run_at=datetime(2026, 3, 28, 8, 0, tzinfo=timezone.utc))
    now = datetime(2026, 3, 28, 9, 0, tzinfo=timezone.utc)

    assert compute_next_run(schedule, now=now) is None


def test_every_schedule_uses_anchor_for_next_run():
    schedule = EverySchedule(
        seconds=300,
        anchor_at=datetime(2026, 3, 28, 8, 0, tzinfo=timezone.utc),
    )
    now = datetime(2026, 3, 28, 8, 7, tzinfo=timezone.utc)

    next_run = compute_next_run(schedule, now=now)

    assert next_run == datetime(2026, 3, 28, 8, 10, tzinfo=timezone.utc)


def test_cron_schedule_honors_timezone():
    schedule = CronExpressionSchedule(
        expression="0 9 * * *",
        timezone="Asia/Shanghai",
    )
    now = datetime(2026, 3, 28, 0, 30, tzinfo=timezone.utc)

    next_run = compute_next_run(schedule, now=now)

    assert next_run == datetime(2026, 3, 28, 1, 0, tzinfo=timezone.utc)
