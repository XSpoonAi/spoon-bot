from datetime import datetime, timezone

from spoon_bot.cron.models import CronJob, EverySchedule
from spoon_bot.cron.store import CronStore


def test_cron_store_round_trip(tmp_path):
    store = CronStore(tmp_path / "jobs.json")
    job = CronJob(
        name="heartbeat",
        prompt="say hello",
        schedule=EverySchedule(
            seconds=60,
            anchor_at=datetime(2026, 3, 28, 0, 0, tzinfo=timezone.utc),
        ),
        target_mode="isolated",
    )

    store.save_jobs([job])
    loaded = store.load_jobs()

    assert len(loaded) == 1
    assert loaded[0].id == job.id
    assert loaded[0].schedule.kind == "every"
    assert loaded[0].prompt == "say hello"


def test_cron_store_round_trip_preserves_conversation_scope(tmp_path):
    store = CronStore(tmp_path / "jobs.json")
    job = CronJob(
        name="scoped",
        prompt="say hello",
        schedule=EverySchedule(
            seconds=60,
            anchor_at=datetime(2026, 3, 28, 0, 0, tzinfo=timezone.utc),
        ),
        target_mode="isolated",
        conversation_scope={
            "channel": "telegram",
            "account_id": "spoon_bot",
            "conversation_id": "123",
            "session_key": "telegram_spoon_bot_123",
        },
    )

    store.save_jobs([job])
    loaded = store.load_jobs()

    assert len(loaded) == 1
    assert loaded[0].conversation_scope is not None
    assert loaded[0].conversation_scope.channel == "telegram"
    assert loaded[0].conversation_scope.conversation_id == "123"
