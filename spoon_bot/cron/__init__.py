"""Cron service public exports."""

from spoon_bot.cron.models import (
    AtSchedule,
    CronConversationScope,
    CronDeliveryTarget,
    CronExecutionResult,
    CronExpressionSchedule,
    CronJob,
    CronJobCreate,
    CronJobPatch,
    CronJobState,
    CronRunLogEntry,
    CronServiceStatus,
    EverySchedule,
)
from spoon_bot.cron.run_log import CronRunLog
from spoon_bot.cron.store import CronStore

try:
    from spoon_bot.cron.executor import CronExecutor
    from spoon_bot.cron.service import CronService, create_cron_service
except Exception:  # pragma: no cover - optional import for offline store/model access
    CronExecutor = None  # type: ignore[assignment]
    CronService = None  # type: ignore[assignment]
    create_cron_service = None  # type: ignore[assignment]

__all__ = [
    "AtSchedule",
    "CronConversationScope",
    "CronDeliveryTarget",
    "CronExecutionResult",
    "CronExecutor",
    "CronExpressionSchedule",
    "CronJob",
    "CronJobCreate",
    "CronJobPatch",
    "CronJobState",
    "CronRunLog",
    "CronRunLogEntry",
    "CronService",
    "CronServiceStatus",
    "CronStore",
    "EverySchedule",
    "create_cron_service",
]
