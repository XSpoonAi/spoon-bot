"""Structured cron tool for natural-language scheduling flows."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger

from spoon_bot.agent.tools.base import Tool
from spoon_bot.agent.tools.web import describe_web_search_capability
from spoon_bot.channels.delivery import (
    DeliveryBinding,
    binding_from_session_key,
    conversation_scope_from_binding,
    conversation_scope_from_parts,
    conversation_scope_from_session_key,
    normalize_channel_name,
)
from spoon_bot.cron.models import (
    AtSchedule,
    CronConversationScope,
    CronDeliveryTarget,
    CronExpressionSchedule,
    CronJob,
    CronJobCreate,
    CronJobPatch,
    CronRunLogEntry,
    EverySchedule,
)

if TYPE_CHECKING:
    from spoon_bot.agent.loop import AgentLoop
    from spoon_bot.cron.service import CronService
    from spoon_bot.session.manager import Session


_PENDING_DRAFT_KEY = "cron_pending_draft"
_LAST_JOB_KEY = "cron_last_job_id"
_CURRENT_DRAFT_VERSION = 1
_MUTATING_ACTIONS = {"create", "update", "delete"}
_LIVE_INFO_KEYWORDS = (
    "latest", "current", "today", "weather", "news", "headline", "price", "stock",
    "market", "update", "updates", "summarize updates", "check website", "monitor", "fetch", "search",
    "latest updates", "news digest", "web", "rss", "newsapi",
    "最新", "当前", "今天", "天气", "新闻", "头条", "价格", "股价", "行情",
    "摘要", "更新", "网页", "网站", "监控", "查询", "获取", "抓取", "检索",
)
_LIVE_INFO_ALLOWED_TOOLS = ["web_search", "web_fetch", "read_file", "list_dir"]
_CRON_PAYLOAD_FIELDS = set(CronJobCreate.model_fields.keys())
_CRON_PATCH_FIELDS = set(CronJobPatch.model_fields.keys())


def _non_empty(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_datetime(value: str, default_timezone: str) -> datetime:
    """Parse ISO datetime and normalize to UTC."""
    from zoneinfo import ZoneInfo

    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(default_timezone))
    return dt.astimezone(timezone.utc)


def _truncate(text: str, max_len: int = 72) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3].rstrip() + "..."


class CronTool(Tool):
    """Manage scheduled tasks from chat using a structured tool contract."""

    def __init__(self, require_confirmation: bool = True) -> None:
        self._require_confirmation = require_confirmation
        self._agent_loop: AgentLoop | None = None
        self._cron_service: CronService | None = None

    def set_agent_loop(self, agent_loop: AgentLoop) -> None:
        """Inject the owning agent loop."""
        self._agent_loop = agent_loop

    def set_cron_service(self, cron_service: CronService | None) -> None:
        """Inject the active cron service."""
        self._cron_service = cron_service

    @property
    def name(self) -> str:
        return "cron"

    @property
    def description(self) -> str:
        return (
            "Manage scheduled tasks and reminders. Use this tool whenever the user wants "
            "something to happen later, on a schedule, or repeatedly. Examples: reminders, "
            "daily digests, every-hour check-ins, 'tomorrow morning message me', or "
            "'cancel that reminder'. Default delivery is the current Telegram chat or "
            "Discord channel when available. For create/update/delete, first call this tool "
            "with confirm=false to prepare a draft and show a confirmation summary; only "
            "call again with confirm=true after the user explicitly confirms. Do NOT use "
            "shell commands to manage cron jobs. For news, weather, website checks, "
            "or other live-data jobs, prefer isolated execution unless the user explicitly "
            "asks to reuse the current chat context."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["status", "list", "create", "update", "delete", "run", "runs"],
                    "description": "Cron action to perform.",
                },
                "job_id": {
                    "type": "string",
                    "description": "Job id for update/delete/run/runs. Defaults to the most recently touched job in this chat when omitted.",
                },
                "name": {
                    "type": "string",
                    "description": "Optional human-readable job name. If omitted during create, a short name is derived from the message/prompt.",
                },
                "message": {
                    "type": "string",
                    "description": "Fixed reminder text to send back. When set, the tool converts it into a prompt that replies with exactly this message.",
                },
                "prompt": {
                    "type": "string",
                    "description": "Full agent prompt to run on schedule. Use this for richer recurring tasks.",
                },
                "schedule_kind": {
                    "type": "string",
                    "enum": ["at", "every", "cron"],
                    "description": "Schedule type.",
                },
                "run_at": {
                    "type": "string",
                    "description": "ISO-8601 datetime for one-shot jobs. Naive datetimes use the cron default timezone.",
                },
                "every_seconds": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Repeat interval in seconds when schedule_kind='every'.",
                },
                "cron_expression": {
                    "type": "string",
                    "description": "Cron expression when schedule_kind='cron', for example '0 9 * * 1'.",
                },
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone for cron expressions or naive one-shot datetimes. Defaults to the configured cron timezone.",
                },
                "target_mode": {
                    "type": "string",
                    "enum": ["session", "current", "main", "isolated"],
                    "description": "Execution mode. Use isolated for independent work, current to reuse this chat session, main for the default shared session, or session for an explicit session key.",
                },
                "session_key": {
                    "type": "string",
                    "description": "Explicit session key when target_mode='session'. For target_mode='current', the current chat session key is used automatically when omitted.",
                },
                "delivery_channel": {
                    "type": "string",
                    "description": "Explicit delivery channel full name, e.g. 'telegram:my_bot' or 'discord:main'. Defaults to the current chat when available.",
                },
                "delivery_account": {
                    "type": "string",
                    "description": "Explicit delivery account id.",
                },
                "chat_id": {
                    "type": "string",
                    "description": "Telegram chat id for delivery.",
                },
                "channel_id": {
                    "type": "string",
                    "description": "Discord channel id for delivery.",
                },
                "enabled": {
                    "type": "boolean",
                    "description": "Whether the job should be enabled after creation/update.",
                },
                "delivery_mode": {
                    "type": "string",
                    "enum": ["announce", "none"],
                    "description": "Whether scheduled runs should announce results back to a channel or keep them silent.",
                },
                "allowed_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional allowlist of tools this job may use while running.",
                },
                "max_attempts": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Maximum execution attempts for transient failures.",
                },
                "backoff_seconds": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 3600,
                    "description": "Fixed backoff between retries for transient failures.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Maximum number of runs to return for action='runs'.",
                },
                "confirm": {
                    "type": "boolean",
                    "description": "Must be true before create/update/delete actually mutates jobs.",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        action = str(kwargs.get("action") or "").strip().lower()
        if not action:
            return "Error: action is required."

        if action in _MUTATING_ACTIONS and self._is_cron_run_session():
            return (
                "Error: cron create/update/delete is blocked inside scheduled-run sessions "
                "to avoid recursive job creation."
            )

        if action == "status":
            return await self._status()
        if action == "list":
            return await self._list_jobs()
        if action == "create":
            return await self._create(kwargs)
        if action == "update":
            return await self._update(kwargs)
        if action == "delete":
            return await self._delete(kwargs)
        if action == "run":
            return await self._run(kwargs)
        if action == "runs":
            return await self._runs(kwargs)
        return f"Error: Unsupported cron action '{action}'."

    async def _status(self) -> str:
        service = self._get_cron_service()
        if service is None:
            return self._service_unavailable_message()

        status = await service.status()
        lines = [
            "Cron service status:",
            f"- Running: {status.running}",
            f"- Total jobs: {status.total_jobs}",
            f"- Enabled jobs: {status.enabled_jobs}",
            f"- Active runs: {status.active_runs}",
            f"- Store: {status.store_path}",
        ]
        if status.next_run_at is not None:
            lines.append(f"- Next run: {status.next_run_at.isoformat()}")
        return "\n".join(lines)

    async def _list_jobs(self) -> str:
        service = self._get_cron_service()
        if service is None:
            return self._service_unavailable_message()

        jobs = await service.list_jobs()
        visible = [job for job in jobs if self._job_matches_current_context(job)]
        if not visible:
            return (
                "No scheduled jobs are visible in this conversation. "
                "If the user wants a new reminder, gather the schedule and message, "
                "then prepare a create draft."
            )

        self._remember_job_id(visible[-1].id)
        lines = ["Scheduled jobs in this conversation:"]
        for job in visible:
            lines.append(
                f"- {job.id}: {job.name} | {self._format_schedule(job)} | "
                f"enabled={job.enabled} | next={job.state.next_run_at.isoformat() if job.state.next_run_at else 'none'}"
            )
        return "\n".join(lines)

    async def _create(self, raw_kwargs: dict[str, Any]) -> str:
        confirm = bool(raw_kwargs.get("confirm", False))
        merged = self._merge_with_pending("create", raw_kwargs)
        normalized, missing_or_error = self._normalize_create_payload(merged)
        if isinstance(missing_or_error, str):
            return missing_or_error

        if missing_or_error:
            self._store_pending_draft("create", normalized)
            return self._render_missing_fields("create", missing_or_error, normalized)

        if self._require_confirmation and not confirm:
            self._store_pending_draft("create", normalized)
            return self._render_confirmation_preview("create", normalized)

        service = self._get_cron_service()
        if service is None:
            return self._service_unavailable_message()

        payload = CronJobCreate(**{
            key: value for key, value in normalized.items() if key in _CRON_PAYLOAD_FIELDS
        })
        job = await service.create_job(payload)
        self._clear_pending_draft()
        self._remember_job_id(job.id)
        return self._render_create_success(job)

    async def _update(self, raw_kwargs: dict[str, Any]) -> str:
        confirm = bool(raw_kwargs.get("confirm", False))
        service = self._get_cron_service()
        if service is None:
            return self._service_unavailable_message()

        merged = self._merge_with_pending("update", raw_kwargs)
        job_id = self._resolve_job_id(merged.get("job_id"))
        if not job_id:
            return (
                "Need the job to update. Ask the user which scheduled task to change, "
                "or call cron list first."
            )

        try:
            job = await service.get_job(job_id)
        except KeyError:
            return f"Error: Cron job not found: {job_id}"

        if not self._job_matches_current_context(job):
            return f"Error: Cron job {job_id} is not accessible from this conversation."

        normalized, missing_or_error = self._normalize_update_payload(job, merged)
        if isinstance(missing_or_error, str):
            return missing_or_error
        if missing_or_error:
            self._store_pending_draft("update", normalized)
            return self._render_missing_fields("update", missing_or_error, normalized)

        if self._require_confirmation and not confirm:
            self._store_pending_draft("update", normalized)
            return self._render_confirmation_preview("update", normalized, current_job=job)

        patch_data = {key: value for key, value in normalized.items() if key != "job_id"}
        patch = CronJobPatch(**{
            key: value for key, value in patch_data.items() if key in _CRON_PATCH_FIELDS
        })
        updated = await service.update_job(job_id, patch)
        self._clear_pending_draft()
        self._remember_job_id(updated.id)
        return self._render_update_success(updated)

    async def _delete(self, raw_kwargs: dict[str, Any]) -> str:
        confirm = bool(raw_kwargs.get("confirm", False))
        service = self._get_cron_service()
        if service is None:
            return self._service_unavailable_message()

        merged = self._merge_with_pending("delete", raw_kwargs)
        job_id = self._resolve_job_id(merged.get("job_id"))
        if not job_id:
            return "Need the job id to delete. Ask the user which scheduled task to remove."

        try:
            job = await service.get_job(job_id)
        except KeyError:
            return f"Error: Cron job not found: {job_id}"

        if not self._job_matches_current_context(job):
            return f"Error: Cron job {job_id} is not accessible from this conversation."

        draft = {"job_id": job.id, "name": job.name}
        if self._require_confirmation and not confirm:
            self._store_pending_draft("delete", draft)
            return (
                "Delete scheduled task draft prepared:\n"
                f"- Job: {job.id}\n"
                f"- Name: {job.name}\n"
                f"- Schedule: {self._format_schedule(job)}\n\n"
                "Ask the user to confirm. After explicit confirmation, call this tool again "
                "with action='delete' and confirm=true."
            )

        deleted = await service.delete_job(job.id)
        self._clear_pending_draft()
        if deleted:
            self._remember_job_id(job.id)
            return f"Deleted scheduled task {job.id} ({job.name})."
        return f"Error: Failed to delete cron job {job.id}."

    async def _run(self, raw_kwargs: dict[str, Any]) -> str:
        service = self._get_cron_service()
        if service is None:
            return self._service_unavailable_message()

        job_id = self._resolve_job_id(raw_kwargs.get("job_id"))
        if not job_id:
            return "Need the job id to run. Ask the user which scheduled task to execute."

        try:
            job = await service.get_job(job_id)
        except KeyError:
            return f"Error: Cron job not found: {job_id}"

        if not self._job_matches_current_context(job):
            return f"Error: Cron job {job_id} is not accessible from this conversation."

        result = await service.run_now(job.id)
        self._remember_job_id(job.id)
        lines = [
            f"Ran scheduled task {job.id} ({job.name}).",
            f"- Status: {result.status}",
            f"- Session: {result.session_key}",
            f"- Delivery: {result.delivery_status or 'unknown'}",
        ]
        if result.error:
            lines.append(f"- Error: {result.error}")
        if result.output:
            lines.append(f"- Output: {_truncate(result.output, 220)}")
        return "\n".join(lines)

    async def _runs(self, raw_kwargs: dict[str, Any]) -> str:
        service = self._get_cron_service()
        if service is None:
            return self._service_unavailable_message()

        job_id = self._resolve_job_id(raw_kwargs.get("job_id"))
        if not job_id:
            return "Need the job id to inspect runs. Ask the user which scheduled task they mean."

        try:
            job = await service.get_job(job_id)
        except KeyError:
            return f"Error: Cron job not found: {job_id}"

        if not self._job_matches_current_context(job):
            return f"Error: Cron job {job_id} is not accessible from this conversation."

        limit = int(raw_kwargs.get("limit") or 5)
        runs = await service.get_runs(job.id, limit=max(1, min(limit, 50)))
        self._remember_job_id(job.id)
        if not runs:
            return f"No recorded runs for scheduled task {job.id}."

        lines = [f"Recent runs for {job.id} ({job.name}):"]
        for run in runs:
            lines.append(self._format_run(run))
        return "\n".join(lines)

    def _normalize_create_payload(self, raw: dict[str, Any]) -> tuple[dict[str, Any], list[str] | str]:
        prompt = self._normalize_prompt(raw)
        name = _non_empty(raw.get("name")) or self._derive_name(prompt)
        schedule, schedule_missing = self._build_schedule(raw, require_schedule=True)
        target_mode = str(raw.get("target_mode") or "isolated").strip().lower()
        if target_mode not in {"session", "current", "main", "isolated"}:
            return {}, "Error: target_mode must be 'session', 'current', 'main', or 'isolated'."
        delivery_mode = str(raw.get("delivery_mode") or "announce").strip().lower()
        if delivery_mode not in {"announce", "none"}:
            return {}, "Error: delivery_mode must be 'announce' or 'none'."
        session_key = _non_empty(raw.get("session_key"))
        if target_mode in {"session", "current"} and not session_key:
            session_key = self._current_session_key()
        elif target_mode in {"main", "isolated"}:
            session_key = None
        delivery = None if delivery_mode == "none" else self._build_delivery(raw)
        conversation_scope = self._build_conversation_scope(
            delivery=delivery,
            session_key=session_key,
        )
        allowed_tools = self._normalize_allowed_tools(raw.get("allowed_tools"))
        capability_warning = self._build_capability_warning(prompt)
        if allowed_tools is None and self._looks_like_live_info_task(prompt):
            allowed_tools = list(_LIVE_INFO_ALLOWED_TOOLS)

        max_attempts = raw.get("max_attempts")
        backoff_seconds = raw.get("backoff_seconds")

        normalized: dict[str, Any] = {
            "name": name,
            "prompt": prompt,
            "schedule": schedule,
            "target_mode": target_mode,
            "session_key": session_key,
            "delivery": delivery,
            "conversation_scope": conversation_scope,
            "delivery_mode": delivery_mode,
            "allowed_tools": allowed_tools,
            "max_attempts": 1 if max_attempts is None else int(max_attempts),
            "backoff_seconds": 0 if backoff_seconds is None else int(backoff_seconds),
            "enabled": bool(raw.get("enabled", True)),
        }
        if capability_warning:
            normalized["capability_warning"] = capability_warning
        if target_mode == "current" and self._looks_like_live_info_task(prompt):
            normalized["target_mode_warning"] = (
                "This job looks like live/background work. current mode reuses the active "
                "chat context and can drift; isolated mode is recommended unless the user "
                "explicitly wants ongoing session context."
            )

        missing: list[str] = []
        if not prompt:
            missing.append("message or prompt")
        if not name:
            missing.append("name")
        if schedule_missing:
            missing.extend(schedule_missing)
        if target_mode in {"session", "current"} and not session_key:
            missing.append(f"session_key for {target_mode} mode")
        return normalized, missing

    def _normalize_update_payload(
        self,
        current_job: CronJob,
        raw: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str] | str]:
        normalized: dict[str, Any] = {"job_id": current_job.id}
        changed = False

        name = _non_empty(raw.get("name"))
        if name is not None:
            normalized["name"] = name
            changed = True

        prompt = self._normalize_prompt(raw)
        if prompt:
            normalized["prompt"] = prompt
            changed = True

        schedule, schedule_missing = self._build_schedule(raw, require_schedule=False)
        if schedule_missing:
            return {}, schedule_missing
        if schedule is not None:
            normalized["schedule"] = schedule
            changed = True

        target_mode = _non_empty(raw.get("target_mode"))
        session_key = _non_empty(raw.get("session_key"))
        delivery_mode = _non_empty(raw.get("delivery_mode"))
        if target_mode is not None:
            target_mode = target_mode.lower()
            if target_mode not in {"session", "current", "main", "isolated"}:
                return {}, "Error: target_mode must be 'session', 'current', 'main', or 'isolated'."
            normalized["target_mode"] = target_mode
            changed = True
            if target_mode in {"session", "current"} and not session_key:
                session_key = self._current_session_key()
            elif target_mode in {"main", "isolated"}:
                normalized["session_key"] = None
        if delivery_mode is not None:
            delivery_mode = delivery_mode.lower()
            if delivery_mode not in {"announce", "none"}:
                return {}, "Error: delivery_mode must be 'announce' or 'none'."
            normalized["delivery_mode"] = delivery_mode
            changed = True
        target_mode_effective = target_mode or current_job.target_mode
        if session_key is not None and target_mode_effective in {"session", "current"}:
            normalized["session_key"] = session_key
            changed = True

        if raw.get("enabled") is not None:
            normalized["enabled"] = bool(raw["enabled"])
            changed = True

        delivery_supplied = any(
            raw.get(key) is not None
            for key in ("delivery_channel", "delivery_account", "chat_id", "channel_id")
        )
        if not delivery_supplied and raw.get("delivery") is not None:
            delivery_supplied = True
        effective_delivery_mode = str(
            normalized.get("delivery_mode") or current_job.delivery_mode
        )
        if delivery_supplied:
            if effective_delivery_mode == "none":
                return {}, "Error: delivery overrides require delivery_mode='announce'."
            delivery = self._build_delivery(raw, allow_default=False)
            if delivery is None:
                return {}, "Error: delivery_channel is required when overriding delivery target."
            normalized["delivery"] = delivery
            changed = True
        elif delivery_mode == "none":
            normalized["delivery"] = None
            changed = True
        elif delivery_mode == "announce" and current_job.delivery is None:
            default_delivery = self._build_delivery(raw, allow_default=True)
            if default_delivery is not None:
                normalized["delivery"] = default_delivery
                changed = True

        if raw.get("max_attempts") is not None:
            normalized["max_attempts"] = int(raw["max_attempts"])
            changed = True
        if raw.get("backoff_seconds") is not None:
            normalized["backoff_seconds"] = int(raw["backoff_seconds"])
            changed = True
        if raw.get("allowed_tools") is not None:
            normalized["allowed_tools"] = self._normalize_allowed_tools(raw.get("allowed_tools"))
            changed = True

        if not changed:
            return {}, ["field(s) to change"]

        target_mode_final = str(normalized.get("target_mode") or current_job.target_mode)
        if target_mode_final in {"main", "isolated"}:
            normalized["session_key"] = None
        session_key_final = normalized.get("session_key", current_job.session_key)
        if target_mode_final in {"session", "current"} and not session_key_final:
            return {}, [f"session_key for {target_mode_final} mode"]

        effective_delivery = normalized.get("delivery", current_job.delivery)
        normalized["conversation_scope"] = (
            current_job.conversation_scope
            or self._build_conversation_scope(
                delivery=effective_delivery,
                session_key=session_key_final,
            )
        )
        prompt_to_report = str(normalized.get("prompt") or current_job.prompt)
        capability_warning = self._build_capability_warning(prompt_to_report)
        if capability_warning:
            normalized["capability_warning"] = capability_warning
        target_mode_to_report = str(normalized.get("target_mode") or current_job.target_mode)
        if target_mode_to_report == "current" and self._looks_like_live_info_task(prompt_to_report):
            normalized["target_mode_warning"] = (
                "This job looks like live/background work. current mode reuses the active "
                "chat context and can drift; isolated mode is recommended unless the user "
                "explicitly wants ongoing session context."
            )

        return normalized, []

    def _normalize_prompt(self, raw: dict[str, Any]) -> str | None:
        prompt = _non_empty(raw.get("prompt"))
        if prompt:
            return prompt

        message = _non_empty(raw.get("message"))
        if not message:
            return None
        return f"Reply with exactly: {message}"

    def _derive_name(self, prompt: str | None) -> str | None:
        if not prompt:
            return None
        base = prompt.replace("Reply with exactly:", "").strip() or prompt
        return _truncate(base, 48)

    @staticmethod
    def _normalize_allowed_tools(raw: Any) -> list[str] | None:
        if raw is None:
            return None
        if not isinstance(raw, list):
            return None
        normalized: list[str] = []
        seen: set[str] = set()
        for item in raw:
            text = _non_empty(item)
            if not text or text in seen:
                continue
            normalized.append(text)
            seen.add(text)
        return normalized or None

    def _looks_like_live_info_task(self, prompt: str | None) -> bool:
        if not prompt:
            return False
        lowered = prompt.lower()
        return any(keyword in lowered for keyword in _LIVE_INFO_KEYWORDS)

    def _build_capability_warning(self, prompt: str | None) -> str | None:
        if not self._looks_like_live_info_task(prompt):
            return None
        available, message = describe_web_search_capability()
        if available:
            return None
        return message

    def _build_schedule(
        self,
        raw: dict[str, Any],
        *,
        require_schedule: bool,
    ) -> tuple[AtSchedule | EverySchedule | CronExpressionSchedule | None, list[str]]:
        existing_schedule = raw.get("schedule")
        kind = _non_empty(raw.get("schedule_kind"))
        run_at = _non_empty(raw.get("run_at"))
        every_seconds = raw.get("every_seconds")
        cron_expression = _non_empty(raw.get("cron_expression"))
        timezone_name = _non_empty(raw.get("timezone")) or self._default_timezone()

        if not any(value is not None for value in (kind, run_at, every_seconds, cron_expression)):
            if isinstance(existing_schedule, (AtSchedule, EverySchedule, CronExpressionSchedule)):
                return existing_schedule, []
            if isinstance(existing_schedule, dict):
                try:
                    schedule_kind = str(existing_schedule.get("kind") or "").strip().lower()
                    if schedule_kind == "at":
                        return AtSchedule.model_validate(existing_schedule), []
                    if schedule_kind == "every":
                        return EverySchedule.model_validate(existing_schedule), []
                    if schedule_kind == "cron":
                        return CronExpressionSchedule.model_validate(existing_schedule), []
                except Exception as exc:
                    return None, [f"valid schedule ({exc})"]

        inferred_kinds = set()
        if run_at is not None:
            inferred_kinds.add("at")
        if every_seconds is not None:
            inferred_kinds.add("every")
        if cron_expression is not None:
            inferred_kinds.add("cron")
        if kind:
            inferred_kinds.add(kind)

        if not inferred_kinds:
            return (None, ["schedule"]) if require_schedule else (None, [])

        if len(inferred_kinds) > 1:
            return None, ["exactly one schedule kind"]

        schedule_kind = inferred_kinds.pop()
        try:
            if schedule_kind == "at":
                if run_at is None:
                    return None, ["run_at"]
                return AtSchedule(run_at=_parse_datetime(run_at, timezone_name)), []
            if schedule_kind == "every":
                if every_seconds is None:
                    return None, ["every_seconds"]
                return EverySchedule(seconds=int(every_seconds)), []
            if schedule_kind == "cron":
                if cron_expression is None:
                    return None, ["cron_expression"]
                return CronExpressionSchedule(expression=cron_expression, timezone=timezone_name), []
        except Exception as exc:
            return None, [f"valid schedule ({exc})"]

        return None, ["valid schedule kind"]

    def _build_delivery(
        self,
        raw: dict[str, Any],
        *,
        allow_default: bool = True,
    ) -> CronDeliveryTarget | None:
        existing_delivery = raw.get("delivery")
        channel = _non_empty(raw.get("delivery_channel"))
        account = _non_empty(raw.get("delivery_account"))
        chat_id = _non_empty(raw.get("chat_id"))
        channel_id = _non_empty(raw.get("channel_id"))

        if (
            channel is None
            and account is None
            and chat_id is None
            and channel_id is None
        ):
            if isinstance(existing_delivery, CronDeliveryTarget):
                return existing_delivery
            if isinstance(existing_delivery, dict):
                try:
                    return CronDeliveryTarget.model_validate(existing_delivery)
                except Exception:
                    pass

        if channel is None and allow_default:
            binding = self._current_binding()
            if binding is not None:
                full_channel, account_id = normalize_channel_name(
                    binding.channel,
                    binding.account_id,
                )
                return CronDeliveryTarget(
                    channel=full_channel or binding.channel,
                    account_id=account_id,
                    target=dict(binding.target),
                    session_key=binding.session_key,
                )

        if channel is None:
            return None

        channel, account = normalize_channel_name(channel, account)

        target: dict[str, str] = {}
        if chat_id is not None:
            target["chat_id"] = chat_id
        if channel_id is not None:
            target["channel_id"] = channel_id

        session_key = self._current_session_key()
        return CronDeliveryTarget(
            channel=channel,
            account_id=account,
            target=target,
            session_key=session_key,
        )

    def _render_missing_fields(
        self,
        action: str,
        missing: list[str],
        normalized: dict[str, Any],
    ) -> str:
        draft_note = " The partial draft has been saved for this conversation." if normalized else ""
        return (
            f"I still need more information before I can {action} this scheduled task: "
            + ", ".join(missing)
            + ". Ask the user for the missing pieces and then call the cron tool again."
            + draft_note
        )

    def _render_confirmation_preview(
        self,
        action: str,
        normalized: dict[str, Any],
        *,
        current_job: CronJob | None = None,
    ) -> str:
        lines = [f"{action.title()} scheduled task draft prepared:"]
        if current_job is not None:
            lines.append(f"- Job: {current_job.id} ({current_job.name})")
        if normalized.get("name"):
            lines.append(f"- Name: {normalized['name']}")
        if normalized.get("prompt"):
            lines.append(f"- Prompt: {_truncate(str(normalized['prompt']), 180)}")
        schedule = normalized.get("schedule")
        if schedule is not None:
            lines.append(f"- Schedule: {self._format_schedule(schedule)}")
        if normalized.get("target_mode"):
            lines.append(f"- Target mode: {normalized['target_mode']}")
        if normalized.get("session_key"):
            lines.append(f"- Session key: {normalized['session_key']}")
        delivery = normalized.get("delivery")
        lines.append(f"- Delivery mode: {normalized.get('delivery_mode', 'announce')}")
        if delivery is not None or normalized.get("delivery_mode") != "none":
            lines.append(f"- Delivery: {self._format_delivery(delivery)}")
        if normalized.get("max_attempts") is not None:
            lines.append(f"- Max attempts: {normalized['max_attempts']}")
        if normalized.get("backoff_seconds") is not None:
            lines.append(f"- Backoff: {normalized['backoff_seconds']}s")
        if normalized.get("allowed_tools"):
            lines.append(f"- Allowed tools: {', '.join(normalized['allowed_tools'])}")
        if normalized.get("enabled") is not None:
            lines.append(f"- Enabled: {normalized['enabled']}")
        if normalized.get("capability_warning"):
            lines.append(f"- Capability warning: {normalized['capability_warning']}")
        if normalized.get("target_mode_warning"):
            lines.append(f"- Target mode note: {normalized['target_mode_warning']}")
        lines.append("")
        lines.append(
            "Do not create/update/delete yet. Ask the user to confirm, then call this tool "
            f"again with action='{action}' and confirm=true."
        )
        return "\n".join(lines)

    def _render_create_success(self, job: CronJob) -> str:
        return (
            "Created scheduled task successfully:\n"
            f"- Job: {job.id}\n"
            f"- Name: {job.name}\n"
            f"- Schedule: {self._format_schedule(job)}\n"
            f"- Target mode: {job.target_mode}\n"
            f"- Delivery mode: {job.delivery_mode}\n"
            f"- Delivery: {self._format_delivery(job.delivery)}\n"
            f"- Allowed tools: {', '.join(job.allowed_tools) if job.allowed_tools else 'automation defaults'}\n"
            f"- Max attempts: {job.max_attempts}\n"
            f"- Backoff: {job.backoff_seconds}s\n"
            f"- Next run: {job.state.next_run_at.isoformat() if job.state.next_run_at else 'none'}"
        )

    def _render_update_success(self, job: CronJob) -> str:
        return (
            "Updated scheduled task successfully:\n"
            f"- Job: {job.id}\n"
            f"- Name: {job.name}\n"
            f"- Schedule: {self._format_schedule(job)}\n"
            f"- Target mode: {job.target_mode}\n"
            f"- Delivery mode: {job.delivery_mode}\n"
            f"- Delivery: {self._format_delivery(job.delivery)}\n"
            f"- Allowed tools: {', '.join(job.allowed_tools) if job.allowed_tools else 'automation defaults'}\n"
            f"- Max attempts: {job.max_attempts}\n"
            f"- Backoff: {job.backoff_seconds}s\n"
            f"- Enabled: {job.enabled}"
        )

    def _format_schedule(self, job_or_schedule: CronJob | AtSchedule | EverySchedule | CronExpressionSchedule) -> str:
        schedule = job_or_schedule.schedule if isinstance(job_or_schedule, CronJob) else job_or_schedule
        if isinstance(schedule, AtSchedule):
            return schedule.run_at.isoformat()
        if isinstance(schedule, EverySchedule):
            return f"every {schedule.seconds}s"
        return f"{schedule.expression} ({schedule.timezone or self._default_timezone()})"

    def _format_delivery(self, delivery: CronDeliveryTarget | None) -> str:
        if delivery is None:
            return "none"
        channel = delivery.channel
        if ":" not in channel and delivery.account_id:
            channel = f"{channel}:{delivery.account_id}"
        return f"{channel} -> {delivery.target or '{}'}"

    def _format_run(self, run: CronRunLogEntry) -> str:
        excerpt = f" | output={_truncate(run.output_excerpt, 120)}" if run.output_excerpt else ""
        error = f" | error={_truncate(run.error, 120)}" if run.error else ""
        return (
            f"- {run.started_at.isoformat()} | status={run.status} | delivery={run.delivery_status or 'unknown'} | attempts={run.attempts}"
            f"{excerpt}{error}"
        )

    def _merge_with_pending(self, action: str, raw_kwargs: dict[str, Any]) -> dict[str, Any]:
        payload = {
            key: value
            for key, value in raw_kwargs.items()
            if key in self.parameters["properties"] and value is not None and key not in {"action", "confirm"}
        }
        pending = self._load_pending_draft(action)
        if pending:
            merged = dict(pending)
            merged.update(payload)
            return merged
        return payload

    def _load_pending_draft(self, action: str) -> dict[str, Any] | None:
        session = self._current_session()
        if session is None:
            return None
        metadata = session.metadata or {}
        pending = metadata.get(_PENDING_DRAFT_KEY)
        if not isinstance(pending, dict):
            return None
        if pending.get("version") != _CURRENT_DRAFT_VERSION:
            return None
        if pending.get("action") != action:
            return None
        payload = pending.get("payload")
        return dict(payload) if isinstance(payload, dict) else None

    def _store_pending_draft(self, action: str, payload: dict[str, Any]) -> None:
        session = self._current_session()
        if session is None:
            return
        metadata = dict(session.metadata or {})
        metadata[_PENDING_DRAFT_KEY] = {
            "version": _CURRENT_DRAFT_VERSION,
            "action": action,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "payload": self._serialize_payload(payload),
        }
        session.metadata = metadata
        self._save_session(session)

    def _clear_pending_draft(self) -> None:
        session = self._current_session()
        if session is None:
            return
        metadata = dict(session.metadata or {})
        if _PENDING_DRAFT_KEY in metadata:
            metadata.pop(_PENDING_DRAFT_KEY, None)
            session.metadata = metadata
            self._save_session(session)

    def _remember_job_id(self, job_id: str) -> None:
        session = self._current_session()
        if session is None:
            return
        metadata = dict(session.metadata or {})
        metadata[_LAST_JOB_KEY] = job_id
        session.metadata = metadata
        self._save_session(session)

    def _resolve_job_id(self, raw_job_id: Any) -> str | None:
        explicit = _non_empty(raw_job_id)
        if explicit:
            return explicit
        session = self._current_session()
        if session is None:
            return None
        return _non_empty((session.metadata or {}).get(_LAST_JOB_KEY))

    def _job_matches_current_context(self, job: CronJob) -> bool:
        binding = self._current_binding()
        current_scope = self._current_conversation_scope()
        current_session_key = self._current_session_key()

        if job.conversation_scope is not None and current_scope is not None:
            return self._conversation_scope_matches(job.conversation_scope, current_scope)

        if job.conversation_scope is not None and current_session_key:
            return job.conversation_scope.session_key == current_session_key

        if binding is None:
            if (
                current_session_key
                and job.target_mode in {"session", "current"}
                and job.session_key == current_session_key
            ):
                return True
            return True

        if job.delivery is not None:
            return self._delivery_matches(job.delivery, binding)

        if job.target_mode in {"session", "current"} and current_session_key:
            return job.session_key == current_session_key

        return False

    def _conversation_scope_matches(
        self,
        job_scope: CronConversationScope,
        current_scope: CronConversationScope,
    ) -> bool:
        if job_scope.channel != current_scope.channel:
            return False
        if job_scope.account_id != current_scope.account_id:
            return False
        if job_scope.conversation_id != current_scope.conversation_id:
            return False
        if job_scope.thread_id or current_scope.thread_id:
            return job_scope.thread_id == current_scope.thread_id
        return True

    def _delivery_matches(self, delivery: CronDeliveryTarget, binding: DeliveryBinding) -> bool:
        delivery_channel = delivery.channel
        if ":" not in delivery_channel and delivery.account_id:
            delivery_channel = f"{delivery_channel}:{delivery.account_id}"
        if delivery_channel != binding.channel:
            return False
        for key, value in binding.target.items():
            if delivery.target.get(key) != value:
                return False
        return True

    def _build_conversation_scope(
        self,
        *,
        delivery: CronDeliveryTarget | None,
        session_key: str | None,
    ) -> CronConversationScope | None:
        current_scope = self._current_conversation_scope()
        if current_scope is not None:
            return current_scope
        if delivery is not None:
            scope = conversation_scope_from_parts(
                channel=delivery.channel,
                account_id=delivery.account_id,
                target=delivery.target,
                session_key=delivery.session_key or session_key,
            )
            if scope is not None:
                return scope
        return conversation_scope_from_session_key(session_key)

    def _current_conversation_scope(self) -> CronConversationScope | None:
        binding = self._current_binding()
        current_session_key = self._current_session_key()
        if binding is not None:
            scope = conversation_scope_from_parts(
                channel=binding.channel,
                account_id=binding.account_id,
                target=binding.target,
                session_key=binding.session_key or current_session_key,
            )
            if scope is not None:
                return scope
            return conversation_scope_from_binding(
                DeliveryBinding(
                    channel=binding.channel,
                    account_id=binding.account_id,
                    session_key=binding.session_key or current_session_key,
                    target=dict(binding.target),
                )
            )
        return conversation_scope_from_session_key(current_session_key)

    def _current_binding(self) -> DeliveryBinding | None:
        session = self._current_session()
        if session is not None:
            raw = (session.metadata or {}).get("delivery_binding")
            if isinstance(raw, dict):
                binding = DeliveryBinding.from_metadata(raw)
                if binding is not None:
                    return binding
        return binding_from_session_key(self._current_session_key())

    def _current_session(self) -> Session | None:
        if self._agent_loop is None:
            return None
        try:
            return self._agent_loop.sessions.get_or_create(self._agent_loop.session_key)
        except Exception as exc:
            logger.debug(f"Could not resolve current session for cron tool: {exc}")
            return None

    def _save_session(self, session: Session) -> None:
        if self._agent_loop is None:
            return
        try:
            self._agent_loop.sessions.save(session)
        except Exception as exc:
            logger.warning(f"Failed to persist cron tool session metadata: {exc}")

    def _current_session_key(self) -> str | None:
        if self._agent_loop is None:
            return None
        return _non_empty(self._agent_loop.session_key)

    def _default_timezone(self) -> str:
        try:
            from spoon_bot.channels.config import load_cron_config

            config_path = getattr(self._agent_loop, "_config_path", None)
            return load_cron_config(config_path).timezone
        except Exception:
            return "UTC"

    def _get_cron_service(self) -> CronService | None:
        if self._cron_service is not None:
            return self._cron_service
        if self._agent_loop is not None:
            attached = getattr(self._agent_loop, "_cron_service", None)
            if attached is not None:
                return attached
        try:
            from spoon_bot.gateway.app import get_cron_service

            return get_cron_service()
        except Exception:
            return None

    def _service_unavailable_message(self) -> str:
        return (
            "Error: cron service is not available in this runtime. Start spoon-bot with "
            "cron.enabled=true (for example via the http-gateway service mode) before using the cron tool."
        )

    def _serialize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in payload.items():
            if value is None:
                continue
            if hasattr(value, "model_dump"):
                result[key] = value.model_dump(mode="json")
            else:
                result[key] = value
        return result

    def _is_cron_run_session(self) -> bool:
        session_key = self._current_session_key() or ""
        return session_key.startswith("cron_")
