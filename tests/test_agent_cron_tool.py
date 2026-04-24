from __future__ import annotations

from datetime import timedelta

import pytest

from spoon_bot.agent.loop import AgentLoop
from spoon_bot.agent.tools.cron import CronTool
from spoon_bot.cron.models import (
    CronExecutionResult,
    CronJob,
    CronJobCreate,
    CronJobPatch,
    CronRunLogEntry,
    CronServiceStatus,
    EverySchedule,
    utc_now,
)
from spoon_bot.session.manager import SessionManager


class DummyAgentLoop:
    def __init__(self, tmp_path, session_key: str = "telegram_spoon_bot_123") -> None:
        self.session_key = session_key
        self._config_path = None
        self._cron_service = None
        self.sessions = SessionManager(tmp_path)


class FakeCronService:
    def __init__(self) -> None:
        self.jobs: dict[str, CronJob] = {}
        self.runs_by_job: dict[str, list[CronRunLogEntry]] = {}

    async def status(self) -> CronServiceStatus:
        return CronServiceStatus(
            running=True,
            total_jobs=len(self.jobs),
            enabled_jobs=sum(1 for job in self.jobs.values() if job.enabled),
            active_runs=0,
            next_run_at=min(
                (job.state.next_run_at for job in self.jobs.values() if job.state.next_run_at is not None),
                default=None,
            ),
            store_path="memory",
        )

    async def list_jobs(self) -> list[CronJob]:
        return list(self.jobs.values())

    async def get_job(self, job_id: str) -> CronJob:
        if job_id not in self.jobs:
            raise KeyError(job_id)
        return self.jobs[job_id]

    async def create_job(self, payload: CronJobCreate) -> CronJob:
        job = CronJob(**payload.model_dump(mode="python"))
        job.state.next_run_at = utc_now() + timedelta(minutes=5)
        self.jobs[job.id] = job
        return job

    async def update_job(self, job_id: str, patch: CronJobPatch) -> CronJob:
        current = await self.get_job(job_id)
        merged = current.model_dump(mode="python")
        merged.update(patch.model_dump(mode="python", exclude_none=True))
        merged["id"] = current.id
        merged["created_at"] = current.created_at
        merged["state"] = current.state
        updated = CronJob(**merged)
        self.jobs[job_id] = updated
        return updated

    async def delete_job(self, job_id: str) -> bool:
        return self.jobs.pop(job_id, None) is not None

    async def run_now(self, job_id: str) -> CronExecutionResult:
        job = await self.get_job(job_id)
        run = CronRunLogEntry(
            job_id=job_id,
            status="success",
            session_key=job.session_key or f"cron_{job_id}",
            delivery_status="sent",
            output_excerpt="CRON_OK",
        )
        self.runs_by_job.setdefault(job_id, []).append(run)
        return CronExecutionResult(
            status="success",
            session_key=run.session_key,
            output="CRON_OK",
            delivered=True,
            delivery_status="sent",
        )

    async def get_runs(self, job_id: str, limit: int = 20) -> list[CronRunLogEntry]:
        return list(reversed(self.runs_by_job.get(job_id, [])[-limit:]))


def _bind_current_chat(agent: DummyAgentLoop, *, channel: str = "telegram:spoon_bot", target_key: str = "chat_id", target_value: str = "123") -> None:
    session = agent.sessions.get_or_create(agent.session_key)
    session.metadata = {
        "delivery_binding": {
            "channel": channel,
            "account_id": channel.split(":", 1)[1],
            "target": {target_key: target_value},
        }
    }
    agent.sessions.save(session)


@pytest.mark.asyncio
async def test_cron_tool_create_preview_and_confirm_uses_current_chat_binding(tmp_path):
    agent = DummyAgentLoop(tmp_path)
    _bind_current_chat(agent)
    service = FakeCronService()
    tool = CronTool()
    tool.set_agent_loop(agent)
    tool.set_cron_service(service)

    preview = await tool.execute(
        action="create",
        message="Drink water",
        schedule_kind="every",
        every_seconds=3600,
    )
    assert "draft prepared" in preview.lower()
    assert service.jobs == {}

    created = await tool.execute(action="create", confirm=True)
    assert "Created scheduled task successfully" in created
    assert len(service.jobs) == 1

    job = next(iter(service.jobs.values()))
    assert job.prompt == "Reply with exactly: Drink water"
    assert job.target_mode == "isolated"
    assert job.delivery is not None
    assert job.delivery.channel == "telegram:spoon_bot"
    assert job.delivery.account_id == "spoon_bot"
    assert job.delivery.target == {"chat_id": "123"}
    assert job.conversation_scope is not None
    assert job.conversation_scope.channel == "telegram"
    assert job.conversation_scope.account_id == "spoon_bot"
    assert job.conversation_scope.conversation_id == "123"


@pytest.mark.asyncio
async def test_cron_tool_live_info_preview_warns_when_search_provider_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    agent = DummyAgentLoop(tmp_path)
    _bind_current_chat(agent)
    service = FakeCronService()
    tool = CronTool()
    tool.set_agent_loop(agent)
    tool.set_cron_service(service)

    preview = await tool.execute(
        action="create",
        prompt="Every 5 minutes, fetch the latest news and summarize it for this chat.",
        schedule_kind="every",
        every_seconds=300,
    )
    assert "Capability warning:" in preview
    assert "No web_search provider is configured" in preview

    created = await tool.execute(action="create", confirm=True)
    assert "Created scheduled task successfully" in created

    job = next(iter(service.jobs.values()))
    assert job.target_mode == "isolated"
    assert job.allowed_tools == ["web_search", "web_fetch", "read_file", "list_dir"]


@pytest.mark.asyncio
async def test_cron_tool_reports_missing_fields_and_saves_partial_draft(tmp_path):
    agent = DummyAgentLoop(tmp_path)
    _bind_current_chat(agent)
    tool = CronTool()
    tool.set_agent_loop(agent)

    result = await tool.execute(action="create", message="Stand up")
    assert "schedule" in result.lower()

    session = agent.sessions.get_or_create(agent.session_key)
    pending = session.metadata.get("cron_pending_draft")
    assert isinstance(pending, dict)
    assert pending["action"] == "create"


@pytest.mark.asyncio
async def test_cron_tool_list_is_scoped_to_current_chat_binding(tmp_path):
    agent = DummyAgentLoop(tmp_path)
    _bind_current_chat(agent)
    service = FakeCronService()
    tool = CronTool()
    tool.set_agent_loop(agent)
    tool.set_cron_service(service)

    current_job = CronJob(
        name="Current chat reminder",
        prompt="Reply with exactly: hi",
        schedule=EverySchedule(seconds=60),
        target_mode="isolated",
        delivery={
            "channel": "telegram",
            "account_id": "spoon_bot",
            "target": {"chat_id": "123"},
            "session_key": agent.session_key,
        },
    )
    other_job = CronJob(
        name="Other chat reminder",
        prompt="Reply with exactly: hi",
        schedule=EverySchedule(seconds=60),
        target_mode="isolated",
        delivery={
            "channel": "telegram",
            "account_id": "spoon_bot",
            "target": {"chat_id": "999"},
            "session_key": "telegram_spoon_bot_999",
        },
    )
    service.jobs[current_job.id] = current_job
    service.jobs[other_job.id] = other_job

    result = await tool.execute(action="list")
    assert current_job.id in result
    assert other_job.id not in result


@pytest.mark.asyncio
async def test_cron_tool_scope_stays_accessible_from_creator_chat_even_if_delivery_changes(tmp_path):
    agent = DummyAgentLoop(tmp_path)
    _bind_current_chat(agent)
    service = FakeCronService()
    tool = CronTool()
    tool.set_agent_loop(agent)
    tool.set_cron_service(service)

    scoped_job = CronJob(
        name="Creator-owned job",
        prompt="Summarize the news",
        schedule=EverySchedule(seconds=3600),
        target_mode="isolated",
        delivery={
            "channel": "telegram",
            "account_id": "spoon_bot",
            "target": {"chat_id": "999"},
            "session_key": "telegram_spoon_bot_999",
        },
        conversation_scope={
            "channel": "telegram",
            "account_id": "spoon_bot",
            "conversation_id": "123",
            "session_key": agent.session_key,
        },
    )
    other_scoped_job = CronJob(
        name="Other chat owned job",
        prompt="Other chat",
        schedule=EverySchedule(seconds=3600),
        target_mode="isolated",
        delivery={
            "channel": "telegram",
            "account_id": "spoon_bot",
            "target": {"chat_id": "123"},
            "session_key": agent.session_key,
        },
        conversation_scope={
            "channel": "telegram",
            "account_id": "spoon_bot",
            "conversation_id": "999",
            "session_key": "telegram_spoon_bot_999",
        },
    )
    service.jobs[scoped_job.id] = scoped_job
    service.jobs[other_scoped_job.id] = other_scoped_job

    result = await tool.execute(action="list")

    assert scoped_job.id in result
    assert other_scoped_job.id not in result


@pytest.mark.asyncio
async def test_cron_tool_delete_uses_last_job_after_confirmation(tmp_path):
    agent = DummyAgentLoop(tmp_path)
    _bind_current_chat(agent)
    service = FakeCronService()
    tool = CronTool()
    tool.set_agent_loop(agent)
    tool.set_cron_service(service)

    await tool.execute(
        action="create",
        message="Pay rent",
        schedule_kind="every",
        every_seconds=86400,
    )
    await tool.execute(action="create", confirm=True)
    job_id = next(iter(service.jobs.keys()))

    preview = await tool.execute(action="delete")
    assert job_id in preview
    assert len(service.jobs) == 1

    deleted = await tool.execute(action="delete", confirm=True)
    assert "Deleted scheduled task" in deleted
    assert service.jobs == {}


@pytest.mark.asyncio
async def test_cron_tool_blocks_mutations_inside_cron_sessions(tmp_path):
    agent = DummyAgentLoop(tmp_path, session_key="cron_123")
    tool = CronTool()
    tool.set_agent_loop(agent)

    result = await tool.execute(
        action="create",
        message="Recursive reminder",
        schedule_kind="every",
        every_seconds=60,
    )
    assert "blocked" in result.lower()


@pytest.mark.asyncio
async def test_cron_tool_supports_complex_prompt_current_mode_and_silent_delivery(tmp_path):
    agent = DummyAgentLoop(tmp_path)
    _bind_current_chat(agent)
    service = FakeCronService()
    tool = CronTool()
    tool.set_agent_loop(agent)
    tool.set_cron_service(service)

    preview = await tool.execute(
        action="create",
        prompt="Every hour, research the day's updates and send me only important changes.",
        schedule_kind="every",
        every_seconds=3600,
        target_mode="current",
        delivery_mode="none",
        max_attempts=3,
        backoff_seconds=30,
    )
    assert "draft prepared" in preview.lower()
    assert "Delivery mode: none" in preview
    assert "Max attempts: 3" in preview

    created = await tool.execute(action="create", confirm=True)
    assert "Created scheduled task successfully" in created

    job = next(iter(service.jobs.values()))
    assert job.prompt == "Every hour, research the day's updates and send me only important changes."
    assert job.target_mode == "current"
    assert job.session_key == agent.session_key
    assert job.delivery_mode == "none"
    assert job.delivery is None
    assert "Target mode note:" in preview
    assert job.max_attempts == 3
    assert job.backoff_seconds == 30


def test_agent_registers_cron_tool(tmp_path):
    loop = AgentLoop.__new__(AgentLoop)
    loop.workspace = tmp_path
    loop.shell_timeout = 30
    loop.shell_max_timeout = 30
    loop.max_output = 8000
    loop.memory = type("M", (), {
        "get_memory_context": lambda self: "",
        "store": lambda self, **kw: None,
        "search": lambda self, **kw: [],
        "get_all": lambda self, **kw: [],
    })()
    loop.yolo_mode = False
    loop._cron_service = None
    loop.session_key = "telegram_spoon_bot_123"
    loop.sessions = SessionManager(tmp_path)
    loop._session = loop.sessions.get_or_create(loop.session_key)

    class _FakeTM:
        def __init__(self):
            self._tools: dict[str, object] = {}

        def register(self, tool):
            self._tools[tool.name] = tool

        def list_tools(self):
            return sorted(self._tools.keys())

        def set_tool_filter(self, **kw):
            pass

        def get_inactive_tools(self):
            return {}

        def get_active_tools(self):
            return self._tools

        def get(self, name):
            return self._tools.get(name)

        def __len__(self):
            return len(self._tools)

    loop.tools = _FakeTM()
    loop.add_tool = lambda name: False
    loop._register_native_tools()
    assert "cron" in loop.tools.list_tools()
