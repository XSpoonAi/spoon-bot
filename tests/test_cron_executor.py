from __future__ import annotations

import pytest

from spoon_bot.cron.executor import CronExecutor
from spoon_bot.cron.models import CronJob, EverySchedule
from spoon_bot.runtime.execution import ExecutionCoordinator


class FakeSourceAgent:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def build_creation_kwargs(self, **overrides):
        self.calls.append(dict(overrides))
        return {
            "enabled_tools": {"cron"},
            "tool_profile": "core",
            **overrides,
        }


class FakeSessions:
    def get(self, session_key: str):
        return None


class FakeToolRegistry:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def set_tool_filter(self, **kwargs) -> None:
        self.calls.append(dict(kwargs))


class FakeRunner:
    def __init__(self, output: str = "ok") -> None:
        self.output = output
        self.sessions = FakeSessions()
        self.tools = FakeToolRegistry()
        self.process_calls: list[tuple[str, str]] = []
        self.cleared: list[str] = []

    async def process(self, prompt: str, session_key: str):
        self.process_calls.append((prompt, session_key))
        return self.output

    def clear_session_history(self, session_key: str) -> None:
        self.cleared.append(session_key)

    async def cleanup(self) -> None:
        return None


class FakeDeliveryService:
    def __init__(self) -> None:
        self.resolve_calls = 0
        self.deliver_calls = 0

    def resolve_binding(self, **kwargs):
        self.resolve_calls += 1
        return None

    async def deliver(self, content, binding):
        self.deliver_calls += 1


@pytest.mark.asyncio
async def test_cron_executor_uses_automation_profile_for_runner(monkeypatch):
    source = FakeSourceAgent()
    delivery = FakeDeliveryService()
    runner = FakeRunner()
    captured: dict[str, object] = {}

    async def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return runner

    monkeypatch.setattr("spoon_bot.cron.executor.create_agent", fake_create_agent)

    executor = CronExecutor(source, ExecutionCoordinator(), delivery)
    await executor._ensure_runner()

    assert captured["tool_profile"] == "automation"
    assert captured["enabled_tools"] is None


@pytest.mark.asyncio
async def test_cron_executor_skips_delivery_for_none_and_uses_extended_target_modes(monkeypatch):
    source = FakeSourceAgent()
    delivery = FakeDeliveryService()
    runner = FakeRunner(output="weather digest")

    async def fake_create_agent(**kwargs):
        return runner

    monkeypatch.setattr("spoon_bot.cron.executor.create_agent", fake_create_agent)

    executor = CronExecutor(source, ExecutionCoordinator(), delivery)

    main_job = CronJob(
        name="Hourly digest",
        prompt="Check today's weather and summarize it.",
        schedule=EverySchedule(seconds=3600),
        target_mode="main",
        delivery_mode="none",
    )
    main_result = await executor.execute(main_job)

    current_job = CronJob(
        name="Current thread follow-up",
        prompt="Continue this project summary.",
        schedule=EverySchedule(seconds=3600),
        target_mode="current",
        session_key="telegram_bot_123",
        delivery_mode="none",
    )
    current_result = await executor.execute(current_job)

    assert main_result.status == "success"
    assert main_result.session_key == "default"
    assert main_result.delivery_status == "skipped:none"
    assert current_result.session_key == "telegram_bot_123"
    assert delivery.resolve_calls == 0
    assert delivery.deliver_calls == 0
    assert runner.process_calls == [
        ("Check today's weather and summarize it.", "default"),
        ("Continue this project summary.", "telegram_bot_123"),
    ]
    assert runner.tools.calls[-1] == {"tool_profile": "automation"}


@pytest.mark.asyncio
async def test_cron_executor_applies_job_allowed_tools(monkeypatch):
    source = FakeSourceAgent()
    delivery = FakeDeliveryService()
    runner = FakeRunner(output="digest")

    async def fake_create_agent(**kwargs):
        return runner

    monkeypatch.setattr("spoon_bot.cron.executor.create_agent", fake_create_agent)

    executor = CronExecutor(source, ExecutionCoordinator(), delivery)
    job = CronJob(
        name="News digest",
        prompt="Fetch the latest news and summarize it.",
        schedule=EverySchedule(seconds=300),
        target_mode="isolated",
        delivery_mode="none",
        allowed_tools=["web_search", "web_fetch", "read_file"],
    )

    result = await executor.execute(job)

    assert result.status == "success"
    assert runner.tools.calls[0] == {"enabled_tools": {"web_search", "web_fetch", "read_file"}}
    assert runner.tools.calls[-1] == {"tool_profile": "automation"}
