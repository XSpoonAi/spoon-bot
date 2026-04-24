from __future__ import annotations

import signal
from datetime import datetime, timezone

import pytest
from typer.testing import CliRunner

from spoon_bot.cli import _run_gateway, app
from spoon_bot.config import CronConfig
from spoon_bot.cron.models import CronJob, CronRunLogEntry, EverySchedule

runner = CliRunner()


class FakeCronClient:
    async def list_jobs(self):
        return [
            CronJob(
                id="cron_123",
                name="heartbeat",
                prompt="say hi",
                schedule=EverySchedule(seconds=60),
                target_mode="isolated",
            )
        ]

    async def list_runs(self, job_id: str, limit: int = 20):
        assert job_id == "cron_123"
        assert limit == 5
        return [
            CronRunLogEntry(
                job_id=job_id,
                status="success",
                session_key="cron_123",
                started_at=datetime(2026, 3, 28, 9, 0, tzinfo=timezone.utc),
                ended_at=datetime(2026, 3, 28, 9, 1, tzinfo=timezone.utc),
                delivery_status="sent",
                output_excerpt="done",
            )
        ]


def test_cron_list_command_outputs_jobs(monkeypatch):
    monkeypatch.setattr("spoon_bot.cli._create_cron_gateway_client", lambda *args, **kwargs: FakeCronClient())

    result = runner.invoke(app, ["cron", "list"])

    assert result.exit_code == 0
    assert "heartbeat" in result.output
    assert "cron_123" in result.output
    assert "every 60s" in result.output


def test_cron_create_requires_exactly_one_schedule(monkeypatch):
    monkeypatch.setattr("spoon_bot.cli._create_cron_gateway_client", lambda *args, **kwargs: FakeCronClient())
    monkeypatch.setattr("spoon_bot.cli._load_cli_cron_config", lambda config: CronConfig())

    result = runner.invoke(
        app,
        [
            "cron",
            "create",
            "--name",
            "job",
            "--prompt",
            "do work",
        ],
    )

    assert result.exit_code != 0
    assert "Invalid value:" in result.output
    assert "Exactly one of --at, --every-seconds, or --cron must be" in result.output
    assert "provided" in result.output


def test_cron_runs_command_outputs_recent_runs(monkeypatch):
    monkeypatch.setattr("spoon_bot.cli._create_cron_gateway_client", lambda *args, **kwargs: FakeCronClient())

    result = runner.invoke(app, ["cron", "runs", "cron_123", "--limit", "5"])

    assert result.exit_code == 0
    assert "cron_123" in result.output
    assert "sent" in result.output
    assert "done" in result.output


@pytest.mark.asyncio
async def test_gateway_keeps_running_without_channels_and_starts_cron(monkeypatch, tmp_path):
    class FakeAgent:
        provider = "openai"
        model = "gpt-5.2"
        _agent = None

        async def cleanup(self):
            return None

    class FakeManager:
        def __init__(self, *args, **kwargs):
            self._running = False

        def set_agent(self, agent, **kwargs):
            self.agent = agent

        async def load_from_config(self, config):
            return None

        async def start_all(self):
            self._running = False

        async def health_check_all(self):
            return {
                "running_channels": 0,
                "total_channels": 0,
                "channels": {},
            }

        async def stop(self):
            return None

        @property
        def channel_names(self):
            return []

        @property
        def is_running(self):
            return self._running

    class FakeCronService:
        def __init__(self):
            self.started = False
            self.stopped = False

        async def start(self):
            self.started = True

        async def stop(self):
            self.stopped = True

    fake_cron_service = FakeCronService()

    async def fake_create_agent(**kwargs):
        return FakeAgent()

    monkeypatch.setattr("spoon_bot.cli._configure_logging", lambda workspace: None)
    monkeypatch.setattr("spoon_bot.agent.loop.create_agent", fake_create_agent)
    monkeypatch.setattr("spoon_bot.channels.manager.ChannelManager", FakeManager)
    monkeypatch.setattr(
        "spoon_bot.channels.config.load_agent_config",
        lambda config: {
            "provider": "openai",
            "model": "gpt-5.2",
            "workspace": str(tmp_path),
        },
    )
    monkeypatch.setattr(
        "spoon_bot.channels.config.load_cron_config",
        lambda config: CronConfig(enabled=True),
    )
    monkeypatch.setattr(
        "spoon_bot.cron.service.create_cron_service",
        lambda **kwargs: fake_cron_service,
    )
    monkeypatch.setattr(signal, "getsignal", lambda *_args, **_kwargs: signal.SIG_DFL)
    monkeypatch.setattr(signal, "signal", lambda *_args, **_kwargs: None)

    await _run_gateway(
        config=None,
        channels=None,
        cli_enabled=None,
        model=None,
        provider=None,
        api_key=None,
        base_url=None,
        tool_profile=None,
        workspace=tmp_path,
    )

    assert fake_cron_service.started is True
    assert fake_cron_service.stopped is True
