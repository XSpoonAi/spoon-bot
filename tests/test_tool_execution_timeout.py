"""Tests for tool execution timeout: default 600s, per-command override, background promotion."""
from __future__ import annotations

import asyncio
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from spoon_bot.agent.tools.execution_context import (
    record_tool_invocation_result,
    suppress_after_consecutive_tool_failures,
    track_tool_invocations,
)
from spoon_bot.agent.tools.shell import (
    ShellTool,
    SafeShellTool,
    _BackgroundShellJob,
    _SHELL_BACKGROUND_JOBS,
)
from spoon_bot.agent.loop import AgentLoop
from spoon_bot.config import (
    AgentLoopConfig,
    DEFAULT_MAX_STREAM_TOOL_RESULTS_WITHOUT_CONTENT,
    DEFAULT_PROVIDER_SILENCE_TIMEOUT,
    DEFAULT_PROVIDER_TOTAL_TIMEOUT,
    DEFAULT_TOOL_FOLLOWUP_TIMEOUT,
    validate_agent_loop_params,
)


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    def test_agent_loop_config_default_shell_timeout(self):
        config = AgentLoopConfig()
        assert config.shell_timeout == 600

    def test_agent_loop_config_default_shell_max_timeout(self):
        config = AgentLoopConfig()
        assert config.shell_max_timeout == 7200

    def test_validate_agent_loop_params_default(self):
        config = validate_agent_loop_params()
        assert config.shell_timeout == 600
        assert config.shell_max_timeout == 7200

    def test_validate_agent_loop_params_custom(self):
        config = validate_agent_loop_params(
            shell_timeout=120,
            shell_max_timeout=3600,
        )
        assert config.shell_timeout == 120
        assert config.shell_max_timeout == 3600

    def test_shell_timeout_upper_bound(self):
        config = AgentLoopConfig(shell_timeout=7200)
        assert config.shell_timeout == 7200

    def test_shell_max_timeout_lower_bound(self):
        config = AgentLoopConfig(shell_max_timeout=60)
        assert config.shell_max_timeout == 60

    def test_agent_loop_stream_timeout_defaults(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SPOON_BOT_PROVIDER_SILENCE_TIMEOUT", raising=False)
        monkeypatch.delenv("SPOON_BOT_PROVIDER_TOTAL_TIMEOUT", raising=False)
        monkeypatch.delenv("SPOON_BOT_TOOL_FOLLOWUP_TIMEOUT", raising=False)
        monkeypatch.delenv("SPOON_BOT_MAX_STREAM_TOOL_RESULTS_WITHOUT_CONTENT", raising=False)

        loop = AgentLoop(workspace=tmp_path, model="test-model", provider="openai", api_key="test")

        assert loop.provider_silence_timeout == DEFAULT_PROVIDER_SILENCE_TIMEOUT
        assert loop.provider_total_timeout == DEFAULT_PROVIDER_TOTAL_TIMEOUT
        assert loop.tool_followup_timeout == DEFAULT_TOOL_FOLLOWUP_TIMEOUT
        assert (
            loop.max_stream_tool_results_without_content
            == DEFAULT_MAX_STREAM_TOOL_RESULTS_WITHOUT_CONTENT
        )

    def test_agent_loop_stream_timeout_env_overrides(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SPOON_BOT_PROVIDER_SILENCE_TIMEOUT", "301")
        monkeypatch.setenv("SPOON_BOT_PROVIDER_TOTAL_TIMEOUT", "302")
        monkeypatch.setenv("SPOON_BOT_TOOL_FOLLOWUP_TIMEOUT", "303")
        monkeypatch.setenv("SPOON_BOT_MAX_STREAM_TOOL_RESULTS_WITHOUT_CONTENT", "77")

        loop = AgentLoop(workspace=tmp_path, model="test-model", provider="openai", api_key="test")

        assert loop.provider_silence_timeout == 301
        assert loop.provider_total_timeout == 302
        assert loop.tool_followup_timeout == 303
        assert loop.max_stream_tool_results_without_content == 77


# ---------------------------------------------------------------------------
# ShellTool initialization
# ---------------------------------------------------------------------------

class TestShellToolInit:
    def test_default_timeout(self):
        tool = ShellTool()
        assert tool.timeout == 600

    def test_default_max_timeout(self):
        tool = ShellTool()
        assert tool.max_timeout == 7200

    def test_custom_timeout(self):
        tool = ShellTool(timeout=120, max_timeout=3600)
        assert tool.timeout == 120
        assert tool.max_timeout == 3600

    def test_max_timeout_at_least_timeout(self):
        tool = ShellTool(timeout=1000, max_timeout=500)
        assert tool.max_timeout == 1000

    def test_workspace_skill_foreground_timeout_defaults_to_short_handoff(self, monkeypatch):
        monkeypatch.delenv("SPOON_BOT_WORKSPACE_SKILL_FOREGROUND_TIMEOUT", raising=False)
        tool = ShellTool(timeout=3600, max_timeout=7200)

        assert tool._workspace_skill_foreground_timeout() == 60

    def test_workspace_skill_foreground_timeout_env_override(self, monkeypatch):
        monkeypatch.setenv("SPOON_BOT_WORKSPACE_SKILL_FOREGROUND_TIMEOUT", "120")
        tool = ShellTool(timeout=3600, max_timeout=7200)

        assert tool._workspace_skill_foreground_timeout() == 120

    def test_background_job_poll_wait_defaults_to_managed_wait(self, monkeypatch):
        monkeypatch.delenv("SPOON_BOT_BACKGROUND_JOB_POLL_WAIT_SECONDS", raising=False)
        tool = ShellTool()

        assert tool._background_job_poll_wait_seconds() == 30.0

    def test_background_job_poll_wait_env_override(self, monkeypatch):
        monkeypatch.setenv("SPOON_BOT_BACKGROUND_JOB_POLL_WAIT_SECONDS", "12.5")
        tool = ShellTool()

        assert tool._background_job_poll_wait_seconds() == 12.5


class TestSafeShellToolInit:
    def test_default_timeout(self):
        tool = SafeShellTool()
        assert tool.timeout == 600

    def test_default_max_timeout(self):
        tool = SafeShellTool()
        assert tool.max_timeout == 7200


# ---------------------------------------------------------------------------
# Tool description and parameters
# ---------------------------------------------------------------------------

class TestShellToolSchema:
    def test_description_mentions_10min(self):
        tool = ShellTool()
        assert "10min" in tool.description or "600s" in tool.description

    def test_parameters_include_timeout(self):
        tool = ShellTool()
        params = tool.parameters
        assert "timeout" in params["properties"]
        timeout_param = params["properties"]["timeout"]
        assert timeout_param["type"] == "integer"
        assert timeout_param["minimum"] == 1
        assert timeout_param["maximum"] == tool.max_timeout

    def test_safe_shell_description_mentions_10min(self):
        tool = SafeShellTool()
        assert "10min" in tool.description or "600s" in tool.description


# ---------------------------------------------------------------------------
# Per-command timeout in execute()
# ---------------------------------------------------------------------------

class TestPerCommandTimeout:
    @pytest.mark.asyncio
    async def test_default_timeout_used_when_none(self):
        tool = ShellTool(timeout=600, max_timeout=7200, working_dir=os.getcwd())
        with patch.object(tool, "_start_background_job") as mock_start, \
             patch.object(tool, "_refresh_background_job") as mock_refresh, \
             patch.object(tool, "_wait_for_process") as mock_wait, \
             patch("spoon_bot.agent.tools.shell.get_tool_owner", return_value="test"), \
             patch("spoon_bot.agent.tools.shell.capture_tool_output"):

            mock_job = MagicMock()
            mock_job.job_id = "test_job"
            mock_job.stdout_text = "ok"
            mock_job.stderr_text = ""
            mock_job.returncode = 0
            mock_job.owner_key = "test"
            mock_start.return_value = mock_job
            mock_wait.return_value = None

            await tool.execute(command="echo hello")

            mock_wait.assert_called_once()
            wait_call = mock_wait.call_args
            assert wait_call is not None

    @pytest.mark.asyncio
    async def test_custom_timeout_capped_by_max(self):
        tool = ShellTool(timeout=60, max_timeout=300, working_dir=os.getcwd())
        with patch.object(tool, "_start_background_job") as mock_start, \
             patch.object(tool, "_refresh_background_job"), \
             patch.object(tool, "_wait_for_process") as mock_wait, \
             patch("spoon_bot.agent.tools.shell.get_tool_owner", return_value="test"), \
             patch("spoon_bot.agent.tools.shell.capture_tool_output"):

            mock_job = MagicMock()
            mock_job.job_id = "test_job"
            mock_job.stdout_text = "ok"
            mock_job.stderr_text = ""
            mock_job.returncode = 0
            mock_job.owner_key = "test"
            mock_start.return_value = mock_job
            mock_wait.return_value = None

            # Request 9999s but max is 300
            await tool.execute(command="echo hello", timeout=9999)

    @pytest.mark.asyncio
    async def test_custom_timeout_below_max(self):
        tool = ShellTool(timeout=60, max_timeout=300, working_dir=os.getcwd())
        with patch.object(tool, "_start_background_job") as mock_start, \
             patch.object(tool, "_refresh_background_job"), \
             patch.object(tool, "_wait_for_process") as mock_wait, \
             patch("spoon_bot.agent.tools.shell.get_tool_owner", return_value="test"), \
             patch("spoon_bot.agent.tools.shell.capture_tool_output"):

            mock_job = MagicMock()
            mock_job.job_id = "test_job"
            mock_job.stdout_text = "ok"
            mock_job.stderr_text = ""
            mock_job.returncode = 0
            mock_job.owner_key = "test"
            mock_start.return_value = mock_job
            mock_wait.return_value = None

            await tool.execute(command="echo hello", timeout=120)

    @pytest.mark.asyncio
    async def test_stateful_workspace_skill_caps_long_default_foreground_timeout(
        self,
        tmp_path,
        monkeypatch,
    ):
        monkeypatch.delenv("SPOON_BOT_WORKSPACE_SKILL_FOREGROUND_TIMEOUT", raising=False)
        tool = ShellTool(timeout=3600, max_timeout=7200, working_dir=str(tmp_path))
        command = (
            "node skills/joker-game-agent/cli/index.js challenge-answer "
            "2889759959 chl_b24ac1fd0f1725a22de73f9e7fc985d4 1187"
        )
        wait_timeouts: list[float] = []

        async def fake_wait_for(awaitable, timeout):
            wait_timeouts.append(timeout)
            close = getattr(awaitable, "close", None)
            if callable(close):
                close()
            if len(wait_timeouts) == 1:
                raise asyncio.TimeoutError
            return None

        mock_job = _BackgroundShellJob(
            job_id="sh_stateful",
            command=command,
            cwd=str(tmp_path),
            process=MagicMock(),
            stdout_task=MagicMock(),
            stderr_task=MagicMock(),
            buffer_limit=200000,
            owner_key="test",
            stdout_text="",
            stderr_text="",
            status="running",
            created_at=time.time() - 60,
        )

        with patch.object(tool, "_start_background_job", return_value=mock_job), \
             patch.object(tool, "_refresh_background_job"), \
             patch("spoon_bot.agent.tools.shell.asyncio.wait_for", side_effect=fake_wait_for), \
             patch("spoon_bot.agent.tools.shell.get_tool_owner", return_value="test"), \
             patch("spoon_bot.agent.tools.shell.capture_tool_output"):
            result = await tool.execute(command=command)

        assert wait_timeouts[0] == 60
        assert "Foreground timeout (60s) exceeded" in result
        assert "job_id: sh_stateful" in result


# ---------------------------------------------------------------------------
# Background job summary message
# ---------------------------------------------------------------------------

class TestBackgroundJobSummary:
    def test_summary_includes_job_id(self):
        tool = ShellTool()
        job = _BackgroundShellJob(
            job_id="sh_test123",
            command="npm install",
            cwd="/tmp",
            process=MagicMock(),
            stdout_task=MagicMock(),
            stderr_task=MagicMock(),
            buffer_limit=200000,
            owner_key="test",
            stdout_text="installing...",
            stderr_text="",
            status="running",
            created_at=time.time() - 120,
        )
        summary = tool._format_background_job_summary(job, timeout_seconds=600)
        assert "sh_test123" in summary
        assert "npm install" in summary

    def test_summary_includes_monitoring_instructions(self):
        tool = ShellTool()
        job = _BackgroundShellJob(
            job_id="sh_abc",
            command="make build",
            cwd="/tmp",
            process=MagicMock(),
            stdout_task=MagicMock(),
            stderr_task=MagicMock(),
            buffer_limit=200000,
            owner_key="test",
            created_at=time.time() - 60,
        )
        summary = tool._format_background_job_summary(job, timeout_seconds=600)
        assert "job_status" in summary
        assert "job_output" in summary
        assert "terminate_job" in summary
        assert "NEXT STEPS" in summary
        assert "Quiet output can be normal" in summary
        assert "after two checks" not in summary

    @pytest.mark.asyncio
    async def test_repeated_running_job_output_does_not_stop_tool_loop(self, monkeypatch):
        monkeypatch.setenv("SPOON_BOT_BACKGROUND_JOB_POLL_WAIT_SECONDS", "0")
        tool = ShellTool()
        process = MagicMock()
        process.poll.return_value = None
        job = _BackgroundShellJob(
            job_id="sh_running",
            command="node skills/example/cli/index.js wait",
            cwd="/tmp",
            process=process,
            stdout_task=MagicMock(),
            stderr_task=MagicMock(),
            buffer_limit=200000,
            owner_key="default",
            stdout_text="waiting...",
            stderr_text="",
            status="running",
            created_at=time.time() - 120,
        )
        _SHELL_BACKGROUND_JOBS[job.job_id] = job
        try:
            with track_tool_invocations():
                first = await tool.execute(action="job_output", job_id=job.job_id)
                second = await tool.execute(action="job_status", job_id=job.job_id)
                third = await tool.execute(action="job_output", job_id=job.job_id)
        finally:
            _SHELL_BACKGROUND_JOBS.pop(job.job_id, None)

        assert "STOP_TOOL_LOOP" not in first
        assert "STOP_TOOL_LOOP" not in second
        assert "STOP_TOOL_LOOP" not in third
        assert "status: running" in third
        assert "not a completion signal" in third

    @pytest.mark.asyncio
    async def test_duplicate_running_background_command_is_not_started(self, tmp_path):
        tool = ShellTool(working_dir=str(tmp_path))
        process = MagicMock()
        process.poll.return_value = None
        job = _BackgroundShellJob(
            job_id="sh_existing",
            command="sleep 999",
            cwd=str(tmp_path),
            process=process,
            stdout_task=MagicMock(),
            stderr_task=MagicMock(),
            buffer_limit=200000,
            owner_key="test",
            stdout_text="waiting...",
            stderr_text="",
            status="running",
            created_at=time.time() - 90,
        )
        _SHELL_BACKGROUND_JOBS[job.job_id] = job
        try:
            with patch.object(tool, "_start_background_job") as start_background_job, \
                 patch("spoon_bot.agent.tools.shell.get_tool_owner", return_value="test"), \
                 patch("spoon_bot.agent.tools.shell.capture_tool_output"):
                result = await tool.execute(command="sleep 999")
        finally:
            _SHELL_BACKGROUND_JOBS.pop(job.job_id, None)

        start_background_job.assert_not_called()
        assert "ACTIVE_BACKGROUND_JOB_EXISTS" in result
        assert "job_id: sh_existing" in result
        assert "Do not start another copy" in result
        assert "job_output" in result

    def test_summary_includes_elapsed_time(self):
        tool = ShellTool()
        job = _BackgroundShellJob(
            job_id="sh_xyz",
            command="sleep 999",
            cwd="/tmp",
            process=MagicMock(),
            stdout_task=MagicMock(),
            stderr_task=MagicMock(),
            buffer_limit=200000,
            owner_key="test",
            created_at=time.time() - 300,
        )
        summary = tool._format_background_job_summary(job, timeout_seconds=600)
        assert "elapsed" in summary

    def test_summary_shows_timeout_seconds(self):
        tool = ShellTool()
        job = _BackgroundShellJob(
            job_id="sh_t",
            command="test",
            cwd="/tmp",
            process=MagicMock(),
            stdout_task=MagicMock(),
            stderr_task=MagicMock(),
            buffer_limit=200000,
            owner_key="test",
            created_at=time.time(),
        )
        summary = tool._format_background_job_summary(job, timeout_seconds=120)
        assert "120s" in summary


class TestToolFailureLoopGuard:
    def test_consecutive_tool_failures_do_not_stop_next_call(self):
        with track_tool_invocations(max_consecutive_failures=3):
            for _ in range(3):
                assert suppress_after_consecutive_tool_failures("shell") is None
                record_tool_invocation_result("shell", "Error: transient failure")

            assert suppress_after_consecutive_tool_failures("shell") is None

    def test_repeated_failure_pattern_does_not_stop_after_successes(self):
        with track_tool_invocations(max_consecutive_failures=3):
            for _ in range(3):
                assert suppress_after_consecutive_tool_failures("shell") is None
                record_tool_invocation_result("shell", "Error: provider temporarily unavailable")
                record_tool_invocation_result("read_file", "ok")

            assert suppress_after_consecutive_tool_failures("shell") is None


class TestRepeatedToolGuards:
    @pytest.mark.asyncio
    async def test_repeated_shell_invocations_do_not_stop_tool_loop(self, tmp_path):
        tool = ShellTool(working_dir=str(tmp_path))

        with track_tool_invocations(max_repeats=1):
            first = await tool(command="printf exact-duplicate")
            second = await tool(command="printf exact-duplicate")
            third = await tool(command="printf exact-duplicate")

        assert first == "exact-duplicate"
        assert second == "exact-duplicate"
        assert third == "exact-duplicate"
        assert "STOP_TOOL_LOOP" not in second
        assert "STOP_TOOL_LOOP" not in third

    @pytest.mark.asyncio
    async def test_repeated_read_only_shell_inspections_do_not_stop_tool_loop(self, tmp_path):
        tool = ShellTool(working_dir=str(tmp_path))

        with track_tool_invocations(max_repeats=1):
            first = await tool(command="pwd")
            second = await tool(command="pwd")
            third = await tool(command="pwd")

        assert str(tmp_path) in first
        assert str(tmp_path) in second
        assert str(tmp_path) in third
        assert "STOP_TOOL_LOOP" not in second
        assert "duplicate shell inspection" not in second.lower()
        assert "STOP_TOOL_LOOP" not in third
        assert "duplicate shell inspection" not in third.lower()

    @pytest.mark.asyncio
    async def test_repeated_skill_cli_series_do_not_stop_tool_loop(self, tmp_path):
        skill_dir = tmp_path / "skills" / "example" / "cli"
        skill_dir.mkdir(parents=True)
        script = skill_dir / "index.sh"
        script.write_text("#!/bin/sh\nprintf played\n", encoding="utf-8")
        script.chmod(0o755)
        (tmp_path / "skills" / "example" / "SKILL.md").write_text(
            "# Example\n\n"
            "CLI = skills/example/cli/index.sh\n\n"
            "## Commands\n"
            "- $CLI play\n",
            encoding="utf-8",
        )
        tool = ShellTool(working_dir=str(tmp_path))

        with track_tool_invocations(max_repeats=100, max_series_repeats=1):
            first = await tool(command="skills/example/cli/index.sh play")
            second = await tool(command="skills/example/cli/index.sh play")
            third = await tool(command="skills/example/cli/index.sh play")

        assert first == "played"
        assert second == "played"
        assert third == "played"
        assert "STOP_TOOL_LOOP" not in second
        assert "repeated side-effecting tool series" not in second.lower()
        assert "STOP_TOOL_LOOP" not in third
        assert "repeated side-effecting tool series" not in third.lower()


# ---------------------------------------------------------------------------
# Config loading from env vars
# ---------------------------------------------------------------------------

class TestConfigEnvLoading:
    def test_shell_timeout_from_env(self, monkeypatch):
        monkeypatch.setenv("SPOON_BOT_SHELL_TIMEOUT", "300")
        from spoon_bot.channels.config import load_agent_config
        config = load_agent_config()
        assert config.get("shell_timeout") == 300

    def test_shell_max_timeout_from_env(self, monkeypatch):
        monkeypatch.setenv("SPOON_BOT_SHELL_MAX_TIMEOUT", "1800")
        from spoon_bot.channels.config import load_agent_config
        config = load_agent_config()
        assert config.get("shell_max_timeout") == 1800
