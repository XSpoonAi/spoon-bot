"""Tests for tool execution timeout: default 600s, per-command override, background promotion."""
from __future__ import annotations

import asyncio
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from spoon_bot.agent.tools.shell import ShellTool, SafeShellTool, _BackgroundShellJob
from spoon_bot.config import AgentLoopConfig, validate_agent_loop_params


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
