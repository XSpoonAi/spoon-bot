"""
Unit tests for spoon-bot tools.

These tests use mocks and do not require API keys to run.
Run with: pytest tests/test_tools.py -v
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure spoon_bot is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestShellTool:
    """Tests for the ShellTool."""

    @pytest.fixture
    def shell_tool(self):
        """Create a ShellTool instance for testing."""
        from spoon_bot.agent.tools.shell import ShellTool
        from spoon_bot.utils.rate_limit import RateLimitConfig

        return ShellTool(
            timeout=5,
            max_output=1000,
            rate_limit_config=RateLimitConfig.unlimited(),
        )

    @pytest.mark.asyncio
    async def test_simple_echo_command(self, shell_tool):
        """Test executing a simple echo command."""
        result = await shell_tool.execute("echo hello")
        assert "hello" in result.lower()

    @pytest.mark.asyncio
    async def test_sensitive_env_vars_not_exposed_to_shell(self, shell_tool, monkeypatch):
        """Shell subprocesses must not inherit secret-bearing env vars."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-secret")
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-secret")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "telegram-secret")
        monkeypatch.setenv("PRIVATE_KEY", "0x" + "11" * 32)
        monkeypatch.setenv("SPOON_BOT_DEFAULT_PROVIDER", "openrouter")

        from spoon_bot.agent.tools.shell import _scrub_env

        result_env = _scrub_env(dict(os.environ))
        result = "\n".join(
            f"{key}={result_env.get(key, '')}"
            for key in (
                "OPENROUTER_API_KEY",
                "TAVILY_API_KEY",
                "TELEGRAM_BOT_TOKEN",
                "PRIVATE_KEY",
                "SPOON_BOT_DEFAULT_PROVIDER",
            )
        )

        assert "OPENROUTER_API_KEY=" in result
        assert "TAVILY_API_KEY=" in result
        assert "TELEGRAM_BOT_TOKEN=" in result
        assert "PRIVATE_KEY=" in result
        assert "OPENROUTER_API_KEY=sk-or-v1-secret" not in result
        assert "TAVILY_API_KEY=tvly-secret" not in result
        assert "TELEGRAM_BOT_TOKEN=telegram-secret" not in result
        assert "PRIVATE_KEY=0x" not in result
        assert "SPOON_BOT_DEFAULT_PROVIDER=openrouter" in result

    def test_shell_tool_does_not_infer_side_effect_series_from_command_text(self, shell_tool):
        """Shell should not guess business side effects from command words."""
        assert not callable(getattr(shell_tool, "tool_invocation_series_key", None))

    def test_shell_detects_workspace_skill_commands_for_timeout_floor(self, tmp_path):
        """Workspace skill CLI commands should keep the default foreground budget."""
        from spoon_bot.agent.tools.shell import ShellTool

        workspace = tmp_path / "workspace"
        (workspace / "skills" / "demo-skill").mkdir(parents=True)
        shell_tool = ShellTool(working_dir=str(workspace))

        assert shell_tool._command_invokes_workspace_skill(
            "node skills/demo-skill/cli/index.js run",
            str(workspace),
        ) is True
        assert shell_tool._command_invokes_workspace_skill(
            "node scripts/build.js",
            str(workspace),
        ) is False
        assert shell_tool._command_invokes_workspace_skill(
            "cd skills/demo-skill/cli && node index.js run",
            str(workspace),
        ) is True

    def test_shell_skips_exact_dedup_for_workspace_skill_commands(self, tmp_path):
        """Skill-owned retry/idempotency should not be blocked by shell dedup."""
        from spoon_bot.agent.tools.shell import ShellTool

        workspace = tmp_path / "workspace"
        (workspace / "skills" / "demo-skill" / "cli").mkdir(parents=True)
        shell_tool = ShellTool(working_dir=str(workspace))

        assert shell_tool.tool_invocation_dedup_key(
            {
                "action": "execute",
                "command": "node skills/demo-skill/cli/index.js join A",
                "working_dir": str(workspace),
            }
        ) is None

    def test_shell_loads_workspace_env_for_skill_commands(self, tmp_path):
        """Installed skill CLIs should receive non-sensitive workspace env values."""
        from spoon_bot.agent.tools.shell import ShellTool

        workspace = tmp_path / "workspace"
        (workspace / "skills" / "demo-skill" / "cli").mkdir(parents=True)
        (workspace / ".env.local").write_text(
            "AGENT_ID=8004\nAGENT_PRIVATE_KEY=0x" + "11" * 32 + "\n",
            encoding="utf-8",
        )
        shell_tool = ShellTool(working_dir=str(workspace))

        command = "node skills/demo-skill/cli/index.js wallet"
        result = shell_tool._prepend_workspace_env_for_skill_command(
            command,
            str(workspace),
        )

        assert result.startswith("export AGENT_ID=8004; ")
        assert "AGENT_PRIVATE_KEY" not in result
        assert "0x" + "11" * 32 not in result
        assert result.endswith(command)

    def test_shell_formats_idempotent_conflict_as_state(self, shell_tool):
        """HTTP 409 already-state output should not look like a failed command."""
        result = shell_tool._build_output_result(
            "",
            'HTTP 409 POST /v1/faucet: {"error":"Address already claimed in this round"}',
            1,
        )

        assert "already satisfied" in result
        assert "Address already claimed in this round" in result
        assert "STDERR" not in result
        assert "Exit code" not in result

    def test_shell_normalizes_windows_python3_store_alias(self, shell_tool):
        """Windows Store python3 alias should fall back to installed python."""
        with patch("spoon_bot.agent.tools.shell.shutil.which") as which:
            which.side_effect = lambda name: {
                "python": r"C:\Python313\python.exe",
                "python3": r"C:\Users\me\AppData\Local\Microsoft\WindowsApps\python3.exe",
            }.get(name)

            result = shell_tool._normalize_windows_python_command("python3 -c \"print(1)\"")

        assert result == "python -c \"print(1)\""
        assert shell_tool._is_git_bash(r"C:\Program Files\Git\bin\bash.exe") is True

    def test_shell_rejects_chained_git_clone_into_workspace_skills(self, tmp_path):
        """Skill installs must go through the skill manager even when clone is chained."""
        from spoon_bot.agent.tools.shell import ShellTool

        workspace = tmp_path / "workspace"
        (workspace / "skills").mkdir(parents=True)
        shell_tool = ShellTool(working_dir=str(workspace))

        result = shell_tool._reject_workspace_skill_clone(
            "cd skills && git clone https://github.com/example-org/example-skill spot-skill",
            str(workspace),
        )

        assert result is not None
        assert "skill_marketplace(action='install_skill'" in result
        assert "workspace/skills" in result

    def test_shell_allows_plain_git_clone_outside_workspace_skills(self, tmp_path):
        """Ordinary repo clones outside the skills tree are not skill-installer bypasses."""
        from spoon_bot.agent.tools.shell import ShellTool

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        shell_tool = ShellTool(working_dir=str(workspace))

        result = shell_tool._reject_workspace_skill_clone(
            "git clone https://github.com/example-org/example-repo repos/example-repo",
            str(workspace),
        )

        assert result is None

    def test_shell_rejects_undocumented_skill_cli_flag(self, tmp_path):
        """Skill CLI commands must not invent flags from user wording."""
        from spoon_bot.agent.tools.shell import ShellTool

        workspace = tmp_path / "workspace"
        skill_dir = workspace / "skills" / "spot-agent-cypher"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "\n".join([
                "---",
                "name: spot-agent-cypher",
                "---",
                "CLI := node skills/spot-agent-cypher/cli/index.js",
                "",
                "## Commands",
                "$CLI faucet [-c <code>]",
                "$CLI register",
                "$CLI join [gameId] [spot]",
                "$CLI game snapshot <gameId>",
            ]),
            encoding="utf-8",
        )
        shell_tool = ShellTool(working_dir=str(workspace))

        result = shell_tool._reject_undocumented_skill_cli_arguments(
            (
                "cd /workspace && "
                "node skills/spot-agent-cypher/cli/index.js register --agent-id 8004 2>&1"
            ),
            str(workspace),
        )

        assert result is not None
        assert "unsupported option '--agent-id'" in result
        assert "skills/spot-agent-cypher/SKILL.md" in result
        assert "do not invent CLI flags" in result

    def test_shell_rejects_undocumented_skill_cli_flag_after_cd(self, tmp_path):
        """Skill CLI guards follow a cd into the skill entrypoint directory."""
        from spoon_bot.agent.tools.shell import ShellTool

        workspace = tmp_path / "workspace"
        skill_dir = workspace / "skills" / "spot-agent-cypher"
        (skill_dir / "cli").mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "\n".join([
                "---",
                "name: spot-agent-cypher",
                "---",
                "CLI := node skills/spot-agent-cypher/cli/index.js",
                "## Commands",
                "$CLI register",
                "$CLI join [gameId] [spot]",
            ]),
            encoding="utf-8",
        )
        shell_tool = ShellTool(working_dir=str(workspace))

        result = shell_tool._reject_undocumented_skill_cli_arguments(
            "cd skills/spot-agent-cypher/cli && node index.js register --agent-id 8004",
            str(workspace),
        )

        assert result is not None
        assert "unsupported option '--agent-id'" in result
        assert "Documented form: node skills/spot-agent-cypher/cli/index.js register" in result

    def test_shell_rejects_extra_skill_cli_positional_arg(self, tmp_path):
        """Values from the user request are allowed only when SKILL.md has placeholders."""
        from spoon_bot.agent.tools.shell import ShellTool

        workspace = tmp_path / "workspace"
        skill_dir = workspace / "skills" / "demo-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "\n".join([
                "---",
                "name: demo-skill",
                "---",
                "CLI := node skills/demo-skill/cli/index.js",
                "RUN $CLI register",
            ]),
            encoding="utf-8",
        )
        shell_tool = ShellTool(working_dir=str(workspace))

        result = shell_tool._reject_undocumented_skill_cli_arguments(
            "node skills/demo-skill/cli/index.js register 8004",
            str(workspace),
        )

        assert result is not None
        assert "too many positional arguments" in result
        assert "Documented form: node skills/demo-skill/cli/index.js register" in result

    def test_shell_allows_documented_skill_cli_args(self, tmp_path):
        """Documented flags, positionals, and nested subcommands remain allowed."""
        from spoon_bot.agent.tools.shell import ShellTool

        workspace = tmp_path / "workspace"
        skill_dir = workspace / "skills" / "spot-agent-cypher"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "\n".join([
                "---",
                "name: spot-agent-cypher",
                "---",
                "CLI := node skills/spot-agent-cypher/cli/index.js",
                "$CLI faucet [-c <code>]",
                "$CLI join [gameId] [spot]",
                "$CLI game snapshot <gameId>",
            ]),
            encoding="utf-8",
        )
        shell_tool = ShellTool(working_dir=str(workspace))

        assert shell_tool._reject_undocumented_skill_cli_arguments(
            "node skills/spot-agent-cypher/cli/index.js faucet -c 3KK57S",
            str(workspace),
        ) is None
        assert shell_tool._reject_undocumented_skill_cli_arguments(
            "node skills/spot-agent-cypher/cli/index.js join 123 A",
            str(workspace),
        ) is None
        assert shell_tool._reject_undocumented_skill_cli_arguments(
            "node skills/spot-agent-cypher/cli/index.js game snapshot 123",
            str(workspace),
        ) is None

    def test_shell_rejects_undocumented_skill_cli_entrypoint(self, tmp_path):
        """Skill CLI guards cover the documented executable path, not only args."""
        from spoon_bot.agent.tools.shell import ShellTool

        workspace = tmp_path / "workspace"
        skill_dir = workspace / "skills" / "spot-agent-cypher"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "\n".join([
                "---",
                "name: spot-agent-cypher",
                "---",
                "CLI := node skills/spot-agent-cypher/cli/index.js",
                "## Commands",
                "$CLI join [gameId] [spot]",
            ]),
            encoding="utf-8",
        )
        shell_tool = ShellTool(working_dir=str(workspace))

        result = shell_tool._reject_undocumented_skill_cli_arguments(
            "node skills/spot-agent-cypher/cli/dist/index.js join --help",
            str(workspace),
        )

        assert result is not None
        assert "entrypoint is not documented" in result
        assert "cli/index.js" in result
        assert "cli/dist/index.js" not in result

    def test_shell_skill_cli_templates_prefer_commands_section(self, tmp_path):
        """Procedural retry prose must not become fake skill CLI arguments."""
        from spoon_bot.agent.tools.shell import ShellTool

        workspace = tmp_path / "workspace"
        skill_dir = workspace / "skills" / "spot-agent-cypher"
        skill_dir.mkdir(parents=True)
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            "\n".join([
                "---",
                "name: spot-agent-cypher",
                "---",
                "CLI := node skills/spot-agent-cypher/cli/index.js",
                "## Setup",
                "RUN $CLI wallet",
                'MATCH output: "No wallet" -> run $CLI wallet again',
                "RUN $CLI join {gameId} {spot} again",
                "",
                "## Commands",
                "```bash",
                "$CLI wallet",
                "$CLI faucet [-c <code>]",
                "$CLI join [gameId] [spot]",
                "$CLI wait <gameId>",
                "$CLI reveal <gameId>",
                "$CLI settlement <gameId>",
                "```",
            ]),
            encoding="utf-8",
        )

        templates = ShellTool._parse_skill_command_templates(skill_md)
        displays = [template.display for template in templates]

        assert "node skills/spot-agent-cypher/cli/index.js wallet again" not in displays
        assert "node skills/spot-agent-cypher/cli/index.js join <gameId> <spot> again" not in displays
        assert "node skills/spot-agent-cypher/cli/index.js settlement <gameId>" in displays

        shell_tool = ShellTool(working_dir=str(workspace))
        assert shell_tool._reject_undocumented_skill_cli_arguments(
            "node skills/spot-agent-cypher/cli/index.js settlement 142",
            str(workspace),
        ) is None

    def test_shell_skill_cli_templates_ignore_inline_prose(self, tmp_path):
        """Fallback parsing accepts explicit RUN lines without scanning whole prose."""
        from spoon_bot.agent.tools.shell import ShellTool

        workspace = tmp_path / "workspace"
        skill_dir = workspace / "skills" / "demo-skill"
        skill_dir.mkdir(parents=True)
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            "\n".join([
                "---",
                "name: demo-skill",
                "---",
                "CLI := node skills/demo-skill/cli/index.js",
                'MATCH output: "No wallet" -> then run $CLI wallet again',
                "RUN $CLI register",
            ]),
            encoding="utf-8",
        )

        displays = [
            template.display
            for template in ShellTool._parse_skill_command_templates(skill_md)
        ]

        assert displays == ["node skills/demo-skill/cli/index.js register"]

    def test_shell_rejects_neighboring_skill_cli_option(self, tmp_path):
        """Nearby option names are rejected when SKILL.md documents a different one."""
        from spoon_bot.agent.tools.shell import ShellTool

        workspace = tmp_path / "workspace"
        skill_dir = workspace / "skills" / "spot-agent-cypher"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "\n".join([
                "---",
                "name: spot-agent-cypher",
                "---",
                "CLI := node skills/spot-agent-cypher/cli/index.js",
                "$CLI faucet [-c <code>]",
            ]),
            encoding="utf-8",
        )
        shell_tool = ShellTool(working_dir=str(workspace))

        result = shell_tool._reject_undocumented_skill_cli_arguments(
            "node skills/spot-agent-cypher/cli/index.js faucet --invite 3KK57S",
            str(workspace),
        )

        assert result is not None
        assert "unsupported option '--invite'" in result
        assert "Allowed options: -c" in result

    def test_shell_rejects_labeled_request_value_on_unrelated_skill_cli_parameter(self, tmp_path):
        """User-labeled values should match the command parameter label."""
        from spoon_bot.agent.tools.execution_context import bind_request_execution_hints
        from spoon_bot.agent.tools.shell import ShellTool

        workspace = tmp_path / "workspace"
        skill_dir = workspace / "skills" / "spot-agent-cypher"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "\n".join([
                "---",
                "name: spot-agent-cypher",
                "---",
                "CLI := node skills/spot-agent-cypher/cli/index.js",
                "$CLI faucet [-c <code>]",
                "$CLI join <spot>",
                "$CLI join [gameId] [spot]",
            ]),
            encoding="utf-8",
        )
        shell_tool = ShellTool(working_dir=str(workspace))

        with bind_request_execution_hints({
            "explicit_request_values": [{
                "value": "3KK57S",
                "labels": ["code", "invited"],
                "label": "Invited Code",
            }]
        }):
            rejected = shell_tool._reject_undocumented_skill_cli_arguments(
                "node skills/spot-agent-cypher/cli/index.js join A 3KK57S",
                str(workspace),
            )
            allowed_code = shell_tool._reject_undocumented_skill_cli_arguments(
                "node skills/spot-agent-cypher/cli/index.js faucet -c 3KK57S",
                str(workspace),
            )
            allowed_join = shell_tool._reject_undocumented_skill_cli_arguments(
                "node skills/spot-agent-cypher/cli/index.js join 123 A",
                str(workspace),
            )

        assert rejected is not None
        assert "labeled value from the user request" in rejected
        assert allowed_code is None
        assert allowed_join is None

    def test_shell_augments_skill_cli_from_matching_labeled_request_value(self, tmp_path):
        """Documented value flags can be filled from structured request facts."""
        from spoon_bot.agent.tools.shell import ShellTool

        workspace = tmp_path / "workspace"
        skill_dir = workspace / "skills" / "spot-agent-cypher"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "\n".join([
                "---",
                "name: spot-agent-cypher",
                "---",
                "CLI := node skills/spot-agent-cypher/cli/index.js",
                "$CLI faucet [-c <code>]",
            ]),
            encoding="utf-8",
        )
        shell_tool = ShellTool(working_dir=str(workspace))
        shell_tool._request_execution_hints = {
            "explicit_request_values": [{
                "value": "3KK57S",
                "labels": ["code", "invited"],
                "label": "Invited Code",
            }]
        }

        augmented = shell_tool._augment_skill_cli_labeled_values(
            "node skills/spot-agent-cypher/cli/index.js faucet",
            str(workspace),
        )

        assert augmented.endswith("faucet -c 3KK57S")

    def test_shell_allows_long_flag_alias_matching_documented_value_label(self, tmp_path):
        """Skill-emitted long flag aliases are accepted when their label matches."""
        from spoon_bot.agent.tools.execution_context import bind_request_execution_hints
        from spoon_bot.agent.tools.shell import ShellTool

        workspace = tmp_path / "workspace"
        skill_dir = workspace / "skills" / "spot-agent-cypher"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "\n".join([
                "---",
                "name: spot-agent-cypher",
                "---",
                "CLI := node skills/spot-agent-cypher/cli/index.js",
                "$CLI faucet-answer <challengeId> <answer> [-c <code>]",
            ]),
            encoding="utf-8",
        )
        shell_tool = ShellTool(working_dir=str(workspace))

        with bind_request_execution_hints({
            "explicit_request_values": [{
                "value": "3KK57S",
                "labels": ["code", "invited"],
                "label": "Invited Code",
            }]
        }):
            rejection = shell_tool._reject_undocumented_skill_cli_arguments(
                "node skills/spot-agent-cypher/cli/index.js faucet-answer abc 4 --invitation-code 3KK57S",
                str(workspace),
            )

        assert rejection is None

    def test_shell_allows_help_for_documented_skill_cli_command(self, tmp_path):
        """Help flags are harmless command introspection, not invented workflow args."""
        from spoon_bot.agent.tools.shell import ShellTool

        workspace = tmp_path / "workspace"
        skill_dir = workspace / "skills" / "demo-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "\n".join([
                "---",
                "name: demo-skill",
                "---",
                "CLI := node skills/demo-skill/cli/index.js",
                "RUN $CLI register",
            ]),
            encoding="utf-8",
        )
        shell_tool = ShellTool(working_dir=str(workspace))

        assert shell_tool._reject_undocumented_skill_cli_arguments(
            "node skills/demo-skill/cli/index.js register --help",
            str(workspace),
        ) is None

    def test_shell_read_only_background_actions_skip_exact_dedup(self, shell_tool):
        """Polling an existing background job should not be mistaken for a loop."""
        for action in ("list_jobs", "job_status", "job_output"):
            assert shell_tool.tool_invocation_dedup_key(
                {"action": action, "job_id": "sh_testjob"}
            ) is None

        assert shell_tool.tool_invocation_dedup_key(
            {"action": "terminate_job", "job_id": "sh_testjob"}
        ) == {"action": "terminate_job", "job_id": "sh_testjob"}

    @pytest.mark.asyncio
    async def test_background_job_status_can_be_polled_repeatedly_in_request_scope(self, shell_tool):
        """The generic duplicate guard should not block repeated status reads."""
        from spoon_bot.agent.tools.execution_context import track_tool_invocations
        from spoon_bot.agent.tools.shell import _BackgroundShellJob, _SHELL_BACKGROUND_JOBS

        class _FakeProcess:
            returncode = None

            async def wait(self):
                return None

            def terminate(self):
                self.returncode = -15

            def kill(self):
                self.returncode = -9

        _SHELL_BACKGROUND_JOBS.clear()
        stdout_task = asyncio.create_task(asyncio.sleep(0))
        stderr_task = asyncio.create_task(asyncio.sleep(0))
        job = _BackgroundShellJob(
            job_id="sh_polljob",
            command="echo hello",
            cwd=os.getcwd(),
            process=_FakeProcess(),
            stdout_task=stdout_task,
            stderr_task=stderr_task,
            buffer_limit=1000,
            stdout_text="line one",
        )
        _SHELL_BACKGROUND_JOBS[job.job_id] = job

        with track_tool_invocations(max_repeats=1):
            first = await shell_tool(action="job_status", job_id=job.job_id)
            second = await shell_tool(action="job_status", job_id=job.job_id)

        assert "job_id: sh_polljob" in first
        assert "job_id: sh_polljob" in second
        assert "STOP_TOOL_LOOP" not in second

    @pytest.mark.asyncio
    async def test_refresh_completed_job_does_not_hang_on_open_child_pipes(self, shell_tool):
        """Detached child processes should not keep a completed shell call open forever."""
        from spoon_bot.agent.tools.shell import _BackgroundShellJob

        class _CompletedProcess:
            returncode = 0

        job = _BackgroundShellJob(
            job_id="sh_orphanpipe",
            command="node server.js &",
            cwd=os.getcwd(),
            process=_CompletedProcess(),
            stdout_task=asyncio.create_task(asyncio.sleep(60)),
            stderr_task=asyncio.create_task(asyncio.sleep(60)),
            buffer_limit=1000,
        )

        refreshed = await shell_tool._refresh_background_job(job)

        assert refreshed.status == "completed"
        assert refreshed.returncode == 0
        assert job.stdout_task.cancelled()
        assert job.stderr_task.cancelled()

    @pytest.mark.asyncio
    async def test_shell_rejects_unmanaged_background_operator(self, shell_tool):
        """Long-lived commands should use managed jobs or dedicated service tools."""
        result = await shell_tool(command="node server.js &")

        assert "unmanaged shell background operator" in result
        assert "managed background jobs" in result

    def test_shell_stops_after_exact_requested_command_failure(self, shell_tool):
        """Exact user-requested shell commands should fail fast without extra tool detours."""
        from spoon_bot.agent.tools.execution_context import bind_request_execution_hints

        with bind_request_execution_hints(
            {"exact_shell_commands": ["node skills/spot-agent-cypher/cli/index.js join A"]}
        ):
            result = shell_tool._maybe_stop_after_exact_command_failure(
                "node skills/spot-agent-cypher/cli/index.js join A",
                "SPOT API GET /api/agent/games/assign failed after 5 attempts: fetch failed",
            )

        assert result.startswith("STOP_TOOL_LOOP: Exact requested shell command failed.")

    def test_shell_exact_command_guard_ignores_non_matching_commands(self, shell_tool):
        """The fast-stop guard should only apply to the exact requested command."""
        from spoon_bot.agent.tools.execution_context import bind_request_execution_hints

        with bind_request_execution_hints(
            {"exact_shell_commands": ["node skills/spot-agent-cypher/cli/index.js join A"]}
        ):
            result = shell_tool._maybe_stop_after_exact_command_failure(
                "node skills/spot-agent-cypher/cli/index.js wallet",
                "Exit code: 1",
            )

        assert result == "Exit code: 1"

    @pytest.mark.asyncio
    async def test_shell_defers_external_probe_until_current_session_fact_check(self, shell_tool):
        """Prior-action disputes must search current session history before shell probes."""
        from spoon_bot.agent.tools.execution_context import (
            bind_request_execution_hints,
            track_tool_invocations,
        )

        with bind_request_execution_hints({"current_session_fact_check_required": True}):
            with track_tool_invocations():
                result = await shell_tool.execute("echo should-not-run")

        assert "Current-session fact check required" in result
        assert "search_history(scope='current')" in result
        assert "should-not-run" not in result

    def test_fact_check_blocker_clears_after_search_history(self):
        """The guard is a sequencing rule, not a permanent ban on external tools."""
        from spoon_bot.agent.tools.execution_context import (
            bind_request_execution_hints,
            current_session_fact_check_blocker,
            suppress_repeated_tool_invocation,
            track_tool_invocations,
        )

        with bind_request_execution_hints({"current_session_fact_check_required": True}):
            with track_tool_invocations():
                assert "search_history(scope='current')" in (
                    current_session_fact_check_blocker() or ""
                )
                suppress_repeated_tool_invocation("search_history", {"query": "previous"})
                assert current_session_fact_check_blocker() is None

    @pytest.mark.asyncio
    async def test_memory_is_not_current_session_fact_check_substitute(self):
        """Long-term memory should not satisfy current-session transcript checks."""
        from spoon_bot.agent.tools.execution_context import (
            bind_request_execution_hints,
            track_tool_invocations,
        )
        from spoon_bot.agent.tools.self_config import MemoryManagementTool

        class _Store:
            def search(self, query):
                return ["stale long-term note"]

        tool = MemoryManagementTool(_Store())
        with bind_request_execution_hints({"current_session_fact_check_required": True}):
            with track_tool_invocations():
                result = await tool.execute(action="search", query="previous")

        assert "search_history(scope='current')" in result
        assert "Long-term memory is not the current-session transcript" in result
        assert "stale long-term note" not in result

    @pytest.mark.asyncio
    async def test_memory_allows_repeated_read_only_actions_in_request_scope(self):
        """Read-only memory checks should not trip exact duplicate suppression."""
        from spoon_bot.agent.tools.execution_context import track_tool_invocations
        from spoon_bot.agent.tools.self_config import MemoryManagementTool

        class _Store:
            def __init__(self):
                self.added = 0

            def get_summary(self):
                return "Long-term memory: 0 entries"

            def search(self, query):
                return [f"hit:{query}"]

            def add_memory(self, content, category):
                self.added += 1

        tool = MemoryManagementTool(_Store())

        assert tool.tool_invocation_dedup_key({"action": "summary"}) is None
        assert tool.tool_invocation_dedup_key({"action": "search", "query": "x"}) is None
        assert tool.tool_invocation_dedup_key({"action": "remember", "content": "x"}) == {
            "action": "remember",
            "content": "x",
        }
        assert tool.tool_invocation_dedup_key({"action": "note", "content": "x"}) == {
            "action": "note",
            "content": "x",
        }
        assert tool.tool_invocation_dedup_key({"action": "forget", "content": "x"}) == {
            "action": "forget",
            "content": "x",
        }

        with track_tool_invocations(max_repeats=1):
            first_summary = await tool(action="summary")
            second_summary = await tool(action="summary")
            first_search = await tool(action="search", query="install")
            second_search = await tool(action="search", query="install")
            first_write = await tool(action="remember", content="install done")
            duplicate_write = await tool(action="remember", content="install done")

        assert first_summary == "Long-term memory: 0 entries"
        assert second_summary == "Long-term memory: 0 entries"
        assert "hit:install" in first_search
        assert "hit:install" in second_search
        assert "Remembered: install done..." == first_write
        assert "STOP_TOOL_LOOP" in duplicate_write
        assert tool._memory_store.added == 1

    def test_shell_rejects_git_clone_into_workspace_skills(self, shell_tool, tmp_path):
        """Workspace skills must go through skill management, not manual clones."""
        workspace = tmp_path / "workspace"
        skills = workspace / "skills"
        skills.mkdir(parents=True)
        shell_tool.working_dir = str(workspace)

        result = shell_tool._reject_workspace_skill_clone(
            "git clone https://github.com/example-org/example-skill.git skills/example-skill",
            str(workspace),
        )

        assert result is not None
        assert not result.startswith("STOP_TOOL_LOOP:")
        assert result.startswith("Rejected:")
        assert "skill_marketplace(action='install_skill'" in result

    def test_shell_allows_git_clone_outside_workspace_skills(self, shell_tool, tmp_path):
        """Regular repository work outside workspace/skills remains allowed."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        shell_tool.working_dir = str(workspace)

        result = shell_tool._reject_workspace_skill_clone(
            "git clone https://github.com/example-org/example-repo.git repos/example-repo",
            str(workspace),
        )

        assert result is None

    def test_shell_rejects_implicit_clone_destination_inside_skills_cwd(self, shell_tool, tmp_path):
        """A clone run from workspace/skills also counts as a manual skill install."""
        workspace = tmp_path / "workspace"
        skills = workspace / "skills"
        skills.mkdir(parents=True)
        shell_tool.working_dir = str(workspace)

        result = shell_tool._reject_workspace_skill_clone(
            "git clone https://github.com/example-org/example-skill.git",
            str(skills),
        )

        assert result is not None
        assert result.startswith("Rejected:")

    def test_shell_description_preserves_protective_wrappers(self, shell_tool):
        """Tool instructions should not invite converting a replay into a live command."""
        description = shell_tool.description

        assert "execute it exactly as provided" in description
        assert "do not remove protective wrappers" in description
        assert "echo/printf" in description

    @pytest.mark.asyncio
    async def test_dangerous_command_blocked(self, shell_tool):
        """Test that dangerous commands are blocked."""
        result = await shell_tool.execute("rm -rf /")
        assert "Security Error" in result or "blocked" in result.lower()

    @pytest.mark.asyncio
    async def test_command_injection_blocked(self, shell_tool):
        """Test that command injection is blocked."""
        # Command chaining should be blocked
        result = await shell_tool.execute("echo hello; rm -rf /")
        assert "Security Error" in result or "blocked" in result.lower() or "injection" in result.lower()

    @pytest.mark.asyncio
    async def test_nonexistent_directory(self, shell_tool):
        """Test handling of nonexistent working directory."""
        result = await shell_tool.execute("echo test", working_dir="/nonexistent/path")
        assert "Error" in result or "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_relative_working_dir_resolves_from_default_workspace(self, tmp_path):
        """Relative cwd overrides should stay inside the configured workspace."""
        from spoon_bot.agent.tools.shell import ShellTool
        from spoon_bot.utils.rate_limit import RateLimitConfig

        marker = tmp_path / "marker.txt"
        marker.write_text("workspace-ok", encoding="utf-8")
        tool = ShellTool(
            timeout=5,
            working_dir=str(tmp_path),
            rate_limit_config=RateLimitConfig.unlimited(),
        )

        result = await tool.execute(
            "python -c \"print(open('marker.txt').read())\"",
            working_dir=".",
        )

        assert "workspace-ok" in result

    def test_windows_posix_drive_working_dir_is_normalized(self):
        """Windows agents may receive /c/... cwd paths from bash-style output."""
        if sys.platform != "win32":
            pytest.skip("Windows-only path normalization")

        from spoon_bot.agent.tools.shell import ShellTool

        assert ShellTool._posix_drive_path_to_windows("/c/Users/Test") == "C:\\Users\\Test"

    @pytest.mark.asyncio
    async def test_output_truncation(self):
        """Test that long output is truncated."""
        from spoon_bot.agent.tools.shell import ShellTool
        from spoon_bot.utils.rate_limit import RateLimitConfig

        tool = ShellTool(
            max_output=50,
            rate_limit_config=RateLimitConfig.unlimited(),
        )

        # Generate long output
        if sys.platform == "win32":
            result = await tool.execute("echo " + "x" * 100)
        else:
            result = await tool.execute("echo " + "x" * 100)

        # Should be truncated
        assert "truncated" in result.lower() or len(result) <= 100

    @pytest.mark.asyncio
    async def test_command_moves_to_background_after_timeout(self, shell_tool):
        """Timed-out commands should stay alive as background jobs instead of erroring out."""
        from spoon_bot.agent.tools.shell import _BackgroundShellJob, _SHELL_BACKGROUND_JOBS

        shell_tool.timeout = 0.01
        shell_tool.background_handoff_timeout = 0

        class _FakeProcess:
            returncode = None

            async def wait(self):
                await asyncio.sleep(1)

            def terminate(self):
                self.returncode = -15

            def kill(self):
                self.returncode = -9

        stdout_task = asyncio.create_task(asyncio.sleep(1))
        stderr_task = asyncio.create_task(asyncio.sleep(1))
        job = _BackgroundShellJob(
            job_id="sh_testjob",
            command="echo hello",
            cwd=os.getcwd(),
            process=_FakeProcess(),
            stdout_task=stdout_task,
            stderr_task=stderr_task,
            buffer_limit=1000,
            stdout_text="still running",
        )

        async def _fake_start_background_job(*args, **kwargs):
            _SHELL_BACKGROUND_JOBS[job.job_id] = job
            return job

        with patch.object(shell_tool, "_start_background_job", AsyncMock(side_effect=_fake_start_background_job)):
            result = await shell_tool.execute("echo hello")

        assert "background" in result.lower()
        assert "sh_testjob" in result
        assert _SHELL_BACKGROUND_JOBS["sh_testjob"] is job
        stdout_task.cancel()
        stderr_task.cancel()

    @pytest.mark.asyncio
    async def test_command_completion_during_timeout_handoff_returns_output(self, shell_tool):
        """Near-timeout completions should not force the agent into background polling."""
        from spoon_bot.agent.tools.shell import _BackgroundShellJob, _SHELL_BACKGROUND_JOBS

        shell_tool.timeout = 0.01
        shell_tool.background_handoff_timeout = 0.2

        class _FakeProcess:
            def __init__(self):
                self.returncode = None

            async def wait(self):
                await asyncio.sleep(0.03)
                self.returncode = 0
                return self.returncode

            def terminate(self):
                self.returncode = -15

            def kill(self):
                self.returncode = -9

        _SHELL_BACKGROUND_JOBS.clear()
        job = _BackgroundShellJob(
            job_id="sh_handoff_done",
            command="echo hello",
            cwd=os.getcwd(),
            process=_FakeProcess(),
            stdout_task=asyncio.create_task(asyncio.sleep(0)),
            stderr_task=asyncio.create_task(asyncio.sleep(0)),
            buffer_limit=1000,
            stdout_text="done",
        )

        async def _fake_start_background_job(*args, **kwargs):
            _SHELL_BACKGROUND_JOBS[job.job_id] = job
            return job

        with patch.object(shell_tool, "_start_background_job", AsyncMock(side_effect=_fake_start_background_job)):
            result = await shell_tool.execute("echo hello")

        assert "done" in result
        assert "command moved to background" not in result
        assert job.job_id not in _SHELL_BACKGROUND_JOBS

    @pytest.mark.asyncio
    async def test_background_job_status_and_terminate_actions(self, shell_tool):
        """Shell background jobs should expose status and terminate actions."""
        from spoon_bot.agent.tools.shell import _BackgroundShellJob, _SHELL_BACKGROUND_JOBS

        class _FakeProcess:
            def __init__(self):
                self.returncode = None

            async def wait(self):
                self.returncode = -15
                return self.returncode

            def terminate(self):
                self.returncode = -15

            def kill(self):
                self.returncode = -9

        stdout_task = asyncio.create_task(asyncio.sleep(0))
        stderr_task = asyncio.create_task(asyncio.sleep(0))
        job = _BackgroundShellJob(
            job_id="sh_statusjob",
            command="echo hello",
            cwd=os.getcwd(),
            process=_FakeProcess(),
            stdout_task=stdout_task,
            stderr_task=stderr_task,
            buffer_limit=1000,
            stdout_text="line one",
        )
        _SHELL_BACKGROUND_JOBS[job.job_id] = job

        status = await shell_tool.execute(action="job_status", job_id=job.job_id)
        assert "job_id: sh_statusjob" in status
        assert "line one" in status

        terminated = await shell_tool.execute(action="terminate_job", job_id=job.job_id)
        assert "Terminated background shell job sh_statusjob" in terminated

    @pytest.mark.asyncio
    async def test_silent_running_background_job_is_not_terminated_without_force(self, shell_tool):
        """Silence alone should not make the agent kill a long-running job."""
        from spoon_bot.agent.tools.shell import _BackgroundShellJob, _SHELL_BACKGROUND_JOBS

        class _FakeProcess:
            def __init__(self):
                self.returncode = None
                self.terminated = False

            async def wait(self):
                self.returncode = -15
                return self.returncode

            def terminate(self):
                self.terminated = True
                self.returncode = -15

            def kill(self):
                self.terminated = True
                self.returncode = -9

        _SHELL_BACKGROUND_JOBS.clear()
        process = _FakeProcess()
        job = _BackgroundShellJob(
            job_id="sh_silentjob",
            command="node cli/index.js long-chain-operation",
            cwd=os.getcwd(),
            process=process,
            stdout_task=asyncio.create_task(asyncio.sleep(0)),
            stderr_task=asyncio.create_task(asyncio.sleep(0)),
            buffer_limit=1000,
        )
        _SHELL_BACKGROUND_JOBS[job.job_id] = job

        result = await shell_tool.execute(action="terminate_job", job_id=job.job_id)

        assert "Not terminated" in result
        assert "Silent output alone is not evidence" in result
        assert process.terminated is False
        assert job.status == "running"

        forced = await shell_tool.execute(
            action="terminate_job",
            job_id=job.job_id,
            force=True,
        )

        assert "Terminated background shell job sh_silentjob" in forced
        assert process.terminated is True

    @pytest.mark.asyncio
    async def test_terminate_completed_background_job_preserves_terminal_status(self, shell_tool):
        """Terminating an already completed job should not relabel successful output."""
        from spoon_bot.agent.tools.shell import _BackgroundShellJob, _SHELL_BACKGROUND_JOBS

        class _FakeProcess:
            returncode = 0

            async def wait(self):
                return 0

            def terminate(self):
                self.returncode = -15

            def kill(self):
                self.returncode = -9

        _SHELL_BACKGROUND_JOBS.clear()
        job = _BackgroundShellJob(
            job_id="sh_donejob",
            command="echo done",
            cwd=os.getcwd(),
            process=_FakeProcess(),
            stdout_task=asyncio.create_task(asyncio.sleep(0)),
            stderr_task=asyncio.create_task(asyncio.sleep(0)),
            buffer_limit=1000,
            stdout_text="NEXT: continue",
            status="completed",
            returncode=0,
            finished_at=1.0,
        )
        _SHELL_BACKGROUND_JOBS[job.job_id] = job

        result = await shell_tool.execute(action="terminate_job", job_id=job.job_id)

        assert "already completed" in result
        assert "NEXT: continue" in result
        assert "Terminated background shell job" not in result
        assert job.status == "completed"

    @pytest.mark.asyncio
    async def test_background_jobs_are_scoped_by_owner(self, shell_tool):
        """Shell background job actions should only expose caller-owned jobs."""
        from spoon_bot.agent.tools.execution_context import bind_tool_owner
        from spoon_bot.agent.tools.shell import _BackgroundShellJob, _SHELL_BACKGROUND_JOBS

        class _FakeProcess:
            returncode = None

            async def wait(self):
                return None

            def terminate(self):
                self.returncode = -15

            def kill(self):
                self.returncode = -9

        _SHELL_BACKGROUND_JOBS.clear()
        owner_a_job = _BackgroundShellJob(
            job_id="sh_owner_a",
            command="echo owner-a",
            cwd=os.getcwd(),
            process=_FakeProcess(),
            stdout_task=asyncio.create_task(asyncio.sleep(0)),
            stderr_task=asyncio.create_task(asyncio.sleep(0)),
            buffer_limit=1000,
            stdout_text="owner-a output",
            owner_key="owner-a",
        )
        owner_b_job = _BackgroundShellJob(
            job_id="sh_owner_b",
            command="echo owner-b",
            cwd=os.getcwd(),
            process=_FakeProcess(),
            stdout_task=asyncio.create_task(asyncio.sleep(0)),
            stderr_task=asyncio.create_task(asyncio.sleep(0)),
            buffer_limit=1000,
            stdout_text="owner-b output",
            owner_key="owner-b",
        )
        _SHELL_BACKGROUND_JOBS[owner_a_job.job_id] = owner_a_job
        _SHELL_BACKGROUND_JOBS[owner_b_job.job_id] = owner_b_job

        with bind_tool_owner("owner-a"):
            listed = await shell_tool.execute(action="list_jobs")
            denied = await shell_tool.execute(action="job_status", job_id=owner_b_job.job_id)

        assert "sh_owner_a" in listed
        assert "sh_owner_b" not in listed
        assert "not found" in denied.lower()

    @pytest.mark.asyncio
    async def test_foreground_completion_evicts_job_from_registry(self, shell_tool):
        """Completed foreground shell runs should not remain in global job registry."""
        from spoon_bot.agent.tools.shell import _BackgroundShellJob, _SHELL_BACKGROUND_JOBS

        class _FakeProcess:
            returncode = 0

            async def wait(self):
                return 0

            def terminate(self):
                self.returncode = -15

            def kill(self):
                self.returncode = -9

        _SHELL_BACKGROUND_JOBS.clear()
        job = _BackgroundShellJob(
            job_id="sh_foreground_done",
            command="echo done",
            cwd=os.getcwd(),
            process=_FakeProcess(),
            stdout_task=asyncio.create_task(asyncio.sleep(0)),
            stderr_task=asyncio.create_task(asyncio.sleep(0)),
            buffer_limit=1000,
            stdout_text="done",
        )

        async def _fake_start_background_job(*args, **kwargs):
            _SHELL_BACKGROUND_JOBS[job.job_id] = job
            return job

        with patch.object(shell_tool, "_start_background_job", AsyncMock(side_effect=_fake_start_background_job)):
            result = await shell_tool.execute("echo done")

        assert "done" in result.lower()
        assert job.job_id not in _SHELL_BACKGROUND_JOBS

    @pytest.mark.asyncio
    async def test_foreground_cancellation_terminates_job(self, shell_tool):
        """Cancelling a foreground shell tool call must not leave the process running."""
        from spoon_bot.agent.tools.shell import _BackgroundShellJob, _SHELL_BACKGROUND_JOBS

        started = asyncio.Event()
        terminated = asyncio.Event()

        class _FakeProcess:
            returncode = None
            pid = None

            async def wait(self):
                started.set()
                while self.returncode is None:
                    await asyncio.sleep(0.01)
                return self.returncode

            def terminate(self):
                self.returncode = -15
                terminated.set()

            def kill(self):
                self.returncode = -9
                terminated.set()

        _SHELL_BACKGROUND_JOBS.clear()
        stdout_task = asyncio.create_task(asyncio.sleep(60))
        stderr_task = asyncio.create_task(asyncio.sleep(60))
        job = _BackgroundShellJob(
            job_id="sh_cancelled",
            command="sleep 60",
            cwd=os.getcwd(),
            process=_FakeProcess(),
            stdout_task=stdout_task,
            stderr_task=stderr_task,
            buffer_limit=1000,
            stdout_text="started",
        )

        async def _fake_start_background_job(*args, **kwargs):
            _SHELL_BACKGROUND_JOBS[job.job_id] = job
            return job

        with patch.object(shell_tool, "_start_background_job", AsyncMock(side_effect=_fake_start_background_job)):
            task = asyncio.create_task(shell_tool.execute("sleep 60", timeout=60))
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert terminated.is_set()
        assert job.status == "cancelled"
        assert job.job_id not in _SHELL_BACKGROUND_JOBS
        assert stdout_task.done()
        assert stderr_task.done()

    def test_windows_shell_uses_user_home_env(self, monkeypatch):
        """Windows bash execution should preserve the user's home path."""
        from spoon_bot.agent.tools.shell import ShellTool
        from spoon_bot.utils.rate_limit import RateLimitConfig

        tool = ShellTool(
            timeout=5,
            max_output=1000,
            rate_limit_config=RateLimitConfig.unlimited(),
        )

        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(tool, "_find_bash", lambda: "C:/Program Files/Git/bin/bash.exe")

        captured: dict[str, object] = {}

        def fake_run(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

            class _Result:
                stdout = b"ok"
                stderr = b""
                returncode = 0

            return _Result()

        with patch("subprocess.run", side_effect=fake_run):
            out, err, code = tool._run_sync("echo hello", str(Path.home()))

        assert out == b"ok"
        assert err == b""
        assert code == 0
        assert captured["args"][0][1] == "-c"
        env = captured["kwargs"]["env"]
        assert env["USERPROFILE"] == str(Path.home())
        assert env["HOME"] == tool._windows_home_to_bash(str(Path.home()))

    def test_windows_home_to_bash_path(self):
        """Windows home paths should be converted to Git Bash POSIX form."""
        from spoon_bot.agent.tools.shell import ShellTool

        assert ShellTool._windows_home_to_bash(r"C:\Users\Ricky") == "/c/Users/Ricky"
        assert ShellTool._windows_home_to_bash("/already/posix") == "/already/posix"


class TestCommandValidator:
    """Tests for the CommandValidator."""

    @pytest.fixture
    def validator(self):
        """Create a CommandValidator instance."""
        from spoon_bot.agent.tools.shell import CommandValidator
        return CommandValidator()

    def test_valid_command(self, validator):
        """Test that valid commands pass validation."""
        is_valid, error = validator.validate("ls -la")
        assert is_valid is True
        assert error is None

    def test_empty_command(self, validator):
        """Test that empty commands are rejected."""
        is_valid, error = validator.validate("")
        assert is_valid is False
        assert error is not None

    def test_dangerous_rm_rf(self, validator):
        """Test that rm -rf / is blocked."""
        is_valid, error = validator.validate("rm -rf /")
        assert is_valid is False
        assert "dangerous" in error.lower() or "blocked" in error.lower()

    def test_fork_bomb(self, validator):
        """Test that fork bombs are blocked."""
        is_valid, error = validator.validate(":(){ :|:& };:")
        assert is_valid is False

    def test_whitelist_mode(self):
        """Test whitelist mode validation."""
        from spoon_bot.agent.tools.shell import CommandValidator

        validator = CommandValidator(whitelist_mode=True)

        # Whitelisted command should pass
        is_valid, _ = validator.validate("ls -la")
        assert is_valid is True

        # Non-whitelisted command should fail
        is_valid, error = validator.validate("custom_dangerous_cmd")
        assert is_valid is False
        assert "whitelist" in error.lower()


class TestFilesystemTools:
    """Tests for filesystem tools.

    Note: Filesystem tools have workspace security that requires paths to be
    within the configured workspace. Tests set the workspace to the temp directory.
    """

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def read_tool(self, temp_dir):
        """Create a ReadFileTool instance with temp_dir as workspace."""
        from spoon_bot.agent.tools.filesystem import ReadFileTool
        tool = ReadFileTool(workspace=temp_dir)
        return tool

    @pytest.fixture
    def write_tool(self, temp_dir):
        """Create a WriteFileTool instance with temp_dir as workspace."""
        from spoon_bot.agent.tools.filesystem import WriteFileTool
        tool = WriteFileTool(workspace=temp_dir)
        return tool

    @pytest.fixture
    def list_tool(self, temp_dir):
        """Create a ListDirTool instance with temp_dir as workspace."""
        from spoon_bot.agent.tools.filesystem import ListDirTool
        tool = ListDirTool(workspace=temp_dir)
        return tool

    @pytest.mark.asyncio
    async def test_read_existing_file(self, read_tool, temp_dir):
        """Test reading an existing file within workspace."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello, World!")

        result = await read_tool.execute(path=str(test_file))
        assert "Hello, World!" in result

    @pytest.mark.asyncio
    async def test_read_skill_md_returns_execution_summary(self, temp_dir):
        """Large SKILL.md reads should surface command contracts to the model."""
        from spoon_bot.agent.tools.filesystem import ReadFileTool

        skill_dir = temp_dir / "skills" / "spot-agent-cypher"
        skill_dir.mkdir(parents=True)
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            "\n".join([
                "---",
                "name: spot-agent-cypher",
                "description: Play SPOT through the local CLI.",
                "---",
                "# Spot Agent",
                "CLI := node skills/spot-agent-cypher/cli/index.js",
                "## Setup",
                *("setup details" for _ in range(300)),
                "RUN $CLI join {spot} again so backend can assign a fresh room.",
                "## Commands",
                "```bash",
                "$CLI wallet",
                "$CLI join [gameId] [spot]",
                "$CLI settlement <gameId>",
                "```",
            ]),
            encoding="utf-8",
        )
        tool = ReadFileTool(workspace=temp_dir, max_output=900)

        result = await tool.execute(path=str(skill_md))

        assert "skill-ref" in result
        assert "[SKILL.md execution summary]" in result
        assert "node skills/spot-agent-cypher/cli/index.js" in result
        assert "Operational contract:" in result
        assert "RUN $CLI join {spot} again so backend can assign a fresh room" in result
        assert "$CLI join [gameId] [spot]" in result
        assert "$CLI settlement <gameId>" in result

    @pytest.mark.asyncio
    async def test_repeated_read_returns_cache_hit(self, temp_dir):
        """Repeated same-range reads should tell the model to continue."""
        from spoon_bot.agent.tools.execution_context import track_tool_invocations
        from spoon_bot.agent.tools.filesystem import ReadFileTool

        target = temp_dir / "notes.txt"
        target.write_text("alpha\nbeta\n", encoding="utf-8")
        tool = ReadFileTool(workspace=temp_dir, max_output=900)

        with track_tool_invocations():
            first = await tool.execute(path=str(target))
            second = await tool.execute(path=str(target))
            third = await tool.execute(path=str(target))

        assert "alpha" in first
        assert "File content already available" in second
        assert "without calling read_file again" in second
        assert "Repeated read skipped" in third
        assert "next non-read action" in third

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, read_tool, temp_dir):
        """Test reading a nonexistent file."""
        result = await read_tool.execute(path=str(temp_dir / "nonexistent.txt"))
        assert "Error" in result or "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_write_file(self, write_tool, temp_dir):
        """Test writing to a file within workspace."""
        test_file = temp_dir / "output.txt"
        result = await write_tool.execute(
            path=str(test_file),
            content="Test content"
        )

        assert "Successfully wrote" in result
        assert test_file.exists()
        assert test_file.read_text() == "Test content"

    @pytest.mark.asyncio
    async def test_write_file_requires_explicit_overwrite_for_existing_file(self, write_tool, temp_dir):
        """Existing files should default to targeted edits, not blind replacement."""
        test_file = temp_dir / "output.txt"
        test_file.write_text("original", encoding="utf-8")

        result = await write_tool.execute(path=str(test_file), content="replacement")

        assert "File already exists" in result
        assert "edit_file" in result
        assert test_file.read_text(encoding="utf-8") == "original"

    @pytest.mark.asyncio
    async def test_write_file_allows_explicit_overwrite_for_existing_file(self, write_tool, temp_dir):
        test_file = temp_dir / "output.txt"
        test_file.write_text("original", encoding="utf-8")

        result = await write_tool.execute(
            path=str(test_file),
            content="replacement",
            overwrite=True,
        )

        assert "Successfully overwrote" in result
        assert test_file.read_text(encoding="utf-8") == "replacement"

    @pytest.mark.asyncio
    async def test_list_directory(self, list_tool, temp_dir):
        """Test listing directory contents within workspace."""
        # Create some test files
        (temp_dir / "file1.txt").write_text("content1")
        (temp_dir / "file2.txt").write_text("content2")

        result = await list_tool.execute(path=str(temp_dir))

        assert "file1.txt" in result
        assert "file2.txt" in result

    @pytest.mark.asyncio
    async def test_read_and_list_allowed_external_skill_symlink(self, temp_dir):
        """Workspace skill links may point at explicitly allowed external roots."""
        from spoon_bot.agent.tools.filesystem import ListDirTool, ReadFileTool

        external_root = temp_dir.parent / f"{temp_dir.name}-external-skill-root"
        external_skill = external_root / "spot-agent-cypher"
        external_skill.mkdir(parents=True)
        (external_skill / "SKILL.md").write_text("CLI := node cli/index.js\n", encoding="utf-8")

        skills_dir = temp_dir / "skills"
        skills_dir.mkdir()
        link_path = skills_dir / "spot-agent-cypher"
        try:
            os.symlink(external_skill, link_path, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"symlink creation unavailable: {exc}")

        list_tool = ListDirTool(workspace=temp_dir, additional_read_paths=[external_root])
        read_tool = ReadFileTool(workspace=temp_dir, additional_read_paths=[external_root])

        listed = await list_tool.execute("skills/spot-agent-cypher")
        read = await read_tool.execute("skills/spot-agent-cypher/SKILL.md")

        assert "SKILL.md" in listed
        assert "CLI := node cli/index.js" in read

    @pytest.mark.asyncio
    async def test_list_nonexistent_directory(self, list_tool, temp_dir):
        """Test listing a nonexistent directory within workspace."""
        # Path must be within workspace but nonexistent
        result = await list_tool.execute(path=str(temp_dir / "nonexistent_dir"))
        assert "Error" in result or "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_read_outside_workspace(self):
        """Test that reading outside workspace is blocked."""
        from spoon_bot.agent.tools.filesystem import ReadFileTool

        # Create tool with restricted workspace
        with tempfile.TemporaryDirectory() as workspace:
            tool = ReadFileTool(workspace=workspace)

            # Try to read a file outside workspace
            result = await tool.execute(path="/etc/passwd")
            assert "Security Error" in result or "outside" in result.lower()

    def test_workspace_under_blocklisted_parent_can_read_workspace_files(self, tmp_path):
        """A Linux workspace under /root should not block its own skill files."""
        from pathlib import PurePosixPath

        from spoon_bot.agent.tools.path_validator import PathValidator

        validator = PathValidator(workspace=tmp_path)
        validator._is_windows = False
        validator._workspace = PurePosixPath("/root/.spoon-bot/workspace")
        validator._blocklist = {"/root", "/.ssh"}

        blocked, reason = validator._is_blocked_path(
            PurePosixPath("/root/.spoon-bot/workspace/skills/demo/SKILL.md")
        )

        assert blocked is False
        assert reason is None

    def test_sensitive_child_path_inside_root_workspace_stays_blocked(self, tmp_path):
        """Only the workspace parent prefix is exempted, not sensitive children."""
        from pathlib import PurePosixPath

        from spoon_bot.agent.tools.path_validator import PathValidator

        validator = PathValidator(workspace=tmp_path)
        validator._is_windows = False
        validator._workspace = PurePosixPath("/root/.spoon-bot/workspace")
        validator._blocklist = {"/root", "/.ssh"}

        blocked, reason = validator._is_blocked_path(
            PurePosixPath("/root/.spoon-bot/workspace/.ssh/id_rsa")
        )

        assert blocked is True
        assert "/.ssh" in (reason or "")


class TestToolRegistry:
    """Tests for the ToolRegistry."""

    @pytest.fixture
    def registry(self):
        """Create a ToolRegistry instance with validation disabled for mocks."""
        from spoon_bot.agent.tools.registry import ToolRegistry
        # Disable parameter validation for mock tool tests
        return ToolRegistry(validate_params=False)

    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool."""
        class MockTool:
            name = "test_tool"
            description = "A test tool"
            parameters = {"type": "object", "properties": {}}

            def __init__(self) -> None:
                self.execute = AsyncMock(return_value="test result")
                self.validate_parameters = MagicMock(return_value=[])
                self.to_schema = MagicMock(return_value={
                    "type": "function",
                    "function": {
                        "name": "test_tool",
                        "description": "A test tool",
                        "parameters": {"type": "object", "properties": {}},
                    }
                })

            async def __call__(self, **kwargs):
                return await self.execute(**kwargs)

        tool = MockTool()
        tool.name = "test_tool"
        tool.description = "A test tool"
        tool.parameters = {"type": "object", "properties": {}}
        return tool

    def test_register_tool(self, registry, mock_tool):
        """Test registering a tool."""
        registry.register(mock_tool)
        assert "test_tool" in registry
        assert len(registry) == 1

    def test_get_tool(self, registry, mock_tool):
        """Test getting a registered tool."""
        registry.register(mock_tool)
        retrieved = registry.get("test_tool")
        assert retrieved is mock_tool

    def test_get_nonexistent_tool(self, registry):
        """Test getting a tool that doesn't exist."""
        result = registry.get("nonexistent")
        assert result is None

    def test_unregister_tool(self, registry, mock_tool):
        """Test unregistering a tool."""
        registry.register(mock_tool)
        result = registry.unregister("test_tool")
        assert result is True
        assert "test_tool" not in registry

    def test_list_tools(self, registry, mock_tool):
        """Test listing registered tools."""
        registry.register(mock_tool)
        tools = registry.list_tools()
        assert "test_tool" in tools

    @pytest.mark.asyncio
    async def test_execute_tool(self, registry, mock_tool):
        """Test executing a registered tool."""
        registry.register(mock_tool)
        result = await registry.execute("test_tool", {"arg": "value"})
        assert result == "test result"
        mock_tool.execute.assert_awaited_once_with(arg="value")

    @pytest.mark.asyncio
    async def test_execute_uses_base_tool_invocation_guardrails(self, registry):
        """Registry execution should share Tool.__call__ loop guardrails."""
        from typing import Any

        from spoon_bot.agent.tools.base import Tool
        from spoon_bot.agent.tools.execution_context import track_tool_invocations

        class CountingTool(Tool):
            def __init__(self) -> None:
                self.calls = 0

            @property
            def name(self) -> str:
                return "counting"

            @property
            def description(self) -> str:
                return "Count executions"

            @property
            def parameters(self) -> dict[str, Any]:
                return {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                }

            async def execute(self, **kwargs: Any) -> str:
                self.calls += 1
                return f"call-{self.calls}"

        tool = CountingTool()
        registry.register(tool)

        with track_tool_invocations(max_repeats=1):
            first = await registry.execute("counting", {"value": "same"})
            second = await registry.execute("counting", {"value": "same"})

        assert first == "call-1"
        assert "STOP_TOOL_LOOP" in second
        assert tool.calls == 1

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, registry):
        """Test executing an unknown tool."""
        result = await registry.execute("unknown_tool", {})
        assert "Error" in result or "Unknown" in result

    def test_get_definitions(self, registry, mock_tool):
        """Test getting tool definitions."""
        registry.register(mock_tool)
        definitions = registry.get_definitions()
        assert len(definitions) == 1
        assert definitions[0]["type"] == "function"


class TestRateLimiting:
    """Tests for rate limiting utilities."""

    @pytest.fixture
    def rate_config(self):
        """Create a rate limit config for testing."""
        from spoon_bot.utils.rate_limit import RateLimitConfig
        return RateLimitConfig(
            requests_per_second=10.0,
            requests_per_minute=60.0,
            burst_size=5,
            enabled=True,
        )

    @pytest.mark.asyncio
    async def test_token_bucket_acquire(self, rate_config):
        """Test token bucket limiter acquire."""
        from spoon_bot.utils.rate_limit import TokenBucketLimiter

        limiter = TokenBucketLimiter.from_config(rate_config)

        # Should be able to acquire immediately (within burst capacity)
        result = await limiter.acquire()
        assert result is True

    @pytest.mark.asyncio
    async def test_token_bucket_burst(self, rate_config):
        """Test token bucket burst behavior."""
        from spoon_bot.utils.rate_limit import TokenBucketLimiter

        limiter = TokenBucketLimiter.from_config(rate_config)

        # Should be able to acquire burst_size tokens immediately
        for _ in range(rate_config.burst_size):
            result = await limiter.acquire()
            assert result is True

    @pytest.mark.asyncio
    async def test_sliding_window_acquire(self, rate_config):
        """Test sliding window limiter acquire."""
        from spoon_bot.utils.rate_limit import SlidingWindowLimiter

        limiter = SlidingWindowLimiter.from_config(rate_config)

        # Should be able to acquire immediately
        result = await limiter.acquire()
        assert result is True

    def test_rate_config_presets(self):
        """Test rate limit config presets."""
        from spoon_bot.utils.rate_limit import RateLimitConfig

        llm_config = RateLimitConfig.for_llm_api()
        assert llm_config.enabled is True
        assert llm_config.requests_per_second <= 5  # Conservative limit

        shell_config = RateLimitConfig.for_shell()
        assert shell_config.enabled is True

        unlimited = RateLimitConfig.unlimited()
        assert unlimited.enabled is False

    @pytest.mark.asyncio
    async def test_rate_limiter_reset(self, rate_config):
        """Test rate limiter reset."""
        from spoon_bot.utils.rate_limit import TokenBucketLimiter

        limiter = TokenBucketLimiter.from_config(rate_config)

        # Exhaust some tokens
        for _ in range(3):
            await limiter.acquire()

        # Reset
        limiter.reset()

        # Should have full capacity again
        assert limiter.tokens == rate_config.burst_size


class TestErrorHandling:
    """Tests for error handling utilities."""

    def test_format_user_error_spoon_error(self):
        """Test formatting SpoonBotError."""
        from spoon_bot.utils.errors import SpoonBotError, format_user_error

        error = SpoonBotError(
            "Technical error",
            user_message="Something went wrong. Please try again."
        )
        result = format_user_error(error)
        assert result == "Something went wrong. Please try again."

    def test_format_user_error_config_error(self):
        """Test formatting ConfigurationError."""
        from spoon_bot.utils.errors import ConfigurationError, format_user_error

        error = ConfigurationError(
            "Missing key",
            user_message="API key not configured."
        )
        result = format_user_error(error, include_type=True)
        assert "Configuration Error" in result
        assert "API key not configured" in result

    def test_format_user_error_api_error(self):
        """Test formatting APIError."""
        from spoon_bot.utils.errors import APIError, format_user_error

        error = APIError(
            "HTTP 401",
            status_code=401,
            provider="Anthropic",
        )
        result = format_user_error(error)
        assert "authentication" in result.lower() or "api key" in result.lower()

    def test_format_user_error_rate_limit(self):
        """Test formatting RateLimitExceeded."""
        from spoon_bot.utils.errors import RateLimitExceeded, format_user_error

        error = RateLimitExceeded(
            resource="API",
            limit=60,
            window=60.0,
            retry_after=5.0,
        )
        result = format_user_error(error)
        assert "too many requests" in result.lower() or "wait" in result.lower()

    def test_format_user_error_generic(self):
        """Test formatting generic exceptions."""
        from spoon_bot.utils.errors import format_user_error

        error = ValueError("Invalid input")
        result = format_user_error(error)
        # Should return a user-friendly message, not the raw error
        assert "Invalid" in result or "value" in result.lower()

    def test_error_suggestions(self):
        """Test getting error suggestions."""
        from spoon_bot.utils.errors import get_error_suggestions

        # API key error
        error = ValueError("ANTHROPIC_API_KEY not set")
        suggestions = get_error_suggestions(error)
        assert len(suggestions) > 0
        assert any("key" in s.lower() for s in suggestions)

        # Connection error
        error = ConnectionError("Connection refused")
        suggestions = get_error_suggestions(error)
        assert any("connection" in s.lower() or "internet" in s.lower() for s in suggestions)


class TestToolBase:
    """Tests for the Tool base class."""

    def test_tool_to_schema(self):
        """Test converting a tool to OpenAI schema format."""
        from spoon_bot.agent.tools.base import Tool

        class TestTool(Tool):
            @property
            def name(self) -> str:
                return "test_tool"

            @property
            def description(self) -> str:
                return "A test tool for testing"

            @property
            def parameters(self) -> dict:
                return {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string"},
                    },
                    "required": ["input"],
                }

            async def execute(self, **kwargs) -> str:
                return f"Result: {kwargs.get('input')}"

        tool = TestTool()
        schema = tool.to_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test_tool"
        assert schema["function"]["description"] == "A test tool for testing"
        assert "properties" in schema["function"]["parameters"]

    def test_tool_repr(self):
        """Test tool string representation."""
        from spoon_bot.agent.tools.base import Tool

        class TestTool(Tool):
            @property
            def name(self):
                return "my_tool"

            @property
            def description(self):
                return "desc"

            @property
            def parameters(self):
                return {}

            async def execute(self, **kwargs):
                return ""

        tool = TestTool()
        assert "my_tool" in repr(tool)


class TestWebFetchTool:
    """Tests for generic web fetch guardrails around local executable skills."""

    @pytest.fixture
    def web_fetch_tool(self):
        from spoon_bot.agent.tools.web import WebFetchTool

        return WebFetchTool(timeout=5)

    @pytest.mark.asyncio
    async def test_web_fetch_defers_skill_derived_remote_probe_before_shell(self, web_fetch_tool):
        """A matching local skill should be executed before probing its backend endpoint."""
        from spoon_bot.agent.tools.execution_context import (
            bind_request_execution_hints,
            track_tool_invocations,
        )

        hints = {
            "explicit_request_urls": [],
            "local_executable_skills": [
                {
                    "name": "spot-agent-cypher",
                    "location": "skills/spot-agent-cypher/SKILL.md",
                    "commands": [
                        "node skills/spot-agent-cypher/cli/index.js join A",
                        "node skills/spot-agent-cypher/cli/index.js wallet",
                    ],
                    "urls": ["http://13.251.72.206:8080/api/agent/games"],
                }
            ],
        }

        with bind_request_execution_hints(hints), track_tool_invocations():
            result = await web_fetch_tool.execute("http://13.251.72.206:8080/api/agent/games")

        assert "Deferred remote fetch from a skill-derived endpoint" in result
        assert "spot-agent-cypher" in result
        assert "join A" in result

    @pytest.mark.asyncio
    async def test_web_fetch_allows_same_endpoint_after_shell_progress(self, web_fetch_tool):
        """Once shell work has happened in the request, the same remote endpoint may be fetched."""
        from spoon_bot.agent.tools.base import Tool
        from spoon_bot.agent.tools.execution_context import (
            bind_request_execution_hints,
            track_tool_invocations,
        )

        hints = {
            "explicit_request_urls": [],
            "local_executable_skills": [
                {
                    "name": "spot-agent-cypher",
                    "location": "skills/spot-agent-cypher/SKILL.md",
                    "commands": ["node skills/spot-agent-cypher/cli/index.js join A"],
                    "urls": ["http://13.251.72.206:8080/api/agent/games"],
                }
            ],
        }

        class _StubShell(Tool):
            @property
            def name(self) -> str:
                return "shell"

            @property
            def description(self) -> str:
                return "stub"

            @property
            def parameters(self):
                return {"type": "object", "properties": {}}

            async def execute(self, **kwargs):
                return "ok"

        fake_response = MagicMock()
        fake_response.headers = {"content-type": "application/json"}
        fake_response.text = '{"ok":true}'
        fake_response.json.return_value = {"ok": True}
        fake_response.raise_for_status.return_value = None

        fake_client = AsyncMock()
        fake_client.request = AsyncMock(return_value=fake_response)

        with bind_request_execution_hints(hints), track_tool_invocations():
            await _StubShell()()
            with patch("spoon_bot.agent.tools.web._get_http_client", return_value=fake_client):
                result = await web_fetch_tool.execute("http://13.251.72.206:8080/api/agent/games")

        assert result == '{\n  "ok": true\n}'

    @pytest.mark.asyncio
    async def test_web_fetch_allows_user_explicit_endpoint_before_shell(self, web_fetch_tool):
        """A URL the user supplied directly is not an accidental skill-backend probe."""
        from spoon_bot.agent.tools.execution_context import (
            bind_request_execution_hints,
            track_tool_invocations,
        )

        hints = {
            "explicit_request_urls": ["http://13.251.72.206:8080/api/agent/games"],
            "local_executable_skills": [
                {
                    "name": "spot-agent-cypher",
                    "location": "skills/spot-agent-cypher/SKILL.md",
                    "commands": ["node skills/spot-agent-cypher/cli/index.js join A"],
                    "urls": ["http://13.251.72.206:8080/api/agent/games"],
                }
            ],
        }

        fake_response = MagicMock()
        fake_response.headers = {"content-type": "application/json"}
        fake_response.text = '{"ok":true}'
        fake_response.json.return_value = {"ok": True}
        fake_response.raise_for_status.return_value = None

        fake_client = AsyncMock()
        fake_client.request = AsyncMock(return_value=fake_response)

        with bind_request_execution_hints(hints), track_tool_invocations():
            with patch("spoon_bot.agent.tools.web._get_http_client", return_value=fake_client):
                result = await web_fetch_tool.execute("http://13.251.72.206:8080/api/agent/games")

        assert result == '{\n  "ok": true\n}'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
