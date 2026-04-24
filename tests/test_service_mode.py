from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from spoon_bot.cli import app
from spoon_bot.services import daemon

runner = CliRunner()


def test_http_gateway_command_uses_uvicorn_and_config_env(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("cron:\n  enabled: true\n", encoding="utf-8")

    command = daemon._get_command(daemon.ServiceMode.HTTP_GATEWAY, config)
    env_overrides = daemon._get_env_overrides(daemon.ServiceMode.HTTP_GATEWAY, config)

    assert command == [
        sys.executable,
        "-m",
        "uvicorn",
        "spoon_bot.gateway.server:create_app",
        "--factory",
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
    ]
    assert env_overrides["GATEWAY_HOST"] == "127.0.0.1"
    assert env_overrides["GATEWAY_PORT"] == "8080"
    assert env_overrides["SPOON_BOT_CONFIG"] == str(config.resolve())


def test_windows_manual_start_uses_create_no_window(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("cron:\n  enabled: true\n", encoding="utf-8")
    monkeypatch.setattr(daemon, "SPOON_BOT_DIR", tmp_path)
    monkeypatch.setattr(daemon.platform, "system", lambda: "Windows")
    monkeypatch.setattr(daemon, "_get_windows_supervisor", lambda mode: daemon.SUPERVISOR_NONE)
    monkeypatch.setattr(daemon, "_wait_for_http_gateway", lambda timeout_sec=10.0: True)
    monkeypatch.setattr(daemon, "_find_windows_listener_pid", lambda port: None)

    captured: dict[str, object] = {}

    class FakeProc:
        pid = 4242

        @staticmethod
        def poll():
            return None

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return FakeProc()

    monkeypatch.setattr(daemon.subprocess, "Popen", fake_popen)

    ok, msg = daemon.start(config=config, mode=daemon.ServiceMode.HTTP_GATEWAY)

    assert ok is True
    assert "manual detached mode" in msg
    assert "http-gateway.log" in msg
    assert daemon._get_pid_file(daemon.ServiceMode.HTTP_GATEWAY).read_text() == "4242"
    assert daemon._get_log_file(daemon.ServiceMode.HTTP_GATEWAY).exists()
    assert captured["cmd"] == [
        "cmd.exe",
        "/d",
        "/s",
        "/c",
        str(daemon._get_windows_service_script(daemon.ServiceMode.HTTP_GATEWAY)),
    ]
    creationflags = captured["kwargs"]["creationflags"]
    assert creationflags & daemon.WINDOWS_CREATE_NO_WINDOW
    assert creationflags & daemon.WINDOWS_DETACHED_PROCESS


def test_windows_manual_start_treats_slow_http_gateway_as_starting(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("cron:\n  enabled: true\n", encoding="utf-8")
    monkeypatch.setattr(daemon, "SPOON_BOT_DIR", tmp_path)
    monkeypatch.setattr(daemon.platform, "system", lambda: "Windows")
    monkeypatch.setattr(daemon, "_get_windows_supervisor", lambda mode: daemon.SUPERVISOR_NONE)
    monkeypatch.setattr(daemon, "_find_windows_listener_pid", lambda port: None)

    waits = iter([False, False])
    monkeypatch.setattr(
        daemon,
        "_wait_for_http_gateway",
        lambda timeout_sec=daemon.HTTP_GATEWAY_STARTUP_TIMEOUT_SEC: next(waits),
    )

    class FakeProc:
        pid = 5150

        @staticmethod
        def poll():
            return None

    monkeypatch.setattr(daemon.subprocess, "Popen", lambda cmd, **kwargs: FakeProc())

    ok, msg = daemon.start(config=config, mode=daemon.ServiceMode.HTTP_GATEWAY)

    assert ok is True
    assert "manual detached mode" in msg
    assert "Startup is still in progress" in msg


def test_windows_manual_start_reports_real_http_gateway_exit(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("cron:\n  enabled: true\n", encoding="utf-8")
    monkeypatch.setattr(daemon, "SPOON_BOT_DIR", tmp_path)
    monkeypatch.setattr(daemon.platform, "system", lambda: "Windows")
    monkeypatch.setattr(daemon, "_get_windows_supervisor", lambda mode: daemon.SUPERVISOR_NONE)
    monkeypatch.setattr(daemon, "_find_windows_listener_pid", lambda port: None)
    monkeypatch.setattr(
        daemon,
        "_wait_for_http_gateway",
        lambda timeout_sec=daemon.HTTP_GATEWAY_STARTUP_TIMEOUT_SEC: False,
    )

    class FakeProc:
        pid = 6262

        @staticmethod
        def poll():
            return 1

    monkeypatch.setattr(daemon.subprocess, "Popen", lambda cmd, **kwargs: FakeProc())

    ok, msg = daemon.start(config=config, mode=daemon.ServiceMode.HTTP_GATEWAY)

    assert ok is False
    assert "exited before binding http://127.0.0.1:8080" in msg
    assert "Check logs:" in msg


def test_windows_start_uses_schtasks_when_installed(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("cron:\n  enabled: true\n", encoding="utf-8")
    monkeypatch.setattr(daemon, "SPOON_BOT_DIR", tmp_path)
    monkeypatch.setattr(daemon.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        daemon,
        "_get_windows_supervisor",
        lambda mode: daemon.WINDOWS_SUPERVISOR_SCHTASKS,
    )
    monkeypatch.setattr(daemon, "_wait_for_http_gateway", lambda timeout_sec=10.0: True)

    calls: list[list[str]] = []

    def fake_run(args):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(daemon, "_run_windows_schtasks", fake_run)

    ok, msg = daemon.start(config=config, mode=daemon.ServiceMode.HTTP_GATEWAY)

    assert ok is True
    assert "Windows Task Scheduler" in msg
    assert calls == [["/Run", "/TN", "SpoonBot HTTP Gateway"]]
    assert daemon._get_windows_service_script(daemon.ServiceMode.HTTP_GATEWAY).exists()


def test_windows_install_prefers_schtasks(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("cron:\n  enabled: true\n", encoding="utf-8")
    monkeypatch.setattr(daemon, "SPOON_BOT_DIR", tmp_path)
    monkeypatch.setattr(daemon.platform, "system", lambda: "Windows")
    calls: list[list[str]] = []

    def fake_run(args):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(daemon, "_run_windows_schtasks", fake_run)

    ok, msg = daemon.install_auto_start(config=config, mode=daemon.ServiceMode.HTTP_GATEWAY)

    assert ok is True
    assert "Windows Task Scheduler task" in msg
    assert daemon._get_windows_service_script(daemon.ServiceMode.HTTP_GATEWAY).exists()
    assert not daemon._get_windows_startup_launcher(daemon.ServiceMode.HTTP_GATEWAY).exists()
    assert calls == [[
        "/Create",
        "/TN",
        "SpoonBot HTTP Gateway",
        "/TR",
        subprocess.list2cmdline(
            [
                "cmd.exe",
                "/d",
                "/c",
                str(daemon._get_windows_service_script(daemon.ServiceMode.HTTP_GATEWAY)),
            ]
        ),
        "/SC",
        "ONLOGON",
        "/F",
    ]]


def test_windows_install_falls_back_to_startup_launcher_on_access_denied(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("cron:\n  enabled: true\n", encoding="utf-8")
    startup_dir = tmp_path / "startup"
    monkeypatch.setattr(daemon, "SPOON_BOT_DIR", tmp_path)
    monkeypatch.setattr(daemon.platform, "system", lambda: "Windows")
    monkeypatch.setattr(daemon, "_get_windows_startup_dir", lambda: startup_dir)
    monkeypatch.setattr(
        daemon,
        "_run_windows_schtasks",
        lambda args: SimpleNamespace(returncode=1, stdout="", stderr="ERROR: Access is denied."),
    )

    ok, msg = daemon.install_auto_start(config=config, mode=daemon.ServiceMode.HTTP_GATEWAY)

    assert ok is True
    assert "Startup-folder fallback" in msg
    launcher = daemon._get_windows_startup_launcher(daemon.ServiceMode.HTTP_GATEWAY)
    assert launcher.exists()
    assert 'start "" /min cmd.exe /d /c' in launcher.read_text(encoding="utf-8")


def test_windows_status_reports_supervisor_and_url(tmp_path, monkeypatch):
    monkeypatch.setattr(daemon, "SPOON_BOT_DIR", tmp_path)
    monkeypatch.setattr(daemon.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        daemon,
        "_get_windows_supervisor",
        lambda mode: daemon.WINDOWS_SUPERVISOR_SCHTASKS,
    )
    monkeypatch.setattr(
        daemon,
        "_query_windows_task",
        lambda mode: SimpleNamespace(
            returncode=0,
            stdout="Status: Running\nLast Run Result: 0x0\n",
            stderr="",
        ),
    )
    monkeypatch.setattr(
        daemon,
        "_is_http_gateway_listening",
        lambda host="127.0.0.1", port=8080: True,
    )
    monkeypatch.setattr(daemon, "_find_windows_listener_pid", lambda port: 9001)
    monkeypatch.setattr(
        daemon,
        "_fetch_gateway_health",
        lambda: {
            "status": "degraded",
            "checks": [
                {
                    "name": "channels",
                    "status": "unhealthy",
                    "message": "0/1 running | telegram:Irene_spoon_bot: httpx.ConnectError",
                }
            ],
        },
    )

    info = daemon.get_status(mode=daemon.ServiceMode.HTTP_GATEWAY)

    assert info["supervisor"] == daemon.WINDOWS_SUPERVISOR_SCHTASKS
    assert info["installed"] is True
    assert info["running"] is True
    assert info["pid"] == 9001
    assert info["task_name"] == "SpoonBot HTTP Gateway"
    assert info["task_status"] == "Running"
    assert info["url"] == "http://127.0.0.1:8080"
    assert info["gateway_health"] == "degraded"
    assert info["channels_health"] == "unhealthy"
    assert "telegram:Irene_spoon_bot" in info["channels_message"]


def test_service_start_command_passes_mode(monkeypatch, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("cron:\n  enabled: true\n", encoding="utf-8")
    calls: dict[str, object] = {}

    def fake_start(config: Path | None = None, mode=daemon.DEFAULT_SERVICE_MODE):
        calls["config"] = config
        calls["mode"] = mode
        return True, "started"

    monkeypatch.setattr("spoon_bot.services.daemon.start", fake_start)

    result = runner.invoke(
        app,
        ["service", "start", "--mode", "http-gateway", "--config", str(config)],
    )

    assert result.exit_code == 0
    assert calls["mode"] == daemon.ServiceMode.HTTP_GATEWAY
    assert calls["config"] == config


def test_service_status_command_passes_mode(monkeypatch):
    calls: dict[str, object] = {}

    def fake_status(mode=daemon.DEFAULT_SERVICE_MODE):
        calls["mode"] = mode
        return {
            "mode": daemon.ServiceMode.HTTP_GATEWAY.value,
            "display_name": "HTTP gateway service",
            "running": False,
            "pid": None,
            "log_file": "http-gateway.log",
            "pid_file": "service-http-gateway.pid",
            "auto_start": False,
            "installed": False,
            "supervisor": daemon.SUPERVISOR_NONE,
            "url": "http://127.0.0.1:8080",
        }

    monkeypatch.setattr("spoon_bot.services.daemon.get_status", fake_status)

    result = runner.invoke(app, ["service", "status", "--mode", "http-gateway"])

    assert result.exit_code == 0
    assert calls["mode"] == daemon.ServiceMode.HTTP_GATEWAY
    assert "http-gateway" in result.output
    assert "manual-detached" not in result.output
    assert "http://127.0.0.1:8080" in result.output


def test_service_stop_and_restart_commands_pass_mode(monkeypatch, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("cron:\n  enabled: true\n", encoding="utf-8")
    stop_calls: dict[str, object] = {}
    restart_calls: dict[str, object] = {}

    def fake_stop(mode=daemon.DEFAULT_SERVICE_MODE):
        stop_calls["mode"] = mode
        return True, "stopped"

    def fake_restart(config: Path | None = None, mode=daemon.DEFAULT_SERVICE_MODE):
        restart_calls["config"] = config
        restart_calls["mode"] = mode
        return True, "restarted"

    monkeypatch.setattr("spoon_bot.services.daemon.stop", fake_stop)
    monkeypatch.setattr("spoon_bot.services.daemon.restart", fake_restart)

    stop_result = runner.invoke(app, ["service", "stop", "--mode", "http-gateway"])
    restart_result = runner.invoke(
        app,
        ["service", "restart", "--mode", "http-gateway", "--config", str(config)],
    )

    assert stop_result.exit_code == 0
    assert restart_result.exit_code == 0
    assert stop_calls["mode"] == daemon.ServiceMode.HTTP_GATEWAY
    assert restart_calls["mode"] == daemon.ServiceMode.HTTP_GATEWAY
    assert restart_calls["config"] == config


def test_service_logs_command_passes_mode(monkeypatch):
    calls: dict[str, object] = {}

    def fake_logs(lines: int = 50, follow: bool = False, mode=daemon.DEFAULT_SERVICE_MODE):
        calls["lines"] = lines
        calls["follow"] = follow
        calls["mode"] = mode

    monkeypatch.setattr("spoon_bot.services.daemon.tail_logs", fake_logs)

    result = runner.invoke(
        app,
        ["service", "logs", "--mode", "http-gateway", "--lines", "25", "--follow"],
    )

    assert result.exit_code == 0
    assert calls["mode"] == daemon.ServiceMode.HTTP_GATEWAY
    assert calls["lines"] == 25
    assert calls["follow"] is True


def test_service_install_and_uninstall_commands_pass_mode(monkeypatch, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("cron:\n  enabled: true\n", encoding="utf-8")
    install_calls: dict[str, object] = {}
    uninstall_calls: dict[str, object] = {}

    def fake_install(config: Path | None = None, mode=daemon.DEFAULT_SERVICE_MODE):
        install_calls["config"] = config
        install_calls["mode"] = mode
        return True, "installed"

    def fake_uninstall(mode=daemon.DEFAULT_SERVICE_MODE):
        uninstall_calls["mode"] = mode
        return True, "uninstalled"

    monkeypatch.setattr("spoon_bot.services.daemon.install_auto_start", fake_install)
    monkeypatch.setattr("spoon_bot.services.daemon.uninstall_auto_start", fake_uninstall)

    install_result = runner.invoke(
        app,
        ["service", "install", "--mode", "http-gateway", "--config", str(config)],
    )
    uninstall_result = runner.invoke(
        app,
        ["service", "uninstall", "--mode", "http-gateway"],
    )

    assert install_result.exit_code == 0
    assert uninstall_result.exit_code == 0
    assert install_calls["mode"] == daemon.ServiceMode.HTTP_GATEWAY
    assert install_calls["config"] == config
    assert uninstall_calls["mode"] == daemon.ServiceMode.HTTP_GATEWAY
