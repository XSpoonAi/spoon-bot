"""Cross-platform background service management for spoon-bot.

Provides start/stop/restart/status/logs commands to run spoon-bot services
as persistent background processes without Docker.

Supports:
  - Manual start/stop via PID file (all platforms)
  - Auto-start at login:
      Windows  -> Task Scheduler (schtasks, no admin required)
      Linux    -> systemd user service (systemctl --user)
      macOS    -> launchd agent (~/Library/LaunchAgents)
"""

from __future__ import annotations

import json
import os
import platform
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import urlopen

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SPOON_BOT_DIR = Path.home() / ".spoon-bot"
DEFAULT_CONFIG_PATH = SPOON_BOT_DIR / "config.yaml"

WINDOWS_SUPERVISOR_SCHTASKS = "schtasks"
WINDOWS_SUPERVISOR_STARTUP = "startup-folder"
SUPERVISOR_MANUAL = "manual-detached"
SUPERVISOR_SYSTEMD = "systemd"
SUPERVISOR_LAUNCHD = "launchd"
SUPERVISOR_NONE = "none"

WINDOWS_DETACHED_PROCESS = 0x00000008
WINDOWS_CREATE_NEW_PROCESS_GROUP = 0x00000200
WINDOWS_CREATE_NO_WINDOW = 0x08000000
HTTP_GATEWAY_STARTUP_TIMEOUT_SEC = 30.0
HTTP_GATEWAY_STARTUP_GRACE_SEC = 5.0


class ServiceMode(str, Enum):
    """Supported background service modes."""

    GATEWAY = "gateway"
    HTTP_GATEWAY = "http-gateway"


DEFAULT_SERVICE_MODE = ServiceMode.GATEWAY


@dataclass(frozen=True)
class ServiceSpec:
    """Runtime metadata for a service mode."""

    mode: ServiceMode
    display_name: str
    description: str
    pid_filename: str
    log_filename: str
    windows_task_name: str
    windows_script_filename: str
    linux_service_name: str
    macos_label: str


_SERVICE_SPECS: dict[ServiceMode, ServiceSpec] = {
    ServiceMode.GATEWAY: ServiceSpec(
        mode=ServiceMode.GATEWAY,
        display_name="Gateway service",
        description="SpoonBot AI Agent Gateway",
        pid_filename="service.pid",
        log_filename="gateway.log",
        windows_task_name="SpoonBot Gateway",
        windows_script_filename="gateway-service.cmd",
        linux_service_name="spoon-bot",
        macos_label="com.xspoon.spoon-bot",
    ),
    ServiceMode.HTTP_GATEWAY: ServiceSpec(
        mode=ServiceMode.HTTP_GATEWAY,
        display_name="HTTP gateway service",
        description="SpoonBot HTTP Gateway",
        pid_filename="service-http-gateway.pid",
        log_filename="http-gateway.log",
        windows_task_name="SpoonBot HTTP Gateway",
        windows_script_filename="http-gateway-service.cmd",
        linux_service_name="spoon-bot-http-gateway",
        macos_label="com.xspoon.spoon-bot.http-gateway",
    ),
}


# ---------------------------------------------------------------------------
# Service metadata helpers
# ---------------------------------------------------------------------------

def normalize_mode(mode: ServiceMode | str = DEFAULT_SERVICE_MODE) -> ServiceMode:
    """Normalize a caller-provided mode string to ``ServiceMode``."""
    if isinstance(mode, ServiceMode):
        return mode
    return ServiceMode(mode)


def _get_service_spec(mode: ServiceMode | str = DEFAULT_SERVICE_MODE) -> ServiceSpec:
    """Return the metadata for *mode*."""
    return _SERVICE_SPECS[normalize_mode(mode)]


def _get_pid_file(mode: ServiceMode | str = DEFAULT_SERVICE_MODE) -> Path:
    """Return the PID file path for *mode*."""
    return SPOON_BOT_DIR / _get_service_spec(mode).pid_filename


def _get_log_file(mode: ServiceMode | str = DEFAULT_SERVICE_MODE) -> Path:
    """Return the log file path for *mode*."""
    return SPOON_BOT_DIR / _get_service_spec(mode).log_filename


def _get_windows_service_script(mode: ServiceMode | str = DEFAULT_SERVICE_MODE) -> Path:
    """Return the generated Windows launcher script path for *mode*."""
    spec = _get_service_spec(mode)
    return SPOON_BOT_DIR / spec.windows_script_filename


def _linux_service_file(mode: ServiceMode | str = DEFAULT_SERVICE_MODE) -> Path:
    """Return the systemd user service path for *mode*."""
    spec = _get_service_spec(mode)
    return Path.home() / ".config" / "systemd" / "user" / f"{spec.linux_service_name}.service"


def _macos_plist_file(mode: ServiceMode | str = DEFAULT_SERVICE_MODE) -> Path:
    """Return the launchd plist path for *mode*."""
    spec = _get_service_spec(mode)
    return Path.home() / "Library" / "LaunchAgents" / f"{spec.macos_label}.plist"


def _resolve_config_path(config: Optional[Path] = None) -> Path | None:
    """Resolve a config path using service-mode conventions."""
    if config and config.exists():
        return config.expanduser().resolve()
    if DEFAULT_CONFIG_PATH.exists():
        return DEFAULT_CONFIG_PATH.expanduser().resolve()
    return None


def _resolve_service_workdir(config: Optional[Path] = None) -> Path:
    """Choose a stable working directory for launcher scripts."""
    resolved_config = _resolve_config_path(config)
    if resolved_config is not None:
        return resolved_config.parent
    return Path.home()


# ---------------------------------------------------------------------------
# Executable detection / command building
# ---------------------------------------------------------------------------

def _get_command(
    mode: ServiceMode | str = DEFAULT_SERVICE_MODE,
    config: Optional[Path] = None,
) -> list[str]:
    """Build the full command list for starting the selected service."""
    service_mode = normalize_mode(mode)
    resolved_config = _resolve_config_path(config)

    if service_mode is ServiceMode.GATEWAY:
        exe = shutil.which("spoon-bot")
        if exe:
            cmd = [exe, "gateway"]
        else:
            # Editable / development install fallback
            cmd = [sys.executable, "-m", "spoon_bot", "gateway"]

        if resolved_config is not None:
            cmd += ["--config", str(resolved_config)]
        return cmd

    return [
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


def _get_env_overrides(
    mode: ServiceMode | str = DEFAULT_SERVICE_MODE,
    config: Optional[Path] = None,
) -> dict[str, str]:
    """Return extra environment variables needed for *mode*."""
    service_mode = normalize_mode(mode)
    resolved_config = _resolve_config_path(config)
    overrides: dict[str, str] = {}

    if service_mode is ServiceMode.HTTP_GATEWAY:
        overrides["GATEWAY_HOST"] = "127.0.0.1"
        overrides["GATEWAY_PORT"] = "8080"
        if resolved_config is not None:
            overrides["SPOON_BOT_CONFIG"] = str(resolved_config)

    return overrides


def _get_start_environment(
    mode: ServiceMode | str = DEFAULT_SERVICE_MODE,
    config: Optional[Path] = None,
) -> dict[str, str]:
    """Return the child-process environment for starting *mode*."""
    env = os.environ.copy()
    env.update(_get_env_overrides(mode, config))
    return env


def _sanitize_windows_filename(value: str) -> str:
    return value.replace("\\", "_").replace("/", "_").replace(":", "_")


def _get_windows_startup_dir() -> Path:
    app_data = os.environ.get("APPDATA")
    if app_data:
        return Path(app_data) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"

    user_profile = os.environ.get("USERPROFILE") or os.environ.get("HOME")
    if not user_profile:
        raise RuntimeError("Windows startup folder unavailable: APPDATA/USERPROFILE not set")

    return (
        Path(user_profile)
        / "AppData"
        / "Roaming"
        / "Microsoft"
        / "Windows"
        / "Start Menu"
        / "Programs"
        / "Startup"
    )


def _get_windows_startup_launcher(mode: ServiceMode | str = DEFAULT_SERVICE_MODE) -> Path:
    spec = _get_service_spec(mode)
    return _get_windows_startup_dir() / f"{_sanitize_windows_filename(spec.windows_task_name)}.cmd"


def _render_windows_service_script(
    mode: ServiceMode | str = DEFAULT_SERVICE_MODE,
    config: Optional[Path] = None,
) -> str:
    """Render the .cmd script that launches the selected service."""
    spec = _get_service_spec(mode)
    workdir = _resolve_service_workdir(config)
    env_overrides = _get_env_overrides(mode, config)
    command = subprocess.list2cmdline(_get_command(mode, config))

    lines = [
        "@echo off",
        f"rem {spec.description}",
        f'cd /d "{workdir}"',
    ]
    for key, value in env_overrides.items():
        lines.append(f'set "{key}={value}"')
    lines.append(command)
    return "\r\n".join(lines) + "\r\n"


def _build_windows_startup_launcher(
    mode: ServiceMode | str = DEFAULT_SERVICE_MODE,
    script_path: Path | None = None,
) -> str:
    """Render the Startup-folder launcher for the selected service."""
    spec = _get_service_spec(mode)
    target = script_path or _get_windows_service_script(mode)
    lines = [
        "@echo off",
        f"rem {spec.description}",
        f'start "" /min cmd.exe /d /c "{target}"',
    ]
    return "\r\n".join(lines) + "\r\n"


def _write_windows_service_script(
    mode: ServiceMode | str = DEFAULT_SERVICE_MODE,
    config: Optional[Path] = None,
) -> Path:
    """Write and return the Windows launcher script path for *mode*."""
    script_path = _get_windows_service_script(mode)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(_render_windows_service_script(mode, config), encoding="utf-8")
    return script_path


def _write_windows_startup_launcher(
    mode: ServiceMode | str = DEFAULT_SERVICE_MODE,
    config: Optional[Path] = None,
) -> Path:
    """Write and return the Windows Startup-folder launcher for *mode*."""
    script_path = _write_windows_service_script(mode, config)
    launcher_path = _get_windows_startup_launcher(mode)
    launcher_path.parent.mkdir(parents=True, exist_ok=True)
    launcher_path.write_text(
        _build_windows_startup_launcher(mode, script_path),
        encoding="utf-8",
    )
    return launcher_path


def _run_windows_schtasks(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run schtasks with captured output."""
    return subprocess.run(
        ["schtasks", *args],
        capture_output=True,
        text=True,
    )


def _windows_task_exists(mode: ServiceMode | str = DEFAULT_SERVICE_MODE) -> bool:
    spec = _get_service_spec(mode)
    result = _run_windows_schtasks(["/Query", "/TN", spec.windows_task_name])
    return result.returncode == 0


def _query_windows_task(mode: ServiceMode | str = DEFAULT_SERVICE_MODE) -> subprocess.CompletedProcess[str]:
    spec = _get_service_spec(mode)
    return _run_windows_schtasks(["/Query", "/TN", spec.windows_task_name, "/V", "/FO", "LIST"])


def _parse_windows_task_status(output: str) -> str | None:
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line.lower().startswith("status:"):
            return line.split(":", 1)[1].strip()
    return None


def _parse_windows_task_last_result(output: str) -> str | None:
    for raw_line in output.splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        if lowered.startswith("last run result:") or lowered.startswith("last result:"):
            return line.split(":", 1)[1].strip()
    return None


def _should_fallback_to_startup_launcher(detail: str) -> bool:
    lowered = detail.lower()
    return "access is denied" in lowered or "denied" in lowered or "0x80070005" in lowered


def _get_windows_supervisor(mode: ServiceMode | str = DEFAULT_SERVICE_MODE) -> str:
    if _windows_task_exists(mode):
        return WINDOWS_SUPERVISOR_SCHTASKS
    if _get_windows_startup_launcher(mode).exists():
        return WINDOWS_SUPERVISOR_STARTUP
    pid = _read_pid(mode)
    if pid is not None and _pid_alive(pid):
        return SUPERVISOR_MANUAL
    return SUPERVISOR_NONE


def _is_http_gateway_listening(host: str = "127.0.0.1", port: int = 8080) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _wait_for_http_gateway(timeout_sec: float = HTTP_GATEWAY_STARTUP_TIMEOUT_SEC) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if _is_http_gateway_listening():
            return True
        time.sleep(0.25)
    return _is_http_gateway_listening()


def _find_windows_listener_pid(port: int) -> Optional[int]:
    try:
        result = subprocess.run(
            ["netstat", "-ano", "-p", "tcp"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    target_suffix = f":{port}"
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line or "LISTENING" not in line.upper():
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        local_address = parts[1]
        pid_text = parts[-1]
        if not local_address.endswith(target_suffix):
            continue
        try:
            return int(pid_text)
        except ValueError:
            continue
    return None


def _terminate_windows_pid(pid: int) -> bool:
    try:
        result = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0 or not _pid_alive(pid)
    except (OSError, subprocess.TimeoutExpired):
        return False


def _start_windows_via_schtasks(
    mode: ServiceMode | str = DEFAULT_SERVICE_MODE,
    config: Optional[Path] = None,
) -> tuple[bool, str]:
    service_mode = normalize_mode(mode)
    spec = _get_service_spec(service_mode)
    _write_windows_service_script(service_mode, config)
    _get_pid_file(service_mode).unlink(missing_ok=True)

    result = _run_windows_schtasks(["/Run", "/TN", spec.windows_task_name])
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        return False, f"Failed to start {spec.display_name.lower()} via Task Scheduler: {detail}"

    if service_mode is ServiceMode.HTTP_GATEWAY and not _wait_for_http_gateway():
        return False, (
            f"{spec.display_name} was launched via Task Scheduler but did not bind "
            "http://127.0.0.1:8080 in time."
        )

    return True, (
        f"{spec.display_name} started via Windows Task Scheduler: '{spec.windows_task_name}'\n"
        f"Logs: {_get_log_file(service_mode)}"
    )


def _start_windows_detached(
    mode: ServiceMode | str = DEFAULT_SERVICE_MODE,
    config: Optional[Path] = None,
    *,
    reason: str,
) -> tuple[bool, str]:
    service_mode = normalize_mode(mode)
    spec = _get_service_spec(service_mode)
    pid_file = _get_pid_file(service_mode)
    log_file = _get_log_file(service_mode)
    script_path = _write_windows_service_script(service_mode, config)
    SPOON_BOT_DIR.mkdir(parents=True, exist_ok=True)

    kwargs: dict[str, object] = {
        "stdin": subprocess.DEVNULL,
        "env": _get_start_environment(service_mode, config),
        "creationflags": (
            WINDOWS_DETACHED_PROCESS
            | WINDOWS_CREATE_NEW_PROCESS_GROUP
            | WINDOWS_CREATE_NO_WINDOW
        ),
        "close_fds": True,
    }

    try:
        with open(log_file, "a", encoding="utf-8") as log_fh:
            kwargs["stdout"] = log_fh
            kwargs["stderr"] = log_fh
            proc = subprocess.Popen(  # noqa: S603
                ["cmd.exe", "/d", "/s", "/c", str(script_path)],
                **kwargs,
            )
        pid_file.write_text(str(proc.pid))

        if service_mode is ServiceMode.HTTP_GATEWAY:
            if not _wait_for_http_gateway():
                if proc.poll() is not None:
                    return False, (
                        f"{spec.display_name} exited before binding http://127.0.0.1:8080 "
                        f"in {reason}. Check logs: {log_file}"
                    )
                if not _wait_for_http_gateway(timeout_sec=HTTP_GATEWAY_STARTUP_GRACE_SEC):
                    if proc.poll() is not None:
                        return False, (
                            f"{spec.display_name} exited before binding http://127.0.0.1:8080 "
                            f"in {reason}. Check logs: {log_file}"
                        )
                    return True, (
                        f"{spec.display_name} started in {reason} (PID: {proc.pid})\n"
                        f"Logs: {log_file}\n"
                        "Startup is still in progress. Check "
                        "'spoon-bot service status --mode http-gateway' if "
                        "http://127.0.0.1:8080 is not reachable yet."
                    )
        else:
            time.sleep(0.5)
            if not _pid_alive(proc.pid):
                return False, f"{spec.display_name} exited immediately after launch in {reason}."

        return True, (
            f"{spec.display_name} started in {reason} (PID: {proc.pid})\n"
            f"Logs: {log_file}"
        )
    except FileNotFoundError:
        return False, f"Cannot find required executable. Script: {script_path}"
    except Exception as exc:  # noqa: BLE001
        return False, f"Failed to start {spec.display_name.lower()} in {reason}: {exc}"


# ---------------------------------------------------------------------------
# PID helpers
# ---------------------------------------------------------------------------

def _read_pid(mode: ServiceMode | str = DEFAULT_SERVICE_MODE) -> Optional[int]:
    pid_file = _get_pid_file(mode)
    if not pid_file.exists():
        return None
    try:
        return int(pid_file.read_text().strip())
    except (ValueError, OSError):
        return None


def _pid_alive(pid: int) -> bool:
    """Return True if a process with *pid* is currently running."""
    try:
        if platform.system() == "Windows":
            import ctypes

            synchronize = 0x00100000
            handle = ctypes.windll.kernel32.OpenProcess(synchronize, False, pid)
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _clear_stale_pid(mode: ServiceMode | str = DEFAULT_SERVICE_MODE) -> None:
    pid_file = _get_pid_file(mode)
    pid = _read_pid(mode)
    if pid is not None and not _pid_alive(pid):
        pid_file.unlink(missing_ok=True)


def _get_runtime_pid(mode: ServiceMode | str = DEFAULT_SERVICE_MODE) -> Optional[int]:
    service_mode = normalize_mode(mode)
    _clear_stale_pid(service_mode)
    pid = _read_pid(service_mode)
    if pid is not None and _pid_alive(pid):
        return pid

    if platform.system() == "Windows" and service_mode is ServiceMode.HTTP_GATEWAY:
        return _find_windows_listener_pid(8080)

    return None


def _detect_supervisor(mode: ServiceMode | str = DEFAULT_SERVICE_MODE) -> str:
    service_mode = normalize_mode(mode)
    system = platform.system()
    if system == "Windows":
        return _get_windows_supervisor(service_mode)
    if system == "Linux" and _linux_service_file(service_mode).exists():
        return SUPERVISOR_SYSTEMD
    if system == "Darwin" and _macos_plist_file(service_mode).exists():
        return SUPERVISOR_LAUNCHD

    pid = _read_pid(service_mode)
    if pid is not None and _pid_alive(pid):
        return SUPERVISOR_MANUAL
    return SUPERVISOR_NONE


# ---------------------------------------------------------------------------
# Public API - process management (all platforms)
# ---------------------------------------------------------------------------

def is_running(mode: ServiceMode | str = DEFAULT_SERVICE_MODE) -> bool:
    """Return True if the service process is alive."""
    service_mode = normalize_mode(mode)
    pid = _get_runtime_pid(service_mode)
    if pid is not None and _pid_alive(pid):
        return True
    if platform.system() == "Windows" and service_mode is ServiceMode.HTTP_GATEWAY:
        return _is_http_gateway_listening()
    return False


def start(
    config: Optional[Path] = None,
    mode: ServiceMode | str = DEFAULT_SERVICE_MODE,
) -> tuple[bool, str]:
    """Start the selected service in the background."""
    service_mode = normalize_mode(mode)
    spec = _get_service_spec(service_mode)

    if is_running(service_mode):
        pid = _get_runtime_pid(service_mode)
        if pid is not None:
            return False, f"{spec.display_name} is already running (PID: {pid})"
        return False, f"{spec.display_name} is already running"

    SPOON_BOT_DIR.mkdir(parents=True, exist_ok=True)

    if platform.system() == "Windows":
        supervisor = _get_windows_supervisor(service_mode)
        if supervisor == WINDOWS_SUPERVISOR_SCHTASKS:
            return _start_windows_via_schtasks(service_mode, config)
        if supervisor == WINDOWS_SUPERVISOR_STARTUP:
            return _start_windows_detached(
                service_mode,
                config,
                reason="startup-folder fallback mode",
            )
        return _start_windows_detached(
            service_mode,
            config,
            reason="manual detached mode",
        )

    pid_file = _get_pid_file(service_mode)
    log_file = _get_log_file(service_mode)
    cmd = _get_command(service_mode, config)
    env = _get_start_environment(service_mode, config)
    kwargs: dict[str, object] = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "stdin": subprocess.DEVNULL,
        "env": env,
        "start_new_session": True,
    }

    try:
        with open(log_file, "a", encoding="utf-8") as log_fh:
            kwargs["stdout"] = log_fh
            kwargs["stderr"] = log_fh
            proc = subprocess.Popen(cmd, **kwargs)  # noqa: S603
        pid_file.write_text(str(proc.pid))
        return True, f"{spec.display_name} started (PID: {proc.pid})\nLogs: {log_file}"
    except FileNotFoundError:
        return False, f"Cannot find required executable. Command: {' '.join(cmd)}"
    except Exception as exc:  # noqa: BLE001
        return False, f"Failed to start {spec.display_name.lower()}: {exc}"


def stop(mode: ServiceMode | str = DEFAULT_SERVICE_MODE) -> tuple[bool, str]:
    """Stop the running service gracefully."""
    service_mode = normalize_mode(mode)
    spec = _get_service_spec(service_mode)
    pid_file = _get_pid_file(service_mode)
    supervisor = _detect_supervisor(service_mode)
    pid = _get_runtime_pid(service_mode)

    if platform.system() == "Windows":
        stopped = False
        details: list[str] = []

        if supervisor == WINDOWS_SUPERVISOR_SCHTASKS:
            result = _run_windows_schtasks(["/End", "/TN", spec.windows_task_name])
            if result.returncode == 0:
                stopped = True
                details.append(f"ended scheduled task '{spec.windows_task_name}'")
            else:
                detail = result.stderr.strip() or result.stdout.strip()
                if detail:
                    details.append(detail)

        if pid is not None and _pid_alive(pid):
            if _terminate_windows_pid(pid):
                stopped = True
                details.append(f"terminated PID {pid}")
            else:
                details.append(f"failed to terminate PID {pid}")

        pid_file.unlink(missing_ok=True)

        if stopped:
            detail_text = "; ".join(dict.fromkeys(details))
            return True, f"{spec.display_name} stopped{f' ({detail_text})' if detail_text else ''}"
        return False, f"{spec.display_name} is not running"

    pid = _read_pid(service_mode)
    if pid is None or not _pid_alive(pid):
        pid_file.unlink(missing_ok=True)
        return False, f"{spec.display_name} is not running"

    try:
        os.kill(pid, signal.SIGTERM)

        for _ in range(20):
            time.sleep(0.5)
            if not _pid_alive(pid):
                break
        else:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        pid_file.unlink(missing_ok=True)
        return True, f"{spec.display_name} stopped (was PID: {pid})"
    except Exception as exc:  # noqa: BLE001
        return False, f"Failed to stop {spec.display_name.lower()}: {exc}"


def restart(
    config: Optional[Path] = None,
    mode: ServiceMode | str = DEFAULT_SERVICE_MODE,
) -> tuple[bool, str]:
    """Stop then start the selected service."""
    service_mode = normalize_mode(mode)
    ok, msg = stop(service_mode)
    if not ok and "not running" not in msg:
        return False, f"Stop failed: {msg}"
    time.sleep(1)
    return start(config=config, mode=service_mode)


def get_status(mode: ServiceMode | str = DEFAULT_SERVICE_MODE) -> dict[str, object]:
    """Return a dict with status information for the selected service."""
    service_mode = normalize_mode(mode)
    spec = _get_service_spec(service_mode)
    pid_file = _get_pid_file(service_mode)
    log_file = _get_log_file(service_mode)
    supervisor = _detect_supervisor(service_mode)
    pid = _get_runtime_pid(service_mode)
    installed = supervisor in {
        WINDOWS_SUPERVISOR_SCHTASKS,
        WINDOWS_SUPERVISOR_STARTUP,
        SUPERVISOR_SYSTEMD,
        SUPERVISOR_LAUNCHD,
    }
    running = pid is not None and _pid_alive(pid)

    info: dict[str, object] = {
        "mode": spec.mode.value,
        "display_name": spec.display_name,
        "running": running,
        "pid": pid if running else None,
        "log_file": str(log_file),
        "pid_file": str(pid_file),
        "auto_start": installed,
        "installed": installed,
        "supervisor": supervisor,
    }
    if service_mode is ServiceMode.HTTP_GATEWAY:
        info["url"] = "http://127.0.0.1:8080"

    if platform.system() == "Windows":
        if supervisor == WINDOWS_SUPERVISOR_SCHTASKS:
            query = _query_windows_task(service_mode)
            if query.returncode == 0:
                task_status = _parse_windows_task_status(query.stdout)
                last_result = _parse_windows_task_last_result(query.stdout)
                if task_status:
                    info["task_status"] = task_status
                    if task_status.lower() == "running":
                        info["running"] = True
                if last_result:
                    info["task_last_result"] = last_result
                if service_mode is ServiceMode.HTTP_GATEWAY and _is_http_gateway_listening():
                    info["running"] = True
                    info["pid"] = _find_windows_listener_pid(8080)
            info["task_name"] = spec.windows_task_name
        elif supervisor == WINDOWS_SUPERVISOR_STARTUP:
            info["startup_entry"] = str(_get_windows_startup_launcher(service_mode))
            if service_mode is ServiceMode.HTTP_GATEWAY and _is_http_gateway_listening():
                info["running"] = True
                info["pid"] = _find_windows_listener_pid(8080)

    if service_mode is ServiceMode.HTTP_GATEWAY and info.get("running"):
        health = _fetch_gateway_health()
        if health is not None:
            info["gateway_health"] = health.get("status")
            checks = health.get("checks", [])
            if isinstance(checks, list):
                for check in checks:
                    if not isinstance(check, dict):
                        continue
                    if check.get("name") == "channels":
                        info["channels_health"] = check.get("status")
                        if check.get("message"):
                            info["channels_message"] = check.get("message")
                        break

    return info


def _fetch_gateway_health() -> dict[str, object] | None:
    try:
        with urlopen("http://127.0.0.1:8080/health", timeout=1.5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, TimeoutError, ValueError, URLError):
        return None
    return payload if isinstance(payload, dict) else None


def tail_logs(
    lines: int = 50,
    follow: bool = False,
    mode: ServiceMode | str = DEFAULT_SERVICE_MODE,
) -> None:
    """Print the last *lines* of the selected service log."""
    log_file = _get_log_file(mode)
    if not log_file.exists():
        print("No log file found yet. Start the service first.")
        return

    if follow:
        with open(log_file, encoding="utf-8", errors="replace") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 8192))
            fh.read()
            print(f"==> {log_file} (Ctrl+C to stop) <==")
            try:
                while True:
                    line = fh.readline()
                    if line:
                        print(line, end="", flush=True)
                    else:
                        time.sleep(0.2)
            except KeyboardInterrupt:
                pass
    else:
        with open(log_file, encoding="utf-8", errors="replace") as fh:
            content = fh.readlines()
        for line in content[-lines:]:
            print(line, end="")


# ---------------------------------------------------------------------------
# Auto-start installation helpers
# ---------------------------------------------------------------------------

def _is_auto_start_installed(mode: ServiceMode | str = DEFAULT_SERVICE_MODE) -> bool:
    service_mode = normalize_mode(mode)
    if platform.system() == "Windows":
        return _windows_task_exists(service_mode) or _get_windows_startup_launcher(service_mode).exists()
    if platform.system() == "Linux":
        return _linux_service_file(service_mode).exists()
    if platform.system() == "Darwin":
        return _macos_plist_file(service_mode).exists()
    return False


def install_auto_start(
    config: Optional[Path] = None,
    mode: ServiceMode | str = DEFAULT_SERVICE_MODE,
) -> tuple[bool, str]:
    """Register the selected service to start automatically at user login."""
    service_mode = normalize_mode(mode)
    system = platform.system()
    if system == "Windows":
        return _install_windows(service_mode, config)
    if system == "Linux":
        return _install_linux(service_mode, config)
    if system == "Darwin":
        return _install_macos(service_mode, config)
    return False, f"Auto-start not supported on {system}"


def uninstall_auto_start(mode: ServiceMode | str = DEFAULT_SERVICE_MODE) -> tuple[bool, str]:
    """Remove the auto-start registration for the selected service."""
    service_mode = normalize_mode(mode)
    system = platform.system()
    if system == "Windows":
        return _uninstall_windows(service_mode)
    if system == "Linux":
        return _uninstall_linux(service_mode)
    if system == "Darwin":
        return _uninstall_macos(service_mode)
    return False, f"Auto-start not supported on {system}"


# ---------------------------------------------------------------------------
# Windows - Task Scheduler / Startup folder
# ---------------------------------------------------------------------------

def _install_windows(
    mode: ServiceMode,
    config: Optional[Path] = None,
) -> tuple[bool, str]:
    spec = _get_service_spec(mode)
    script_path = _write_windows_service_script(mode, config)
    task_cmd = subprocess.list2cmdline(["cmd.exe", "/d", "/c", str(script_path)])

    result = _run_windows_schtasks(
        [
            "/Create",
            "/TN",
            spec.windows_task_name,
            "/TR",
            task_cmd,
            "/SC",
            "ONLOGON",
            "/F",
        ],
    )
    if result.returncode == 0:
        _get_windows_startup_launcher(mode).unlink(missing_ok=True)
        return True, (
            f"Installed as Windows Task Scheduler task: '{spec.windows_task_name}'\n"
            f"Script: {script_path}\n"
            "The service will start automatically at next login."
        )

    detail = result.stderr.strip() or result.stdout.strip()
    if _should_fallback_to_startup_launcher(detail):
        launcher_path = _write_windows_startup_launcher(mode, config)
        return True, (
            "Task Scheduler install was denied; installed Startup-folder fallback instead.\n"
            f"Launcher: {launcher_path}\n"
            f"Script: {script_path}"
        )

    return False, f"schtasks failed: {detail}"


def _uninstall_windows(mode: ServiceMode) -> tuple[bool, str]:
    spec = _get_service_spec(mode)
    removed: list[str] = []

    if _windows_task_exists(mode):
        result = _run_windows_schtasks(["/Delete", "/TN", spec.windows_task_name, "/F"])
        if result.returncode == 0:
            removed.append(f"Windows Task Scheduler task '{spec.windows_task_name}'")
        else:
            detail = result.stderr.strip() or result.stdout.strip()
            return False, f"schtasks failed: {detail}"

    launcher_path = _get_windows_startup_launcher(mode)
    if launcher_path.exists():
        launcher_path.unlink()
        removed.append(f"Startup-folder launcher '{launcher_path.name}'")

    _get_windows_service_script(mode).unlink(missing_ok=True)

    if removed:
        return True, f"Removed {' and '.join(removed)}"
    return False, "No Windows auto-start entry was installed"


# ---------------------------------------------------------------------------
# Linux - systemd user service
# ---------------------------------------------------------------------------

def _install_linux(
    mode: ServiceMode,
    config: Optional[Path] = None,
) -> tuple[bool, str]:
    spec = _get_service_spec(mode)
    cmd = _get_command(mode, config)
    exec_start = shlex.join(cmd)
    log_file = _get_log_file(mode)
    env_overrides = _get_env_overrides(mode, config)
    env_lines = "".join(
        f'Environment="{key}={str(value).replace(chr(34), r"\\\"")}"\n'
        for key, value in env_overrides.items()
    )

    unit = f"""\
[Unit]
Description={spec.description}
After=network.target

[Service]
{env_lines}ExecStart={exec_start}
WorkingDirectory={Path.home()}
Restart=always
RestartSec=5
StandardOutput=append:{log_file}
StandardError=append:{log_file}

[Install]
WantedBy=default.target
"""
    service_file = _linux_service_file(mode)
    service_file.parent.mkdir(parents=True, exist_ok=True)
    service_file.write_text(unit)

    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    subprocess.run(["systemctl", "--user", "enable", spec.linux_service_name], check=False)

    username = os.environ.get("USER", "")
    if username:
        subprocess.run(["loginctl", "enable-linger", username], check=False)

    return True, (
        f"Installed systemd user service: {service_file}\n"
        "The service will start automatically at boot.\n"
        f"Start it now with: spoon-bot service start --mode {mode.value}"
    )


def _uninstall_linux(mode: ServiceMode) -> tuple[bool, str]:
    spec = _get_service_spec(mode)
    subprocess.run(["systemctl", "--user", "disable", spec.linux_service_name], check=False)
    service_file = _linux_service_file(mode)
    service_file.unlink(missing_ok=True)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    return True, f"Removed systemd user service: {service_file}"


# ---------------------------------------------------------------------------
# macOS - launchd agent
# ---------------------------------------------------------------------------

def _install_macos(
    mode: ServiceMode,
    config: Optional[Path] = None,
) -> tuple[bool, str]:
    import plistlib

    spec = _get_service_spec(mode)
    cmd = _get_command(mode, config)
    env_overrides = _get_env_overrides(mode, config)
    log_file = _get_log_file(mode)
    plist_file = _macos_plist_file(mode)
    plist_file.parent.mkdir(parents=True, exist_ok=True)

    plist: dict[str, object] = {
        "Label": spec.macos_label,
        "ProgramArguments": cmd,
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(log_file),
        "StandardErrorPath": str(log_file),
    }
    if env_overrides:
        plist["EnvironmentVariables"] = dict(env_overrides)

    with open(plist_file, "wb") as fh:
        plistlib.dump(plist, fh)

    subprocess.run(["launchctl", "load", str(plist_file)], check=False)

    return True, (
        f"Installed launchd agent: {plist_file}\n"
        "The service will start automatically at login."
    )


def _uninstall_macos(mode: ServiceMode) -> tuple[bool, str]:
    plist_file = _macos_plist_file(mode)
    if plist_file.exists():
        subprocess.run(["launchctl", "unload", str(plist_file)], check=False)
        plist_file.unlink()
        return True, f"Removed launchd agent: {plist_file}"
    return False, "launchd agent not installed"
