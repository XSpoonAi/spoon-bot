"""Cross-platform background service management for spoon-bot.

Provides start/stop/restart/status/logs commands to run spoon-bot gateway
as a persistent background process without Docker.

Supports:
  - Manual start/stop via PID file (all platforms)
  - Auto-start at login:
      Windows  → Task Scheduler (schtasks, no admin required)
      Linux    → systemd user service (systemctl --user)
      macOS    → launchd agent (~/Library/LaunchAgents)
"""

from __future__ import annotations

import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SPOON_BOT_DIR = Path.home() / ".spoon-bot"
PID_FILE = SPOON_BOT_DIR / "service.pid"
LOG_FILE = SPOON_BOT_DIR / "gateway.log"

_WINDOWS_TASK_NAME = "SpoonBot Gateway"
_LINUX_SERVICE_NAME = "spoon-bot"
_MACOS_LABEL = "com.xspoon.spoon-bot"


# ---------------------------------------------------------------------------
# Executable detection
# ---------------------------------------------------------------------------

def _get_command(config: Optional[Path] = None) -> list[str]:
    """Build the full command list for starting the gateway."""
    exe = shutil.which("spoon-bot")
    if exe:
        cmd = [exe, "gateway"]
    else:
        # Editable / development install fallback
        cmd = [sys.executable, "-m", "spoon_bot", "gateway"]

    if config and config.exists():
        cmd += ["--config", str(config)]
    elif (SPOON_BOT_DIR / "config.yaml").exists():
        cmd += ["--config", str(SPOON_BOT_DIR / "config.yaml")]

    return cmd


# ---------------------------------------------------------------------------
# PID helpers
# ---------------------------------------------------------------------------

def _read_pid() -> Optional[int]:
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return None


def _pid_alive(pid: int) -> bool:
    """Return True if a process with *pid* is currently running."""
    try:
        if platform.system() == "Windows":
            import ctypes
            SYNCHRONIZE = 0x00100000
            handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        else:
            os.kill(pid, 0)
            return True
    except (OSError, ProcessLookupError):
        return False


def _clear_stale_pid() -> None:
    pid = _read_pid()
    if pid is not None and not _pid_alive(pid):
        PID_FILE.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Public API — process management (all platforms)
# ---------------------------------------------------------------------------

def is_running() -> bool:
    """Return True if the service process is alive."""
    _clear_stale_pid()
    pid = _read_pid()
    return pid is not None and _pid_alive(pid)


def start(config: Optional[Path] = None) -> tuple[bool, str]:
    """Start the gateway in the background.

    Returns (success, message).
    """
    if is_running():
        pid = _read_pid()
        return False, f"Service is already running (PID: {pid})"

    SPOON_BOT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = _get_command(config)

    log_fh = open(LOG_FILE, "a", encoding="utf-8")
    kwargs: dict = {"stdout": log_fh, "stderr": log_fh, "stdin": subprocess.DEVNULL}

    if platform.system() == "Windows":
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        kwargs["creationflags"] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        kwargs["close_fds"] = True
    else:
        kwargs["start_new_session"] = True

    try:
        proc = subprocess.Popen(cmd, **kwargs)  # noqa: S603
        PID_FILE.write_text(str(proc.pid))
        return True, f"Service started (PID: {proc.pid})\nLogs: {LOG_FILE}"
    except FileNotFoundError:
        return False, f"Cannot find spoon-bot executable. Command: {' '.join(cmd)}"
    except Exception as exc:  # noqa: BLE001
        return False, f"Failed to start service: {exc}"


def stop() -> tuple[bool, str]:
    """Stop the running service gracefully.

    Returns (success, message).
    """
    pid = _read_pid()
    if pid is None or not _pid_alive(pid):
        PID_FILE.unlink(missing_ok=True)
        return False, "Service is not running"

    try:
        if platform.system() == "Windows":
            import ctypes
            PROCESS_TERMINATE = 0x0001
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
            if handle:
                ctypes.windll.kernel32.TerminateProcess(handle, 0)
                ctypes.windll.kernel32.CloseHandle(handle)
        else:
            os.kill(pid, signal.SIGTERM)

        # Wait up to 10 s for graceful shutdown
        for _ in range(20):
            time.sleep(0.5)
            if not _pid_alive(pid):
                break
        else:
            # Force kill if still alive
            if platform.system() != "Windows":
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

        PID_FILE.unlink(missing_ok=True)
        return True, f"Service stopped (was PID: {pid})"
    except Exception as exc:  # noqa: BLE001
        return False, f"Failed to stop service: {exc}"


def restart(config: Optional[Path] = None) -> tuple[bool, str]:
    """Stop then start the service."""
    ok, msg = stop()
    if not ok and "not running" not in msg:
        return False, f"Stop failed: {msg}"
    time.sleep(1)
    return start(config)


def get_status() -> dict:
    """Return a dict with status information."""
    _clear_stale_pid()
    pid = _read_pid()
    running = pid is not None and _pid_alive(pid)
    if not running:
        pid = None
    return {
        "running": running,
        "pid": pid,
        "log_file": str(LOG_FILE),
        "pid_file": str(PID_FILE),
        "auto_start": _is_auto_start_installed(),
    }


def tail_logs(lines: int = 50, follow: bool = False) -> None:
    """Print the last *lines* of the log, optionally streaming new output."""
    if not LOG_FILE.exists():
        print("No log file found yet. Start the service first.")
        return

    if follow:
        # Cross-platform tail -f: read new bytes as they appear
        with open(LOG_FILE, encoding="utf-8", errors="replace") as fh:
            # Seek to end minus ~8 KB so we see some history
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 8192))
            fh.read()  # discard partial line at start
            print(f"==> {LOG_FILE} (Ctrl+C to stop) <==")
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
        # Read last N lines
        with open(LOG_FILE, encoding="utf-8", errors="replace") as fh:
            content = fh.readlines()
        for line in content[-lines:]:
            print(line, end="")


# ---------------------------------------------------------------------------
# Auto-start installation helpers
# ---------------------------------------------------------------------------

def _is_auto_start_installed() -> bool:
    system = platform.system()
    if system == "Windows":
        result = subprocess.run(
            ["schtasks", "/Query", "/TN", _WINDOWS_TASK_NAME],
            capture_output=True,
        )
        return result.returncode == 0
    elif system == "Linux":
        service_file = _linux_service_file()
        return service_file.exists()
    elif system == "Darwin":
        plist_file = _macos_plist_file()
        return plist_file.exists()
    return False


def install_auto_start(config: Optional[Path] = None) -> tuple[bool, str]:
    """Register the gateway to start automatically at user login."""
    system = platform.system()
    if system == "Windows":
        return _install_windows(config)
    elif system == "Linux":
        return _install_linux(config)
    elif system == "Darwin":
        return _install_macos(config)
    else:
        return False, f"Auto-start not supported on {system}"


def uninstall_auto_start() -> tuple[bool, str]:
    """Remove the auto-start registration."""
    system = platform.system()
    if system == "Windows":
        return _uninstall_windows()
    elif system == "Linux":
        return _uninstall_linux()
    elif system == "Darwin":
        return _uninstall_macos()
    else:
        return False, f"Auto-start not supported on {system}"


# ---------------------------------------------------------------------------
# Windows — Task Scheduler
# ---------------------------------------------------------------------------

def _install_windows(config: Optional[Path] = None) -> tuple[bool, str]:
    cmd = _get_command(config)
    # Wrap each token in quotes for the TR argument
    task_cmd = " ".join(f'"{c}"' if " " in c else c for c in cmd)

    result = subprocess.run(
        [
            "schtasks", "/Create",
            "/TN", _WINDOWS_TASK_NAME,
            "/TR", task_cmd,
            "/SC", "ONLOGON",
            "/F",  # overwrite if exists
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True, (
            f"Installed as Windows Task Scheduler task: '{_WINDOWS_TASK_NAME}'\n"
            "The gateway will start automatically at next login."
        )
    return False, f"schtasks failed: {result.stderr.strip() or result.stdout.strip()}"


def _uninstall_windows() -> tuple[bool, str]:
    result = subprocess.run(
        ["schtasks", "/Delete", "/TN", _WINDOWS_TASK_NAME, "/F"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True, f"Removed Windows Task Scheduler task: '{_WINDOWS_TASK_NAME}'"
    return False, f"schtasks failed: {result.stderr.strip() or result.stdout.strip()}"


# ---------------------------------------------------------------------------
# Linux — systemd user service
# ---------------------------------------------------------------------------

def _linux_service_file() -> Path:
    return Path.home() / ".config" / "systemd" / "user" / f"{_LINUX_SERVICE_NAME}.service"


def _install_linux(config: Optional[Path] = None) -> tuple[bool, str]:
    cmd = _get_command(config)
    exec_start = " ".join(cmd)

    unit = f"""\
[Unit]
Description=SpoonBot AI Agent Gateway
After=network.target

[Service]
ExecStart={exec_start}
WorkingDirectory={Path.home()}
Restart=always
RestartSec=5
StandardOutput=append:{LOG_FILE}
StandardError=append:{LOG_FILE}

[Install]
WantedBy=default.target
"""
    service_file = _linux_service_file()
    service_file.parent.mkdir(parents=True, exist_ok=True)
    service_file.write_text(unit)

    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    subprocess.run(["systemctl", "--user", "enable", _LINUX_SERVICE_NAME], check=False)

    # Enable lingering so the service starts at boot even without a login session
    username = os.environ.get("USER", "")
    if username:
        subprocess.run(["loginctl", "enable-linger", username], check=False)

    return True, (
        f"Installed systemd user service: {service_file}\n"
        "The gateway will start automatically at boot.\n"
        "Start it now with: spoon-bot service start"
    )


def _uninstall_linux() -> tuple[bool, str]:
    subprocess.run(["systemctl", "--user", "disable", _LINUX_SERVICE_NAME], check=False)
    service_file = _linux_service_file()
    service_file.unlink(missing_ok=True)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    return True, f"Removed systemd user service: {service_file}"


# ---------------------------------------------------------------------------
# macOS — launchd agent
# ---------------------------------------------------------------------------

def _macos_plist_file() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{_MACOS_LABEL}.plist"


def _install_macos(config: Optional[Path] = None) -> tuple[bool, str]:
    import plistlib

    cmd = _get_command(config)
    plist_file = _macos_plist_file()
    plist_file.parent.mkdir(parents=True, exist_ok=True)

    plist: dict = {
        "Label": _MACOS_LABEL,
        "ProgramArguments": cmd,
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(LOG_FILE),
        "StandardErrorPath": str(LOG_FILE),
    }

    with open(plist_file, "wb") as fh:
        plistlib.dump(plist, fh)

    subprocess.run(["launchctl", "load", str(plist_file)], check=False)

    return True, (
        f"Installed launchd agent: {plist_file}\n"
        "The gateway will start automatically at login."
    )


def _uninstall_macos() -> tuple[bool, str]:
    plist_file = _macos_plist_file()
    if plist_file.exists():
        subprocess.run(["launchctl", "unload", str(plist_file)], check=False)
        plist_file.unlink()
        return True, f"Removed launchd agent: {plist_file}"
    return False, "launchd agent not installed"
