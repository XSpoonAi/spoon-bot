#!/usr/bin/env python3
"""Run local services in the background and expose them through Cloudflare."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

WINDOWS_DETACHED_PROCESS = 0x00000008
WINDOWS_CREATE_NEW_PROCESS_GROUP = 0x00000200
WINDOWS_CREATE_NO_WINDOW = 0x08000000
TRYCLOUDFLARE_RE = re.compile(r"https://[-a-zA-Z0-9]+\.trycloudflare\.com")


def _state_root() -> Path:
    raw = os.environ.get("SPOON_BOT_SERVICE_EXPOSE_DIR")
    return Path(raw).expanduser().resolve() if raw else Path.home() / ".spoon-bot" / "service-expose"


def _registry_path() -> Path:
    return _state_root() / "registry.json"


def _logs_dir() -> Path:
    return _state_root() / "logs"


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _json_result(payload: dict[str, Any]) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=True)


def _load_params() -> dict[str, Any]:
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON input: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Input must be a JSON object")
    return payload


def _load_registry() -> dict[str, Any]:
    path = _registry_path()
    if not path.exists():
        return {"services": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"services": {}}
    if not isinstance(payload, dict):
        return {"services": {}}
    payload.setdefault("services", {})
    return payload


def _save_registry(registry: dict[str, Any]) -> None:
    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(registry, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)


def _safe_name(name: str | None) -> str:
    raw = str(name or "").strip()
    if not raw:
        raise ValueError("name is required")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-")
    if not safe:
        raise ValueError("name must contain at least one letter or digit")
    return safe[:80]


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        if os.name == "nt":
            import ctypes

            handle = ctypes.windll.kernel32.OpenProcess(0x00100000, False, int(pid))
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _stop_pid(pid: int | None, *, process_group: bool = True) -> bool:
    if not _pid_alive(pid):
        return True
    if os.name == "nt":
        result = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0 or not _pid_alive(pid)

    assert pid is not None
    try:
        if process_group:
            os.killpg(int(pid), signal.SIGTERM)
        else:
            os.kill(int(pid), signal.SIGTERM)
    except ProcessLookupError:
        return True
    except OSError:
        try:
            os.kill(int(pid), signal.SIGTERM)
        except OSError:
            return False
    for _ in range(20):
        if not _pid_alive(pid):
            return True
        time.sleep(0.1)
    try:
        if process_group:
            os.killpg(int(pid), signal.SIGKILL)
        else:
            os.kill(int(pid), signal.SIGKILL)
    except OSError:
        pass
    return not _pid_alive(pid)


def _tail(path: str | None, tail_chars: int = 4000) -> str:
    if not path:
        return ""
    log_path = Path(path)
    if not log_path.exists():
        return ""
    try:
        with open(log_path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - max(1, int(tail_chars))))
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _spawn_shell(command: str, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    flags = 0
    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "cwd": str(cwd),
        "shell": True,
        "env": os.environ.copy(),
    }
    if os.name == "nt":
        flags = WINDOWS_DETACHED_PROCESS | WINDOWS_CREATE_NEW_PROCESS_GROUP | WINDOWS_CREATE_NO_WINDOW
        kwargs["creationflags"] = flags
    else:
        kwargs["start_new_session"] = True

    with open(log_path, "ab") as log_fh:
        kwargs["stdout"] = log_fh
        kwargs["stderr"] = subprocess.STDOUT
        proc = subprocess.Popen(command, **kwargs)  # noqa: S602
    return int(proc.pid)


def _spawn_argv(argv: list[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "cwd": str(cwd),
        "env": os.environ.copy(),
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            WINDOWS_DETACHED_PROCESS | WINDOWS_CREATE_NEW_PROCESS_GROUP | WINDOWS_CREATE_NO_WINDOW
        )
    else:
        kwargs["start_new_session"] = True

    with open(log_path, "ab") as log_fh:
        kwargs["stdout"] = log_fh
        kwargs["stderr"] = subprocess.STDOUT
        proc = subprocess.Popen(argv, **kwargs)  # noqa: S603
    return int(proc.pid)


def _local_url(params: dict[str, Any], entry: dict[str, Any] | None = None) -> str:
    url = str(params.get("url") or "").strip()
    if url:
        return url
    if entry and entry.get("local_url"):
        return str(entry["local_url"])
    port = params.get("port") if params.get("port") is not None else (entry or {}).get("port")
    if port is None:
        raise ValueError("port or url is required for tunnel exposure")
    host = str(params.get("host") or (entry or {}).get("host") or "127.0.0.1").strip() or "127.0.0.1"
    scheme = str(params.get("scheme") or (entry or {}).get("scheme") or "http").strip() or "http"
    return f"{scheme}://{host}:{int(port)}"


def _parse_public_url(log_path: Path) -> str | None:
    text = _tail(str(log_path), 20000)
    match = TRYCLOUDFLARE_RE.search(text)
    return match.group(0) if match else None


def _start_tunnel_for_entry(
    name: str,
    entry: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    cloudflared = os.environ.get("CLOUDFLARED_PATH") or shutil.which("cloudflared")
    if not cloudflared:
        return {
            "success": False,
            "error": "cloudflared not found. Install cloudflared or set CLOUDFLARED_PATH.",
        }

    local_url = _local_url(params, entry)
    tunnel_log = _logs_dir() / f"{name}.cloudflared.log"
    pid = _spawn_argv([cloudflared, "tunnel", "--url", local_url], Path.cwd(), tunnel_log)
    public_url = None
    for _ in range(int(params.get("tunnel_wait_seconds", 25)) * 2):
        public_url = _parse_public_url(tunnel_log)
        if public_url:
            break
        if not _pid_alive(pid):
            break
        time.sleep(0.5)

    tunnel = {
        "pid": pid,
        "local_url": local_url,
        "public_url": public_url,
        "log_path": str(tunnel_log),
        "started_at": _now(),
        "status": "running" if _pid_alive(pid) else "exited",
    }
    entry["tunnel"] = tunnel
    return {"success": public_url is not None, **tunnel}


def _refresh_entry(entry: dict[str, Any]) -> dict[str, Any]:
    service_pid = entry.get("pid")
    entry["status"] = "running" if _pid_alive(service_pid) else "stopped"
    tunnel = entry.get("tunnel")
    if isinstance(tunnel, dict):
        tunnel["status"] = "running" if _pid_alive(tunnel.get("pid")) else "stopped"
    return entry


def _action_start(params: dict[str, Any]) -> dict[str, Any]:
    name = _safe_name(params.get("name"))
    command = str(params.get("command") or "").strip()
    if not command:
        raise ValueError("command is required for action=start")
    cwd = Path(params.get("cwd") or os.getcwd()).expanduser().resolve()
    if not cwd.is_dir():
        raise ValueError(f"cwd is not a directory: {cwd}")

    registry = _load_registry()
    services = registry.setdefault("services", {})
    existing = services.get(name)
    if isinstance(existing, dict):
        _refresh_entry(existing)
        if existing.get("status") == "running" and not params.get("replace"):
            return {
                "success": False,
                "error": f"Service '{name}' is already running",
                "service": existing,
            }
        if params.get("replace"):
            _stop_entry(existing, stop_tunnel=True)

    log_path = _logs_dir() / f"{name}.service.log"
    pid = _spawn_shell(command, cwd, log_path)
    local_url = None
    try:
        local_url = _local_url(params)
    except ValueError:
        pass

    entry = {
        "name": name,
        "command": command,
        "cwd": str(cwd),
        "pid": pid,
        "status": "running" if _pid_alive(pid) else "exited",
        "host": str(params.get("host") or "127.0.0.1"),
        "port": params.get("port"),
        "scheme": str(params.get("scheme") or "http"),
        "local_url": local_url,
        "log_path": str(log_path),
        "started_at": _now(),
        "tunnel": None,
    }
    if params.get("start_tunnel"):
        entry["tunnel"] = _start_tunnel_for_entry(name, entry, params)
    services[name] = entry
    _save_registry(registry)
    return {"success": True, "service": _refresh_entry(entry)}


def _stop_entry(entry: dict[str, Any], *, stop_tunnel: bool) -> dict[str, Any]:
    stopped_tunnel = None
    if stop_tunnel and isinstance(entry.get("tunnel"), dict):
        tunnel = entry["tunnel"]
        stopped_tunnel = _stop_pid(tunnel.get("pid"))
        tunnel["status"] = "stopped" if stopped_tunnel else "stop_failed"
        tunnel["stopped_at"] = _now()
    stopped_service = _stop_pid(entry.get("pid"))
    entry["status"] = "stopped" if stopped_service else "stop_failed"
    entry["stopped_at"] = _now()
    return {"service_stopped": stopped_service, "tunnel_stopped": stopped_tunnel}


def _action_stop(params: dict[str, Any], *, tunnel_only: bool = False) -> dict[str, Any]:
    name = _safe_name(params.get("name"))
    registry = _load_registry()
    entry = registry.get("services", {}).get(name)
    if not isinstance(entry, dict):
        return {"success": False, "error": f"Service '{name}' not found"}

    if tunnel_only:
        tunnel = entry.get("tunnel")
        if not isinstance(tunnel, dict):
            return {"success": False, "error": f"Service '{name}' has no tunnel"}
        stopped = _stop_pid(tunnel.get("pid"))
        tunnel["status"] = "stopped" if stopped else "stop_failed"
        tunnel["stopped_at"] = _now()
        result = {"tunnel_stopped": stopped}
    else:
        result = _stop_entry(entry, stop_tunnel=True)
    _save_registry(registry)
    return {"success": bool(all(v is not False for v in result.values())), "service": _refresh_entry(entry), **result}


def _action_tunnel(params: dict[str, Any]) -> dict[str, Any]:
    name = _safe_name(params.get("name"))
    registry = _load_registry()
    services = registry.setdefault("services", {})
    entry = services.get(name)
    if not isinstance(entry, dict):
        entry = {
            "name": name,
            "command": None,
            "cwd": str(Path.cwd()),
            "pid": None,
            "status": "external",
            "host": str(params.get("host") or "127.0.0.1"),
            "port": params.get("port"),
            "scheme": str(params.get("scheme") or "http"),
            "local_url": _local_url(params),
            "log_path": None,
            "started_at": _now(),
        }
        services[name] = entry

    existing_tunnel = entry.get("tunnel")
    if isinstance(existing_tunnel, dict) and _pid_alive(existing_tunnel.get("pid")):
        if not params.get("replace"):
            return {"success": False, "error": f"Tunnel for '{name}' is already running", "service": entry}
        _stop_pid(existing_tunnel.get("pid"))

    tunnel = _start_tunnel_for_entry(name, entry, params)
    _save_registry(registry)
    return {"success": bool(tunnel.get("public_url")), "service": _refresh_entry(entry), "tunnel": tunnel}


def _action_status(params: dict[str, Any]) -> dict[str, Any]:
    registry = _load_registry()
    services = registry.setdefault("services", {})
    name = params.get("name")
    if name:
        safe = _safe_name(str(name))
        entry = services.get(safe)
        if not isinstance(entry, dict):
            return {"success": False, "error": f"Service '{safe}' not found"}
        return {"success": True, "service": _refresh_entry(entry)}
    refreshed = {key: _refresh_entry(value) for key, value in services.items() if isinstance(value, dict)}
    return {"success": True, "services": refreshed}


def _action_logs(params: dict[str, Any]) -> dict[str, Any]:
    name = _safe_name(params.get("name"))
    target = str(params.get("target") or "service")
    tail_chars = int(params.get("tail_chars") or 4000)
    registry = _load_registry()
    entry = registry.get("services", {}).get(name)
    if not isinstance(entry, dict):
        return {"success": False, "error": f"Service '{name}' not found"}

    result: dict[str, Any] = {"success": True, "name": name}
    if target in {"service", "all"}:
        result["service_log"] = _tail(entry.get("log_path"), tail_chars)
    if target in {"tunnel", "all"}:
        tunnel = entry.get("tunnel") if isinstance(entry.get("tunnel"), dict) else {}
        result["tunnel_log"] = _tail(tunnel.get("log_path"), tail_chars)
    return result


def main() -> None:
    try:
        params = _load_params()
        action = str(params.get("action") or "").strip().lower()
        if action == "start":
            result = _action_start(params)
        elif action == "tunnel":
            result = _action_tunnel(params)
        elif action in {"status", "list"}:
            result = _action_status(params)
        elif action == "logs":
            result = _action_logs(params)
        elif action == "stop":
            result = _action_stop(params)
        elif action == "stop_tunnel":
            result = _action_stop(params, tunnel_only=True)
        else:
            result = {"success": False, "error": "Unsupported action. Use start, tunnel, status, list, logs, stop, or stop_tunnel."}
    except Exception as exc:
        result = {"success": False, "error": str(exc)}
    _json_result(result)


if __name__ == "__main__":
    main()
