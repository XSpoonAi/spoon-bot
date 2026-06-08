#!/usr/bin/env python3
"""Run local services in the background and expose them through Cloudflare."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

WINDOWS_DETACHED_PROCESS = 0x00000008
WINDOWS_CREATE_NEW_PROCESS_GROUP = 0x00000200
WINDOWS_CREATE_NO_WINDOW = 0x08000000
TRYCLOUDFLARE_RE = re.compile(r"https://[-a-zA-Z0-9]+\.trycloudflare\.com")
CLOUDFLARED_LATEST_BASE_URL = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download"
)


def _state_root() -> Path:
    raw = os.environ.get("SPOON_BOT_SERVICE_EXPOSE_DIR")
    return Path(raw).expanduser().resolve() if raw else Path.home() / ".spoon-bot" / "service-expose"


def _registry_path() -> Path:
    return _state_root() / "registry.json"


def _logs_dir() -> Path:
    return _state_root() / "logs"


def _bin_dir() -> Path:
    return _state_root() / "bin"


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _cloudflared_asset_name() -> str | None:
    system = platform.system().lower()
    machine = platform.machine().lower()
    arch = ""
    if machine in {"x86_64", "amd64"}:
        arch = "amd64"
    elif machine in {"aarch64", "arm64"}:
        arch = "arm64"
    elif machine.startswith("arm"):
        arch = "arm"
    if not arch:
        return None
    if system == "linux":
        return f"cloudflared-linux-{arch}"
    if system == "darwin":
        return f"cloudflared-darwin-{arch}.tgz"
    if system == "windows" and arch == "amd64":
        return "cloudflared-windows-amd64.exe"
    return None


def _cloudflared_executable_name() -> str:
    return "cloudflared.exe" if platform.system().lower() == "windows" else "cloudflared"


def _download_cloudflared(destination: Path) -> str | None:
    asset_name = _cloudflared_asset_name()
    if not asset_name:
        return f"unsupported platform for automatic cloudflared install: {platform.system()} {platform.machine()}"
    if asset_name.endswith(".tgz"):
        return "automatic cloudflared install is not supported for archive assets yet; install cloudflared or set CLOUDFLARED_PATH"
    url = f"{CLOUDFLARED_LATEST_BASE_URL}/{asset_name}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    try:
        req = Request(url, headers={"User-Agent": "spoon-bot-service-expose/1.0"})
        with urlopen(req, timeout=60) as response, tmp.open("wb") as handle:  # noqa: S310 - official Cloudflare release asset
            shutil.copyfileobj(response, handle)
        tmp.replace(destination)
        if os.name != "nt":
            destination.chmod(0o755)
    except Exception as exc:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return f"failed to download cloudflared from {url}: {exc}"
    return None


def _resolve_cloudflared() -> tuple[str | None, str | None]:
    explicit = os.environ.get("CLOUDFLARED_PATH")
    if explicit:
        return explicit, None
    discovered = shutil.which("cloudflared")
    if discovered:
        return discovered, None
    if str(os.environ.get("SPOON_BOT_AUTO_INSTALL_CLOUDFLARED", "1")).strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return None, "cloudflared not found. Install cloudflared, set CLOUDFLARED_PATH, or enable SPOON_BOT_AUTO_INSTALL_CLOUDFLARED."
    destination = _bin_dir() / _cloudflared_executable_name()
    if destination.exists():
        return str(destination), None
    error = _download_cloudflared(destination)
    if error:
        return None, error
    return str(destination), None


def _json_result(payload: dict[str, Any]) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=True)


def _load_params() -> dict[str, Any]:
    try:
        raw = sys.stdin.read().strip()
        if not raw and len(sys.argv) > 1:
            raw = " ".join(sys.argv[1:]).strip()
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
            from ctypes import wintypes

            process_query_limited_information = 0x1000
            still_active = 259
            handle = ctypes.windll.kernel32.OpenProcess(
                process_query_limited_information,
                False,
                int(pid),
            )
            if not handle:
                return False
            try:
                exit_code = wintypes.DWORD()
                if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return False
                return exit_code.value == still_active
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
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


def _spawn_shell(
    command: str,
    cwd: Path,
    log_path: Path,
    *,
    extra_env: dict[str, str] | None = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    flags = 0
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "cwd": str(cwd),
        "shell": True,
        "env": env,
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


def _clear_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _connect_host(host: str | None) -> str:
    raw = str(host or "").strip()
    if raw in {"", "0.0.0.0", "::", "[::]"}:
        return "127.0.0.1"
    return raw.strip("[]") if raw.startswith("[") and raw.endswith("]") else raw


def _display_host_for_url(host: str) -> str:
    if ":" in host and not (host.startswith("[") and host.endswith("]")):
        return f"[{host}]"
    return host


def _connection_probe_hosts(host: str) -> list[str]:
    normalized = _connect_host(host)
    hosts = [normalized]
    if normalized in {"127.0.0.1", "localhost"}:
        hosts.extend(["localhost", "::1"])
    seen: set[str] = set()
    result: list[str] = []
    for item in hosts:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _can_connect(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=0.25):
            return True
    except OSError:
        return False


def _port_is_free(host: str, port: int) -> bool:
    for probe_host in _connection_probe_hosts(host):
        if _can_connect(probe_host, port):
            return False

    bind_host = str(host or "127.0.0.1").strip()
    if bind_host in {"::", "[::]"}:
        bind_host = "::"
    family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.bind((bind_host, int(port)))
        return True
    except OSError:
        return False


def _find_free_port(host: str) -> int:
    bind_host = str(host or "127.0.0.1").strip()
    if bind_host in {"::", "[::]"}:
        bind_host = "::"
    family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.bind((bind_host, 0))
        return int(sock.getsockname()[1])


def _reserved_preview_ports() -> set[int]:
    raw = os.environ.get("SPOON_BOT_SERVICE_EXPOSE_RESERVED_PORTS", "")
    ports: set[int] = set()
    for item in raw.replace(";", ",").split(","):
        value = item.strip()
        if not value:
            continue
        try:
            port = int(value)
        except ValueError:
            continue
        if 1 <= port <= 65535:
            ports.add(port)
    return ports


def _find_free_unreserved_port(host: str) -> int:
    reserved = _reserved_preview_ports()
    for _ in range(50):
        port = _find_free_port(host)
        if port not in reserved:
            return port
    raise ValueError("could not find a free non-reserved port for preview service")


def _local_url(params: dict[str, Any], entry: dict[str, Any] | None = None) -> str:
    url = str(params.get("url") or "").strip()
    if url:
        return url
    if entry and entry.get("local_url"):
        return str(entry["local_url"])
    port = params.get("port") if params.get("port") is not None else (entry or {}).get("port")
    if port is None:
        raise ValueError("port or url is required for tunnel exposure")
    host = _connect_host(params.get("host") or (entry or {}).get("host") or "127.0.0.1")
    scheme = str(params.get("scheme") or (entry or {}).get("scheme") or "http").strip() or "http"
    return f"{scheme}://{_display_host_for_url(host)}:{int(port)}"


def _has_explicit_target(params: dict[str, Any]) -> bool:
    return bool(str(params.get("url") or "").strip()) or params.get("port") is not None


def _apply_explicit_target(entry: dict[str, Any], params: dict[str, Any]) -> None:
    url = str(params.get("url") or "").strip()
    if url:
        entry["local_url"] = url
        if params.get("host"):
            entry["host"] = str(params["host"])
        if params.get("port") is not None:
            entry["port"] = int(params["port"])
        if params.get("scheme"):
            entry["scheme"] = str(params["scheme"])
        return

    if params.get("port") is None:
        return

    port = int(params["port"])
    if port <= 0:
        raise ValueError("port must be greater than 0 for action=tunnel; use action=start with port=0 for auto-selection")
    host = _connect_host(params.get("host") or entry.get("host") or "127.0.0.1")
    scheme = str(params.get("scheme") or entry.get("scheme") or "http").strip() or "http"
    entry["host"] = host
    entry["port"] = port
    entry["scheme"] = scheme
    entry["local_url"] = f"{scheme}://{_display_host_for_url(host)}:{port}"


def _http_body(url: str, timeout: float = 5.0) -> str:
    req = Request(url, headers={"User-Agent": "spoon-bot-service-expose/1.0"})
    with urlopen(req, timeout=timeout) as response:  # noqa: S310 - user-requested local/public preview verification
        data = response.read(512_000)
    return data.decode("utf-8", errors="replace")


def _verify_url(
    url: str | None,
    *,
    expected_text: str | None,
    wait_seconds: float,
) -> dict[str, Any] | None:
    if not url:
        return None
    expected = str(expected_text or "").strip()
    deadline = time.monotonic() + max(0.0, wait_seconds)
    last_error = ""
    while True:
        try:
            body = _http_body(str(url))
            matched = bool(expected and expected in body)
            if not expected or matched:
                return {
                    "url": url,
                    "ok": True,
                    "matched": matched,
                    "expected_text": expected or None,
                }
            last_error = f"Expected text not found: {expected!r}"
        except HTTPError as exc:
            if not expected and exc.code == 426:
                return {
                    "url": url,
                    "ok": True,
                    "matched": False,
                    "expected_text": None,
                    "http_status": 426,
                    "method": "websocket-upgrade-probe",
                }
            last_error = f"HTTP Error {exc.code}: {exc.reason}"
        except (OSError, URLError, TimeoutError) as exc:
            last_error = str(exc)
        if time.monotonic() >= deadline:
            return {
                "url": url,
                "ok": False,
                "matched": False,
                "expected_text": expected or None,
                "error": last_error,
            }
        time.sleep(0.25)


def _parse_public_url(log_path: Path) -> str | None:
    text = _tail(str(log_path), 20000)
    match = TRYCLOUDFLARE_RE.search(text)
    return match.group(0) if match else None


def _infer_port_from_log(log_path: Path) -> int | None:
    text = _tail(str(log_path), 20000)
    candidates: list[int] = []
    for pattern in (
        r"https?://(?:localhost|127\.0\.0\.1|0\.0\.0\.0|\[?::1\]?):(\d{2,5})",
        r"\bport\s+(\d{2,5})\b",
    ):
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                port = int(match.group(1))
            except (TypeError, ValueError):
                continue
            if 1 <= port <= 65535:
                candidates.append(port)
    return candidates[-1] if candidates else None


def _process_tree_pids(pid: Any) -> list[int]:
    """Return a best-effort list of pid and descendants on Linux."""

    try:
        root = int(pid)
    except (TypeError, ValueError):
        return []
    if root <= 0:
        return []

    seen: set[int] = set()
    queue = [root]
    while queue:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        children_path = Path("/proc") / str(current) / "task" / str(current) / "children"
        try:
            raw_children = children_path.read_text(encoding="utf-8").split()
        except OSError:
            raw_children = []
        for raw_child in raw_children:
            try:
                child = int(raw_child)
            except ValueError:
                continue
            if child > 0 and child not in seen:
                queue.append(child)
    return list(seen)


def _socket_inodes_for_pids(pids: list[int]) -> set[str]:
    inodes: set[str] = set()
    for pid in pids:
        fd_dir = Path("/proc") / str(pid) / "fd"
        try:
            entries = list(fd_dir.iterdir())
        except OSError:
            continue
        for entry in entries:
            try:
                target = os.readlink(entry)
            except OSError:
                continue
            if not (target.startswith("socket:[") and target.endswith("]")):
                continue
            inode = target[len("socket:[") : -1]
            if inode:
                inodes.add(inode)
    return inodes


def _listening_tcp_ports_by_inode() -> dict[str, int]:
    ports: dict[str, int] = {}
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            lines = table.read_text(encoding="utf-8").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 10:
                continue
            state = parts[3]
            if state != "0A":
                continue
            local_address = parts[1]
            if ":" not in local_address:
                continue
            _, port_hex = local_address.rsplit(":", 1)
            inode = parts[9]
            try:
                port = int(port_hex, 16)
            except ValueError:
                continue
            if 1 <= port <= 65535 and inode:
                ports[inode] = port
    return ports


def _infer_port_from_process(pid: Any) -> int | None:
    pids = _process_tree_pids(pid)
    if not pids:
        return None
    socket_inodes = _socket_inodes_for_pids(pids)
    if not socket_inodes:
        return None
    ports_by_inode = _listening_tcp_ports_by_inode()
    candidates = sorted(
        {
            port
            for inode, port in ports_by_inode.items()
            if inode in socket_inodes and 1 <= port <= 65535
        }
    )
    return candidates[0] if len(candidates) == 1 else None


def _render_command_placeholders(command: str, params: dict[str, Any]) -> str:
    """Render generic service command placeholders after host/port resolution."""

    replacements = {
        "{port}": str(params.get("port") or ""),
        "{PORT}": str(params.get("port") or ""),
        "{host}": str(params.get("host") or "127.0.0.1"),
        "{HOST}": str(params.get("host") or "127.0.0.1"),
    }
    rendered = str(command or "")
    for placeholder, value in replacements.items():
        if placeholder in rendered and value:
            rendered = rendered.replace(placeholder, value)
    return rendered


def _tunnel_registered(log_path: Path) -> bool:
    text = _tail(str(log_path), 20000)
    return "Registered tunnel connection" in text


def _normalize_action(action: Any) -> str:
    normalized = str(action or "").strip().lower()
    aliases = {
        "expose": "tunnel",
        "start_tunnel": "tunnel",
        "inspect": "status",
    }
    return aliases.get(normalized, normalized)


def _tunnel_protocol(params: dict[str, Any]) -> str:
    protocol = str(
        params.get("tunnel_protocol")
        or os.environ.get("SPOON_BOT_CLOUDFLARED_PROTOCOL")
        or "http2"
    ).strip().lower()
    return protocol or "http2"


def _tunnel_attempts(params: dict[str, Any]) -> int:
    raw = params.get("tunnel_attempts") or os.environ.get("SPOON_BOT_CLOUDFLARED_TUNNEL_ATTEMPTS") or 3
    try:
        attempts = int(raw)
    except (TypeError, ValueError):
        attempts = 3
    return max(1, min(attempts, 5))


def _tunnel_public_settle_seconds(params: dict[str, Any]) -> float:
    raw = (
        params.get("tunnel_public_settle_seconds")
        or os.environ.get("SPOON_BOT_CLOUDFLARE_PUBLIC_SETTLE_SECONDS")
        or 8
    )
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        seconds = 8.0
    return max(0.0, min(seconds, 30.0))


def _public_verification_error_is_transient(verification: dict[str, Any] | None) -> bool:
    if not isinstance(verification, dict):
        return False
    error = str(verification.get("error") or "").casefold()
    return any(
        marker in error
        for marker in (
            "name or service not known",
            "temporary failure in name resolution",
            "errno -2",
            "http error 530",
            "timed out",
        )
    )


def _cloudflare_quick_tunnel_rate_limited(log_path: Path) -> bool:
    text = _tail(str(log_path), 20000).casefold()
    return "429 too many requests" in text or "error code: 1015" in text


def _tunnel_retry_after(entry: dict[str, Any]) -> int | None:
    tunnel = entry.get("tunnel")
    if not isinstance(tunnel, dict):
        return None
    try:
        retry_after_epoch = float(tunnel.get("retry_after_epoch") or 0)
    except (TypeError, ValueError):
        return None
    remaining = int(retry_after_epoch - time.time())
    return remaining if remaining > 0 else None


def _start_tunnel_for_entry(
    name: str,
    entry: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    cloudflared, cloudflared_error = _resolve_cloudflared()
    if not cloudflared:
        return {
            "success": False,
            "error": cloudflared_error
            or "cloudflared not found. Install cloudflared or set CLOUDFLARED_PATH.",
        }

    local_url = _local_url(params, entry)
    tunnel_log = _logs_dir() / f"{name}.cloudflared.log"
    protocol = _tunnel_protocol(params)
    verify_text = params.get("verify_text")
    if not verify_text and isinstance(entry.get("verification"), dict):
        verify_text = entry["verification"].get("expected_text")
    public_settle_seconds = _tunnel_public_settle_seconds(params)

    attempts: list[dict[str, Any]] = []
    last_tunnel: dict[str, Any] | None = None
    max_attempts = _tunnel_attempts(params)
    for attempt in range(1, max_attempts + 1):
        _clear_log(tunnel_log)
        pid = _spawn_argv(
            [cloudflared, "tunnel", "--protocol", protocol, "--url", local_url],
            Path.cwd(),
            tunnel_log,
        )
        public_url = None
        registered = False
        for _ in range(int(params.get("tunnel_wait_seconds", 60)) * 2):
            public_url = _parse_public_url(tunnel_log)
            registered = _tunnel_registered(tunnel_log)
            if public_url and registered:
                break
            if not _pid_alive(pid):
                break
            time.sleep(0.5)

        rate_limited = _cloudflare_quick_tunnel_rate_limited(tunnel_log)
        tunnel = {
            "pid": pid,
            "local_url": local_url,
            "public_url": public_url,
            "registered": registered,
            "rate_limited": rate_limited,
            "protocol": protocol,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "log_path": str(tunnel_log),
            "started_at": _now(),
            "status": "running" if _pid_alive(pid) else "exited",
        }
        if not local_url and params.get("start_tunnel"):
            inferred_port = _infer_port_from_log(tunnel_log)
            if inferred_port:
                params["port"] = inferred_port
                entry["port"] = inferred_port
                entry["local_url"] = _local_url(params)
                local_url = entry["local_url"]

        if public_url:
            if registered and public_settle_seconds > 0:
                time.sleep(public_settle_seconds)
            tunnel["verification"] = _verify_url(
                public_url,
                expected_text=str(verify_text) if verify_text else None,
                wait_seconds=float(params.get("verify_wait_seconds") or 20),
            )
            if not tunnel["verification"].get("ok"):
                tunnel["verification_error"] = tunnel["verification"].get("error")
        success = (
            public_url is not None
            and registered
            and tunnel["status"] == "running"
            and bool(tunnel.get("verification", {}).get("ok"))
        )
        if tunnel.get("verification") and not tunnel["verification"].get("ok"):
            success = False
        if success:
            if attempts:
                tunnel["attempts"] = attempts + [
                    {"attempt": attempt, "success": True, "registered": registered}
                ]
            entry["tunnel"] = tunnel
            return {"success": True, **tunnel}

        if _pid_alive(pid):
            _stop_pid(pid)
            tunnel["status"] = "stopped"
        tunnel["public_url"] = None
        tunnel["public_url_omitted_reason"] = "unverified"
        attempts.append(
            {
                "attempt": attempt,
                "success": False,
                "registered": registered,
                "rate_limited": rate_limited,
                "verification_error": tunnel.get("verification_error"),
            }
        )
        last_tunnel = tunnel
        if rate_limited:
            tunnel["retry_after_seconds"] = int(params.get("rate_limit_retry_after_seconds") or 600)
            tunnel["retry_after_epoch"] = time.time() + tunnel["retry_after_seconds"]
            break
        if not (
            attempt < max_attempts
            and registered
            and _public_verification_error_is_transient(tunnel.get("verification"))
        ):
            break
        time.sleep(min(5.0, 1.5 * attempt))

    tunnel = last_tunnel or {
        "pid": None,
        "local_url": local_url,
        "public_url": None,
        "registered": False,
        "protocol": protocol,
        "log_path": str(tunnel_log),
        "started_at": _now(),
        "status": "exited",
        "public_url_omitted_reason": "unverified",
    }
    if attempts:
        tunnel["attempts"] = attempts
    entry["tunnel"] = tunnel
    return {"success": False, **tunnel}


def _refresh_entry(entry: dict[str, Any]) -> dict[str, Any]:
    service_pid = entry.get("pid")
    entry["status"] = "running" if _pid_alive(service_pid) else "stopped"
    tunnel = entry.get("tunnel")
    if isinstance(tunnel, dict):
        tunnel_running = _pid_alive(tunnel.get("pid"))
        tunnel["status"] = "running" if tunnel_running else "stopped"
        if not tunnel_running or tunnel.get("success") is False:
            tunnel.pop("candidate_public_url", None)
            tunnel["public_url"] = None
            tunnel["public_url_omitted_reason"] = "not_running_or_unverified"
    return entry


def _same_workspace_unexposed_services(
    name: str,
    entry: dict[str, Any],
    services: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return running sibling services that a public browser cannot reach."""

    tunnel = entry.get("tunnel") if isinstance(entry.get("tunnel"), dict) else {}
    if not isinstance(tunnel, dict) or not tunnel.get("public_url"):
        return []
    try:
        entry_cwd = Path(str(entry.get("cwd") or "")).resolve()
    except Exception:
        return []

    blocked: list[dict[str, Any]] = []
    for other_name, other in services.items():
        if other_name == name or not isinstance(other, dict):
            continue
        if not _pid_alive(other.get("pid")):
            continue
        try:
            other_cwd = Path(str(other.get("cwd") or "")).resolve()
        except Exception:
            continue
        if other_cwd != entry_cwd:
            continue
        other_tunnel = other.get("tunnel") if isinstance(other.get("tunnel"), dict) else {}
        if isinstance(other_tunnel, dict) and other_tunnel.get("public_url"):
            continue
        blocked.append(
            {
                "name": str(other.get("name") or other_name),
                "port": other.get("port"),
                "local_url": other.get("local_url"),
                "command": other.get("command"),
            }
        )
    return blocked


def _apply_public_readiness(
    name: str,
    entry: dict[str, Any],
    services: dict[str, Any],
) -> None:
    """Warn when a public app URL still depends on local-only sibling services."""

    blocked = _same_workspace_unexposed_services(name, entry, services)
    if not blocked:
        return
    readiness = {
        "ok": False,
        "blocking": True,
        "reason": "same-workspace-services-not-public",
        "message": (
            "A public browser URL was created, but other running services in "
            "the same app workspace are still local-only. If the public page "
            "uses any of these API/WebSocket/backend ports, do not report the "
            "app as complete until those dependencies are exposed too or routed "
            "through the same public origin."
        ),
        "unexposed_services": blocked,
        "required_next_steps": [
            "Expose each browser-required local service with service_expose, or",
            "serve/proxy the frontend and API/WebSocket from one public origin, then",
            "verify the public URL with a real browser/client request before finalizing.",
        ],
    }
    entry["public_readiness"] = readiness
    tunnel = entry.get("tunnel")
    if isinstance(tunnel, dict):
        tunnel["public_readiness"] = readiness


def _action_start(params: dict[str, Any]) -> dict[str, Any]:
    name = _safe_name(params.get("name"))
    command = str(params.get("command") or "").strip()
    if not command:
        raise ValueError("command is required for action=start")
    cwd = Path(params.get("cwd") or os.getcwd()).expanduser().resolve()
    if not cwd.is_dir():
        raise ValueError(f"cwd is not a directory: {cwd}")

    host = str(params.get("host") or "127.0.0.1").strip() or "127.0.0.1"
    bind_host = host.strip("[]") if host.startswith("[") and host.endswith("]") else host
    port = params.get("port")
    if port is not None:
        port = int(port)
        if port <= 0:
            port = _find_free_unreserved_port(_connect_host(bind_host))
            params["port"] = port
        elif port in _reserved_preview_ports() and not params.get("allow_reserved_port"):
            return {
                "success": False,
                "error": (
                    f"Port {port} is reserved for another local service in this "
                    "runtime. Use port=0 for generated preview apps, or set "
                    "allow_reserved_port=true only when the user explicitly owns "
                    "that reserved service."
                ),
                "port": port,
                "host": _connect_host(bind_host),
                "port_reserved": True,
                "reserved_ports": sorted(_reserved_preview_ports()),
            }
        elif not _port_is_free(_connect_host(bind_host), port):
            return {
                "success": False,
                "error": (
                    f"Port {port} on {_connect_host(bind_host)} is already in use. "
                    "Choose a different free port or pass port=0 for auto-selection."
                ),
                "port": port,
                "host": _connect_host(bind_host),
                "port_available": False,
            }

    rendered_command = _render_command_placeholders(command, params)

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
    _clear_log(log_path)
    extra_env: dict[str, str] = {}
    if port is not None:
        extra_env["PORT"] = str(port)
    if host:
        extra_env["HOST"] = host
    pid = _spawn_shell(rendered_command, cwd, log_path, extra_env=extra_env)
    local_url = None
    try:
        local_url = _local_url(params)
    except ValueError:
        pass

    entry = {
        "name": name,
        "command": rendered_command,
        "cwd": str(cwd),
        "pid": pid,
        "status": "running" if _pid_alive(pid) else "exited",
        "host": host,
        "port": port,
        "scheme": str(params.get("scheme") or "http"),
        "local_url": local_url,
        "log_path": str(log_path),
        "started_at": _now(),
        "tunnel": None,
    }

    startup_wait = float(params.get("startup_wait_seconds") or 1.5)
    if startup_wait > 0:
        time.sleep(min(startup_wait, 10.0))
        entry["status"] = "running" if _pid_alive(pid) else "exited"
        if entry["status"] != "running":
            services[name] = entry
            _save_registry(registry)
            return {
                "success": False,
                "error": "Service process exited during startup",
                "service": entry,
                "service_log": _tail(str(log_path), int(params.get("tail_chars") or 4000)),
            }

    verify_text = params.get("verify_text")
    if local_url and (verify_text or params.get("start_tunnel")):
        entry["verification"] = _verify_url(
            local_url,
            expected_text=str(verify_text) if verify_text else None,
            wait_seconds=float(params.get("verify_wait_seconds") or 10),
        )
        if (
            not entry["verification"].get("ok")
            and not verify_text
            and port is not None
            and _can_connect(_connect_host(host), int(port))
        ):
            entry["verification"] = {
                "url": local_url,
                "ok": True,
                "matched": False,
                "expected_text": None,
                "method": "tcp-port-probe",
                "http_error": entry["verification"].get("error"),
            }
        if not entry["verification"].get("ok"):
            log_inferred_port = _infer_port_from_log(log_path)
            inferred_port = log_inferred_port or _infer_port_from_process(pid)
            if inferred_port and inferred_port != port:
                port = inferred_port
                params["port"] = inferred_port
                entry["port"] = inferred_port
                entry["local_url"] = _local_url(params)
                entry["port_inferred_from"] = (
                    "service_log" if log_inferred_port == inferred_port else "process_listening_socket"
                )
                local_url = entry["local_url"]
                entry["verification"] = _verify_url(
                    local_url,
                    expected_text=str(verify_text) if verify_text else None,
                    wait_seconds=float(params.get("verify_wait_seconds") or 10),
                )
                if (
                    not entry["verification"].get("ok")
                    and not verify_text
                    and _can_connect(_connect_host(host), int(port))
                ):
                    entry["verification"] = {
                        "url": local_url,
                        "ok": True,
                        "matched": False,
                        "expected_text": None,
                        "method": "tcp-port-probe",
                        "http_error": entry["verification"].get("error"),
                    }
        if not entry["verification"].get("ok"):
            services[name] = entry
            _save_registry(registry)
            return {
                "success": False,
                "error": "Local URL verification failed",
                "service": entry,
                "verification": entry["verification"],
                "service_log": _tail(str(log_path), int(params.get("tail_chars") or 4000)),
            }

    if params.get("start_tunnel"):
        entry["tunnel"] = _start_tunnel_for_entry(name, entry, params)
        if not entry["tunnel"].get("success"):
            services[name] = entry
            _save_registry(registry)
            return {
                "success": False,
                "error": "Cloudflare tunnel did not become reachable",
                "service": _refresh_entry(entry),
                "tunnel": entry["tunnel"],
                "service_log": _tail(str(log_path), int(params.get("tail_chars") or 4000)),
            }
    services[name] = entry
    _apply_public_readiness(name, entry, services)
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
        if stopped:
            tunnel["public_url"] = None
            tunnel["public_url_omitted_reason"] = "stopped"
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
    has_explicit_target = _has_explicit_target(params)
    if not isinstance(entry, dict):
        if not has_explicit_target:
            return {
                "success": False,
                "error": f"Service '{name}' not found; pass url/port or start it first.",
            }
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
    elif has_explicit_target:
        _apply_explicit_target(entry, params)
    else:
        _refresh_entry(entry)
        if entry.get("status") != "running":
            return {
                "success": False,
                "error": f"Service '{name}' is not running; pass url/port or start it first.",
                "service": entry,
            }

    existing_tunnel = entry.get("tunnel")
    if isinstance(existing_tunnel, dict) and _pid_alive(existing_tunnel.get("pid")):
        if not params.get("replace"):
            return {"success": False, "error": f"Tunnel for '{name}' is already running", "service": entry}
        _stop_pid(existing_tunnel.get("pid"))
    retry_after = _tunnel_retry_after(entry)
    if retry_after and not params.get("force"):
        return {
            "success": False,
            "error": (
                "Cloudflare Quick Tunnel is currently rate-limited for this "
                "service. Do not retry in this turn; wait for retry_after_seconds "
                "or use an authenticated/named tunnel."
            ),
            "retry_after_seconds": retry_after,
            "service": _refresh_entry(entry),
        }

    tunnel = _start_tunnel_for_entry(name, entry, params)
    _apply_public_readiness(name, entry, services)
    _save_registry(registry)
    return {"success": bool(tunnel.get("success")), "service": _refresh_entry(entry), "tunnel": tunnel}


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
        tunnel_log = _tail(tunnel.get("log_path"), tail_chars)
        if isinstance(tunnel, dict) and not tunnel.get("public_url"):
            tunnel_log = TRYCLOUDFLARE_RE.sub("<unverified-trycloudflare-url-redacted>", tunnel_log)
            result["tunnel_log_redacted"] = True
            result["tunnel_log_note"] = (
                "Unverified trycloudflare candidate URLs are redacted. "
                "Report only service_expose public_url values from successful tunnel results."
            )
        result["tunnel_log"] = tunnel_log
    return result


def main() -> None:
    try:
        params = _load_params()
        action = _normalize_action(params.get("action"))
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
