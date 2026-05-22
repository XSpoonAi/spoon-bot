from __future__ import annotations

import json
import io
import os
import socket
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPT = (
    PROJECT_ROOT
    / "spoon_bot"
    / "skills"
    / "builtin"
    / "service_expose"
    / "scripts"
    / "service_expose.py"
)


def run_service_script(state_dir: Path, payload: dict) -> dict:
    env = {"SPOON_BOT_SERVICE_EXPOSE_DIR": str(state_dir)}
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=20,
        cwd=str(PROJECT_ROOT),
        env={**dict(os.environ), **env},
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def run_service_script_argv(state_dir: Path, payload: dict) -> dict:
    env = {"SPOON_BOT_SERVICE_EXPOSE_DIR": str(state_dir)}
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), json.dumps(payload)],
        capture_output=True,
        text=True,
        timeout=20,
        cwd=str(PROJECT_ROOT),
        env={**dict(os.environ), **env},
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_service_expose_start_status_logs_and_stop(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    command = subprocess.list2cmdline([
        sys.executable,
        "-c",
        "import time; print('service-ready', flush=True); time.sleep(60)",
    ])

    started = run_service_script(
        state_dir,
        {
            "action": "start",
            "name": "test-service",
            "command": command,
            "cwd": str(PROJECT_ROOT),
            "port": 18765,
        },
    )
    assert started["success"] is True
    assert started["service"]["status"] == "running"

    try:
        status = run_service_script(state_dir, {"action": "status", "name": "test-service"})
        assert status["success"] is True
        assert status["service"]["local_url"] == "http://127.0.0.1:18765"

        logs = run_service_script(
            state_dir,
            {"action": "logs", "name": "test-service", "tail_chars": 2000},
        )
        assert logs["success"] is True
        assert "service_log" in logs
    finally:
        stopped = run_service_script(state_dir, {"action": "stop", "name": "test-service"})
        assert stopped["success"] is True
        assert stopped["service"]["status"] == "stopped"


def test_service_expose_rejects_occupied_port(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        port = int(sock.getsockname()[1])

        result = run_service_script(
            state_dir,
            {
                "action": "start",
                "name": "occupied-port",
                "command": subprocess.list2cmdline([sys.executable, "-c", "import time; time.sleep(5)"]),
                "cwd": str(PROJECT_ROOT),
                "port": port,
            },
        )

    assert result["success"] is False
    assert result["port_available"] is False
    assert "already in use" in result["error"]


def test_service_expose_port_probe_detects_listening_socket() -> None:
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("service_expose_script", SCRIPT)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        port = int(sock.getsockname()[1])

        assert module._port_is_free("127.0.0.1", port) is False


def test_service_expose_local_url_uses_loopback_for_wildcard_host(tmp_path: Path) -> None:
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("service_expose_script", SCRIPT)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._local_url({"port": 12345, "host": "0.0.0.0"}) == "http://127.0.0.1:12345"
    assert module._local_url({"port": 12345, "host": "::"}) == "http://127.0.0.1:12345"


def test_service_expose_accepts_json_payload_from_argv(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"

    result = run_service_script_argv(state_dir, {"action": "list"})

    assert result["success"] is True
    assert result["services"] == {}


def test_service_expose_normalizes_action_aliases() -> None:
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("service_expose_script", SCRIPT)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._normalize_action("expose") == "tunnel"
    assert module._normalize_action("start_tunnel") == "tunnel"
    assert module._normalize_action("inspect") == "status"


def test_service_expose_refuses_stopped_entry_without_explicit_target(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "registry.json").write_text(
        json.dumps(
            {
                "services": {
                    "stale-preview": {
                        "name": "stale-preview",
                        "command": None,
                        "cwd": str(PROJECT_ROOT),
                        "pid": None,
                        "status": "stopped",
                        "host": "127.0.0.1",
                        "port": 3000,
                        "scheme": "http",
                        "local_url": "http://127.0.0.1:3000",
                        "log_path": None,
                        "started_at": "2026-01-01T00:00:00Z",
                        "tunnel": {
                            "pid": None,
                            "local_url": "http://127.0.0.1:3000",
                            "public_url": "https://old-preview.trycloudflare.com",
                            "status": "stopped",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = run_service_script(state_dir, {"action": "tunnel", "name": "stale-preview"})

    assert result["success"] is False
    assert "not running" in result["error"]
    assert "url/port" in result["error"]


def test_service_expose_tool_exposes_structured_schema() -> None:
    from importlib.util import module_from_spec, spec_from_file_location

    tools_path = PROJECT_ROOT / "spoon_bot" / "skills" / "builtin" / "service_expose" / "tools.py"
    spec = spec_from_file_location("service_expose_tools", tools_path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    tool = module.ServiceExposeTool()
    assert tool.name == "service_expose"
    assert "manual background shell commands" in tool.description
    assert "port" in tool.parameters["properties"]
    assert "verify_text" in tool.parameters["properties"]
    assert tool.parameters["properties"]["action"]["enum"] == [
        "start",
        "tunnel",
        "expose",
        "start_tunnel",
        "status",
        "list",
        "inspect",
        "logs",
        "stop",
        "stop_tunnel",
    ]


def test_service_expose_parses_cloudflare_url_from_tunnel_log(tmp_path: Path) -> None:
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("service_expose_script", SCRIPT)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    log_path = tmp_path / "cloudflared.log"
    log_path.write_text(
        "INFO Requesting new quick Tunnel on trycloudflare.com...\n"
        "INF +--------------------------------------------------------------------------------------------+\n"
        "INF |  Your quick Tunnel has been created! Visit it at https://sample-demo.trycloudflare.com  |\n",
        encoding="utf-8",
    )
    assert module._parse_public_url(log_path) == "https://sample-demo.trycloudflare.com"


def test_service_expose_infers_port_from_service_log(tmp_path: Path) -> None:
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("service_expose_script", SCRIPT)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    log_path = tmp_path / "service.log"
    log_path.write_text(
        "WebSocket Chat Server running on port 3456\n",
        encoding="utf-8",
    )

    assert module._infer_port_from_log(log_path) == 3456


def test_service_expose_clears_old_tunnel_log_before_reuse(tmp_path: Path) -> None:
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("service_expose_script", SCRIPT)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    log_path = tmp_path / "cloudflared.log"
    log_path.write_text("old https://old-preview.trycloudflare.com\n", encoding="utf-8")

    module._clear_log(log_path)

    assert log_path.read_text(encoding="utf-8") == ""
    assert module._parse_public_url(log_path) is None


def test_service_expose_auto_downloads_cloudflared_when_missing(tmp_path: Path, monkeypatch) -> None:
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("service_expose_script", SCRIPT)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.delenv("CLOUDFLARED_PATH", raising=False)
    monkeypatch.setenv("SPOON_BOT_AUTO_INSTALL_CLOUDFLARED", "1")
    monkeypatch.setattr(module, "_state_root", lambda: tmp_path)
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(module.platform, "machine", lambda: "x86_64")
    requested_urls: list[str] = []

    def fake_urlopen(req, timeout=0):
        requested_urls.append(req.full_url)
        return FakeResponse(b"#!/bin/sh\n")

    monkeypatch.setattr(module, "urlopen", fake_urlopen)

    path, error = module._resolve_cloudflared()

    assert error is None
    assert path == str(tmp_path / "bin" / "cloudflared")
    assert Path(path).exists()
    assert requested_urls == [
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    ]


def test_service_expose_hides_unverified_public_url(monkeypatch) -> None:
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("service_expose_script", SCRIPT)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    stopped: list[int] = []
    monkeypatch.setenv("CLOUDFLARED_PATH", "cloudflared")
    monkeypatch.setattr(module, "_spawn_argv", lambda *args, **kwargs: 12345)
    monkeypatch.setattr(module, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(module, "_parse_public_url", lambda log_path: "https://bad-preview.trycloudflare.com")
    monkeypatch.setattr(module, "_tunnel_registered", lambda log_path: True)
    monkeypatch.setattr(
        module,
        "_verify_url",
        lambda *args, **kwargs: {"ok": False, "error": "verification failed"},
    )
    monkeypatch.setattr(module, "_stop_pid", lambda pid: stopped.append(pid) or True)

    result = module._start_tunnel_for_entry(
        "demo",
        {"local_url": "http://127.0.0.1:1234"},
        {"verify_text": "READY", "verify_wait_seconds": 0, "tunnel_wait_seconds": 1},
    )

    assert result["success"] is False
    assert result["public_url"] is None
    assert result["public_url_omitted_reason"] == "unverified"
    assert stopped == [12345]


def test_service_expose_requires_registered_tunnel(monkeypatch) -> None:
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("service_expose_script", SCRIPT)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    stopped: list[int] = []
    monkeypatch.setenv("CLOUDFLARED_PATH", "cloudflared")
    monkeypatch.setattr(module, "_spawn_argv", lambda *args, **kwargs: 12345)
    monkeypatch.setattr(module, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(module, "_parse_public_url", lambda log_path: "https://early.trycloudflare.com")
    monkeypatch.setattr(module, "_tunnel_registered", lambda log_path: False)
    monkeypatch.setattr(module, "_verify_url", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(module, "_stop_pid", lambda pid: stopped.append(pid) or True)

    result = module._start_tunnel_for_entry(
        "demo",
        {"local_url": "http://127.0.0.1:1234"},
        {"tunnel_wait_seconds": 1},
    )

    assert result["success"] is False
    assert result["public_url"] is None
    assert result["public_url_omitted_reason"] == "unverified"
    assert stopped == [12345]


def test_service_expose_reuses_local_verification_text_for_tunnel(monkeypatch) -> None:
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("service_expose_script", SCRIPT)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    verified: list[str | None] = []
    monkeypatch.setenv("CLOUDFLARED_PATH", "cloudflared")
    monkeypatch.setattr(module, "_spawn_argv", lambda *args, **kwargs: 12345)
    monkeypatch.setattr(module, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(module, "_parse_public_url", lambda log_path: "https://ok-preview.trycloudflare.com")
    monkeypatch.setattr(module, "_tunnel_registered", lambda log_path: True)

    def _verify(*args, **kwargs):
        verified.append(kwargs.get("expected_text"))
        return {"ok": True, "matched": True}

    monkeypatch.setattr(module, "_verify_url", _verify)

    result = module._start_tunnel_for_entry(
        "demo",
        {
            "local_url": "http://127.0.0.1:1234",
            "verification": {"expected_text": "READY"},
        },
        {"tunnel_wait_seconds": 1},
    )

    assert result["success"] is True
    assert result["public_url"] == "https://ok-preview.trycloudflare.com"
    assert verified == ["READY"]
