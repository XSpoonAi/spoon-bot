from __future__ import annotations

import json
import os
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
