"""Tool wrapper for the built-in service_expose skill."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from spoon_ai.tools.base import BaseTool


_SCRIPT = Path(__file__).resolve().parent / "scripts" / "service_expose.py"


class ServiceExposeTool(BaseTool):
    """Start, inspect, and expose local preview services."""

    name: str = "service_expose"
    description: str = (
        "Start, stop, inspect, and Cloudflare-expose local frontend/backend "
        "preview services. Use for local URLs, WebSocket apps, public "
        "links, trycloudflare/cloudflare tunnels, and follow-up requests asking "
        "for the current preview link. Prefer this tool over manual background "
        "shell commands for preview services. Supports free-port selection with "
        "port=0 and app-specific URL verification via verify_text. Only report "
        "a public link when the tool result has success=true and public_url is "
        "non-null; trycloudflare URLs in logs are candidates, not verified links. "
        "For WebSocket-only services, HTTP 426 or TCP reachability can be valid "
        "when no verify_text was requested. If the result contains "
        "public_readiness.blocking=true, expose the missing browser dependencies "
        "or route them through the same public origin before finalizing. For "
        "browser apps with WebSocket/API dependencies, prefer one service that "
        "serves the page and upgrades WebSocket/API paths on the same public origin."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
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
                ],
                "description": "Action to perform. expose/start_tunnel alias tunnel; inspect aliases status.",
            },
            "name": {
                "type": "string",
                "description": "Stable service name.",
            },
            "command": {
                "type": "string",
                "description": "Command to run for action=start.",
            },
            "cwd": {
                "type": "string",
                "description": "Working directory for the service command.",
            },
            "port": {
                "type": "integer",
                "description": "Local port to expose. Use 0 to auto-select a free port.",
            },
            "host": {
                "type": "string",
                "description": "Local host. Defaults to 127.0.0.1.",
            },
            "scheme": {
                "type": "string",
                "description": "URL scheme. Defaults to http.",
            },
            "url": {
                "type": "string",
                "description": "Full local URL to expose, for example http://127.0.0.1:3000.",
            },
            "start_tunnel": {
                "type": "boolean",
                "description": "Start Cloudflare tunnel after service start.",
            },
            "replace": {
                "type": "boolean",
                "description": "Stop and replace an existing process/tunnel with the same name.",
            },
            "verify_text": {
                "type": "string",
                "description": (
                    "App-specific text that must appear in the local/public HTTP "
                    "response before the URL is reported."
                ),
            },
            "verify_wait_seconds": {
                "type": "integer",
                "description": "Seconds to wait for HTTP verification.",
            },
            "startup_wait_seconds": {
                "type": "number",
                "description": "Seconds to wait after start before checking for early exit.",
            },
            "tunnel_wait_seconds": {
                "type": "integer",
                "description": "Seconds to wait for a trycloudflare URL.",
            },
            "tunnel_protocol": {
                "type": "string",
                "description": (
                    "cloudflared transport protocol. Defaults to http2 for "
                    "Docker/Linux networks where QUIC/UDP may be unavailable."
                ),
            },
            "tunnel_attempts": {
                "type": "integer",
                "description": (
                    "Number of transient tunnel verification attempts. Defaults "
                    "to 3 and is capped at 5."
                ),
            },
            "tunnel_public_settle_seconds": {
                "type": "number",
                "description": (
                    "Seconds to wait after cloudflared registration before the "
                    "first public URL verification. Defaults to 8 to avoid "
                    "negative DNS caching on new trycloudflare names."
                ),
            },
            "tail_chars": {
                "type": "integer",
                "description": "Log tail size for logs or failures.",
            },
            "target": {
                "type": "string",
                "enum": ["service", "tunnel", "all"],
                "description": "Log target for action=logs.",
            },
        },
        "required": ["action"],
    }

    async def execute(self, **kwargs: Any) -> str:
        timeout = 90
        action = str(kwargs.get("action") or "").strip().lower()
        if action in {"start", "tunnel", "expose", "start_tunnel"} or kwargs.get("start_tunnel"):
            tunnel_wait = int(kwargs.get("tunnel_wait_seconds") or 60)
            verify_wait = int(kwargs.get("verify_wait_seconds") or 20)
            try:
                attempts = int(kwargs.get("tunnel_attempts") or 3)
            except (TypeError, ValueError):
                attempts = 3
            attempts = max(1, min(attempts, 5))
            startup_wait = int(float(kwargs.get("startup_wait_seconds") or 2))
            timeout = max(timeout, attempts * (tunnel_wait + verify_wait + 5) + startup_wait + 30)
        payload = {k: v for k, v in kwargs.items() if v is not None}

        def _run() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                [sys.executable, str(_SCRIPT)],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(Path(os.environ.get("SPOON_BOT_WORKSPACE_PATH") or os.getcwd())),
                env=os.environ.copy(),
            )

        try:
            proc = await asyncio.to_thread(_run)
        except subprocess.TimeoutExpired as exc:
            return json.dumps(
                {
                    "success": False,
                    "error": f"service_expose timed out after {timeout}s",
                    "stdout": exc.stdout or "",
                    "stderr": exc.stderr or "",
                },
                ensure_ascii=True,
            )
        if proc.returncode != 0:
            return json.dumps(
                {
                    "success": False,
                    "error": f"service_expose exited with code {proc.returncode}",
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                },
                ensure_ascii=True,
            )
        return proc.stdout.strip() or json.dumps({"success": True}, ensure_ascii=True)
