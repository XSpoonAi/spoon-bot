"""Modal deployment entrypoint for spoon-bot Gateway.

Deploy with:

    modal deploy modal_spoon_bot.py

Required Modal secret:

    modal secret create spoon-bot-openrouter OPENROUTER_API_KEY=...

For a fresh game identity, deploy with a fresh volume name:

    $env:MODAL_SPOON_BOT_VOLUME = "spoon-bot-deepseek-v4-$(Get-Date -Format yyyyMMddHHmmss)"
    modal deploy modal_spoon_bot.py
"""

from __future__ import annotations

import os
from pathlib import Path

import modal


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

APP_NAME = os.environ.get("MODAL_SPOON_BOT_APP", "spoon-bot-deepseek-v4")
VOLUME_NAME = os.environ.get("MODAL_SPOON_BOT_VOLUME", f"{APP_NAME}-data")
SECRET_NAME = os.environ.get("MODAL_SPOON_BOT_SECRET", "spoon-bot-openrouter")
FUNCTION_TIMEOUT = int(os.environ.get("MODAL_SPOON_BOT_FUNCTION_TIMEOUT", "86400"))
image = modal.Image.from_dockerfile(
    "Dockerfile.modal",
    context_dir=".",
    ignore=[
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "sessions",
    ],
).env({
    "GATEWAY_AUTH_REQUIRED": "false",
    "SPOON_BOT_CONFIG": "/app/modal_config.yaml",
    "SPOON_BOT_DEFAULT_PROVIDER": os.environ.get("SPOON_BOT_DEFAULT_PROVIDER", "openrouter"),
    "SPOON_BOT_DEFAULT_MODEL": os.environ.get(
        "SPOON_BOT_DEFAULT_MODEL",
        "deepseek/deepseek-v4-pro",
    ),
    "OPENROUTER_BASE_URL": os.environ.get("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL),
    "SPOON_BOT_WORKSPACE_PATH": "/modal-data/workspace",
    "SPOON_BOT_WALLET_PATH": "/modal-data/agent-wallet",
    "SANDBOX_PROVIDER": "modal",
    "SPOON_BOT_YOLO_MODE": "true",
    "SPOON_BOT_MAX_ITERATIONS": "100",
    "SPOON_BOT_CONTEXT_WINDOW": os.environ.get("SPOON_BOT_CONTEXT_WINDOW", "1000000"),
    "SPOON_BOT_SHELL_TIMEOUT": "3600",
    "SPOON_BOT_SHELL_MAX_TIMEOUT": "7200",
    "SPOON_BOT_SERVICE_EXPOSE_RESERVED_PORTS": os.environ.get(
        "SPOON_BOT_SERVICE_EXPOSE_RESERVED_PORTS",
        "3000",
    ),
    "SPOON_BOT_PROVIDER_TOTAL_TIMEOUT": "0",
    "SPOON_BOT_PROVIDER_ASK_TIMEOUT": os.environ.get(
        "SPOON_BOT_PROVIDER_ASK_TIMEOUT",
        "0",
    ),
    "SPOON_BOT_TOOL_FOLLOWUP_TIMEOUT": os.environ.get(
        "SPOON_BOT_TOOL_FOLLOWUP_TIMEOUT",
        "600",
    ),
    "SPOON_BOT_NON_SHELL_TOOL_ACTIVE_TIMEOUT": os.environ.get(
        "SPOON_BOT_NON_SHELL_TOOL_ACTIVE_TIMEOUT",
        "120",
    ),
    "SPOON_BOT_POST_TOOL_RESULT_SILENCE_TIMEOUT": os.environ.get(
        "SPOON_BOT_POST_TOOL_RESULT_SILENCE_TIMEOUT",
        "30",
    ),
    "SPOON_BOT_OPENROUTER_REQUIRED_TOOL_CHOICE": os.environ.get(
        "SPOON_BOT_OPENROUTER_REQUIRED_TOOL_CHOICE",
        "false",
    ),
    "SPOON_BOT_INTERNAL_RECOVERY_TIMEOUT": os.environ.get(
        "SPOON_BOT_INTERNAL_RECOVERY_TIMEOUT",
        "180",
    ),
    "SPOON_BOT_CONTEXT_SNAPSHOT_ENABLED": os.environ.get(
        "SPOON_BOT_CONTEXT_SNAPSHOT_ENABLED",
        "true",
    ),
    "SPOON_BOT_CONTEXT_SNAPSHOT_MAX_CHARS": os.environ.get(
        "SPOON_BOT_CONTEXT_SNAPSHOT_MAX_CHARS",
        "60000",
    ),
    "SPOON_BOT_MAX_OUTPUT": os.environ.get("SPOON_BOT_MAX_OUTPUT", "80000"),
    "GATEWAY_TIMEOUT_REQUEST_MS": "0",
    "GATEWAY_TIMEOUT_STREAM_MS": "0",
    "SPOON_BOT_LOG_LEVEL": "INFO",
    "PYTHONUNBUFFERED": "1",
})

app = modal.App(APP_NAME, image=image)
data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
function_secrets = [modal.Secret.from_name(SECRET_NAME)]


@app.function(
    secrets=function_secrets,
    volumes={"/modal-data": data_volume},
    timeout=FUNCTION_TIMEOUT,
    max_containers=1,
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app(label="gateway")
def gateway():
    """Return the spoon-bot FastAPI gateway as a Modal ASGI Web Function."""
    os.makedirs("/modal-data/workspace", exist_ok=True)
    wallet_dir = Path("/modal-data/agent-wallet")
    wallet_dir.mkdir(parents=True, exist_ok=True)

    legacy_wallet_dir = Path.home() / ".agent-wallet"
    if not legacy_wallet_dir.exists():
        legacy_wallet_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            legacy_wallet_dir.symlink_to(wallet_dir, target_is_directory=True)
        except OSError:
            legacy_wallet_dir.mkdir(parents=True, exist_ok=True)

    from spoon_bot.gateway.server import create_app

    return create_app()
