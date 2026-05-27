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


APP_NAME = os.environ.get("MODAL_SPOON_BOT_APP", "spoon-bot-deepseek-v4")
VOLUME_NAME = os.environ.get("MODAL_SPOON_BOT_VOLUME", f"{APP_NAME}-data")
SECRET_NAME = os.environ.get("MODAL_SPOON_BOT_SECRET", "spoon-bot-openrouter")
RUNTIME_SECRET_KEYS = (
    "TAVILY_API_KEY",
    "SANDBOX_PROVIDER",
    "SPOON_BOT_YOLO_MODE",
)
runtime_secret_values = {
    key: value
    for key in RUNTIME_SECRET_KEYS
    if (value := os.environ.get(key))
}

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
    "GATEWAY_TIMEOUT_REQUEST_MS": "900000",
    "GATEWAY_TIMEOUT_STREAM_MS": "900000",
    "SPOON_BOT_CONFIG": "/app/modal_config.yaml",
    "SPOON_BOT_DEFAULT_PROVIDER": "openrouter",
    "SPOON_BOT_DEFAULT_MODEL": "deepseek/deepseek-v4-pro",
    "SPOON_BOT_WORKSPACE_PATH": "/modal-data/workspace",
    "SPOON_BOT_WALLET_PATH": "/modal-data/agent-wallet",
    "SANDBOX_PROVIDER": "modal",
    "SPOON_BOT_YOLO_MODE": "true",
    "SPOON_BOT_MAX_ITERATIONS": "20",
    "SPOON_BOT_SHELL_TIMEOUT": "900",
    "SPOON_BOT_MAX_OUTPUT": "12000",
    "SPOON_BOT_LOG_LEVEL": "INFO",
    "PYTHONUNBUFFERED": "1",
})

app = modal.App(APP_NAME, image=image)
data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
function_secrets = [modal.Secret.from_name(SECRET_NAME)]
if runtime_secret_values:
    function_secrets.append(modal.Secret.from_dict(runtime_secret_values))


@app.function(
    secrets=function_secrets,
    volumes={"/modal-data": data_volume},
    timeout=900,
    scaledown_window=900,
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
