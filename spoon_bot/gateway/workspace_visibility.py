"""Shared visibility policy for workspace file listings."""

from __future__ import annotations

import os
from pathlib import Path

SYSTEM_ENTRY_NAMES = frozenset({"__pycache__", "node_modules", ".git"})
_DEFAULT_WALLET_ROOT_NAME = ".agent-wallet"
_WALLET_ENV_VARS = ("SPOON_BOT_WALLET_PATH", "AGENT_WALLET_DIR")


def configured_wallet_runtime_names() -> set[str]:
    """Return wallet runtime directory names that should stay out of file views."""
    names = {
        _DEFAULT_WALLET_ROOT_NAME,
        f".home-{_DEFAULT_WALLET_ROOT_NAME}",
        ".home-agent-wallet",
    }
    for env_name in _WALLET_ENV_VARS:
        raw = os.environ.get(env_name, "").strip()
        if not raw:
            continue
        wallet_name = Path(raw).expanduser().name
        if not wallet_name:
            continue
        names.add(wallet_name)
        names.add(f".home-{wallet_name}")
    return names


def is_workspace_wallet_runtime_path(
    path: Path,
    *,
    workspace_root: Path | None = None,
) -> bool:
    """Return whether *path* is the wallet runtime or one of its descendants."""
    try:
        candidate = path.resolve(strict=False)
        if workspace_root is not None:
            relative = candidate.relative_to(workspace_root.resolve(strict=False))
            parts = relative.parts or (candidate.name,)
        else:
            parts = candidate.parts
    except (OSError, RuntimeError, ValueError):
        parts = path.parts

    runtime_names = configured_wallet_runtime_names()
    return any(part in runtime_names for part in parts)


def should_hide_workspace_path(
    path: Path,
    *,
    workspace_root: Path | None = None,
    include_hidden: bool = False,
) -> bool:
    """Return whether *path* should be omitted from user-visible workspace views."""
    if is_workspace_wallet_runtime_path(path, workspace_root=workspace_root):
        return True
    if path.name in SYSTEM_ENTRY_NAMES:
        return True
    return not include_hidden and path.name.startswith(".")
