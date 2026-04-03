"""Helpers for managing built-in skills shipped with spoon-bot."""

from __future__ import annotations

import shutil
from pathlib import Path


def builtin_skills_root() -> Path:
    """Return the repository directory that stores built-in skill templates."""
    return Path(__file__).resolve().parent / "builtin"


def ensure_builtin_skills(workspace: Path | str) -> list[Path]:
    """Install missing built-in skills into the target workspace.

    Existing workspace skills are left untouched so user-local edits win.
    Returns the list of newly copied skill directories.
    """
    workspace_path = Path(workspace).expanduser().resolve()
    skills_dir = workspace_path / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    installed: list[Path] = []
    source_root = builtin_skills_root()
    if not source_root.is_dir():
        return installed

    for source in sorted(source_root.iterdir()):
        if not source.is_dir() or not (source / "SKILL.md").exists():
            continue
        target = skills_dir / source.name
        if target.exists():
            continue
        shutil.copytree(source, target)
        installed.append(target)

    return installed
