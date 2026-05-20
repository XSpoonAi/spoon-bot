"""Helpers for managing built-in skills shipped with spoon-bot."""

from __future__ import annotations

import shutil
from pathlib import Path


_BUILTIN_MANIFEST = ".spoon-bot-builtin"


def builtin_skills_root() -> Path:
    """Return the repository directory that stores built-in skill templates."""
    return Path(__file__).resolve().parent / "builtin"


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _is_refreshable_builtin_copy(source: Path, target: Path) -> bool:
    """Return True when an existing workspace skill is a managed built-in copy."""
    if not target.is_dir():
        return False
    if (target / _BUILTIN_MANIFEST).exists():
        return True
    source_skill = _read_text(source / "SKILL.md")
    target_skill = _read_text(target / "SKILL.md")
    return source_skill is not None and source_skill == target_skill


def _builtin_source_files_differ(source: Path, target: Path) -> bool:
    """Return True when any bundled file is missing or different in the target."""
    for item in source.rglob("*"):
        if not item.is_file():
            continue
        dest = target / item.relative_to(source)
        if not dest.is_file():
            return True
        try:
            if item.read_bytes() != dest.read_bytes():
                return True
        except OSError:
            return True
    return False


def _needs_builtin_refresh(source: Path, target: Path) -> bool:
    """Return True when a managed built-in copy should be refreshed."""
    if not _is_refreshable_builtin_copy(source, target):
        return False
    if not (target / _BUILTIN_MANIFEST).exists():
        return True
    return _builtin_source_files_differ(source, target)


def _merge_builtin_skill(source: Path, target: Path) -> None:
    """Copy built-in files over an existing managed skill without deleting extras."""
    target.mkdir(parents=True, exist_ok=True)
    for item in source.rglob("*"):
        rel = item.relative_to(source)
        dest = target / rel
        if item.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, dest)
    (target / _BUILTIN_MANIFEST).write_text(source.name, encoding="utf-8")


def ensure_builtin_skills(workspace: Path | str) -> list[Path]:
    """Install or refresh managed built-in skills into the target workspace.

    Existing user-customized skills are left untouched. A workspace skill is
    considered a managed built-in copy when it has the built-in manifest or
    when its SKILL.md still matches the bundled template exactly.
    Returns the list of copied or refreshed skill directories.
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
            if _needs_builtin_refresh(source, target):
                _merge_builtin_skill(source, target)
                installed.append(target)
            continue
        shutil.copytree(source, target)
        (target / _BUILTIN_MANIFEST).write_text(source.name, encoding="utf-8")
        installed.append(target)

    return installed
