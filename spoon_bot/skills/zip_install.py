"""Safe installation helpers for skill zip archives."""

from __future__ import annotations

import re
import shutil
import stat
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


@dataclass(frozen=True)
class InstalledSkillZip:
    """Result of installing a skill archive into workspace/skills."""

    name: str
    source: Path
    target_dir: Path
    skill_md: Path
    file_count: int
    reinstalled: bool


def install_skill_zip_archive(
    archive_path: Path | str,
    workspace: Path | str,
    *,
    name_hint: str | None = None,
    reinstall: bool = True,
) -> InstalledSkillZip | None:
    """Install a zip archive containing a SKILL.md under workspace/skills.

    Returns None when the archive is valid zip but does not contain SKILL.md.
    Raises ValueError for unsafe paths or ambiguous multi-skill archives.
    """
    archive = Path(archive_path).expanduser().resolve()
    workspace_path = Path(workspace).expanduser().resolve()
    skills_dir = workspace_path / "skills"

    with zipfile.ZipFile(archive) as zf:
        entries = _safe_zip_entries(zf)
        skill_roots = _skill_roots(entries)
        if not skill_roots:
            return None
        root = _select_skill_root(skill_roots)
        skill_md_text = _read_skill_md_text(zf, entries, root)
        metadata_name = _extract_skill_metadata_name(skill_md_text)
        skill_name = _derive_skill_name(
            metadata_name or name_hint or archive.stem,
            root,
            prefer_archive_root=metadata_name is None,
        )

        skills_dir.mkdir(parents=True, exist_ok=True)
        target = (skills_dir / skill_name).resolve(strict=False)
        _ensure_within(target, skills_dir)

        tmp_target = (skills_dir / f".installing-{skill_name}-{uuid.uuid4().hex}").resolve(strict=False)
        _ensure_within(tmp_target, skills_dir)
        if tmp_target.exists():
            shutil.rmtree(tmp_target)
        tmp_target.mkdir(parents=True)

        file_count = 0
        try:
            for info, path in entries:
                rel = _relative_archive_path(path, root)
                if rel is None:
                    continue
                destination = (tmp_target / Path(*rel.parts)).resolve(strict=False)
                _ensure_within(destination, tmp_target)
                if info.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as source, destination.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
                _apply_zip_mode(destination, info.external_attr)
                file_count += 1

            skill_md = tmp_target / "SKILL.md"
            if not skill_md.exists():
                raise ValueError("Skill archive did not normalize to a root SKILL.md")

            existed = target.exists()
            if existed:
                if not reinstall:
                    raise FileExistsError(f"Skill already exists: {target}")
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()

            shutil.move(str(tmp_target), str(target))
            return InstalledSkillZip(
                name=skill_name,
                source=archive,
                target_dir=target,
                skill_md=target / "SKILL.md",
                file_count=file_count,
                reinstalled=existed,
            )
        finally:
            if tmp_target.exists():
                shutil.rmtree(tmp_target, ignore_errors=True)


def _safe_zip_entries(zf: zipfile.ZipFile) -> list[tuple[zipfile.ZipInfo, PurePosixPath]]:
    entries: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
    for info in zf.infolist():
        raw_name = str(info.filename or "").replace("\\", "/")
        if not raw_name or raw_name.startswith("__MACOSX/"):
            continue
        path = PurePosixPath(raw_name)
        if _is_unsafe_zip_path(path):
            raise ValueError(f"Unsafe zip entry path: {info.filename}")
        entries.append((info, path))
    return entries


def _is_unsafe_zip_path(path: PurePosixPath) -> bool:
    if path.is_absolute():
        return True
    for part in path.parts:
        if part in {"", ".", ".."} or part.endswith(":"):
            return True
    return False


def _skill_roots(entries: Iterable[tuple[zipfile.ZipInfo, PurePosixPath]]) -> set[PurePosixPath]:
    roots: set[PurePosixPath] = set()
    for info, path in entries:
        if info.is_dir():
            continue
        if path.name == "SKILL.md":
            roots.add(path.parent)
    return roots


def _select_skill_root(skill_roots: set[PurePosixPath]) -> PurePosixPath:
    if PurePosixPath(".") in skill_roots:
        return PurePosixPath(".")
    if len(skill_roots) == 1:
        return next(iter(skill_roots))
    raise ValueError(
        "Skill archive contains multiple SKILL.md roots; attach one skill per zip"
    )


def _read_skill_md_text(
    zf: zipfile.ZipFile,
    entries: Iterable[tuple[zipfile.ZipInfo, PurePosixPath]],
    root: PurePosixPath,
) -> str:
    target = root / "SKILL.md" if str(root) != "." else PurePosixPath("SKILL.md")
    for info, path in entries:
        if path != target:
            continue
        with zf.open(info, "r") as source:
            return source.read(64 * 1024).decode("utf-8", errors="replace")
    return ""


def _extract_skill_metadata_name(text: str) -> str | None:
    for line in text.splitlines()[:80]:
        stripped = line.strip()
        if not stripped or stripped in {"---", "..."}:
            continue
        match = re.match(r"^name\s*:\s*['\"]?([^'\"#\r\n]+)", stripped, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            return value or None
    return None


def _derive_skill_name(
    name_hint: str,
    root: PurePosixPath,
    *,
    prefer_archive_root: bool = True,
) -> str:
    raw = root.name if prefer_archive_root and str(root) != "." else name_hint
    raw = Path(raw).stem.strip()
    raw = re.sub(r"-\d{8}-\d{6}$", "", raw)
    raw = re.sub(r"-[0-9a-fA-F]{7,40}$", "", raw)
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-._").lower()
    if not normalized:
        normalized = "attached-skill"
    return normalized


def _relative_archive_path(path: PurePosixPath, root: PurePosixPath) -> PurePosixPath | None:
    if str(root) == ".":
        return path
    if path == root:
        return None
    try:
        return path.relative_to(root)
    except ValueError:
        return None


def _ensure_within(path: Path, root: Path) -> None:
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Resolved path escapes target directory: {path}") from exc


def _apply_zip_mode(path: Path, external_attr: int) -> None:
    mode = (external_attr >> 16) & 0o777
    if mode:
        try:
            path.chmod(stat.S_IMODE(mode))
        except OSError:
            pass
