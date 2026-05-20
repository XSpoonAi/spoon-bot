"""Skill marketplace tool — search, install, remove skills from GitHub / skills.sh."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from spoon_ai.tools.base import BaseTool

logger = logging.getLogger(__name__)

# GitHub URL pattern: https://github.com/owner/repo[/tree/branch[/path]]
_GITHUB_URL_RE = re.compile(
    r"https?://github\.com/"
    r"(?P<owner>[^/\s]+)/(?P<repo>[^/\s]+)"
    r"(?:/tree/(?P<branch>[^/\s]+)(?:/(?P<path>[^\s]*))?)?"
    r"\s*$"
)
_SKILLS_SH_API = "https://skills.sh/api/search"
_SKILL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")


def _get_workspace() -> Path:
    """Resolve the workspace directory."""
    ws = os.environ.get("SPOON_BOT_WORKSPACE_PATH")
    if ws:
        return Path(ws)
    return Path.home() / ".spoon-bot" / "workspace"


def _normalize_subpath(raw: str) -> str:
    """Normalize a remote skill path into a stable POSIX-like form."""
    subpath = raw.strip().replace("\\", "/").strip("/")
    while "//" in subpath:
        subpath = subpath.replace("//", "/")
    return subpath


def _validate_subpath(raw: str, *, allow_empty: bool = False) -> str:
    """Validate a remote skill path and optionally allow repo root installs."""
    subpath = _normalize_subpath(raw)
    if not subpath:
        if allow_empty:
            return ""
        raise ValueError("Invalid skill path")

    parts = [p for p in subpath.split("/") if p]
    if not parts:
        raise ValueError("Invalid skill path")
    if any(p in {".", ".."} for p in parts):
        raise ValueError("Invalid skill path: path traversal is not allowed")
    if any(p.startswith(".") for p in parts):
        raise ValueError("Invalid skill path: hidden segments are not allowed")

    return "/".join(parts)


def _looks_like_github_repo_reference(value: str) -> bool:
    """Return True for GitHub URLs or ``owner/repo`` shorthand, not local filesystem paths."""
    candidate = (value or "").strip()
    if not candidate:
        return False
    if "github.com" in candidate:
        return True
    if candidate.startswith(("/", "\\", ".", "~")):
        return False
    if re.match(r"^[A-Za-z]:[\\/]", candidate):
        return False
    return bool(re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/.*)?$", candidate))


def _parse_github_url(url: str) -> tuple[str, str, str, str]:
    """Parse a GitHub URL -> (owner, repo, branch, subpath)."""
    m = _GITHUB_URL_RE.match(url.strip())
    if not m:
        raise ValueError(f"Not a recognised GitHub URL: {url!r}")
    return (
        m.group("owner"),
        m.group("repo"),
        m.group("branch") or "main",
        (m.group("path") or "").rstrip("/"),
    )


def _is_safe_skill_name(value: str) -> bool:
    """Return True when a frontmatter name is safe as a workspace directory."""
    name = value.strip()
    return bool(_SKILL_NAME_RE.fullmatch(name)) and ".." not in name


def _read_skill_name_from_frontmatter(skill_md: Path, fallback: str) -> str:
    """Read ``name:`` from SKILL.md frontmatter, falling back to the path-derived name."""
    try:
        text = skill_md.read_text(encoding="utf-8")
    except OSError:
        return fallback

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return fallback

    frontmatter: list[str] = []
    for line in lines[1:]:
        if line.strip() == "---":
            break
        frontmatter.append(line)
    else:
        return fallback

    for line in frontmatter:
        match = re.match(r"^\s*name\s*:\s*(.+?)\s*$", line)
        if not match:
            continue
        raw_name = match.group(1).split("#", 1)[0].strip()
        skill_name = raw_name.strip("'\"")
        if _is_safe_skill_name(skill_name):
            return skill_name

    return fallback


def _github_headers() -> dict[str, str]:
    """Build GitHub API headers, including auth token if available."""
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _build_git_clone_command(clone_url: str, branch: str, destination: Path, subpath: str) -> list[str]:
    """Build the git clone command for either repo-root or nested skill installs."""
    cmd = [
        "git",
        "clone",
        "--depth=1",
        "--filter=blob:none",
        "--branch",
        branch,
    ]
    if subpath:
        cmd.append("--sparse")
    cmd.extend([clone_url, str(destination)])
    return cmd


def _select_skill_root_subpath(skill_md_paths: list[str], requested_subpath: str = "") -> str:
    """Choose the directory that should be installed based on ``SKILL.md`` location."""
    requested = _normalize_subpath(requested_subpath)
    skill_roots: list[str] = []
    seen: set[str] = set()

    for raw_path in skill_md_paths:
        normalized = _normalize_subpath(raw_path)
        if not normalized or normalized.endswith("/SKILL.md") is False and normalized != "SKILL.md":
            continue
        root = normalized[:-len("/SKILL.md")] if normalized != "SKILL.md" else ""
        if root not in seen:
            seen.add(root)
            skill_roots.append(root)

    if requested:
        skill_roots = [
            root
            for root in skill_roots
            if root == requested or root.startswith(requested + "/")
        ]

    if not skill_roots:
        scope = requested or "repository root"
        raise RuntimeError(f"No SKILL.md found under {scope}")

    if "" in skill_roots:
        return ""

    if requested and requested in skill_roots:
        return requested

    if len(skill_roots) == 1:
        return skill_roots[0]

    candidates = ", ".join(sorted(skill_roots))
    raise RuntimeError(
        "Multiple skill roots found; provide a GitHub tree URL to the exact skill folder. "
        f"Candidates: {candidates}"
    )


async def _resolve_github_skill_source(
    owner: str,
    repo: str,
    branch: str,
    subpath: str,
) -> tuple[str, str, str]:
    """Resolve the branch, skill root subpath, and installed skill name for a GitHub repo."""
    import httpx

    async def _fetch_repo_tree_paths(client: httpx.AsyncClient, candidate_branch: str) -> list[str]:
        resp = await client.get(
            f"https://api.github.com/repos/{owner}/{repo}/git/trees/{candidate_branch}",
            params={"recursive": "1"},
            headers=_github_headers(),
        )
        resp.raise_for_status()
        payload = resp.json()
        tree = payload.get("tree", [])
        if not isinstance(tree, list):
            raise RuntimeError("GitHub tree API returned an unexpected payload")
        return [
            str(item.get("path", ""))
            for item in tree
            if isinstance(item, dict)
            and item.get("type") == "blob"
            and str(item.get("path", "")).endswith("SKILL.md")
        ]

    requested_subpath = _normalize_subpath(subpath)
    resolved_branch = branch

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                skill_md_paths = await _fetch_repo_tree_paths(client, resolved_branch)
            except httpx.HTTPStatusError as exc:
                should_retry_default_branch = exc.response.status_code == 404 and resolved_branch == "main"
                if not should_retry_default_branch:
                    raise
                repo_resp = await client.get(
                    f"https://api.github.com/repos/{owner}/{repo}",
                    headers=_github_headers(),
                )
                repo_resp.raise_for_status()
                repo_data = repo_resp.json()
                fallback_branch = str(repo_data.get("default_branch") or "").strip()
                if not fallback_branch or fallback_branch == resolved_branch:
                    raise
                resolved_branch = fallback_branch
                skill_md_paths = await _fetch_repo_tree_paths(client, resolved_branch)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to resolve skill root for {owner}/{repo}@{branch}: {exc}"
        ) from exc

    resolved_subpath = _select_skill_root_subpath(skill_md_paths, requested_subpath)
    skill_name = resolved_subpath.rsplit("/", 1)[-1] if resolved_subpath else repo
    return resolved_branch, resolved_subpath, skill_name


async def _download_via_git(
    owner: str, repo: str, branch: str, subpath: str, target: Path
) -> int:
    """Download skill files using git sparse-checkout. Returns file count.

    This avoids GitHub API rate limits by using git protocol directly.
    """
    clone_url = f"https://github.com/{owner}/{repo}.git"
    tmp_dir = Path(tempfile.mkdtemp(prefix="spoon_skill_"))

    try:
        # Nested skills use sparse checkout; repo-root skills clone fully.
        clone_cmd = _build_git_clone_command(
            clone_url,
            branch,
            tmp_dir / "repo",
            subpath,
        )
        proc = await asyncio.create_subprocess_exec(
            *clone_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
        if proc.returncode != 0:
            raise RuntimeError(f"git clone failed: {stderr.decode().strip()}")

        repo_dir = tmp_dir / "repo"

        if subpath:
            proc = await asyncio.create_subprocess_exec(
                "git", "sparse-checkout", "set", subpath,
                cwd=str(repo_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"git sparse-checkout failed: {stderr.decode().strip()}"
                )

        # Copy skill files to target
        source = repo_dir / subpath if subpath else repo_dir
        if not source.is_dir():
            raise RuntimeError(f"Path '{subpath}' not found in {owner}/{repo}@{branch}")

        target.mkdir(parents=True, exist_ok=True)
        total_files = 0
        for item in source.rglob("*"):
            if item.is_file() and ".git" not in item.parts:
                rel = item.relative_to(source)
                dest = target / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(item), str(dest))
                total_files += 1

        return total_files

    finally:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)


async def _download_via_api(
    owner: str, repo: str, branch: str, subpath: str, target: Path
) -> int:
    """Download skill files via GitHub Contents API. Returns file count."""
    import httpx

    headers = _github_headers()
    base_api = f"https://api.github.com/repos/{owner}/{repo}/contents"
    raw_base = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}"
    total_files = 0

    async def download_dir(
        api_path: str, dest_dir: Path, client: httpx.AsyncClient
    ) -> None:
        nonlocal total_files
        url = f"{base_api}/{api_path}?ref={branch}" if api_path else f"{base_api}?ref={branch}"
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        items: list[dict[str, Any]] = resp.json()
        for item in items:
            itype = item.get("type")
            ipath = item.get("path", "")
            iname = item.get("name", "")
            if itype == "file":
                dest = dest_dir / iname
                dest.parent.mkdir(parents=True, exist_ok=True)
                file_resp = await client.get(f"{raw_base}/{ipath}")
                file_resp.raise_for_status()
                dest.write_bytes(file_resp.content)
                total_files += 1
            elif itype == "dir":
                sub_dest = dest_dir / iname
                sub_dest.mkdir(parents=True, exist_ok=True)
                await download_dir(ipath, sub_dest, client)

    target.mkdir(parents=True, exist_ok=True)
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            await download_dir(subpath, target, client)
    except Exception:
        shutil.rmtree(str(target), ignore_errors=True)
        raise

    return total_files


async def _download_skill_files(
    owner: str, repo: str, branch: str, subpath: str, target: Path
) -> int:
    """Download skill files. Tries git sparse-checkout first, falls back to API."""

    # Try git clone first (avoids API rate limits)
    try:
        count = await _download_via_git(owner, repo, branch, subpath, target)
        if count > 0 and (target / "SKILL.md").exists():
            logger.info(f"Downloaded {count} files via git sparse-checkout")
            return count
        # Git succeeded but no files or no SKILL.md — clean up and try API
        if target.exists():
            shutil.rmtree(str(target), ignore_errors=True)
    except Exception as e:
        logger.debug(f"git download failed ({e}), falling back to API")
        if target.exists():
            shutil.rmtree(str(target), ignore_errors=True)

    # Fallback: GitHub Contents API
    count = await _download_via_api(owner, repo, branch, subpath, target)

    if count == 0:
        shutil.rmtree(str(target), ignore_errors=True)
        raise RuntimeError(
            f"No files found under '{subpath}' in {owner}/{repo}@{branch}"
        )

    if not (target / "SKILL.md").exists():
        shutil.rmtree(str(target), ignore_errors=True)
        raise RuntimeError("No SKILL.md found — not a valid skill directory")

    return count


def _retarget_skill_dir(
    workspace: Path,
    current_target: Path,
    fallback_name: str,
    *,
    overwrite_existing: bool = False,
) -> tuple[str, Path, bool]:
    """Move a downloaded skill to its frontmatter-declared directory name."""
    skill_name = _read_skill_name_from_frontmatter(
        current_target / "SKILL.md",
        fallback_name,
    )
    final_target = workspace / "skills" / skill_name
    if final_target == current_target:
        return skill_name, current_target, False

    if final_target.exists():
        if not overwrite_existing and (final_target / "SKILL.md").exists():
            shutil.rmtree(str(current_target), ignore_errors=True)
            return skill_name, final_target, True
        shutil.rmtree(str(final_target), ignore_errors=True)

    final_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(current_target), str(final_target))
    return skill_name, final_target, False


class SkillMarketplaceTool(BaseTool):
    """Search, install, and remove skills from skills.sh / GitHub / local paths."""

    name: str = "skill_marketplace"
    description: str = (
        "Install, update, remove, or search skills. "
        "Use install actions only when the user explicitly asks to install an Agent Skill "
        "or after you have already confirmed the source contains a skill SKILL.md. "
        "Do NOT call this tool for arbitrary GitHub repositories, GitHub files, or local files; "
        "inspect/review those first with the normal file, shell, or web tools. "
        "Actions: "
        "install_skill (url required) — install a confirmed Agent Skill from GitHub URL; "
        "install_local (url required) — copy a confirmed local skill directory containing SKILL.md; "
        "update_skill (url required) — re-download and update an already-installed skill; "
        "search_skills (query required) — search skills.sh; "
        "remove_skill (skill_name required) — remove installed skill; "
        "skill_info (skill_name required) — show SKILL.md. "
        "For GitHub repos, resolve the folder containing SKILL.md and install only that folder into workspace/skills. "
        "Do NOT create shortcuts/symlinks or manual clones for GitHub URLs. "
        "After install/update/remove, call self_upgrade(action='reload_skills') to activate."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "search_skills",
                    "install_skill",
                    "install_local",
                    "update_skill",
                    "remove_skill",
                    "skill_info",
                ],
                "description": "Action to perform",
            },
            "query": {
                "type": "string",
                "description": "Search query for search_skills",
            },
            "url": {
                "type": "string",
                "description": (
                    "Confirmed skill GitHub URL to install from, or a local skill directory path "
                    "for install_local. Do not pass generic repos/files before inspection. "
                    "e.g. 'https://github.com/openclaw/skills/tree/main/skills/tezatezaz/clawcast-wallet' "
                    "or '/home/user/my-skills/my-skill' or 'C:\\Projects\\my-skill'."
                ),
            },
            "skill_name": {
                "type": "string",
                "description": "Skill name for remove_skill or skill_info",
            },
        },
        "required": ["action"],
    }

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "search_skills")
        query = kwargs.get("query", "").strip()
        url = kwargs.get("url", "").strip()
        skill_name = kwargs.get("skill_name", "").strip()
        workspace = _get_workspace()

        # ----------------------------------------------------------
        # search_skills
        # ----------------------------------------------------------
        if action == "search_skills":
            if not query:
                return "Error: 'query' is required for search_skills"
            try:
                import httpx
            except ImportError:
                return "Error: httpx is required"
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.get(_SKILLS_SH_API, params={"q": query})
                    resp.raise_for_status()
                    data = resp.json()
            except Exception as e:
                return f"Error searching skills.sh: {e}"

            skills = data.get("skills", [])
            if not skills:
                return f"No skills found for '{query}' on skills.sh"

            lines = [
                f"Found {len(skills)} skill(s) matching '{query}' on skills.sh:\n"
            ]
            for s in skills:
                source = s.get("source", "")
                sid = s.get("skillId", s.get("name", ""))
                name = s.get("name", sid)
                installs = s.get("installs", 0)
                shorthand = f"{source}/{sid}" if source else sid
                lines.append(
                    f"  - {name}  [{installs} installs]"
                    f"\n    Install: skill_marketplace install_skill url='{shorthand}'"
                )
            return "\n".join(lines)

        # ----------------------------------------------------------
        # install_skill
        # ----------------------------------------------------------
        elif action == "install_skill":
            if not url:
                return "Error: 'url' is required for install_skill"
            try:
                _url = url
                if not _url.startswith(("http://", "https://")):
                    if "github.com" in _url:
                        _url = "https://" + _url.lstrip("/")
                    elif "/tree/" in _url:
                        _url = "https://github.com/" + _url.lstrip("/")

                if _url.startswith(("http://", "https://")):
                    owner, repo, branch, subpath = _parse_github_url(_url)
                    subpath = _validate_subpath(subpath, allow_empty=True)
                    branch, subpath, skill_name_derived = await _resolve_github_skill_source(
                        owner, repo, branch, subpath
                    )
                else:
                    parts = [p for p in url.split("/") if p]
                    if len(parts) < 2:
                        return (
                            "Error: provide a full GitHub URL or "
                            "'owner/repo[/<skill_path>]' shorthand"
                        )
                    owner, repo = parts[0], parts[1]
                    branch = "main"
                    rest = parts[2:]
                    if rest and rest[0] == "tree":
                        if len(rest) < 2:
                            return (
                                "Error: provide a full GitHub tree URL, e.g. "
                                "owner/repo/tree/main[/skills/foo]"
                            )
                        branch = rest[1] if len(rest) > 1 else "main"
                        rest = rest[2:]
                    subpath = _validate_subpath("/".join(rest), allow_empty=True)
                    branch, subpath, skill_name_derived = await _resolve_github_skill_source(
                        owner, repo, branch, subpath
                    )

                target = workspace / "skills" / skill_name_derived
                _rel_path = f"skills/{skill_name_derived}"

                if target.exists() and (target / "SKILL.md").exists():
                    skill_name_derived, target, _already_installed = _retarget_skill_dir(
                        workspace,
                        target,
                        skill_name_derived,
                    )
                    _rel_path = f"skills/{skill_name_derived}"
                    return (
                        f"Skill '{skill_name_derived}' is already installed.\n"
                        "If the user asked to UPDATE, use action='update_skill' instead.\n"
                        "Otherwise, read the skill instructions before proceeding:\n"
                        f"  read_file(path='{_rel_path}/SKILL.md')\n"
                        "Then follow the commands and persistence rules described in SKILL.md."
                    )
                elif target.exists():
                    shutil.rmtree(str(target), ignore_errors=True)

                target.parent.mkdir(parents=True, exist_ok=True)
                count = await _download_skill_files(
                    owner, repo, branch, subpath, target
                )
                skill_name_derived, target, already_installed = _retarget_skill_dir(
                    workspace,
                    target,
                    skill_name_derived,
                )
                _rel_path = f"skills/{skill_name_derived}"

                if already_installed:
                    return (
                        f"Skill '{skill_name_derived}' is already installed.\n"
                        "If the user asked to UPDATE, use action='update_skill' instead.\n"
                        "Otherwise, read the skill instructions before proceeding:\n"
                        f"  read_file(path='{_rel_path}/SKILL.md')\n"
                        "Then follow the commands and persistence rules described in SKILL.md."
                    )

                installed_files = []
                for f in sorted(target.rglob("*")):
                    if f.is_file():
                        installed_files.append(str(f.relative_to(target)))

                return (
                    f"SUCCESS: Skill '{skill_name_derived}' installed ({count} files).\n"
                    f"Source: {owner}/{repo}@{branch}"
                    + (f" / {subpath}" if subpath else " / <repo-root>")
                    + "\n"
                    f"Files: {', '.join(installed_files[:15])}\n\n"
                    "NEXT STEPS:\n"
                    "1) Call `self_upgrade(action='reload_skills')` to activate.\n"
                    f"2) read_file(path='{_rel_path}/SKILL.md') for instructions.\n"
                    "3) Follow the commands and persistence rules described in SKILL.md."
                )

            except Exception as e:
                logger.error(f"install_skill failed: {e}")
                try:
                    if target.exists():
                        has_skill = (target / "SKILL.md").exists()
                        if not has_skill:
                            shutil.rmtree(str(target), ignore_errors=True)
                except NameError:
                    pass
                message = str(e)
                if "No SKILL.md" in message or "not a valid skill" in message:
                    return (
                        "Not installed: this source was not confirmed as an Agent Skill "
                        f"({message}). For a regular repository or file, inspect/review it first "
                        "with normal tools instead of copying it into workspace/skills."
                    )
                return f"Error installing skill: {e}"

        # ----------------------------------------------------------
        # install_local — copy/link a skill from a local directory
        # ----------------------------------------------------------
        elif action == "install_local":
            if not url:
                return "Error: 'url' (local path) is required for install_local"
            if _looks_like_github_repo_reference(url):
                return (
                    "Error: GitHub repos can use action='install_skill' only after the source "
                    "is confirmed to be an Agent Skill with SKILL.md. Inspect generic repos/files first."
                )
            try:
                source = Path(url.strip()).expanduser().resolve()
                if not source.is_dir():
                    return f"Error: '{url}' is not a valid directory"
                if not (source / "SKILL.md").exists():
                    return f"Error: '{url}' does not contain SKILL.md — not a valid skill"

                skill_name_derived = _read_skill_name_from_frontmatter(
                    source / "SKILL.md",
                    source.name,
                )
                target = workspace / "skills" / skill_name_derived
                _rel_path = f"skills/{skill_name_derived}"

                if target.exists() and (target / "SKILL.md").exists():
                    return (
                        f"Skill '{skill_name_derived}' is already installed at {_rel_path}.\n"
                        "Use action='update_skill' or remove it first.\n"
                        f"To use: read_file(path='{_rel_path}/SKILL.md')"
                    )
                elif target.exists():
                    shutil.rmtree(str(target), ignore_errors=True)

                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(
                    str(source), str(target),
                    ignore=shutil.ignore_patterns(
                        ".git", "node_modules", ".pnpm-store",
                        "__pycache__", ".venv", "*.pyc",
                    ),
                )

                installed_files = []
                for f in sorted(target.rglob("*")):
                    if f.is_file():
                        installed_files.append(str(f.relative_to(target)))

                return (
                    f"SUCCESS: Skill '{skill_name_derived}' installed from local path "
                    f"({len(installed_files)} files).\n"
                    f"Source: {source}\n"
                    f"Files: {', '.join(installed_files[:15])}\n\n"
                    "NEXT STEPS:\n"
                    "1) Call `self_upgrade(action='reload_skills')` to activate.\n"
                    f"2) read_file(path='{_rel_path}/SKILL.md') for instructions.\n"
                    "3) Execute CLI commands from SKILL.md."
                )
            except Exception as e:
                logger.error(f"install_local failed: {e}")
                return f"Error installing local skill: {e}"

        # ----------------------------------------------------------
        # update_skill — re-download an already-installed skill
        # ----------------------------------------------------------
        elif action == "update_skill":
            if not url:
                return "Error: 'url' is required for update_skill"
            try:
                _url = url
                if not _url.startswith(("http://", "https://")):
                    if "github.com" in _url:
                        _url = "https://" + _url.lstrip("/")
                    elif "/tree/" in _url:
                        _url = "https://github.com/" + _url.lstrip("/")

                if _url.startswith(("http://", "https://")):
                    owner, repo, branch, subpath = _parse_github_url(_url)
                    subpath = _validate_subpath(subpath, allow_empty=True)
                    branch, subpath, skill_name_derived = await _resolve_github_skill_source(
                        owner, repo, branch, subpath
                    )
                else:
                    parts = [p for p in url.split("/") if p]
                    if len(parts) < 2:
                        return (
                            "Error: provide a full GitHub URL or "
                            "'owner/repo[/<skill_path>]' shorthand"
                        )
                    owner, repo = parts[0], parts[1]
                    branch = "main"
                    rest = parts[2:]
                    if rest and rest[0] == "tree":
                        if len(rest) < 2:
                            return (
                                "Error: provide a full GitHub tree URL, e.g. "
                                "owner/repo/tree/main[/skills/foo]"
                            )
                        branch = rest[1] if len(rest) > 1 else "main"
                        rest = rest[2:]
                    subpath = _validate_subpath("/".join(rest), allow_empty=True)
                    branch, subpath, skill_name_derived = await _resolve_github_skill_source(
                        owner, repo, branch, subpath
                    )

                target = workspace / "skills" / skill_name_derived
                _rel_path = f"skills/{skill_name_derived}"

                if target.exists():
                    shutil.rmtree(str(target), ignore_errors=True)

                target.parent.mkdir(parents=True, exist_ok=True)
                count = await _download_skill_files(
                    owner, repo, branch, subpath, target
                )
                skill_name_derived, target, _already_installed = _retarget_skill_dir(
                    workspace,
                    target,
                    skill_name_derived,
                    overwrite_existing=True,
                )
                _rel_path = f"skills/{skill_name_derived}"

                return (
                    f"SUCCESS: Skill '{skill_name_derived}' updated ({count} files).\n\n"
                    f"Source: {owner}/{repo}@{branch}"
                    + (f" / {subpath}" if subpath else " / <repo-root>")
                    + "\n"
                    "NEXT STEPS:\n"
                    "1) Call `self_upgrade(action='reload_skills')` to activate.\n"
                    f"2) read_file(path='{_rel_path}/SKILL.md') for updated instructions.\n"
                    "3) Record the update in soul.md."
                )
            except Exception as e:
                logger.error(f"update_skill failed: {e}")
                return f"Error updating skill: {e}"

        # ----------------------------------------------------------
        # remove_skill
        # ----------------------------------------------------------
        elif action == "remove_skill":
            if not skill_name:
                return "Error: 'skill_name' is required for remove_skill"
            target = workspace / "skills" / skill_name
            if not target.exists():
                return f"Skill '{skill_name}' not found in {workspace / 'skills'}"
            shutil.rmtree(str(target), ignore_errors=True)
            return (
                f"Skill '{skill_name}' removed.\n"
                "Call `self_upgrade reload_skills` to apply changes."
            )

        # ----------------------------------------------------------
        # skill_info
        # ----------------------------------------------------------
        elif action == "skill_info":
            if not skill_name:
                return "Error: 'skill_name' is required for skill_info"
            skill_dir = workspace / "skills" / skill_name
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                return f"Skill '{skill_name}' not found"
            files = [str(f.relative_to(skill_dir)) for f in sorted(skill_dir.rglob("*")) if f.is_file()]
            _rel_path = f"skills/{skill_name}"

            return (
                f"Skill: {skill_name}\n"
                f"SKILL.md: {_rel_path}/SKILL.md\n"
                f"Files: {', '.join(files[:15])}\n\n"
                f"To use: read_file(path='{_rel_path}/SKILL.md')"
            )

        return f"Unknown action: {action!r}"
