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


def _get_workspace() -> Path:
    """Resolve the workspace directory."""
    ws = os.environ.get("SPOON_BOT_WORKSPACE_PATH")
    if ws:
        return Path(ws)
    return Path.home() / ".spoon-bot" / "workspace"


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


def _github_headers() -> dict[str, str]:
    """Build GitHub API headers, including auth token if available."""
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def _download_via_git(
    owner: str, repo: str, branch: str, subpath: str, target: Path
) -> int:
    """Download skill files using git sparse-checkout. Returns file count.

    This avoids GitHub API rate limits by using git protocol directly.
    """
    clone_url = f"https://github.com/{owner}/{repo}.git"
    tmp_dir = Path(tempfile.mkdtemp(prefix="spoon_skill_"))

    try:
        # git clone with sparse checkout — only download the subpath we need
        proc = await asyncio.create_subprocess_exec(
            "git", "clone", "--depth=1", "--filter=blob:none",
            "--sparse", "--branch", branch,
            clone_url, str(tmp_dir / "repo"),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
        if proc.returncode != 0:
            raise RuntimeError(f"git clone failed: {stderr.decode().strip()}")

        repo_dir = tmp_dir / "repo"

        # Set sparse-checkout to only include the skill subpath
        if subpath:
            proc = await asyncio.create_subprocess_exec(
                "git", "sparse-checkout", "set", subpath,
                cwd=str(repo_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=30)

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
        url = f"{base_api}/{api_path}?ref={branch}"
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


class SkillMarketplaceTool(BaseTool):
    """Search, install, and remove skills from skills.sh / GitHub."""

    name: str = "skill_marketplace"
    description: str = (
        "Install, update, remove, or search skills. "
        "WHEN THE USER GIVES A GITHUB URL, call this tool with action='install_skill' and url=<the URL>. "
        "Actions: "
        "install_skill (url required) — install a skill from GitHub URL; "
        "update_skill (url required) — re-download and update an already-installed skill; "
        "search_skills (query required) — search skills.sh; "
        "remove_skill (skill_name required) — remove installed skill; "
        "skill_info (skill_name required) — show SKILL.md. "
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
                    "GitHub URL to install from. Pass the EXACT URL the user gave you, "
                    "e.g. 'https://github.com/openclaw/skills/tree/main/skills/tezatezaz/clawcast-wallet'. "
                    "Also accepts 'owner/repo/skillId' shorthand."
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
                    skill_name_derived = (
                        subpath.rsplit("/", 1)[-1] if subpath else repo
                    )
                else:
                    parts = [p for p in url.split("/") if p]
                    if len(parts) < 2:
                        return (
                            "Error: provide a full GitHub URL or "
                            "'owner/repo/skillId' shorthand"
                        )
                    owner, repo = parts[0], parts[1]
                    branch = "main"
                    rest = parts[2:]
                    if rest and rest[0] == "tree":
                        branch = rest[1] if len(rest) > 1 else "main"
                        rest = rest[2:]
                    skill_id = rest[-1] if rest else ""
                    subpath = "/".join(rest) if rest else ""
                    skill_name_derived = skill_id or repo

                target = workspace / "skills" / skill_name_derived
                _rel_path = f"skills/{skill_name_derived}"

                if target.exists() and (target / "SKILL.md").exists():
                    return (
                        f"Skill '{skill_name_derived}' is already installed.\n"
                        "If the user asked to UPDATE, use action='update_skill' instead.\n"
                        "Otherwise, proceed to USE the skill:\n"
                        f"  read_file(path='{_rel_path}/SKILL.md')\n"
                        "Then execute CLI commands from SKILL.md. Record in soul.md."
                    )
                elif target.exists():
                    shutil.rmtree(str(target), ignore_errors=True)

                target.parent.mkdir(parents=True, exist_ok=True)
                count = await _download_skill_files(
                    owner, repo, branch, subpath, target
                )

                installed_files = []
                for f in sorted(target.rglob("*")):
                    if f.is_file():
                        installed_files.append(str(f.relative_to(target)))

                return (
                    f"SUCCESS: Skill '{skill_name_derived}' installed ({count} files).\n"
                    f"Files: {', '.join(installed_files[:15])}\n\n"
                    "NEXT STEPS:\n"
                    "1) Call `self_upgrade(action='reload_skills')` to activate.\n"
                    f"2) read_file(path='{_rel_path}/SKILL.md') for instructions.\n"
                    "3) Execute CLI commands from SKILL.md (cast, curl — NOT scripts).\n"
                    "4) Record the result in soul.md."
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
                return f"Error installing skill: {e}"

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
                    skill_name_derived = (
                        subpath.rsplit("/", 1)[-1] if subpath else repo
                    )
                else:
                    parts = [p for p in url.split("/") if p]
                    if len(parts) < 2:
                        return "Error: provide a full GitHub URL or 'owner/repo/skillId' shorthand"
                    owner, repo = parts[0], parts[1]
                    branch = "main"
                    rest = parts[2:]
                    if rest and rest[0] == "tree":
                        branch = rest[1] if len(rest) > 1 else "main"
                        rest = rest[2:]
                    skill_id = rest[-1] if rest else ""
                    subpath = "/".join(rest) if rest else ""
                    skill_name_derived = skill_id or repo

                target = workspace / "skills" / skill_name_derived
                _rel_path = f"skills/{skill_name_derived}"

                if target.exists():
                    shutil.rmtree(str(target), ignore_errors=True)

                target.parent.mkdir(parents=True, exist_ok=True)
                count = await _download_skill_files(
                    owner, repo, branch, subpath, target
                )

                return (
                    f"SUCCESS: Skill '{skill_name_derived}' updated ({count} files).\n\n"
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
