"""Skill marketplace tool — search, install, remove skills from GitHub / skills.sh."""

from __future__ import annotations

import logging
import os
import re
import shutil
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


async def _find_skill_subpath(
    owner: str, repo: str, skill_id: str, branch: str
) -> str:
    """Locate a skill directory inside a repo via GitHub Contents API."""
    import httpx

    headers = {"Accept": "application/vnd.github+json"}
    base_api = f"https://api.github.com/repos/{owner}/{repo}/contents"

    async def search_dir(
        api_path: str, client: httpx.AsyncClient, depth: int = 0
    ) -> str | None:
        if depth > 6:
            return None
        url = (
            f"{base_api}/{api_path}?ref={branch}"
            if api_path
            else f"{base_api}?ref={branch}"
        )
        resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            return None
        items: list[dict[str, Any]] = resp.json()
        for item in items:
            if item.get("type") == "dir":
                if item.get("name", "").lower() == skill_id.lower():
                    chk = await client.get(
                        f"{base_api}/{item['path']}/SKILL.md?ref={branch}",
                        headers=headers,
                    )
                    if chk.status_code == 200:
                        return item["path"]
                found = await search_dir(item["path"], client, depth + 1)
                if found is not None:
                    return found
        return None

    async with httpx.AsyncClient(timeout=30) as client:
        result = await search_dir("", client)
    if result is None:
        raise ValueError(
            f"Skill '{skill_id}' not found in {owner}/{repo}@{branch}. "
            "Check the skillId or provide the full GitHub URL."
        )
    return result


async def _download_skill_files(
    owner: str, repo: str, branch: str, subpath: str, target: Path
) -> int:
    """Download skill files via GitHub Contents API. Returns file count."""
    import httpx

    headers = {"Accept": "application/vnd.github+json"}
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
    async with httpx.AsyncClient(timeout=30) as client:
        await download_dir(subpath, target, client)

    if total_files == 0:
        shutil.rmtree(str(target), ignore_errors=True)
        raise RuntimeError(
            f"No files found under '{subpath}' in {owner}/{repo}@{branch}"
        )

    if not (target / "SKILL.md").exists():
        shutil.rmtree(str(target), ignore_errors=True)
        raise RuntimeError("No SKILL.md found — not a valid skill directory")

    return total_files


class SkillMarketplaceTool(BaseTool):
    """Search, install, and remove skills from skills.sh / GitHub."""

    name: str = "skill_marketplace"
    description: str = (
        "Skill marketplace — search, install, and remove skills.\n"
        "Actions:\n"
        "- search_skills <query>: Search skills.sh for available skills\n"
        "- install_skill <url>: Install from GitHub URL or 'owner/repo/skillId'\n"
        "- remove_skill <skill_name>: Remove an installed skill\n"
        "- skill_info <skill_name>: Show SKILL.md of an installed skill\n\n"
        "After install/remove, call self_upgrade reload_skills to activate changes."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "search_skills",
                    "install_skill",
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
                    "For install_skill: full GitHub URL "
                    "(https://github.com/owner/repo/tree/branch/path) "
                    "or shorthand 'owner/repo/skillId'"
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
                if url.startswith("http://") or url.startswith("https://"):
                    owner, repo, branch, subpath = _parse_github_url(url)
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
                    skill_id = parts[2] if len(parts) > 2 else ""
                    if skill_id:
                        subpath = await _find_skill_subpath(
                            owner, repo, skill_id, branch
                        )
                        skill_name_derived = skill_id
                    else:
                        subpath = ""
                        skill_name_derived = repo

                target = workspace / "skills" / skill_name_derived
                if target.exists():
                    return (
                        f"Skill '{skill_name_derived}' is already installed. "
                        "Use remove_skill first if you want to reinstall."
                    )

                target.parent.mkdir(parents=True, exist_ok=True)
                count = await _download_skill_files(
                    owner, repo, branch, subpath, target
                )

                # Read installed SKILL.md so the agent knows how to use it
                skill_md_content = ""
                skill_md_path = target / "SKILL.md"
                if skill_md_path.exists():
                    skill_md_content = skill_md_path.read_text(encoding="utf-8")

                result_parts = [
                    f"Skill '{skill_name_derived}' installed successfully "
                    f"from {owner}/{repo} ({count} files).",
                    "",
                    "IMPORTANT: To activate the new skill, first activate "
                    "the self_upgrade tool using "
                    "`activate_tool(action='activate', tool_name='self_upgrade')`, "
                    "then call `self_upgrade(action='reload_skills')`.",
                ]
                if skill_md_content:
                    result_parts.append("")
                    result_parts.append("--- Installed SKILL.md ---")
                    result_parts.append(skill_md_content)

                return "\n".join(result_parts)

            except Exception as e:
                logger.error(f"install_skill failed: {e}")
                return f"Error installing skill: {e}"

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
            skill_md = workspace / "skills" / skill_name / "SKILL.md"
            if not skill_md.exists():
                return f"Skill '{skill_name}' not found"
            content = skill_md.read_text(encoding="utf-8")
            return f"Skill: {skill_name}\n\n{content}"

        return f"Unknown action: {action!r}"
