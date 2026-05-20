from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def skill_manager_module(monkeypatch: pytest.MonkeyPatch):
    class BaseTool:
        pass

    spoon_ai_pkg = types.ModuleType("spoon_ai")
    tools_pkg = types.ModuleType("spoon_ai.tools")
    base_pkg = types.ModuleType("spoon_ai.tools.base")
    base_pkg.BaseTool = BaseTool
    spoon_ai_pkg.tools = tools_pkg
    tools_pkg.base = base_pkg

    monkeypatch.setitem(sys.modules, "spoon_ai", spoon_ai_pkg)
    monkeypatch.setitem(sys.modules, "spoon_ai.tools", tools_pkg)
    monkeypatch.setitem(sys.modules, "spoon_ai.tools.base", base_pkg)

    repo_root = Path(__file__).resolve().parent.parent
    tools_path = repo_root / "workspace" / "skills" / "skill-manager" / "tools.py"
    spec = importlib.util.spec_from_file_location(
        "skill_manager_tools_under_test",
        tools_path,
    )
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_skill_marketplace_description_does_not_route_generic_github_urls(
    skill_manager_module,
):
    description = skill_manager_module.SkillMarketplaceTool.description

    assert "WHEN THE USER GIVES A GITHUB URL" not in description
    assert "arbitrary GitHub repositories" in description
    assert "inspect/review those first" in description


def test_builtin_skill_manager_requires_confirmed_skill_sources():
    repo_root = Path(__file__).resolve().parent.parent
    skill_md = repo_root / "workspace" / "skills" / "skill-manager" / "SKILL.md"
    content = skill_md.read_text(encoding="utf-8")
    keywords_line = next(line for line in content.splitlines() if "keywords:" in line)

    assert "Do not use this skill for arbitrary GitHub repositories" in content
    assert "github.com" not in keywords_line
    assert "confirmed Agent Skill sources" in content


@pytest.mark.asyncio
async def test_install_skill_writes_into_workspace_skills_directory(
    skill_manager_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setenv("SPOON_BOT_WORKSPACE_PATH", str(tmp_path))
    recorded_targets: list[Path] = []

    async def fake_resolve(owner, repo, branch, subpath):
        return branch, subpath, "wallet"

    monkeypatch.setattr(skill_manager_module, "_resolve_github_skill_source", fake_resolve)

    async def fake_download(owner, repo, branch, subpath, target):
        recorded_targets.append(target)
        target.mkdir(parents=True, exist_ok=True)
        (target / "SKILL.md").write_text("# skill", encoding="utf-8")
        return 1

    monkeypatch.setattr(skill_manager_module, "_download_skill_files", fake_download)

    tool = skill_manager_module.SkillMarketplaceTool()
    result = await tool.execute(
        action="install_skill",
        url="openclaw/skills/skills/demo/wallet",
    )

    installed_dir = tmp_path / "skills" / "wallet"
    assert "SUCCESS" in result
    assert recorded_targets == [installed_dir]
    assert installed_dir.exists()
    assert (installed_dir / "SKILL.md").exists()
    assert not (tmp_path / "wallet").exists()


@pytest.mark.asyncio
async def test_install_root_skill_repo_writes_into_workspace_skills_directory(
    skill_manager_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setenv("SPOON_BOT_WORKSPACE_PATH", str(tmp_path))
    recorded_calls: list[tuple[str, str, str, str, Path]] = []

    async def fake_resolve(owner, repo, branch, subpath):
        return branch, "", "joker-game-agent"

    monkeypatch.setattr(skill_manager_module, "_resolve_github_skill_source", fake_resolve)

    async def fake_download(owner, repo, branch, subpath, target):
        recorded_calls.append((owner, repo, branch, subpath, target))
        target.mkdir(parents=True, exist_ok=True)
        (target / "SKILL.md").write_text("# joker", encoding="utf-8")
        (target / "README.md").write_text("# readme", encoding="utf-8")
        return 2

    monkeypatch.setattr(skill_manager_module, "_download_skill_files", fake_download)

    tool = skill_manager_module.SkillMarketplaceTool()
    result = await tool.execute(
        action="install_skill",
        url="https://github.com/Agent-Cypher-Lab/joker-game-agent",
    )

    installed_dir = tmp_path / "skills" / "joker-game-agent"
    assert "SUCCESS" in result
    assert recorded_calls == [
        ("Agent-Cypher-Lab", "joker-game-agent", "main", "", installed_dir)
    ]
    assert installed_dir.exists()
    assert (installed_dir / "SKILL.md").exists()
    assert not (tmp_path / "SKILL.md").exists()
    assert not (tmp_path / "README.md").exists()


@pytest.mark.asyncio
async def test_install_root_skill_repo_uses_skill_frontmatter_name(
    skill_manager_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setenv("SPOON_BOT_WORKSPACE_PATH", str(tmp_path))
    recorded_targets: list[Path] = []

    async def fake_resolve(owner, repo, branch, subpath):
        return branch, "", "agent-spot-cypher"

    monkeypatch.setattr(skill_manager_module, "_resolve_github_skill_source", fake_resolve)

    async def fake_download(owner, repo, branch, subpath, target):
        recorded_targets.append(target)
        target.mkdir(parents=True, exist_ok=True)
        (target / "SKILL.md").write_text(
            "---\n"
            "name: spot-agent-cypher\n"
            "description: Use when playing SPOT games\n"
            "---\n"
            "# Spot Agent Cypher\n",
            encoding="utf-8",
        )
        return 1

    monkeypatch.setattr(skill_manager_module, "_download_skill_files", fake_download)

    tool = skill_manager_module.SkillMarketplaceTool()
    result = await tool.execute(
        action="install_skill",
        url="https://github.com/Agent-Cypher-Lab/agent-spot-cypher",
    )

    installed_dir = tmp_path / "skills" / "spot-agent-cypher"
    fallback_dir = tmp_path / "skills" / "agent-spot-cypher"
    assert "SUCCESS: Skill 'spot-agent-cypher' installed" in result
    assert recorded_targets == [fallback_dir]
    assert installed_dir.exists()
    assert (installed_dir / "SKILL.md").exists()
    assert not fallback_dir.exists()


@pytest.mark.asyncio
async def test_install_skill_missing_skill_md_does_not_install_generic_repo(
    skill_manager_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setenv("SPOON_BOT_WORKSPACE_PATH", str(tmp_path))

    async def fake_resolve(owner, repo, branch, subpath):
        raise RuntimeError("No SKILL.md found under repository root")

    monkeypatch.setattr(skill_manager_module, "_resolve_github_skill_source", fake_resolve)

    tool = skill_manager_module.SkillMarketplaceTool()
    result = await tool.execute(
        action="install_skill",
        url="https://github.com/XSpoonAi/spoon-core",
    )

    assert result.startswith("Not installed:")
    assert "inspect/review it first" in result
    assert "workspace/skills" in result
    assert not (tmp_path / "skills" / "spoon-core").exists()


def test_git_clone_command_uses_full_clone_for_root_skill_repo(skill_manager_module):
    destination = Path("C:/tmp/repo")
    cmd = skill_manager_module._build_git_clone_command(
        "https://github.com/Agent-Cypher-Lab/joker-game-agent.git",
        "main",
        destination,
        "",
    )

    assert "--sparse" not in cmd
    assert cmd[-2:] == [
        "https://github.com/Agent-Cypher-Lab/joker-game-agent.git",
        str(destination),
    ]


def test_git_clone_command_uses_sparse_clone_for_nested_skill(skill_manager_module):
    cmd = skill_manager_module._build_git_clone_command(
        "https://github.com/openclaw/skills.git",
        "main",
        Path("C:/tmp/repo"),
        "skills/demo/wallet",
    )

    assert "--sparse" in cmd


def test_select_skill_root_subpath_prefers_root(skill_manager_module):
    resolved = skill_manager_module._select_skill_root_subpath(
        ["SKILL.md", "examples/demo/SKILL.md"],
    )

    assert resolved == ""


def test_select_skill_root_subpath_picks_unique_nested_skill(skill_manager_module):
    resolved = skill_manager_module._select_skill_root_subpath(
        ["skills/demo/joker-game-agent/SKILL.md"],
    )

    assert resolved == "skills/demo/joker-game-agent"


def test_select_skill_root_subpath_scopes_to_requested_parent(skill_manager_module):
    resolved = skill_manager_module._select_skill_root_subpath(
        [
            "skills/demo/joker-game-agent/SKILL.md",
            "skills/other/unrelated/SKILL.md",
        ],
        requested_subpath="skills/demo",
    )

    assert resolved == "skills/demo/joker-game-agent"
