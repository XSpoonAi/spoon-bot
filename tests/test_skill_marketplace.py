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


@pytest.mark.asyncio
async def test_install_skill_writes_into_workspace_skills_directory(
    skill_manager_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setenv("SPOON_BOT_WORKSPACE_PATH", str(tmp_path))
    recorded_targets: list[Path] = []

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


def test_git_clone_command_uses_full_clone_for_root_skill_repo(skill_manager_module):
    cmd = skill_manager_module._build_git_clone_command(
        "https://github.com/Agent-Cypher-Lab/joker-game-agent.git",
        "main",
        Path("C:/tmp/repo"),
        "",
    )

    assert "--sparse" not in cmd
    assert cmd[-2:] == [
        "https://github.com/Agent-Cypher-Lab/joker-game-agent.git",
        "C:\\tmp\\repo",
    ]


def test_git_clone_command_uses_sparse_clone_for_nested_skill(skill_manager_module):
    cmd = skill_manager_module._build_git_clone_command(
        "https://github.com/openclaw/skills.git",
        "main",
        Path("C:/tmp/repo"),
        "skills/demo/wallet",
    )

    assert "--sparse" in cmd
