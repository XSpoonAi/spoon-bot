from __future__ import annotations

from pathlib import Path

from spoon_bot.skills.builtin import builtin_skills_root, ensure_builtin_skills


def test_ensure_builtin_skills_installs_wallet(tmp_path):
    workspace = tmp_path / "workspace"

    installed = ensure_builtin_skills(workspace)

    installed_names = {path.name for path in installed}
    assert "wallet" in installed_names
    assert (workspace / "skills" / "wallet" / "SKILL.md").exists()
    assert not (workspace / "skills" / "wallet" / "scripts").exists()
    assert not (workspace / "skills" / "wallet" / "assets").exists()


def test_ensure_builtin_skills_does_not_overwrite_existing_skill(tmp_path):
    workspace = tmp_path / "workspace"
    target = workspace / "skills" / "wallet"
    target.mkdir(parents=True, exist_ok=True)
    skill_file = target / "SKILL.md"
    skill_file.write_text("custom skill", encoding="utf-8")

    installed = ensure_builtin_skills(workspace)

    assert "wallet" not in {path.name for path in installed}
    assert skill_file.read_text(encoding="utf-8") == "custom skill"


def test_builtin_skills_root_contains_wallet():
    root = builtin_skills_root()
    assert root.is_dir()
    assert (root / "wallet" / "SKILL.md").exists()
    assert not (root / "wallet" / "scripts").exists()
    assert not (root / "wallet" / "assets").exists()
