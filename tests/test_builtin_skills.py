from __future__ import annotations

from spoon_bot.skills.builtin import builtin_skills_root, ensure_builtin_skills


def test_ensure_builtin_skills_installs_builtin_skill_set(tmp_path):
    workspace = tmp_path / "workspace"

    installed = ensure_builtin_skills(workspace)

    installed_names = {path.name for path in installed}
    assert {"wallet", "subagents", "service_expose", "skill-manager"}.issubset(
        installed_names
    )
    assert (workspace / "skills" / "wallet" / "SKILL.md").exists()
    assert (workspace / "skills" / "subagents" / "SKILL.md").exists()
    assert (workspace / "skills" / "service_expose" / "SKILL.md").exists()
    assert (workspace / "skills" / "skill-manager" / "SKILL.md").exists()
    assert (workspace / "skills" / "skill-manager" / "tools.py").exists()
    assert (workspace / "skills" / "service_expose" / "scripts" / "service_expose.py").exists()
    assert (workspace / "skills" / "skill-manager" / ".spoon-bot-builtin").exists()
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


def test_ensure_builtin_skills_skips_current_managed_copy(tmp_path):
    workspace = tmp_path / "workspace"

    ensure_builtin_skills(workspace)
    installed = ensure_builtin_skills(workspace)

    assert installed == []


def test_ensure_builtin_skills_refreshes_managed_builtin_copy(tmp_path):
    workspace = tmp_path / "workspace"
    source = builtin_skills_root() / "skill-manager"
    target = workspace / "skills" / "skill-manager"
    target.mkdir(parents=True, exist_ok=True)
    (target / "SKILL.md").write_text(
        (source / "SKILL.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (target / "tools.py").write_text("# stale copy\n", encoding="utf-8")

    installed = ensure_builtin_skills(workspace)

    assert "skill-manager" in {path.name for path in installed}
    assert (target / "tools.py").read_text(encoding="utf-8") == (
        source / "tools.py"
    ).read_text(encoding="utf-8")
    assert (target / ".spoon-bot-builtin").read_text(encoding="utf-8") == "skill-manager"


def test_builtin_skills_root_contains_builtin_skill_set():
    root = builtin_skills_root()
    assert root.is_dir()
    assert (root / "wallet" / "SKILL.md").exists()
    assert (root / "subagents" / "SKILL.md").exists()
    assert (root / "service_expose" / "SKILL.md").exists()
    assert (root / "skill-manager" / "SKILL.md").exists()
    assert (root / "skill-manager" / "tools.py").exists()
    assert (root / "service_expose" / "scripts" / "service_expose.py").exists()
    assert not (root / "wallet" / "scripts").exists()
    assert not (root / "wallet" / "assets").exists()
