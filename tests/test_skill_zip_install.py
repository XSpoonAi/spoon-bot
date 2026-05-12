from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from spoon_bot.agent.loop import AgentLoop
from spoon_bot.skills.zip_install import install_skill_zip_archive


def _write_zip(path: Path, files: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)


def test_flat_skill_zip_installs_under_workspace_skills_and_strips_timestamp(tmp_path: Path) -> None:
    archive = tmp_path / "agent-spot-cypher-skill-20260509-110247.zip"
    _write_zip(
        archive,
        {
            "SKILL.md": "# Spot skill",
            "cli/index.js": "console.log('join')",
            "references/api.md": "# API",
        },
    )

    result = install_skill_zip_archive(archive, tmp_path)

    assert result is not None
    assert result.name == "agent-spot-cypher-skill"
    assert result.target_dir == tmp_path / "skills" / "agent-spot-cypher-skill"
    assert result.skill_md.exists()
    assert (result.target_dir / "cli" / "index.js").exists()
    assert not (tmp_path / "skills" / "SKILL.md").exists()


def test_skill_zip_prefers_skill_md_name_for_install_directory(tmp_path: Path) -> None:
    archive = tmp_path / "agent-spot-cypher-skill-20260509-110247.zip"
    _write_zip(
        archive,
        {
            "SKILL.md": "---\nname: spot-agent-cypher\n---\n# Spot skill",
            "cli/index.js": "console.log('join')",
        },
    )

    result = install_skill_zip_archive(archive, tmp_path)

    assert result is not None
    assert result.name == "spot-agent-cypher"
    assert result.target_dir == tmp_path / "skills" / "spot-agent-cypher"
    assert (result.target_dir / "cli" / "index.js").exists()
    assert not (tmp_path / "skills" / "agent-spot-cypher-skill").exists()


def test_skill_zip_reinstall_replaces_existing_directory(tmp_path: Path) -> None:
    archive = tmp_path / "demo-skill.zip"
    _write_zip(archive, {"SKILL.md": "# New", "cli/index.js": "new"})
    existing = tmp_path / "skills" / "demo-skill"
    existing.mkdir(parents=True)
    (existing / "old.txt").write_text("old", encoding="utf-8")

    result = install_skill_zip_archive(archive, tmp_path)

    assert result is not None
    assert result.reinstalled is True
    assert (existing / "SKILL.md").read_text(encoding="utf-8") == "# New"
    assert not (existing / "old.txt").exists()


def test_nested_skill_zip_installs_single_root(tmp_path: Path) -> None:
    archive = tmp_path / "packaged.zip"
    _write_zip(
        archive,
        {
            "custom-skill/SKILL.md": "# Nested",
            "custom-skill/scripts/run.sh": "echo ok",
        },
    )

    result = install_skill_zip_archive(archive, tmp_path)

    assert result is not None
    assert result.name == "custom-skill"
    assert (tmp_path / "skills" / "custom-skill" / "SKILL.md").exists()
    assert (tmp_path / "skills" / "custom-skill" / "scripts" / "run.sh").exists()


def test_skill_zip_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    _write_zip(archive, {"SKILL.md": "# Bad", "../escape.txt": "bad"})

    with pytest.raises(ValueError, match="Unsafe zip entry path"):
        install_skill_zip_archive(archive, tmp_path)


@pytest.mark.asyncio
async def test_agent_loop_installs_attached_skill_zip_before_planning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    archive = uploads / "agent-spot-cypher-skill-20260509-110247.zip"
    _write_zip(archive, {"SKILL.md": "# Spot", "cli/index.js": "console.log('join')"})

    loop = AgentLoop(model="test-model", provider="openai", workspace=tmp_path)
    reload_calls = 0

    async def fake_reload_skills() -> dict:
        nonlocal reload_calls
        reload_calls += 1
        return {"before": [], "after": ["agent-spot-cypher-skill"], "added": ["agent-spot-cypher-skill"], "removed": []}

    monkeypatch.setattr(loop, "reload_skills", fake_reload_skills)

    installed = await loop._install_skill_zip_attachments([
        {
            "name": archive.name,
            "mime_type": "application/zip",
            "workspace_path": "/workspace/uploads/agent-spot-cypher-skill-20260509-110247.zip",
        }
    ])

    assert len(installed) == 1
    assert reload_calls == 1
    assert installed[0].name == "agent-spot-cypher-skill"
    assert (tmp_path / "skills" / "agent-spot-cypher-skill" / "SKILL.md").exists()
    assert not (tmp_path / "skills" / "SKILL.md").exists()
    assert "skills/agent-spot-cypher-skill/SKILL.md" in loop._current_turn_skill_zip_context()
