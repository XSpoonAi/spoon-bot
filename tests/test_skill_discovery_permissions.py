from pathlib import Path

from spoon_bot.agent.loop import AgentLoop


def _make_loop(workspace: Path) -> AgentLoop:
    loop = AgentLoop.__new__(AgentLoop)
    loop.workspace = workspace
    loop._skill_paths = []
    loop._touched_paths = set()
    return loop


def test_workspace_skill_discovery_ignores_lost_found(tmp_path: Path, monkeypatch) -> None:
    lost_found = tmp_path / "lost+found"
    lost_found.mkdir()
    skill_dir = tmp_path / "example"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# Example\n", encoding="utf-8")

    original_exists = Path.exists

    def guarded_exists(path: Path) -> bool:
        if path == lost_found / "SKILL.md":
            raise AssertionError("lost+found must not be inspected as a skill")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", guarded_exists)

    candidates = _make_loop(tmp_path)._iter_skill_candidates(include_dormant=True)

    assert [name for name, *_ in candidates] == ["example"]


def test_workspace_skill_discovery_skips_inaccessible_candidates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    skill_dir = tmp_path / "example"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# Example\n", encoding="utf-8")

    original_exists = Path.exists

    def permission_denied(path: Path) -> bool:
        if path == blocked / "SKILL.md":
            raise PermissionError(f"permission denied: {path}")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", permission_denied)

    candidates = _make_loop(tmp_path)._iter_skill_candidates(include_dormant=True)

    assert [name for name, *_ in candidates] == ["example"]
