from __future__ import annotations

import io
import importlib.util
import sys
import types
import zipfile
from pathlib import Path

import pytest


def _zip_bytes(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for path, content in files.items():
            zf.writestr(path, content)
    return buffer.getvalue()


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


def test_skill_install_server_disconnect_is_retryable(skill_manager_module):
    error = RuntimeError("Server disconnected without sending a response.")

    assert skill_manager_module._is_transient_install_error(error) is True
    assert "Retry the same skill_marketplace" in skill_manager_module._format_install_error(error)


@pytest.mark.asyncio
async def test_resolve_github_skill_source_retries_public_api_after_bad_token(
    skill_manager_module,
    monkeypatch: pytest.MonkeyPatch,
):
    import httpx

    monkeypatch.setenv("GITHUB_TOKEN", "stale-token")
    calls: list[dict[str, str]] = []

    class FakeAsyncClient:
        def __init__(self, timeout: int):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url: str, **kwargs):
            headers = kwargs.get("headers") or {}
            calls.append(headers)
            request = httpx.Request("GET", url)
            if "Authorization" in headers:
                return httpx.Response(
                    401,
                    request=request,
                    json={"message": "Bad credentials"},
                )
            return httpx.Response(
                200,
                request=request,
                json={"tree": [{"type": "blob", "path": "SKILL.md"}]},
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    resolved = await skill_manager_module._resolve_github_skill_source(
        "owner",
        "repo",
        "main",
        "",
    )

    assert resolved == ("main", "", "repo")
    assert calls[0]["Authorization"] == "Bearer stale-token"
    assert "Authorization" not in calls[1]


@pytest.mark.asyncio
async def test_download_via_api_retries_public_api_after_bad_token(
    skill_manager_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    import httpx

    monkeypatch.setenv("GITHUB_TOKEN", "stale-token")
    calls: list[tuple[str, dict[str, str]]] = []

    class FakeAsyncClient:
        def __init__(self, timeout: int):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url: str, **kwargs):
            headers = kwargs.get("headers") or {}
            calls.append((url, headers))
            request = httpx.Request("GET", url)
            if url.startswith("https://api.github.com/") and "Authorization" in headers:
                return httpx.Response(
                    401,
                    request=request,
                    json={"message": "Bad credentials"},
                )
            if url.startswith("https://api.github.com/"):
                return httpx.Response(
                    200,
                    request=request,
                    json=[{"type": "file", "path": "SKILL.md", "name": "SKILL.md"}],
                )
            return httpx.Response(
                200,
                request=request,
                content=b"---\nname: repo\n---\n# Repo\n",
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    target = tmp_path / "repo"
    count = await skill_manager_module._download_via_api(
        "owner",
        "repo",
        "main",
        "",
        target,
    )

    assert count == 1
    assert (target / "SKILL.md").read_text(encoding="utf-8").startswith("---")
    assert calls[0][1]["Authorization"] == "Bearer stale-token"
    assert "Authorization" not in calls[1][1]


@pytest.mark.asyncio
async def test_resolve_github_skill_source_falls_back_to_public_archive_after_api_rate_limit(
    skill_manager_module,
    monkeypatch: pytest.MonkeyPatch,
):
    import httpx

    archive = _zip_bytes({
        "agent-spot-cypher-main/SKILL.md": "---\nname: spot-agent-cypher\n---\n",
        "agent-spot-cypher-main/cli/index.js": "console.log('ok')",
    })
    calls: list[str] = []

    class FakeAsyncClient:
        def __init__(self, timeout: int, follow_redirects: bool = False):
            self.timeout = timeout
            self.follow_redirects = follow_redirects

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url: str, **kwargs):
            calls.append(url)
            request = httpx.Request("GET", url)
            if url.startswith("https://api.github.com/"):
                return httpx.Response(
                    403,
                    request=request,
                    json={"message": "rate limit exceeded"},
                )
            if url.startswith("https://codeload.github.com/"):
                return httpx.Response(200, request=request, content=archive)
            return httpx.Response(404, request=request)

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    resolved = await skill_manager_module._resolve_github_skill_source(
        "Agent-Cypher-Lab",
        "agent-spot-cypher",
        "main",
        "",
    )

    assert resolved == ("main", "", "agent-spot-cypher")
    assert any(url.startswith("https://api.github.com/") for url in calls)
    assert any(url.startswith("https://codeload.github.com/") for url in calls)


@pytest.mark.asyncio
async def test_download_skill_files_uses_public_archive_without_git_or_api_token(
    skill_manager_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    import httpx

    archive = _zip_bytes({
        "repo-main/README.md": "# Repo",
        "repo-main/skills/demo/SKILL.md": "---\nname: demo\n---\n# Demo\n",
        "repo-main/skills/demo/cli/index.js": "console.log('ok')",
    })
    calls: list[str] = []

    async def git_missing(*args, **kwargs):
        raise FileNotFoundError("git")

    async def api_must_not_run(*args, **kwargs):
        raise AssertionError("GitHub Contents API should not be required")

    class FakeAsyncClient:
        def __init__(self, timeout: int, follow_redirects: bool = False):
            self.timeout = timeout
            self.follow_redirects = follow_redirects

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url: str, **kwargs):
            calls.append(url)
            request = httpx.Request("GET", url)
            if url.startswith("https://codeload.github.com/"):
                return httpx.Response(200, request=request, content=archive)
            return httpx.Response(404, request=request)

    monkeypatch.setattr(skill_manager_module, "_download_via_git", git_missing)
    monkeypatch.setattr(skill_manager_module, "_download_via_api", api_must_not_run)
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    target = tmp_path / "demo"
    count = await skill_manager_module._download_skill_files(
        "owner",
        "repo",
        "main",
        "skills/demo",
        target,
    )

    assert count == 2
    assert (target / "SKILL.md").exists()
    assert (target / "cli" / "index.js").exists()
    assert not (target / "README.md").exists()
    assert any(url.startswith("https://codeload.github.com/") for url in calls)


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
    assert "Complete any Setup/Prerequisites/Install/dependency steps first" in result
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
