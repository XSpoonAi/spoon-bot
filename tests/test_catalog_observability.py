from __future__ import annotations

from pathlib import Path

from spoon_bot.agent.loop import AgentLoop


def test_agent_loop_skill_catalog_lists_workspace_skill(tmp_path: Path):
    skill_dir = tmp_path / "skills" / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: demo-skill\n"
        "description: Demo skill for catalog tests\n"
        "paths:\n"
        "  - 'src/**'\n"
        "---\n"
        "# Demo\n",
        encoding="utf-8",
    )

    loop = AgentLoop(
        workspace=tmp_path,
        model="test-model",
        provider="openai",
        enable_skills=False,
        auto_commit=False,
    )

    catalog = loop.get_skill_catalog()

    demo = next(item for item in catalog if item["name"] == "demo-skill")
    assert demo["description"] == "Demo skill for catalog tests"
    assert demo["source"] == "workspace"
    assert demo["status"] == "dormant"
    assert demo["organized"] is True
    assert demo["skill_md"].endswith("SKILL.md")


def test_agent_loop_mcp_catalog_lists_configured_servers(tmp_path: Path):
    loop = AgentLoop(
        workspace=tmp_path,
        model="test-model",
        provider="openai",
        mcp_config={
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", str(tmp_path)],
            }
        },
        enable_skills=False,
        auto_commit=False,
    )

    catalog = loop.get_mcp_catalog()

    assert catalog == [
        {
            "name": "filesystem",
            "transport": "stdio",
            "command": "npx",
            "url": None,
            "status": "configured",
            "tool_count": 0,
            "tools": [],
        }
    ]
