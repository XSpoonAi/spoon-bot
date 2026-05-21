"""Tests for dynamic system prompt generation and ActivateToolTool.

Validates that:
- _build_dynamic_tools_prompt correctly lists all inactive tools
- ActivateToolTool supports activate (single + multi) and list actions
- No hardcoded topic/keyword mapping exists
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from spoon_bot.agent.loop import AgentLoop


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_tool(name: str, description: str) -> MagicMock:
    """Create a mock Tool with name and description properties."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    return tool


def _inactive_tools_from_names(names_descs: dict[str, str]) -> dict[str, MagicMock]:
    """Build an {name: MockTool} dict from {name: description} mapping."""
    return {n: _make_mock_tool(n, d) for n, d in names_descs.items()}


# ---------------------------------------------------------------------------
# Tests: _build_dynamic_tools_prompt
# ---------------------------------------------------------------------------

class TestBuildDynamicToolsPrompt:
    """Tests for AgentLoop._build_dynamic_tools_prompt (no topic config)."""

    def test_header_present(self):
        """Prompt should include the dynamic tools header."""
        inactive = _inactive_tools_from_names({
            "some_tool": "Does something",
        })
        prompt = AgentLoop._build_dynamic_tools_prompt(inactive)
        assert "## Dynamically Loadable Tools" in prompt

    def test_all_tools_listed(self):
        """Every inactive tool should appear in the prompt."""
        names = {
            "get_token_price": "Get token price from DEX",
            "balance_check": "Check wallet balance",
            "document_parse": "Parse documents",
            "custom_xyz": "Custom tool",
        }
        inactive = _inactive_tools_from_names(names)
        prompt = AgentLoop._build_dynamic_tools_prompt(inactive)
        for name, desc in names.items():
            assert f"`{name}`" in prompt
            assert desc in prompt

    def test_activate_tool_instructions(self):
        """Prompt should explain how to use activate_tool."""
        inactive = _inactive_tools_from_names({
            "my_tool": "My tool description",
        })
        prompt = AgentLoop._build_dynamic_tools_prompt(inactive)
        assert "activate_tool" in prompt
        assert "action='activate'" in prompt

    def test_prefer_specialized_hint(self):
        """Prompt should hint to prefer specialized tools over web_search."""
        inactive = _inactive_tools_from_names({
            "get_token_price": "Price data",
        })
        prompt = AgentLoop._build_dynamic_tools_prompt(inactive)
        assert "prefer specialized tools" in prompt.lower()

    def test_empty_inactive_returns_minimal(self):
        """With no inactive tools, still has header but no tool entries."""
        prompt = AgentLoop._build_dynamic_tools_prompt({})
        assert "## Dynamically Loadable Tools" in prompt

    def test_no_topic_references(self):
        """Prompt should NOT contain any hardcoded topic references."""
        inactive = _inactive_tools_from_names({
            "get_token_price": "Price data",
            "balance_check": "Balance check",
        })
        prompt = AgentLoop._build_dynamic_tools_prompt(inactive)
        # No topic-based activation
        assert "action='recommend'" not in prompt
        assert "topic=" not in prompt


# ---------------------------------------------------------------------------
# Tests: ActivateToolTool
# ---------------------------------------------------------------------------

class TestActivateToolTool:
    """Tests for the simplified ActivateToolTool."""

    def _make_tool(self, activated_tracker: list | None = None):
        from spoon_bot.agent.tools.self_config import ActivateToolTool

        activated = activated_tracker if activated_tracker is not None else []

        def fake_activate(name: str) -> bool:
            activated.append(name)
            return True

        def fake_list_inactive():
            return [
                {"name": "tool_a", "description": "Description A"},
                {"name": "tool_b", "description": "Description B"},
            ]

        return ActivateToolTool(
            activate_fn=fake_activate,
            list_inactive_fn=fake_list_inactive,
        )

    def test_description_no_topics(self):
        """Description should NOT contain topic-based instructions."""
        tool = self._make_tool()
        desc = tool.description
        # Should mention activation but not hardcoded topics
        assert "activate" in desc.lower()
        assert "topic=" not in desc
        assert "recommend" not in desc

    def test_parameters_no_topic(self):
        """Parameters should not have a 'topic' field."""
        tool = self._make_tool()
        params = tool.parameters
        assert "topic" not in params["properties"]
        assert "action" in params["properties"]
        assert "tool_name" in params["properties"]

    def test_parameters_actions(self):
        """Only 'activate' and 'list' actions should exist."""
        tool = self._make_tool()
        actions = tool.parameters["properties"]["action"]["enum"]
        assert actions == ["activate", "list"]

    @pytest.mark.asyncio
    async def test_list_action(self):
        """List action returns inactive tools."""
        tool = self._make_tool()
        result = await tool.execute(action="list")
        assert "tool_a" in result
        assert "tool_b" in result

    @pytest.mark.asyncio
    async def test_activate_single(self):
        """Activate a single tool."""
        activated = []
        tool = self._make_tool(activated)
        result = await tool.execute(action="activate", tool_name="tool_a")
        assert "Activated" in result
        assert "tool_a" in activated

    @pytest.mark.asyncio
    async def test_activate_multiple_comma_separated(self):
        """Activate multiple tools via comma-separated names."""
        activated = []
        tool = self._make_tool(activated)
        result = await tool.execute(
            action="activate",
            tool_name="tool_a,tool_b,tool_c",
        )
        assert "Activated" in result
        assert activated == ["tool_a", "tool_b", "tool_c"]

    @pytest.mark.asyncio
    async def test_activate_missing_name(self):
        """Activate without tool_name returns error."""
        tool = self._make_tool()
        result = await tool.execute(action="activate")
        assert "Error" in result


class TestToolProfileSkillManagement:
    """Skill management belongs to the default tool profile, not prompt routing."""

    def test_core_profile_includes_skill_marketplace(self):
        from spoon_bot.agent.tools.registry import CORE_TOOLS

        assert "skill_marketplace" in CORE_TOOLS

    def test_core_profile_includes_search_history_for_prior_fact_checks(self):
        from spoon_bot.agent.tools.registry import CORE_TOOLS

        assert "search_history" in CORE_TOOLS

    def test_prior_action_dispute_sets_current_session_fact_check_hint(self):
        from spoon_bot.agent.request_hints import build_request_execution_hints

        hints = build_request_execution_hints(
            "不是啊，你刚刚不是已经运行过很多次，你忘记了吗？",
            [],
        )

        assert hints["current_session_fact_check_required"] is True

    def test_local_skill_context_uses_distinctive_name_token_without_route(self, tmp_path):
        from spoon_bot.agent.request_hints import (
            build_request_execution_hints,
            format_local_executable_skill_context,
        )

        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text(
            "---\n"
            "name: spot-agent-cypher\n"
            "description: Play a local game workflow through its CLI.\n"
            "---\n"
            "CLI = node skills/spot-agent-cypher/cli/index.js\n"
            "$CLI wallet\n"
            "$CLI join <spot>\n",
            encoding="utf-8",
        )

        hints = build_request_execution_hints(
            "继续加入spot游戏",
            [
                {
                    "name": "spot-agent-cypher",
                    "skill_md": skill_md,
                    "is_organized": True,
                    "description": "Play a local game workflow through its CLI.",
                    "when_to_use": "",
                }
            ],
        )
        context = format_local_executable_skill_context(hints)

        assert hints["local_executable_skills"][0]["name"] == "spot-agent-cypher"
        assert "[LOCAL SKILL EXECUTION CONTEXT]:" in context
        assert "skills/spot-agent-cypher/SKILL.md" in context
        assert "neighboring subcommands" in context
        assert "Agent-Cypher-Lab" not in context

    def test_github_skill_install_context_points_to_skill_manager(self):
        from spoon_bot.agent.request_hints import (
            build_request_execution_hints,
            format_github_skill_install_context,
        )

        hints = build_request_execution_hints(
            "https://github.com/example-org/example-skill install the skill in this repo",
            [],
        )
        context = format_github_skill_install_context(hints)

        assert hints["github_skill_install_request"]["urls"] == [
            "https://github.com/example-org/example-skill"
        ]
        assert "skill_marketplace(action='install_skill'" in context
        assert "do not clone directly into workspace/skills" in context
        assert "example-org/example-skill" in context

    def test_request_scoped_activation_hook_has_no_prompt_route(self):
        from spoon_bot.agent.loop import AgentLoop

        loop = AgentLoop.__new__(AgentLoop)

        activated = AgentLoop._activate_tools_for_request_hints(
            loop,
            {"arbitrary": "hint"},
        )

        assert activated == []


# ---------------------------------------------------------------------------
# Tests: No hardcoded config in registry
# ---------------------------------------------------------------------------

class TestRegistryClean:
    """Ensure registry.py has no hardcoded topic config."""

    def test_no_topic_keywords(self):
        """TOPIC_KEYWORDS should not exist in registry."""
        import spoon_bot.agent.tools.registry as reg
        assert not hasattr(reg, "TOPIC_KEYWORDS")

    def test_no_topic_meta(self):
        """TOPIC_META should not exist in registry."""
        import spoon_bot.agent.tools.registry as reg
        assert not hasattr(reg, "TOPIC_META")

    def test_no_tool_recommendations(self):
        """TOOL_RECOMMENDATIONS should not exist in registry."""
        import spoon_bot.agent.tools.registry as reg
        assert not hasattr(reg, "TOOL_RECOMMENDATIONS")

    def test_core_tools_still_exist(self):
        """CORE_TOOLS should still exist."""
        from spoon_bot.agent.tools.registry import CORE_TOOLS
        assert "activate_tool" in CORE_TOOLS
        assert "web_search" in CORE_TOOLS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
