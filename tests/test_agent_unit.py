"""Agent unit tests: dynamic tools, document parsing, and performance benchmarks.

Merged from:
  - test_dynamic_tools_prompt.py (§1: dynamic prompt + ActivateToolTool + registry clean)
  - test_document.py             (§2: DocumentParseTool)
  - test_perf_startup.py         (§3: startup / registration / prompt speed benchmarks)
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ═══════════════════════════════════════════════════════════════════
# §1  Dynamic System Prompt & ActivateToolTool
# ═══════════════════════════════════════════════════════════════════

from spoon_bot.agent.loop import AgentLoop


# -- helpers --

def _make_mock_tool(name: str, description: str) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    tool.description = description
    return tool


def _inactive_tools_from_names(nd: dict[str, str]) -> dict[str, MagicMock]:
    return {n: _make_mock_tool(n, d) for n, d in nd.items()}


class TestBuildDynamicToolsPrompt:

    def test_header_present(self):
        inactive = _inactive_tools_from_names({"some_tool": "Does something"})
        prompt = AgentLoop._build_dynamic_tools_prompt(inactive)
        assert "## Dynamically Loadable Tools" in prompt

    def test_all_tools_listed(self):
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
        inactive = _inactive_tools_from_names({"my_tool": "My tool description"})
        prompt = AgentLoop._build_dynamic_tools_prompt(inactive)
        assert "activate_tool" in prompt
        assert "action='activate'" in prompt

    def test_prefer_specialized_hint(self):
        inactive = _inactive_tools_from_names({"get_token_price": "Price data"})
        prompt = AgentLoop._build_dynamic_tools_prompt(inactive)
        assert "prefer specialized tools" in prompt.lower()

    def test_empty_inactive(self):
        prompt = AgentLoop._build_dynamic_tools_prompt({})
        assert "## Dynamically Loadable Tools" in prompt

    def test_no_topic_references(self):
        inactive = _inactive_tools_from_names({
            "get_token_price": "Price data",
            "balance_check": "Balance check",
        })
        prompt = AgentLoop._build_dynamic_tools_prompt(inactive)
        assert "action='recommend'" not in prompt
        assert "topic=" not in prompt


class TestActivateToolTool:

    def _make_tool(self, tracker: list | None = None):
        from spoon_bot.agent.tools.self_config import ActivateToolTool
        activated = tracker if tracker is not None else []

        def fake_activate(name: str) -> bool:
            activated.append(name)
            return True

        def fake_list_inactive():
            return [
                {"name": "tool_a", "description": "A"},
                {"name": "tool_b", "description": "B"},
            ]

        return ActivateToolTool(activate_fn=fake_activate, list_inactive_fn=fake_list_inactive)

    def test_no_topic_in_desc(self):
        desc = self._make_tool().description
        assert "activate" in desc.lower()
        assert "topic=" not in desc

    def test_params_no_topic(self):
        params = self._make_tool().parameters
        assert "topic" not in params["properties"]
        assert "action" in params["properties"]
        assert "tool_name" in params["properties"]

    def test_only_activate_list(self):
        actions = self._make_tool().parameters["properties"]["action"]["enum"]
        assert actions == ["activate", "list"]

    @pytest.mark.asyncio
    async def test_list_action(self):
        r = await self._make_tool().execute(action="list")
        assert "tool_a" in r and "tool_b" in r

    @pytest.mark.asyncio
    async def test_activate_single(self):
        a = []
        r = await self._make_tool(a).execute(action="activate", tool_name="tool_a")
        assert "Activated" in r and "tool_a" in a

    @pytest.mark.asyncio
    async def test_activate_multi(self):
        a = []
        r = await self._make_tool(a).execute(action="activate", tool_name="tool_a,tool_b,tool_c")
        assert "Activated" in r
        assert a == ["tool_a", "tool_b", "tool_c"]

    @pytest.mark.asyncio
    async def test_activate_missing_name(self):
        r = await self._make_tool().execute(action="activate")
        assert "Error" in r


class TestRegistryClean:

    def test_no_topic_keywords(self):
        import spoon_bot.agent.tools.registry as reg
        assert not hasattr(reg, "TOPIC_KEYWORDS")

    def test_no_topic_meta(self):
        import spoon_bot.agent.tools.registry as reg
        assert not hasattr(reg, "TOPIC_META")

    def test_no_tool_recommendations(self):
        import spoon_bot.agent.tools.registry as reg
        assert not hasattr(reg, "TOOL_RECOMMENDATIONS")

    def test_core_tools_exist(self):
        from spoon_bot.agent.tools.registry import CORE_TOOLS
        assert "activate_tool" in CORE_TOOLS
        assert "web_search" in CORE_TOOLS


# ═══════════════════════════════════════════════════════════════════
# §2  Document Parse Tool
# ═══════════════════════════════════════════════════════════════════

from spoon_bot.agent.tools.document import DocumentParseTool


class TestDocumentParseTool:

    def setup_method(self) -> None:
        self.tool = DocumentParseTool(workspace="/tmp/test_ws")

    def test_name_and_description(self) -> None:
        assert self.tool.name == "document_parse"
        assert "PDF" in self.tool.description

    def test_page_range_all(self) -> None:
        assert self.tool._parse_page_range(None, 5) == [0, 1, 2, 3, 4]

    def test_page_range_single(self) -> None:
        assert self.tool._parse_page_range("3", 10) == [2]

    def test_page_range_range(self) -> None:
        assert self.tool._parse_page_range("2-4", 10) == [1, 2, 3]

    def test_page_range_mixed(self) -> None:
        assert self.tool._parse_page_range("1,3,5-7", 10) == [0, 2, 4, 5, 6]

    def test_page_range_out_of_bounds(self) -> None:
        assert self.tool._parse_page_range("50", 10) == []

    @pytest.mark.asyncio
    async def test_file_not_found(self) -> None:
        r = await self.tool.execute(file_path="/nonexistent/file.pdf")
        assert "Error" in r
        assert "not found" in r or "pymupdf" in r

    @pytest.mark.asyncio
    async def test_not_pdf(self, tmp_path: Path) -> None:
        (tmp_path / "test.txt").write_text("hello")
        r = await self.tool.execute(file_path=str(tmp_path / "test.txt"))
        assert "PDF" in r

    @pytest.mark.asyncio
    async def test_pymupdf_not_installed(self, tmp_path: Path) -> None:
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        with patch.dict("sys.modules", {"fitz": None}):
            import importlib
            from spoon_bot.agent.tools import document as doc_mod
            importlib.reload(doc_mod)
            t = doc_mod.DocumentParseTool(workspace=str(tmp_path))
            r = await t.execute(file_path=str(pdf))
            assert isinstance(r, str)

    @pytest.mark.asyncio
    async def test_parse_basic_text(self, tmp_path: Path) -> None:
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF")
        t = DocumentParseTool(workspace=str(tmp_path))
        r = await t.execute(file_path=str(pdf))
        assert isinstance(r, str)
        if "pymupdf" not in r:
            assert "Parsed" in r or "Error" in r

    @pytest.mark.asyncio
    async def test_empty_path(self) -> None:
        r = await self.tool.execute(file_path="")
        assert "Error" in r


# ═══════════════════════════════════════════════════════════════════
# §3  Performance & Startup Benchmarks
# ═══════════════════════════════════════════════════════════════════


def test_toolkit_adapter_load_time():
    """ToolkitAdapter.load_all() concurrent loading benchmark."""
    try:
        from spoon_bot.toolkit.adapter import ToolkitAdapter
    except ImportError:
        pytest.skip("spoon_bot.toolkit not available")

    times: list[float] = []
    for _ in range(3):
        start = time.perf_counter()
        adapter = ToolkitAdapter()
        tools = adapter.load_all()
        times.append(time.perf_counter() - start)

    avg = statistics.mean(times)
    print(f"\n[PERF] ToolkitAdapter.load_all() avg {avg:.3f}s")
    assert avg < 10.0


def test_native_tool_registration_time():
    """_register_native_tools() registration benchmark."""
    times: list[float] = []
    for _ in range(3):
        start = time.perf_counter()
        loop = AgentLoop.__new__(AgentLoop)
        loop.workspace = __import__("pathlib").Path(".")
        loop.shell_timeout = 30
        loop.max_output = 8000
        loop.memory = type("M", (), {
            "get_memory_context": lambda self: "",
            "store": lambda self, **kw: None,
            "search": lambda self, **kw: [],
            "get_all": lambda self, **kw: [],
        })()

        class _FakeTM:
            def __init__(self):
                self._tools: dict = {}
            def register(self, tool):
                self._tools[tool.name] = tool
            def list_tools(self):
                return sorted(self._tools.keys())
            def set_tool_filter(self, **kw):
                pass
            def get_inactive_tools(self):
                return {}
            def get_active_tools(self):
                return self._tools
            def __len__(self):
                return len(self._tools)

        loop.tools = _FakeTM()
        loop.add_tool = lambda name: False
        loop._register_native_tools()
        times.append(time.perf_counter() - start)

    avg = statistics.mean(times)
    print(f"\n[PERF] _register_native_tools() avg {avg:.3f}s, tools={loop.tools.list_tools()}")
    assert avg < 5.0


@pytest.mark.asyncio
async def test_http_client_pool_reuse():
    """Shared httpx client singleton check."""
    from spoon_bot.agent.tools.web import _get_http_client
    assert _get_http_client() is _get_http_client()


def test_build_dynamic_prompt_speed():
    """_build_dynamic_tools_prompt() timing (20 tools × 100 iterations)."""

    class FakeTool:
        def __init__(self, n, d):
            self.name = n
            self.description = d

    inactive = {f"tool_{i}": FakeTool(f"tool_{i}", f"Desc {i}") for i in range(20)}
    times = []
    for _ in range(100):
        start = time.perf_counter()
        AgentLoop._build_dynamic_tools_prompt(inactive)
        times.append(time.perf_counter() - start)

    avg_ms = statistics.mean(times) * 1000
    print(f"\n[PERF] prompt build avg {avg_ms:.3f}ms")
    assert avg_ms < 10.0


@pytest.mark.asyncio
async def test_activate_tool_multi_speed():
    """ActivateToolTool multi-activate benchmark (10 tools)."""
    from spoon_bot.agent.tools.self_config import ActivateToolTool
    a: list[str] = []
    tool = ActivateToolTool(
        activate_fn=lambda n: (a.append(n), True)[1],
        list_inactive_fn=lambda: [],
    )
    names = ",".join(f"tool_{i}" for i in range(10))
    start = time.perf_counter()
    r = await tool.execute(action="activate", tool_name=names)
    ms = (time.perf_counter() - start) * 1000
    print(f"\n[PERF] multi-activate 10 tools: {ms:.2f}ms")
    assert len(a) == 10
    assert "Activated" in r
    assert ms < 50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
