"""
Performance test: measure startup, tool registration, and prompt build timing.

Run:  python -m pytest tests/test_perf_startup.py -v -s
"""
from __future__ import annotations

import time
import statistics
import pytest

# --------------- Toolkit concurrent loading benchmark ---------------

def test_toolkit_adapter_load_time():
    """Measure how long ToolkitAdapter.load_all() takes (concurrent loading)."""
    try:
        from spoon_bot.toolkit.adapter import ToolkitAdapter
    except ImportError:
        pytest.skip("spoon_bot.toolkit not available")

    times: list[float] = []
    loaded_counts: list[int] = []
    for _ in range(3):
        start = time.perf_counter()
        adapter = ToolkitAdapter()
        tools = adapter.load_all()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        loaded_counts.append(len(tools))

    avg = statistics.mean(times)
    print(f"\n[PERF] ToolkitAdapter.load_all() — avg {avg:.3f}s "
          f"(min={min(times):.3f}s, max={max(times):.3f}s), "
          f"tools loaded: {loaded_counts[0]}")
    # Concurrent loading should keep this under 10s
    assert avg < 10.0, f"Toolkit loading too slow: avg {avg:.3f}s"


# --------------- Native tool registration benchmark ---------------

def test_native_tool_registration_time():
    """Measure how long _register_native_tools() takes."""
    from spoon_bot.agent.loop import AgentLoop

    times: list[float] = []
    for _ in range(3):
        start = time.perf_counter()
        loop = AgentLoop.__new__(AgentLoop)
        # Set minimal attributes needed for registration
        loop.workspace = __import__("pathlib").Path(".")
        loop.shell_timeout = 30
        loop.max_output = 8000
        loop.memory = type("M", (), {
            "get_memory_context": lambda self: "",
            "store": lambda self, **kw: None,
            "search": lambda self, **kw: [],
            "get_all": lambda self, **kw: [],
        })()

        # Lightweight ToolManager mock
        class _FakeTM:
            def __init__(self):
                self._tools: dict = {}
                self._filter: set | None = None

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
        # add_tool stub (needed for ActivateToolTool)
        loop.add_tool = lambda name: False

        loop._register_native_tools()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg = statistics.mean(times)
    tool_names = loop.tools.list_tools()
    print(f"\n[PERF] _register_native_tools() — avg {avg:.3f}s "
          f"(min={min(times):.3f}s, max={max(times):.3f}s), "
          f"tools={len(tool_names)}: {tool_names}")
    # Registration should be near-instant (<5s even with toolkit)
    assert avg < 5.0, f"Tool registration too slow: avg {avg:.3f}s"


# --------------- HTTP client pool reuse benchmark ---------------

@pytest.mark.asyncio
async def test_http_client_pool_reuse():
    """Verify shared httpx client is reused across calls."""
    from spoon_bot.agent.tools.web import _get_http_client

    client1 = _get_http_client()
    client2 = _get_http_client()
    assert client1 is client2, "HTTP client should be reused (singleton)"
    print(f"\n[PERF] Shared httpx client confirmed: id={id(client1)}")


# --------------- System prompt build benchmark ---------------

def test_build_dynamic_prompt_speed():
    """Measure _build_dynamic_tools_prompt() timing."""
    from spoon_bot.agent.loop import AgentLoop

    # Create fake inactive tools
    class FakeTool:
        def __init__(self, name: str, desc: str):
            self.name = name
            self.description = desc

    inactive = {
        f"tool_{i}": FakeTool(f"tool_{i}", f"Description for tool {i}")
        for i in range(20)
    }

    times: list[float] = []
    for _ in range(100):
        start = time.perf_counter()
        prompt = AgentLoop._build_dynamic_tools_prompt(inactive)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_ms = statistics.mean(times) * 1000
    print(f"\n[PERF] _build_dynamic_tools_prompt() — avg {avg_ms:.3f}ms "
          f"(20 tools, 100 iterations)")
    assert avg_ms < 10.0, f"Prompt build too slow: avg {avg_ms:.3f}ms"
    assert len(prompt) > 0


# --------------- ActivateToolTool multi-activate benchmark ---------------

@pytest.mark.asyncio
async def test_activate_tool_multi_speed():
    """Measure ActivateToolTool multi-activate (comma-separated) speed."""
    from spoon_bot.agent.tools.self_config import ActivateToolTool

    activated: list[str] = []

    tool = ActivateToolTool(
        activate_fn=lambda name: (activated.append(name), True)[1],
        list_inactive_fn=lambda: [],
    )

    names = ",".join(f"tool_{i}" for i in range(10))

    start = time.perf_counter()
    result = await tool.execute(action="activate", tool_name=names)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"\n[PERF] ActivateToolTool multi-activate (10 tools) — {elapsed_ms:.2f}ms")
    assert len(activated) == 10
    assert "Activated" in result
    assert elapsed_ms < 50.0
