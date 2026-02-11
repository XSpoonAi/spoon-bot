"""
Comprehensive test for dynamic tool loading architecture.

Tests:
1. ToolRegistry: CORE_TOOLS, filtering, activate/deactivate, caching
2. AgentLoop: core-only default, add_tool, remove_tool, get_available_tools
3. Integration: end-to-end agent with dynamic tool injection
"""
import asyncio
import os
import sys
import time
from pathlib import Path

# Load .env with override
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)

sys.path.insert(0, str(Path(__file__).parent))

from spoon_bot.agent.tools.registry import CORE_TOOLS, TOOL_PROFILES, ToolRegistry
from spoon_bot.agent.tools.base import Tool, ToolParameterSchema

# ──────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        msg = f"  [FAIL] {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)


class DummyTool(Tool):
    """Minimal tool for testing."""

    def __init__(self, tool_name: str, tool_desc: str = "dummy"):
        self._name = tool_name
        self._desc = tool_desc

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._desc

    @property
    def parameters(self) -> ToolParameterSchema:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> str:
        return f"executed:{self._name}"


# ──────────────────────────────────────────────────
# Test 1: ToolRegistry unit tests
# ──────────────────────────────────────────────────


def test_registry():
    print("\n=== Test 1: ToolRegistry ===\n")

    # 1a: CORE_TOOLS constant
    check(
        "CORE_TOOLS has 5 tools",
        len(CORE_TOOLS) == 5,
        f"got {len(CORE_TOOLS)}: {CORE_TOOLS}",
    )
    for name in ("shell", "read_file", "write_file", "edit_file", "list_dir"):
        check(f"CORE_TOOLS contains '{name}'", name in CORE_TOOLS)

    # 1b: TOOL_PROFILES
    check("'core' profile exists", "core" in TOOL_PROFILES)
    check("'core' profile == CORE_TOOLS", TOOL_PROFILES["core"] == CORE_TOOLS)
    check("'full' profile exists", "full" in TOOL_PROFILES)
    check("'coding' profile exists", "coding" in TOOL_PROFILES)
    check("'web3' profile exists", "web3" in TOOL_PROFILES)
    check("'research' profile exists", "research" in TOOL_PROFILES)

    # 1c: Registry filtering
    reg = ToolRegistry()
    for n in ("shell", "read_file", "write_file", "edit_file", "list_dir",
              "web_search", "web_fetch", "balance_check"):
        reg.register(DummyTool(n, f"Tool: {n}"))

    check("All 8 tools registered", len(reg._tools) == 8)
    check("No filter => all active", len(reg.get_active_tools()) == 8)

    # Apply core filter
    reg.set_tool_filter(enabled_tools=set(CORE_TOOLS))
    check("Core filter => 5 active", len(reg.get_active_tools()) == 5)
    check("Core filter => 3 inactive", len(reg.get_inactive_tools()) == 3)
    check(
        "list_tools returns 5",
        len(reg.list_tools()) == 5,
        f"got {reg.list_tools()}",
    )

    # 1d: activate / deactivate
    activated = reg.activate_tool("web_search")
    check("activate_tool('web_search') returns True", activated)
    check("web_search now active", "web_search" in reg.get_active_tools())
    check("Now 6 active", len(reg.get_active_tools()) == 6)

    deactivated = reg.deactivate_tool("web_search")
    check("deactivate_tool('web_search') returns True", deactivated)
    check("web_search now inactive", "web_search" not in reg.get_active_tools())
    check("Back to 5 active", len(reg.get_active_tools()) == 5)

    # activate unknown tool
    check(
        "activate unknown tool returns False",
        not reg.activate_tool("nonexistent_tool"),
    )

    # 1e: get_all_tool_summaries
    summaries = reg.get_all_tool_summaries()
    check("get_all_tool_summaries returns 8 items", len(summaries) == 8)
    active_summaries = [s for s in summaries if s["active"]]
    inactive_summaries = [s for s in summaries if not s["active"]]
    check("5 active in summaries", len(active_summaries) == 5)
    check("3 inactive in summaries", len(inactive_summaries) == 3)

    # 1f: Definition caching
    defs1 = reg.get_definitions()
    defs2 = reg.get_definitions()
    check("get_definitions returns 5 items", len(defs1) == 5)
    check("Cached result is same object", defs1 is defs2)

    reg.activate_tool("web_fetch")
    defs3 = reg.get_definitions()
    check("After activate, cache invalidated", defs3 is not defs2)
    check("Now 6 definitions", len(defs3) == 6)

    # 1g: execute works on ALL tools (even inactive)
    async def _test_execute():
        # balance_check is inactive but should still execute
        result = await reg.execute("balance_check", {})
        return result

    result = asyncio.get_event_loop().run_until_complete(_test_execute())
    check(
        "execute() works on inactive tool",
        "executed:balance_check" in result,
        f"got: {result}",
    )

    # 1h: clear_tool_filter
    reg.clear_tool_filter()
    check("After clear_tool_filter => all active", len(reg.get_active_tools()) == 8)


# ──────────────────────────────────────────────────
# Test 2: AgentLoop unit tests (no LLM needed)
# ──────────────────────────────────────────────────


def test_agent_loop_tools():
    print("\n=== Test 2: AgentLoop tool management ===\n")

    from spoon_bot.agent.loop import AgentLoop

    workspace = Path(__file__).parent / "test_workspace"
    workspace.mkdir(exist_ok=True)

    agent = AgentLoop(
        workspace=str(workspace),
        model="test-model",
        provider="openai",
        api_key="test-key",
        auto_commit=False,
        enable_skills=False,
    )

    # 2a: Default is core-only
    active = agent.tools.get_active_tools()
    total = agent.tools._tools
    check(
        "Default: 5 core tools active",
        len(active) == 5,
        f"got {len(active)}: {list(active.keys())}",
    )
    check(
        "Total registered > 5",
        len(total) > 5,
        f"got {len(total)}",
    )

    # 2b: Core tools are the right ones
    for name in CORE_TOOLS:
        check(f"Core tool '{name}' is active", name in active)

    # 2c: Non-core tools are inactive
    inactive = agent.tools.get_inactive_tools()
    check(
        "Inactive tools exist",
        len(inactive) > 0,
        f"got {len(inactive)}",
    )
    for name in ("web_search", "web_fetch", "balance_check"):
        if name in total:
            check(f"'{name}' is inactive by default", name in inactive)

    # 2d: get_available_tools
    all_tools = agent.get_available_tools()
    check(
        "get_available_tools returns all tools",
        len(all_tools) == len(total),
        f"got {len(all_tools)} vs {len(total)}",
    )
    active_list = [t for t in all_tools if t["active"]]
    inactive_list = [t for t in all_tools if not t["active"]]
    check("active list has 5", len(active_list) == 5)
    check(
        "inactive list matches",
        len(inactive_list) == len(total) - 5,
    )

    # 2e: add_tool
    added = agent.add_tool("web_search")
    check("add_tool('web_search') returns True", added)
    check(
        "web_search now active",
        "web_search" in agent.tools.get_active_tools(),
    )
    check(
        "Now 6 active tools",
        len(agent.tools.get_active_tools()) == 6,
    )

    # add_tool duplicate
    dup = agent.add_tool("web_search")
    check("add_tool duplicate returns False", not dup)

    # add_tool unknown
    unknown = agent.add_tool("nonexistent_xyz")
    check("add_tool unknown returns False", not unknown)

    # 2f: add_tools (multiple)
    activated = agent.add_tools("web_fetch", "balance_check")
    check(
        "add_tools returns activated names",
        set(activated) == {"web_fetch", "balance_check"},
        f"got {activated}",
    )
    check(
        "Now 8 active tools",
        len(agent.tools.get_active_tools()) == 8,
    )

    # 2g: remove_tool
    removed = agent.remove_tool("web_search")
    check("remove_tool returns True", removed)
    check(
        "web_search no longer active",
        "web_search" not in agent.tools.get_active_tools(),
    )

    # remove_tool on already inactive
    removed_again = agent.remove_tool("web_search")
    check("remove_tool on inactive returns False", not removed_again)


# ──────────────────────────────────────────────────
# Test 3: AgentLoop with tool_profile
# ──────────────────────────────────────────────────


def test_tool_profiles():
    print("\n=== Test 3: Tool profiles ===\n")

    from spoon_bot.agent.loop import AgentLoop

    workspace = Path(__file__).parent / "test_workspace"
    workspace.mkdir(exist_ok=True)

    # Full profile
    agent_full = AgentLoop(
        workspace=str(workspace),
        model="test-model",
        provider="openai",
        api_key="test-key",
        auto_commit=False,
        enable_skills=False,
        tool_profile="full",
    )
    full_active = agent_full.tools.get_active_tools()
    check(
        "tool_profile='full' loads all tools",
        len(full_active) == len(agent_full.tools._tools),
        f"active={len(full_active)}, total={len(agent_full.tools._tools)}",
    )

    # Coding profile
    agent_coding = AgentLoop(
        workspace=str(workspace),
        model="test-model",
        provider="openai",
        api_key="test-key",
        auto_commit=False,
        enable_skills=False,
        tool_profile="coding",
    )
    coding_active = agent_coding.tools.get_active_tools()
    expected_coding = TOOL_PROFILES["coding"]
    # Only count tools that are both in the profile AND registered
    registered_coding = {n for n in expected_coding if n in agent_coding.tools._tools}
    check(
        f"tool_profile='coding' loads {len(registered_coding)} tools",
        len(coding_active) == len(registered_coding),
        f"active={len(coding_active)}, expected={len(registered_coding)}",
    )

    # Explicit enabled_tools
    agent_custom = AgentLoop(
        workspace=str(workspace),
        model="test-model",
        provider="openai",
        api_key="test-key",
        auto_commit=False,
        enable_skills=False,
        enabled_tools={"shell", "web_search"},
    )
    custom_active = agent_custom.tools.get_active_tools()
    check(
        "enabled_tools={'shell','web_search'} loads 2",
        len(custom_active) == 2,
        f"got {list(custom_active.keys())}",
    )


# ──────────────────────────────────────────────────
# Test 4: End-to-end with LLM (optional, needs API)
# ──────────────────────────────────────────────────


async def test_e2e():
    print("\n=== Test 4: End-to-end agent execution ===\n")

    api_key = os.environ.get("OPENAI_API_KEY", "").strip("'\"")
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip("'\"")
    model = os.environ.get("SPOON_BOT_DEFAULT_MODEL", "gpt-5.3-codex")
    provider = os.environ.get("SPOON_BOT_DEFAULT_PROVIDER", "openai")

    if not api_key or api_key.startswith("sk-placeholder"):
        print("  [SKIP] No API key configured, skipping E2E test")
        return

    from spoon_bot.agent.loop import create_agent

    workspace = Path(__file__).parent / "test_workspace"
    workspace.mkdir(exist_ok=True)

    print(f"  Provider: {provider}, Model: {model}")
    print(f"  Base URL: {base_url[:40]}...")

    # 4a: Create agent with core tools
    agent = await create_agent(
        model=model,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        workspace=str(workspace),
        enable_skills=False,
        max_iterations=10,
        auto_commit=False,
    )

    active = agent.tools.get_active_tools()
    total = agent.tools._tools
    check(
        "E2E: Agent created with core tools",
        len(active) == 5,
        f"active={len(active)}/{len(total)}",
    )

    # 4b: Simple task using core tools (write_file)
    print("\n  Running write_file task...")
    t0 = time.time()
    response = await agent.process(
        "Create a file called hello.txt with the content 'Hello from dynamic tools test'. "
        "Use the write_file tool. Reply with just 'Done' when finished."
    )
    t1 = time.time()
    print(f"  Response ({t1-t0:.1f}s): {response[:120]}...")

    hello_file = workspace / "hello.txt"
    check(
        "E2E: hello.txt created by core tools",
        hello_file.exists(),
        f"exists={hello_file.exists()}",
    )
    if hello_file.exists():
        content = hello_file.read_text()
        check(
            "E2E: hello.txt has correct content",
            "Hello from dynamic tools test" in content,
            f"content={content[:60]}",
        )
        hello_file.unlink()  # cleanup

    # 4c: Dynamically add web_search tool
    added = agent.add_tool("web_search")
    check("E2E: add_tool('web_search') succeeded", added)
    check(
        "E2E: Now 6 active tools",
        len(agent.tools.get_active_tools()) == 6,
    )

    # Verify the tool was injected into agent's ToolManager
    if agent._agent and hasattr(agent._agent, "available_tools"):
        tm = agent._agent.available_tools
        check(
            "E2E: web_search in agent's ToolManager",
            "web_search" in tm.tool_map,
            f"tool_map keys: {list(tm.tool_map.keys())}",
        )

    # 4d: Remove a tool
    removed = agent.remove_tool("web_search")
    check("E2E: remove_tool('web_search') succeeded", removed)
    check(
        "E2E: Back to 5 active tools",
        len(agent.tools.get_active_tools()) == 5,
    )


# ──────────────────────────────────────────────────
# Test 5: Schema caching performance
# ──────────────────────────────────────────────────


def test_schema_caching():
    print("\n=== Test 5: Schema caching performance ===\n")

    reg = ToolRegistry()
    # Register 20 tools
    for i in range(20):
        reg.register(DummyTool(f"tool_{i}", f"Description for tool {i}"))

    # First call: cold
    t0 = time.time()
    for _ in range(1000):
        defs = reg.get_definitions()
    t1 = time.time()
    cold_time = t1 - t0

    # Second call: warm (cached)
    t2 = time.time()
    for _ in range(1000):
        defs = reg.get_definitions()
    t3 = time.time()
    warm_time = t3 - t2

    # The warm time should be significantly faster (cache hit)
    check(
        f"Caching: 1000 calls, cold={cold_time*1000:.1f}ms warm={warm_time*1000:.1f}ms",
        warm_time <= cold_time * 1.1,  # warm should be <= cold (with margin)
        f"ratio={warm_time/cold_time:.2f}",
    )
    check("get_definitions returns 20 schemas", len(defs) == 20)


# ──────────────────────────────────────────────────
# Test 6: ToolManager (spoon-core) caching
# ──────────────────────────────────────────────────


def test_tool_manager_caching():
    print("\n=== Test 6: ToolManager (spoon-core) caching ===\n")

    from spoon_ai.tools import ToolManager, BaseTool
    from pydantic import Field

    class FakeTool(BaseTool):
        name: str = "fake"
        description: str = "A fake tool"
        parameters: dict = {"type": "object", "properties": {}}

        async def execute(self, **kwargs):
            return "ok"

    tools = [FakeTool(name=f"tool_{i}", description=f"desc_{i}") for i in range(10)]
    tm = ToolManager(tools)

    # First call
    params1 = tm.to_params()
    check("to_params returns 10 items", len(params1) == 10)

    # Second call (cached)
    params2 = tm.to_params()
    check("Cached to_params returns same object", params1 is params2)

    # Add a tool - cache should invalidate
    tm.add_tool(FakeTool(name="tool_new", description="new tool"))
    params3 = tm.to_params()
    check("After add_tool, cache invalidated", params3 is not params2)
    check("Now 11 items", len(params3) == 11)

    # Remove a tool
    tm.remove_tool("tool_new")
    params4 = tm.to_params()
    check("After remove_tool, cache invalidated", params4 is not params3)
    check("Back to 10 items", len(params4) == 10)


# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────

def main():
    global passed, failed

    print("=" * 60)
    print("Dynamic Tool Loading Architecture - Test Suite")
    print("=" * 60)

    # Unit tests (no LLM needed)
    test_registry()
    test_agent_loop_tools()
    test_tool_profiles()
    test_schema_caching()
    test_tool_manager_caching()

    # E2E test (needs API key)
    asyncio.run(test_e2e())

    # Summary
    total = passed + failed
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
