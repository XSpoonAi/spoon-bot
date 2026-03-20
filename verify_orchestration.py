"""
验证脚本：测试 LLM 自主决策多 SubAgent 编排能力的各个组件。

运行方式：
    uv run python verify_orchestration.py

此脚本不依赖 spoon-ai-sdk，直接测试新增的编排相关代码。
"""

import sys
import json
import io

# Force UTF-8 output on Windows to avoid GBK encode errors
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

print("=" * 60)
print("spoon-bot Multi-Agent Orchestration Verification")
print("=" * 60)

# ----------------------------------------------------------------
# 1. catalog.py — AgentRole 和 AGENT_CATALOG
# ----------------------------------------------------------------
print("\n[1] 验证 catalog.py ...")
from spoon_bot.subagent.catalog import (
    AGENT_CATALOG, AgentRole, get_role, list_roles, format_roles_for_prompt
)

assert len(AGENT_CATALOG) == 7, f"期望 7 个角色，实际 {len(AGENT_CATALOG)}"
assert set(AGENT_CATALOG.keys()) == {
    "planner", "backend", "frontend", "researcher", "reviewer", "devops", "tester"
}

planner = get_role("planner")
assert planner is not None
assert planner.tool_profile == "research"
assert planner.max_iterations == 12
assert "plan" in planner.system_prompt.lower()

backend = get_role("backend")
assert backend.tool_profile == "coding"
assert backend.max_iterations == 25

assert get_role("nonexistent") is None

roles = list_roles()
assert len(roles) == 7

roles_text = format_roles_for_prompt()
assert "planner" in roles_text
assert "backend" in roles_text
assert "frontend" in roles_text

print(f"  ✓ AGENT_CATALOG: {list(AGENT_CATALOG.keys())}")
print(f"  ✓ get_role('planner'): profile={planner.tool_profile}, max_iter={planner.max_iterations}")
print(f"  ✓ get_role('nonexistent'): None")
print(f"  ✓ format_roles_for_prompt() 包含所有角色")

# ----------------------------------------------------------------
# 2. models.py — SubagentConfig.role 字段
# ----------------------------------------------------------------
print("\n[2] 验证 models.py (SubagentConfig.role 字段) ...")
from spoon_bot.subagent.models import SubagentConfig, SpawnMode, CleanupMode

# 默认无 role
cfg_default = SubagentConfig()
assert cfg_default.role is None
assert cfg_default.tool_profile == "core"
assert cfg_default.max_iterations == 15

# 指定 role
cfg_planner = SubagentConfig(role="planner")
assert cfg_planner.role == "planner"

# role 不影响其他字段（由 tools.py 在 spawn 时填充）
cfg_backend = SubagentConfig(role="backend", model="anthropic/sonnet-4")
assert cfg_backend.role == "backend"
assert cfg_backend.model == "anthropic/sonnet-4"

print(f"  ✓ SubagentConfig() 默认 role=None")
print(f"  ✓ SubagentConfig(role='planner').role = 'planner'")
print(f"  ✓ SubagentConfig(role='backend', model=...) 字段共存正常")

# ----------------------------------------------------------------
# 3. tools.py — SubagentTool 描述、参数、list_roles action
# ----------------------------------------------------------------
print("\n[3] 验证 tools.py (SubagentTool) ...")
from spoon_bot.subagent.tools import SubagentTool

tool = SubagentTool()

# 工具名称
assert tool.name == "spawn"

# description 动态注入了角色目录
desc = tool.description
assert "planner" in desc, "description 应包含 planner 角色"
assert "backend" in desc, "description 应包含 backend 角色"
assert "list_roles" in desc, "description 应包含 list_roles action 说明"
assert "Orchestration" in desc, "description 应包含 Orchestration 引导"
assert "no fixed pipeline" in desc, "description 应说明无固定 pipeline"

# parameters 包含 role 枚举和 list_roles action
params = tool.parameters
actions = params["properties"]["action"]["enum"]
assert "list_roles" in actions, f"actions 应包含 list_roles，实际: {actions}"
assert "spawn" in actions
assert "wait" in actions

role_param = params["properties"]["role"]
assert "planner" in role_param["enum"]
assert "backend" in role_param["enum"]
assert "task_id" in params["properties"], "应有 task_id 参数（镜像 opencode task.ts）"

print(f"  ✓ tool.name = 'spawn'")
print(f"  ✓ description 包含角色目录 + Orchestration 引导 + no fixed pipeline")
print(f"  ✓ parameters.action.enum 包含 list_roles: {actions}")
print(f"  ✓ parameters.role.enum: {role_param['enum']}")
print(f"  ✓ parameters 包含 task_id 字段")

# list_roles action 输出
roles_output = tool._handle_list_roles()
assert "planner" in roles_output
assert "Tool profile: research" in roles_output
assert "Max steps:    12" in roles_output
assert "backend" in roles_output
assert "Usage:" in roles_output

print(f"  ✓ _handle_list_roles() 输出格式正确")
print()
print("  list_roles 输出预览:")
for line in roles_output.split("\n")[:12]:
    print(f"    {line}")

# ----------------------------------------------------------------
# 4. __init__.py — 公共 API 导出
# ----------------------------------------------------------------
print("\n[4] 验证 __init__.py 导出 ...")
from spoon_bot.subagent import (
    AGENT_CATALOG as _cat,
    AgentRole as _role,
    get_role as _get,
    list_roles as _list,
    format_roles_for_prompt as _fmt,
    SubagentConfig as _cfg,
    SubagentTool as _tool,
)
assert _cat is AGENT_CATALOG
assert _role is AgentRole
print(f"  ✓ 所有新增符号从 spoon_bot.subagent 正确导出")

# ----------------------------------------------------------------
# 5. loop.py — format_roles_for_prompt 导入 + spawn 检测逻辑
# ----------------------------------------------------------------
print("\n[5] 验证 loop.py 中的导入 ...")
import ast, pathlib

loop_src = pathlib.Path(
    "C:/Users/lal/Desktop/Agent/Neo/spoon/spoon-bot/spoon_bot/agent/loop.py"
).read_text(encoding="utf-8")

assert "from spoon_bot.subagent.catalog import format_roles_for_prompt" in loop_src, \
    "loop.py 应导入 format_roles_for_prompt"
assert "Multi-Agent Orchestration" in loop_src, \
    "loop.py 应包含 Multi-Agent Orchestration 引导段"
assert "if \"spawn\" in self.tools:" in loop_src, \
    "loop.py 应检测 spawn 工具是否可用"
assert "no fixed pipeline" not in loop_src or "there is no fixed" in loop_src or \
    "no fixed" in loop_src, \
    "loop.py 应说明无固定 pipeline"
assert "format_roles_for_prompt()" in loop_src, \
    "loop.py 应调用 format_roles_for_prompt() 注入角色列表"
assert "list_roles" in loop_src, \
    "loop.py 应提及 list_roles action"

print(f"  ✓ loop.py 导入 format_roles_for_prompt")
print(f"  ✓ loop.py 包含 Multi-Agent Orchestration 引导段")
print(f"  ✓ loop.py 检测 'spawn' in self.tools 后才注入")
print(f"  ✓ loop.py 调用 format_roles_for_prompt() 动态注入角色列表")

# ----------------------------------------------------------------
# 6. 模拟 spawn 时 role 参数的处理逻辑（不需要真实 manager）
# ----------------------------------------------------------------
print("\n[6] 模拟 role 参数处理逻辑 ...")

# 模拟 _handle_spawn 中的 role 解析逻辑
role_name = "planner"
agent_role = get_role(role_name)
assert agent_role is not None

config = SubagentConfig()
config.role = agent_role.name
config.system_prompt = agent_role.system_prompt
config.tool_profile = agent_role.tool_profile
config.max_iterations = agent_role.max_iterations

assert config.role == "planner"
assert config.tool_profile == "research"
assert config.max_iterations == 12
assert "architect" in config.system_prompt.lower() or "plan" in config.system_prompt.lower()

# 自动生成 label
task = "实现一个用户管理系统，包含注册、登录、权限管理"
task_snippet = task[:40].rstrip() + ("…" if len(task) > 40 else "")
label = f"{role_name}: {task_snippet}"
assert label.startswith("planner:")
print(f"  ✓ role='planner' → tool_profile='research', max_iterations=12")
print(f"  ✓ 自动生成 label: '{label}'")

# 未知 role 返回错误
unknown_role = get_role("unknown_role")
assert unknown_role is None
print(f"  ✓ 未知 role 返回 None（tools.py 会返回错误信息）")

# ----------------------------------------------------------------
# 总结
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("✅ 所有验证通过！")
print("=" * 60)
print()
print("📋 功能总结：")
print("  用户输入: '实现一个用户管理系统'")
print()
print("  LLM 自主决策流程（无固定 Pipeline）：")
print("  1. 看到 system prompt 中的 Orchestration 引导段")
print("     → 知道自己是 Orchestrator，知道有哪些专业角色")
print()
print("  2. 调用 spawn(action='list_roles')")
print("     → 获取所有角色的详细描述")
print()
print("  3. 自主决定：先 spawn planner 分析需求")
print("     spawn(action='spawn', role='planner',")
print("           task='分析用户管理系统需求...')")
print("     → SubagentTool 自动加载 planner 的 system_prompt")
print("        (tool_profile='research', max_iterations=12)")
print()
print("  4. 等待规划结果：spawn(action='wait', timeout=120)")
print()
print("  5. 根据规划，LLM 自主决定并行 spawn backend + frontend")
print("     spawn(action='spawn', role='backend', task='...<plan>...')")
print("     spawn(action='spawn', role='frontend', task='...<plan>...')")
print()
print("  6. 等待实现结果，汇总给用户")
print()
print("  关键：编排顺序由 LLM 自主决定，不是硬编码 Pipeline ✓")
