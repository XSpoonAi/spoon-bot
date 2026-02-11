"""Test script: use spoon-bot agent to create a 2048 game."""
import asyncio
import os
import sys
from pathlib import Path

# Load .env with override to replace system env vars
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from spoon_bot.agent.loop import create_agent


async def main():
    api_key = os.environ.get("OPENAI_API_KEY", "").strip("'\"")
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip("'\"")
    model = os.environ.get("SPOON_BOT_DEFAULT_MODEL", "gpt-5.3-codex")
    provider = os.environ.get("SPOON_BOT_DEFAULT_PROVIDER", "openai")

    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Base URL: {base_url}")
    print(f"API Key: {api_key[:10]}...{api_key[-4:]}")
    print()

    workspace = Path(__file__).parent / "test_workspace"
    workspace.mkdir(exist_ok=True)

    print("Creating agent (core tools only)...")
    agent = await create_agent(
        model=model,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        workspace=str(workspace),
        enable_skills=False,
        max_iterations=30,
        auto_commit=False,
    )

    # Show tool loading info
    all_tools = agent.get_available_tools()
    active = [t for t in all_tools if t["active"]]
    inactive = [t for t in all_tools if not t["active"]]
    print(f"Active tools ({len(active)}): {[t['name'] for t in active]}")
    print(f"Available for loading ({len(inactive)}): {[t['name'] for t in inactive]}")
    print()

    prompt = """请用 HTML + CSS + JavaScript 实现一个完整的 2048 游戏，要求：
1. 单个 HTML 文件（内嵌 CSS 和 JS）
2. 4x4 网格，支持键盘方向键操作
3. 数字方块有不同颜色
4. 显示当前分数
5. 游戏结束检测
6. 保存为 2048.html

请直接使用工具创建文件。"""

    print("Sending prompt to agent...")
    print("=" * 60)
    response = await agent.process(prompt)
    print("=" * 60)
    print(f"\nAgent response:\n{response}")

    # Check if file was created
    game_file = workspace / "2048.html"
    if game_file.exists():
        size = game_file.stat().st_size
        print(f"\n✓ 2048.html created successfully! ({size} bytes)")
    else:
        print("\n✗ 2048.html was not created")
        # Check what files exist
        for f in workspace.iterdir():
            if f.is_file():
                print(f"  Found: {f.name} ({f.stat().st_size} bytes)")


if __name__ == "__main__":
    asyncio.run(main())
