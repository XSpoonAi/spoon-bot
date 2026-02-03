"""
Real API test for spoon-bot.

Run with: ANTHROPIC_API_KEY=your_key python tests/test_agent_real.py
"""

import asyncio
import os
import sys

# Ensure the package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    """Run real API tests."""
    from spoon_bot.agent.loop import create_agent

    print("=== spoon-bot Real API Test ===\n")

    # Create agent
    print("Creating agent...")
    agent = await create_agent(workspace="./workspace")
    print(f"✓ Agent created with model: {agent.model}")
    print(f"✓ Tools available: {agent.tools.list_tools()}")
    print(f"✓ Skills loaded: {agent.skills.list()}")

    # Test 1: Simple greeting
    print("\n--- Test 1: Simple greeting ---")
    response = await agent.process("Say hello and tell me what tools you have available.")
    print(f"Response:\n{response}\n")

    # Test 2: Directory listing
    print("\n--- Test 2: List directory ---")
    response = await agent.process("Use the list_dir tool to show me what's in the current directory.")
    print(f"Response:\n{response}\n")

    # Test 3: Shell command
    print("\n--- Test 3: Shell command ---")
    response = await agent.process("Run 'echo Hello from spoon-bot!' using the shell tool.")
    print(f"Response:\n{response}\n")

    # Test 4: File operations
    print("\n--- Test 4: File operations ---")
    response = await agent.process("Read the pyproject.toml file and tell me the project version.")
    print(f"Response:\n{response}\n")

    # Test 5: Memory
    print("\n--- Test 5: Memory ---")
    response = await agent.process("Remember that my favorite color is blue.")
    print(f"Response:\n{response}\n")

    print("\n=== All tests completed! ===")


if __name__ == "__main__":
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        print("\nUsage:")
        print("  export ANTHROPIC_API_KEY=your_key")
        print("  python tests/test_agent_real.py")
        sys.exit(1)

    asyncio.run(main())
