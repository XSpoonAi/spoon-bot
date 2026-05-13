"""Live integration tests requiring a running gateway.

Merged from:
  - test_integration_crypto.py (§1: crypto price queries)
  - test_more_scenarios.py     (§2: non-crypto, edge cases, tool list)
  - test_agent_real.py         (§3: real API agent test)

Run:
    python tests/test_integration_live.py [--gateway URL]

Requires gateway at http://127.0.0.1:9090 (or specify --gateway).
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

import httpx

GATEWAY = "http://127.0.0.1:9090"
ENDPOINT = f"{GATEWAY}/v1/agent/chat"
TIMEOUT = 180


async def chat(message: str, session_id: str | None = None) -> dict:
    sid = session_id or f"test-{int(time.time())}"
    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        r = await c.post(ENDPOINT, json={
            "message": message,
            "session_id": sid,
            "stream": False,
        })
        return r.json()


def extract(data: dict) -> str:
    d = data.get("data", {})
    return d.get("response", "") if isinstance(d, dict) else str(d)


# ═══════════════════════════════════════════════════════════════════
# §1  Crypto Integration
# ═══════════════════════════════════════════════════════════════════


async def test_btc_price():
    """BTC price query — numeric answer, not code."""
    print("\n━━━ §1 BTC Price ━━━")
    data = await chat("What is the BTC price now?", "test-btc-price")
    resp = extract(data)
    print(f"  Reply: {resp[:300]}")
    assert data.get("success"), f"Fail: {data}"
    assert resp, "Empty"
    assert any(c.isdigit() for c in resp), "Should contain price"
    assert "```" not in resp, "Should not contain code blocks"
    print("  ✅ PASSED")


async def test_eth_price():
    """ETH price via auto-selected provider."""
    print("\n━━━ §1 ETH Price ━━━")
    data = await chat("What is ETH trading at now?", "test-eth-price")
    resp = extract(data)
    print(f"  Reply: {resp[:300]}")
    assert data.get("success"), f"Fail: {data}"
    assert resp and any(c.isdigit() for c in resp), "Should contain price"
    print("  ✅ PASSED")


async def test_sol_price():
    """SOL price — Raydium provider."""
    print("\n━━━ §1 SOL Price ━━━")
    data = await chat("How is SOL performing lately? What is the current price?", "test-sol-price")
    resp = extract(data)
    print(f"  Reply: {resp[:300]}")
    assert data.get("success"), f"Fail: {data}"
    assert resp and any(c.isdigit() for c in resp), "Should contain price"
    print("  ✅ PASSED")


async def test_ambiguous_crypto():
    """Ambiguous crypto query with comparison."""
    print("\n━━━ §1 Ambiguous Crypto ━━━")
    data = await chat("What is the difference between WBTC and BTC? How far apart are their prices?", "test-ambiguous")
    resp = extract(data)
    print(f"  Reply: {resp[:400]}")
    assert data.get("success"), f"Fail: {data}"
    assert resp, "Empty"
    print("  ✅ PASSED")


# ═══════════════════════════════════════════════════════════════════
# §2  Non-crypto & Edge Cases
# ═══════════════════════════════════════════════════════════════════


async def test_coding_question():
    """Non-crypto coding question should NOT activate crypto tools."""
    print("\n━━━ §2 Coding Question ━━━")
    data = await chat("How do I read a JSON file in Python?", "test-coding-01")
    resp = extract(data)
    print(f"  Reply: {resp[:400]}")
    assert data.get("success"), f"Fail: {data}"
    assert resp, "Empty"
    lower = resp.lower()
    assert any(kw in lower for kw in ["json", "open", "import", "load"]), \
        "Should contain Python JSON content"
    print("  ✅ PASSED")


async def test_single_word_token():
    """Single token 'ETH' should trigger crypto tools."""
    print("\n━━━ §2 Single Token Symbol ━━━")
    data = await chat("ETH", "test-single-token-01")
    resp = extract(data)
    print(f"  Reply: {resp[:400]}")
    assert data.get("success") and resp, f"Fail: {data}"
    print("  ✅ PASSED")


async def test_crypto_symbol_slang():
    """Lowercase crypto shorthand should still trigger tool activation."""
    print("\n━━━ §2 Crypto Symbol Slang ━━━")
    data = await chat("btc price now?", "test-slang-01")
    resp = extract(data)
    print(f"  Reply: {resp[:400]}")
    assert data.get("success") and resp, f"Fail: {data}"
    assert any(c.isdigit() for c in resp), "Should contain a price"
    print("  ✅ PASSED")


async def test_follow_up_in_session():
    """Follow-up in same session maintains context."""
    print("\n━━━ §2 Follow-up Session ━━━")
    sid = "test-followup-01"
    data1 = await chat("What is the SOL price?", sid)
    assert data1.get("success"), f"Q1 Fail: {data1}"
    data2 = await chat("What about ETH?", sid)
    resp2 = extract(data2)
    print(f"  Q2 Reply: {resp2[:200]}")
    assert data2.get("success"), f"Q2 Fail: {data2}"
    assert "eth" in resp2.lower() or any(c.isdigit() for c in resp2), \
        "Follow-up should mention ETH"
    print("  ✅ PASSED")


async def test_math_question():
    """Math question should NOT activate specialized tools."""
    print("\n━━━ §2 Math Question ━━━")
    data = await chat("What is 1+1?", "test-math-01")
    resp = extract(data)
    print(f"  Reply: {resp[:200]}")
    assert data.get("success"), f"Fail: {data}"
    assert "2" in resp
    print("  ✅ PASSED")


async def test_general_search():
    """Non-crypto query uses web_search, not crypto tools."""
    print("\n━━━ §2 General Search ━━━")
    data = await chat("How is the weather today?", "test-general-search")
    resp = extract(data)
    print(f"  Reply: {resp[:300]}")
    assert data.get("success") and resp, f"Fail: {data}"
    print("  ✅ PASSED")


async def test_tools_list():
    """GET /v1/tools returns non-empty tool list."""
    print("\n━━━ §2 Tools API ━━━")
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{GATEWAY}/v1/tools")
        data = r.json()
        assert r.status_code == 200 and len(data) > 0
        print(f"  {len(data)} tools, sample: {data[0].get('name', '?')}")
    print("  ✅ PASSED")


# ═══════════════════════════════════════════════════════════════════
# §3  Real Agent API (no gateway needed)
# ═══════════════════════════════════════════════════════════════════


async def test_agent_real():
    """Direct agent API test (requires LLM API key)."""
    print("\n━━━ §3 Real Agent API ━━━")
    try:
        from spoon_bot.agent.loop import create_agent
    except ImportError:
        print("  ⏭ Skipped (import error)")
        return

    if not any(os.environ.get(k) for k in [
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
        "OPENROUTER_API_KEY", "GEMINI_API_KEY",
    ]):
        print("  ⏭ Skipped (no API key)")
        return

    agent = await create_agent(workspace="./workspace")
    print(f"  Model: {agent.model}, Tools: {len(agent.tools.list_tools())}")

    r = await agent.process("Say hello and list your available tools.")
    print(f"  Reply: {r[:200]}")
    assert r, "Empty response"
    print("  ✅ PASSED")


# ═══════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════


async def main():
    global GATEWAY, ENDPOINT
    if "--gateway" in sys.argv:
        idx = sys.argv.index("--gateway")
        GATEWAY = sys.argv[idx + 1]
        ENDPOINT = f"{GATEWAY}/v1/agent/chat"

    # Pre-check
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{GATEWAY}/health")
            if r.status_code != 200:
                print(f"Gateway not healthy: {r.status_code}")
                sys.exit(1)
    except Exception as e:
        print(f"Cannot reach gateway: {e}")
        sys.exit(1)

    print(f"Gateway healthy at {GATEWAY}")

    tests = [
        # §1 Crypto
        test_btc_price, test_eth_price, test_sol_price, test_ambiguous_crypto,
        # §2 Non-crypto & edge cases
        test_tools_list, test_math_question, test_coding_question,
        test_single_word_token, test_crypto_symbol_slang,
        test_follow_up_in_session, test_general_search,
        # §3 Real agent
        test_agent_real,
    ]

    passed = failed = 0
    errors: list[str] = []
    for fn in tests:
        try:
            await fn()
            passed += 1
        except AssertionError as e:
            failed += 1
            errors.append(f"  FAIL: {fn.__name__}: {e}")
            print(f"  ❌ FAILED: {e}")
        except Exception as e:
            failed += 1
            errors.append(f"  ERROR: {fn.__name__}: {e}")
            print(f"  ❌ ERROR: {e}")

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    for e in errors:
        print(e)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
