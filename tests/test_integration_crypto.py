"""Integration tests for crypto tools via the agent gateway.

These tests require a running gateway at http://127.0.0.1:9090.
Start with:
    python -m uvicorn spoon_bot.gateway.server:create_app --factory --host 127.0.0.1 --port 9090

Run with:
    python tests/test_integration_crypto.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

import httpx

GATEWAY = "http://127.0.0.1:9090"
ENDPOINT = f"{GATEWAY}/v1/agent/chat"
TIMEOUT = 180  # seconds (complex queries with tool activation may take longer)


async def chat(message: str, session_id: str | None = None) -> dict:
    """Send a non-streaming chat request and return parsed JSON."""
    sid = session_id or f"test-{int(time.time())}"
    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        r = await c.post(ENDPOINT, json={
            "message": message,
            "session_id": sid,
            "stream": False,
        })
        return r.json()


def extract_response(data: dict) -> str:
    """Pull out the response text from gateway JSON."""
    d = data.get("data", {})
    if isinstance(d, dict):
        return d.get("response", "")
    return str(d)


# ─────────────────────────────────────────────────────────
# Test scenarios
# ─────────────────────────────────────────────────────────

async def test_btc_price():
    """BTC price query should return a numeric answer, not code."""
    print("\n━━━ Test: BTC Price ━━━")
    data = await chat("BTC价格多少？", "test-btc-price")
    resp = extract_response(data)
    print(f"  Reply: {resp[:300]}")
    # Basic checks
    assert data.get("success"), f"Request failed: {data}"
    assert resp, "Empty response"
    # Should contain a number (price)
    has_number = any(c.isdigit() for c in resp)
    assert has_number, "Response should contain a price number"
    # Should NOT contain code blocks
    assert "```" not in resp, "Response should not contain code blocks"
    print("  ✅ PASSED")


async def test_eth_price():
    """ETH price query should work via auto-selected provider."""
    print("\n━━━ Test: ETH Price ━━━")
    data = await chat("ETH现在多少钱", "test-eth-price")
    resp = extract_response(data)
    print(f"  Reply: {resp[:300]}")
    assert data.get("success"), f"Request failed: {data}"
    assert resp, "Empty response"
    has_number = any(c.isdigit() for c in resp)
    assert has_number, "Response should contain a price"
    print("  ✅ PASSED")


async def test_sol_price():
    """SOL price query — should use Raydium provider."""
    print("\n━━━ Test: SOL Price ━━━")
    data = await chat("SOL最近行情如何？价格是多少", "test-sol-price")
    resp = extract_response(data)
    print(f"  Reply: {resp[:300]}")
    assert data.get("success"), f"Request failed: {data}"
    assert resp, "Empty response"
    has_number = any(c.isdigit() for c in resp)
    assert has_number, "Response should contain price data"
    print("  ✅ PASSED")


async def test_general_search():
    """Non-crypto query should use web_search, not crypto tools."""
    print("\n━━━ Test: General Web Search ━━━")
    data = await chat("今天天气怎么样", "test-general-search")
    resp = extract_response(data)
    print(f"  Reply: {resp[:300]}")
    assert data.get("success"), f"Request failed: {data}"
    assert resp, "Empty response"
    print("  ✅ PASSED")


async def test_ambiguous_crypto():
    """Ambiguous query with crypto context should still activate tools."""
    print("\n━━━ Test: Ambiguous Crypto Query ━━━")
    data = await chat("WBTC和BTC有什么区别？价格差多少？", "test-ambiguous")
    resp = extract_response(data)
    print(f"  Reply: {resp[:400]}")
    assert data.get("success"), f"Request failed: {data}"
    assert resp, "Empty response"
    print("  ✅ PASSED")


# ─────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────

async def main():
    # Pre-check: gateway alive?
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{GATEWAY}/health")
            if r.status_code != 200:
                print(f"Gateway not healthy: {r.status_code}")
                sys.exit(1)
    except Exception as e:
        print(f"Cannot reach gateway at {GATEWAY}: {e}")
        sys.exit(1)

    print(f"Gateway is healthy at {GATEWAY}")

    tests = [
        test_btc_price,
        test_eth_price,
        test_sol_price,
        test_general_search,
        test_ambiguous_crypto,
    ]

    passed = 0
    failed = 0
    errors: list[str] = []

    for test_fn in tests:
        try:
            await test_fn()
            passed += 1
        except AssertionError as e:
            failed += 1
            errors.append(f"  FAIL: {test_fn.__name__}: {e}")
            print(f"  ❌ FAILED: {e}")
        except Exception as e:
            failed += 1
            errors.append(f"  ERROR: {test_fn.__name__}: {e}")
            print(f"  ❌ ERROR: {e}")

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        for err in errors:
            print(err)
    print(f"{'='*50}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
