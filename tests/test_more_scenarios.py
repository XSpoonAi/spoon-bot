"""Additional integration test scenarios for the agent.

Covers: non-crypto queries, edge cases, tool category selection.

Run: python tests/test_more_scenarios.py
Requires: gateway at http://127.0.0.1:9090
"""

from __future__ import annotations

import asyncio
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


# ─────────────────────────────────────────────────────────
# Scenarios
# ─────────────────────────────────────────────────────────

async def test_coding_question():
    """Non-crypto coding question should NOT activate crypto tools."""
    print("\n━━━ Test: Coding Question ━━━")
    data = await chat("Python如何读取JSON文件？", "test-coding-01")
    resp = extract(data)
    print(f"  Reply: {resp[:400]}")
    assert data.get("success"), f"Fail: {data}"
    assert resp, "Empty"
    # Should mention json/open/load
    lower = resp.lower()
    assert any(kw in lower for kw in ["json", "open", "import", "load"]), \
        "Should contain Python JSON-related content"
    print("  ✅ PASSED")


async def test_single_word_token():
    """Single token symbol 'ETH' should trigger crypto tools."""
    print("\n━━━ Test: Single Token Symbol ━━━")
    data = await chat("ETH", "test-single-token-01")
    resp = extract(data)
    print(f"  Reply: {resp[:400]}")
    assert data.get("success"), f"Fail: {data}"
    assert resp, "Empty"
    # Should either give price or ask clarifying questions
    print("  ✅ PASSED")


async def test_chinese_crypto_slang():
    """Chinese crypto slang '大饼' (BTC) should trigger tool activation."""
    print("\n━━━ Test: Chinese Crypto Slang ━━━")
    data = await chat("大饼现在什么价", "test-slang-01")
    resp = extract(data)
    print(f"  Reply: {resp[:400]}")
    assert data.get("success"), f"Fail: {data}"
    assert resp, "Empty"
    # Should recognize BTC/大饼 and give price
    has_number = any(c.isdigit() for c in resp)
    assert has_number, "Should contain a price number"
    print("  ✅ PASSED")


async def test_follow_up_in_session():
    """Follow-up question in same session should maintain context."""
    print("\n━━━ Test: Follow-up in Session ━━━")
    sid = "test-followup-01"
    data1 = await chat("SOL多少钱", sid)
    resp1 = extract(data1)
    print(f"  Q1 Reply: {resp1[:200]}")
    assert data1.get("success"), f"Q1 Fail: {data1}"

    data2 = await chat("那ETH呢", sid)
    resp2 = extract(data2)
    print(f"  Q2 Reply: {resp2[:200]}")
    assert data2.get("success"), f"Q2 Fail: {data2}"
    # Should mention ETH price
    assert "eth" in resp2.lower() or any(c.isdigit() for c in resp2), \
        "Follow-up should contain ETH info"
    print("  ✅ PASSED")


async def test_math_question():
    """Math question should NOT activate any specialized tools."""
    print("\n━━━ Test: Math Question ━━━")
    data = await chat("1+1等于多少", "test-math-01")
    resp = extract(data)
    print(f"  Reply: {resp[:200]}")
    assert data.get("success"), f"Fail: {data}"
    assert "2" in resp, "Should answer 2"
    print("  ✅ PASSED")


async def test_tools_list():
    """GET /v1/tools should list all tools with status."""
    print("\n━━━ Test: Tools API ━━━")
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{GATEWAY}/v1/tools")
        data = r.json()
        print(f"  Status: {r.status_code}, tools count: {len(data)}")
        assert r.status_code == 200
        assert len(data) > 0, "Should have tools"
        # Check structure
        sample = data[0]
        assert "name" in sample, "Tool should have name"
        print(f"  Sample tool: {sample.get('name')}")
    print("  ✅ PASSED")


# ─────────────────────────────────────────────────────────

async def main():
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
        test_tools_list,
        test_math_question,
        test_coding_question,
        test_single_word_token,
        test_chinese_crypto_slang,
        test_follow_up_in_session,
    ]

    passed = 0
    failed = 0
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
    if errors:
        for err in errors:
            print(err)
    print(f"{'='*50}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
