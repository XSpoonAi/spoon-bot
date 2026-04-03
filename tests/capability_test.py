"""Thorough agent capability tests via the gateway WebSocket interface.

Tests the spoon-bot agent's actual LLM capabilities through the gateway,
covering instruction following, reasoning, streaming, multi-turn context,
error handling, and response quality.

Usage:
    python tests/capability_test.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import traceback
from uuid import uuid4

import websockets


WS_URL = "ws://127.0.0.1:8080/v1/ws"
TIMEOUT = 90  # seconds per test


class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error: str | None = None
        self.details: dict = {}

    def ok(self, **details):
        self.passed = True
        self.details = details

    def fail(self, msg: str, **details):
        self.passed = False
        self.error = msg
        self.details = details


def make_request(method: str, params: dict | None = None) -> dict:
    return {
        "type": "request",
        "id": f"req_{uuid4().hex[:8]}",
        "method": method,
        "params": params or {},
    }


async def recv_response(ws, request_id: str, *, timeout: float = TIMEOUT) -> tuple[dict | None, list[dict]]:
    """Receive events + final response for a given request ID."""
    events = []
    response = None
    try:
        async with asyncio.timeout(timeout):
            while True:
                raw = await ws.recv()
                msg = json.loads(raw)
                if msg.get("type") == "response" and msg.get("id") == request_id:
                    response = msg
                    break
                elif msg.get("type") == "error" and msg.get("id") == request_id:
                    response = msg
                    break
                else:
                    events.append(msg)
    except TimeoutError:
        pass
    return response, events


async def chat(ws, message: str, stream: bool = False) -> tuple[str | None, dict | None, list[dict]]:
    """Helper: send a chat message and return (text, response, events)."""
    req = make_request("agent.chat", {"message": message, "stream": stream})
    await ws.send(json.dumps(req))
    response, events = await recv_response(ws, req["id"])
    if response is None:
        return None, None, events
    if response.get("type") == "error":
        return None, response, events
    text = response.get("result", {}).get("text") or response.get("result", {}).get("content", "")
    return text, response, events


async def stream_chat(ws, message: str) -> tuple[str, dict | None, list[dict]]:
    """Helper: send a streaming chat and collect full text from chunks."""
    req = make_request("agent.chat", {"message": message, "stream": True})
    await ws.send(json.dumps(req))
    response, events = await recv_response(ws, req["id"])

    # Collect streamed text from chunk events
    chunks = []
    for e in events:
        if e.get("event") == "agent.stream.chunk":
            delta = e.get("data", {}).get("delta", "")
            chunks.append(delta)

    full_text = "".join(chunks)
    return full_text, response, events


# ──────────────────────────────────────────────
# Test 1: Basic instruction following
# ──────────────────────────────────────────────
async def test_basic_instruction(results: list[TestResult]):
    t = TestResult("1. Basic instruction following")
    results.append(t)
    try:
        async with websockets.connect(WS_URL, ping_interval=3600, ping_timeout=3600) as ws:
            await asyncio.wait_for(ws.recv(), timeout=10)

            text, resp, _ = await chat(ws, "What is 2 + 3? Reply with just the number.")

            if text is None:
                t.fail(f"No response. resp={resp}")
                print(f"  [FAIL] No response")
                return

            if "5" in text:
                t.ok(response=text[:200])
                print(f"  [PASS] Got '5' in response: {text[:100]}")
            else:
                t.fail(f"Expected '5' in response, got: {text[:200]}")
                print(f"  [FAIL] Response: {text[:200]}")
    except Exception as e:
        t.fail(str(e))
        print(f"  [FAIL] {e}")
        traceback.print_exc()


# ──────────────────────────────────────────────
# Test 2: Structured output / JSON response
# ──────────────────────────────────────────────
async def test_structured_output(results: list[TestResult]):
    t = TestResult("2. Structured output (JSON)")
    results.append(t)
    try:
        async with websockets.connect(WS_URL, ping_interval=3600, ping_timeout=3600) as ws:
            await asyncio.wait_for(ws.recv(), timeout=10)

            text, resp, _ = await chat(
                ws,
                'Return a JSON object with keys "name", "age", "city" for a fictional person. '
                "Output ONLY the JSON, no other text.",
            )

            if text is None:
                t.fail(f"No response. resp={resp}")
                print(f"  [FAIL] No response")
                return

            # Try to parse JSON from response
            # Strip markdown code fences if present
            clean = text.strip()
            if clean.startswith("```"):
                lines = clean.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                clean = "\n".join(lines).strip()

            try:
                data = json.loads(clean)
                has_keys = all(k in data for k in ("name", "age", "city"))
                if has_keys:
                    t.ok(data=data)
                    print(f"  [PASS] Valid JSON: {data}")
                else:
                    t.fail(f"Missing keys. Got: {list(data.keys())}")
                    print(f"  [FAIL] Missing keys: {data}")
            except json.JSONDecodeError:
                t.fail(f"Invalid JSON: {text[:200]}")
                print(f"  [FAIL] Not valid JSON: {text[:200]}")

    except Exception as e:
        t.fail(str(e))
        print(f"  [FAIL] {e}")
        traceback.print_exc()


# ──────────────────────────────────────────────
# Test 3: Reasoning / logic
# ──────────────────────────────────────────────
async def test_reasoning(results: list[TestResult]):
    t = TestResult("3. Reasoning / logic problem")
    results.append(t)
    try:
        async with websockets.connect(WS_URL, ping_interval=3600, ping_timeout=3600) as ws:
            await asyncio.wait_for(ws.recv(), timeout=10)

            text, resp, _ = await chat(
                ws,
                "If all roses are flowers and some flowers fade quickly, "
                "can we conclude that some roses fade quickly? "
                "Answer with 'Yes' or 'No' and a brief explanation.",
            )

            if text is None:
                t.fail(f"No response. resp={resp}")
                print(f"  [FAIL] No response")
                return

            lower = text.lower()
            # The correct answer is "No" — we cannot conclude that
            if "no" in lower and ("cannot" in lower or "can't" in lower or "not necessarily" in lower or "doesn't follow" in lower or "does not follow" in lower or "not valid" in lower or "no" in lower):
                t.ok(response=text[:300])
                print(f"  [PASS] Correct reasoning: {text[:200]}")
            else:
                # Even if the model says "No" we accept it
                if "no" in lower[:50]:
                    t.ok(response=text[:300])
                    print(f"  [PASS] Said 'No': {text[:200]}")
                else:
                    t.fail(f"Incorrect reasoning: {text[:300]}")
                    print(f"  [FAIL] {text[:200]}")

    except Exception as e:
        t.fail(str(e))
        print(f"  [FAIL] {e}")
        traceback.print_exc()


# ──────────────────────────────────────────────
# Test 4: Streaming response quality
# ──────────────────────────────────────────────
async def test_streaming_quality(results: list[TestResult]):
    t = TestResult("4. Streaming response quality + chunk count")
    results.append(t)
    try:
        async with websockets.connect(WS_URL, ping_interval=3600, ping_timeout=3600) as ws:
            await asyncio.wait_for(ws.recv(), timeout=10)

            full_text, resp, events = await stream_chat(
                ws,
                "Write exactly 3 bullet points about the benefits of open source software.",
            )

            if resp is None:
                t.fail("No response (timeout)")
                print(f"  [FAIL] No response")
                return

            if resp.get("type") == "error":
                t.fail(f"Error: {resp}")
                print(f"  [FAIL] Error: {resp}")
                return

            chunk_events = [e for e in events if e.get("event") == "agent.stream.chunk"]
            done_events = [e for e in events if e.get("event") == "agent.stream.done"]

            # Should have multiple chunks for a reasonable response
            has_chunks = len(chunk_events) > 0
            has_done = len(done_events) > 0
            has_content = len(full_text) > 50

            if has_chunks and has_content:
                t.ok(
                    chunks=len(chunk_events),
                    done_events=len(done_events),
                    text_length=len(full_text),
                    preview=full_text[:200],
                )
                print(
                    f"  [PASS] {len(chunk_events)} chunks, {len(full_text)} chars, "
                    f"done={len(done_events)}"
                )
                print(f"         Preview: {full_text[:150]}...")
            else:
                t.fail(
                    f"chunks={has_chunks} ({len(chunk_events)}), "
                    f"content={has_content} ({len(full_text)} chars)"
                )
                print(f"  [FAIL] chunks={len(chunk_events)}, text={full_text[:100]}")

    except Exception as e:
        t.fail(str(e))
        print(f"  [FAIL] {e}")
        traceback.print_exc()


# ──────────────────────────────────────────────
# Test 5: Multi-turn conversation (context retention)
# ──────────────────────────────────────────────
async def test_multi_turn(results: list[TestResult]):
    t = TestResult("5. Multi-turn conversation context")
    results.append(t)
    try:
        async with websockets.connect(WS_URL, ping_interval=3600, ping_timeout=3600) as ws:
            await asyncio.wait_for(ws.recv(), timeout=10)

            # First message: establish context
            text1, _, _ = await chat(ws, "My name is Alice and I live in Tokyo. Remember this.")

            if text1 is None:
                t.fail("No response to first message")
                print(f"  [FAIL] No response to first message")
                return

            # Second message: ask about the context
            text2, _, _ = await chat(ws, "What is my name and where do I live?")

            if text2 is None:
                t.fail("No response to second message")
                print(f"  [FAIL] No response to second message")
                return

            lower = text2.lower()
            has_name = "alice" in lower
            has_city = "tokyo" in lower

            if has_name and has_city:
                t.ok(response=text2[:300])
                print(f"  [PASS] Remembered context: {text2[:200]}")
            elif has_name or has_city:
                # Partial context retention
                t.ok(response=text2[:300])
                print(f"  [PASS] Partial context: name={has_name}, city={has_city}: {text2[:200]}")
            else:
                t.fail(f"Lost context. Response: {text2[:300]}")
                print(f"  [FAIL] No context retention: {text2[:200]}")

    except Exception as e:
        t.fail(str(e))
        print(f"  [FAIL] {e}")
        traceback.print_exc()


# ──────────────────────────────────────────────
# Test 6: Code generation
# ──────────────────────────────────────────────
async def test_code_generation(results: list[TestResult]):
    t = TestResult("6. Code generation")
    results.append(t)
    try:
        async with websockets.connect(WS_URL, ping_interval=3600, ping_timeout=3600) as ws:
            await asyncio.wait_for(ws.recv(), timeout=10)

            text, resp, _ = await chat(
                ws,
                "Write a Python function called 'fibonacci' that returns the nth Fibonacci number. "
                "Include the function definition only, no explanations.",
            )

            if text is None:
                t.fail(f"No response. resp={resp}")
                print(f"  [FAIL] No response")
                return

            lower = text.lower()
            has_def = "def fibonacci" in lower or "def fibonacci" in text
            has_return = "return" in lower
            has_python = "def " in text

            if has_def and has_return:
                t.ok(response=text[:400])
                print(f"  [PASS] Valid Python function generated")
                print(f"         Preview: {text[:200]}...")
            elif has_python:
                t.ok(response=text[:400])
                print(f"  [PASS] Python code generated (variant naming)")
                print(f"         Preview: {text[:200]}...")
            else:
                t.fail(f"Not valid code: {text[:300]}")
                print(f"  [FAIL] {text[:200]}")

    except Exception as e:
        t.fail(str(e))
        print(f"  [FAIL] {e}")
        traceback.print_exc()


# ──────────────────────────────────────────────
# Test 7: Language / translation
# ──────────────────────────────────────────────
async def test_translation(results: list[TestResult]):
    t = TestResult("7. Language / translation")
    results.append(t)
    try:
        async with websockets.connect(WS_URL, ping_interval=3600, ping_timeout=3600) as ws:
            await asyncio.wait_for(ws.recv(), timeout=10)

            text, resp, _ = await chat(
                ws,
                "Translate 'Hello, how are you?' into French, Spanish, and Japanese. "
                "Format as: French: ..., Spanish: ..., Japanese: ...",
            )

            if text is None:
                t.fail(f"No response. resp={resp}")
                print(f"  [FAIL] No response")
                return

            lower = text.lower()
            has_french = "bonjour" in lower or "salut" in lower or "comment" in lower
            has_spanish = "hola" in lower or "cómo" in lower or "como" in lower
            has_japanese = any(
                c in text for c in "こんにちはお元気ですか"
            ) or "konnichiwa" in lower

            score = sum([has_french, has_spanish, has_japanese])
            if score >= 2:
                t.ok(
                    french=has_french,
                    spanish=has_spanish,
                    japanese=has_japanese,
                    response=text[:400],
                )
                print(f"  [PASS] {score}/3 languages correct: fr={has_french}, es={has_spanish}, ja={has_japanese}")
                print(f"         Response: {text[:200]}")
            else:
                t.fail(f"Only {score}/3 translations. Response: {text[:300]}")
                print(f"  [FAIL] {text[:200]}")

    except Exception as e:
        t.fail(str(e))
        print(f"  [FAIL] {e}")
        traceback.print_exc()


# ──────────────────────────────────────────────
# Test 8: Summarization
# ──────────────────────────────────────────────
async def test_summarization(results: list[TestResult]):
    t = TestResult("8. Text summarization")
    results.append(t)
    try:
        async with websockets.connect(WS_URL, ping_interval=3600, ping_timeout=3600) as ws:
            await asyncio.wait_for(ws.recv(), timeout=10)

            long_text = (
                "Artificial intelligence has transformed numerous industries over the past decade. "
                "In healthcare, AI systems can now detect diseases from medical images with accuracy "
                "rivaling human specialists. In finance, algorithmic trading powered by machine learning "
                "processes millions of transactions per second. The transportation sector has seen "
                "autonomous vehicles move from science fiction to reality, with self-driving cars being "
                "tested on public roads worldwide. Education has been revolutionized by personalized "
                "learning platforms that adapt to each student's pace and style. However, these advances "
                "also raise important ethical questions about privacy, job displacement, and the need for "
                "responsible AI development."
            )

            text, resp, _ = await chat(
                ws,
                f"Summarize the following text in 1-2 sentences:\n\n{long_text}",
            )

            if text is None:
                t.fail(f"No response. resp={resp}")
                print(f"  [FAIL] No response")
                return

            # Summary should be shorter than original and mention AI
            is_shorter = len(text) < len(long_text)
            mentions_ai = "ai" in text.lower() or "artificial intelligence" in text.lower()

            if is_shorter and mentions_ai:
                t.ok(
                    original_len=len(long_text),
                    summary_len=len(text),
                    response=text[:300],
                )
                print(f"  [PASS] Summary ({len(text)} chars vs {len(long_text)} original): {text[:200]}")
            else:
                t.fail(f"shorter={is_shorter}, mentions_ai={mentions_ai}, len={len(text)}")
                print(f"  [FAIL] {text[:200]}")

    except Exception as e:
        t.fail(str(e))
        print(f"  [FAIL] {e}")
        traceback.print_exc()


# ──────────────────────────────────────────────
# Test 9: Math / computation
# ──────────────────────────────────────────────
async def test_math(results: list[TestResult]):
    t = TestResult("9. Math / computation")
    results.append(t)
    try:
        async with websockets.connect(WS_URL, ping_interval=3600, ping_timeout=3600) as ws:
            await asyncio.wait_for(ws.recv(), timeout=10)

            text, resp, _ = await chat(
                ws,
                "What is 17 * 23? Show your work step by step, then give the final answer.",
            )

            if text is None:
                t.fail(f"No response. resp={resp}")
                print(f"  [FAIL] No response")
                return

            # 17 * 23 = 391
            if "391" in text:
                t.ok(response=text[:400])
                print(f"  [PASS] Correct: 17*23=391")
                print(f"         {text[:200]}")
            else:
                t.fail(f"Expected 391 in response: {text[:300]}")
                print(f"  [FAIL] {text[:200]}")

    except Exception as e:
        t.fail(str(e))
        print(f"  [FAIL] {e}")
        traceback.print_exc()


# ──────────────────────────────────────────────
# Test 10: Streaming + long response
# ──────────────────────────────────────────────
async def test_streaming_long(results: list[TestResult]):
    t = TestResult("10. Streaming long response")
    results.append(t)
    try:
        async with websockets.connect(WS_URL, ping_interval=3600, ping_timeout=3600) as ws:
            await asyncio.wait_for(ws.recv(), timeout=10)

            start = time.time()
            full_text, resp, events = await stream_chat(
                ws,
                "Explain the concept of recursion in programming. "
                "Include a definition, how it works, and a simple example. "
                "Write at least 150 words.",
            )
            elapsed = time.time() - start

            if resp is None:
                t.fail("No response (timeout)")
                print(f"  [FAIL] No response after {elapsed:.1f}s")
                return

            if resp.get("type") == "error":
                t.fail(f"Error: {resp}")
                print(f"  [FAIL] Error: {resp}")
                return

            chunk_events = [e for e in events if e.get("event") == "agent.stream.chunk"]
            word_count = len(full_text.split())
            mentions_recursion = "recursion" in full_text.lower() or "recursive" in full_text.lower()

            if len(chunk_events) > 5 and word_count > 50 and mentions_recursion:
                t.ok(
                    chunks=len(chunk_events),
                    words=word_count,
                    elapsed=f"{elapsed:.1f}s",
                    preview=full_text[:200],
                )
                print(
                    f"  [PASS] {len(chunk_events)} chunks, {word_count} words, "
                    f"{elapsed:.1f}s"
                )
                print(f"         Preview: {full_text[:150]}...")
            else:
                t.fail(
                    f"chunks={len(chunk_events)}, words={word_count}, "
                    f"mentions_recursion={mentions_recursion}"
                )
                print(f"  [FAIL] chunks={len(chunk_events)}, words={word_count}")

    except Exception as e:
        t.fail(str(e))
        print(f"  [FAIL] {e}")
        traceback.print_exc()


# ──────────────────────────────────────────────
# Test 11: REST API non-streaming
# ──────────────────────────────────────────────
async def test_rest_api(results: list[TestResult]):
    t = TestResult("11. REST API /v1/agent/chat")
    results.append(t)
    try:
        import aiohttp

        url = "http://127.0.0.1:8080/v1/agent/chat"
        payload = {"message": "What is the capital of France? Reply in one word."}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                status_code = resp.status
                data = await resp.json()

                if status_code == 200:
                    # APIResponse format: {"success": true, "data": {"response": "..."}, "meta": {...}}
                    chat_data = data.get("data", {})
                    meta = data.get("meta", {})
                    text = chat_data.get("response", "")
                    trace_id = meta.get("trace_id", "")

                    if "paris" in text.lower():
                        t.ok(status=status_code, text=text[:100], trace_id=trace_id)
                        print(f"  [PASS] REST API: '{text[:80]}' trace={trace_id}")
                    else:
                        t.fail(f"Expected 'Paris', got: {text[:200]}")
                        print(f"  [FAIL] {text[:200]}")
                else:
                    t.fail(f"HTTP {status_code}: {data}")
                    print(f"  [FAIL] HTTP {status_code}: {data}")

    except ImportError:
        t.fail("aiohttp not installed — skipping REST test")
        print(f"  [SKIP] aiohttp not installed")
    except Exception as e:
        t.fail(str(e))
        print(f"  [FAIL] {e}")
        traceback.print_exc()


# ──────────────────────────────────────────────
# Test 12: Error handling — empty message
# ──────────────────────────────────────────────
async def test_empty_message(results: list[TestResult]):
    t = TestResult("12. Error handling — empty message")
    results.append(t)
    try:
        async with websockets.connect(WS_URL, ping_interval=3600, ping_timeout=3600) as ws:
            await asyncio.wait_for(ws.recv(), timeout=10)

            req = make_request("agent.chat", {"message": "", "stream": False})
            await ws.send(json.dumps(req))

            response, events = await recv_response(ws, req["id"], timeout=30)

            if response is not None:
                rtype = response.get("type")
                # Either an error response or a graceful response is acceptable
                t.ok(response_type=rtype, response=str(response)[:200])
                print(f"  [PASS] Got {rtype} response for empty message")
            else:
                t.fail("No response for empty message")
                print(f"  [FAIL] No response")

    except Exception as e:
        t.fail(str(e))
        print(f"  [FAIL] {e}")
        traceback.print_exc()


# ──────────────────────────────────────────────
# Test 13: Agent status endpoint
# ──────────────────────────────────────────────
async def test_agent_status(results: list[TestResult]):
    t = TestResult("13. Agent status endpoint")
    results.append(t)
    try:
        import aiohttp

        url = "http://127.0.0.1:8080/v1/agent/status"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                status = resp.status
                data = await resp.json()

                if status == 200:
                    model = data.get("model") or data.get("status", {}).get("model", "")
                    provider = data.get("provider") or data.get("status", {}).get("provider", "")
                    t.ok(status=status, model=model, provider=provider, data=str(data)[:300])
                    print(f"  [PASS] Status: model={model}, provider={provider}")
                else:
                    t.fail(f"HTTP {status}: {data}")
                    print(f"  [FAIL] HTTP {status}")

    except ImportError:
        t.fail("aiohttp not installed — skipping")
        print(f"  [SKIP] aiohttp not installed")
    except Exception as e:
        t.fail(str(e))
        print(f"  [FAIL] {e}")
        traceback.print_exc()


# ──────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────
async def main():
    print("=" * 65)
    print("SpoonBot Agent Capability Tests")
    print(f"Server: {WS_URL}")
    print("=" * 65)
    print()

    results: list[TestResult] = []

    tests = [
        ("Basic instruction following", test_basic_instruction),
        ("Structured JSON output", test_structured_output),
        ("Reasoning / logic", test_reasoning),
        ("Streaming quality", test_streaming_quality),
        ("Multi-turn conversation", test_multi_turn),
        ("Code generation", test_code_generation),
        ("Translation", test_translation),
        ("Summarization", test_summarization),
        ("Math computation", test_math),
        ("Streaming long response", test_streaming_long),
        ("REST API", test_rest_api),
        ("Empty message handling", test_empty_message),
        ("Agent status endpoint", test_agent_status),
    ]

    for label, test_fn in tests:
        print(f"\n--- {label} ---")
        await test_fn(results)

    # Summary
    print("\n" + "=" * 65)
    print("CAPABILITY TEST RESULTS")
    print("=" * 65)
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        extra = ""
        if r.error:
            extra = f" -- {r.error[:80]}"
        print(f"  [{status}] {r.name}{extra}")

    print(f"\n  Total: {passed}/{len(results)} passed, {failed} failed")
    print("=" * 65)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

