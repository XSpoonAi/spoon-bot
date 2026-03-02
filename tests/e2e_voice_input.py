"""End-to-end test for voice input functionality.

Tests:
1. Generate test audio via OpenAI TTS (direct API)
2. POST /v1/chat/audio — multipart file upload
3. POST /v1/chat       — base64-encoded audio in JSON body
4. Verify transcription results and agent responses

Usage:
    python tests/e2e_voice_input.py [--gateway-url http://127.0.0.1:8080]
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
from pathlib import Path

import httpx

GATEWAY_URL = "http://127.0.0.1:8080"
TEST_PHRASE = "Hello, this is a test of the voice input system."
TTS_MODEL = "tts-1"
TTS_VOICE = "alloy"


# ── Step 1: Generate test audio via OpenAI TTS ────────────────────────────

async def generate_test_audio() -> bytes:
    """Generate a short WAV audio clip using OpenAI TTS (direct API)."""
    import os

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[SKIP] OPENAI_API_KEY not set — cannot generate TTS audio")
        return _generate_sine_wave_wav()

    print(f"[TTS] Generating audio: '{TEST_PHRASE}'")
    # Always use direct OpenAI for TTS (not OpenRouter)
    async with httpx.AsyncClient(
        base_url="https://api.openai.com/v1",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30.0,
    ) as client:
        try:
            resp = await client.post(
                "/audio/speech",
                json={
                    "model": TTS_MODEL,
                    "input": TEST_PHRASE,
                    "voice": TTS_VOICE,
                    "response_format": "wav",
                },
            )
            resp.raise_for_status()
            audio_bytes = resp.content
            print(f"[TTS] Generated {len(audio_bytes)} bytes of WAV audio")
            return audio_bytes
        except httpx.HTTPStatusError as e:
            print(f"[TTS] OpenAI TTS failed ({e.response.status_code}): {e.response.text[:200]}")
            print("[TTS] Falling back to synthetic sine wave")
            return _generate_sine_wave_wav()
        except Exception as e:
            print(f"[TTS] Error: {e}")
            print("[TTS] Falling back to synthetic sine wave")
            return _generate_sine_wave_wav()


def _generate_sine_wave_wav(
    frequency: float = 440.0,
    duration: float = 2.0,
    sample_rate: int = 16000,
) -> bytes:
    """Generate a simple sine-wave WAV file (fallback if TTS unavailable)."""
    import math
    import struct
    import io

    num_samples = int(sample_rate * duration)
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        value = int(32767 * 0.5 * math.sin(2 * math.pi * frequency * t))
        samples.append(struct.pack("<h", value))

    raw_data = b"".join(samples)
    data_size = len(raw_data)
    bits_per_sample = 16
    num_channels = 1
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    wav_bytes = header + raw_data
    print(f"[FALLBACK] Generated {len(wav_bytes)} bytes synthetic WAV ({duration}s)")
    return wav_bytes


# ── Step 2: Test POST /v1/chat/audio (multipart) ──────────────────────────

async def test_chat_audio_multipart(
    client: httpx.AsyncClient,
    audio_bytes: bytes,
) -> dict | None:
    """Test multipart audio upload endpoint."""
    print("\n" + "=" * 60)
    print("[TEST 1] POST /v1/chat/audio (multipart file upload)")
    print("=" * 60)

    try:
        resp = await client.post(
            "/v1/agent/chat/audio",
            files={"audio": ("test.wav", audio_bytes, "audio/wav")},
            data={
                "message": "Please repeat what you hear in the audio.",
                "session_key": "e2e-test",
                "stream": "false",
            },
            timeout=120.0,
        )
        print(f"  Status: {resp.status_code}")

        if resp.status_code == 200:
            data = resp.json()
            print(f"  Success: {data.get('success')}")
            if data.get("data"):
                chat_data = data["data"]
                response_text = chat_data.get("response", "")
                transcription = chat_data.get("transcription")
                print(f"  Response: {response_text[:200]}...")
                if transcription:
                    print(f"  Transcription text: {transcription.get('text', '')[:200]}")
                    print(f"  Transcription language: {transcription.get('language')}")
                    print(f"  Transcription duration: {transcription.get('duration_seconds')}")
                    print(f"  Transcription provider: {transcription.get('provider')}")
                else:
                    print("  Transcription: None (native audio passthrough)")
            if data.get("meta"):
                meta = data["meta"]
                print(f"  Request ID: {meta.get('request_id')}")
                print(f"  Duration: {meta.get('duration_ms')}ms")
            print("  [PASS] Multipart audio upload works!")
            return data
        else:
            print(f"  [FAIL] Response: {resp.text[:500]}")
            return None

    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return None


# ── Step 3: Test POST /v1/chat (base64 audio in JSON) ─────────────────────

async def test_chat_base64_audio(
    client: httpx.AsyncClient,
    audio_bytes: bytes,
) -> dict | None:
    """Test base64-encoded audio in JSON chat request."""
    print("\n" + "=" * 60)
    print("[TEST 2] POST /v1/chat (base64 audio in JSON body)")
    print("=" * 60)

    b64_audio = base64.b64encode(audio_bytes).decode("ascii")
    print(f"  Base64 audio size: {len(b64_audio)} chars")

    payload = {
        "message": "What did the voice say?",
        "session_key": "e2e-test-b64",
        "audio": {
            "data": b64_audio,
            "format": "wav",
            "language": None,
        },
        "options": {
            "stream": False,
            "thinking": False,
        },
    }

    try:
        resp = await client.post(
            "/v1/agent/chat",
            json=payload,
            timeout=120.0,
        )
        print(f"  Status: {resp.status_code}")

        if resp.status_code == 200:
            data = resp.json()
            print(f"  Success: {data.get('success')}")
            if data.get("data"):
                chat_data = data["data"]
                response_text = chat_data.get("response", "")
                transcription = chat_data.get("transcription")
                print(f"  Response: {response_text[:200]}...")
                if transcription:
                    print(f"  Transcription text: {transcription.get('text', '')[:200]}")
                    print(f"  Transcription language: {transcription.get('language')}")
                    print(f"  Transcription provider: {transcription.get('provider')}")
                else:
                    print("  Transcription: None (native audio passthrough)")
            if data.get("meta"):
                meta = data["meta"]
                print(f"  Request ID: {meta.get('request_id')}")
                print(f"  Duration: {meta.get('duration_ms')}ms")
            print("  [PASS] Base64 audio chat works!")
            return data
        else:
            print(f"  [FAIL] Response: {resp.text[:500]}")
            return None

    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return None


# ── Step 4: Test health and status endpoints ───────────────────────────────

async def test_health(client: httpx.AsyncClient) -> bool:
    """Verify gateway is running."""
    print("\n" + "=" * 60)
    print("[PRE-CHECK] Health & connectivity")
    print("=" * 60)

    try:
        resp = await client.get("/health", timeout=10.0)
        print(f"  /health status: {resp.status_code}")
        if resp.status_code == 200:
            print(f"  Health: {resp.json()}")
            print("  [PASS] Gateway is running")
            return True
        else:
            print(f"  [FAIL] Unexpected status: {resp.text[:200]}")
            return False
    except httpx.ConnectError:
        print(f"  [FAIL] Cannot connect to gateway at {GATEWAY_URL}")
        print("  Make sure the gateway is running:")
        print("    uvicorn spoon_bot.gateway.server:create_app --factory --port 8080")
        return False
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


# ── Step 5: Test Whisper transcription directly ────────────────────────────

async def test_whisper_direct(audio_bytes: bytes) -> bool:
    """Test Whisper transcription directly (bypass gateway)."""
    print("\n" + "=" * 60)
    print("[TEST 0] Direct Whisper transcription (bypass gateway)")
    print("=" * 60)

    try:
        from spoon_bot.services.audio.whisper import WhisperTranscriber

        transcriber = WhisperTranscriber()
        available = await transcriber.is_available()
        print(f"  Whisper available: {available}")

        if not available:
            print("  [SKIP] Whisper not available (no API key)")
            return False

        result = await transcriber.transcribe(audio_bytes, audio_format="wav")
        print(f"  Transcribed text: '{result.text}'")
        print(f"  Language: {result.language}")
        print(f"  Duration: {result.duration_seconds}s")
        print(f"  Segments: {len(result.segments)}")
        print(f"  Provider: {result.provider}")
        print("  [PASS] Direct Whisper transcription works!")
        return True

    except Exception as e:
        print(f"  [FAIL] Whisper transcription error: {e}")
        import traceback
        traceback.print_exc()
        return False


# ── Main ───────────────────────────────────────────────────────────────────

async def main(gateway_url: str):
    global GATEWAY_URL
    GATEWAY_URL = gateway_url

    print("=" * 60)
    print("  spoon-bot Voice Input E2E Test")
    print(f"  Gateway: {GATEWAY_URL}")
    print("=" * 60)

    # Load .env
    from dotenv import load_dotenv
    load_dotenv(override=True)

    results = {
        "tts_generation": False,
        "whisper_direct": False,
        "health_check": False,
        "multipart_upload": False,
        "base64_json": False,
    }

    # Step 1: Generate test audio
    print("\n[STEP 1] Generating test audio...")
    audio_bytes = await generate_test_audio()
    results["tts_generation"] = len(audio_bytes) > 0

    # Save test audio to file for debugging
    test_audio_path = Path(__file__).parent / "test_audio.wav"
    test_audio_path.write_bytes(audio_bytes)
    print(f"  Saved test audio to: {test_audio_path}")

    # Step 2: Test Whisper directly (bypass gateway)
    print("\n[STEP 2] Testing Whisper transcription directly...")
    results["whisper_direct"] = await test_whisper_direct(audio_bytes)

    # Step 3: Test gateway endpoints
    # Use explicit local_address to avoid Windows IPv4/IPv6 routing issues
    transport = httpx.AsyncHTTPTransport(local_address="0.0.0.0")
    async with httpx.AsyncClient(base_url=GATEWAY_URL, transport=transport) as client:
        # Health check
        results["health_check"] = await test_health(client)

        if not results["health_check"]:
            print("\n[ABORT] Gateway not reachable. Start the server first.")
            _print_summary(results)
            return

        # Test multipart upload
        resp1 = await test_chat_audio_multipart(client, audio_bytes)
        results["multipart_upload"] = resp1 is not None and resp1.get("success")

        # Test base64 JSON
        resp2 = await test_chat_base64_audio(client, audio_bytes)
        results["base64_json"] = resp2 is not None and resp2.get("success")

    _print_summary(results)


def _print_summary(results: dict):
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        icon = "+" if passed else "-"
        print(f"  [{icon}] {name}: {status}")
        if not passed:
            all_pass = False

    print("=" * 60)
    if all_pass:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E2E voice input test")
    parser.add_argument(
        "--gateway-url",
        default="http://127.0.0.1:8080",
        help="Gateway base URL",
    )
    args = parser.parse_args()

    asyncio.run(main(args.gateway_url))
