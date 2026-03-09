"""Run organized pytest suites for spoon-bot.

Usage:
  uv run python scripts/run_test_suite.py
  uv run python scripts/run_test_suite.py --suite extended
  uv run python scripts/run_test_suite.py --suite all -- -q
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

SUITES: dict[str, list[str]] = {
    # Fast, deterministic regression suite used for day-to-day development.
    "core": [
        "tests/test_agent_unit.py",
        "tests/test_gateway_ws_bugs.py",
        "tests/test_gateway_tracing.py",
        "tests/test_mcp_tool_import_fallback.py",
        "tests/test_security.py",
        "tests/test_semantic_memory_store.py",
        "tests/test_session_persistence.py",
        "tests/test_skills_scripts.py",
    ],
    # Higher-cost / high-churn scenarios; often require extra env setup,
    # platform-specific assumptions, or live services.
    "extended": [
        "tests/capability_test.py",
        "tests/e2e_gateway.py",
        "tests/e2e_gateway_tracing_timeout_real_api.py",
        "tests/e2e_voice_input.py",
        "tests/test_config_and_channels.py",
        "tests/test_e2e_gateway_live.py",
        "tests/test_integration_live.py",
        "tests/test_semantic_memory_e2e.py",
        "tests/test_streaming_thinking.py",
        "tests/test_tools.py",
    ],
    # Keep raw full run available when needed.
    "all": ["tests"],
}

CORE_ENV_UNSET = (
    "SPOON_BOT_DEFAULT_PROVIDER",
    "SPOON_BOT_DEFAULT_MODEL",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run organized pytest suites.")
    parser.add_argument(
        "--suite",
        choices=tuple(SUITES.keys()),
        default="core",
        help="Test suite to run (default: core).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List test files in each suite and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved pytest command and exit without running.",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Additional args forwarded to pytest. Prefix with '--'.",
    )
    return parser.parse_args()


def _normalize_pytest_args(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def _print_suites() -> None:
    for suite_name, files in SUITES.items():
        print(f"[{suite_name}]")
        for path in files:
            print(f"  - {path}")
        print()


def _build_run_env(suite: str) -> tuple[dict[str, str], list[str]]:
    env = os.environ.copy()
    removed: list[str] = []
    if suite == "core":
        for key in CORE_ENV_UNSET:
            if key in env:
                removed.append(key)
                env.pop(key, None)
    return env, removed


def main() -> int:
    args = _parse_args()

    if args.list:
        _print_suites()
        return 0

    suite_files = SUITES[args.suite]
    extra = _normalize_pytest_args(args.pytest_args)
    try:
        import pytest  # noqa: F401
        cmd = [sys.executable, "-m", "pytest", *suite_files, *extra]
    except ModuleNotFoundError:
        uv_bin = shutil.which("uv")
        if uv_bin is None:
            print(
                "pytest is not available in the current environment and `uv` was not found.\n"
                "Install dev dependencies first: `uv sync --extra dev` "
                "or `pip install -e \".[dev]\"`."
            )
            return 2
        # Fallback keeps this script usable on fresh environments.
        cmd = [uv_bin, "run", "--extra", "dev", "python", "-m", "pytest", *suite_files, *extra]

    run_env, removed_env = _build_run_env(args.suite)
    print(f"Running suite: {args.suite}")
    print("Command:", " ".join(cmd))
    if removed_env:
        print("Sanitized env for core:", ", ".join(removed_env))

    if args.dry_run:
        return 0

    completed = subprocess.run(cmd, cwd=REPO_ROOT, env=run_env)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
