"""Integration test: verify joker-game-agent wallet works after private key scrubbing.

This script tests that:
1. The joker-game-agent CLI can load the wallet via KeystoreSigner
   (without PRIVATE_KEY in the subprocess environment)
2. The wallet command succeeds and shows address + balances
3. A second invocation in the same process also works (session continuity)
4. No private key hex appears in the captured output

Run:
    python -m pytest tests/test_joker_wallet_compat.py -v -s
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path.home() / ".spoon-bot" / "workspace" / "skills" / "joker-game-agent"
CLI_PATH = SKILL_DIR / "cli" / "index.js"


def _has_skill():
    return CLI_PATH.exists() and (SKILL_DIR / "cli" / "node_modules").exists()


@pytest.mark.skipif(not _has_skill(), reason="joker-game-agent skill not installed")
class TestJokerWalletCompat:
    """Verify the joker-game-agent CLI works without PRIVATE_KEY in env."""

    @staticmethod
    def _run_cli(*args: str, scrub_private_key: bool = True) -> tuple[str, str, int]:
        """Run the joker-game-agent CLI synchronously with env scrubbing."""
        import subprocess

        env = os.environ.copy()
        if scrub_private_key:
            env.pop("PRIVATE_KEY", None)
            env.pop("SECRET_KEY", None)
            env.pop("MNEMONIC", None)

        node = "node"
        cmd = [node, str(CLI_PATH)] + list(args)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
            cwd=str(SKILL_DIR / "cli"),
        )
        return result.stdout, result.stderr, result.returncode

    @staticmethod
    def _get_private_key_hex() -> str | None:
        """Read the raw private key for leak detection."""
        pk_file = Path.home() / ".agent-wallet" / "privatekey.tmp"
        if pk_file.exists():
            return pk_file.read_text(encoding="utf-8").strip()
        return None

    def test_wallet_command_without_private_key_env(self):
        """The CLI should load wallet via keystore when PRIVATE_KEY is absent."""
        stdout, stderr, rc = self._run_cli("wallet", scrub_private_key=True)
        combined = stdout + stderr
        assert rc == 0, f"wallet command failed: {combined}"
        assert "Wallet=" in stdout, f"Expected wallet output, got: {stdout}"

    def test_wallet_shows_address(self):
        """Output should contain a valid Ethereum address."""
        stdout, _, rc = self._run_cli("wallet", scrub_private_key=True)
        assert rc == 0
        match = re.search(r"Wallet=(0x[0-9a-fA-F]{40})", stdout)
        assert match, f"No wallet address found in output: {stdout}"

    def test_no_private_key_in_output(self):
        """The raw private key must not appear in CLI output."""
        pk = self._get_private_key_hex()
        if not pk:
            pytest.skip("No privatekey.tmp found")

        stdout, stderr, _ = self._run_cli("wallet", scrub_private_key=True)
        combined = stdout + stderr
        assert pk not in combined, "Private key leaked in wallet output!"

    def test_two_invocations_same_keystore(self):
        """Two consecutive runs should use the same wallet address (keystore path)."""
        stdout1, _, rc1 = self._run_cli("wallet", scrub_private_key=True)
        stdout2, _, rc2 = self._run_cli("wallet", scrub_private_key=True)
        assert rc1 == 0 and rc2 == 0

        addr1 = re.search(r"Wallet=(0x[0-9a-fA-F]{40})", stdout1)
        addr2 = re.search(r"Wallet=(0x[0-9a-fA-F]{40})", stdout2)
        assert addr1 and addr2
        assert addr1.group(1).lower() == addr2.group(1).lower(), (
            f"Addresses differ: {addr1.group(1)} vs {addr2.group(1)}"
        )

    def test_game_list_without_private_key_env(self):
        """game list should work without PRIVATE_KEY in env (read-only, no signing)."""
        stdout, stderr, rc = self._run_cli("game", "list", scrub_private_key=True)
        combined = stdout + stderr
        # rc=0 means success; rc=1 with network errors is acceptable in CI
        if rc != 0:
            assert any(kw in combined.lower() for kw in ["timeout", "econnrefused", "fetch"]), (
                f"Unexpected error: {combined}"
            )


@pytest.mark.skipif(not _has_skill(), reason="joker-game-agent skill not installed")
class TestShellToolEnvScrubbing:
    """Verify ShellTool scrubs secrets before spawning subprocesses."""

    @pytest.mark.asyncio
    async def test_shell_env_printenv_no_private_key(self):
        """Running 'printenv PRIVATE_KEY' via ShellTool should return empty."""
        from spoon_bot.agent.tools.shell import ShellTool

        os.environ["PRIVATE_KEY"] = "0x_test_secret_value"
        try:
            tool = ShellTool(
                timeout=30,
                working_dir=os.getcwd(),
                allow_chaining=True,
            )
            result = await tool.execute(command="printenv PRIVATE_KEY")
            # After scrubbing, PRIVATE_KEY should not be in subprocess env
            assert "0x_test_secret_value" not in result
        finally:
            os.environ.pop("PRIVATE_KEY", None)

    @pytest.mark.asyncio
    async def test_shell_env_command_no_leak(self):
        """Running 'env' via ShellTool should not show PRIVATE_KEY."""
        from spoon_bot.agent.tools.shell import ShellTool

        os.environ["PRIVATE_KEY"] = "0xdeadbeef1234567890"
        try:
            tool = ShellTool(
                timeout=30,
                working_dir=os.getcwd(),
                allow_chaining=True,
            )
            result = await tool.execute(command="env")
            assert "0xdeadbeef1234567890" not in result
            assert "PRIVATE_KEY=" not in result or "***masked***" in result
        finally:
            os.environ.pop("PRIVATE_KEY", None)
