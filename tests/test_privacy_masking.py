"""Tests for privacy masking — especially bare hex private key detection."""
from __future__ import annotations

import os

import pytest

from spoon_bot.utils.privacy import mask_secrets


# ---------------------------------------------------------------------------
# Bare hex private key masking
# ---------------------------------------------------------------------------

_FAKE_KEY = "0x" + "ab" * 32  # 0x + 64 hex chars


class TestBareHexPrivateKeyMasking:
    """mask_secrets must redact 0x+64-hex strings that look like private keys."""

    def test_standalone_key_is_masked(self):
        text = f"the key is {_FAKE_KEY} here"
        assert "***masked_private_key***" in mask_secrets(text)
        assert _FAKE_KEY not in mask_secrets(text)

    def test_key_on_its_own_line(self):
        text = f"line1\n{_FAKE_KEY}\nline3"
        result = mask_secrets(text)
        assert _FAKE_KEY not in result
        assert "0x***masked_private_key***" in result

    def test_key_in_env_var_assignment_already_masked(self):
        text = f"PRIVATE_KEY={_FAKE_KEY}"
        result = mask_secrets(text)
        assert _FAKE_KEY not in result

    def test_key_in_cat_output(self):
        text = f"$ cat ~/.agent-wallet/privatekey.tmp\n{_FAKE_KEY}\n"
        result = mask_secrets(text)
        assert _FAKE_KEY not in result

    def test_address_not_masked(self):
        addr = "0x" + "ab" * 20  # 40 hex = address
        text = f"wallet address: {addr}"
        result = mask_secrets(text)
        assert addr in result

    def test_tx_hash_with_context_preserved(self):
        tx_hash = "0x" + "cd" * 32
        text = f"transaction hash: {tx_hash}"
        result = mask_secrets(text)
        assert tx_hash in result

    def test_tx_hash_equals_context_preserved(self):
        tx_hash = "0x" + "cd" * 32
        text = f"tx_hash= {tx_hash}"
        result = mask_secrets(text)
        assert tx_hash in result

    def test_receipt_hash_context_preserved(self):
        tx_hash = "0x" + "ef" * 32
        text = f"receipt_hash: {tx_hash}"
        result = mask_secrets(text)
        assert tx_hash in result

    def test_block_hash_context_preserved(self):
        block_hash = "0x" + "11" * 32
        text = f"block hash: {block_hash}"
        result = mask_secrets(text)
        assert block_hash in result

    def test_bare_hash_label_is_masked(self):
        """'hash=<key>' should be masked — 'hash' alone is too ambiguous."""
        key = "0x" + "ee" * 32
        text = f"hash={key}"
        result = mask_secrets(text)
        assert key not in result

    def test_myhash_label_is_masked(self):
        """'myhash=<key>' must NOT bypass masking (word boundary enforcement)."""
        key = "0x" + "ff" * 32
        text = f"myhash={key}"
        result = mask_secrets(text)
        assert key not in result

    def test_ambiguous_tx_equals_is_masked(self):
        """Bare 'tx=<hex>' is ambiguous and should be masked for safety."""
        key = "0x" + "cd" * 32
        text = f"tx= {key}"
        result = mask_secrets(text)
        assert key not in result

    def test_parent_hash_preserved(self):
        h = "0x" + "99" * 32
        text = f'{{"parentHash":"{h}"}}'
        result = mask_secrets(text)
        assert h in result

    def test_key_longer_than_64_hex_not_masked(self):
        long_hex = "0x" + "ab" * 33  # 66 hex chars
        text = f"data: {long_hex}"
        result = mask_secrets(text)
        assert long_hex in result

    def test_no_0x_prefix_not_masked(self):
        bare = "ab" * 32  # 64 hex but no 0x prefix
        text = f"data: {bare}"
        result = mask_secrets(text)
        assert bare in result

    def test_multiple_keys_masked(self):
        key1 = "0x" + "11" * 32
        key2 = "0x" + "22" * 32
        text = f"keys: {key1} and {key2}"
        result = mask_secrets(text)
        assert key1 not in result
        assert key2 not in result
        assert result.count("***masked_private_key***") == 2

    def test_json_transactionHash_preserved(self):
        """JSON receipt output like {"transactionHash":"0x..."} must not be masked."""
        tx_hash = "0x" + "aa" * 32
        text = f'{{"transactionHash":"{tx_hash}","blockNumber":123}}'
        result = mask_secrets(text)
        assert tx_hash in result

    def test_json_tx_hash_preserved(self):
        tx_hash = "0x" + "bb" * 32
        text = f'{{"tx_hash": "{tx_hash}"}}'
        result = mask_secrets(text)
        assert tx_hash in result

    def test_json_blockHash_preserved(self):
        block_hash = "0x" + "cc" * 32
        text = f'{{"blockHash":"{block_hash}"}}'
        result = mask_secrets(text)
        assert block_hash in result

    def test_json_receipt_hash_preserved(self):
        tx_hash = "0x" + "dd" * 32
        text = f'{{"receiptHash":"{tx_hash}"}}'
        result = mask_secrets(text)
        assert tx_hash in result


# ---------------------------------------------------------------------------
# Existing env var masking (regression)
# ---------------------------------------------------------------------------

class TestEnvVarMasking:
    def test_private_key_assignment(self):
        text = "PRIVATE_KEY=0xsecretvalue123"
        result = mask_secrets(text)
        assert "0xsecretvalue123" not in result
        assert "***masked***" in result

    def test_export_private_key(self):
        text = "export SECRET_KEY=my_super_secret"
        result = mask_secrets(text)
        assert "my_super_secret" not in result

    def test_bearer_token(self):
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig"
        result = mask_secrets(text)
        assert "eyJhbGciOiJIUzI1NiJ9" not in result

    def test_non_sensitive_env_preserved(self):
        text = "HOME=/home/user\nPATH=/usr/bin"
        result = mask_secrets(text)
        assert result == text


# ---------------------------------------------------------------------------
# Shell env scrubbing
# ---------------------------------------------------------------------------

class TestShellEnvScrubbing:
    def test_scrub_env_removes_private_key(self):
        from spoon_bot.agent.tools.shell import _scrub_env

        env = {
            "PATH": "/usr/bin",
            "PRIVATE_KEY": "0xsecret",
            "HOME": "/home/user",
            "SECRET_KEY": "s3cret",
            "MNEMONIC": "word1 word2",
        }
        result = _scrub_env(env)
        assert "PRIVATE_KEY" not in result
        assert "SECRET_KEY" not in result
        assert "MNEMONIC" not in result
        assert result["PATH"] == "/usr/bin"
        assert result["HOME"] == "/home/user"

    def test_scrub_env_noop_when_absent(self):
        from spoon_bot.agent.tools.shell import _scrub_env

        env = {"PATH": "/usr/bin", "HOME": "/home/user"}
        result = _scrub_env(env)
        assert result == {"PATH": "/usr/bin", "HOME": "/home/user"}


# ---------------------------------------------------------------------------
# Wallet state.env no longer contains PRIVATE_KEY_FILE
# ---------------------------------------------------------------------------

class TestWalletStateEnv:
    def test_state_env_excludes_private_key_file(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        monkeypatch.delenv("SPOON_BOT_WALLET_PATH", raising=False)
        for name in ("WALLET_ADDRESS", "PRIVATE_KEY", "WALLET_NETWORK"):
            monkeypatch.delenv(name, raising=False)

        from spoon_bot.wallet import ensure_wallet_runtime

        runtime = ensure_wallet_runtime(tmp_path / "workspace")
        state_content = runtime.state_file.read_text(encoding="utf-8")

        assert "PRIVATE_KEY_FILE" not in state_content
        assert "ADDRESS=" in state_content
        assert "KEYSTORE_FILE=" in state_content
