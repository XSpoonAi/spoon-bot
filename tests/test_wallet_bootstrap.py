from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest

from spoon_bot.agent.tools.web3 import (
    BalanceCheckTool,
    ContractCallTool,
    SwapTool,
    TransferTool,
)
from spoon_bot.wallet import ensure_wallet_runtime, resolve_wallet_network


def test_ensure_wallet_runtime_creates_wallet_and_exports_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("SPOON_BOT_WALLET_PATH", raising=False)
    for name in (
        "WALLET_ADDRESS",
        "PRIVATE_KEY",
        "WALLET_NETWORK",
        "NEOX_RPC_URL",
        "NEOX_TESTNET_RPC_URL",
    ):
        monkeypatch.delenv(name, raising=False)

    runtime = ensure_wallet_runtime(tmp_path / "workspace")

    assert runtime.address.startswith("0x")
    assert runtime.private_key.startswith("0x")
    assert runtime.network.key == "neox"
    assert runtime.wallet_root == tmp_path / ".agent-wallet"
    assert runtime.keystore_path.exists()
    assert runtime.password_file.exists()
    assert runtime.state_file.exists()
    assert runtime.keystore_path.name == "keystore.json"
    assert runtime.password_file.name == "pw.txt"
    assert runtime.state_file.name == "state.env"
    assert (runtime.wallet_root / "privatekey.tmp").read_text(encoding="utf-8").strip() == runtime.private_key
    state_env = runtime.state_file.read_text(encoding="utf-8")
    assert "KEYSTORE_FILE=" in state_env
    assert "PASSWORD_FILE=" in state_env
    assert "PRIVATE_KEY_FILE=" not in state_env
    assert "ADDRESS=" in state_env
    assert runtime.address == os.environ["WALLET_ADDRESS"]
    assert runtime.private_key == os.environ["PRIVATE_KEY"]
    assert os.environ["WALLET_NETWORK"] == "neox"
    assert "ETH_RPC_URL" in state_env
    assert "NEOX_TESTNET_RPC_URL" in os.environ
    assert os.environ["SPOON_BOT_WALLET_AUTO_CREATED"] == "1"


def test_ensure_wallet_runtime_reuses_existing_wallet(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("SPOON_BOT_WALLET_PATH", raising=False)
    workspace = tmp_path / "workspace"
    first = ensure_wallet_runtime(workspace)
    second = ensure_wallet_runtime(workspace)

    assert first.address == second.address
    assert first.private_key == second.private_key
    assert first.keystore_path == second.keystore_path
    assert (first.wallet_root / "privatekey.tmp").read_text(encoding="utf-8").strip() == first.private_key
    assert os.environ["SPOON_BOT_WALLET_AUTO_CREATED"] == "0"


def test_ensure_wallet_runtime_rejects_partial_wallet_state(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("SPOON_BOT_WALLET_PATH", raising=False)
    workspace = tmp_path / "workspace"

    runtime = ensure_wallet_runtime(workspace)
    keystore_before = runtime.keystore_path.read_text(encoding="utf-8")

    runtime.password_file.unlink()

    with pytest.raises(RuntimeError, match="partial state"):
        ensure_wallet_runtime(workspace)

    assert runtime.keystore_path.read_text(encoding="utf-8") == keystore_before
    assert runtime.state_file.exists()
    assert not runtime.password_file.exists()


def test_ensure_wallet_runtime_does_not_override_rpc_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("SPOON_BOT_WALLET_PATH", raising=False)
    monkeypatch.setenv("RPC_URL", "https://rpc.example.ethereum")
    monkeypatch.setenv("ETH_RPC_URL", "https://rpc.example.eth-alt")

    ensure_wallet_runtime(tmp_path / "workspace")

    assert os.environ["RPC_URL"] == "https://rpc.example.ethereum"
    assert os.environ["ETH_RPC_URL"] == "https://rpc.example.eth-alt"


def test_ensure_wallet_runtime_preserves_existing_private_key_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("SPOON_BOT_WALLET_PATH", raising=False)
    monkeypatch.setenv("PRIVATE_KEY", "0x" + "11" * 32)
    monkeypatch.setenv("WALLET_ADDRESS", "0xExistingWalletAddress")

    runtime = ensure_wallet_runtime(tmp_path / "workspace")

    assert runtime.private_key != os.environ["PRIVATE_KEY"]
    assert os.environ["PRIVATE_KEY"] == "0x" + "11" * 32
    assert os.environ["WALLET_ADDRESS"] == "0xExistingWalletAddress"


def test_resolve_wallet_network_supports_neox_testnet():
    network = resolve_wallet_network("neox_testnet")
    assert network.chain_id == 12227332
    assert network.env_var == "NEOX_TESTNET_RPC_URL"


def test_wallet_root_prefers_userprofile_when_home_is_root(monkeypatch, tmp_path):
    from spoon_bot import wallet as wallet_mod

    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(wallet_mod.Path, "home", lambda: Path("/root"))
    monkeypatch.delenv("SPOON_BOT_WALLET_PATH", raising=False)

    runtime = ensure_wallet_runtime(tmp_path / "workspace")
    assert runtime.wallet_root == tmp_path / ".agent-wallet"


@pytest.mark.asyncio
async def test_create_agent_bootstraps_wallet(monkeypatch, tmp_path):
    from spoon_bot.agent import loop as loop_mod

    calls: list[Path | None] = []

    class FakeAgentLoop:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def initialize(self):
            return None

    def fake_ensure_wallet_runtime(workspace):
        calls.append(workspace)
        return None

    monkeypatch.setattr(loop_mod, "AgentLoop", FakeAgentLoop)
    monkeypatch.setattr(loop_mod, "ensure_wallet_runtime", fake_ensure_wallet_runtime)

    agent = await loop_mod.create_agent(workspace=tmp_path / "workspace")

    assert calls == [tmp_path / "workspace"]
    assert agent.kwargs["workspace"] == tmp_path / "workspace"


@pytest.mark.asyncio
async def test_create_agent_skips_wallet_failure_when_not_required(monkeypatch, tmp_path):
    from spoon_bot.agent import loop as loop_mod

    class FakeAgentLoop:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def initialize(self):
            return None

    def fake_ensure_wallet_runtime(workspace):
        raise OSError(f"read-only fs at {workspace}")

    monkeypatch.setattr(loop_mod, "AgentLoop", FakeAgentLoop)
    monkeypatch.setattr(loop_mod, "ensure_wallet_runtime", fake_ensure_wallet_runtime)

    agent = await loop_mod.create_agent(workspace=tmp_path / "workspace")

    assert agent.kwargs["workspace"] == tmp_path / "workspace"
    assert os.environ["SPOON_BOT_WALLET_AUTO_CREATED"] == "0"


@pytest.mark.asyncio
async def test_create_agent_raises_wallet_failure_when_required(monkeypatch, tmp_path):
    from spoon_bot.agent import loop as loop_mod

    class FakeAgentLoop:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def initialize(self):
            return None

    def fake_ensure_wallet_runtime(workspace):
        raise RuntimeError(f"wallet failed for {workspace}")

    monkeypatch.setattr(loop_mod, "AgentLoop", FakeAgentLoop)
    monkeypatch.setattr(loop_mod, "ensure_wallet_runtime", fake_ensure_wallet_runtime)

    with pytest.raises(RuntimeError):
        await loop_mod.create_agent(
            workspace=tmp_path / "workspace",
            enabled_tools={"transfer"},
        )


@pytest.mark.parametrize(
    ("tool_cls", "expected_default"),
    [
        (BalanceCheckTool, "neox"),
        (TransferTool, "neox"),
        (SwapTool, "neox"),
        (ContractCallTool, "neox"),
    ],
)
def test_web3_tools_default_to_neox(tool_cls, expected_default):
    default_value = inspect.signature(tool_cls.execute).parameters["chain"].default
    assert default_value == expected_default
    assert "neox" in tool_cls.SUPPORTED_CHAINS
    assert "neox_testnet" in tool_cls.SUPPORTED_CHAINS
