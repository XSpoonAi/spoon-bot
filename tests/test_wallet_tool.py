from __future__ import annotations

import json
import os

import pytest

from spoon_bot.agent.tools.wallet import WalletTool
from spoon_bot.agent.tools.web3 import ContractCallTool, TransferTool
from spoon_bot.wallet import ensure_wallet_runtime


@pytest.fixture
def wallet_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("SPOON_BOT_WALLET_PATH", raising=False)
    for name in (
        "WALLET_ADDRESS",
        "PRIVATE_KEY",
        "WALLET_NETWORK",
        "WALLET_KEYSTORE_PATH",
        "WALLET_PASSWORD_FILE",
        "CHAIN_ID",
        "AGENT_WALLET_DIR",
        "SPOON_BOT_WALLET_AUTO_CREATED",
        "SPOON_BOT_DISABLE_PRIVATE_KEY_EXPORT",
        "SPOON_BOT_DISABLE_WALLET_REPLACE",
        "SPOON_BOT_DISABLE_LIVE_TRANSFERS",
    ):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


@pytest.mark.asyncio
async def test_wallet_tool_status_redacts_private_key(wallet_home):
    runtime = ensure_wallet_runtime(wallet_home / "workspace")
    result = await WalletTool().execute(action="status")
    payload = json.loads(result)

    assert payload["address"] == runtime.address
    assert payload["private_key_available"] is True
    assert "private_key" not in payload


@pytest.mark.asyncio
async def test_wallet_tool_create_new_backs_up_existing_wallet(wallet_home):
    first = ensure_wallet_runtime(wallet_home / "workspace")
    result = await WalletTool().execute(action="create_new", confirm=True, network="neox_testnet")
    payload = json.loads(result)

    assert payload["success"] is True
    assert payload["address"] != first.address
    assert payload["network"] == "neox_testnet"
    backup_dir = payload["backup_dir"]
    assert backup_dir
    assert os.path.exists(os.path.join(backup_dir, "keystore.json"))
    assert os.path.exists(os.path.join(backup_dir, "pw.txt"))
    assert os.path.exists(os.path.join(backup_dir, "state.env"))


@pytest.mark.asyncio
async def test_wallet_tool_export_private_key_requires_confirmation(wallet_home, tmp_path):
    runtime = ensure_wallet_runtime(wallet_home / "workspace")

    denied = await WalletTool().execute(action="export_private_key", reveal_private_key=True)
    assert "Safety check failed" in denied

    output_path = tmp_path / "exported.key"
    result = await WalletTool().execute(
        action="export_private_key",
        confirm=True,
        reveal_private_key=True,
        output_path=str(output_path),
    )
    payload = json.loads(result)
    assert payload["private_key"] == runtime.private_key
    assert output_path.read_text(encoding="utf-8") == runtime.private_key


@pytest.mark.asyncio
async def test_wallet_create_new_disable_env_blocks_confirm(wallet_home, monkeypatch):
    ensure_wallet_runtime(wallet_home / "workspace")
    monkeypatch.setenv("SPOON_BOT_DISABLE_WALLET_REPLACE", "true")

    denied = await WalletTool().execute(action="create_new", confirm=True, network="neox_testnet")

    assert "wallet replacement is disabled by runtime configuration" in denied


@pytest.mark.asyncio
async def test_wallet_export_disable_env_blocks_confirm(wallet_home, monkeypatch):
    ensure_wallet_runtime(wallet_home / "workspace")
    monkeypatch.setenv("SPOON_BOT_DISABLE_PRIVATE_KEY_EXPORT", "true")

    denied = await WalletTool().execute(
        action="export_private_key",
        confirm=True,
        reveal_private_key=True,
    )

    assert "private key export is disabled by runtime configuration" in denied


@pytest.mark.asyncio
async def test_transfer_disable_env_blocks_confirm(wallet_home, monkeypatch):
    tool = TransferTool()
    monkeypatch.setenv("SPOON_BOT_DISABLE_LIVE_TRANSFERS", "true")

    denied_without_confirm = await tool.execute(
        to_address="0x0000000000000000000000000000000000000001",
        amount="0.1",
        chain="neox",
    )
    assert "Transfer not confirmed" in denied_without_confirm

    denied_with_confirm = await tool.execute(
        to_address="0x0000000000000000000000000000000000000001",
        amount="0.1",
        chain="neox",
        confirm=True,
    )
    assert "live blockchain writes are disabled by runtime configuration" in denied_with_confirm
    assert "SPOON_BOT_DISABLE_LIVE_TRANSFERS" in denied_with_confirm


@pytest.mark.asyncio
async def test_contract_write_disable_env_blocks_confirm(wallet_home, monkeypatch):
    tool = ContractCallTool()
    monkeypatch.setenv("SPOON_BOT_DISABLE_LIVE_TRANSFERS", "true")

    denied = await tool.execute(
        contract_address="0x0000000000000000000000000000000000000001",
        method="transfer",
        args=["0x0000000000000000000000000000000000000002", 1],
        chain="neox",
        is_write=True,
        confirm=True,
    )

    assert "live blockchain writes are disabled by runtime configuration" in denied


@pytest.mark.asyncio
async def test_wallet_tool_signs_message_and_prepared_transaction(wallet_home):
    runtime = ensure_wallet_runtime(wallet_home / "workspace")
    tool = WalletTool()

    signed_message = json.loads(await tool.execute(action="sign_message", message="hello"))
    assert signed_message["address"] == runtime.address
    assert signed_message["signature"].startswith("0x")
    assert signed_message["kind"] == "eip191"

    signed_typed_data = json.loads(
        await tool.execute(
            action="sign_typed_data",
            typed_data={
                "types": {
                    "EIP712Domain": [
                        {"name": "name", "type": "string"},
                        {"name": "version", "type": "string"},
                        {"name": "chainId", "type": "uint256"},
                    ],
                    "Permit": [
                        {"name": "owner", "type": "address"},
                        {"name": "spender", "type": "address"},
                        {"name": "value", "type": "uint256"},
                    ],
                },
                "primaryType": "Permit",
                "domain": {"name": "Spoon Test", "version": "1", "chainId": 1},
                "message": {
                    "owner": "0x0000000000000000000000000000000000000001",
                    "spender": "0x0000000000000000000000000000000000000002",
                    "value": 1,
                },
            },
        )
    )
    assert signed_typed_data["address"] == runtime.address
    assert signed_typed_data["signature"].startswith("0x")
    assert signed_typed_data["kind"] == "eip712"

    signed_tx = json.loads(
        await tool.execute(
            action="sign_transaction",
            transaction={
                "to": "0x0000000000000000000000000000000000000001",
                "value": 0,
                "nonce": 0,
                "gas": 21000,
                "gasPrice": 1,
                "chainId": 1,
            },
        )
    )
    assert signed_tx["address"] == runtime.address
    assert signed_tx["raw_transaction"].startswith("0x")
    assert signed_tx["transaction_hash"].startswith("0x")
