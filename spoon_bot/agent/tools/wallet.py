"""Built-in wallet management and signing tool."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from spoon_bot.agent.tools.base import Tool
from spoon_bot.wallet import (
    backup_wallet,
    create_new_wallet,
    ensure_wallet_runtime,
    sign_wallet_message,
    sign_wallet_transaction,
    sign_wallet_typed_data,
    wallet_summary,
)


def _env_enabled(name: str) -> bool:
    raw = os.environ.get(name, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class WalletTool(Tool):
    """Manage the local built-in EVM wallet without shelling out."""

    @property
    def name(self) -> str:
        return "wallet"

    @property
    def description(self) -> str:
        return (
            "Manage spoon-bot's built-in local EVM wallet: status, backup, create a new "
            "wallet with automatic backup, export the private key on explicit request, "
            "and sign EIP-191 messages, EIP-712 typed data, or prepared transactions."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "status",
                        "backup",
                        "create_new",
                        "export_private_key",
                        "sign_message",
                        "sign_typed_data",
                        "sign_transaction",
                    ],
                    "description": "Wallet action to perform.",
                },
                "network": {
                    "type": "string",
                    "description": "Wallet network for create_new, for example neox or neox_testnet.",
                },
                "confirm": {
                    "type": "boolean",
                    "description": "Required for create_new and export_private_key.",
                },
                "reveal_private_key": {
                    "type": "boolean",
                    "description": "When exporting, include the raw private key in the tool result.",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional file path to write an exported private key.",
                },
                "message": {
                    "type": "string",
                    "description": "Plain text message for EIP-191 signing.",
                },
                "hexstr": {
                    "type": "string",
                    "description": "Hex encoded message for EIP-191 signing.",
                },
                "typed_data": {
                    "type": "object",
                    "description": "Full EIP-712 typed-data object.",
                },
                "domain_data": {
                    "type": "object",
                    "description": "EIP-712 domain data when not using typed_data.",
                },
                "message_types": {
                    "type": "object",
                    "description": "EIP-712 custom type definitions when not using typed_data.",
                },
                "message_data": {
                    "type": "object",
                    "description": "EIP-712 message data when not using typed_data.",
                },
                "transaction": {
                    "type": "object",
                    "description": "Prepared EVM transaction object to sign without broadcasting.",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        network: str | None = None,
        confirm: bool = False,
        reveal_private_key: bool = False,
        output_path: str | None = None,
        message: str | None = None,
        hexstr: str | None = None,
        typed_data: dict[str, Any] | str | None = None,
        domain_data: dict[str, Any] | str | None = None,
        message_types: dict[str, Any] | str | None = None,
        message_data: dict[str, Any] | str | None = None,
        transaction: dict[str, Any] | str | None = None,
        **_: Any,
    ) -> str:
        try:
            if action == "status":
                runtime = ensure_wallet_runtime()
                summary = wallet_summary(runtime)
                summary.pop("private_key", None)
                summary["private_key_available"] = True
                return json.dumps(summary, ensure_ascii=True)

            if action == "backup":
                backup_dir = backup_wallet()
                return json.dumps(
                    {
                        "success": True,
                        "backup_dir": str(backup_dir) if backup_dir else None,
                        "message": "No wallet files existed to back up" if backup_dir is None else "Wallet backed up",
                    },
                    ensure_ascii=True,
                )

            if action == "create_new":
                if not confirm:
                    return (
                        "Safety check failed: creating a new wallet replaces the active wallet files. "
                        "Call again with confirm=true; existing wallet files will be backed up first."
                    )
                if _env_enabled("SPOON_BOT_DISABLE_WALLET_REPLACE"):
                    return (
                        "Safety check failed: wallet replacement is disabled by runtime "
                        "configuration. Unset SPOON_BOT_DISABLE_WALLET_REPLACE or set it "
                        "to false, then retry after explicit user confirmation."
                    )
                runtime, backup_dir = create_new_wallet(network=network, backup_existing=True)
                return json.dumps(
                    {
                        "success": True,
                        "address": runtime.address,
                        "network": runtime.network.key,
                        "wallet_root": str(runtime.wallet_root),
                        "backup_dir": str(backup_dir) if backup_dir else None,
                    },
                    ensure_ascii=True,
                )

            if action == "export_private_key":
                if not confirm:
                    return (
                        "Safety check failed: private key export requires confirm=true and should only "
                        "be used when the user explicitly asks for the secret."
                    )
                if _env_enabled("SPOON_BOT_DISABLE_PRIVATE_KEY_EXPORT"):
                    return (
                        "Safety check failed: private key export is disabled by runtime "
                        "configuration. Unset SPOON_BOT_DISABLE_PRIVATE_KEY_EXPORT or set it "
                        "to false, then retry after explicit user confirmation."
                    )
                runtime = ensure_wallet_runtime()
                payload: dict[str, Any] = {
                    "success": True,
                    "address": runtime.address,
                    "network": runtime.network.key,
                }
                if output_path:
                    output = Path(output_path).expanduser().resolve()
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(runtime.private_key, encoding="utf-8")
                    try:
                        output.chmod(0o600)
                    except OSError:
                        pass
                    payload["output_path"] = str(output)
                if reveal_private_key:
                    payload["private_key"] = runtime.private_key
                if not output_path and not reveal_private_key:
                    payload["message"] = "Private key export confirmed, but no output_path or reveal_private_key was requested."
                return json.dumps(payload, ensure_ascii=True)

            if action == "sign_message":
                result = sign_wallet_message(message=message, hexstr=hexstr)
                return json.dumps(result, ensure_ascii=True)

            if action == "sign_typed_data":
                result = sign_wallet_typed_data(
                    full_message=self._object_or_json(typed_data, "typed_data"),
                    domain_data=self._object_or_json(domain_data, "domain_data"),
                    message_types=self._object_or_json(message_types, "message_types"),
                    message_data=self._object_or_json(message_data, "message_data"),
                )
                return json.dumps(result, ensure_ascii=True)

            if action == "sign_transaction":
                tx = self._object_or_json(transaction, "transaction")
                if tx is None:
                    return "Error: transaction is required for sign_transaction"
                result = sign_wallet_transaction(tx)
                return json.dumps(result, ensure_ascii=True)

            return f"Error: Unsupported wallet action '{action}'"
        except Exception as exc:
            return f"Error: wallet action failed: {exc}"

    @staticmethod
    def _object_or_json(value: dict[str, Any] | str | None, field_name: str) -> dict[str, Any] | None:
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{field_name} must be a JSON object") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"{field_name} must be a JSON object")
            return parsed
        raise ValueError(f"{field_name} must be a JSON object")
