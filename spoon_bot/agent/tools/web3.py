"""Web3 blockchain tools: balance check, transfer, swap, contract calls."""

from __future__ import annotations

import json
import os
from decimal import Decimal, InvalidOperation
from typing import Any

from eth_account import Account

from spoon_bot.agent.tools.base import Tool
from spoon_bot.wallet import DEFAULT_WALLET_NETWORK, ensure_wallet_runtime

SUPPORTED_WEB3_CHAINS = frozenset({
    "ethereum", "polygon", "arbitrum", "optimism", "base",
    "bsc", "avalanche", "fantom", "sepolia", "goerli",
    "neox", "neox_testnet",
})

NETWORK_SYMBOLS = {
    "ethereum": "ETH",
    "sepolia": "ETH",
    "goerli": "ETH",
    "polygon": "MATIC",
    "arbitrum": "ETH",
    "optimism": "ETH",
    "base": "ETH",
    "bsc": "BNB",
    "avalanche": "AVAX",
    "fantom": "FTM",
    "neox": "GAS",
    "neox_testnet": "GAS",
}


def _decimal_to_units(amount: str, decimals: int) -> int:
    try:
        value = Decimal(str(amount))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid amount '{amount}'. Must be a valid number.") from exc
    if value <= 0:
        raise ValueError("Amount must be greater than 0")
    units = value * (Decimal(10) ** int(decimals))
    if units != units.to_integral_value():
        raise ValueError(f"Amount has more than {decimals} decimal places")
    return int(units)


def _get_env_rpc_url(chain: str, default_rpc_url: str | None = None) -> str | None:
    chain_env_var = f"{chain.upper()}_RPC_URL"
    rpc_url = os.environ.get(chain_env_var)
    if not rpc_url and os.environ.get("WALLET_NETWORK") == chain:
        rpc_url = os.environ.get("ETH_RPC_URL")
    if not rpc_url and chain == "ethereum":
        rpc_url = os.environ.get("RPC_URL") or os.environ.get("ETH_RPC_URL") or default_rpc_url
    return rpc_url


def _get_default_wallet_address() -> str | None:
    address = os.environ.get("WALLET_ADDRESS")
    if address:
        return address
    try:
        return ensure_wallet_runtime().address
    except Exception:
        return None


def _get_signing_private_key() -> str | None:
    private_key = os.environ.get("PRIVATE_KEY")
    if private_key:
        return private_key
    try:
        return ensure_wallet_runtime().private_key
    except Exception:
        return None


def _signed_raw_transaction(signed: Any) -> Any:
    return getattr(signed, "raw_transaction", None) or getattr(signed, "rawTransaction", None)


def _fee_fields(web3: Any) -> dict[str, int]:
    try:
        latest = web3.eth.get_block("latest")
        base_fee = latest.get("baseFeePerGas") if hasattr(latest, "get") else latest["baseFeePerGas"]
    except Exception:
        base_fee = None

    if base_fee is not None:
        try:
            priority_fee = int(web3.eth.max_priority_fee)
        except Exception:
            priority_fee = int(web3.to_wei(Decimal("1.5"), "gwei"))
        return {
            "maxPriorityFeePerGas": priority_fee,
            "maxFeePerGas": int(base_fee) * 2 + priority_fee,
        }

    return {"gasPrice": int(web3.eth.gas_price)}


def _env_enabled(name: str) -> bool:
    raw = os.environ.get(name, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _live_transfer_disabled_message(operation: str) -> str:
    return (
        "SAFETY CHECK FAILED: live blockchain writes are disabled by runtime configuration.\n\n"
        f"Blocked operation: {operation}\n"
        "Unset SPOON_BOT_DISABLE_LIVE_TRANSFERS or set it to false, then retry "
        "after the user explicitly confirms the exact operation."
    )


def _to_hex(web3: Any, value: Any) -> str:
    return web3.to_hex(value) if hasattr(web3, "to_hex") else value.hex()


ERC20_ABI: list[dict[str, Any]] = [
    {
        "constant": True,
        "inputs": [{"name": "owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function",
    },
    {
        "constant": False,
        "inputs": [
            {"name": "to", "type": "address"},
            {"name": "value", "type": "uint256"},
        ],
        "name": "transfer",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function",
    },
]


class BalanceCheckTool(Tool):
    """
    Tool to check wallet balance across multiple chains.

    Supports checking native token balances and ERC20 token balances.
    Requires RPC_URL environment variable or provider configuration.
    """

    # Supported chains and their default RPC endpoints (env var overrides)
    SUPPORTED_CHAINS = SUPPORTED_WEB3_CHAINS

    def __init__(self, default_rpc_url: str | None = None):
        """
        Initialize balance check tool.

        Args:
            default_rpc_url: Default RPC URL for Ethereum mainnet.
        """
        self._default_rpc_url = default_rpc_url

    @property
    def name(self) -> str:
        return "balance_check"

    @property
    def description(self) -> str:
        return (
            "Check wallet balance for native tokens and ERC20 tokens. "
            "Defaults to Neo X mainnet and also supports Neo X testnet plus common EVM networks."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "address": {
                    "type": "string",
                    "description": "Wallet address to check. If omitted, uses configured wallet.",
                },
                "chain": {
                    "type": "string",
                    "description": "Blockchain network (default: ethereum)",
                    "enum": list(self.SUPPORTED_CHAINS),
                },
                "tokens": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of ERC20 token addresses to check. Omit for native balance only.",
                },
            },
            "required": [],
        }

    def _get_rpc_url(self, chain: str) -> str | None:
        """Get RPC URL for the specified chain from environment."""
        return _get_env_rpc_url(chain, self._default_rpc_url)

    def _get_default_address(self) -> str | None:
        """Get default wallet address from environment."""
        return _get_default_wallet_address()

    async def execute(
        self,
        address: str | None = None,
        chain: str = DEFAULT_WALLET_NETWORK,
        tokens: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Check wallet balance.

        Args:
            address: Wallet address to check.
            chain: Blockchain network.
            tokens: Optional list of ERC20 token addresses.

        Returns:
            Formatted balance information or error message.
        """
        # Validate chain
        if chain not in self.SUPPORTED_CHAINS:
            return f"Error: Unsupported chain '{chain}'. Supported: {', '.join(sorted(self.SUPPORTED_CHAINS))}"

        # Get address
        wallet_address = address or self._get_default_address()
        if not wallet_address:
            return (
                "Error: No wallet address provided. "
                "Either pass 'address' parameter or set WALLET_ADDRESS environment variable."
            )

        # Check RPC configuration
        rpc_url = self._get_rpc_url(chain)
        if not rpc_url:
            chain_env_var = f"{chain.upper()}_RPC_URL"
            return (
                f"Error: Web3 provider not configured for {chain}. "
                f"Set {chain_env_var} environment variable with your RPC URL.\n\n"
                f"Example:\n"
                f"  export {chain_env_var}=https://mainnet.infura.io/v3/YOUR_KEY\n\n"
                f"Or use services like:\n"
                f"  - Infura: https://infura.io\n"
                f"  - Alchemy: https://alchemy.com\n"
                f"  - QuickNode: https://quicknode.com"
            )

        try:
            from web3 import Web3
        except ImportError:
            return (
                "Error: Web3 provider not available. Install web3.py to enable blockchain tools.\n"
                "  pip install web3"
            )

        def _native_symbol(chain_name: str) -> str:
            return NETWORK_SYMBOLS.get(chain_name, "NATIVE")

        def _fetch_balances() -> dict[str, Any]:
            web3 = Web3(Web3.HTTPProvider(rpc_url))
            if not web3.is_connected():
                raise RuntimeError("RPC connection failed")

            checksum_address = web3.to_checksum_address(wallet_address)
            result: dict[str, Any] = {
                "chain": chain,
                "address": checksum_address,
                "native": {},
                "tokens": [],
            }

            native_balance = web3.eth.get_balance(checksum_address)
            result["native"] = {
                "symbol": _native_symbol(chain),
                "wei": str(native_balance),
                "amount": str(web3.from_wei(native_balance, "ether")),
            }

            if tokens:
                for token in tokens:
                    token_address = web3.to_checksum_address(token)
                    contract = web3.eth.contract(address=token_address, abi=ERC20_ABI)
                    balance = contract.functions.balanceOf(checksum_address).call()
                    try:
                        decimals = contract.functions.decimals().call()
                    except Exception:
                        decimals = 18
                    try:
                        symbol = contract.functions.symbol().call()
                    except Exception:
                        symbol = "TOKEN"

                    amount = balance / (10 ** decimals)
                    result["tokens"].append(
                        {
                            "address": token_address,
                            "symbol": symbol,
                            "decimals": decimals,
                            "raw": str(balance),
                            "amount": str(amount),
                        }
                    )

            return result

        try:
            import asyncio

            balances = await asyncio.to_thread(_fetch_balances)
            return json.dumps(balances, ensure_ascii=True)
        except Exception as e:
            return f"Error: Balance check failed: {e}"


class TransferTool(Tool):
    """
    Tool to transfer native tokens or ERC20 tokens.

    SAFETY: Requires explicit confirmation before executing transfers.
    This tool handles real value and transactions are irreversible.
    """

    SUPPORTED_CHAINS = SUPPORTED_WEB3_CHAINS

    def __init__(self, require_confirmation: bool = True):
        """
        Initialize transfer tool.

        Args:
            require_confirmation: If True, requires explicit confirmation.
        """
        self._require_confirmation = require_confirmation

    @property
    def name(self) -> str:
        return "transfer"

    @property
    def description(self) -> str:
        return (
            "Transfer native tokens or ERC20 tokens to another address. "
            "IMPORTANT: Requires confirmation parameter set to true. "
            "Transactions are irreversible - verify all parameters carefully."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "to_address": {
                    "type": "string",
                    "description": "Recipient wallet address",
                },
                "amount": {
                    "type": "string",
                    "description": "Amount to transfer (in human-readable units, e.g., '1.5' for 1.5 ETH)",
                },
                "token": {
                    "type": "string",
                    "description": "Token to transfer: 'native' for chain native token, or ERC20 contract address",
                },
                "chain": {
                    "type": "string",
                    "description": "Blockchain network (default: ethereum)",
                    "enum": list(self.SUPPORTED_CHAINS),
                },
                "confirm": {
                    "type": "boolean",
                    "description": "REQUIRED: Must be true to execute. Safety check to prevent accidental transfers.",
                },
            },
            "required": ["to_address", "amount", "confirm"],
        }

    def _get_rpc_url(self, chain: str) -> str | None:
        """Get RPC URL for the specified chain from environment."""
        return _get_env_rpc_url(chain)

    def _get_private_key(self) -> str | None:
        """Get private key from environment."""
        return _get_signing_private_key()

    async def execute(
        self,
        to_address: str,
        amount: str,
        token: str = "native",
        chain: str = DEFAULT_WALLET_NETWORK,
        confirm: bool = False,
        **kwargs: Any,
    ) -> str:
        """
        Transfer tokens.

        Args:
            to_address: Recipient address.
            amount: Amount to transfer.
            token: 'native' or ERC20 contract address.
            chain: Blockchain network.
            confirm: Must be True to execute.

        Returns:
            Transaction hash or error message.
        """
        # Safety check: require confirmation
        if self._require_confirmation and not confirm:
            return (
                "SAFETY CHECK FAILED: Transfer not confirmed.\n\n"
                "This operation will transfer real tokens and is IRREVERSIBLE.\n"
                "Please review the transfer details:\n"
                f"  To: {to_address}\n"
                f"  Amount: {amount}\n"
                f"  Token: {token}\n"
                f"  Chain: {chain}\n\n"
                "To proceed, call this tool again with confirm=true after the user "
                "explicitly confirms the exact operation."
            )
        if self._require_confirmation and _env_enabled("SPOON_BOT_DISABLE_LIVE_TRANSFERS"):
            return _live_transfer_disabled_message(
                f"transfer {amount} {token} on {chain} to {to_address}"
            )

        # Validate chain
        if chain not in self.SUPPORTED_CHAINS:
            return f"Error: Unsupported chain '{chain}'. Supported: {', '.join(sorted(self.SUPPORTED_CHAINS))}"

        # Validate address format (basic check)
        if not to_address.startswith("0x") or len(to_address) != 42:
            return "Error: Invalid address format. Expected 0x followed by 40 hex characters."

        # Check RPC configuration
        rpc_url = self._get_rpc_url(chain)
        if not rpc_url:
            chain_env_var = f"{chain.upper()}_RPC_URL"
            return (
                f"Error: Web3 provider not configured for {chain}. "
                f"Set {chain_env_var} environment variable."
            )

        # Check private key
        private_key = self._get_private_key()
        if not private_key:
            return (
                "Error: Private key not configured. "
                "Set PRIVATE_KEY environment variable.\n\n"
                "WARNING: Never share your private key. Use a dedicated wallet for bot operations."
            )

        try:
            from web3 import Web3
        except ImportError:
            return (
                "Error: Web3 provider not available. Install web3.py to enable blockchain tools.\n"
                "  pip install web3"
            )

        try:
            web3 = Web3(Web3.HTTPProvider(rpc_url))
            if not web3.is_connected():
                return "Error: RPC connection failed"

            account = Account.from_key(private_key)
            sender = web3.to_checksum_address(account.address)
            recipient = web3.to_checksum_address(to_address)
            nonce = web3.eth.get_transaction_count(sender)
            chain_id = int(web3.eth.chain_id)
            base_tx: dict[str, Any] = {
                "from": sender,
                "nonce": nonce,
                "chainId": chain_id,
                **_fee_fields(web3),
            }

            if token == "native":
                value = _decimal_to_units(amount, 18)
                tx: dict[str, Any] = {
                    **base_tx,
                    "to": recipient,
                    "value": value,
                }
                tx["gas"] = int(web3.eth.estimate_gas(tx))
                symbol = NETWORK_SYMBOLS.get(chain, "NATIVE")
            else:
                token_address = web3.to_checksum_address(token)
                contract = web3.eth.contract(address=token_address, abi=ERC20_ABI)
                try:
                    decimals = int(contract.functions.decimals().call())
                except Exception:
                    decimals = 18
                try:
                    symbol = str(contract.functions.symbol().call())
                except Exception:
                    symbol = "TOKEN"
                value = _decimal_to_units(amount, decimals)
                tx = contract.functions.transfer(recipient, value).build_transaction(base_tx)
                if "gas" not in tx:
                    tx["gas"] = int(web3.eth.estimate_gas(tx))

            signed = web3.eth.account.sign_transaction(tx, private_key)
            tx_hash = web3.eth.send_raw_transaction(_signed_raw_transaction(signed))
            return json.dumps(
                {
                    "success": True,
                    "chain": chain,
                    "chain_id": chain_id,
                    "from": sender,
                    "to": recipient,
                    "token": token,
                    "symbol": symbol,
                    "amount": str(amount),
                    "tx_hash": _to_hex(web3, tx_hash),
                },
                ensure_ascii=True,
            )
        except Exception as e:
            return f"Error: Transfer failed: {e}"


class SwapTool(Tool):
    """
    Tool to perform DEX swaps.

    SAFETY: Requires explicit confirmation before executing swaps.
    Swaps involve real value and slippage - verify all parameters carefully.
    """

    SUPPORTED_CHAINS = frozenset({
        "ethereum", "polygon", "arbitrum", "optimism", "base",
        "bsc", "avalanche", "neox", "neox_testnet",
    })

    # Common DEX routers by chain (would be used in real implementation)
    DEX_ROUTERS = {
        "ethereum": "Uniswap V3",
        "polygon": "QuickSwap / Uniswap V3",
        "arbitrum": "Uniswap V3 / Camelot",
        "optimism": "Velodrome / Uniswap V3",
        "base": "Aerodrome / Uniswap V3",
        "bsc": "PancakeSwap",
        "avalanche": "TraderJoe / Pangolin",
        "neox": "Neo X DEX / custom router",
        "neox_testnet": "Neo X testnet DEX / custom router",
    }

    def __init__(self, require_confirmation: bool = True, max_slippage: float = 5.0):
        """
        Initialize swap tool.

        Args:
            require_confirmation: If True, requires explicit confirmation.
            max_slippage: Maximum allowed slippage percentage.
        """
        self._require_confirmation = require_confirmation
        self._max_slippage = max_slippage

    @property
    def name(self) -> str:
        return "swap"

    @property
    def description(self) -> str:
        return (
            "Swap tokens on decentralized exchanges (DEX). "
            "IMPORTANT: Requires confirmation parameter set to true. "
            "Swaps are irreversible and subject to slippage and fees."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "from_token": {
                    "type": "string",
                    "description": "Token to swap from: 'native' for chain token (ETH, MATIC, etc.) or ERC20 address",
                },
                "to_token": {
                    "type": "string",
                    "description": "Token to swap to: 'native' or ERC20 address",
                },
                "amount": {
                    "type": "string",
                    "description": "Amount of from_token to swap (in human-readable units)",
                },
                "chain": {
                    "type": "string",
                    "description": "Blockchain network (default: ethereum)",
                    "enum": list(self.SUPPORTED_CHAINS),
                },
                "slippage": {
                    "type": "number",
                    "description": "Maximum slippage tolerance in percentage (default: 0.5)",
                },
                "confirm": {
                    "type": "boolean",
                    "description": "REQUIRED: Must be true to execute. Safety check to prevent accidental swaps.",
                },
            },
            "required": ["from_token", "to_token", "amount", "confirm"],
        }

    def _get_rpc_url(self, chain: str) -> str | None:
        """Get RPC URL for the specified chain from environment."""
        return _get_env_rpc_url(chain)

    def _get_private_key(self) -> str | None:
        """Get private key from environment."""
        return _get_signing_private_key()

    async def execute(
        self,
        from_token: str,
        to_token: str,
        amount: str,
        chain: str = DEFAULT_WALLET_NETWORK,
        slippage: float = 0.5,
        confirm: bool = False,
        **kwargs: Any,
    ) -> str:
        """
        Execute DEX swap.

        Args:
            from_token: Token to swap from.
            to_token: Token to swap to.
            amount: Amount to swap.
            chain: Blockchain network.
            slippage: Slippage tolerance percentage.
            confirm: Must be True to execute.

        Returns:
            Swap result or error message.
        """
        # Safety check: require confirmation
        if self._require_confirmation and not confirm:
            dex_info = self.DEX_ROUTERS.get(chain, "Unknown DEX")
            return (
                "SAFETY CHECK FAILED: Swap not confirmed.\n\n"
                "This operation will swap real tokens and is IRREVERSIBLE.\n"
                "Please review the swap details:\n"
                f"  From: {amount} {from_token}\n"
                f"  To: {to_token}\n"
                f"  Chain: {chain}\n"
                f"  DEX: {dex_info}\n"
                f"  Max Slippage: {slippage}%\n\n"
                "WARNINGS:\n"
                "  - Actual received amount depends on market conditions\n"
                "  - DEX fees will be deducted\n"
                "  - Large swaps may have significant price impact\n\n"
                "To proceed, call this tool again with confirm=true."
            )

        # Validate chain
        if chain not in self.SUPPORTED_CHAINS:
            return f"Error: Unsupported chain '{chain}'. Supported: {', '.join(sorted(self.SUPPORTED_CHAINS))}"

        # Validate slippage
        if slippage < 0:
            return "Error: Slippage cannot be negative"
        if slippage > self._max_slippage:
            return f"Error: Slippage {slippage}% exceeds maximum allowed {self._max_slippage}%"

        # Check amount is valid
        try:
            amount_float = float(amount)
            if amount_float <= 0:
                return "Error: Amount must be greater than 0"
        except ValueError:
            return f"Error: Invalid amount '{amount}'. Must be a valid number."

        # Validate token addresses if not 'native'
        for token_name, token_addr in [("from_token", from_token), ("to_token", to_token)]:
            if token_addr != "native":
                if not token_addr.startswith("0x") or len(token_addr) != 42:
                    return f"Error: Invalid {token_name} address format. Expected 'native' or 0x followed by 40 hex characters."

        # Check RPC configuration
        rpc_url = self._get_rpc_url(chain)
        if not rpc_url:
            chain_env_var = f"{chain.upper()}_RPC_URL"
            return (
                f"Error: Web3 provider not configured for {chain}. "
                f"Set {chain_env_var} environment variable."
            )

        # Check private key
        private_key = self._get_private_key()
        if not private_key:
            return (
                "Error: Private key not configured. "
                "Set PRIVATE_KEY environment variable.\n\n"
                "WARNING: Never share your private key. Use a dedicated wallet for bot operations."
            )

        # Stub implementation
        dex_info = self.DEX_ROUTERS.get(chain, "Unknown DEX")
        return (
            f"[STUB] Swap would execute:\n"
            f"  From: {amount} {from_token}\n"
            f"  To: {to_token}\n"
            f"  Chain: {chain}\n"
            f"  DEX: {dex_info}\n"
            f"  Slippage: {slippage}%\n\n"
            f"To enable Web3 functionality, install web3.py and DEX SDK:\n"
            f"  pip install web3 uniswap-python\n\n"
            f"This is a stub - no actual swap was executed."
        )


class ContractCallTool(Tool):
    """
    Tool to make generic smart contract calls.

    Supports both read (view/pure) and write (state-changing) operations.
    SAFETY: Write operations require explicit confirmation.
    """

    SUPPORTED_CHAINS = SUPPORTED_WEB3_CHAINS

    def __init__(self, require_confirmation: bool = True):
        """
        Initialize contract call tool.

        Args:
            require_confirmation: If True, requires confirmation for write operations.
        """
        self._require_confirmation = require_confirmation

    @property
    def name(self) -> str:
        return "contract_call"

    @property
    def description(self) -> str:
        return (
            "Call a smart contract method. Supports read (view) and write operations. "
            "For write operations, confirmation is required and value can be sent. "
            "Requires ABI or method signature to be known."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "contract_address": {
                    "type": "string",
                    "description": "Smart contract address (0x...)",
                },
                "method": {
                    "type": "string",
                    "description": "Method name or signature (e.g., 'balanceOf' or 'balanceOf(address)')",
                },
                "args": {
                    "type": "array",
                    "items": {},
                    "description": "Arguments to pass to the method",
                },
                "chain": {
                    "type": "string",
                    "description": "Blockchain network (default: ethereum)",
                    "enum": list(self.SUPPORTED_CHAINS),
                },
                "value": {
                    "type": "string",
                    "description": "ETH/native token value to send with transaction (for payable methods)",
                },
                "is_write": {
                    "type": "boolean",
                    "description": "Set to true if this is a state-changing (write) operation",
                },
                "abi": {
                    "type": "string",
                    "description": "Contract ABI as JSON string (optional - will try to fetch from explorer)",
                },
                "confirm": {
                    "type": "boolean",
                    "description": "REQUIRED for write operations: Must be true to execute.",
                },
            },
            "required": ["contract_address", "method"],
        }

    def _get_rpc_url(self, chain: str) -> str | None:
        """Get RPC URL for the specified chain from environment."""
        return _get_env_rpc_url(chain)

    def _get_private_key(self) -> str | None:
        """Get private key from environment."""
        return _get_signing_private_key()

    async def execute(
        self,
        contract_address: str,
        method: str,
        args: list[Any] | None = None,
        chain: str = DEFAULT_WALLET_NETWORK,
        value: str | None = None,
        is_write: bool = False,
        abi: str | None = None,
        confirm: bool = False,
        **kwargs: Any,
    ) -> str:
        """
        Execute contract call.

        Args:
            contract_address: Contract address.
            method: Method name or signature.
            args: Method arguments.
            chain: Blockchain network.
            value: Native token value to send.
            is_write: True if state-changing operation.
            abi: Contract ABI JSON.
            confirm: Must be True for write operations.

        Returns:
            Call result or error message.
        """
        args = args or []

        # Validate chain
        if chain not in self.SUPPORTED_CHAINS:
            return f"Error: Unsupported chain '{chain}'. Supported: {', '.join(sorted(self.SUPPORTED_CHAINS))}"

        # Validate contract address
        if not contract_address.startswith("0x") or len(contract_address) != 42:
            return "Error: Invalid contract address format. Expected 0x followed by 40 hex characters."

        # Safety check for write operations
        if is_write and self._require_confirmation and not confirm:
            value_display = value if value else "0"
            return (
                "SAFETY CHECK FAILED: Write operation not confirmed.\n\n"
                "This operation will modify blockchain state and may be IRREVERSIBLE.\n"
                "Please review the transaction details:\n"
                f"  Contract: {contract_address}\n"
                f"  Method: {method}\n"
                f"  Arguments: {args}\n"
                f"  Value: {value_display}\n"
                f"  Chain: {chain}\n\n"
                "WARNINGS:\n"
                "  - This will submit a transaction to the blockchain\n"
                "  - Gas fees will be charged\n"
                "  - State changes may be irreversible\n\n"
                "To proceed, call this tool again with confirm=true after the user "
                "explicitly confirms the exact operation."
            )
        if is_write and self._require_confirmation and _env_enabled("SPOON_BOT_DISABLE_LIVE_TRANSFERS"):
            return _live_transfer_disabled_message(
                f"contract write {method} on {chain} at {contract_address}"
            )

        # Check RPC configuration
        rpc_url = self._get_rpc_url(chain)
        if not rpc_url:
            chain_env_var = f"{chain.upper()}_RPC_URL"
            return (
                f"Error: Web3 provider not configured for {chain}. "
                f"Set {chain_env_var} environment variable."
            )

        # For write operations, check private key
        if is_write:
            private_key = self._get_private_key()
            if not private_key:
                return (
                    "Error: Private key not configured for write operation. "
                    "Set PRIVATE_KEY environment variable.\n\n"
                    "WARNING: Never share your private key. Use a dedicated wallet for bot operations."
                )

        # Validate value if provided
        if value:
            try:
                value_float = float(value)
                if value_float < 0:
                    return "Error: Value cannot be negative"
            except ValueError:
                return f"Error: Invalid value '{value}'. Must be a valid number."

        try:
            from web3 import Web3
        except ImportError:
            return (
                "Error: Web3 provider not available. Install web3.py to enable blockchain tools.\n"
                "  pip install web3"
            )

        try:
            parsed_abi = json.loads(abi) if isinstance(abi, str) and abi.strip() else None
        except json.JSONDecodeError as exc:
            return f"Error: Invalid ABI JSON: {exc}"
        if not parsed_abi:
            return "Error: ABI is required for contract_call"

        try:
            web3 = Web3(Web3.HTTPProvider(rpc_url))
            if not web3.is_connected():
                return "Error: RPC connection failed"

            checksum_contract = web3.to_checksum_address(contract_address)
            contract = web3.eth.contract(address=checksum_contract, abi=parsed_abi)
            function_name = method.split("(", 1)[0].strip()
            if not function_name:
                return "Error: Method name is required"
            try:
                fn = contract.get_function_by_name(function_name)(*args)
            except Exception:
                fn = getattr(contract.functions, function_name)(*args)

            if not is_write:
                call_result = fn.call()
                return json.dumps(
                    {
                        "success": True,
                        "operation": "call",
                        "chain": chain,
                        "contract": checksum_contract,
                        "method": function_name,
                        "result": call_result,
                    },
                    ensure_ascii=True,
                    default=str,
                )

            private_key = self._get_private_key()
            if not private_key:
                return (
                    "Error: Private key not configured for write operation. "
                    "Set PRIVATE_KEY environment variable or initialize the built-in wallet."
                )
            account = Account.from_key(private_key)
            sender = web3.to_checksum_address(account.address)
            tx_value = _decimal_to_units(value, 18) if value else 0
            tx_params: dict[str, Any] = {
                "from": sender,
                "nonce": web3.eth.get_transaction_count(sender),
                "chainId": int(web3.eth.chain_id),
                "value": tx_value,
                **_fee_fields(web3),
            }
            tx = fn.build_transaction(tx_params)
            if "gas" not in tx:
                tx["gas"] = int(web3.eth.estimate_gas(tx))
            signed = web3.eth.account.sign_transaction(tx, private_key)
            tx_hash = web3.eth.send_raw_transaction(_signed_raw_transaction(signed))
            return json.dumps(
                {
                    "success": True,
                    "operation": "transaction",
                    "chain": chain,
                    "chain_id": int(web3.eth.chain_id),
                    "from": sender,
                    "contract": checksum_contract,
                    "method": function_name,
                    "tx_hash": _to_hex(web3, tx_hash),
                },
                ensure_ascii=True,
            )
        except Exception as e:
            return f"Error: Contract call failed: {e}"
