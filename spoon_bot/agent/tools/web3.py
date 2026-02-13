"""Web3 blockchain tools: balance check, transfer, swap, contract calls."""

from __future__ import annotations

import os
from typing import Any

from spoon_bot.agent.tools.base import Tool


class BalanceCheckTool(Tool):
    """
    Tool to check wallet balance across multiple chains.

    Supports checking native token balances and ERC20 token balances.
    Requires RPC_URL environment variable or provider configuration.
    """

    # Supported chains and their default RPC endpoints (env var overrides)
    SUPPORTED_CHAINS = frozenset({
        "ethereum", "polygon", "arbitrum", "optimism", "base",
        "bsc", "avalanche", "fantom", "sepolia", "goerli"
    })

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
            "Supports multiple chains: ethereum, polygon, arbitrum, optimism, base, bsc, avalanche."
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
        # Check chain-specific env var first
        chain_env_var = f"{chain.upper()}_RPC_URL"
        rpc_url = os.environ.get(chain_env_var)

        if not rpc_url and chain == "ethereum":
            rpc_url = os.environ.get("RPC_URL") or self._default_rpc_url

        return rpc_url

    def _get_default_address(self) -> str | None:
        """Get default wallet address from environment."""
        return os.environ.get("WALLET_ADDRESS")

    async def execute(
        self,
        address: str | None = None,
        chain: str = "ethereum",
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

        # Stub implementation - would use web3.py in real implementation
        return (
            f"[STUB] Balance check would execute:\n"
            f"  Chain: {chain}\n"
            f"  Address: {wallet_address}\n"
            f"  RPC: {rpc_url[:30]}...\n"
            f"  Tokens: {tokens or ['native']}\n\n"
            f"To enable Web3 functionality, install web3.py:\n"
            f"  pip install web3\n\n"
            f"And configure the BalanceCheckTool with a real implementation."
        )


class TransferTool(Tool):
    """
    Tool to transfer native tokens or ERC20 tokens.

    SAFETY: Requires explicit confirmation before executing transfers.
    This tool handles real value and transactions are irreversible.
    """

    SUPPORTED_CHAINS = frozenset({
        "ethereum", "polygon", "arbitrum", "optimism", "base",
        "bsc", "avalanche", "fantom", "sepolia", "goerli"
    })

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
        chain_env_var = f"{chain.upper()}_RPC_URL"
        rpc_url = os.environ.get(chain_env_var)
        if not rpc_url and chain == "ethereum":
            rpc_url = os.environ.get("RPC_URL")
        return rpc_url

    def _get_private_key(self) -> str | None:
        """Get private key from environment."""
        return os.environ.get("PRIVATE_KEY")

    async def execute(
        self,
        to_address: str,
        amount: str,
        token: str = "native",
        chain: str = "ethereum",
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
                "To proceed, call this tool again with confirm=true."
            )

        # Validate chain
        if chain not in self.SUPPORTED_CHAINS:
            return f"Error: Unsupported chain '{chain}'. Supported: {', '.join(sorted(self.SUPPORTED_CHAINS))}"

        # Validate address format (basic check)
        if not to_address.startswith("0x") or len(to_address) != 42:
            return f"Error: Invalid address format. Expected 0x followed by 40 hex characters."

        # Check amount is valid number
        try:
            amount_float = float(amount)
            if amount_float <= 0:
                return "Error: Amount must be greater than 0"
        except ValueError:
            return f"Error: Invalid amount '{amount}'. Must be a valid number."

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
        return (
            f"[STUB] Transfer would execute:\n"
            f"  To: {to_address}\n"
            f"  Amount: {amount}\n"
            f"  Token: {token}\n"
            f"  Chain: {chain}\n\n"
            f"To enable Web3 functionality, install web3.py:\n"
            f"  pip install web3\n\n"
            f"This is a stub - no actual transaction was sent."
        )


class SwapTool(Tool):
    """
    Tool to perform DEX swaps.

    SAFETY: Requires explicit confirmation before executing swaps.
    Swaps involve real value and slippage - verify all parameters carefully.
    """

    SUPPORTED_CHAINS = frozenset({
        "ethereum", "polygon", "arbitrum", "optimism", "base",
        "bsc", "avalanche"
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
        chain_env_var = f"{chain.upper()}_RPC_URL"
        rpc_url = os.environ.get(chain_env_var)
        if not rpc_url and chain == "ethereum":
            rpc_url = os.environ.get("RPC_URL")
        return rpc_url

    def _get_private_key(self) -> str | None:
        """Get private key from environment."""
        return os.environ.get("PRIVATE_KEY")

    async def execute(
        self,
        from_token: str,
        to_token: str,
        amount: str,
        chain: str = "ethereum",
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

    SUPPORTED_CHAINS = frozenset({
        "ethereum", "polygon", "arbitrum", "optimism", "base",
        "bsc", "avalanche", "fantom", "sepolia", "goerli"
    })

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
        chain_env_var = f"{chain.upper()}_RPC_URL"
        rpc_url = os.environ.get(chain_env_var)
        if not rpc_url and chain == "ethereum":
            rpc_url = os.environ.get("RPC_URL")
        return rpc_url

    def _get_private_key(self) -> str | None:
        """Get private key from environment."""
        return os.environ.get("PRIVATE_KEY")

    async def execute(
        self,
        contract_address: str,
        method: str,
        args: list[Any] | None = None,
        chain: str = "ethereum",
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
                "To proceed, call this tool again with confirm=true."
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

        # Stub implementation
        operation_type = "Write (Transaction)" if is_write else "Read (Call)"
        return (
            f"[STUB] Contract {operation_type.lower()} would execute:\n"
            f"  Contract: {contract_address}\n"
            f"  Method: {method}\n"
            f"  Arguments: {args}\n"
            f"  Chain: {chain}\n"
            f"  Value: {value or '0'}\n"
            f"  Operation: {operation_type}\n\n"
            f"To enable Web3 functionality, install web3.py:\n"
            f"  pip install web3\n\n"
            f"This is a stub - no actual contract call was made."
        )
