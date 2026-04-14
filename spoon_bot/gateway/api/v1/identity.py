"""Agent on-chain identity endpoint."""

from __future__ import annotations

import os

from fastapi import APIRouter

router = APIRouter()


def _query_identity(wallet_address: str) -> dict:
    """Query on-chain identity — delegates to SpoonCoreIdentity."""
    from spoon_bot.gateway.core_integration import SpoonCoreIdentity

    return SpoonCoreIdentity.query_identity(wallet_address)


def format_identity_text(wallet_address: str | None = None) -> str:
    """Return a human-readable identity summary (used by /myid commands)."""
    addr = wallet_address or os.environ.get("WALLET_ADDRESS")
    if not addr:
        return "No wallet configured for this agent."

    info = _query_identity(addr)
    if info.get("error"):
        return (
            f"Wallet: {addr[:8]}...{addr[-6:]}\n"
            f"Identity lookup failed: {info['error']}"
        )
    if info["registered"]:
        return (
            f"AgentID: {info['agent_id']}\n"
            f"Wallet: {addr[:8]}...{addr[-6:]}\n"
            f"Chain: {info['chain']} ({info['chain_id']})\n"
            f"Status: Registered"
        )
    return (
        f"Wallet: {addr[:8]}...{addr[-6:]}\n"
        f"Chain: {info['chain']} ({info['chain_id']})\n"
        f"Status: Not registered"
    )


@router.get("/identity")
async def get_identity() -> dict:
    """Return the current agent's on-chain identity.

    Uses the built-in wallet address from ``WALLET_ADDRESS`` env var.
    """
    wallet_address = os.environ.get("WALLET_ADDRESS")
    if not wallet_address:
        return {"error": "No wallet configured", "registered": False}
    return _query_identity(wallet_address)


@router.get("/identity/{address}")
async def get_identity_for_address(address: str) -> dict:
    """Look up on-chain identity for an arbitrary EVM address."""
    return _query_identity(address)
