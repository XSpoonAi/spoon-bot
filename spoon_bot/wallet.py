"""Built-in EVM wallet bootstrap for spoon-bot.

The runtime layout intentionally matches the legacy ``clawcast-wallet`` files
so existing wallet-aware skills can keep working unchanged.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eth_account import Account
from eth_account.messages import encode_defunct


@dataclass(frozen=True)
class WalletNetwork:
    """Static metadata for a supported wallet network."""

    key: str
    name: str
    chain_id: int
    rpc_url: str
    explorer_url: str
    currency_symbol: str
    wrapped_native_symbol: str
    wrapped_native_address: str
    env_var: str


@dataclass(frozen=True)
class WalletRuntime:
    """Resolved runtime state for the built-in wallet."""

    address: str
    private_key: str
    keystore_path: Path
    password_file: Path
    state_file: Path
    wallet_root: Path
    network: WalletNetwork


_NETWORKS: dict[str, WalletNetwork] = {
    "neox": WalletNetwork(
        key="neox",
        name="Neo X Mainnet",
        chain_id=47763,
        rpc_url="https://mainnet-1.rpc.banelabs.org",
        explorer_url="https://xexplorer.neo.org",
        currency_symbol="GAS",
        wrapped_native_symbol="WGAS",
        wrapped_native_address="0xdE41591ED1f8ED1484aC2CD8ca0876428de60EfF",
        env_var="NEOX_RPC_URL",
    ),
    "neox_testnet": WalletNetwork(
        key="neox_testnet",
        name="Neo X Testnet T4",
        chain_id=12227332,
        rpc_url="https://neoxt4seed1.ngd.network",
        explorer_url="https://xt4scan.ngd.network",
        currency_symbol="GAS",
        wrapped_native_symbol="WGAS",
        wrapped_native_address="0x1CE16390FD09040486221e912B87551E4e44Ab17",
        env_var="NEOX_TESTNET_RPC_URL",
    ),
}

_NETWORK_ALIASES = {
    "neox_mainnet": "neox",
    "neo_x": "neox",
    "neo_x_mainnet": "neox",
    "t4": "neox_testnet",
    "neoxt4": "neox_testnet",
    "neo_x_testnet": "neox_testnet",
    "neox_t4": "neox_testnet",
}

DEFAULT_WALLET_NETWORK = "neox"


def _effective_home() -> Path:
    """Resolve user home robustly across native and mixed Windows environments."""
    home = Path.home()
    userprofile = os.environ.get("USERPROFILE", "").strip()
    home_posix = home.as_posix().lower()
    if userprofile and (home_posix == "/root" or home_posix.endswith("/root")):
        return Path(userprofile).expanduser()
    return home


def _wallet_root(_: Path | str | None = None) -> Path:
    """Return the legacy wallet location expected by existing skills."""
    override = os.environ.get("SPOON_BOT_WALLET_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return (_effective_home() / ".agent-wallet").resolve()


def _state_file(wallet_root: Path) -> Path:
    return wallet_root / "state.env"


def _keystore_file(wallet_root: Path) -> Path:
    return wallet_root / "keystore.json"


def _password_file(wallet_root: Path) -> Path:
    return wallet_root / "pw.txt"


def _private_key_file(wallet_root: Path) -> Path:
    return wallet_root / "privatekey.tmp"


def _wallet_files(wallet_root: Path) -> list[Path]:
    return [
        _state_file(wallet_root),
        _keystore_file(wallet_root),
        _password_file(wallet_root),
        _private_key_file(wallet_root),
    ]


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _quote_env(value: str) -> str:
    escaped = value.replace("'", "'\"'\"'")
    return f"'{escaped}'"


def _write_secret(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _sync_private_key_file(wallet_root: Path, private_key: str) -> Path:
    """Keep the legacy private-key compatibility file in sync with the wallet."""
    private_key_path = _private_key_file(wallet_root)
    _write_secret(private_key_path, private_key)
    return private_key_path


def _write_state_env(
    path: Path,
    *,
    wallet_root: Path,
    address: str,
    network: WalletNetwork,
    keystore_path: Path,
    password_path: Path,
) -> None:
    values = {
        "APP_DIR": str(wallet_root),
        "STATE_FILE": str(path),
        "ADDRESS": address,
        "KEYSTORE_FILE": str(keystore_path),
        "PASSWORD_FILE": str(password_path),
        "NETWORK": network.name,
        "NETWORK_KEY": network.key,
        "CHAIN_ID": str(network.chain_id),
        "ETH_RPC_URL": network.rpc_url,
    }
    content = "\n".join(f"{key}={_quote_env(value)}" for key, value in values.items()) + "\n"
    _write_secret(path, content)


def _parse_state_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key.strip()] = value
    return values


def resolve_wallet_network(name: str | None = None) -> WalletNetwork:
    """Resolve a configured wallet network with Neo X mainnet as default."""
    raw = (name or os.environ.get("SPOON_BOT_WALLET_NETWORK") or DEFAULT_WALLET_NETWORK).strip().lower()
    key = _NETWORK_ALIASES.get(raw, raw)
    network = _NETWORKS.get(key)
    if network is None:
        supported = ", ".join(sorted(_NETWORKS))
        raise ValueError(f"Unsupported wallet network '{raw}'. Supported: {supported}")
    return network


def _create_wallet(wallet_root: Path, network: WalletNetwork) -> WalletRuntime:
    wallet_root.mkdir(parents=True, exist_ok=True)

    account = Account.create(secrets.token_hex(32))
    private_key = account.key.hex()
    if not private_key.startswith("0x"):
        private_key = f"0x{private_key}"

    password = secrets.token_urlsafe(32)
    keystore = Account.encrypt(account.key, password)

    keystore_path = _keystore_file(wallet_root)
    password_path = _password_file(wallet_root)
    state_path = _state_file(wallet_root)

    _write_json(keystore_path, keystore)
    _write_secret(password_path, password)
    _sync_private_key_file(wallet_root, private_key)
    _write_state_env(
        state_path,
        wallet_root=wallet_root,
        address=account.address,
        network=network,
        keystore_path=keystore_path,
        password_path=password_path,
    )

    return WalletRuntime(
        address=account.address,
        private_key=private_key,
        keystore_path=keystore_path,
        password_file=password_path,
        state_file=state_path,
        wallet_root=wallet_root,
        network=network,
    )


def _load_existing_wallet(wallet_root: Path) -> WalletRuntime | None:
    state_path = _state_file(wallet_root)
    keystore_path = _keystore_file(wallet_root)
    password_path = _password_file(wallet_root)
    existing_paths = {
        state_path: state_path.exists(),
        keystore_path: keystore_path.exists(),
        password_path: password_path.exists(),
    }
    existing_count = sum(existing_paths.values())
    if existing_count == 0:
        return None
    if existing_count != len(existing_paths):
        missing = ", ".join(str(path.name) for path, exists in existing_paths.items() if not exists)
        present = ", ".join(str(path.name) for path, exists in existing_paths.items() if exists)
        raise RuntimeError(
            "Built-in wallet is in a partial state. "
            f"Present: {present or 'none'}. Missing: {missing or 'none'}."
        )

    state = _parse_state_env(state_path)
    requested_network = os.environ.get("SPOON_BOT_WALLET_NETWORK") or state.get("NETWORK_KEY") or DEFAULT_WALLET_NETWORK
    network = resolve_wallet_network(requested_network)
    password = password_path.read_text(encoding="utf-8").strip()
    keystore = json.loads(keystore_path.read_text(encoding="utf-8"))
    private_key_bytes = Account.decrypt(keystore, password)
    private_key = private_key_bytes.hex()
    if not private_key.startswith("0x"):
        private_key = f"0x{private_key}"
    address = str(state.get("ADDRESS") or Account.from_key(private_key).address)
    _sync_private_key_file(wallet_root, private_key)

    # Keep state.env aligned with the currently requested network.
    _write_state_env(
        state_path,
        wallet_root=wallet_root,
        address=address,
        network=network,
        keystore_path=keystore_path,
        password_path=password_path,
    )

    return WalletRuntime(
        address=address,
        private_key=private_key,
        keystore_path=keystore_path,
        password_file=password_path,
        state_file=state_path,
        wallet_root=wallet_root,
        network=network,
    )


def export_wallet_environment(runtime: WalletRuntime) -> None:
    """Expose wallet and Neo X network defaults to the process environment."""
    os.environ.setdefault("WALLET_ADDRESS", runtime.address)
    os.environ.setdefault("PRIVATE_KEY", runtime.private_key)
    os.environ["WALLET_NETWORK"] = runtime.network.key
    os.environ["WALLET_KEYSTORE_PATH"] = str(runtime.keystore_path)
    os.environ["WALLET_PASSWORD_FILE"] = str(runtime.password_file)
    os.environ["CHAIN_ID"] = str(runtime.network.chain_id)
    os.environ["AGENT_WALLET_DIR"] = str(runtime.wallet_root)

    for network in _NETWORKS.values():
        os.environ.setdefault(network.env_var, network.rpc_url)


def ensure_wallet_runtime(workspace: Path | str | None = None) -> WalletRuntime:
    """Create or load the built-in wallet and export Neo X defaults.

    The ``workspace`` parameter is accepted for API symmetry, but the canonical
    storage location is ``~/.agent-wallet`` to preserve compatibility with
    existing skills and CLIs.
    """
    wallet_root = _wallet_root(workspace)
    runtime = _load_existing_wallet(wallet_root)
    auto_created = False

    if runtime is None:
        auto_create = _bool_env("SPOON_BOT_WALLET_AUTO_CREATE", True)
        if not auto_create:
            raise RuntimeError("Built-in wallet is missing and SPOON_BOT_WALLET_AUTO_CREATE=false")
        runtime = _create_wallet(wallet_root, resolve_wallet_network())
        auto_created = True

    export_wallet_environment(runtime)
    os.environ["SPOON_BOT_WALLET_AUTO_CREATED"] = "1" if auto_created else "0"
    return runtime


def backup_wallet(
    workspace: Path | str | None = None,
    *,
    backup_root: Path | str | None = None,
) -> Path | None:
    """Copy the current wallet files into a timestamped backup directory.

    Returns ``None`` when no wallet files exist yet.
    """
    wallet_root = _wallet_root(workspace)
    existing_files = [path for path in _wallet_files(wallet_root) if path.exists()]
    if not existing_files:
        return None

    root = (
        Path(backup_root).expanduser().resolve()
        if backup_root is not None
        else wallet_root / "backups"
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = root / stamp
    suffix = 1
    while backup_dir.exists():
        backup_dir = root / f"{stamp}-{suffix}"
        suffix += 1

    backup_dir.mkdir(parents=True, exist_ok=False)
    for source in existing_files:
        shutil.copy2(source, backup_dir / source.name)

    metadata = {
        "backed_up_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source": str(wallet_root),
        "files": [path.name for path in existing_files],
    }
    _write_json(backup_dir / "metadata.json", metadata)
    try:
        backup_dir.chmod(0o700)
    except OSError:
        pass
    return backup_dir


def create_new_wallet(
    workspace: Path | str | None = None,
    *,
    network: str | None = None,
    backup_existing: bool = True,
) -> tuple[WalletRuntime, Path | None]:
    """Create a fresh wallet, backing up existing wallet files first."""
    wallet_root = _wallet_root(workspace)
    backup_dir = backup_wallet(workspace) if backup_existing else None
    runtime = _create_wallet(wallet_root, resolve_wallet_network(network))
    export_wallet_environment(runtime)
    os.environ["SPOON_BOT_WALLET_AUTO_CREATED"] = "1"
    return runtime, backup_dir


def _hex(value: Any) -> str:
    raw = value.hex() if hasattr(value, "hex") else str(value)
    return raw if raw.startswith("0x") or raw == "" else f"0x{raw}"


def _signed_payload(runtime: WalletRuntime, signed: Any) -> dict[str, Any]:
    return {
        "address": runtime.address,
        "signature": _hex(signed.signature),
        "message_hash": _hex(getattr(signed, "message_hash", "")),
        "v": int(signed.v),
        "r": hex(int(signed.r)),
        "s": hex(int(signed.s)),
    }


def sign_wallet_message(
    *,
    message: str | None = None,
    hexstr: str | None = None,
    workspace: Path | str | None = None,
) -> dict[str, Any]:
    """Sign an EIP-191 compatible text or hex message with the built-in wallet."""
    supplied = [value is not None and str(value) != "" for value in (message, hexstr)]
    if sum(supplied) != 1:
        raise ValueError("Provide exactly one of message or hexstr")

    runtime = ensure_wallet_runtime(workspace)
    signable = (
        encode_defunct(hexstr=hexstr)
        if hexstr is not None
        else encode_defunct(text=str(message))
    )
    signed = Account.sign_message(signable, private_key=runtime.private_key)
    payload = _signed_payload(runtime, signed)
    payload["kind"] = "eip191"
    return payload


def sign_wallet_typed_data(
    *,
    full_message: dict[str, Any] | None = None,
    domain_data: dict[str, Any] | None = None,
    message_types: dict[str, Any] | None = None,
    message_data: dict[str, Any] | None = None,
    workspace: Path | str | None = None,
) -> dict[str, Any]:
    """Sign EIP-712 typed data with the built-in wallet."""
    runtime = ensure_wallet_runtime(workspace)
    if full_message is not None:
        signed = Account.sign_typed_data(runtime.private_key, full_message=full_message)
    else:
        signed = Account.sign_typed_data(
            runtime.private_key,
            domain_data=domain_data,
            message_types=message_types,
            message_data=message_data,
        )
    payload = _signed_payload(runtime, signed)
    payload["kind"] = "eip712"
    return payload


def sign_wallet_transaction(
    transaction: dict[str, Any],
    *,
    workspace: Path | str | None = None,
) -> dict[str, Any]:
    """Sign a prepared EVM transaction without broadcasting it."""
    if not isinstance(transaction, dict):
        raise TypeError("transaction must be an object")

    runtime = ensure_wallet_runtime(workspace)
    tx = dict(transaction)
    tx.setdefault("from", Account.from_key(runtime.private_key).address)
    signed = Account.sign_transaction(tx, runtime.private_key)
    raw_transaction = getattr(signed, "raw_transaction", None) or getattr(signed, "rawTransaction", None)
    return {
        "address": runtime.address,
        "transaction_hash": _hex(signed.hash),
        "raw_transaction": _hex(raw_transaction),
    }


def wallet_summary(runtime: WalletRuntime) -> dict[str, Any]:
    """Return a serializable wallet summary for logs/tests."""
    payload = asdict(runtime)
    payload["keystore_path"] = str(runtime.keystore_path)
    payload["password_file"] = str(runtime.password_file)
    payload["state_file"] = str(runtime.state_file)
    payload["wallet_root"] = str(runtime.wallet_root)
    payload["network"] = asdict(runtime.network)
    return payload
