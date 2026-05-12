"""Privacy masking utilities for filtering sensitive data from logs and output."""

from __future__ import annotations

import re

# Env vars that must be scrubbed from subprocess environments to prevent
# leaking secrets into the LLM conversation via `env` / `printenv` output.
# Shared between privacy.py and shell.py so sensitive-var definitions are
# co-located.
SCRUBBED_ENV_VARS: frozenset[str] = frozenset({
    "PRIVATE_KEY",
    "SECRET_KEY",
    "MNEMONIC",
})

_SENSITIVE_ENV_NAME_RE = re.compile(
    r'(?:^|_)(?:API[_-]?KEY|ACCESS[_-]?KEY|AUTH[_-]?TOKEN|TOKEN|SECRET|PASSWORD|'
    r'PASSWD|MNEMONIC|PRIVATE[_-]?KEY|CREDENTIAL|PASSPHRASE)(?:_|$)',
    re.IGNORECASE,
)


def is_sensitive_env_var(name: str) -> bool:
    """Return True when an environment variable name likely contains a secret."""
    normalized = str(name or "").strip()
    return bool(normalized and _SENSITIVE_ENV_NAME_RE.search(normalized))

_SENSITIVE_VAR_RE = re.compile(
    r'^(\s*(?:export\s+)?'
    r'(?:\w*(?:PRIVATE_KEY|SECRET_KEY|SECRET|API_KEY|ACCESS_KEY|AUTH_TOKEN'
    r'|TOKEN|PASSWORD|MNEMONIC|CREDENTIAL|PASSPHRASE)\w*)'
    r'\s*=\s*)(.+)$',
    re.MULTILINE | re.IGNORECASE,
)

_BEARER_RE = re.compile(
    r'(Bearer\s+)[A-Za-z0-9\-._~+/]+=*',
    re.IGNORECASE,
)

_AUTHORIZATION_HEADER_RE = re.compile(
    r'((?:Authorization|X-Api-Key|X-Auth-Token)\s*:\s*)\S+',
    re.IGNORECASE,
)

_INLINE_KEY_RE = re.compile(
    r'((?:api[_-]?key|secret|token|password|auth)\s*[=:]\s*["\']?)([A-Za-z0-9\-._~+/]{20,})',
    re.IGNORECASE,
)

# Bare hex private keys: 0x + exactly 64 hex chars.
# EVM addresses are 0x + 40 hex, tx hashes are 0x + 64 hex.
# To distinguish private keys from tx hashes we mask 0x+64 hex only when NOT
# preceded by labels that indicate a hash/address context (tx, hash, receipt,
# transaction, block).  This is intentionally aggressive — a leaked private key
# in the LLM context is far worse than a masked tx hash (the agent can always
# re-fetch a hash).
_BARE_HEX_PRIVATE_KEY_RE = re.compile(
    r'(?<!\w)'           # not preceded by word char
    r'(?<!tx[:\s])'      # not preceded by "tx:" / "tx "
    r'(?<!hash[:\s])'    # not preceded by "hash:" / "hash "
    r'(0x[0-9a-fA-F]{64})'
    r'(?![0-9a-fA-F])',  # not followed by more hex (longer hash)
)

# Explicit allowlist of field names that indicate a tx/block hash, not a key.
# Uses word boundary (\b) so substrings like "myhash" don't bypass masking.
# Handles:
#   plain text  — "transaction hash: 0x..", "tx hash= 0x.."
#   env assign  — "TX_HASH=0x.."
#   JSON output — {"transactionHash":"0x.."}, {"blockHash":"0x.."}
_HASH_CONTEXT_FIELDS = (
    r'transactionHash'       # JSON receipt
    r'|transaction[_\s]*hash'  # transaction_hash, transaction hash
    r'|tx[_\s]*hash'          # tx_hash, tx hash
    r'|block[_\s]*hash'       # blockHash, block_hash
    r'|receipt[_\s]*hash'     # receiptHash
    r'|parent[_\s]*hash'      # parentHash
    r'|uncles[_\s]*hash'      # unclesHash
)
_HASH_CONTEXT_RE = re.compile(
    r'(?:^|["\'{(\s,])'                                           # word start or JSON/struct boundary
    r'(?:' + _HASH_CONTEXT_FIELDS + r')'
    r'[\s"\']*[=:]+[\s"\']*$',                                    # delimiter: =  :  ":"  = " etc.
    re.IGNORECASE,
)


def _mask_bare_hex_keys(text: str) -> str:
    """Replace bare 0x+64-hex strings that look like private keys."""
    def _replace(m: re.Match[str]) -> str:
        start = m.start(1)
        # Look back up to 40 chars for a hash-context label
        prefix = text[max(0, start - 40):start]
        if _HASH_CONTEXT_RE.search(prefix):
            return m.group(0)
        return "0x***masked_private_key***"
    return _BARE_HEX_PRIVATE_KEY_RE.sub(_replace, text)


def mask_secrets(text: str) -> str:
    """Mask sensitive values in text output.

    Targets: env var assignments with sensitive names, Bearer tokens,
    Authorization headers, inline key=value patterns, and bare hex strings
    that look like EVM private keys (0x + 64 hex chars).
    """
    text = _SENSITIVE_VAR_RE.sub(lambda m: f"{m.group(1)}***masked***", text)
    text = _BEARER_RE.sub(r'\1***masked***', text)
    text = _AUTHORIZATION_HEADER_RE.sub(r'\1***masked***', text)
    text = _INLINE_KEY_RE.sub(r'\1***masked***', text)
    text = _mask_bare_hex_keys(text)
    return text
