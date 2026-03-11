"""Privacy masking utilities for filtering sensitive data from logs and output."""

from __future__ import annotations

import re

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


def mask_secrets(text: str) -> str:
    """Mask sensitive values in text output.

    Targets: env var assignments with sensitive names, Bearer tokens,
    Authorization headers, inline key=value patterns.
    Does NOT mask general hex strings (addresses, tx hashes) as agents
    need those for on-chain interaction.
    """
    text = _SENSITIVE_VAR_RE.sub(lambda m: f"{m.group(1)}***masked***", text)
    text = _BEARER_RE.sub(r'\1***masked***', text)
    text = _AUTHORIZATION_HEADER_RE.sub(r'\1***masked***', text)
    text = _INLINE_KEY_RE.sub(r'\1***masked***', text)
    return text
