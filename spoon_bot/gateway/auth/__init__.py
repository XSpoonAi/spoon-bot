"""Authentication module."""

from spoon_bot.gateway.auth.jwt import (
    create_access_token,
    create_refresh_token,
    verify_token,
    TokenData,
)
from spoon_bot.gateway.auth.api_key import verify_api_key
from spoon_bot.gateway.auth.dependencies import (
    get_current_user,
    get_current_user_optional,
    require_scope,
)

__all__ = [
    "create_access_token",
    "create_refresh_token",
    "verify_token",
    "TokenData",
    "verify_api_key",
    "get_current_user",
    "get_current_user_optional",
    "require_scope",
]
