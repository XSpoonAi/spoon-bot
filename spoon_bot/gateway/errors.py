"""Gateway-specific error codes and error response builders."""

from __future__ import annotations

from enum import Enum
from typing import Any

from spoon_bot.gateway.models.responses import ErrorDetail, ErrorResponse, MetaInfo


class TimeoutCode(str, Enum):
    """Standardized timeout error codes."""

    TIMEOUT_UPSTREAM = "TIMEOUT_UPSTREAM"
    TIMEOUT_TOOL = "TIMEOUT_TOOL"
    TIMEOUT_TOTAL = "TIMEOUT_TOTAL"


class GatewayErrorCode(str, Enum):
    """All gateway error codes (extends existing codes)."""

    # Existing codes (for reference, not redefined)
    # AUTH_REQUIRED, AUTH_INVALID, AUTH_EXPIRED, FORBIDDEN, NOT_FOUND, etc.

    # New timeout codes
    TIMEOUT_UPSTREAM = "TIMEOUT_UPSTREAM"
    TIMEOUT_TOOL = "TIMEOUT_TOOL"
    TIMEOUT_TOTAL = "TIMEOUT_TOTAL"

    # Budget
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"

    # Cancellation
    CANCELLED = "CANCELLED"


def build_timeout_error_detail(
    timeout_code: TimeoutCode | str,
    elapsed_ms: int,
    limit_ms: int,
    context: str | None = None,
) -> ErrorDetail:
    """Build an ErrorDetail for a timeout.

    Args:
        timeout_code: The timeout error code.
        elapsed_ms: Elapsed time in ms when timeout occurred.
        limit_ms: The configured limit in ms.
        context: Optional context (e.g., tool name).

    Returns:
        ErrorDetail with standardized timeout info.
    """
    code = timeout_code if isinstance(timeout_code, str) else timeout_code.value

    messages = {
        "TIMEOUT_UPSTREAM": "Upstream service timed out",
        "TIMEOUT_TOOL": f"Tool execution timed out{f' ({context})' if context else ''}",
        "TIMEOUT_TOTAL": "Total request time budget exhausted",
    }

    return ErrorDetail(
        code=code,
        message=messages.get(code, f"Timeout: {code}"),
        details={
            "elapsed_ms": elapsed_ms,
            "limit_ms": limit_ms,
            **({"context": context} if context else {}),
        },
    )


def build_error_response(
    error_detail: ErrorDetail,
    request_id: str,
    trace_id: str | None = None,
    timing: dict[str, Any] | None = None,
) -> ErrorResponse:
    """Build a complete ErrorResponse with meta info.

    Args:
        error_detail: The error detail.
        request_id: Request ID.
        trace_id: Optional trace ID.
        timing: Optional timing info.

    Returns:
        Complete ErrorResponse.
    """
    return ErrorResponse(
        error=error_detail,
        meta=MetaInfo(
            request_id=request_id,
            trace_id=trace_id,
            timing=timing,
        ),
    )
