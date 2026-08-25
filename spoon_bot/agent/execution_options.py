"""Request-scoped model execution options."""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Any, AsyncGenerator, Callable, Iterator, TypeVar

_MAX_CONTEXT_WINDOW = 10_000_000


@dataclass(frozen=True, slots=True)
class AgentExecutionOptions:
    """Immutable model metadata for one agent turn."""

    model: str
    context_window: int
    model_sku: str | None = None
    model_catalog_version: str | None = None
    explicit: bool = False

    @classmethod
    def resolve(
        cls,
        *,
        default_model: str | None,
        default_context_window: int | None,
        model: str | None = None,
        context_window: int | None = None,
        model_sku: str | None = None,
        model_catalog_version: str | None = None,
    ) -> "AgentExecutionOptions":
        inherited = current_agent_execution_options()
        requested_model = _optional_string(model, "model")
        effective_model = (
            requested_model
            or (inherited.model if inherited is not None else "")
            or str(default_model or "").strip()
        )
        if not effective_model:
            raise ValueError("A default or request model is required")

        if context_window is None:
            effective_context_window = (
                inherited.context_window
                if inherited is not None
                else _positive_int(default_context_window, "default_context_window")
            )
        else:
            effective_context_window = _positive_int(context_window, "context_window")

        return cls(
            model=effective_model,
            context_window=effective_context_window,
            model_sku=(
                _optional_string(model_sku, "model_sku")
                or (inherited.model_sku if inherited is not None else None)
            ),
            model_catalog_version=(
                _optional_string(model_catalog_version, "model_catalog_version")
                or (inherited.model_catalog_version if inherited is not None else None)
            ),
            explicit=bool(requested_model)
            or bool(inherited is not None and inherited.explicit),
        )


_CURRENT_EXECUTION_OPTIONS: ContextVar[AgentExecutionOptions | None] = ContextVar(
    "spoon_bot_agent_execution_options",
    default=None,
)


def _optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if len(normalized) > 256:
        raise ValueError(f"{field_name} must be at most 256 characters")
    return normalized


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if normalized <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    if normalized > _MAX_CONTEXT_WINDOW:
        raise ValueError(
            f"{field_name} must not exceed {_MAX_CONTEXT_WINDOW:,}"
        )
    return normalized


def current_agent_execution_options() -> AgentExecutionOptions | None:
    return _CURRENT_EXECUTION_OPTIONS.get()


@contextmanager
def bind_agent_execution_options(
    options: AgentExecutionOptions,
) -> Iterator[AgentExecutionOptions]:
    token = _CURRENT_EXECUTION_OPTIONS.set(options)
    try:
        yield options
    finally:
        _CURRENT_EXECUTION_OPTIONS.reset(token)


_F = TypeVar("_F", bound=Callable[..., Any])


def request_model_execution(func: _F) -> _F:
    """Bind model kwargs for an AgentLoop async method or async generator."""

    def _resolve(
        self: Any,
        kwargs: dict[str, Any],
    ) -> AgentExecutionOptions | None:
        if (
            current_agent_execution_options() is None
            and not kwargs.get("model")
            and not getattr(self, "model", None)
        ):
            return None
        return AgentExecutionOptions.resolve(
            default_model=getattr(self, "model", None),
            default_context_window=getattr(self, "context_window", None),
            model=kwargs.get("model"),
            context_window=kwargs.get("context_window"),
            model_sku=kwargs.get("model_sku"),
            model_catalog_version=kwargs.get("model_catalog_version"),
        )

    if inspect.isasyncgenfunction(func):

        @wraps(func)
        async def async_generator_wrapper(
            self: Any,
            *args: Any,
            **kwargs: Any,
        ) -> AsyncGenerator[Any, None]:
            options = _resolve(self, kwargs)
            if options is None:
                async for item in func(self, *args, **kwargs):
                    yield item
                return
            iterator = func(self, *args, **kwargs)
            try:
                while True:
                    with bind_agent_execution_options(options):
                        try:
                            item = await anext(iterator)
                        except StopAsyncIteration:
                            return
                    yield item
            finally:
                with bind_agent_execution_options(options):
                    await iterator.aclose()

        return async_generator_wrapper  # type: ignore[return-value]

    @wraps(func)
    async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        options = _resolve(self, kwargs)
        if options is None:
            return await func(self, *args, **kwargs)
        with bind_agent_execution_options(options):
            return await func(self, *args, **kwargs)

    return async_wrapper  # type: ignore[return-value]
