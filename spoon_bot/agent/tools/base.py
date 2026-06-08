"""Base class for agent tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypedDict

from spoon_bot.agent.tools.execution_context import (
    bind_tool_invocation,
    cancelled_tool_run_blocker,
    classify_tool_invocation_category,
    finalize_tool_invocation,
    get_tool_owner,
    mark_current_tool_invocation_guardrail,
    mark_current_tool_invocation_progress_recorded,
    read_only_skill_inspection_budget_blocker,
    record_tool_invocation_result,
    suppress_after_consecutive_tool_failures,
    suppress_repeated_tool_invocation,
    suppress_repeated_tool_series,
)
from spoon_bot.agent.execution_ledger import record_tool_capture_in_ledger

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired


class ToolPropertySchema(TypedDict, total=False):
    """Schema for a single tool property."""

    type: str
    description: str
    enum: list[str]
    default: Any
    minimum: int | float
    maximum: int | float
    minLength: int
    maxLength: int
    pattern: str
    items: dict[str, Any]


class ToolParameterSchema(TypedDict):
    """JSON Schema for tool parameters following OpenAI format."""

    type: str  # Always "object" for tool parameters
    properties: dict[str, ToolPropertySchema]
    required: NotRequired[list[str]]


class ToolFunctionSchema(TypedDict):
    """Schema for the function definition."""

    name: str
    description: str
    parameters: ToolParameterSchema


class ToolSchema(TypedDict):
    """Complete tool schema in OpenAI function format."""

    type: str  # Always "function"
    function: ToolFunctionSchema


class Tool(ABC):
    """
    Abstract base class for agent tools.

    Tools are capabilities that the agent can use to interact with
    the environment, such as reading files, executing commands, etc.

    Native OS tools are always available as priority.
    MCP and toolkit tools can be loaded dynamically.

    Subclasses must implement:
    - name: Unique tool identifier (used in function calls)
    - description: Clear description of what the tool does
    - parameters: JSON Schema defining expected parameters
    - execute: Async method that performs the tool's action

    Example:
        class MyTool(Tool):
            @property
            def name(self) -> str:
                return "my_tool"

            @property
            def description(self) -> str:
                return "Does something useful"

            @property
            def parameters(self) -> ToolParameterSchema:
                return {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string", "description": "Input value"}
                    },
                    "required": ["input"]
                }

            async def execute(self, input: str, **kwargs: Any) -> str:
                return f"Processed: {input}"
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Tool name used in function calls.

        Must be unique across all registered tools.
        Should be lowercase with underscores (e.g., 'read_file').
        """
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Description of what the tool does.

        Should be clear and concise, as it's used by the LLM
        to decide when to use this tool.
        """
        ...

    @property
    @abstractmethod
    def parameters(self) -> ToolParameterSchema:
        """
        JSON Schema for tool parameters.

        Must follow the OpenAI function calling format:
        {
            "type": "object",
            "properties": {...},
            "required": [...]
        }
        """
        ...

    _path_touch_callback: Any = None

    @staticmethod
    def _canonical_parameter_key(key: Any) -> str:
        """Return a casing-insensitive comparison key for tool parameters."""
        return "".join(
            char
            for char in str(key or "")
            if char not in {"_", "-", " "}
        ).casefold()

    def _normalize_invocation_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Normalize common model-emitted parameter casing to schema keys.

        Providers sometimes emit ``filePath`` for a tool whose schema declares
        ``path``. Keep this data-driven from the tool schema so individual tool
        loops do not need route-specific repairs.
        """
        if not isinstance(kwargs, dict) or not kwargs:
            return kwargs

        try:
            schema_properties = dict(self.parameters.get("properties") or {})
        except Exception:
            schema_properties = {}
        if not schema_properties:
            return kwargs

        canonical_to_schema_key = {
            self._canonical_parameter_key(key): str(key)
            for key in schema_properties.keys()
        }
        alias_to_schema_key = {
            "filepath": "path",
            "filename": "path",
            "file": "path",
            "oldstring": "old_text",
            "oldstr": "old_text",
            "old": "old_text",
            "find": "old_text",
            "search": "old_text",
            "newstring": "new_text",
            "newstr": "new_text",
            "new": "new_text",
            "replace": "new_text",
            "replacement": "new_text",
        }
        normalized = dict(kwargs)
        for raw_key, value in list(kwargs.items()):
            if raw_key in schema_properties:
                continue
            canonical = self._canonical_parameter_key(raw_key)
            schema_key = canonical_to_schema_key.get(canonical)
            if not schema_key:
                candidate = alias_to_schema_key.get(canonical)
                if candidate in schema_properties:
                    schema_key = candidate
            if schema_key and schema_key not in normalized:
                normalized[schema_key] = value
        return normalized

    def runtime_invocation_category(self, kwargs: dict[str, Any]) -> str | None:
        """Return request-flow category for this invocation when known.

        Tool subclasses may return ``read_only``, ``setup``, or ``stateful``.
        The default uses shared tool metadata and the registered tool name.
        """
        return classify_tool_invocation_category(self.name, kwargs)

    async def __call__(self, *args: Any, **kwargs: Any) -> str:
        """
        Make the tool callable - compatibility with spoon-core SDK.

        This allows the tool to be used as: `result = await tool(arg=value)`
        """
        result: Any = None
        executed = False
        kwargs = self._normalize_invocation_kwargs(kwargs)
        invocation_category = self.runtime_invocation_category(kwargs)
        with bind_tool_invocation(self.name, kwargs) as invocation_id:
            try:
                cancelled_result = cancelled_tool_run_blocker()
                if cancelled_result is not None:
                    result = cancelled_result
                    mark_current_tool_invocation_guardrail(
                        "request_cancelled",
                        message=cancelled_result,
                    )
                    return result
                failure_loop_result = suppress_after_consecutive_tool_failures(self.name)
                if failure_loop_result is not None:
                    result = failure_loop_result
                    return result
                inspection_budget_result = read_only_skill_inspection_budget_blocker(
                    self.name,
                    kwargs,
                    invocation_category=invocation_category,
                )
                if inspection_budget_result is not None:
                    result = inspection_budget_result
                    mark_current_tool_invocation_guardrail(
                        "read_only_skill_inspection_budget",
                        message=inspection_budget_result,
                    )
                    return result
                dedup_key_func = getattr(self, "tool_invocation_dedup_key", None)
                dedup_arguments = dedup_key_func(kwargs) if callable(dedup_key_func) else kwargs
                if dedup_arguments is not None:
                    duplicate_result = suppress_repeated_tool_invocation(
                        self.name,
                        dedup_arguments,
                    )
                    if duplicate_result is not None:
                        formatter = getattr(
                            self,
                            "format_duplicate_invocation_result",
                            None,
                        )
                        if callable(formatter):
                            duplicate_result = formatter(
                                duplicate_result,
                                kwargs,
                                dedup_arguments,
                            )
                        result = duplicate_result
                        return result
                series_key_func = getattr(self, "tool_invocation_series_key", None)
                if callable(series_key_func):
                    series_result = suppress_repeated_tool_series(
                        self.name,
                        series_key_func(kwargs),
                    )
                    if series_result is not None:
                        result = series_result
                        return result
                executed = True
                result = await self.execute(**kwargs)
            finally:
                progress_hint = record_tool_invocation_result(
                    self.name,
                    result,
                    arguments=kwargs,
                    invocation_category=invocation_category,
                )
                if progress_hint and isinstance(result, str):
                    if progress_hint not in result:
                        result = result.rstrip() + "\n\n" + progress_hint
                mark_current_tool_invocation_progress_recorded()
                finalize_tool_invocation(result)
                if invocation_id is None:
                    try:
                        record_tool_capture_in_ledger(
                            owner=get_tool_owner(),
                            tool_name=self.name,
                            arguments=kwargs,
                            summary_output=result,
                            full_output=result,
                            category=invocation_category,
                        )
                    except Exception:
                        pass

        if executed and self._path_touch_callback is not None:
            _path = kwargs.get("path") or kwargs.get("file_path") or kwargs.get("directory")
            if _path:
                try:
                    self._path_touch_callback(_path)
                except Exception:
                    pass

        return result

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """
        Execute the tool with given parameters.

        Args:
            **kwargs: Tool-specific parameters matching the schema.

        Returns:
            String result of the tool execution.
            Should return error messages as strings (not raise exceptions)
            to allow the agent to recover gracefully.
        """
        ...

    def validate_parameters(self, **kwargs: Any) -> list[str]:
        """
        Validate parameters against the schema.

        Args:
            **kwargs: Parameters to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: list[str] = []
        schema = self.parameters
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Check required parameters
        for param in required:
            if param not in kwargs or kwargs[param] is None:
                errors.append(f"Missing required parameter: {param}")

        # Validate parameter types (basic validation)
        for param, value in kwargs.items():
            if param in properties:
                prop_schema = properties[param]
                expected_type = prop_schema.get("type")

                if expected_type and value is not None:
                    type_valid = self._check_type(value, expected_type)
                    if not type_valid:
                        errors.append(
                            f"Parameter '{param}' has wrong type: "
                            f"expected {expected_type}, got {type(value).__name__}"
                        )

        return errors

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected JSON Schema type."""
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        expected = type_map.get(expected_type)
        if expected is None:
            return True  # Unknown type, assume valid
        return isinstance(value, expected)

    def to_schema(self) -> ToolSchema:
        """
        Convert tool to OpenAI function schema format.

        Returns:
            Complete tool schema suitable for LLM function calling.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

    def to_param(self) -> dict:
        """
        Alias for to_schema() - compatibility with spoon-core SDK.

        Returns:
            Tool definition in OpenAI function format.
        """
        return self.to_schema()

    def __repr__(self) -> str:
        return f"<Tool {self.name}>"

    def __str__(self) -> str:
        return f"Tool({self.name}: {self.description[:50]}...)"
