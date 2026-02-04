"""Base class for agent tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypedDict, NotRequired


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

    def __repr__(self) -> str:
        return f"<Tool {self.name}>"

    def __str__(self) -> str:
        return f"Tool({self.name}: {self.description[:50]}...)"
