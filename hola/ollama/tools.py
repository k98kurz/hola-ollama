"""
Tool system for function registration and execution.
"""

import asyncio
import inspect
import logging
from types import UnionType
from typing import Any, Callable

from hola.ollama.types import (
    FunctionDefinition,
    ToolCall,
    ToolDefinition,
)

logger = logging.getLogger(__name__)


def _generate_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """Generate JSON schema from function type hints."""
    sig = inspect.signature(func)
    parameters = {}

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue

        param_type = param.annotation
        if param_type is inspect.Parameter.empty:
            param_type = Any

        param_schema = _type_to_schema(param_type)

        if param.default is not inspect.Parameter.empty:
            param_schema["default"] = param.default

        parameters[param_name] = param_schema

    required = [
        name
        for name, param in sig.parameters.items()
        if name != "self" and param.default is inspect.Parameter.empty
    ]

    return {
        "type": "object",
        "properties": parameters,
        "required": required,
    }


def _type_to_schema(type_hint: Any) -> dict[str, Any]:
    """Convert Python type hint to JSON Schema."""
    origin = getattr(type_hint, "__origin__", None)

    if origin is None:
        if type_hint is str:
            return {"type": "string"}
        if type_hint is int:
            return {"type": "integer"}
        if type_hint is float:
            return {"type": "number"}
        if type_hint is bool:
            return {"type": "boolean"}

        return {"type": "object"}

    if origin is list:
        item_type = getattr(type_hint, "__args__", (Any,))[0]
        return {
            "type": "array",
            "items": _type_to_schema(item_type),
        }

    if origin is dict:
        value_type = getattr(type_hint, "__args__", (Any, Any))[1]
        return {
            "type": "object",
            "additionalProperties": _type_to_schema(value_type),
        }

    if origin is UnionType:
        types = getattr(type_hint, "__args__", ())
        if type(None) in types:
            non_none_types = [t for t in types if t is not type(None)]
            if len(non_none_types) == 1:
                base_schema = _type_to_schema(non_none_types[0])
                base_schema["nullable"] = True
                return base_schema
            schemas = [_type_to_schema(t) for t in non_none_types]
            return {"anyOf": schemas, "nullable": True}

        schemas = [_type_to_schema(t) for t in types]
        return {"anyOf": schemas}

    return {"type": "object"}


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._functions: dict[str, Callable[..., Any]] = {}

    def register_tool(
        self,
        tool: ToolDefinition,
        func: Callable[..., Any],
    ) -> None:
        self._tools[tool.function.name] = tool
        self._functions[tool.function.name] = func

    def get_tool(
        self,
        name: str,
    ) -> ToolDefinition | None:
        return self._tools.get(name)

    def list_tools(
        self,
    ) -> list[ToolDefinition]:
        return list(self._tools.values())

    async def execute_tool_call(
        self,
        tool_call: ToolCall,
    ) -> Any:
        name = tool_call.function.name
        func = self._functions.get(name)

        if func is None:
            raise ValueError(f"Tool not found: {name}")

        args = tool_call.function.arguments

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(**args)
            else:
                result = func(**args)
        except Exception as e:
            logger.error(
                "Error executing tool %s: %s",
                name,
                e,
                exc_info=True,
            )
            raise

        logger.debug("Tool %s executed successfully", name)
        return result


_registry = ToolRegistry()


def tool(
    name: str,
    description: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to auto-register functions as tools."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        schema = _generate_schema(func)

        func_def = FunctionDefinition(
            name=name,
            description=description,
            parameters=schema,
        )

        tool_def = ToolDefinition(type="function", function=func_def)

        _registry.register_tool(tool_def, func)

        logger.debug("Registered tool: %s", name)
        return func

    return decorator


def register_tool(
    tool: ToolDefinition,
    func: Callable[..., Any],
) -> None:
    """Register tool in global registry."""
    return _registry.register_tool(tool, func)


def get_tool(name: str) -> ToolDefinition | None:
    """Get tool from global registry."""
    return _registry.get_tool(name)


def list_tools() -> list[ToolDefinition]:
    """List all tools from global registry."""
    return _registry.list_tools()


async def execute_tool_call(tool_call: ToolCall) -> Any:
    """Execute tool call using global registry."""
    return await _registry.execute_tool_call(tool_call)
