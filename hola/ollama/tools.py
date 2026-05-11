"""
Tool system for function registration and execution.
"""

from types import UnionType
from typing import Any, Callable
from hola.ollama.exceptions import ToolExecutionError
from hola.ollama.types import (
    FunctionDefinition,
    ToolCall,
    ToolCallFunction,
    ToolDefinition,
)
import asyncio
import inspect
import logging


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


def _validate_args(
        args: dict[str, Any],
        schema: dict[str, Any],
    ) -> None:
    """Basic type checking for tool arguments against schema."""
    properties = schema.get("properties", {})

    for key, value in args.items():
        if key not in properties:
            logger.warning(
                "Unknown argument %s provided for tool execution",
                key,
            )
            continue

        param_schema = properties[key]
        expected_type = param_schema.get("type")

        if expected_type is None:
            continue

        if expected_type == "string" and not isinstance(value, str):
            raise ToolExecutionError(
                f"Invalid type for argument `{key}`: expected "
                f"string, got {type(value).__name__}"
            )

        if expected_type == "integer" and not isinstance(value, int):
            raise ToolExecutionError(
                f"Invalid type for argument `{key}`: expected "
                f"integer, got {type(value).__name__}"
            )

        if expected_type == "number" and not isinstance(value, (int, float)):
            raise ToolExecutionError(
                f"Invalid type for argument `{key}`: expected "
                f"number, got {type(value).__name__}"
            )

        if expected_type == "boolean" and not isinstance(value, bool):
            raise ToolExecutionError(
                f"Invalid type for argument `{key}`: expected "
                f"boolean, got {type(value).__name__}"
            )


def _type_to_schema(type_hint: Any) -> dict[str, Any]:
    """Convert Python type hint to JSON Schema."""
    if isinstance(type_hint, UnionType):
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

    return {"type": "object"}


def _convert_type(
        value: Any,
        target_type: type,
    ) -> Any:
    """Convert value to target type if possible."""
    if target_type is str:
        return str(value)

    if target_type is int:
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.split('.')[0])
            except ValueError:
                raise ToolExecutionError(
                    f"Cannot convert {value!r} to int"
                )
        raise ToolExecutionError(
            f"Cannot convert {type(value).__name__} to int"
        )

    if target_type is float:
        if isinstance(value, (int, float, str)):
            try:
                return float(value)
            except (ValueError, TypeError):
                raise ToolExecutionError(
                    f"Cannot convert {value!r} to float"
                )
        raise ToolExecutionError(
            f"Cannot convert {type(value).__name__} to float"
        )

    if target_type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        raise ToolExecutionError(
            f"Cannot convert {type(value).__name__} to bool"
        )

    if target_type is list:
        if isinstance(value, list):
            return value
        return [value]

    if target_type is dict:
        if isinstance(value, dict):
            return value
        raise ToolExecutionError(
            f"Cannot convert {type(value).__name__} to dict"
        )

    return value


def _resolve_execution_order(
        tool_call: ToolCall,
    ) -> list[ToolCall]:
    """Resolve execution order for nested tool calls using depth-first traversal."""
    order = []

    def _traverse(call: ToolCall) -> None:
        if call.nested_calls:
            for nested in call.nested_calls:
                _traverse(nested)
        order.append(call)

    _traverse(tool_call)
    return order


def _check_list_parameter_compatibility(
        tool_def: ToolDefinition,
        num_children: int,
    ) -> bool:
    """Check if tool can accept a list parameter for nested results."""
    params = tool_def.function.parameters.get("properties", {})

    data_param = params.get("data")
    if data_param and data_param.get("type") == "array":
        return True

    for param_name, param_schema in params.items():
        if param_schema.get("type") == "array":
            return True

    return False


async def _execute_nested_tool_call(
        tool_call: ToolCall,
        registry: "ToolRegistry",
    ) -> Any:
    """Execute a nested tool call with its children first."""
    if tool_call.nested_calls:
        child_results = []

        for nested_call in tool_call.nested_calls:
            try:
                result = await _execute_nested_tool_call(
                    nested_call,
                    registry,
                )
                child_results.append(result)
            except Exception as e:
                logger.error(
                    "Nested tool call failed for %s: %s",
                    nested_call.function.name,
                    e,
                )
                raise ToolExecutionError(
                    f"Nested tool call failed: {e}"
                ) from e

        tool_def = registry.get_tool(tool_call.function.name)

        if tool_def is None:
            raise ToolExecutionError(
                f"Tool not found: {tool_call.function.name}"
            )

        if len(child_results) > 1:
            if not _check_list_parameter_compatibility(
                tool_def,
                len(child_results),
            ):
                raise ToolExecutionError(
                    f"Tool {tool_call.function.name} has multiple "
                    "nested calls but does not accept a list parameter"
                )

        args = tool_call.function.arguments.copy()

        if child_results:
            if len(child_results) == 1:
                args["data"] = child_results[0]
            else:
                args["data"] = child_results

        args = _convert_arguments_for_tool(args, tool_def)

        return await registry.execute_tool_call(
            ToolCall(
                function=ToolCallFunction(
                    name=tool_call.function.name,
                    arguments=args,
                ),
            ),
        )

    return await registry.execute_tool_call(tool_call)


def _convert_arguments_for_tool(
        args: dict[str, Any],
        tool_def: ToolDefinition,
    ) -> dict[str, Any]:
    """Convert argument types to match tool schema."""
    converted = {}
    params = tool_def.function.parameters.get("properties", {})

    for key, value in args.items():
        if key in params:
            param_schema = params[key]
            param_type = param_schema.get("type")

            if param_type == "string":
                converted[key] = _convert_type(value, str)
            elif param_type == "integer":
                converted[key] = _convert_type(value, int)
            elif param_type == "number":
                converted[key] = _convert_type(value, float)
            elif param_type == "boolean":
                converted[key] = _convert_type(value, bool)
            elif param_type == "array":
                if not isinstance(value, list):
                    value = [value]
                item_schema = param_schema.get("items", {})
                item_type = item_schema.get("type")
                if item_type:
                    converted[key] = [
                        _convert_type(item, _python_type_from_schema(item_type))
                        for item in value
                    ]
                else:
                    converted[key] = value
            elif param_type == "object":
                converted[key] = _convert_type(value, dict)
            else:
                converted[key] = value
        else:
            converted[key] = value

    return converted


def _python_type_from_schema(schema_type: str) -> type:
    """Convert JSON schema type to Python type."""
    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    return type_map.get(schema_type, object)


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
        if tool_call.nested_calls:
            return await _execute_nested_tool_call(tool_call, self)

        name = tool_call.function.name
        func = self._functions.get(name)

        if func is None:
            raise ToolExecutionError(f"Tool not found: {name}")

        args = tool_call.function.arguments
        tool_def = self._tools.get(name)

        if tool_def is not None:
            try:
                _validate_args(args, tool_def.function.parameters)
            except ToolExecutionError as e:
                logger.error(
                    "Argument validation failed for tool %s: %s",
                    name,
                    e,
                )
                raise

        logger.debug(
            "Executing tool %s with arguments: %s",
            name,
            args,
        )

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
            )
            raise ToolExecutionError(
                f"Tool execution failed for {name}: {e}"
            ) from e

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
    if tool_call.nested_calls:
        return await _execute_nested_tool_call(tool_call, _registry)
    return await _registry.execute_tool_call(tool_call)
