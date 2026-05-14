"""
Tool system for function registration and execution.
"""

from __future__ import annotations
from types import UnionType
from typing import Any, Callable, TYPE_CHECKING
from hola.ollama.exceptions import ToolExecutionError
from hola.tools.types import (
    ToolCall,
    ToolCallFunction,
    ToolDefinition,
)
import asyncio
import inspect
import logging
import re

if TYPE_CHECKING:
    from hola.session.session import Session


logger = logging.getLogger(__name__)


def _interpolate_template(
        template: str,
        session: "Session",
    ) -> str:
    """Interpolate template variables using session memories. Supports
        ${key}, ${key.path}, and ${key[0]} syntax with backslash
        escaping for literal ${}.
    """
    def replace_match(match: re.Match) -> str:
        key_path = match.group(1)

        tokens = []
        current_token = ""
        i = 0

        while i < len(key_path):
            char = key_path[i]
            if char == '.':
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                tokens.append('.')
                i += 1
            elif char == '[':
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                tokens.append('[')
                i += 1
            elif char == ']':
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                tokens.append(']')
                i += 1
            else:
                current_token += char
                i += 1

        if current_token:
            tokens.append(current_token)

        base_key = tokens[0] if tokens else key_path

        current = session.recall(base_key)

        if current is None:
            raise ToolExecutionError(
                f"Memory key not found: {base_key}"
            )

        i = 1
        while i < len(tokens):
            token = tokens[i]

            if token == '.':
                i += 1
                if i >= len(tokens):
                    raise ToolExecutionError(
                        "Invalid path: ends with dot"
                    )
                next_token = tokens[i]
                if not isinstance(current, dict):
                    raise ToolExecutionError(
                        f"Cannot access key {next_token} on non-dict value"
                    )
                if next_token not in current:
                    raise ToolExecutionError(
                        f"Key {next_token} not found in nested structure"
                    )
                current = current[next_token]
                i += 1

            elif token == '[':
                i += 1
                if i >= len(tokens):
                    raise ToolExecutionError(
                        "Invalid path: unclosed bracket"
                    )
                index_token = tokens[i]
                i += 1

                if i >= len(tokens) or tokens[i] != ']':
                    raise ToolExecutionError(
                        "Invalid path: missing closing bracket"
                    )
                i += 1

                if not isinstance(current, list):
                    raise ToolExecutionError(
                        f"Cannot index non-list value at {index_token}"
                    )
                try:
                    idx = int(index_token)
                except ValueError:
                    raise ToolExecutionError(
                        f"Invalid list index: {index_token}"
                    )
                if idx < 0 or idx >= len(current):
                    raise ToolExecutionError(
                        f"Index {idx} out of range for list of length "
                        f"{len(current)}"
                    )
                current = current[idx]

            else:
                raise ToolExecutionError(
                    f"Invalid path token: {token}"
                )

        return str(current)

    result = re.sub(r'(?<!\\)\$\{([^}]+)\}', replace_match, template)
    result = result.replace(r'\${', '${')

    return result


def _interpolate_arguments(
        args: dict[str, Any],
        session: "Session" | None,
    ) -> dict[str, Any]:
    """Recursively interpolate template variables in arguments."""
    if session is None:
        return args

    def process_value(value: Any) -> Any:
        if isinstance(value, str):
            return _interpolate_template(value, session)
        elif isinstance(value, list):
            return [process_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: process_value(v) for k, v in value.items()}
        else:
            return value

    return {key: process_value(val) for key, val in args.items()}


def _generate_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """Generate JSON schema from function type hints."""
    sig = inspect.signature(func)
    parameters = {}

    for param_name, param in sig.parameters.items():
        if param_name == "self" or param_name == "session":
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
        if name != "self" and name != "session"
        and param.default is inspect.Parameter.empty
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


def _check_list_parameter_compatibility(
        tool_def: ToolDefinition,
        num_children: int,
    ) -> bool:
    """Check if tool can accept a list parameter for nested results."""
    params = tool_def.parameters.get("properties", {})

    data_param = params.get("data")
    if data_param and data_param.get("type") == "array":
        return True

    for param_name, param_schema in params.items():
        if param_schema.get("type") == "array":
            return True

    return False


async def _execute_nested_tool_call(
        tool_call: ToolCall,
        registry: ToolRegistry,
        session: "Session" | None = None,
    ) -> Any:
    """Execute a nested tool call with its children first."""
    if tool_call.nested_calls:
        child_results = []

        for nested_call in tool_call.nested_calls:
            try:
                result = await _execute_nested_tool_call(
                    nested_call,
                    registry,
                    session,
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
            session,
        )

    return await registry.execute_tool_call(tool_call, session)


def _convert_arguments_for_tool(
        args: dict[str, Any],
        tool_def: ToolDefinition,
    ) -> dict[str, Any]:
    """Convert argument types to match tool schema."""
    converted = {}
    params = tool_def.parameters.get("properties", {})

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
        self._tools[tool.name] = tool
        self._functions[tool.name] = func

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
            session: "Session" | None = None,
        ) -> Any:
        if tool_call.nested_calls:
            return await _execute_nested_tool_call(tool_call, self, session)

        name = tool_call.function.name
        func = self._functions.get(name)

        if func is None:
            raise ToolExecutionError(f"Tool not found: {name}")

        args = tool_call.function.arguments
        original_args = args

        if session is not None:
            args = _interpolate_arguments(args, session)

        tool_def = self._tools.get(name)

        if tool_def is not None:
            if session is not None and args != original_args:
                args = _convert_arguments_for_tool(args, tool_def)

            try:
                _validate_args(args, tool_def.parameters)
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

        sig = inspect.signature(func)
        if "session" in sig.parameters and session is not None:
            args = args.copy()
            args["session"] = session

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

        tool_def = ToolDefinition(
            name=name,
            description=description,
            parameters=schema,
        )

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


async def execute_tool_call(
        tool_call: ToolCall,
        session: "Session" | None = None,
    ) -> Any:
    """Execute tool call using global registry."""
    if tool_call.nested_calls:
        return await _execute_nested_tool_call(tool_call, _registry, session)
    return await _registry.execute_tool_call(tool_call, session)
