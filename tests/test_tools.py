"""
Tests for tool system functionality.
"""

import asyncio
from unittest.mock import AsyncMock, patch
import unittest
from hola.ollama.tools import (
    ToolRegistry,
    tool,
    register_tool,
    get_tool,
    list_tools,
    execute_tool_call,
)
from hola.ollama.types import (
    ToolDefinition,
    ToolCall,
    FunctionDefinition,
    ToolCallFunction,
)
from hola.ollama.exceptions import ToolExecutionError


class TestTools(unittest.IsolatedAsyncioTestCase):
    """Test cases for tool system."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = ToolRegistry()

    async def test_tool_registry_registration(self):
        """Test tool registration."""
        def test_func(x: int, y: str) -> str:
            return f"{x}:{y}"

        tool_def = ToolDefinition(
            type="function",
            function=FunctionDefinition(
                name="test_tool",
                description="Test tool",
                parameters={
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer"},
                        "y": {"type": "string"},
                    },
                    "required": ["x", "y"],
                },
            ),
        )

        self.registry.register_tool(tool_def, test_func)

        retrieved = self.registry.get_tool("test_tool")
        assert retrieved is not None, retrieved
        assert retrieved.function.name == "test_tool", (
            retrieved.function.name,
            "test_tool",
        )

    async def test_tool_registry_get_tool_not_found(self):
        """Test getting non-existent tool."""
        retrieved = self.registry.get_tool("nonexistent")
        assert retrieved is None, retrieved

    async def test_tool_registry_list_tools(self):
        """Test listing all tools."""
        def func1(x: int) -> int:
            return x

        def func2(y: str) -> str:
            return y

        tool_def1 = ToolDefinition(
            type="function",
            function=FunctionDefinition(
                name="tool1",
                description="Tool 1",
                parameters={
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                },
            ),
        )

        tool_def2 = ToolDefinition(
            type="function",
            function=FunctionDefinition(
                name="tool2",
                description="Tool 2",
                parameters={
                    "type": "object",
                    "properties": {"y": {"type": "string"}},
                    "required": ["y"],
                },
            ),
        )

        self.registry.register_tool(tool_def1, func1)
        self.registry.register_tool(tool_def2, func2)

        tools = self.registry.list_tools()
        assert len(tools) == 2, len(tools)

    async def test_tool_decorator(self):
        """Test @tool decorator."""
        @tool(name="add_numbers", description="Add two numbers")
        def add(x: int, y: int) -> int:
            return x + y

        retrieved = get_tool("add_numbers")
        assert retrieved is not None, retrieved
        assert retrieved.function.name == "add_numbers", (
            retrieved.function.name,
            "add_numbers",
        )

    async def test_schema_generation_basic_types(self):
        """Test JSON schema generation for basic types."""
        @tool(name="test_tool", description="Test tool")
        def test_func(
            name: str,
            count: int,
            price: float,
            enabled: bool,
        ) -> dict:
            return {}

        schema = get_tool("test_tool").function.parameters

        assert schema["properties"]["name"]["type"] == "string", (
            schema["properties"]["name"]["type"],
            "string",
        )
        assert schema["properties"]["count"]["type"] == "integer", (
            schema["properties"]["count"]["type"],
            "integer",
        )
        assert schema["properties"]["price"]["type"] == "number", (
            schema["properties"]["price"]["type"],
            "number",
        )
        assert schema["properties"]["enabled"]["type"] == "boolean", (
            schema["properties"]["enabled"]["type"],
            "boolean",
        )

    async def test_schema_generation_optional_types(self):
        """Test JSON schema generation for optional types."""
        @tool(name="test_tool", description="Test tool")
        def test_func(
            name: str,
            count: int | None = None,
            tags: list[str] | None = None,
        ) -> dict:
            return {}

        schema = get_tool("test_tool").function.parameters

        assert "name" in schema["required"], schema["required"]
        assert "count" not in schema["required"], schema["required"]
        assert "tags" not in schema["required"], schema["required"]

        assert schema["properties"]["count"]["nullable"] is True, (
            schema["properties"]["count"]["nullable"],
            True,
        )
        assert schema["properties"]["tags"]["nullable"] is True, (
            schema["properties"]["tags"]["nullable"],
            True,
        )

    async def test_schema_generation_complex_types(self):
        """Test JSON schema generation for complex types."""
        @tool(name="test_tool", description="Test tool")
        def test_func(
            items: list[int],
            mapping: dict[str, float],
        ) -> dict:
            return {}

        schema = get_tool("test_tool").function.parameters

        assert schema["properties"]["items"]["type"] == "array", (
            schema["properties"]["items"]["type"],
            "array",
        )
        assert schema["properties"]["items"]["items"]["type"] == "integer", (
            schema["properties"]["items"]["items"]["type"],
            "integer",
        )

        assert schema["properties"]["mapping"]["type"] == "object", (
            schema["properties"]["mapping"]["type"],
            "object",
        )
        assert schema["properties"]["mapping"]["additionalProperties"]["type"] == (
            "number"
        ), (
            schema["properties"]["mapping"]["additionalProperties"]["type"],
            "number",
        )

    async def test_tool_execution_sync(self):
        """Test synchronous tool execution."""
        @tool(name="add_numbers", description="Add two numbers")
        def add(x: int, y: int) -> int:
            return x + y

        tool_call = ToolCall(
            function=ToolCallFunction(
                name="add_numbers",
                arguments={"x": 5, "y": 3},
            ),
        )

        result = await execute_tool_call(tool_call)
        assert result == 8, result

    async def test_tool_execution_async(self):
        """Test asynchronous tool execution."""
        @tool(name="async_tool", description="Async tool")
        async def async_func(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        tool_call = ToolCall(
            function=ToolCallFunction(
                name="async_tool",
                arguments={"x": 5},
            ),
        )

        result = await execute_tool_call(tool_call)
        assert result == 10, result

    async def test_tool_validation_string_type(self):
        """Test tool validation for string type."""
        @tool(name="test_tool", description="Test tool")
        def test_func(name: str) -> str:
            return name

        tool_call = ToolCall(
            function=ToolCallFunction(
                name="test_tool",
                arguments={"name": 123},
            ),
        )

        with self.assertRaises(ToolExecutionError):
            await execute_tool_call(tool_call)

    async def test_tool_validation_integer_type(self):
        """Test tool validation for integer type."""
        @tool(name="test_tool", description="Test tool")
        def test_func(count: int) -> int:
            return count

        tool_call = ToolCall(
            function=ToolCallFunction(
                name="test_tool",
                arguments={"count": "not_an_int"},
            ),
        )

        with self.assertRaises(ToolExecutionError):
            await execute_tool_call(tool_call)

    async def test_tool_validation_number_type(self):
        """Test tool validation for number type."""
        @tool(name="test_tool", description="Test tool")
        def test_func(price: float) -> float:
            return price

        tool_call = ToolCall(
            function=ToolCallFunction(
                name="test_tool",
                arguments={"price": [1, 2, 3]},
            ),
        )

        with self.assertRaises(ToolExecutionError):
            await execute_tool_call(tool_call)

    async def test_tool_validation_boolean_type(self):
        """Test tool validation for boolean type."""
        @tool(name="test_tool", description="Test tool")
        def test_func(enabled: bool) -> bool:
            return enabled

        tool_call = ToolCall(
            function=ToolCallFunction(
                name="test_tool",
                arguments={"enabled": "true"},
            ),
        )

        with self.assertRaises(ToolExecutionError):
            await execute_tool_call(tool_call)

    async def test_tool_execution_not_found_error(self):
        """Test tool execution when tool not found."""
        tool_call = ToolCall(
            function=ToolCallFunction(
                name="nonexistent_tool",
                arguments={},
            ),
        )

        with self.assertRaises(ToolExecutionError):
            await execute_tool_call(tool_call)

    async def test_tool_execution_error_wrapper(self):
        """Test tool execution error wrapping."""
        @tool(name="failing_tool", description="Failing tool")
        def failing_func(x: int) -> int:
            raise ValueError("Intentional error")

        tool_call = ToolCall(
            function=ToolCallFunction(
                name="failing_tool",
                arguments={"x": 5},
            ),
        )

        with self.assertRaises(ToolExecutionError):
            await execute_tool_call(tool_call)

    async def test_nested_tool_call_single_level(self):
        """Test single level nested tool call."""
        @tool(name="inner", description="Inner tool")
        def inner_tool(x: int) -> int:
            return x * 2

        @tool(name="outer", description="Outer tool")
        def outer_tool(data: int) -> int:
            return data + 10

        nested_call = ToolCall(
            function=ToolCallFunction(
                name="inner",
                arguments={"x": 5},
            ),
        )

        outer_call = ToolCall(
            function=ToolCallFunction(
                name="outer",
                arguments={},
            ),
            nested_calls=[nested_call],
        )

        result = await execute_tool_call(outer_call)
        assert result == 20, result

    async def test_nested_tool_call_deep_nesting(self):
        """Test deeply nested tool calls."""
        @tool(name="level1", description="Level 1")
        def level1(x: int) -> int:
            return x + 1

        @tool(name="level2", description="Level 2")
        def level2(data: int) -> int:
            return data * 2

        @tool(name="level3", description="Level 3")
        def level3(data: int) -> int:
            return data - 5

        call1 = ToolCall(
            function=ToolCallFunction(
                name="level1",
                arguments={"x": 5},
            ),
        )

        call2 = ToolCall(
            function=ToolCallFunction(
                name="level2",
                arguments={},
            ),
            nested_calls=[call1],
        )

        call3 = ToolCall(
            function=ToolCallFunction(
                name="level3",
                arguments={},
            ),
            nested_calls=[call2],
        )

        result = await execute_tool_call(call3)
        assert result == 7, result

    async def test_nested_tool_call_multiple_children_list_param(self):
        """Test nested tool call with multiple children and list parameter."""
        @tool(name="add_one", description="Add one")
        def add_one(x: int) -> int:
            return x + 1

        @tool(name="sum_all", description="Sum all numbers")
        def sum_all(data: list[int]) -> int:
            return sum(data)

        call1 = ToolCall(
            function=ToolCallFunction(
                name="add_one",
                arguments={"x": 1},
            ),
        )

        call2 = ToolCall(
            function=ToolCallFunction(
                name="add_one",
                arguments={"x": 2},
            ),
        )

        call3 = ToolCall(
            function=ToolCallFunction(
                name="add_one",
                arguments={"x": 3},
            ),
        )

        outer_call = ToolCall(
            function=ToolCallFunction(
                name="sum_all",
                arguments={},
            ),
            nested_calls=[call1, call2, call3],
        )

        result = await execute_tool_call(outer_call)
        assert result == 9, result

    async def test_nested_tool_call_multiple_children_no_list_param_error(self):
        """Test error when multiple children but no list parameter."""
        @tool(name="inner", description="Inner tool")
        def inner_tool(x: int) -> int:
            return x

        @tool(name="outer", description="Outer tool")
        def outer_tool(data: int) -> int:
            return data

        call1 = ToolCall(
            function=ToolCallFunction(
                name="inner",
                arguments={"x": 1},
            ),
        )

        call2 = ToolCall(
            function=ToolCallFunction(
                name="inner",
                arguments={"x": 2},
            ),
        )

        outer_call = ToolCall(
            function=ToolCallFunction(
                name="outer",
                arguments={},
            ),
            nested_calls=[call1, call2],
        )

        with self.assertRaises(ToolExecutionError):
            await execute_tool_call(outer_call)

    async def test_nested_tool_call_inner_error_propagation(self):
        """Test that inner tool errors prevent outer execution."""
        @tool(name="failing_inner", description="Failing inner tool")
        def failing_inner(x: int) -> int:
            raise ValueError("Intentional failure")

        @tool(name="outer", description="Outer tool")
        def outer_tool(data: int) -> int:
            return data + 10

        nested_call = ToolCall(
            function=ToolCallFunction(
                name="failing_inner",
                arguments={"x": 5},
            ),
        )

        outer_call = ToolCall(
            function=ToolCallFunction(
                name="outer",
                arguments={},
            ),
            nested_calls=[nested_call],
        )

        with self.assertRaises(ToolExecutionError):
            await execute_tool_call(outer_call)

    async def test_type_conversion_int_to_float(self):
        """Test automatic type conversion from int to float."""
        @tool(name="inner", description="Inner tool")
        def inner_tool(x: int) -> int:
            return x

        @tool(name="outer", description="Outer tool")
        def outer_tool(data: float) -> float:
            return data * 1.5

        nested_call = ToolCall(
            function=ToolCallFunction(
                name="inner",
                arguments={"x": 4},
            ),
        )

        outer_call = ToolCall(
            function=ToolCallFunction(
                name="outer",
                arguments={},
            ),
            nested_calls=[nested_call],
        )

        result = await execute_tool_call(outer_call)
        assert result == 6.0, result

    async def test_type_conversion_str_to_int(self):
        """Test automatic type conversion from str to int."""
        @tool(name="inner", description="Inner tool")
        def inner_tool(x: str) -> str:
            return x

        @tool(name="outer", description="Outer tool")
        def outer_tool(data: int) -> int:
            return data * 2

        nested_call = ToolCall(
            function=ToolCallFunction(
                name="inner",
                arguments={"x": "5"},
            ),
        )

        outer_call = ToolCall(
            function=ToolCallFunction(
                name="outer",
                arguments={},
            ),
            nested_calls=[nested_call],
        )

        result = await execute_tool_call(outer_call)
        assert result == 10, result

    async def test_type_conversion_bool_from_string(self):
        """Test boolean conversion from string."""
        @tool(name="inner", description="Inner tool")
        def inner_tool(x: str) -> str:
            return x

        @tool(name="outer", description="Outer tool")
        def outer_tool(data: bool) -> str:
            return "yes" if data else "no"

        nested_call = ToolCall(
            function=ToolCallFunction(
                name="inner",
                arguments={"x": "true"},
            ),
        )

        outer_call = ToolCall(
            function=ToolCallFunction(
                name="outer",
                arguments={},
            ),
            nested_calls=[nested_call],
        )

        result = await execute_tool_call(outer_call)
        assert result == "yes", result

    async def test_dependency_resolution_execution_order(self):
        """Test that nested calls execute in correct order (depth-first)."""
        execution_order = []

        @tool(name="first", description="First tool")
        def first_tool(x: int) -> int:
            execution_order.append("first")
            return x

        @tool(name="second", description="Second tool")
        def second_tool(data: int) -> int:
            execution_order.append("second")
            return data + 1

        @tool(name="third", description="Third tool")
        def third_tool(data: int) -> int:
            execution_order.append("third")
            return data + 2

        call1 = ToolCall(
            function=ToolCallFunction(
                name="first",
                arguments={"x": 1},
            ),
        )

        call2 = ToolCall(
            function=ToolCallFunction(
                name="second",
                arguments={},
            ),
            nested_calls=[call1],
        )

        call3 = ToolCall(
            function=ToolCallFunction(
                name="third",
                arguments={},
            ),
            nested_calls=[call2],
        )

        await execute_tool_call(call3)

        assert execution_order == ["first", "second", "third"], execution_order


if __name__ == "__main__":
    unittest.main()
