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


if __name__ == "__main__":
    unittest.main()
