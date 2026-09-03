"""
Tests for tool system functionality and XML compatibility layer.
"""

from pathlib import Path
from hola.session.session import Session
from hola.tools import (
    ToolRegistry,
    tool,
    register_tool,
    get_tool,
    list_tools,
    execute_tool_call,
    ToolCall,
    ToolCallFunction,
    ToolDefinition,
    ToolExecutionError,
)
from hola.tools.compat import (
    convert_parameter_value,
    parse_tool_calls_from_xml,
    tools_to_xml,
    tool_calls_to_xml,
    process_assistant_message,
)
from hola.tools.core import (
    _interpolate_template,
    _interpolate_arguments,
)
import asyncio
import shutil
import unittest
import packify


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
        )

        self.registry.register_tool(tool_def, test_func)

        retrieved = self.registry.get_tool("test_tool")
        assert retrieved is not None, retrieved
        assert retrieved.name == "test_tool", (
            retrieved.name,
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
            name="tool1",
            description="Tool 1",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
        )

        tool_def2 = ToolDefinition(
            name="tool2",
            description="Tool 2",
            parameters={
                "type": "object",
                "properties": {"y": {"type": "string"}},
                "required": ["y"],
            },
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
        assert retrieved.name == "add_numbers", (
            retrieved.name,
            "add_numbers",
        )

    async def test_tool_def_attribute(self):
        """Test that tool_def attribute is set on function objects."""
        def test_func(x: int, y: str) -> str:
            return f"{x}:{y}"

        tool_def = ToolDefinition(
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
        )

        self.registry.register_tool(tool_def, test_func)

        assert hasattr(test_func, "tool_def")
        assert test_func.tool_def is not None
        assert test_func.tool_def.name == "test_tool"
        assert test_func.tool_def.description == "Test tool"
        assert test_func.tool_def.parameters is not None

        @tool(name="decorated_tool", description="Decorated tool")
        def decorated_func(a: int) -> int:
            return a * 2

        assert hasattr(decorated_func, "tool_def")
        assert decorated_func.tool_def is not None
        assert decorated_func.tool_def.name == "decorated_tool"

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

        schema = get_tool("test_tool").parameters

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

        schema = get_tool("test_tool").parameters

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

        schema = get_tool("test_tool").parameters

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

    async def test_parameter_type_validation(self):
        """Test parameter type validation for string, int, number, bool."""
        @tool(name="test_tool_string", description="Test tool")
        def test_func_string(name: str) -> str:
            return name

        tool_call = ToolCall(
            function=ToolCallFunction(
                name="test_tool_string",
                arguments={"name": 123},
            ),
        )

        with self.assertRaises(ToolExecutionError) as e:
            await execute_tool_call(tool_call)
        assert "expected " in str(e.exception), e.exception
        assert "got " in str(e.exception), e.exception

        @tool(name="test_tool_int", description="Test tool")
        def test_func_int(count: int) -> int:
            return count

        tool_call = ToolCall(
            function=ToolCallFunction(
                name="test_tool_int",
                arguments={"count": "not_an_int"},
            ),
        )

        with self.assertRaises(ToolExecutionError) as e:
            await execute_tool_call(tool_call)
        assert "expected " in str(e.exception), e.exception
        assert "got " in str(e.exception), e.exception

        @tool(name="test_tool_float", description="Test tool")
        def test_func_float(price: float) -> float:
            return price

        tool_call = ToolCall(
            function=ToolCallFunction(
                name="test_tool_float",
                arguments={"price": [1, 2, 3]},
            ),
        )

        with self.assertRaises(ToolExecutionError) as e:
            await execute_tool_call(tool_call)
        assert "expected " in str(e.exception), e.exception
        assert "got " in str(e.exception), e.exception

        @tool(name="test_tool_bool", description="Test tool")
        def test_func_bool(enabled: bool) -> bool:
            return enabled

        tool_call = ToolCall(
            function=ToolCallFunction(
                name="test_tool_bool",
                arguments={"enabled": "true"},
            ),
        )

        with self.assertRaises(ToolExecutionError):
            await execute_tool_call(tool_call)
        assert "expected " in str(e.exception), e.exception
        assert "got " in str(e.exception), e.exception

    async def test_error_scenarios(self):
        """Test not found, execution error, error wrapper, inner error
            propagation, multiple children error.
        """
        tool_call = ToolCall(
            function=ToolCallFunction(
                name="nonexistent_tool",
                arguments={},
            ),
        )

        with self.assertRaises(ToolExecutionError) as e:
            await execute_tool_call(tool_call)
        assert "Tool not found" in str(e.exception), e.exception

        @tool(name="failing_tool", description="Failing tool")
        def failing_func(x: int) -> int:
            raise ValueError("Intentional error")

        tool_call = ToolCall(
            function=ToolCallFunction(
                name="failing_tool",
                arguments={"x": 5},
            ),
        )

        with self.assertRaises(ToolExecutionError) as e:
            await execute_tool_call(tool_call)
        assert "Tool execution failed" in str(e.exception), e.exception

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
                arguments={"data": nested_call},
            ),
        )

        result = await execute_tool_call(outer_call)
        assert result == 20, result

        @tool(name="failing_inner", description="Failing inner tool")
        def failing_inner(x: int) -> int:
            raise ValueError("Intentional failure")

        nested_call = ToolCall(
            function=ToolCallFunction(
                name="failing_inner",
                arguments={"x": 5},
            ),
        )

        outer_call = ToolCall(
            function=ToolCallFunction(
                name="outer",
                arguments={"data": nested_call},
            ),
        )

        with self.assertRaises(ToolExecutionError) as e:
            await execute_tool_call(outer_call)
        assert "Tool execution failed" in str(e.exception), e.exception
        assert "failing_inner" in str(e.exception), e.exception

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
                arguments={"data": [call1, call2]},
            ),
        )

        with self.assertRaises(ToolExecutionError) as e:
            await execute_tool_call(outer_call)
        assert "Invalid type for argument" in str(e.exception), e.exception

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
                arguments={"data": nested_call},
            ),
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
                arguments={"data": call1},
            ),
        )

        call3 = ToolCall(
            function=ToolCallFunction(
                name="level3",
                arguments={"data": call2},
            ),
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
                arguments={"data": [call1, call2, call3]},
            ),
        )

        result = await execute_tool_call(outer_call)
        assert result == 9, result

    async def test_type_conversion_in_nested_calls(self):
        """Test automatic type conversions and dependency execution order."""
        @tool(name="inner_int", description="Inner tool int")
        def inner_int(x: int) -> int:
            return x

        @tool(name="outer_float", description="Outer tool float")
        def outer_float(data: float) -> float:
            return data * 1.5

        nested_call = ToolCall(
            function=ToolCallFunction(
                name="inner_int",
                arguments={"x": 4},
            ),
        )

        outer_call = ToolCall(
            function=ToolCallFunction(
                name="outer_float",
                arguments={"data": nested_call},
            ),
        )

        result = await execute_tool_call(outer_call)
        assert result == 6.0, result

        @tool(name="inner_str", description="Inner tool str")
        def inner_str(x: str) -> str:
            return x

        @tool(name="outer_int", description="Outer tool int")
        def outer_int(data: int) -> int:
            return data * 2

        nested_call = ToolCall(
            function=ToolCallFunction(
                name="inner_str",
                arguments={"x": "5"},
            ),
        )

        outer_call = ToolCall(
            function=ToolCallFunction(
                name="outer_int",
                arguments={"data": nested_call},
            ),
        )

        result = await execute_tool_call(outer_call)
        assert result == 10, result

        @tool(name="inner_str2", description="Inner tool str2")
        def inner_str2(x: str) -> str:
            return x

        @tool(name="outer_bool", description="Outer tool bool")
        def outer_bool(data: bool) -> str:
            return "yes" if data else "no"

        nested_call = ToolCall(
            function=ToolCallFunction(
                name="inner_str2",
                arguments={"x": "true"},
            ),
        )

        outer_call = ToolCall(
            function=ToolCallFunction(
                name="outer_bool",
                arguments={"data": nested_call},
            ),
        )

        result = await execute_tool_call(outer_call)
        assert result == "yes", result

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
                arguments={"data": call1},
            ),
        )

        call3 = ToolCall(
            function=ToolCallFunction(
                name="third",
                arguments={"data": call2},
            ),
        )

        await execute_tool_call(call3)

        assert execution_order == ["first", "second", "third"], execution_order

    async def test_tool_call_auto_id_generation(self):
        """Test that ToolCall objects automatically get unique call_ids."""
        call1 = ToolCall(
            function=ToolCallFunction(
                name="read_file",
                arguments={"path": "/tmp/file.txt"},
            ),
        )

        call2 = ToolCall(
            function=ToolCallFunction(
                name="grep",
                arguments={"pattern": "test"},
            ),
        )

        assert call1.call_id is not None, call1.call_id
        assert call2.call_id is not None, call2.call_id
        assert call1.call_id.startswith("call_"), call1.call_id
        assert call2.call_id.startswith("call_"), call2.call_id
        assert len(call1.call_id) == 21, call1.call_id  # "call_" + 16 chars
        assert len(call2.call_id) == 21, call2.call_id
        assert call1.call_id != call2.call_id, f"{call1.call_id} == {call2.call_id}"

    async def test_tool_call_pack_unpack_nested(self):
        outer_call = ToolCall(
            function=ToolCallFunction(
                name="outer_tool",
                arguments={
                    "data": ToolCall(
                        function=ToolCallFunction(
                            name="inner_tool",
                            arguments={"x": 5},
                        ),
                    ),
                    "other": ToolCall(
                        function=ToolCallFunction(
                            name="another_inner",
                            arguments={"y": 10},
                        ),
                    ),
                },
            ),
        )

        original_outer_id = outer_call.call_id
        original_inner_id = outer_call.function.arguments["data"].call_id
        original_another_id = outer_call.function.arguments["other"].call_id

        packed = packify.pack(outer_call)
        assert isinstance(packed, bytes), type(packed)

        unpacked = packify.unpack(packed, inject={**globals()})

        assert unpacked.call_id == original_outer_id, unpacked.call_id
        assert unpacked.function.name == "outer_tool", unpacked.function.name
        assert "data" in unpacked.function.arguments, unpacked.function.arguments
        assert "other" in unpacked.function.arguments, unpacked.function.arguments
        arg_data = unpacked.function.arguments["data"]
        assert arg_data.call_id == original_inner_id, arg_data.call_id
        assert arg_data.function.name == "inner_tool", arg_data.function.name
        arg_other = unpacked.function.arguments["other"]
        assert arg_other.call_id == original_another_id, arg_other.call_id
        assert arg_other.function.name == "another_inner", arg_other.function.name

    async def test_e2e_nested_tool_calls_with_memory(self):
        """End-to-end test with nested tool calls, memory
            interpolation, and list parameters using direct ToolCall
            construction. Tests parsing, interpolation, and execution
            together.
        """
        @tool(name="sum", description="Sum a list of numbers")
        def sum_numbers(numbers: list[int]) -> int:
            return sum(numbers)

        @tool(name="get_number", description="Get a fixed number")
        def get_number() -> int:
            return 42

        @tool(name="dummy", description="Dummy tool")
        def dummy(does: str) -> str:
            return does

        session = Session("test_e2e_direct", portable=True)
        session.memorize("numbers", [1, 2, 3])

        inner_sum_memory = ToolCall(
            function=ToolCallFunction(
                name="sum",
                arguments={"numbers": "${numbers}"},
            ),
        )

        inner_get_number = ToolCall(
            function=ToolCallFunction(
                name="get_number",
                arguments={},
            ),
        )

        inner_sum_values = ToolCall(
            function=ToolCallFunction(
                name="sum",
                arguments={"numbers": [1, 2, inner_get_number]},
            ),
        )

        root_sum = ToolCall(
            function=ToolCallFunction(
                name="sum",
                arguments={"numbers": [inner_sum_memory, inner_sum_values]},
            ),
        )

        dummy_call = ToolCall(
            function=ToolCallFunction(
                name="dummy",
                arguments={"does": "nothing"},
            ),
        )

        result1 = await execute_tool_call(root_sum, session)
        assert result1 == 51, result1

        result2 = await execute_tool_call(dummy_call, session)
        assert result2 == "nothing", result2


class TestXMLCompatibility(unittest.TestCase):
    """Test cases for XML compatibility layer."""
    def test_format_single_tool(self):
        """Test formatting a single tool with simple parameters."""
        tool_def = ToolDefinition(
            name="add_numbers",
            description="Adds two integers together",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        )
        xml, instructions = tools_to_xml([tool_def])
        assert isinstance(xml, str), xml
        assert isinstance(instructions, str), instructions
        assert "<tools>" in xml, xml
        assert "</tools>" in xml, xml
        assert "<tool name=\"add_numbers\">" in xml, xml
        assert "<parameter name=\"a\">42</parameter>" in xml, xml
        assert "<parameter name=\"b\">42</parameter>" in xml, xml
        assert "<!-- add_numbers: Adds two integers together -->" in xml, xml

    def test_format_array_parameters(self):
        """Test formatting a tool with array parameters."""
        tool_def = ToolDefinition(
            name="concatenate",
            description="Joins strings together",
            parameters={
                "type": "object",
                "properties": {
                    "parts": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["parts"],
            },
        )
        xml, instructions = tools_to_xml([tool_def])
        assert isinstance(xml, str), xml
        assert isinstance(instructions, str), instructions
        assert "<tool name=\"concatenate\">" in xml, xml
        assert xml.count("<parameter name=\"parts\">") == 2, xml
        assert "<parameter name=\"parts\">\"example\"</parameter>" in xml, xml

    def test_format_multiple_tools(self):
        """Test formatting multiple tools."""
        tool_def1 = ToolDefinition(
            name="add_numbers",
            description="Adds two integers",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
        )
        tool_def2 = ToolDefinition(
            name="multiply",
            description="Multiplies two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        )
        xml, instructions = tools_to_xml([tool_def1, tool_def2])
        assert isinstance(xml, str), xml
        assert isinstance(instructions, str), instructions
        assert "<tool name=\"add_numbers\">" in xml, xml
        assert "<tool name=\"multiply\">" in xml, xml
        assert xml.count("<tool name=") == 2, xml

    def test_format_basic_parameter_types(self):
        """Test formatting boolean, string, and number parameters."""
        tool_def_bool = ToolDefinition(
            name="toggle",
            description="Toggle a feature",
            parameters={
                "type": "object",
                "properties": {"enabled": {"type": "boolean"}},
                "required": ["enabled"],
            },
        )
        xml, instructions = tools_to_xml([tool_def_bool])
        assert isinstance(xml, str), xml
        assert isinstance(instructions, str), instructions
        assert "<parameter name=\"enabled\">true</parameter>" in xml, xml

        tool_def_str = ToolDefinition(
            name="greet",
            description="Say hello",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        )
        xml, instructions = tools_to_xml([tool_def_str])
        assert isinstance(xml, str), xml
        assert isinstance(instructions, str), instructions
        assert "<parameter name=\"name\">\"example\"</parameter>" in xml, xml

        tool_def_num = ToolDefinition(
            name="calculate",
            description="Calculate something",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "number"}},
                "required": ["value"],
            },
        )
        xml, instructions = tools_to_xml([tool_def_num])
        assert isinstance(xml, str), xml
        assert isinstance(instructions, str), instructions
        assert "<parameter name=\"value\">3.14</parameter>" in xml, xml

    def test_format_edge_cases(self):
        """Test formatting empty tool list and tool without properties."""
        xml, instructions = tools_to_xml([])
        assert isinstance(xml, str), xml
        assert isinstance(instructions, str), instructions
        assert xml == "<tools>\n</tools>", xml

        tool_def = ToolDefinition(
            name="ping",
            description="Simple ping",
            parameters={"type": "object", "properties": {}},
        )
        xml, instructions = tools_to_xml([tool_def])
        assert isinstance(xml, str), xml
        assert isinstance(instructions, str), instructions
        assert "<tool name=\"ping\">\n  </tool>" in xml, xml

    def test_parse_simple_tool(self):
        """Test parsing a simple tool call."""
        xml_text = """
        <tool name="add_numbers">
            <parameter name="a">5</parameter>
            <parameter name="b">3</parameter>
        </tool>
        """
        tool_calls = parse_tool_calls_from_xml(xml_text)
        assert len(tool_calls) == 1, len(tool_calls)
        assert tool_calls[0].function.name == "add_numbers"
        assert tool_calls[0].function.arguments == {"a": 5, "b": 3}

    def test_parse_multiple_tools(self):
        """Test parsing multiple tool calls."""
        xml_text = """
        <tool name="add">
            <parameter name="x">1</parameter>
        </tool>
        <tool name="subtract">
            <parameter name="x">2</parameter>
        </tool>
        """
        tool_calls = parse_tool_calls_from_xml(xml_text)
        assert len(tool_calls) == 2, len(tool_calls)
        assert tool_calls[0].function.name == "add"
        assert tool_calls[1].function.name == "subtract"

    def test_parse_with_tools_root(self):
        """Test parsing XML with <tools> root element."""
        xml_text = """
        <tools>
            <tool name="test">
                <parameter name="value">42</parameter>
            </tool>
        </tools>
        """
        tool_calls = parse_tool_calls_from_xml(xml_text)
        assert len(tool_calls) == 1, len(tool_calls)
        assert tool_calls[0].function.name == "test"

    def test_parse_with_malformed_calls(self):
        """Test parsing XML with some malformed tool calls."""
        xml_text = """
        <tool name="valid">
            <parameter name="x">1</parameter>
        </tool>
        <tool name="">
        </tool>
        <tool name="another_valid">
            <parameter name="y">2</parameter>
        </tool>
        """
        tool_calls = parse_tool_calls_from_xml(xml_text)
        assert len(tool_calls) == 2, len(tool_calls)
        assert tool_calls[0].function.name == "valid"
        assert tool_calls[1].function.name == "another_valid"

    def test_parse_with_markdown(self):
        """Test parsing tool calls from markdown code blocks."""
        xml_text = """
        Here's some text before.

        ```xml
        <tool name="test">
            <parameter name="value">123</parameter>
        </tool>
        ```

        And some text after.
        """
        tool_calls = parse_tool_calls_from_xml(xml_text)
        assert len(tool_calls) == 1, len(tool_calls)
        assert tool_calls[0].function.name == "test"

    def test_parse_empty_xml(self):
        """Test parsing empty XML."""
        xml_text = ""
        tool_calls = parse_tool_calls_from_xml(xml_text)
        assert len(tool_calls) == 0, len(tool_calls)

    def test_convert_parameter_value(self):
        """Test parameter conversion for booleans, numbers, and strings."""
        assert convert_parameter_value("true") is True
        assert convert_parameter_value("false") is False
        assert convert_parameter_value("yes") is True
        assert convert_parameter_value("no") is False
        assert convert_parameter_value("1") is True
        assert convert_parameter_value("0") is False
        assert convert_parameter_value("TRUE") is True
        assert convert_parameter_value("FALSE") is False

        assert convert_parameter_value("42") == 42
        assert convert_parameter_value("3.14") == 3.14
        assert convert_parameter_value("-5") == -5
        assert convert_parameter_value("0.5") == 0.5

        assert convert_parameter_value("hello") == "hello"
        assert convert_parameter_value("not_a_number") == "not_a_number"
        assert convert_parameter_value("maybe") == "maybe"

    def test_parse_nested_calls(self):
        """Test parsing nested tool calls."""
        xml_text = """
        <tool name="outer">
            <parameter name="data">
                <tool name="inner">
                    <parameter name="x">5</parameter>
                </tool>
            </parameter>
        </tool>
        """
        tool_calls = parse_tool_calls_from_xml(xml_text)
        assert len(tool_calls) == 1, len(tool_calls)
        assert tool_calls[0].function.name == "outer"
        assert "data" in tool_calls[0].function.arguments
        nested_call = tool_calls[0].function.arguments["data"]
        assert isinstance(nested_call, ToolCall), type(nested_call)
        assert nested_call.function.name == "inner"
        assert nested_call.function.arguments == {"x": 5}

    def test_parse_multiple_nested_calls(self):
        """Test parsing multiple nested tool calls."""
        xml_text = """
        <tool name="outer">
            <parameter name="data1">
                <tool name="inner1">
                    <parameter name="x">1</parameter>
                </tool>
            </parameter>
            <parameter name="data2">
                <tool name="inner2">
                    <parameter name="y">2</parameter>
                </tool>
            </parameter>
        </tool>
        <tool name="another">
            <parameter name="purpose">To prove it does not interfere.</parameter>
        </tool>
        """
        tool_calls = parse_tool_calls_from_xml(xml_text)
        assert len(tool_calls) == 2, len(tool_calls)
        assert tool_calls[0].function.name == "outer"
        assert "data1" in tool_calls[0].function.arguments
        assert "data2" in tool_calls[0].function.arguments
        nested1 = tool_calls[0].function.arguments["data1"]
        nested2 = tool_calls[0].function.arguments["data2"]
        assert isinstance(nested1, ToolCall), type(nested1)
        assert isinstance(nested2, ToolCall), type(nested2)
        assert nested1.function.name == "inner1"
        assert nested2.function.name == "inner2"

    def test_parse_no_nested_calls(self):
        """Test parsing tool calls without nested calls."""
        xml_text = """
        <tool name="simple">
            <parameter name="value">42</parameter>
        </tool>
        """
        tool_calls = parse_tool_calls_from_xml(xml_text)
        assert len(tool_calls) == 1, len(tool_calls)
        assert tool_calls[0].function.name == "simple"
        assert tool_calls[0].function.arguments == {"value": 42}

    def test_round_trip(self):
        """Test round-trip: ToolDefinition → XML → ToolCall."""
        tool_def = ToolDefinition(
            name="add_numbers",
            description="Adds two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        )
        xml, instructions = tools_to_xml([tool_def])
        assert isinstance(xml, str), xml
        assert isinstance(instructions, str), instructions
        xml_with_values = xml.replace("42", "5").replace('"example"', '"hello"')
        tool_calls = parse_tool_calls_from_xml(xml_with_values)
        assert len(tool_calls) == 1, len(tool_calls)
        assert tool_calls[0].function.name == "add_numbers"
        assert "a" in tool_calls[0].function.arguments
        assert "b" in tool_calls[0].function.arguments

    def test_execute_parsed_tools(self):
        """Test executing tools parsed from XML."""
        @tool(name="multiply", description="Multiply two numbers")
        def multiply(a: int, b: int) -> int:
            return a * b
        xml_text = """
        <tool name="multiply">
            <parameter name="a">6</parameter>
            <parameter name="b">7</parameter>
        </tool>
        """
        tool_calls = parse_tool_calls_from_xml(xml_text)
        assert len(tool_calls) == 1, len(tool_calls)
        result = asyncio.run(execute_tool_call(tool_calls[0]))
        assert result == 42, result

    def test_round_trip_with_nested_calls(self):
        """Test round-trip with nested tool calls."""
        @tool(name="double", description="Double a number")
        def double(x: int) -> int:
            return x * 2

        @tool(name="add_ten", description="Add ten")
        def add_ten(number: int) -> int:
            return number + 10
        xml_text = """
        <tool name="add_ten">
            <parameter name="number">
                <tool name="double">
                    <parameter name="x">5</parameter>
                </tool>
            </parameter>
        </tool>
        """
        tool_calls = parse_tool_calls_from_xml(xml_text)
        assert len(tool_calls) == 1, len(tool_calls)
        assert tool_calls[0].function.name == "add_ten"
        assert "number" in tool_calls[0].function.arguments
        nested_call = tool_calls[0].function.arguments["number"]
        assert isinstance(nested_call, ToolCall), type(nested_call)
        assert nested_call.function.name == "double"
        result = asyncio.run(execute_tool_call(tool_calls[0]))
        assert result == 20, result

    def test_e2e_xml_nested_tool_calls_with_memory(self):
        """End-to-end XML compatibility test with nested tool calls,
            memory interpolation, and list parameters. Tests parsing,
            interpolation, and execution together via XML.
        """
        @tool(name="sum", description="Sum a list of numbers")
        def sum_numbers(numbers: list[int]) -> int:
            return sum(numbers)

        @tool(name="get_number", description="Get a fixed number")
        def get_number() -> int:
            return 42

        @tool(name="dummy", description="Dummy tool")
        def dummy(does: str) -> str:
            return does

        session = Session("test_e2e_xml", portable=True)
        session.memorize("numbers", [1, 2, 3])

        xml_text = """
        <tool name="sum">
            <parameter name="numbers">
                <tool name="sum">
                    <parameter name="numbers">${numbers}</parameter>
                </tool>
            </parameter>
            <parameter name="numbers">
                <tool name="sum">
                    <parameter name="numbers">1</parameter>
                    <parameter name="numbers">2</parameter>
                    <parameter name="numbers">
                        <tool name="get_number"></tool>
                    </parameter>
                </tool>
            </parameter>
        </tool>
        <tool name="dummy">
            <parameter name="does">nothing</parameter>
        </tool>
        """

        tool_calls = parse_tool_calls_from_xml(xml_text)
        assert len(tool_calls) == 2, len(tool_calls)

        result1 = asyncio.run(execute_tool_call(tool_calls[0], session))
        assert result1 == 51, result1

        result2 = asyncio.run(execute_tool_call(tool_calls[1], session))
        assert result2 == "nothing", result2

    def test_serialize_simple_tool_call(self):
        """Test serializing a simple tool call with autogenerated call_id."""
        tool_call = ToolCall(
            function=ToolCallFunction(
                name="sum",
                arguments={"numbers": [1, 2, 3]}
            )
        )

        xml = tool_calls_to_xml(tool_call)
        assert f'call_id="{tool_call.call_id}"' in xml, xml
        assert 'name="sum"' in xml, xml
        assert '<parameter name="numbers">1</parameter>' in xml, xml
        assert '<parameter name="numbers">2</parameter>' in xml, xml
        assert '<parameter name="numbers">3</parameter>' in xml, xml
        assert xml.count('<parameter name="numbers">') == 3, xml

    def test_serialize_nested_and_empty_tool_calls(self):
        """Test serializing nested and empty tool calls with call_ids."""
        nested_call = ToolCall(
            function=ToolCallFunction(
                name="get_number",
                arguments={}
            )
        )

        outer_call = ToolCall(
            function=ToolCallFunction(
                name="sum",
                arguments={"numbers": [1, 2, nested_call]}
            )
        )

        xml = tool_calls_to_xml(outer_call)
        assert f'call_id="{outer_call.call_id}"' in xml, xml
        assert f'call_id="{nested_call.call_id}"' in xml, xml
        assert '<tool name="get_number"' in xml, xml
        assert '<parameter name="numbers">1</parameter>' in xml, xml
        assert '<parameter name="numbers">2</parameter>' in xml, xml
        substr = '<parameter name="numbers">\n        <tool name="get_number"'
        assert substr in xml, xml

        xml_empty = tool_calls_to_xml(nested_call)
        assert f'call_id="{nested_call.call_id}"' in xml_empty, xml_empty
        assert xml_empty == (f'<tool name="get_number" call_id="'
            f'{nested_call.call_id}"></tool>'), xml_empty
        assert '\n' not in xml_empty, xml_empty

    def test_serialize_and_parse_multiple_with_call_ids(self):
        """Test serializing multiple tool calls and parser ignores
            call_id attributes.
        """
        call1 = ToolCall(
            function=ToolCallFunction(
                name="get_number",
                arguments={}
            )
        )

        call2 = ToolCall(
            function=ToolCallFunction(
                name="sum",
                arguments={"numbers": [1, 2, 3]}
            )
        )

        call3 = ToolCall(
            function=ToolCallFunction(
                name="greet",
                arguments={"name": "Alice"}
            )
        )

        xml = tool_calls_to_xml([call1, call2, call3])
        assert f'call_id="{call1.call_id}"' in xml, xml
        assert f'call_id="{call2.call_id}"' in xml, xml
        assert f'call_id="{call3.call_id}"' in xml, xml
        assert xml.count('<tool name=') == 3, xml

        parsed_calls = parse_tool_calls_from_xml(xml)
        assert len(parsed_calls) == 3, len(parsed_calls)

        assert parsed_calls[0].call_id != call1.call_id, parsed_calls[0].call_id
        assert parsed_calls[1].call_id != call2.call_id, parsed_calls[1].call_id
        assert parsed_calls[2].call_id != call3.call_id, parsed_calls[2].call_id

        assert parsed_calls[0].function.name == "get_number", (
            parsed_calls[0].function.name
        )
        assert parsed_calls[1].function.name == "sum", (
            parsed_calls[1].function.name
        )
        assert parsed_calls[2].function.name == "greet", (
            parsed_calls[2].function.name
        )

    def test_serialize_parse_roundtrip_with_nested_calls(self):
        """Test serializing nested tool call to XML and parsing it back."""
        inner_call = ToolCall(
            function=ToolCallFunction(
                name="get_number",
                arguments={}
            )
        )

        outer_call = ToolCall(
            function=ToolCallFunction(
                name="sum",
                arguments={"numbers": [1, 2, inner_call]}
            )
        )

        xml = tool_calls_to_xml(outer_call)

        assert f'call_id="{outer_call.call_id}"' in xml, xml
        assert f'call_id="{inner_call.call_id}"' in xml, xml
        assert '<tool name="sum"' in xml, xml
        assert '<tool name="get_number"' in xml, xml

        parsed_calls = parse_tool_calls_from_xml(xml)
        assert len(parsed_calls) == 1, len(parsed_calls)

        parsed_outer = parsed_calls[0]

        assert parsed_outer.function.name == "sum", parsed_outer.function.name
        assert "numbers" in parsed_outer.function.arguments

        parsed_numbers = parsed_outer.function.arguments["numbers"]
        assert isinstance(parsed_numbers, list), type(parsed_numbers)
        assert len(parsed_numbers) == 3, len(parsed_numbers)

        assert parsed_numbers[0] == 1, parsed_numbers[0]
        assert parsed_numbers[1] == 2, parsed_numbers[1]

        parsed_inner = parsed_numbers[2]
        assert isinstance(parsed_inner, ToolCall), type(parsed_inner)
        assert parsed_inner.function.name == "get_number", parsed_inner.function.name
        assert parsed_inner.function.arguments == {}, parsed_inner.function.arguments

        assert parsed_outer.call_id != outer_call.call_id, parsed_outer.call_id
        assert parsed_inner.call_id != inner_call.call_id, parsed_inner.call_id

    def test_process_assistant_message_single_tool(self):
        """Test processing message with single tool call."""
        content = ("I'll search for that.\n<tool name=\"search\"><parameter "
            "name=\"query\">cats</parameter></tool>")
        modified, tool_calls = process_assistant_message(content)

        assert "[Tool call intercepted" in modified, modified
        assert "I'll search for that." in modified, modified
        assert "<tool name=\"search\" call_id=" in modified, modified
        assert "[Tool calls]" in modified, modified
        assert len(tool_calls) == 1, len(tool_calls)
        assert tool_calls[0].function.name == "search", tool_calls[0].function.name
        assert tool_calls[0].function.arguments == {"query": "cats"}

    def test_process_assistant_message_multiple_tools(self):
        """Test processing message with multiple tool calls."""
        content = ("I'll search and sum.\n<tool name=\"search\"><parameter "
            "name=\"query\">cats</parameter></tool>\n<tool name=\"sum\">"
            "<parameter name=\"numbers\">1</parameter><parameter name=\"numbers\">"
            "2</parameter><parameter name=\"numbers\">3</parameter></tool>")
        modified, tool_calls = process_assistant_message(content)

        assert "I'll search and sum." in modified, modified
        assert "[Tool call intercepted" in modified, modified
        intercepted_count = modified.count("[Tool call intercepted")
        assert intercepted_count == 2, intercepted_count
        assert "[Tool calls]" in modified, modified
        assert "call_id=\"call_" in modified, modified
        assert len(tool_calls) == 2, len(tool_calls)
        assert tool_calls[0].function.name == "search", tool_calls[0].function.name
        assert tool_calls[1].function.name == "sum", tool_calls[1].function.name

    def test_process_assistant_message_nested_tool(self):
        """Test processing message with nested tool call."""
        content = ("I'll chain tools.\n<tool name=\"outer\"><parameter name="
            "\"data\">\n<tool name=\"inner\"><parameter name=\"x\">5</parameter>"
            "</tool>\n</parameter></tool>")
        modified, tool_calls = process_assistant_message(content)

        assert "I'll chain tools." in modified, modified
        assert "[Tool call intercepted" in modified, modified
        assert modified.count("[Tool call intercepted") == 1, (
            modified.count("[Tool call intercepted")
        )
        assert "[Tool calls]" in modified, modified
        assert "call_id=\"call_" in modified, modified
        assert len(tool_calls) == 1, len(tool_calls)
        assert tool_calls[0].function.name == "outer", tool_calls[0].function.name
        assert "data" in tool_calls[0].function.arguments
        nested_call = tool_calls[0].function.arguments["data"]
        assert isinstance(nested_call, ToolCall), type(nested_call)
        assert nested_call.function.name == "inner", nested_call.function.name

    def test_process_assistant_message_no_tools(self):
        """Test processing message with no tool calls."""
        content = "This is just a regular message with no tools."
        modified, tool_calls = process_assistant_message(content)

        assert modified == content, modified
        assert len(tool_calls) == 0, len(tool_calls)
        assert "[Tool calls]" not in modified, modified

    def test_process_assistant_message_text_before_between_after(self):
        """Test processing text before, between, and after tools."""
        content = ("Start.\n<tool name=\"search\"><parameter name=\"query\">"
            "cats</parameter></tool>\nMiddle.\n<tool name=\"sum\"><parameter "
            "name=\"numbers\">1</parameter><parameter name=\"numbers\">2"
            "</parameter></tool>\nEnd.")
        modified, tool_calls = process_assistant_message(content)

        assert "Start." in modified, modified
        assert "Middle." in modified, modified
        assert "End." in modified, modified
        assert "[Tool call intercepted" in modified, modified
        assert "[Tool calls]" in modified, modified
        assert len(tool_calls) == 2, len(tool_calls)

    def test_process_assistant_message_empty_message(self):
        """Test processing empty message."""
        content = ""
        modified, tool_calls = process_assistant_message(content)

        assert modified == "", modified
        assert len(tool_calls) == 0, len(tool_calls)

    def test_process_assistant_message_roundtrip(self):
        """Test roundtrip: original → process → parse again → same structure."""
        original_content = ("Searching.\n<tool name=\"search\"><parameter name="
            "\"query\">cats</parameter></tool>")
        modified_content, original_tool_calls = process_assistant_message(
            original_content
        )

        reparsed_calls = parse_tool_calls_from_xml(modified_content)

        assert len(reparsed_calls) == len(original_tool_calls), (
            len(reparsed_calls), len(original_tool_calls)
        )
        rc = reparsed_calls[0]
        otc = original_tool_calls[0]
        assert rc.function.name == otc.function.name, (rc, otc)
        assert rc.function.arguments == otc.function.arguments, (rc, otc)


class TestTemplateInterpolation(unittest.IsolatedAsyncioTestCase):
    """Test cases for template interpolation with session memories."""
    def setUp(self):
        """Set up test fixtures."""
        @tool(name="echo", description="Echo a message")
        def echo(message: str) -> str:
            return message

        self.session = Session("test_template_interpolation", portable=True)

    def tearDown(self):
        hola_dir = Path.cwd() / ".hola"
        if hola_dir.exists():
            shutil.rmtree(hola_dir)

    async def test_basic_interpolation(self):
        """Test simple key interpolation."""
        self.session.memorize("name", "Alice")

        result = _interpolate_template("Hello ${name}", self.session)
        assert result == "Hello Alice", result

    async def test_integer_value_conversion(self):
        """Test that integer values are converted to strings."""
        self.session.memorize("age", 25)

        result = _interpolate_template("Age: ${age}", self.session)
        assert result == "Age: 25", result

    async def test_dot_notation_dict(self):
        """Test dot notation for nested dictionary access."""
        self.session.memorize("user", {"name": "Bob"})

        result = _interpolate_template("User: ${user.name}", self.session)
        assert result == "User: Bob", result

    async def test_list_indexing(self):
        """Test list indexing with bracket notation."""
        self.session.memorize("users", ["Alice", "Bob"])

        result = _interpolate_template("First: ${users[0]}", self.session)
        assert result == "First: Alice", result

    async def test_nested_list_access(self):
        """Test nested list and dict access."""
        self.session.memorize("user", {
             "friends": [
                 {"name": "Alice", "id": "123"},
                 {"name": "Bob", "id": "456"},
             ]
        })

        result = _interpolate_template(
            "Friend: ${user.friends[0].name}, ID: ${user.friends[0].id}",
            self.session
        )
        assert result == "Friend: Alice, ID: 123", result

    async def test_multiple_interpolations(self):
        """Test multiple interpolations in one string."""
        self.session.memorize("name", "Alice")
        self.session.memorize("age", 25)

        result = _interpolate_template(
            "Hello ${name}, age ${age}", self.session
        )
        assert result == "Hello Alice, age 25", result

    async def test_missing_key_error(self):
        """Test that missing keys raise ToolExecutionError."""
        with self.assertRaises(ToolExecutionError) as cm:
            _interpolate_template("Hello ${missing}", self.session)

        assert "not found" in str(cm.exception).lower(), str(cm.exception)

    async def test_invalid_nested_path_error(self):
        """Test error on invalid nested path."""
        self.session.memorize("user", {"name": "Alice"})

        with self.assertRaises(ToolExecutionError) as cm:
            _interpolate_template("${user.invalid}", self.session)

        assert "not found" in str(cm.exception).lower(), str(cm.exception)

    async def test_list_out_of_range_error(self):
        """Test error when list index is out of range."""
        self.session.memorize("users", ["Alice"])

        with self.assertRaises(ToolExecutionError) as cm:
            _interpolate_template("${users[5]}", self.session)

        assert "out of range" in str(cm.exception).lower(), str(cm.exception)

    async def test_backslash_escaping(self):
        """Test backslash escaping of template syntax."""
        result = _interpolate_template(r"Literal \${name}", self.session)
        assert result == "Literal ${name}", result

    async def test_dollar_signs_before_interpolation(self):
        """Test dollar signs before interpolation."""
        self.session.memorize("name", "Alice")

        result = _interpolate_template("$${name}", self.session)
        assert result == "$Alice", result
        result = _interpolate_template("$$${name}", self.session)
        assert result == "$$Alice", result

    async def test_non_string_arguments_unchanged(self):
        """Test that non-string arguments remain unchanged."""
        args = {"x": 123, "y": True, "z": 3.14}
        result = _interpolate_arguments(args, self.session)
        assert result == args, result

    async def test_list_argument_interpolation(self):
        """Test interpolation in list arguments."""
        self.session.memorize("name", "Alice")

        args = {"data": [1, "${name}", 3]}
        result = _interpolate_arguments(args, self.session)
        assert result == {"data": [1, "Alice", 3]}, result

    async def test_dict_argument_interpolation(self):
        """Test interpolation in dict arguments."""
        self.session.memorize("key", "value")

        args = {"nested": {"x": "${key}", "y": 1}}
        result = _interpolate_arguments(args, self.session)
        assert result == {"nested": {"x": "value", "y": 1}}, result

    async def test_no_session_no_interpolation(self):
        """Test strings pass through unchanged when no session is provided."""
        result = _interpolate_arguments({"msg": "${name}"}, None)
        assert result == {"msg": "${name}"}, result

    async def test_integration_with_tool_execution(self):
        """Test interpolation in actual tool execution."""
        @tool(name="greet", description="Greet someone")
        def greet(name: str) -> str:
            return f"Hello {name}"

        self.session.memorize("name", "Alice")

        tool_call = ToolCall(
            function=ToolCallFunction(
                name="greet",
                arguments={"name": "${name}"},
            ),
        )

        result = await execute_tool_call(tool_call, self.session)
        assert result == "Hello Alice", result

    async def test_integration_with_nested_tool_calls(self):
        """Test interpolation across nested tool calls."""
        @tool(name="double", description="Double a number")
        def double(x: int) -> int:
            return x * 2

        @tool(name="add_ten", description="Add ten")
        def add_ten(data: int) -> int:
            return data + 10

        self.session.memorize("value", "5")

        inner_call = ToolCall(
            function=ToolCallFunction(
                name="double",
                arguments={"x": "${value}"},
            ),
        )

        outer_call = ToolCall(
            function=ToolCallFunction(
                name="add_ten",
                arguments={"data": inner_call},
            ),
        )

        result = await execute_tool_call(outer_call, self.session)
        assert result == 20, result


class TestSessionParameterInjection(unittest.IsolatedAsyncioTestCase):
    """Test cases for session parameter injection in tool execution."""
    def setUp(self):
        """Set up test fixtures."""
        @tool(name="no_session", description="Tool without session")
        def no_session_tool(x: int) -> int:
            return x * 2

        @tool(name="with_session", description="Tool with session")
        def with_session_tool(x: int, session: "Session") -> int:
            session.memorize("test_key", x * 2)
            return x * 2

        @tool(name="session_required", description="Tool requiring session")
        def session_required_tool(session: "Session") -> str:
            session.memorize("required_key", "value")
            return "done"

    async def test_session_injection_works(self):
        """Test session is injected and accessible in tools."""
        session = Session("test_injection", portable=True)
        session.delete()

        session = Session("test_injection", portable=True)

        call1 = ToolCall(
            function=ToolCallFunction(
                name="with_session",
                arguments={"x": 5},
            ),
        )

        call2 = ToolCall(
            function=ToolCallFunction(
                name="session_required",
                arguments={},
            ),
        )

        result1 = await execute_tool_call(call1, session)
        result2 = await execute_tool_call(call2, session)

        assert result1 == 10, result1
        assert result2 == "done", result2

        assert session.recall("test_key") == 10
        assert session.recall("required_key") == "value"

        session.delete()

    async def test_tools_without_session_still_work(self):
        """Test backward compatibility for tools without session parameter."""
        tool_call = ToolCall(
            function=ToolCallFunction(
                name="no_session",
                arguments={"x": 3},
            ),
        )

        result = await execute_tool_call(tool_call, None)
        assert result == 6, result

    async def test_session_excluded_from_schemas(self):
        """Test session parameter is hidden from JSON schemas."""
        tools_with_session = [
            get_tool("with_session"),
            get_tool("session_required"),
        ]

        for tool in tools_with_session:
            assert tool is not None

            params = tool.parameters.get("properties", {})
            assert "session" not in params, params

            required = tool.parameters.get("required", [])
            assert "session" not in required, required


if __name__ == "__main__":
    unittest.main()
