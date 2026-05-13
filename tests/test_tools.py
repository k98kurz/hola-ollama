"""
Tests for tool system functionality and XML compatibility layer.
"""

import asyncio
from unittest.mock import Mock, patch
import unittest
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
    handle_nested_calls,
    parse_tool_calls_from_xml,
    tools_to_xml,
)
from hola.tools.core import (
    _interpolate_template,
    _interpolate_arguments,
)


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
        xml = tools_to_xml([tool_def])
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
        xml = tools_to_xml([tool_def])
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
        xml = tools_to_xml([tool_def1, tool_def2])
        assert "<tool name=\"add_numbers\">" in xml, xml
        assert "<tool name=\"multiply\">" in xml, xml
        assert xml.count("<tool name=") == 2, xml

    def test_format_boolean_parameters(self):
        """Test formatting boolean parameters."""
        tool_def = ToolDefinition(
            name="toggle",
            description="Toggle a feature",
            parameters={
                "type": "object",
                "properties": {"enabled": {"type": "boolean"}},
                "required": ["enabled"],
            },
        )
        xml = tools_to_xml([tool_def])
        assert "<parameter name=\"enabled\">true</parameter>" in xml, xml

    def test_format_string_parameters(self):
        """Test formatting string parameters."""
        tool_def = ToolDefinition(
            name="greet",
            description="Say hello",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        )
        xml = tools_to_xml([tool_def])
        assert "<parameter name=\"name\">\"example\"</parameter>" in xml, xml

    def test_format_number_parameters(self):
        """Test formatting number (float) parameters."""
        tool_def = ToolDefinition(
            name="calculate",
            description="Calculate something",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "number"}},
                "required": ["value"],
            },
        )
        xml = tools_to_xml([tool_def])
        assert "<parameter name=\"value\">3.14</parameter>" in xml, xml

    def test_format_empty_tool_list(self):
        """Test formatting an empty list of tools."""
        xml = tools_to_xml([])
        assert xml == "<tools>\n</tools>", xml

    def test_format_tool_without_properties(self):
        """Test formatting a tool with no parameters."""
        tool_def = ToolDefinition(
            name="ping",
            description="Simple ping",
            parameters={"type": "object", "properties": {}},
        )
        xml = tools_to_xml([tool_def])
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

    def test_convert_parameter_value_booleans(self):
        """Test boolean string conversion for compat layer."""
        assert convert_parameter_value("true") is True
        assert convert_parameter_value("false") is False
        assert convert_parameter_value("yes") is True
        assert convert_parameter_value("no") is False
        assert convert_parameter_value("1") is True
        assert convert_parameter_value("0") is False
        assert convert_parameter_value("TRUE") is True
        assert convert_parameter_value("FALSE") is False

    def test_convert_parameter_value_numbers(self):
        """Test numeric string conversion for compat layer."""
        assert convert_parameter_value("42") == 42
        assert convert_parameter_value("3.14") == 3.14
        assert convert_parameter_value("-5") == -5
        assert convert_parameter_value("0.5") == 0.5

    def test_convert_parameter_value_strings(self):
        """Test that non-convertible strings remain strings for compat layer."""
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
        assert tool_calls[0].nested_calls is not None
        assert len(tool_calls[0].nested_calls) == 1
        assert tool_calls[0].nested_calls[0].function.name == "inner"
        assert tool_calls[0].nested_calls[0].function.arguments == {"x": 5}

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
        """
        tool_calls = parse_tool_calls_from_xml(xml_text)
        assert len(tool_calls) == 1, len(tool_calls)
        assert tool_calls[0].function.name == "outer"
        assert tool_calls[0].nested_calls is not None
        assert len(tool_calls[0].nested_calls) == 2
        assert tool_calls[0].nested_calls[0].function.name == "inner1"
        assert tool_calls[0].nested_calls[1].function.name == "inner2"

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
        assert tool_calls[0].nested_calls is None

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
        xml = tools_to_xml([tool_def])
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
        def add_ten(data: int) -> int:
            return data + 10
        xml_text = """
        <tool name="add_ten">
            <parameter name="data">
                <tool name="double">
                    <parameter name="x">5</parameter>
                </tool>
            </parameter>
        </tool>
        """
        tool_calls = parse_tool_calls_from_xml(xml_text)
        assert len(tool_calls) == 1, len(tool_calls)
        assert tool_calls[0].function.name == "add_ten"
        assert tool_calls[0].nested_calls is not None
        assert len(tool_calls[0].nested_calls) == 1
        result = asyncio.run(execute_tool_call(tool_calls[0]))
        assert result == 20, result


class TestTemplateInterpolation(unittest.IsolatedAsyncioTestCase):
    """Test cases for template interpolation with session memories."""
    def setUp(self):
        """Set up test fixtures."""
        self.registry = ToolRegistry()

        @tool(name="echo", description="Echo a message")
        def echo(message: str) -> str:
            return message

        self.session = Mock()
        self.session.recall = Mock()

    async def test_basic_interpolation(self):
        """Test simple key interpolation."""
        self.session.recall.side_effect = lambda key: "Alice" if key == "name" else None

        result = _interpolate_template("Hello ${name}", self.session)
        assert result == "Hello Alice", result

    async def test_integer_value_conversion(self):
        """Test that integer values are converted to strings."""
        self.session.recall.side_effect = lambda key: 25 if key == "age" else None

        result = _interpolate_template("Age: ${age}", self.session)
        assert result == "Age: 25", result

    async def test_dot_notation_dict(self):
        """Test dot notation for nested dictionary access."""
        self.session.recall.side_effect = lambda key: (
            {"name": "Bob"} if key == "user" else None
        )

        result = _interpolate_template("User: ${user.name}", self.session)
        assert result == "User: Bob", result

    async def test_list_indexing(self):
        """Test list indexing with bracket notation."""
        self.session.recall.side_effect = lambda key: (
            ["Alice", "Bob"] if key == "users" else None
        )

        result = _interpolate_template("First: ${users[0]}", self.session)
        assert result == "First: Alice", result

    async def test_nested_list_access(self):
        """Test nested list and dict access."""
        self.session.recall.side_effect = lambda key: (
            {
                "friends": [
                    {"name": "Alice", "id": "123"},
                    {"name": "Bob", "id": "456"},
                ]
            }
            if key == "user"
            else None
        )

        result = _interpolate_template(
            "Friend: ${user.friends[0].name}, ID: ${user.friends[0].id}",
            self.session
        )
        assert result == "Friend: Alice, ID: 123", result

    async def test_multiple_interpolations(self):
        """Test multiple interpolations in one string."""
        self.session.recall.side_effect = lambda key: {
            "name": "Alice",
            "age": 25,
        }.get(key)

        result = _interpolate_template(
            "Hello ${name}, age ${age}", self.session
        )
        assert result == "Hello Alice, age 25", result

    async def test_missing_key_error(self):
        """Test that missing keys raise ToolExecutionError."""
        self.session.recall.return_value = None

        with self.assertRaises(ToolExecutionError) as cm:
            _interpolate_template("Hello ${missing}", self.session)

        assert "not found" in str(cm.exception).lower(), str(cm.exception)

    async def test_invalid_nested_path_error(self):
        """Test error on invalid nested path."""
        self.session.recall.side_effect = lambda key: {"name": "Alice"} if key == "user" else None

        with self.assertRaises(ToolExecutionError) as cm:
            _interpolate_template("${user.invalid}", self.session)

        assert "not found" in str(cm.exception).lower(), str(cm.exception)

    async def test_list_out_of_range_error(self):
        """Test error when list index is out of range."""
        self.session.recall.side_effect = lambda key: ["Alice"] if key == "users" else None

        with self.assertRaises(ToolExecutionError) as cm:
            _interpolate_template("${users[5]}", self.session)

        assert "out of range" in str(cm.exception).lower(), str(cm.exception)

    async def test_backslash_escaping(self):
        """Test backslash escaping of template syntax."""
        self.session.recall.return_value = None

        result = _interpolate_template(r"Literal \${name}", self.session)
        assert result == "Literal ${name}", result

    async def test_dollar_before_interpolation(self):
        """Test dollar signs before interpolation."""
        self.session.recall.side_effect = lambda key: "Alice" if key == "name" else None

        result = _interpolate_template("$${name}", self.session)
        assert result == "$Alice", result

    async def test_double_dollar_before_interpolation(self):
        """Test double dollar signs before interpolation."""
        self.session.recall.side_effect = lambda key: "Alice" if key == "name" else None

        result = _interpolate_template("$$${name}", self.session)
        assert result == "$$Alice", result

    async def test_non_string_arguments_unchanged(self):
        """Test that non-string arguments remain unchanged."""
        self.session.recall.return_value = None

        args = {"x": 123, "y": True, "z": 3.14}
        result = _interpolate_arguments(args, self.session)
        assert result == args, result

    async def test_list_argument_interpolation(self):
        """Test interpolation in list arguments."""
        self.session.recall.side_effect = lambda key: "Alice" if key == "name" else None

        args = {"data": [1, "${name}", 3]}
        result = _interpolate_arguments(args, self.session)
        assert result == {"data": [1, "Alice", 3]}, result

    async def test_dict_argument_interpolation(self):
        """Test interpolation in dict arguments."""
        self.session.recall.side_effect = lambda key: "value" if key == "key" else None

        args = {"nested": {"x": "${key}", "y": 1}}
        result = _interpolate_arguments(args, self.session)
        assert result == {"nested": {"x": "value", "y": 1}}, result

    async def test_no_session_no_interpolation(self):
        """Test that strings pass through unchanged when no session is provided."""
        result = _interpolate_arguments({"msg": "${name}"}, None)
        assert result == {"msg": "${name}"}, result

    async def test_integration_with_tool_execution(self):
        """Test interpolation in actual tool execution."""
        @tool(name="greet", description="Greet someone")
        def greet(name: str) -> str:
            return f"Hello {name}"

        self.session.recall.side_effect = lambda key: "Alice" if key == "name" else None

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

        self.session.recall.side_effect = lambda key: "5" if key == "value" else None

        inner_call = ToolCall(
            function=ToolCallFunction(
                name="double",
                arguments={"x": "${value}"},
            ),
        )

        outer_call = ToolCall(
            function=ToolCallFunction(
                name="add_ten",
                arguments={},
            ),
            nested_calls=[inner_call],
        )

        result = await execute_tool_call(outer_call, self.session)
        assert result == 20, result


if __name__ == "__main__":
    unittest.main()
