"""
XML compatibility layer for non-tool-calling LLMs.
"""

from __future__ import annotations
from hola.tools.types import ToolDefinition, ToolCall, ToolCallFunction
from typing import Any
import re
import xml.etree.ElementTree as ET


_instructions = """You can call tools with syntax like the following:
```xml
<tool name="tool_name">
    <parameter name="param_name">value</parameter>
</tool>
```

You can also chain tool calls, using the output of one tool call as the parameter \
value for another, for example:
```xml
<tool name="outer_tool>
    <parameter name="outer_param_name">
        <tool name="inner_tool">
            <parameter name="inner_param_name">...</parameter>
        </tool>
    </parameter>
</tool>
```
If a necessary tool does not exist, inform the user that you cannot fulfill that \
requirement."""

def tools_to_xml(tools: list[ToolDefinition]) -> tuple[str, str]:
    """Convert `ToolDefinition` objects to XML format for inclusion in
        LLM prompts. Returns XML string with `<tools>` root element
        containing all tool definitions.
    """
    xml_parts = ["<tools>"]

    for tool in tools:
        xml_parts.append(f"  <!-- {tool.name}: {tool.description} -->")
        xml_parts.append(f"  <tool name=\"{tool.name}\">")

        properties = tool.parameters.get("properties", {})

        for param_name, param_def in properties.items():
            param_type = param_def.get("type", "string")

            if param_type == "array":
                items_type = param_def.get("items", {}).get("type", "string")
                example_value = _get_example_value(items_type)
                xml_parts.append(
                    f"    <parameter name=\"{param_name}\">{example_value}</parameter>"
                )
                xml_parts.append(
                    f"    <parameter name=\"{param_name}\">{example_value}</parameter>"
                )
            else:
                example_value = _get_example_value(param_type)
                xml_parts.append(
                    f"    <parameter name=\"{param_name}\">{example_value}</parameter>"
                )

        xml_parts.append("  </tool>")

    xml_parts.append("</tools>")

    return "\n".join(xml_parts), _instructions


def _get_example_value(param_type: str) -> str:
    """Get an example value for a given parameter type. Returns an
        appropriate example value string based on the JSON Schema type.
    """
    examples = {
        "string": '"example"',
        "integer": "42",
        "number": "3.14",
        "boolean": "true",
        "array": '"item"',
        "object": '{"key": "value"}',
    }

    return examples.get(param_type, '"example"')


TOOL_OPEN_PATTERN = re.compile(r'<tool\s+name="([^"]+)">', re.DOTALL)

TOOL_CLOSE_TAG = '</tool>'


def parse_tool_calls_from_xml(
        xml_text: str,
        tools: list[ToolDefinition] | None = None
    ) -> list[ToolCall]:
    """Extract `ToolCall` objects from LLM output using lenient XML
        parsing. Returns a list of successfully parsed ToolCall objects,
        skipping malformed elements. Handles nested tool calls via
        recursive parsing of parameter values.
    """
    open_matches = list(TOOL_OPEN_PATTERN.finditer(xml_text))

    all_tool_info: list[tuple[str, int, int, str]] = []

    for open_match in open_matches:
        tool_name = open_match.group(1)
        start_pos = open_match.start()

        if not tool_name:
            continue

        depth = 1
        search_pos = open_match.end()

        while depth > 0 and search_pos < len(xml_text):
            next_open = TOOL_OPEN_PATTERN.search(xml_text, search_pos)
            next_close = xml_text.find(TOOL_CLOSE_TAG, search_pos)

            if next_close == -1:
                break

            if next_open and next_open.start() < next_close:
                depth += 1
                search_pos = next_open.end()
            else:
                depth -= 1
                if depth == 0:
                    end_pos = next_close + len(TOOL_CLOSE_TAG)
                    tool_content = xml_text[open_match.end():next_close]
                    all_tool_info.append((tool_name, start_pos, end_pos, tool_content))
                    break
                else:
                    search_pos = next_close + len(TOOL_CLOSE_TAG)

    all_tool_info.sort(key=lambda x: x[1])

    top_level_tools: list[tuple[str, int, int, str]] = []

    for i, (tool_name, start_pos, end_pos, tool_content) in enumerate(all_tool_info):
        is_nested = False
        for j, (other_name, other_start, other_end, other_content) in enumerate(
                all_tool_info
            ):
            if i == j:
                continue
            if start_pos > other_start and end_pos < other_end:
                is_nested = True
                break

        if not is_nested:
            top_level_tools.append((tool_name, start_pos, end_pos, tool_content))

    tool_calls: list[ToolCall] = []

    for tool_name, start_pos, end_pos, tool_content in top_level_tools:
        arguments: dict[str, Any] = {}
        nested_calls: list[ToolCall] = []

        try:
            tool_wrapper = f"<tool_wrapper>{tool_content}</tool_wrapper>"
            tool_elem = ET.fromstring(tool_wrapper)

            for param_elem in tool_elem.findall("parameter"):
                param_name = param_elem.get("name")
                if not param_name:
                    continue

                param_value_inner = "".join(
                    ET.tostring(child, encoding="unicode") for child in param_elem
                )

                extracted_nested = handle_nested_calls(param_value_inner, tools)
                if extracted_nested:
                    nested_calls.extend(extracted_nested)

                param_value_text = "".join(param_elem.itertext()).strip()
                converted_value = convert_parameter_value(param_value_text)

                if param_name in arguments:
                    existing = arguments[param_name]
                    if isinstance(existing, list):
                        existing.append(converted_value)
                    else:
                        arguments[param_name] = [existing, converted_value]
                else:
                    arguments[param_name] = converted_value
        except ET.ParseError:
            pass

        tool_calls.append(
            ToolCall(
                function=ToolCallFunction(
                    name=tool_name,
                    arguments=arguments,
                ),
                nested_calls=nested_calls if nested_calls else None,
            )
        )

    return tool_calls


def convert_parameter_value(value: str) -> int | float | bool | str:
    """Convert string values to appropriate Python types. Converts
        boolean strings (true/false/yes/no/1/0), numeric strings
        (int/float), or returns the original string if conversion fails.
    """
    lower_value = value.strip().lower()

    if lower_value in ("true", "yes", "1"):
        return True
    if lower_value in ("false", "no", "0"):
        return False

    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass

    return value


def handle_nested_calls(
        parameter_content: str,
        tools: list[ToolDefinition] | None
    ) -> list[ToolCall]:
    """Recursively parse nested `<tool>` elements in parameter content.
        Returns a list of nested ToolCall objects found in the parameter
        content, extracting parameters and recursively handling any
        further nesting.
    """
    nested_calls: list[ToolCall] = []

    open_matches = list(TOOL_OPEN_PATTERN.finditer(parameter_content))

    for open_match in open_matches:
        tool_name = open_match.group(1)
        start_pos = open_match.start()

        if not tool_name:
            continue

        depth = 1
        search_pos = open_match.end()

        while depth > 0 and search_pos < len(parameter_content):
            next_open = TOOL_OPEN_PATTERN.search(parameter_content, search_pos)
            next_close = parameter_content.find(TOOL_CLOSE_TAG, search_pos)

            if next_close == -1:
                break

            if next_open and next_open.start() < next_close:
                depth += 1
                search_pos = next_open.end()
            else:
                depth -= 1
                if depth == 0:
                    tool_content = parameter_content[open_match.end():next_close]
                    break
                else:
                    search_pos = next_close + len(TOOL_CLOSE_TAG)

        if depth != 0:
            continue

        arguments: dict[str, Any] = {}

        try:
            tool_wrapper = f"<tool_wrapper>{tool_content}</tool_wrapper>"
            tool_elem = ET.fromstring(tool_wrapper)

            for param_elem in tool_elem.findall("parameter"):
                param_name = param_elem.get("name")
                if not param_name:
                    continue

                param_value_text = "".join(param_elem.itertext()).strip()
                converted_value = convert_parameter_value(param_value_text)

                if param_name in arguments:
                    existing = arguments[param_name]
                    if isinstance(existing, list):
                        existing.append(converted_value)
                    else:
                        arguments[param_name] = [existing, converted_value]
                else:
                    arguments[param_name] = converted_value
        except ET.ParseError:
            pass

        nested_calls.append(
            ToolCall(
                function=ToolCallFunction(
                    name=tool_name,
                    arguments=arguments,
                ),
                nested_calls=None,
            )
        )

    return nested_calls
