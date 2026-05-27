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


TOOL_OPEN_PATTERN = re.compile(
    r'<tool\s+name="([^"]+)"(?:\s+call_id="([^"]+)")?>', re.DOTALL
)
SELF_CLOSING_TOOL_PATTERN = re.compile(
    r'<tool\s+name="([^"]+)"(?:\s+call_id="([^"]+)")?\s*/>'
)
TOOL_CLOSE_TAG = '</tool>'


def _find_tool_positions(xml_text: str) -> list[tuple[int, int]]:
    """Find start and end positions of all top-level tool tags in text.
        Returns list of (start_pos, end_pos) tuples sorted by position.
        Does not include nested tool tags - only top-level tools.
    """
    open_matches = list(TOOL_OPEN_PATTERN.finditer(xml_text))

    all_tool_info: list[tuple[int, int]] = []

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
                    all_tool_info.append((start_pos, end_pos))
                    break
                else:
                    search_pos = next_close + len(TOOL_CLOSE_TAG)

    all_tool_info.sort(key=lambda x: x[0])

    top_level_tools: list[tuple[int, int]] = []

    for i, (start_pos, end_pos) in enumerate(all_tool_info):
        is_nested = False
        for j, (other_start, other_end) in enumerate(all_tool_info):
            if i == j:
                continue
            if start_pos > other_start and end_pos < other_end:
                is_nested = True
                break

        if not is_nested:
            top_level_tools.append((start_pos, end_pos))

    return top_level_tools


def parse_tool_calls_from_xml(
        xml_text: str,
        tools: list[ToolDefinition] | None = None
    ) -> list[ToolCall]:
    """Extract `ToolCall` objects from LLM output using lenient XML
        parsing. Returns a list of successfully parsed ToolCall objects,
        skipping malformed elements. Handles nested tool calls via
        monadic pattern where nested calls appear inline as parameter
        values.
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

                param_value_inner = SELF_CLOSING_TOOL_PATTERN.sub(
                    r'<tool name="\1"></tool>', param_value_inner
                )

                extracted_nested = handle_nested_calls(param_value_inner, tools)
                if extracted_nested:
                    if param_name in arguments:
                        existing = arguments[param_name]
                        if isinstance(existing, list):
                            existing.append(extracted_nested)
                        else:
                            arguments[param_name] = [existing, extracted_nested]
                    else:
                        arguments[param_name] = extracted_nested
                else:
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
    ) -> ToolCall | None:
    """Parse a single nested `<tool>` element in parameter content.
        Returns a ToolCall object if found, or None if no tool is present.
        Handles recursive nested tool calls within the nested tool's
        parameters.
    """
    parameter_content = SELF_CLOSING_TOOL_PATTERN.sub(
        r'<tool name="\1"></tool>', parameter_content
    )

    open_match = TOOL_OPEN_PATTERN.search(parameter_content)
    if not open_match:
        return None

    tool_name = open_match.group(1)
    if not tool_name:
        return None

    start_pos = open_match.start()
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
        return None

    arguments: dict[str, Any] = {}

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
                if param_name in arguments:
                    existing = arguments[param_name]
                    if isinstance(existing, list):
                        existing.append(extracted_nested)
                    else:
                        arguments[param_name] = [existing, extracted_nested]
                else:
                    arguments[param_name] = extracted_nested
            else:
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

    except Exception:
        return None

    return ToolCall(
        function=ToolCallFunction(
            name=tool_name,
            arguments=arguments,
        ),
    )


def _serialize_value(value: Any, indent: str, depth: int) -> str:
    """Serialize a value for XML parameter element."""
    if isinstance(value, ToolCall):
        return _serialize_tool_call(value, indent, depth + 1)
    elif isinstance(value, str):
        return value
    elif isinstance(value, list):
        items = [_serialize_value(item, indent, depth) for item in value]
        return "\n".join(items)
    elif isinstance(value, dict):
        return str(value)
    else:
        return str(value)


def _serialize_tool_call(tool_call: ToolCall, indent: str, depth: int) -> str:
    """Recursively serialize a ToolCall to XML with call_id attribute."""
    tool_name = tool_call.function.name
    call_id = tool_call.call_id
    arguments = tool_call.function.arguments

    prefix = indent * depth
    if not arguments:
        return f'{prefix}<tool name="{tool_name}" call_id="{call_id}"></tool>'

    inner_prefix = indent * (depth + 1)

    parts = [f'{prefix}<tool name="{tool_name}" call_id="{call_id}">']

    for param_name, param_value in arguments.items():
        value_str = _serialize_value(param_value, indent, depth + 1)

        if isinstance(param_value, list) and len(param_value) > 1:
            if any(isinstance(item, ToolCall) for item in param_value):
                for item in param_value:
                    item_str = _serialize_value(item, indent, depth + 1)
                    if '<tool' in item_str:
                        parts.append(f'{inner_prefix}<parameter name="{param_name}">')
                        parts.append(item_str)
                        parts.append(f'{inner_prefix}</parameter>')
                    else:
                        parts.append(
                            f'{inner_prefix}<parameter name="{param_name}">'
                            f'{item_str}</parameter>'
                        )
            else:
                for item in param_value:
                    parts.append(
                        f'{inner_prefix}<parameter name="{param_name}">{item}'
                        '</parameter>'
                    )
        elif '<tool' in value_str:
            parts.append(f'{inner_prefix}<parameter name="{param_name}">')
            parts.append(value_str)
            parts.append(f'{inner_prefix}</parameter>')
        else:
            parts.append(
                f'{inner_prefix}<parameter name="{param_name}">{value_str}'
                '</parameter>'
            )

    parts.append(f'{prefix}</tool>')

    return "\n".join(parts)


def tool_calls_to_xml(
        tool_calls: list[ToolCall] | ToolCall,
        indent: str = "    ",
    ) -> str:
    """Serialize ToolCall objects to XML with call_id attributes.
        Handles nested tool calls recursively. Returns XML string
        that can be embedded in Message content.
    """
    if isinstance(tool_calls, ToolCall):
        return _serialize_tool_call(tool_calls, indent, 0)

    return "\n".join(_serialize_tool_call(tc, indent, 0) for tc in tool_calls)


def process_assistant_message(
        content: str,
        tools: list[ToolDefinition] | None = None
    ) -> tuple[str, list[ToolCall]]:
    """Process LLM assistant message by extracting tool calls, replacing them
        with a notice, and appending serialized XML with call_ids. Returns
        (modified_content, tool_calls_with_call_ids). This prevents the LLM
        from hallucinating call_ids while maintaining proper tool call tracking.
    """
    tool_calls = parse_tool_calls_from_xml(content, tools)

    if not tool_calls:
        return content, tool_calls

    tool_positions = _find_tool_positions(content)

    placeholder = "[Tool call intercepted and processed by harness. See below for tool call with call_id assigned by the system.]"

    content_parts = list(content)
    offset = 0

    for start_pos, end_pos in reversed(tool_positions):
        actual_start = start_pos
        actual_end = end_pos
        content_parts[actual_start:actual_end] = [placeholder]

    modified_content = "".join(content_parts)

    serialized_tools = tool_calls_to_xml(tool_calls)
    modified_content += f"\n\n[Tool calls]\n{serialized_tools}"

    return modified_content, tool_calls
