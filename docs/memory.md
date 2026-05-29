# Agent Memory

This file serves as a persistent memory store for agents working on this
project. It contains important architectural decisions, patterns, and gotchas
that have been learned through development and may be useful for future work.

## Permission System Architecture

**Permission Resolution** (Session.get_file_permission)
Permissions are resolved using a hierarchical approach:
1. Normalize path to absolute (use `Path.absolute()`, not `resolve()` to avoid symlink issues)
2. Check exact matches in `files` dictionary
3. Check glob patterns (sorted by length, most specific first, first match wins)
4. Check inheritance (parent directories)
5. Apply default permissions:
   - Session temp files: full (7)
   - Session CWD: read-only (4)
   - Everything else: no permissions (0)

**Pattern Matching**
Custom `_match_path_with_glob()` function supports `**` recursive patterns.
Patterns sorted by `(-len(pattern), pattern)` to prioritize specific matches.

**Important**: Always set `self.dbsession.permissions = dict`, never assign to individual
keys in the dictionary, to ensure packify serialization works correctly.

## Monadic Tool Calling

Nested tool calls appear inline in arguments/parameter values rather than in a separate `nested_calls` field. A ToolCall object can be placed directly in the arguments dictionary under a specific parameter name. The execution engine recursively evaluates these nested ToolCall objects before executing the parent call.

## XML Compatibility Layer for Tool Calling

**Primary Entry Points**
The compatibility layer provides two main functions for harness-level tool calling:

1. `tools_to_xml(tool_definitions)` - Convert tool definitions to XML format for inclusion in LLM prompts. Returns a tuple of (xml_string, instructions) where `xml_string` contains `<tools>` with all tool definitions and `instructions` provides guidance syntax examples for the LLM. Use this when setting up a conversation to inform the LLM what tools are available.

2. `process_assistant_message(content, tools)` - Process complete LLM assistant messages by extracting tool calls, replacing them with a notice, and appending serialized XML with call_ids. This is the primary function for handling LLM responses. Returns a tuple of (modified_content, tool_calls_with_call_ids). The modified content removes raw XML tool tags (preventing LLM hallucination of call_ids) while the tool_calls list contains structured ToolCall objects with tracking identifiers.

**Note**: `parse_tool_calls_from_xml()` and `tool_calls_to_xml()` are internal implementation details. Agents should not use these directly when building harness logic - use `process_assistant_message()` instead.

**Typical Harness Workflow**
```
# When setting up conversation with LLM
tools_xml, instructions = tools_to_xml(available_tools)
system_prompt += f"\n\nAvailable tools:\n{tools_xml}\n\n{instructions}"

# When processing LLM response
modified_content, tool_calls = process_assistant_message(llm_output)
# Returns: ("I'll search.\n[Tool call intercepted...]\n\n[Tool calls]\n<tool name=\"search\" call_id=\"...\">...", [ToolCall(...)])

# Execute tool_calls using execution engine...
# Display modified_content to user (raw XML removed)
# Use tool_calls (with call_ids) to track and correlate results
```

The `process_assistant_message` function prevents LLMs from seeing call_id attributes (reducing hallucination risk) while maintaining proper tracking through the appended XML section.

**XML Serialization with Call IDs**
- `tool_calls_to_xml()` function serializes ToolCall objects to XML with `call_id` attributes
- XML format: `<tool name="tool_name" call_id="call_xyz123"><parameter name="param">value</parameter></tool>`
- Call IDs are for tracking tool results back to specific calls
- Empty tool calls (no parameters) serialize as single line: `<tool name="..." call_id="..."></tool>`

**Parser Behavior with Call IDs**
- Parser ignores `call_id` attributes in incoming XML
- Always generates fresh call_ids via `make_call_id()` when parsing
- This allows re-serializing parsed XML without conflicts
- Pattern: `TOOL_OPEN_PATTERN = re.compile(r'<tool\s+name="([^"]+)"(?:\s+call_id="([^"]+)")?>', re.DOTALL)`

**ElementTree Serialization Gotcha**
ElementTree can transform XML structure during serialization:
- Empty elements may convert to self-closing: `<tool></tool>` → `<tool />`
- When custom attributes exist, self-closing pattern must account for them
- Pattern must handle: `<tool name="..." call_id="..." />`
- Solution: Use flexible regex with optional attribute groups

**Indentation Depth in Serialization**
When recursively serializing nested tool calls, use `depth + 1` for nested content, not `depth + 2`. The `_serialize_value()` function already adds one level when it detects a ToolCall, so the caller should only add one level to get proper indentation:
```python
# Correct
value_str = _serialize_value(param_value, indent, depth + 1)

# Incorrect - causes excessive indentation
value_str = _serialize_value(param_value, indent, depth + 2)
```

**Self-Closing Tag Conversion**
Parser converts self-closing tags to non-self-closing format before recursive parsing:
```python
SELF_CLOSING_TOOL_PATTERN = re.compile(
    r'<tool\s+name="([^"]+)"(?:\s+call_id="([^"]+)")?\s*/>'
)
# Converts: <tool name="..." call_id="..." /> → <tool name="..." call_id="..."></tool>
```

This ensures consistent parsing regardless of how XML was generated or transformed.

## Context Size Tracking

- The `ChatResponse.prompt_tokens` contains the number of tokens processed before response generation began
- The DB Message model has `tokens` and `estimated_tokens` columns for tracking the tokens used by each message
- Each phase of a workflow starts with a single system message that is not stored in the database
- The harness will read the `prompt_tokens` from the first request as a baseline
- Construct a running tally by adding the baseline to the sum of the `tokens` fields from all persisted messages
- Set `estimated_tokens` in `session.add_message` and `session.add_tool_output`
- Each subsequent turn within the phase, subtract the running tally from the `prompt_tokens` and set the last message `tokens` field to that difference, then save the new message from the response
- Thus, each db `message.tokens` field is 0 until it has been processed as part of the prompt
- In the cast that multiple messages saved to the database in the same phase do not have `tokens` assigned, assign proportional to the `estimated_tokens` such that the sum of the new `tokens` values across those messages equals the difference calculated from the API response and cumulative tally
