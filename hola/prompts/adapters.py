"""
Adapter functions for transforming Message lists into provider-specific
request formats.
"""

from hola.types import Adapter, ChatRequest, GenerateRequest, Message, Role
from hola.tools.compat import tools_to_xml
from hola.tools.types import ToolDefinition


def _partition_messages(
        messages: list[Message]
    ) -> tuple[list[Message], list[Message]]:
    """Split `messages` into the leading run of system messages and
        the remainder.
    """
    prelude, rest, in_prelude = [], [], True
    for m in messages:
        if in_prelude and m.role is Role.SYSTEM:
            prelude.append(m)
            continue
        in_prelude = False
        rest.append(m)
    return prelude, rest


def _combine_system_messages(messages: list[Message]) -> str:
    """Join the contents of `messages` with blank lines."""
    return '\n\n'.join([m.content for m in messages])


def _combine_non_leading_system_messages(messages: list[Message]) -> str:
    """Render `messages` as chat-style prompt blocks."""
    parts = []
    for m in messages:
        parts.append(f"<|im_start|>{m.role.value.lower()}\n{m.content}<|im_end|>")
    return '\n\n'.join(parts)


def tool_enabled_chat_adapter(
        phase_instructions: str,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        model_name: str = 'granite4.1:8b',
        rag_cag: str | None = None,
    ) -> ChatRequest:
    prelude, rest = _partition_messages(messages)
    system_prompt = phase_instructions + '\n\n'
    if rag_cag:
        system_prompt += rag_cag + '\n\n'
    system_prompt += _combine_system_messages(prelude)
    combined_messages = [Message(
        role=m.role,
        content=m.content,
        tool_calls=m.tool_calls,
        thinking=m.thinking,
    ) for m in rest]

    combined_messages.insert(0, Message(
        role=Role.SYSTEM,
        content=system_prompt.strip(),
    ))

    return ChatRequest(
        model=model_name,
        messages=combined_messages,
        tools=tools if tools else None,
    )


def xml_workaround_chat_adapter(
        phase_instructions: str,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        model_name: str = 'granite4.1:8b',
        rag_cag: str | None = None,
    ) -> ChatRequest:
    """Transform messages into ChatRequest with XML tool definitions.
        Combines leading system messages into one, includes tool
        definitions in XML format within the system message for
        non-tool-calling models.
    """
    prelude, rest = _partition_messages(messages)
    system_prompt = phase_instructions + '\n\n'

    if tools:
        xml_tools, tool_instructions = tools_to_xml(tools)
        system_prompt += xml_tools + '\n\n' + tool_instructions + '\n\n'

    if rag_cag:
        system_prompt += rag_cag + '\n\n'

    system_prompt += _combine_system_messages(prelude)

    combined_messages = [Message(
        role=m.role,
        content=m.content,
        tool_calls=m.tool_calls,
        thinking=m.thinking,
    ) for m in rest]

    combined_messages.insert(0, Message(
        role=Role.SYSTEM,
        content=system_prompt.strip(),
    ))

    return ChatRequest(
        model=model_name,
        messages=combined_messages,
        tools=None,
    )


def generate_endpoint_adapter(
        phase_instructions: str,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        model_name: str = 'granite4.1:8b',
        rag_cag: str | None = None,
    ) -> GenerateRequest:
    prelude, rest = _partition_messages(messages)
    system = phase_instructions + '\n\n'

    if tools:
        xml_tools, tool_instructions = tools_to_xml(tools)
        system += xml_tools + '\n\n' + tool_instructions + '\n\n'

    if rag_cag:
        system += rag_cag + '\n\n'

    system += _combine_system_messages(prelude)
    prompt = _combine_non_leading_system_messages(rest)
    return GenerateRequest(
        model=model_name,
        prompt=prompt,
        system=system.strip() if system else None,
    )


ADAPTERS: dict[str, Adapter] = {
    'tool_enabled_chat_adapter': tool_enabled_chat_adapter,
    'xml_workaround_chat_adapter': xml_workaround_chat_adapter,
    'generate_endpoint_adapter': generate_endpoint_adapter,
}
