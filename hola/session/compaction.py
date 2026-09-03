"""
Compaction logic for managing conversation context.
"""

from hola.context import calculate_context_size
from hola.errors import type_assert
from hola.prompts.adapters import (
    tool_enabled_chat_adapter,
    xml_workaround_chat_adapter,
    generate_endpoint_adapter,
)
from hola.prompts.generators import generate_prompt
from hola.types import (
    Adapter, ChatResponse, GenerateResponse, LLMClient, Message, Role,
    SessionProtocol, GenerateRequest,
)


DEFAULT_MAX_CONTEXT: int = 4096
DEFAULT_CONTEXT_THRESHOLD_RATIO: float = 0.7
DEFAULT_COMPACTION_RATIO: float = 0.5
COMPACTION_FAILED_MESSAGE: str = (
    "Compaction failed for older messages. "
    "Previous context was truncated. "
    "Useful information may be found in session memories."
)


def check_context_threshold(session: SessionProtocol, phase: str) -> bool:
    """Check if current phase context exceeds threshold."""
    threshold_ratio = session.config.get(
        'context_threshold_ratio', DEFAULT_CONTEXT_THRESHOLD_RATIO
    )
    max_context = session.config.get('max_context', {}).get(
        phase, DEFAULT_MAX_CONTEXT
    )
    threshold = session.config.get(
        'context_threshold', threshold_ratio * max_context
    )
    messages = session.get_history(phase)
    context_size = calculate_context_size(messages)
    return context_size >= threshold


def get_messages_to_compact(
        session: SessionProtocol, phase: str
    ) -> list[Message]:
    """Returns a list of messages that should be compacted."""
    compaction_ratio = session.config.get(
        'context_compaction_ratio', DEFAULT_COMPACTION_RATIO
    )
    max_context = session.config.get('max_context', {}).get(
        phase, DEFAULT_MAX_CONTEXT
    )
    default_target = session.config.get('context_compaction_target', {}).get(
        'default', compaction_ratio * max_context
    )
    target = session.config.get('context_compaction_target', {}).get(
        phase, default_target
    )
    messages = session.get_history(phase)
    messages.reverse()
    to_compact = []
    total = 0
    while total < target and len(messages):
        to_compact.append(messages.pop())
        total += calculate_context_size([to_compact[-1]])
    return to_compact


async def compact_messages(
        client: LLMClient, model: str, messages: list[Message],
        adapter: Adapter, tools = [], **kwargs
    ) -> ChatResponse | GenerateResponse:
    instructions = generate_prompt("COMPACTION", **kwargs)
    request = adapter(instructions, messages, tools, model)
    request.stream = False
    if isinstance(request, GenerateRequest):
        return await client.generate(request)
    return await client.chat(request)


async def autocompact(
        session: SessionProtocol, client: LLMClient, phase: str,
        *, debug: bool = False, **kwargs
    ) -> Message | None:
    if not check_context_threshold(session, phase):
        return

    messages = get_messages_to_compact(session, phase)
    models_config = session.config.get('models')
    model = models_config.get('COMPACTION', models_config.get('DEFAULT'))
    type_assert(type(model) is str, 'missing model name config')
    adapter_config = session.config.get('adapters', {})
    adapter_name = adapter_config.get(model, 'tool_enabled_chat_adapter')
    adapter = globals()[adapter_name]
    result = None

    for _ in range(session.config.get('compaction_retries', 3)):
        try:
            response = await compact_messages(
                client, model, messages, adapter, **kwargs
            )
            if isinstance(response, GenerateResponse):
                result = Message(
                    role=Role.SYSTEM, content=response.response, model=model,
                )
            else:
                result = response.message
            break
        except Exception:
            if debug:
                raise
    if not result:
        result = Message(
            role=Role.SYSTEM, content=COMPACTION_FAILED_MESSAGE, model=model
        )

    sequences = [m.sequence for m in messages]
    session.insert_compaction(model, phase, result.content, sequences)
    result.role = Role.SYSTEM
    return result

