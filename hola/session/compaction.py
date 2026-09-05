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
from hola.tools.types import ToolCall
from hola.types import (
    Adapter, ChatResponse, GenerateResponse, LLMClient, Message, Role,
    SessionProtocol, GenerateRequest,
)


DEFAULT_MAX_CONTEXT: int = 4096
DEFAULT_CONTEXT_THRESHOLD_RATIO: float = 0.7
DEFAULT_COMPACTION_RATIO: float = 0.5
DEFAULT_COMPACTION_RESERVE_RATIO: float = 0.25
SPLIT_BOUNDARY_WINDOW: int = 512
DIGEST_MAX_VALUE_CHARS: int = 100
DIGEST_MAX_TOTAL_CHARS: int = 1024
DIGEST_OMITTED_MARKER: str = '... (+{} more tool calls omitted)'
CONTINUED_MESSAGE_HEADER: str = '# Continued from Previous Message'
TOOL_CALLS_HEADER: str = '## Tool Calls'
COMPACTION_FAILED_MESSAGE: str = (
    "Compaction failed for older messages. "
    "Previous context was truncated. "
    "Useful information may be found in session memories."
)


def _get_threshold(session: SessionProtocol, phase: str) -> int | float:
    """Returns the context size threshold for a phase."""
    threshold_ratio = session.config.get(
        'context_threshold_ratio', DEFAULT_CONTEXT_THRESHOLD_RATIO
    )
    max_context = session.config.get('max_context', {}).get(
        phase, DEFAULT_MAX_CONTEXT
    )
    return session.config.get(
        'context_threshold', threshold_ratio * max_context
    )


def _get_target(session: SessionProtocol, phase: str) -> int | float:
    """Returns the compaction target size for a phase."""
    compaction_ratio = session.config.get(
        'context_compaction_ratio', DEFAULT_COMPACTION_RATIO
    )
    max_context = session.config.get('max_context', {}).get(
        phase, DEFAULT_MAX_CONTEXT
    )
    default_target = session.config.get(
        'context_compaction_target', {}
    ).get('default', compaction_ratio * max_context)
    return session.config.get('context_compaction_target', {}).get(
        phase, default_target
    )


def check_context_threshold(session: SessionProtocol, phase: str) -> bool:
    """Check if current phase context exceeds threshold."""
    threshold = _get_threshold(session, phase)
    messages = session.get_history(phase)
    context_size = calculate_context_size(messages)
    return context_size >= threshold


def get_messages_to_compact(
        session: SessionProtocol, phase: str
    ) -> list[Message]:
    """Returns a list of messages that should be compacted."""
    target = _get_target(session, phase)
    messages = session.get_history(phase)
    messages.reverse()
    to_compact = []
    total = 0
    while total < target and len(messages):
        to_compact.append(messages.pop())
        total += calculate_context_size([to_compact[-1]])
    return to_compact


def _format_digest_value(value: object) -> str:
    """Renders an argument value for the tool call digest, abbreviating
        nested tool calls as placeholders and truncating long values.
    """
    if isinstance(value, ToolCall):
        return f'<nested: {value.function.name}>'
    if isinstance(value, list):
        return '[' + ', '.join(
            _format_digest_value(item) for item in value
        ) + ']'
    if isinstance(value, dict):
        return '{' + ', '.join(
            f'{_format_digest_value(k)}: {_format_digest_value(v)}'
            for k, v in value.items()
        ) + '}'
    rendered = str(value)
    if len(rendered) > DIGEST_MAX_VALUE_CHARS:
        rendered = rendered[:DIGEST_MAX_VALUE_CHARS] + '...'
    return rendered


def _cap_digest_lines(lines: list[str], call_count: int) -> str:
    """Truncates rendered digest lines at `DIGEST_MAX_TOTAL_CHARS`
        characters, keeping whole call lines where they fit,
        hard-truncating the first call line when it alone exceeds the
        cap, and noting how many calls were omitted.
    """
    kept = 1
    size = len(lines[0])
    for line in lines[1:]:
        if size + 1 + len(line) > DIGEST_MAX_TOTAL_CHARS:
            break
        size += 1 + len(line)
        kept += 1
    result = lines[:kept]
    if kept == 1 and call_count > 0:
        avail = max(1, DIGEST_MAX_TOTAL_CHARS - size - 4)
        result.append(lines[1][:avail] + '...')
        kept += 1
    if kept < len(lines):
        result.append(DIGEST_OMITTED_MARKER.format(len(lines) - kept))
    return '\n'.join(result)


def _format_tool_calls_digest(
        tool_calls: list[ToolCall] | None,
    ) -> str:
    """Formats tool calls as a compact textual digest with one line
        per top-level call, e.g. `- [name] [params]`, capped at
        `DIGEST_MAX_TOTAL_CHARS` characters. Returns an empty string
        when there are no tool calls.
    """
    if not tool_calls:
        return ''
    lines = [TOOL_CALLS_HEADER]
    for tc in tool_calls:
        params = ', '.join([
            f'{k}: {_format_digest_value(v)}'
            for k, v in tc.function.arguments.items()
        ])
        lines.append(f'- [{tc.function.name}] [{params}]')
    return _cap_digest_lines(lines, len(tool_calls))


def _snap_split_index(content: str, ideal: int) -> int:
    """Snaps a split index backward to a newline (then whitespace)
        within a small window so lines and words are not cut in half.
    """
    ideal = max(0, min(ideal, len(content)))
    window_start = max(0, ideal - SPLIT_BOUNDARY_WINDOW)
    idx = content.rfind('\n', window_start, ideal)
    if idx != -1:
        return idx + 1
    idx = content.rfind(' ', window_start, ideal)
    if idx != -1:
        return idx + 1
    return ideal


def _snap_split_index_forward(content: str, hard_limit: int) -> int:
    """Snaps a hard split limit forward to a newline (then whitespace)
        within a small window so lines and words are not cut in half;
        returns the limit unchanged when no boundary is nearby.
    """
    window_end = min(len(content), hard_limit + SPLIT_BOUNDARY_WINDOW)
    idx = content.find('\n', hard_limit, window_end)
    if idx != -1:
        return idx + 1
    idx = content.find(' ', hard_limit, window_end)
    if idx != -1:
        return idx + 1
    return hard_limit


def split_message(
        message: Message, retain_chars: int,
    ) -> tuple[Message, str]:
    """Splits a message into a head for summarization and a tail
        section for retention. The head keeps the original role,
        sequence, thinking, and real `tool_calls` for summarizer
        fidelity, but gets a fresh token estimate. The tail section
        starts with the continuation header, contains the newest part
        of the content, and ends with a textual digest of the tool
        calls, if any. `retain_chars` is a strict character budget for
        the entire tail section.
    """
    digest = _format_tool_calls_digest(message.tool_calls)
    digest_section = f'\n\n{digest}' if digest else ''
    overhead = len(CONTINUED_MESSAGE_HEADER) + 1 + len(digest_section)
    avail = max(0, retain_chars - overhead)
    ideal = len(message.content) - avail
    idx = _snap_split_index(message.content, ideal)
    hard_limit = len(message.content) - avail
    if idx < hard_limit:
        idx = _snap_split_index_forward(message.content, hard_limit)
    head = Message(
        role=message.role, content=message.content[:idx],
        tool_calls=message.tool_calls, thinking=message.thinking,
        model=message.model, sequence=message.sequence,
    )
    tail_section = (
        f'{CONTINUED_MESSAGE_HEADER}\n{message.content[idx:]}'
        f'{digest_section}'
    )
    return head, tail_section


def split_oversized_last_message(
        session: SessionProtocol, phase: str, to_compact: list[Message],
    ) -> tuple[list[Message], str | None]:
    """Splits the phase's most recent message when it is too large to
        retain whole without keeping the context over the threshold
        after compaction. Returns the updated compaction selection plus
        the tail section to append to the compaction message content,
        or `None` for the tail section when no split is needed. The
        retained size is computed in one pass: the threshold minus a
        reserve for the compaction message itself (config key
        `context_compaction_reserve`, in tokens, defaulting to
        `DEFAULT_COMPACTION_RESERVE_RATIO` of the target and clamped to
        half the threshold) minus the size of the other retained
        messages. The token budget is converted to a character budget
        for the tail using the message's own characters-per-token
        density, derived from measured tokens when available and
        clamped to [0.5, 8.0].
    """
    history = session.get_history(phase)
    if not history:
        return to_compact, None
    last = history[-1]
    if last.is_compaction:
        return to_compact, None
    threshold = _get_threshold(session, phase)
    reserve = session.config.get('context_compaction_reserve')
    if reserve is None:
        reserve = DEFAULT_COMPACTION_RESERVE_RATIO * _get_target(
            session, phase
        )
    reserve = min(reserve, threshold / 2)
    selected = {m.sequence for m in to_compact}
    other_retained = sum(
        calculate_context_size([m]) for m in history
        if m.sequence not in selected and m.sequence != last.sequence
    )
    budget = threshold - reserve - other_retained
    last_size = calculate_context_size([last])
    if budget <= 0 or last_size <= budget:
        return to_compact, None
    chars = len(last.content) + last.extra_context_chars()
    density = min(max(chars / last_size, 0.5), 8.0)
    retain_chars = int(budget * density)
    head, tail_section = split_message(last, retain_chars)
    updated = [m for m in to_compact if m.sequence != last.sequence]
    updated.append(head)
    return updated, tail_section


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
        *, debug: bool = False, force: bool = False, **kwargs
    ) -> Message | None:
    """Compact old messages for a phase. `force` skips the threshold
        check, e.g. for compaction at a phase transition. Selection
        covers the messages since the phase's last compaction (the
        phase's full history on the first compaction), stopping at
        the compaction target, an estimate based on `max_context` for
        the phase. Already-compacted messages are never re-selected.
        When the phase's most recent message is too large to retain
        whole without keeping the context over the threshold, it is
        split: the older part is summarized along with the rest of the
        selection, and the newest part is retained verbatim in the
        compaction message under a continuation header. Returns the
        compaction message, or `None` if nothing was compacted; if the
        only candidate is the phase's existing compaction message, it
        is returned as-is.
    """
    if not force and not check_context_threshold(session, phase):
        return

    messages = get_messages_to_compact(session, phase)
    if not messages:
        return
    messages, tail_section = split_oversized_last_message(
        session, phase, messages
    )
    if  (   tail_section is None
            and len(messages) == 1
            and messages[0].is_compaction
    ):
        return messages[0]
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
    if tail_section is not None:
        result.content = f'{result.content}\n\n{tail_section}'

    sequences = [m.sequence for m in messages]
    session.insert_compaction(model, phase, result.content, sequences)
    result.role = Role.SYSTEM
    return result

