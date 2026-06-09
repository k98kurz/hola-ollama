"""
Compaction logic for managing conversation context.
"""

from math import ceil
from hola.session.models import Message as DBMessage
from hola.types import Message, SessionProtocol
import packify


DEFAULT_MAX_CONTEXT: int = 4096
DEFAULT_CONTEXT_THRESHOLD_RATIO: float = 0.7
DEFAULT_COMPACTION_RATIO: float = 0.5
ESTIMATED_CHARS_PER_TOKEN_RATIO: float = 4.0

COMPACTION_FAILED_MESSAGE: str = (
    "Compaction failed for older messages. "
    "Previous context was truncated. "
    "Useful information may be found in session memories."
)


def estimate_tokens(message: Message | DBMessage) -> int:
    """Estimate tokens with a simple mechanism."""
    chars = len(message.content)
    if isinstance(message, DBMessage):
        chars += len(message.data['details'])
    else:
        chars += len(packify.pack(message.tool_calls))
    return ceil(chars / ESTIMATED_CHARS_PER_TOKEN_RATIO)


def calculate_context_size(messages: list[Message]) -> int:
    """Calculate the context size, using either the measured or
        estimated tokens for each message.
    """
    return sum([
        m.tokens if m.tokens
        else m.estimated_tokens if m.estimated_tokens
        else estimate_tokens(m)
        for m in messages
    ])


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
