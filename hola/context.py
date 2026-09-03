from math import ceil
from hola.types import Message, TokenEstimatable


ESTIMATED_CHARS_PER_TOKEN_RATIO: float = 4.0


def estimate_tokens(message: TokenEstimatable) -> int:
    """Estimate tokens with a simple mechanism."""
    chars = len(message.content) + message.extra_context_chars()
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
