"""
Session types for Hola session management.
"""

from dataclasses import dataclass, field, asdict
from typing import Any
from packify import Packable, pack, unpack
from hola.ollama.types import Message


@dataclass
class SessionState(Packable):
    """In-memory session data."""
    conversation_history: list[tuple[str, Message]] = field(
        default_factory=list
    )
    tool_outputs: dict[str, Any] = field(default_factory=dict)
    fsm_state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)

    def pack(self) -> bytes:
        """Pack the instance into bytes."""
        data_dict = asdict(self)
        return pack(data_dict)

    @classmethod
    def unpack(cls, data: bytes, inject: dict = {}) -> "SessionState":
        """Unpack an instance from bytes."""
        data_dict = unpack(data, inject=inject)
        return cls(**data_dict)
