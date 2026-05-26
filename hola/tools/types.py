"""
Core types for tool system.
"""

from __future__ import annotations
from dataclasses import field
from typing import Any
from hola.errors import validated_dataclass
from hola.helpers import make_call_id
import packify


@validated_dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]

    def pack(self) -> bytes:
        return packify.pack({
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters,
        })

    @classmethod
    def unpack(cls, data: bytes, inject: dict = {}):
        fields = packify.unpack(data, inject={**globals(), **inject})
        return cls(**fields)


@validated_dataclass
class ToolCallFunction:
    name: str
    arguments: dict[str, Any]

    def pack(self) -> bytes:
        return packify.pack({
            'name': self.name,
            'arguments': self.arguments,
        })

    @classmethod
    def unpack(cls, data: bytes, inject: dict = {}):
        fields = packify.unpack(data, inject={**globals(), **inject})
        return cls(**fields)


@validated_dataclass
class ToolCall:
    function: ToolCallFunction
    nested_calls: list[ToolCall] | None = None
    call_id: str = field(default_factory=make_call_id)

    def pack(self) -> bytes:
        return packify.pack({
            'function': self.function,
            'nested_calls': self.nested_calls,
            'call_id': self.call_id,
        })

    @classmethod
    def unpack(cls, data: bytes, inject: dict = {}):
        fields = packify.unpack(data, inject={**globals(), **inject})
        return cls(**fields)
