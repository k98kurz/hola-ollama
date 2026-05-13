"""
Core types for tool system.
"""

from __future__ import annotations
from typing import Any
from hola.errors import validated_dataclass
import packify


@validated_dataclass
class Permission:
    """Integer-based file permissions (chmod-style, one digit: 0-7).
        Permission bits: read=4, write=2, execute=1.
    """
    value: int = 0

    @property
    def read(self) -> bool:
        """Check if read permission is granted."""
        return (self.value & 4) != 0

    @property
    def write(self) -> bool:
        """Check if write permission is granted."""
        return (self.value & 2) != 0

    @property
    def execute(self) -> bool:
        """Check if execute permission is granted."""
        return (self.value & 1) != 0

    @classmethod
    def from_bits(
            cls, read: bool = False, write: bool = False,
            execute: bool = False
        ) -> Permission:
        """Create Permission from individual bit flags."""
        value = (4 if read else 0) | (2 if write else 0) | (
            1 if execute else 0
        )
        return cls(value)

    def pack(self) -> bytes:
        return packify.pack(self.value)

    @classmethod
    def unpack(cls, data: bytes, inject: dict = {}) -> Permission:
        return cls(packify.unpack(data))


@validated_dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]


@validated_dataclass
class ToolCallFunction:
    name: str
    arguments: dict[str, Any]


@validated_dataclass
class ToolCall:
    function: ToolCallFunction
    nested_calls: list[ToolCall] | None = None
