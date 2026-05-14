"""
Core types for tool system.
"""

from __future__ import annotations
from typing import Any
from hola.errors import validated_dataclass


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
