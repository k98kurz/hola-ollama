"""
Core types for LLM client interactions.
"""

from __future__ import annotations
from collections.abc import AsyncGenerator
from enum import Enum
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable
from hola.errors import validated_dataclass
from hola.tools.types import ToolCall, ToolCallFunction, ToolDefinition


__all__ = [
    'Role', 'Message', 'SessionProtocol',
    'ChatResponseChunk', 'ChatResponse',
    'ModelInfo', 'ModelMetadata',
    'GenerateChunk', 'GenerateResponse', 'GenerationOptions',
    'ChatRequest', 'GenerateRequest',
    'LLMClient', 'PromptGenerator', 'Adapter',
    'ToolCall', 'ToolDefinition',
]


class Role(Enum):
    """Message role enumeration."""
    SYSTEM = 'system'
    USER = 'user'
    ASSISTANT = 'assistant'
    TOOL = 'tool'


@validated_dataclass
class Message:
    role: Role
    content: str
    tool_calls: list[ToolCall] | None = None
    thinking: str | None = None
    tokens: int = 0
    estimated_tokens: int = 0


class SessionProtocol(Protocol):
    """Minimal session interface for tool execution."""
    @property
    def cwd(self) -> str:
        """Current working directory."""
        ...

    @property
    def file_permissions(self) -> dict[str, dict[str, bool]]:
        """Get all file permissions as a simple dict format."""
        ...

    @property
    def config(self) -> dict[str, object]:
        """Access the session config."""
        ...

    def get_history(
            self, phase: str = None, model: str | None = None, limit: int = None,
            only_since_last_compaction: bool = True
        ) -> list[tuple[str, Message]]:
        """Get conversation history, optionally filtered by phase and
            model. `only_since_last_compaction` is only effective if
            `phase` is provided; by default, if `phase` is provided,
            `only_since_last_compaction` should be True and should
            return the most recent compaction for the phase and the
            messages that followed.
        """
        ...

    def clear_history(
            self, phase: str | None = None, after_sequence: int | None = None
        ) -> None:
        """Clear conversation history, optionally only for a specific
            phase or beyond a specific sequence number.
        """
        ...

    def get_file_permission(self, path: Path | str) -> Permission:
        """Get effective permission for a path."""
        ...

    def set_token_baseline(self, phase: str, tokens: int) -> None:
        """Sets the token baseline for a given phase. Used in context
            management.
        """
        ...

    def get_token_baseline(self, phase: str) -> int:
        """Gets the token baseline for a given phase. Used in context
            management.
        """
        ...

    def memorize(self, key: str, value: object) -> None:
        """Store a key-value pair in session memory."""
        ...

    def recall(self, key: str) -> object:
        """Retrieve a value from session memory by key."""
        ...

    def accept_task(self, phase: str, sequence: int) -> None:
        """Mark the task as accepted by the review agent."""
        ...

    def reject_task(self, phase: str, sequence: int, reason: str) -> None:
        """Mark the task as rejcted with a reason by the review agent."""
        ...

    def accept(self) -> None:
        """Mark the workflow as accepted by the reviewer agent."""
        ...

    def reject(self, reason: str) -> None:
        """Mark the workflow as rejected with a reason by the review agent."""
        ...


@validated_dataclass
class ChatResponseChunk:
    model: str
    message: Message
    done: bool
    done_reason: str | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_tokens: int | None = None
    prompt_duration: int | None = None
    output_tokens: int | None = None
    output_duration: int | None = None


@validated_dataclass
class ChatResponse:
    model: str
    message: Message
    done: bool
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_tokens: int | None = None
    prompt_duration: int | None = None
    output_tokens: int | None = None
    output_duration: int | None = None


@validated_dataclass
class ModelInfo:
    name: str
    modified_at: str
    size: int
    digest: str


@validated_dataclass
class ModelMetadata:
    model_name: str
    context_length: int
    architecture: str | None = None
    parameter_count: int | None = None
    quantization_level: str | None = None
    family: str | None = None


@validated_dataclass
class GenerateChunk:
    model: str
    response: str
    done: bool
    done_reason: str | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_tokens: int | None = None
    prompt_duration: int | None = None
    output_tokens: int | None = None
    output_duration: int | None = None


@validated_dataclass
class GenerateResponse:
    model: str
    response: str
    done: bool
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_tokens: int | None = None
    prompt_duration: int | None = None
    output_tokens: int | None = None
    output_duration: int | None = None


@validated_dataclass
class GenerationOptions:
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    num_predict: int | None = None
    seed: int | None = None


@validated_dataclass
class ChatRequest:
    model: str
    messages: list[Message]
    tools: list[ToolDefinition] | None = None
    stream: bool = True
    options: GenerationOptions | None = None
    keep_alive: str | None = None


@validated_dataclass
class GenerateRequest:
    model: str
    prompt: str
    system: str | None = None
    stream: bool = True
    options: GenerationOptions | None = None
    keep_alive: str | None = None


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM client implementations."""
    async def chat(
            self, request: ChatRequest
        ) -> AsyncGenerator[ChatResponseChunk, None] | ChatResponse:
        """Supports both streaming and non-streaming based on request.stream."""
        ...

    async def generate(
            self, request: GenerateRequest
        ) -> AsyncGenerator[GenerateChunk, None] | GenerateResponse:
        """Supports both streaming and non-streaming based on request.stream."""
        ...

    async def list_models(self) -> list[ModelInfo]:
        """List all available models."""
        ...

    async def get_model_info(self, model: str) -> ModelMetadata:
        """Get metadata about a specific model."""
        ...

    async def close(self) -> None:
        """Close the client and cleanup resources."""
        ...


PromptGenerator = Callable[
    [SessionProtocol, str, dict[str, str] | None],
    list[Message]
]

Adapter = Callable[
    [list[Message], list[ToolDefinition]],
    ChatRequest | GenerateRequest
]
