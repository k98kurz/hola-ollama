"""
Core types for Ollama API compatibility.
"""

from __future__ import annotations
from collections.abc import AsyncGenerator
from enum import Enum
from typing import Any, Protocol, runtime_checkable
from hola.errors import validated_dataclass
from hola.tools.types import ToolCall, ToolDefinition


class Role(Enum):
    """Message role enumeration for Ollama API compatibility."""
    SYSTEM = 'system'
    USER = 'user'
    ASSISTANT = 'assistant'
    TOOL = 'tool'


@validated_dataclass
class Message:
    role: Role
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_name: str | None = None


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
class ChatResponseChunk:
    model: str
    message: Message
    done: bool
    done_reason: str | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None


@validated_dataclass
class ChatResponse:
    model: str
    message: Message
    done: bool
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None


@validated_dataclass
class ModelInfo:
    name: str
    modified_at: str
    size: int
    digest: str


@validated_dataclass
class RunningModel:
    name: str
    expires_at: str
    size: int
    vm_peak: int | None = None
    vram_total: int | None = None


@validated_dataclass
class ModelMetadata:
    model_name: str
    architecture: str
    context_length: int
    parameter_count: int | None = None
    quantization_level: str | None = None
    family: str | None = None


@validated_dataclass
class GenerateRequest:
    model: str
    prompt: str
    system: str | None = None
    stream: bool = True
    options: GenerationOptions | None = None
    keep_alive: str | None = None


@validated_dataclass
class GenerateChunk:
    model: str
    response: str
    done: bool
    done_reason: str | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None


@validated_dataclass
class GenerateResponse:
    model: str
    response: str
    done: bool
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM client implementations."""

    async def chat(
            self, request: ChatRequest
        ) -> AsyncGenerator[ChatResponseChunk, None]:
        """Stream chat responses from the model."""
        ...

    def chat_sync(self, request: ChatRequest) -> ChatResponse:
        """Non-streaming chat response from the model."""
        ...

    async def generate(
            self, request: GenerateRequest
        ) -> AsyncGenerator[GenerateChunk, None]:
        """Stream text generation from the model."""
        ...

    def generate_sync(self, request: GenerateRequest) -> GenerateResponse:
        """Non-streaming text generation from the model."""
        ...

    async def load_model(self, model: str, keep_alive: str = "24h") -> None:
        """Explicitly load a model into memory."""
        ...

    async def unload_model(self, model: str) -> None:
        """Explicitly unload a model from memory."""
        ...

    async def list_models(self) -> list[ModelInfo]:
        """List all available models."""
        ...

    async def list_running_models(self) -> list[RunningModel]:
        """List currently loaded/running models."""
        ...

    async def get_model_info(self, model: str) -> ModelMetadata:
        """Get metadata about a specific model."""
        ...

    async def close(self) -> None:
        """Close the client and cleanup resources."""
        ...
