"""
Ollama-specific API types.
"""

from __future__ import annotations
from collections.abc import AsyncGenerator
from hola.errors import validated_dataclass
from hola.types import ChatRequest, GenerateRequest, Message


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
