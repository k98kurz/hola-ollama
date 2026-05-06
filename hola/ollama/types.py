"""
Core types for Ollama API compatibility.
"""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: list["ToolCall"] | None = None
    tool_name: str | None = None


@dataclass
class FunctionDefinition:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolDefinition:
    type: Literal["function"]
    function: FunctionDefinition


@dataclass
class ToolCallFunction:
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolCall:
    function: ToolCallFunction


@dataclass
class GenerationOptions:
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    num_predict: int | None = None
    seed: int | None = None


@dataclass
class ChatRequest:
    model: str
    messages: list[Message]
    tools: list[ToolDefinition] | None = None
    stream: bool = True
    options: GenerationOptions | None = None
    keep_alive: str | None = None


@dataclass
class ChatResponseChunk:
    model: str
    message: Message
    done: bool
    done_reason: str | None = None


@dataclass
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


@dataclass
class ModelInfo:
    name: str
    modified_at: str
    size: int
    digest: str


@dataclass
class RunningModel:
    name: str
    expires_at: str
    size: int
    vm_peak: int | None = None
    vram_total: int | None = None
